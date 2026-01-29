//! DASH (Dynamic Adaptive Streaming over HTTP) output sink.
//!
//! This module provides the [`DashSink`] element for outputting media
//! as DASH streams with MPD (Media Presentation Description) manifests.
//!
//! # DASH Overview
//!
//! DASH segments media into `.m4s` (fragmented MP4) files and generates
//! an MPD manifest that players use to discover and download segments.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        DASH Structure                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  output_dir/                                                     │
//! │    ├── manifest.mpd       (MPD manifest)                        │
//! │    ├── init.mp4           (Initialization segment)              │
//! │    ├── chunk_000001.m4s   (Media segment)                       │
//! │    ├── chunk_000002.m4s                                         │
//! │    └── chunk_000003.m4s                                         │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::streaming::{DashSink, DashConfig};
//!
//! let config = DashConfig {
//!     output_dir: "/var/www/stream".into(),
//!     segment_duration: 4.0,
//!     ..Default::default()
//! };
//!
//! let sink = DashSink::new(config)?;
//! pipeline.add_node("dash", Snk(sink));
//! ```

use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use quick_xml::Writer;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, Event};

use super::segment::{SegmentBoundaryDetector, SegmentInfo, SegmentWriter};
use crate::buffer::Buffer;
use crate::element::{ExecutionHints, SimpleSink};
use crate::error::{Error, Result};

/// DASH output sink configuration.
#[derive(Debug, Clone)]
pub struct DashConfig {
    /// Output directory for segments and manifests.
    pub output_dir: PathBuf,
    /// Target segment duration in seconds (default: 4.0).
    pub segment_duration: f64,
    /// Number of segments to keep for live streams (default: 5).
    /// Set to 0 for VOD (all segments kept).
    pub segment_window: u32,
    /// Manifest filename (default: "manifest.mpd").
    pub manifest_name: String,
    /// Segment filename prefix (default: "chunk").
    pub segment_prefix: String,
    /// Initialization segment filename (default: "init.mp4").
    pub init_segment_name: String,
    /// Whether this is a live stream (dynamic) or VOD (static).
    pub is_live: bool,
    /// Minimum buffer time in seconds (default: 6.0).
    pub min_buffer_time: f64,
    /// Suggested presentation delay for live (default: 10.0).
    pub suggested_presentation_delay: f64,
    /// Adaptation sets.
    pub adaptation_sets: Vec<DashAdaptationSet>,
    /// DASH profile (default: "urn:mpeg:dash:profile:isoff-live:2011").
    pub profile: String,
}

impl Default for DashConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("."),
            segment_duration: 4.0,
            segment_window: 5,
            manifest_name: "manifest.mpd".to_string(),
            segment_prefix: "chunk".to_string(),
            init_segment_name: "init.mp4".to_string(),
            is_live: false,
            min_buffer_time: 6.0,
            suggested_presentation_delay: 10.0,
            adaptation_sets: Vec::new(),
            profile: "urn:mpeg:dash:profile:isoff-live:2011".to_string(),
        }
    }
}

/// DASH adaptation set configuration.
#[derive(Debug, Clone)]
pub struct DashAdaptationSet {
    /// Adaptation set ID.
    pub id: u32,
    /// Content type ("video", "audio", "text").
    pub content_type: String,
    /// MIME type (e.g., "video/mp4", "audio/mp4").
    pub mime_type: String,
    /// Codec string (e.g., "avc1.64001f").
    pub codecs: String,
    /// Representations (quality levels).
    pub representations: Vec<DashRepresentation>,
}

impl DashAdaptationSet {
    /// Create a video adaptation set.
    pub fn video(id: u32, codecs: &str) -> Self {
        Self {
            id,
            content_type: "video".to_string(),
            mime_type: "video/mp4".to_string(),
            codecs: codecs.to_string(),
            representations: Vec::new(),
        }
    }

    /// Create an audio adaptation set.
    pub fn audio(id: u32, codecs: &str) -> Self {
        Self {
            id,
            content_type: "audio".to_string(),
            mime_type: "audio/mp4".to_string(),
            codecs: codecs.to_string(),
            representations: Vec::new(),
        }
    }

    /// Add a representation.
    pub fn with_representation(mut self, rep: DashRepresentation) -> Self {
        self.representations.push(rep);
        self
    }
}

/// DASH representation (quality level).
#[derive(Debug, Clone)]
pub struct DashRepresentation {
    /// Representation ID.
    pub id: String,
    /// Bandwidth in bits per second.
    pub bandwidth: u32,
    /// Video width (for video).
    pub width: Option<u32>,
    /// Video height (for video).
    pub height: Option<u32>,
    /// Frame rate (for video, e.g., "30/1").
    pub frame_rate: Option<String>,
    /// Audio sample rate (for audio).
    pub sample_rate: Option<u32>,
    /// Audio channels (for audio).
    pub channels: Option<u32>,
}

impl DashRepresentation {
    /// Create a video representation.
    pub fn video(id: &str, bandwidth: u32, width: u32, height: u32) -> Self {
        Self {
            id: id.to_string(),
            bandwidth,
            width: Some(width),
            height: Some(height),
            frame_rate: None,
            sample_rate: None,
            channels: None,
        }
    }

    /// Create an audio representation.
    pub fn audio(id: &str, bandwidth: u32, sample_rate: u32, channels: u32) -> Self {
        Self {
            id: id.to_string(),
            bandwidth,
            width: None,
            height: None,
            frame_rate: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
        }
    }

    /// Set frame rate.
    pub fn with_frame_rate(mut self, frame_rate: &str) -> Self {
        self.frame_rate = Some(frame_rate.to_string());
        self
    }
}

/// DASH output sink.
///
/// Receives fragmented MP4 data and splits it into segments with an MPD manifest.
///
/// # Input Requirements
///
/// The input should be fragmented MP4 (fMP4) data. For raw encoded data,
/// use an fMP4 muxer before this element.
///
/// # Output
///
/// Creates:
/// - `manifest.mpd` - MPD manifest
/// - `init.mp4` - Initialization segment (codec parameters)
/// - `chunk_XXXXXX.m4s` - Media segments
pub struct DashSink {
    /// Configuration.
    config: DashConfig,
    /// Segment writer.
    segment_writer: SegmentWriter,
    /// Segment boundary detector.
    boundary_detector: SegmentBoundaryDetector,
    /// Recent segments for manifest.
    segments: VecDeque<SegmentInfo>,
    /// Current buffer for accumulating data.
    current_buffer: Vec<u8>,
    /// Current segment start PTS.
    current_pts: i64,
    /// Last keyframe PTS.
    last_keyframe_pts: Option<i64>,
    /// Segment sequence number (1-based for DASH).
    segment_number: u64,
    /// Whether we've received initialization data.
    has_init_segment: bool,
    /// Whether we've written the first media segment.
    first_segment: bool,
    /// Statistics: total segments written.
    total_segments: u64,
    /// Statistics: total bytes written.
    total_bytes: u64,
    /// Stream start time for live manifest.
    availability_start_time: Option<String>,
    /// Total duration for VOD.
    total_duration: f64,
}

impl DashSink {
    /// Create a new DASH sink.
    pub fn new(config: DashConfig) -> Result<Self> {
        // Create output directory
        fs::create_dir_all(&config.output_dir).map_err(|e| {
            Error::Element(format!(
                "Failed to create DASH output directory {:?}: {}",
                config.output_dir, e
            ))
        })?;

        let segment_writer =
            SegmentWriter::new(config.output_dir.clone(), &config.segment_prefix, "m4s")?;

        let boundary_detector = SegmentBoundaryDetector::new(config.segment_duration);

        // Generate availability start time for live streams
        let availability_start_time = if config.is_live {
            Some(chrono_format_now())
        } else {
            None
        };

        Ok(Self {
            config,
            segment_writer,
            boundary_detector,
            segments: VecDeque::new(),
            current_buffer: Vec::with_capacity(1024 * 1024), // 1MB initial
            current_pts: 0,
            last_keyframe_pts: None,
            segment_number: 1,
            has_init_segment: false,
            first_segment: true,
            total_segments: 0,
            total_bytes: 0,
            availability_start_time,
            total_duration: 0.0,
        })
    }

    /// Write initialization segment.
    ///
    /// Call this once with the codec initialization data (SPS/PPS for H.264,
    /// or the moov atom for fMP4).
    pub fn write_init_segment(&mut self, data: &[u8]) -> Result<()> {
        let init_path = self.config.output_dir.join(&self.config.init_segment_name);
        let mut file = fs::File::create(&init_path).map_err(|e| {
            Error::Element(format!(
                "Failed to create init segment {:?}: {}",
                init_path, e
            ))
        })?;

        file.write_all(data)
            .map_err(|e| Error::Element(format!("Failed to write init segment: {}", e)))?;

        self.has_init_segment = true;
        self.total_bytes += data.len() as u64;

        Ok(())
    }

    /// Write the MPD manifest.
    fn write_manifest(&self) -> Result<()> {
        let manifest_path = self.config.output_dir.join(&self.config.manifest_name);
        let mut file = fs::File::create(&manifest_path).map_err(|e| {
            Error::Element(format!(
                "Failed to create manifest {:?}: {}",
                manifest_path, e
            ))
        })?;

        let content = self.generate_mpd()?;
        file.write_all(content.as_bytes())
            .map_err(|e| Error::Element(format!("Failed to write manifest: {}", e)))?;

        Ok(())
    }

    /// Generate MPD manifest content.
    fn generate_mpd(&self) -> Result<String> {
        let mut buffer = Vec::new();
        let mut writer = Writer::new_with_indent(&mut buffer, b' ', 2);

        // XML declaration
        writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        // MPD element
        let mut mpd = BytesStart::new("MPD");
        mpd.push_attribute(("xmlns", "urn:mpeg:dash:schema:mpd:2011"));
        mpd.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
        mpd.push_attribute((
            "xsi:schemaLocation",
            "urn:mpeg:dash:schema:mpd:2011 DASH-MPD.xsd",
        ));
        mpd.push_attribute(("profiles", self.config.profile.as_str()));

        if self.config.is_live {
            mpd.push_attribute(("type", "dynamic"));
            if let Some(ref ast) = self.availability_start_time {
                mpd.push_attribute(("availabilityStartTime", ast.as_str()));
            }
            mpd.push_attribute((
                "suggestedPresentationDelay",
                format!("PT{}S", self.config.suggested_presentation_delay).as_str(),
            ));
            mpd.push_attribute((
                "minimumUpdatePeriod",
                format!("PT{}S", self.config.segment_duration).as_str(),
            ));
        } else {
            mpd.push_attribute(("type", "static"));
            mpd.push_attribute((
                "mediaPresentationDuration",
                format!("PT{:.3}S", self.total_duration).as_str(),
            ));
        }

        mpd.push_attribute((
            "minBufferTime",
            format!("PT{}S", self.config.min_buffer_time).as_str(),
        ));

        writer
            .write_event(Event::Start(mpd))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        // Period element
        let mut period = BytesStart::new("Period");
        period.push_attribute(("id", "0"));
        period.push_attribute(("start", "PT0S"));
        writer
            .write_event(Event::Start(period))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        // Write adaptation sets
        self.write_adaptation_sets(&mut writer)?;

        // Close Period
        writer
            .write_event(Event::End(BytesEnd::new("Period")))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        // Close MPD
        writer
            .write_event(Event::End(BytesEnd::new("MPD")))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        String::from_utf8(buffer).map_err(|e| Error::Element(format!("UTF-8 error: {}", e)))
    }

    /// Write adaptation sets to MPD.
    fn write_adaptation_sets<W: std::io::Write>(&self, writer: &mut Writer<W>) -> Result<()> {
        // If no adaptation sets configured, create a default one
        if self.config.adaptation_sets.is_empty() {
            self.write_default_adaptation_set(writer)?;
        } else {
            for adaptation_set in &self.config.adaptation_sets {
                self.write_adaptation_set(writer, adaptation_set)?;
            }
        }
        Ok(())
    }

    /// Write default adaptation set (when none configured).
    fn write_default_adaptation_set<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
    ) -> Result<()> {
        let mut as_elem = BytesStart::new("AdaptationSet");
        as_elem.push_attribute(("id", "0"));
        as_elem.push_attribute(("contentType", "video"));
        as_elem.push_attribute(("mimeType", "video/mp4"));
        as_elem.push_attribute(("segmentAlignment", "true"));
        as_elem.push_attribute(("startWithSAP", "1"));
        writer
            .write_event(Event::Start(as_elem))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        // Default representation
        let mut rep = BytesStart::new("Representation");
        rep.push_attribute(("id", "1"));
        rep.push_attribute(("bandwidth", "2000000"));
        writer
            .write_event(Event::Start(rep))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        // Segment template
        self.write_segment_template(writer)?;

        writer
            .write_event(Event::End(BytesEnd::new("Representation")))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        writer
            .write_event(Event::End(BytesEnd::new("AdaptationSet")))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        Ok(())
    }

    /// Write a configured adaptation set.
    fn write_adaptation_set<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        adaptation_set: &DashAdaptationSet,
    ) -> Result<()> {
        let mut as_elem = BytesStart::new("AdaptationSet");
        as_elem.push_attribute(("id", adaptation_set.id.to_string().as_str()));
        as_elem.push_attribute(("contentType", adaptation_set.content_type.as_str()));
        as_elem.push_attribute(("mimeType", adaptation_set.mime_type.as_str()));
        as_elem.push_attribute(("codecs", adaptation_set.codecs.as_str()));
        as_elem.push_attribute(("segmentAlignment", "true"));
        as_elem.push_attribute(("startWithSAP", "1"));
        writer
            .write_event(Event::Start(as_elem))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        // Write representations
        for representation in &adaptation_set.representations {
            self.write_representation(writer, representation)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("AdaptationSet")))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        Ok(())
    }

    /// Write a representation.
    fn write_representation<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
        representation: &DashRepresentation,
    ) -> Result<()> {
        let mut rep = BytesStart::new("Representation");
        rep.push_attribute(("id", representation.id.as_str()));
        rep.push_attribute(("bandwidth", representation.bandwidth.to_string().as_str()));

        if let Some(width) = representation.width {
            rep.push_attribute(("width", width.to_string().as_str()));
        }
        if let Some(height) = representation.height {
            rep.push_attribute(("height", height.to_string().as_str()));
        }
        if let Some(ref frame_rate) = representation.frame_rate {
            rep.push_attribute(("frameRate", frame_rate.as_str()));
        }
        if let Some(sample_rate) = representation.sample_rate {
            rep.push_attribute(("audioSamplingRate", sample_rate.to_string().as_str()));
        }

        writer
            .write_event(Event::Start(rep))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        // Audio channel configuration
        if let Some(channels) = representation.channels {
            let mut acc = BytesStart::new("AudioChannelConfiguration");
            acc.push_attribute((
                "schemeIdUri",
                "urn:mpeg:dash:23003:3:audio_channel_configuration:2011",
            ));
            acc.push_attribute(("value", channels.to_string().as_str()));
            writer
                .write_event(Event::Empty(acc))
                .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;
        }

        // Segment template
        self.write_segment_template(writer)?;

        writer
            .write_event(Event::End(BytesEnd::new("Representation")))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        Ok(())
    }

    /// Write segment template.
    fn write_segment_template<W: std::io::Write>(&self, writer: &mut Writer<W>) -> Result<()> {
        let mut st = BytesStart::new("SegmentTemplate");

        // Timescale (nanoseconds to match our PTS)
        st.push_attribute(("timescale", "1000000000"));
        st.push_attribute(("initialization", self.config.init_segment_name.as_str()));
        st.push_attribute((
            "media",
            format!("{}$Number%06d$.m4s", self.config.segment_prefix).as_str(),
        ));
        st.push_attribute(("startNumber", "1"));

        writer
            .write_event(Event::Start(st))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        // Segment timeline
        if !self.segments.is_empty() {
            let stl = BytesStart::new("SegmentTimeline");
            writer
                .write_event(Event::Start(stl))
                .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

            for segment in &self.segments {
                let mut s = BytesStart::new("S");
                s.push_attribute(("t", segment.start_pts.to_string().as_str()));
                s.push_attribute((
                    "d",
                    (segment.end_pts - segment.start_pts).to_string().as_str(),
                ));
                writer
                    .write_event(Event::Empty(s))
                    .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;
            }

            writer
                .write_event(Event::End(BytesEnd::new("SegmentTimeline")))
                .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("SegmentTemplate")))
            .map_err(|e| Error::Element(format!("XML write error: {}", e)))?;

        Ok(())
    }

    /// Rotate to a new segment.
    fn rotate_segment(&mut self, pts: i64) -> Result<()> {
        // Finalize current segment if open
        if self.segment_writer.is_open() {
            // Write buffered data
            if !self.current_buffer.is_empty() {
                self.segment_writer.write(&self.current_buffer)?;
                self.total_bytes += self.current_buffer.len() as u64;
                self.current_buffer.clear();
            }

            // Finalize segment
            if let Some(segment_info) = self.segment_writer.finalize(pts)? {
                self.total_duration += segment_info.duration;
                self.segments.push_back(segment_info);
                self.total_segments += 1;

                // Remove old segments for live streams
                if self.config.segment_window > 0 {
                    while self.segments.len() > self.config.segment_window as usize {
                        if let Some(old) = self.segments.pop_front() {
                            // Delete old segment file
                            let _ = fs::remove_file(&old.path);
                        }
                    }
                }

                // Update manifest
                self.write_manifest()?;
            }
        }

        // Start new segment
        let is_keyframe = self.last_keyframe_pts == Some(pts);
        self.segment_writer.start_segment(pts, is_keyframe)?;
        self.boundary_detector.segment_cut(pts);
        self.current_pts = pts;
        self.segment_number += 1;

        Ok(())
    }

    /// Get statistics.
    pub fn stats(&self) -> DashStats {
        DashStats {
            total_segments: self.total_segments,
            total_bytes: self.total_bytes,
            current_segments: self.segments.len(),
            segment_number: self.segment_number,
            total_duration: self.total_duration,
        }
    }

    /// Finalize the stream (call at EOS).
    pub fn finalize(&mut self) -> Result<()> {
        // Write any remaining buffered data
        if !self.current_buffer.is_empty() && self.segment_writer.is_open() {
            self.segment_writer.write(&self.current_buffer)?;
            self.total_bytes += self.current_buffer.len() as u64;
            self.current_buffer.clear();
        }

        // Finalize last segment
        if self.segment_writer.is_open() {
            if let Some(segment_info) = self.segment_writer.finalize(self.current_pts)? {
                self.total_duration += segment_info.duration;
                self.segments.push_back(segment_info);
                self.total_segments += 1;
            }
        }

        // Write final manifest
        self.write_manifest()?;

        Ok(())
    }
}

impl SimpleSink for DashSink {
    fn consume(&mut self, buffer: &Buffer) -> Result<()> {
        let data = buffer.as_bytes();
        let pts = if buffer.metadata().pts.is_none() {
            self.current_pts
        } else {
            buffer.metadata().pts.nanos() as i64
        };

        // Check for initialization segment flag
        let is_init = buffer
            .metadata()
            .get::<bool>("mp4/init_segment")
            .copied()
            .unwrap_or(false);

        if is_init && !self.has_init_segment {
            self.write_init_segment(data)?;
            return Ok(());
        }

        // Check for keyframe
        let is_keyframe = buffer
            .metadata()
            .get::<bool>("video/keyframe")
            .copied()
            .unwrap_or(false);

        if is_keyframe {
            self.last_keyframe_pts = Some(pts);
        }

        // Start first segment
        if self.first_segment {
            self.segment_writer.start_segment(pts, is_keyframe)?;
            self.boundary_detector.segment_cut(pts);
            self.current_pts = pts;
            self.first_segment = false;
        }

        // Check if we should rotate segments
        if self.boundary_detector.should_cut(pts, is_keyframe) {
            self.rotate_segment(pts)?;
        }

        // Buffer the data
        self.current_buffer.extend_from_slice(data);
        self.current_pts = pts;

        // Periodically flush buffer to disk to prevent excessive memory use
        if self.current_buffer.len() >= 512 * 1024 {
            // 512KB threshold
            self.segment_writer.write(&self.current_buffer)?;
            self.total_bytes += self.current_buffer.len() as u64;
            self.current_buffer.clear();
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "DashSink"
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

impl Drop for DashSink {
    fn drop(&mut self) {
        // Try to finalize on drop
        let _ = self.finalize();
    }
}

/// Format current time as ISO 8601 for MPD.
fn chrono_format_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    // Simple ISO 8601 format: 2024-01-15T12:00:00Z
    let secs = now.as_secs();
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;

    // Approximate date calculation (good enough for DASH)
    let mut year = 1970u32;
    let mut remaining_days = days_since_epoch as u32;

    loop {
        let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
            366
        } else {
            365
        };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let is_leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_months: [u32; 12] = if is_leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u32;
    for days in days_in_months.iter() {
        if remaining_days < *days {
            break;
        }
        remaining_days -= *days;
        month += 1;
    }
    let day = remaining_days + 1;

    let hours = (time_of_day / 3600) as u32;
    let minutes = ((time_of_day % 3600) / 60) as u32;
    let seconds = (time_of_day % 60) as u32;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// DASH sink statistics.
#[derive(Debug, Clone)]
pub struct DashStats {
    /// Total segments written.
    pub total_segments: u64,
    /// Total bytes written.
    pub total_bytes: u64,
    /// Current segment count in window.
    pub current_segments: usize,
    /// Current segment number.
    pub segment_number: u64,
    /// Total stream duration.
    pub total_duration: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DashConfig::default();
        assert_eq!(config.segment_duration, 4.0);
        assert_eq!(config.segment_window, 5);
        assert_eq!(config.manifest_name, "manifest.mpd");
        assert!(!config.is_live);
    }

    #[test]
    fn test_adaptation_set_video() {
        let as_ = DashAdaptationSet::video(0, "avc1.64001f")
            .with_representation(DashRepresentation::video("1080p", 5_000_000, 1920, 1080));

        assert_eq!(as_.id, 0);
        assert_eq!(as_.content_type, "video");
        assert_eq!(as_.mime_type, "video/mp4");
        assert_eq!(as_.representations.len(), 1);
        assert_eq!(as_.representations[0].bandwidth, 5_000_000);
    }

    #[test]
    fn test_adaptation_set_audio() {
        let as_ = DashAdaptationSet::audio(1, "mp4a.40.2")
            .with_representation(DashRepresentation::audio("audio", 128_000, 48000, 2));

        assert_eq!(as_.id, 1);
        assert_eq!(as_.content_type, "audio");
        assert_eq!(as_.representations.len(), 1);
        assert_eq!(as_.representations[0].sample_rate, Some(48000));
    }

    #[test]
    fn test_chrono_format() {
        let formatted = chrono_format_now();
        // Should be ISO 8601 format
        assert!(formatted.contains("T"));
        assert!(formatted.ends_with("Z"));
        assert_eq!(formatted.len(), 20); // YYYY-MM-DDTHH:MM:SSZ
    }

    #[test]
    fn test_mpd_generation() {
        let config = DashConfig {
            output_dir: PathBuf::from("/tmp/dash_test"),
            is_live: false,
            ..Default::default()
        };

        // We can't fully test without creating a sink (needs filesystem)
        // but we can verify config is correct
        assert!(!config.is_live);
        assert_eq!(config.profile, "urn:mpeg:dash:profile:isoff-live:2011");
    }
}
