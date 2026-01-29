//! HLS (HTTP Live Streaming) output sink.
//!
//! This module provides the [`HlsSink`] element for outputting media
//! as HLS streams with M3U8 playlists.
//!
//! # HLS Overview
//!
//! HLS segments media into `.ts` (MPEG-TS) files and generates M3U8
//! playlists that players use to discover and download segments.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        HLS Structure                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  output_dir/                                                     │
//! │    ├── playlist.m3u8      (Media playlist)                      │
//! │    ├── segment_000001.ts  (Media segment)                       │
//! │    ├── segment_000002.ts                                        │
//! │    └── segment_000003.ts                                        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! For adaptive bitrate (ABR), a master playlist references multiple
//! variant streams at different bitrates.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::streaming::{HlsSink, HlsConfig};
//!
//! let config = HlsConfig {
//!     output_dir: "/var/www/stream".into(),
//!     segment_duration: 6.0,
//!     playlist_length: 5,
//!     ..Default::default()
//! };
//!
//! let sink = HlsSink::new(config)?;
//! pipeline.add_node("hls", Snk(sink));
//! ```

use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use super::segment::{SegmentBoundaryDetector, SegmentInfo, SegmentWriter};
use crate::buffer::Buffer;
use crate::element::{ExecutionHints, SimpleSink};
use crate::error::{Error, Result};

/// HLS output sink configuration.
#[derive(Debug, Clone)]
pub struct HlsConfig {
    /// Output directory for segments and playlists.
    pub output_dir: PathBuf,
    /// Target segment duration in seconds (default: 6.0).
    pub segment_duration: f64,
    /// Number of segments to keep in playlist for live streams (default: 5).
    /// Set to 0 for VOD (all segments kept).
    pub playlist_length: u32,
    /// Playlist filename (default: "playlist.m3u8").
    pub playlist_name: String,
    /// Segment filename prefix (default: "segment").
    pub segment_prefix: String,
    /// HLS version (default: 3).
    pub hls_version: u32,
    /// Whether this is a VOD stream (adds EXT-X-ENDLIST).
    pub is_vod: bool,
    /// Multiple variants for ABR (optional).
    pub variants: Vec<HlsVariant>,
    /// Enable discontinuity markers.
    pub enable_discontinuity: bool,
}

impl Default for HlsConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("."),
            segment_duration: 6.0,
            playlist_length: 5,
            playlist_name: "playlist.m3u8".to_string(),
            segment_prefix: "segment".to_string(),
            hls_version: 3,
            is_vod: false,
            variants: Vec::new(),
            enable_discontinuity: false,
        }
    }
}

/// HLS variant for adaptive bitrate streaming.
#[derive(Debug, Clone)]
pub struct HlsVariant {
    /// Variant name (used for subdirectory).
    pub name: String,
    /// Bandwidth in bits per second.
    pub bandwidth: u32,
    /// Video width (optional).
    pub width: Option<u32>,
    /// Video height (optional).
    pub height: Option<u32>,
    /// Codec string (e.g., "avc1.64001f,mp4a.40.2").
    pub codecs: Option<String>,
}

impl HlsVariant {
    /// Create a new HLS variant.
    pub fn new(name: &str, bandwidth: u32) -> Self {
        Self {
            name: name.to_string(),
            bandwidth,
            width: None,
            height: None,
            codecs: None,
        }
    }

    /// Set video resolution.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set codec string.
    pub fn with_codecs(mut self, codecs: &str) -> Self {
        self.codecs = Some(codecs.to_string());
        self
    }
}

/// HLS output sink.
///
/// Receives MPEG-TS data and splits it into segments with M3U8 playlists.
///
/// # Input Requirements
///
/// The input must be MPEG-TS muxed data. Use `TsMux` before this element.
///
/// # Output
///
/// Creates:
/// - `playlist.m3u8` - Media playlist
/// - `segment_XXXXXX.ts` - Media segments
/// - `master.m3u8` - Master playlist (if variants are configured)
pub struct HlsSink {
    /// Configuration.
    config: HlsConfig,
    /// Segment writer.
    segment_writer: SegmentWriter,
    /// Segment boundary detector.
    boundary_detector: SegmentBoundaryDetector,
    /// Recent segments for playlist.
    segments: VecDeque<SegmentInfo>,
    /// Current buffer for accumulating data.
    current_buffer: Vec<u8>,
    /// Current segment start PTS.
    current_pts: i64,
    /// Last keyframe PTS.
    last_keyframe_pts: Option<i64>,
    /// Media sequence number for playlist.
    media_sequence: u64,
    /// Whether we've written the first segment.
    first_segment: bool,
    /// Statistics: total segments written.
    total_segments: u64,
    /// Statistics: total bytes written.
    total_bytes: u64,
}

impl HlsSink {
    /// Create a new HLS sink.
    pub fn new(config: HlsConfig) -> Result<Self> {
        // Create output directory
        fs::create_dir_all(&config.output_dir).map_err(|e| {
            Error::Element(format!(
                "Failed to create HLS output directory {:?}: {}",
                config.output_dir, e
            ))
        })?;

        let segment_writer =
            SegmentWriter::new(config.output_dir.clone(), &config.segment_prefix, "ts")?;

        let boundary_detector = SegmentBoundaryDetector::new(config.segment_duration);

        Ok(Self {
            config,
            segment_writer,
            boundary_detector,
            segments: VecDeque::new(),
            current_buffer: Vec::with_capacity(1024 * 1024), // 1MB initial
            current_pts: 0,
            last_keyframe_pts: None,
            media_sequence: 1,
            first_segment: true,
            total_segments: 0,
            total_bytes: 0,
        })
    }

    /// Write the media playlist (M3U8).
    fn write_playlist(&self) -> Result<()> {
        let playlist_path = self.config.output_dir.join(&self.config.playlist_name);
        let mut file = fs::File::create(&playlist_path).map_err(|e| {
            Error::Element(format!(
                "Failed to create playlist {:?}: {}",
                playlist_path, e
            ))
        })?;

        let content = self.generate_media_playlist();
        file.write_all(content.as_bytes())
            .map_err(|e| Error::Element(format!("Failed to write playlist: {}", e)))?;

        Ok(())
    }

    /// Generate media playlist content.
    fn generate_media_playlist(&self) -> String {
        let mut playlist = String::new();

        playlist.push_str("#EXTM3U\n");
        playlist.push_str(&format!("#EXT-X-VERSION:{}\n", self.config.hls_version));

        // Calculate target duration (must be >= max segment duration)
        let target_duration = self
            .segments
            .iter()
            .map(|s| s.duration.ceil() as u32)
            .max()
            .unwrap_or(self.config.segment_duration.ceil() as u32);

        playlist.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", target_duration));
        playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", self.media_sequence));

        // Add segments
        for segment in &self.segments {
            playlist.push_str(&format!("#EXTINF:{:.3},\n", segment.duration));
            playlist.push_str(&format!("{}\n", segment.filename));
        }

        // Add endlist for VOD
        if self.config.is_vod {
            playlist.push_str("#EXT-X-ENDLIST\n");
        }

        playlist
    }

    /// Write master playlist for ABR.
    pub fn write_master_playlist(&self) -> Result<()> {
        if self.config.variants.is_empty() {
            return Ok(());
        }

        let master_path = self.config.output_dir.join("master.m3u8");
        let mut file = fs::File::create(&master_path).map_err(|e| {
            Error::Element(format!(
                "Failed to create master playlist {:?}: {}",
                master_path, e
            ))
        })?;

        let content = self.generate_master_playlist();
        file.write_all(content.as_bytes())
            .map_err(|e| Error::Element(format!("Failed to write master playlist: {}", e)))?;

        Ok(())
    }

    /// Generate master playlist content.
    fn generate_master_playlist(&self) -> String {
        let mut playlist = String::new();

        playlist.push_str("#EXTM3U\n");

        for variant in &self.config.variants {
            let mut stream_inf = format!("#EXT-X-STREAM-INF:BANDWIDTH={}", variant.bandwidth);

            if let (Some(w), Some(h)) = (variant.width, variant.height) {
                stream_inf.push_str(&format!(",RESOLUTION={}x{}", w, h));
            }

            if let Some(ref codecs) = variant.codecs {
                stream_inf.push_str(&format!(",CODECS=\"{}\"", codecs));
            }

            playlist.push_str(&stream_inf);
            playlist.push('\n');
            playlist.push_str(&format!("{}/{}\n", variant.name, self.config.playlist_name));
        }

        playlist
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
                self.segments.push_back(segment_info);
                self.total_segments += 1;

                // Remove old segments for live streams
                if self.config.playlist_length > 0 {
                    while self.segments.len() > self.config.playlist_length as usize {
                        if let Some(old) = self.segments.pop_front() {
                            // Delete old segment file
                            let _ = fs::remove_file(&old.path);
                            self.media_sequence += 1;
                        }
                    }
                }

                // Update playlist
                self.write_playlist()?;
            }
        }

        // Start new segment
        let is_keyframe = self.last_keyframe_pts == Some(pts);
        self.segment_writer.start_segment(pts, is_keyframe)?;
        self.boundary_detector.segment_cut(pts);
        self.current_pts = pts;

        Ok(())
    }

    /// Get statistics.
    pub fn stats(&self) -> HlsStats {
        HlsStats {
            total_segments: self.total_segments,
            total_bytes: self.total_bytes,
            current_segments: self.segments.len(),
            media_sequence: self.media_sequence,
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
                self.segments.push_back(segment_info);
                self.total_segments += 1;
            }
        }

        // Write final playlist (with ENDLIST for VOD)
        self.write_playlist()?;

        // Write master playlist if configured
        self.write_master_playlist()?;

        Ok(())
    }
}

impl SimpleSink for HlsSink {
    fn consume(&mut self, buffer: &Buffer) -> Result<()> {
        let data = buffer.as_bytes();
        let pts = if buffer.metadata().pts.is_none() {
            self.current_pts
        } else {
            buffer.metadata().pts.nanos() as i64
        };

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
        "HlsSink"
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

impl Drop for HlsSink {
    fn drop(&mut self) {
        // Try to finalize on drop
        let _ = self.finalize();
    }
}

/// HLS sink statistics.
#[derive(Debug, Clone)]
pub struct HlsStats {
    /// Total segments written.
    pub total_segments: u64,
    /// Total bytes written.
    pub total_bytes: u64,
    /// Current segment count in playlist.
    pub current_segments: usize,
    /// Current media sequence number.
    pub media_sequence: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = HlsConfig::default();
        assert_eq!(config.segment_duration, 6.0);
        assert_eq!(config.playlist_length, 5);
        assert_eq!(config.playlist_name, "playlist.m3u8");
    }

    #[test]
    fn test_variant() {
        let variant = HlsVariant::new("720p", 2_000_000)
            .with_resolution(1280, 720)
            .with_codecs("avc1.64001f,mp4a.40.2");

        assert_eq!(variant.name, "720p");
        assert_eq!(variant.bandwidth, 2_000_000);
        assert_eq!(variant.width, Some(1280));
        assert_eq!(variant.height, Some(720));
    }

    #[test]
    fn test_media_playlist_generation() {
        let config = HlsConfig::default();
        let sink = HlsSink {
            config,
            segment_writer: SegmentWriter::new(PathBuf::from("/tmp"), "seg", "ts").unwrap(),
            boundary_detector: SegmentBoundaryDetector::new(6.0),
            segments: VecDeque::from([
                SegmentInfo {
                    sequence: 1,
                    duration: 6.0,
                    filename: "segment_000001.ts".to_string(),
                    path: PathBuf::from("/tmp/segment_000001.ts"),
                    start_pts: 0,
                    end_pts: 6_000_000_000,
                    starts_with_keyframe: true,
                },
                SegmentInfo {
                    sequence: 2,
                    duration: 6.0,
                    filename: "segment_000002.ts".to_string(),
                    path: PathBuf::from("/tmp/segment_000002.ts"),
                    start_pts: 6_000_000_000,
                    end_pts: 12_000_000_000,
                    starts_with_keyframe: true,
                },
            ]),
            current_buffer: Vec::new(),
            current_pts: 0,
            last_keyframe_pts: None,
            media_sequence: 1,
            first_segment: false,
            total_segments: 2,
            total_bytes: 0,
        };

        let playlist = sink.generate_media_playlist();

        assert!(playlist.contains("#EXTM3U"));
        assert!(playlist.contains("#EXT-X-VERSION:3"));
        assert!(playlist.contains("#EXT-X-TARGETDURATION:6"));
        assert!(playlist.contains("#EXT-X-MEDIA-SEQUENCE:1"));
        assert!(playlist.contains("#EXTINF:6.000"));
        assert!(playlist.contains("segment_000001.ts"));
        assert!(playlist.contains("segment_000002.ts"));
        // Not VOD, so no ENDLIST
        assert!(!playlist.contains("#EXT-X-ENDLIST"));
    }

    #[test]
    fn test_master_playlist_generation() {
        let config = HlsConfig {
            variants: vec![
                HlsVariant::new("1080p", 5_000_000)
                    .with_resolution(1920, 1080)
                    .with_codecs("avc1.640028,mp4a.40.2"),
                HlsVariant::new("720p", 2_500_000)
                    .with_resolution(1280, 720)
                    .with_codecs("avc1.64001f,mp4a.40.2"),
            ],
            ..Default::default()
        };

        let sink = HlsSink {
            config,
            segment_writer: SegmentWriter::new(PathBuf::from("/tmp"), "seg", "ts").unwrap(),
            boundary_detector: SegmentBoundaryDetector::new(6.0),
            segments: VecDeque::new(),
            current_buffer: Vec::new(),
            current_pts: 0,
            last_keyframe_pts: None,
            media_sequence: 1,
            first_segment: true,
            total_segments: 0,
            total_bytes: 0,
        };

        let master = sink.generate_master_playlist();

        assert!(master.contains("#EXTM3U"));
        assert!(master.contains("BANDWIDTH=5000000"));
        assert!(master.contains("RESOLUTION=1920x1080"));
        assert!(master.contains("1080p/playlist.m3u8"));
        assert!(master.contains("720p/playlist.m3u8"));
    }
}
