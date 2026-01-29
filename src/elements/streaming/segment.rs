//! Segment management for streaming protocols.
//!
//! This module provides utilities for managing media segments in
//! HLS and DASH streaming.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Information about a media segment.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment sequence number.
    pub sequence: u64,
    /// Segment duration in seconds.
    pub duration: f64,
    /// Segment filename (relative).
    pub filename: String,
    /// Segment file path (absolute).
    pub path: PathBuf,
    /// Segment start PTS in nanoseconds.
    pub start_pts: i64,
    /// Segment end PTS in nanoseconds.
    pub end_pts: i64,
    /// Whether this segment starts with a keyframe.
    pub starts_with_keyframe: bool,
}

/// Writes media segments to disk.
pub struct SegmentWriter {
    /// Output directory for segments.
    output_dir: PathBuf,
    /// Filename prefix for segments.
    prefix: String,
    /// File extension (.ts for HLS, .m4s for DASH).
    extension: String,
    /// Current segment file.
    current_file: Option<File>,
    /// Current segment sequence number.
    current_sequence: u64,
    /// Current segment start PTS.
    current_start_pts: i64,
    /// Current segment byte count.
    current_bytes: usize,
    /// Whether current segment has a keyframe.
    has_keyframe: bool,
}

impl SegmentWriter {
    /// Create a new segment writer.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory to write segments to
    /// * `prefix` - Filename prefix (e.g., "segment")
    /// * `extension` - File extension (e.g., "ts" or "m4s")
    pub fn new(output_dir: PathBuf, prefix: &str, extension: &str) -> Result<Self> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(&output_dir).map_err(|e| {
            Error::Element(format!(
                "Failed to create segment directory {:?}: {}",
                output_dir, e
            ))
        })?;

        Ok(Self {
            output_dir,
            prefix: prefix.to_string(),
            extension: extension.to_string(),
            current_file: None,
            current_sequence: 0,
            current_start_pts: 0,
            current_bytes: 0,
            has_keyframe: false,
        })
    }

    /// Start a new segment.
    ///
    /// If there's a current segment open, it will be finalized first.
    pub fn start_segment(&mut self, pts: i64, is_keyframe: bool) -> Result<()> {
        // Close current segment if open
        self.finalize_current()?;

        self.current_sequence += 1;
        self.current_start_pts = pts;
        self.current_bytes = 0;
        self.has_keyframe = is_keyframe;

        let filename = self.segment_filename(self.current_sequence);
        let path = self.output_dir.join(&filename);

        let file = File::create(&path).map_err(|e| {
            Error::Element(format!("Failed to create segment file {:?}: {}", path, e))
        })?;

        self.current_file = Some(file);

        Ok(())
    }

    /// Write data to the current segment.
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        if let Some(ref mut file) = self.current_file {
            file.write_all(data)
                .map_err(|e| Error::Element(format!("Failed to write segment data: {}", e)))?;
            self.current_bytes += data.len();
            Ok(())
        } else {
            Err(Error::Element("No segment open for writing".to_string()))
        }
    }

    /// Mark that the current segment contains a keyframe.
    pub fn mark_keyframe(&mut self) {
        self.has_keyframe = true;
    }

    /// Finalize the current segment.
    ///
    /// Returns segment info if a segment was finalized.
    pub fn finalize(&mut self, end_pts: i64) -> Result<Option<SegmentInfo>> {
        if self.current_file.is_none() {
            return Ok(None);
        }

        // Flush and close file
        if let Some(mut file) = self.current_file.take() {
            file.flush()
                .map_err(|e| Error::Element(format!("Failed to flush segment: {}", e)))?;
        }

        let filename = self.segment_filename(self.current_sequence);
        let path = self.output_dir.join(&filename);

        // Calculate duration
        let duration_ns = end_pts - self.current_start_pts;
        let duration = duration_ns as f64 / 1_000_000_000.0;

        Ok(Some(SegmentInfo {
            sequence: self.current_sequence,
            duration,
            filename,
            path,
            start_pts: self.current_start_pts,
            end_pts,
            starts_with_keyframe: self.has_keyframe,
        }))
    }

    /// Finalize current segment without returning info.
    fn finalize_current(&mut self) -> Result<()> {
        if let Some(mut file) = self.current_file.take() {
            file.flush()
                .map_err(|e| Error::Element(format!("Failed to flush segment: {}", e)))?;
        }
        Ok(())
    }

    /// Generate segment filename for a sequence number.
    fn segment_filename(&self, sequence: u64) -> String {
        format!("{}_{:06}.{}", self.prefix, sequence, self.extension)
    }

    /// Get the current segment sequence number.
    pub fn current_sequence(&self) -> u64 {
        self.current_sequence
    }

    /// Get the current segment byte count.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Get the output directory.
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Check if a segment is currently open.
    pub fn is_open(&self) -> bool {
        self.current_file.is_some()
    }
}

impl Drop for SegmentWriter {
    fn drop(&mut self) {
        // Ensure file is closed
        self.current_file.take();
    }
}

/// Detects segment boundaries based on timing and keyframes.
///
/// HLS and DASH segments should start at keyframes for proper playback.
/// This detector helps determine when to cut segments.
pub struct SegmentBoundaryDetector {
    /// Target segment duration in seconds.
    target_duration: f64,
    /// Minimum duration before considering a cut (prevents very short segments).
    min_duration: f64,
    /// Maximum duration before forcing a cut (even without keyframe).
    max_duration: f64,
    /// Current segment start PTS in nanoseconds.
    segment_start_pts: i64,
    /// Last seen PTS.
    last_pts: i64,
    /// Whether we've seen any data.
    started: bool,
}

impl SegmentBoundaryDetector {
    /// Create a new boundary detector.
    ///
    /// # Arguments
    ///
    /// * `target_duration` - Target segment duration in seconds
    pub fn new(target_duration: f64) -> Self {
        Self {
            target_duration,
            min_duration: target_duration * 0.5, // Don't cut below 50% of target
            max_duration: target_duration * 2.0, // Force cut at 200% of target
            segment_start_pts: 0,
            last_pts: 0,
            started: false,
        }
    }

    /// Create with custom min/max durations.
    pub fn with_limits(target_duration: f64, min_duration: f64, max_duration: f64) -> Self {
        Self {
            target_duration,
            min_duration,
            max_duration,
            segment_start_pts: 0,
            last_pts: 0,
            started: false,
        }
    }

    /// Check if we should start a new segment.
    ///
    /// # Arguments
    ///
    /// * `pts` - Current presentation timestamp in nanoseconds
    /// * `is_keyframe` - Whether the current frame is a keyframe
    ///
    /// # Returns
    ///
    /// `true` if a new segment should be started
    pub fn should_cut(&mut self, pts: i64, is_keyframe: bool) -> bool {
        if !self.started {
            self.segment_start_pts = pts;
            self.started = true;
            return false; // First frame, don't cut yet
        }

        self.last_pts = pts;

        let duration_ns = pts - self.segment_start_pts;
        let duration = duration_ns as f64 / 1_000_000_000.0;

        // Check if we should cut
        if is_keyframe && duration >= self.min_duration {
            // Cut at keyframe when at or past minimum duration
            if duration >= self.target_duration * 0.9 {
                return true;
            }
        }

        // Force cut if way over target (should rarely happen with proper encoding)
        if duration >= self.max_duration {
            tracing::warn!(
                "Forcing segment cut at {:.2}s without keyframe (target: {:.2}s)",
                duration,
                self.target_duration
            );
            return true;
        }

        false
    }

    /// Notify that a segment was cut.
    ///
    /// Call this after cutting a segment to reset the detector.
    pub fn segment_cut(&mut self, pts: i64) {
        self.segment_start_pts = pts;
    }

    /// Get the current segment duration in seconds.
    pub fn current_duration(&self) -> f64 {
        if !self.started {
            return 0.0;
        }
        let duration_ns = self.last_pts - self.segment_start_pts;
        duration_ns as f64 / 1_000_000_000.0
    }

    /// Get the target segment duration.
    pub fn target_duration(&self) -> f64 {
        self.target_duration
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.segment_start_pts = 0;
        self.last_pts = 0;
        self.started = false;
    }
}

/// Delete old segments beyond a retention count.
///
/// Used for live streams to limit disk usage.
pub fn cleanup_old_segments(
    output_dir: &Path,
    prefix: &str,
    extension: &str,
    keep_count: usize,
) -> io::Result<Vec<PathBuf>> {
    let mut segments: Vec<PathBuf> = fs::read_dir(output_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(prefix) && n.ends_with(extension))
                .unwrap_or(false)
        })
        .collect();

    // Sort by name (which includes sequence number)
    segments.sort();

    let mut deleted = Vec::new();

    // Delete oldest segments beyond keep_count
    if segments.len() > keep_count {
        let to_delete = segments.len() - keep_count;
        for path in segments.into_iter().take(to_delete) {
            if fs::remove_file(&path).is_ok() {
                deleted.push(path);
            }
        }
    }

    Ok(deleted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_detector_basic() {
        let mut detector = SegmentBoundaryDetector::new(6.0);

        // First frame - don't cut
        assert!(!detector.should_cut(0, true));

        // 3 seconds in, keyframe - don't cut (below min)
        assert!(!detector.should_cut(3_000_000_000, true));

        // 5.5 seconds in, keyframe - cut (above 90% of target)
        assert!(detector.should_cut(5_500_000_000, true));
    }

    #[test]
    fn test_boundary_detector_force_cut() {
        let mut detector = SegmentBoundaryDetector::new(6.0);

        assert!(!detector.should_cut(0, true));

        // 13 seconds in, no keyframe - force cut (above max)
        assert!(detector.should_cut(13_000_000_000, false));
    }

    #[test]
    fn test_segment_info() {
        let info = SegmentInfo {
            sequence: 1,
            duration: 6.0,
            filename: "segment_000001.ts".to_string(),
            path: PathBuf::from("/tmp/segment_000001.ts"),
            start_pts: 0,
            end_pts: 6_000_000_000,
            starts_with_keyframe: true,
        };

        assert_eq!(info.sequence, 1);
        assert_eq!(info.duration, 6.0);
        assert!(info.starts_with_keyframe);
    }
}
