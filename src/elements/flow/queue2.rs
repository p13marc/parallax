//! Advanced buffering queue with network-oriented strategies.
//!
//! `Queue2` extends the basic [`Queue`](super::Queue) with buffering modes
//! designed for network streaming:
//!
//! - **Stream**: In-memory ring buffer with watermark-based pause/resume
//! - **Download**: File-backed progressive download with random access
//! - **Timeshift**: Circular file buffer for DVR-like rewind
//!
//! Buffering progress (0-100%) is reported via pipeline bus messages.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::flow::Queue2;
//! use parallax::pipeline::bus::BufferingMode;
//!
//! // Stream mode (default): pause when empty, resume when refilled
//! let queue = Queue2::stream(10 * 1024 * 1024); // 10 MB buffer
//!
//! // Download mode: progressive download to disk
//! let queue = Queue2::download("/tmp/parallax_dl.tmp", Some(total_size));
//!
//! // Timeshift mode: rewind up to 60 seconds of live stream
//! let queue = Queue2::timeshift("/tmp/parallax_ts.tmp", 60 * 1024 * 1024);
//! ```

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::pipeline::bus::{BufferingMode, BusHandle, MessageKind};

// ============================================================================
// Configuration
// ============================================================================

/// Buffering configuration for Queue2.
#[derive(Debug, Clone, PartialEq)]
pub struct BufferingConfig {
    /// Buffering mode.
    pub mode: BufferingMode,
    /// High watermark percentage (resume playback). Default: 95.
    pub high_percent: u32,
    /// Low watermark percentage (pause playback). Default: 10.
    pub low_percent: u32,
    /// Maximum buffer size in bytes.
    pub max_size_bytes: usize,
    /// File path for download/timeshift modes.
    pub temp_file: Option<PathBuf>,
    /// Total expected size (for download mode percentage calculation).
    pub total_size: Option<u64>,
}

impl Default for BufferingConfig {
    fn default() -> Self {
        Self {
            mode: BufferingMode::Stream,
            high_percent: 95,
            low_percent: 10,
            max_size_bytes: 10 * 1024 * 1024, // 10 MB
            temp_file: None,
            total_size: None,
        }
    }
}

// ============================================================================
// Buffering Action
// ============================================================================

/// Action determined by the buffering state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferingAction {
    /// Forward buffer downstream normally.
    Forward,
    /// Hold buffer (don't forward yet, still buffering).
    Hold,
    /// Pause downstream (underrun).
    Pause,
    /// Resume downstream (buffer refilled).
    Resume,
}

// ============================================================================
// Downloaded Ranges (for download mode)
// ============================================================================

/// Tracks non-contiguous downloaded byte ranges for seek UI.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DownloadedRanges {
    ranges: Vec<(u64, u64)>,
}

impl DownloadedRanges {
    /// Add a downloaded range.
    pub fn add(&mut self, offset: u64, size: u64) {
        if size == 0 {
            return;
        }
        let end = offset + size;
        self.ranges.push((offset, end));
        self.merge();
    }

    /// Check if a byte offset is within a downloaded range.
    pub fn contains(&self, offset: u64) -> bool {
        self.ranges
            .iter()
            .any(|(start, end)| offset >= *start && offset < *end)
    }

    /// Get the downloaded ranges as (start, end) pairs.
    pub fn as_slice(&self) -> &[(u64, u64)] {
        &self.ranges
    }

    /// Total downloaded bytes.
    pub fn total_bytes(&self) -> u64 {
        self.ranges.iter().map(|(s, e)| e - s).sum()
    }

    fn merge(&mut self) {
        self.ranges.sort_by_key(|r| r.0);
        let mut merged = Vec::new();
        for range in &self.ranges {
            if let Some(last) = merged.last_mut() {
                let last: &mut (u64, u64) = last;
                if range.0 <= last.1 {
                    last.1 = last.1.max(range.1);
                    continue;
                }
            }
            merged.push(*range);
        }
        self.ranges = merged;
    }
}

// ============================================================================
// Queue2 Statistics
// ============================================================================

/// Statistics for Queue2.
#[derive(Debug, Clone, PartialEq)]
pub struct Queue2Stats {
    /// Current buffering percentage (0-100).
    pub percent: u32,
    /// Whether the queue is in buffering mode.
    pub is_buffering: bool,
    /// Total bytes received.
    pub bytes_in: u64,
    /// Total bytes sent downstream.
    pub bytes_out: u64,
    /// Estimated input rate (bytes/sec).
    pub avg_in_rate: f64,
    /// Estimated output rate (bytes/sec).
    pub avg_out_rate: f64,
    /// Current buffer size in bytes.
    pub current_size: usize,
    /// Total buffers received.
    pub buffers_in: u64,
    /// Total buffers forwarded.
    pub buffers_out: u64,
}

impl std::fmt::Display for Queue2Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}% ({}B in, {}B out, {} bufs, buffering={})",
            self.percent, self.bytes_in, self.bytes_out, self.buffers_in, self.is_buffering,
        )
    }
}

// ============================================================================
// Queue2
// ============================================================================

/// Advanced buffering queue with stream, download, and timeshift modes.
///
/// Unlike [`Queue`](super::Queue), Queue2 is designed for network streaming
/// scenarios where buffering strategies are needed to handle variable
/// network conditions.
pub struct Queue2 {
    config: BufferingConfig,
    /// In-memory buffer (stream mode).
    ring: VecDeque<Buffer>,
    /// Current buffer size in bytes.
    current_bytes: usize,
    /// Buffering state.
    is_buffering: bool,
    /// Buffering percentage (0-100).
    percent: u32,
    /// Total bytes received.
    bytes_in: u64,
    /// Total bytes forwarded.
    bytes_out: u64,
    /// Total buffers received.
    buffers_in: u64,
    /// Total buffers forwarded.
    buffers_out: u64,
    /// Rate estimation state.
    rate_state: RateState,
    /// Bus handle for posting buffering messages.
    bus: Option<BusHandle>,
    /// File for download/timeshift modes.
    file: Option<File>,
    /// Current write position in file.
    file_write_pos: u64,
    /// Current read position in file (for timeshift mode).
    #[allow(dead_code)]
    file_read_pos: u64,
    /// Downloaded ranges (for download mode).
    downloaded_ranges: DownloadedRanges,
}

struct RateState {
    last_time: Instant,
    last_bytes_in: u64,
    last_bytes_out: u64,
    avg_in_rate: f64,
    avg_out_rate: f64,
}

impl Queue2 {
    /// Create a Queue2 in stream buffering mode.
    pub fn stream(max_size_bytes: usize) -> Self {
        Self::with_config(BufferingConfig {
            mode: BufferingMode::Stream,
            max_size_bytes,
            ..Default::default()
        })
    }

    /// Create a Queue2 in download buffering mode.
    pub fn download(path: impl AsRef<Path>, total_size: Option<u64>) -> Self {
        Self::with_config(BufferingConfig {
            mode: BufferingMode::Download,
            temp_file: Some(path.as_ref().to_path_buf()),
            total_size,
            ..Default::default()
        })
    }

    /// Create a Queue2 in timeshift buffering mode.
    pub fn timeshift(path: impl AsRef<Path>, max_size_bytes: usize) -> Self {
        Self::with_config(BufferingConfig {
            mode: BufferingMode::Timeshift,
            temp_file: Some(path.as_ref().to_path_buf()),
            max_size_bytes,
            ..Default::default()
        })
    }

    /// Create a Queue2 with custom configuration.
    pub fn with_config(config: BufferingConfig) -> Self {
        Self {
            config,
            ring: VecDeque::new(),
            current_bytes: 0,
            is_buffering: true, // Start in buffering mode
            percent: 0,
            bytes_in: 0,
            bytes_out: 0,
            buffers_in: 0,
            buffers_out: 0,
            rate_state: RateState {
                last_time: Instant::now(),
                last_bytes_in: 0,
                last_bytes_out: 0,
                avg_in_rate: 0.0,
                avg_out_rate: 0.0,
            },
            bus: None,
            file: None,
            file_write_pos: 0,
            file_read_pos: 0,
            downloaded_ranges: DownloadedRanges::default(),
        }
    }

    /// Set watermark percentages.
    pub fn with_watermarks(mut self, low: u32, high: u32) -> Self {
        self.config.low_percent = low;
        self.config.high_percent = high;
        self
    }

    /// Get current statistics.
    pub fn stats(&self) -> Queue2Stats {
        Queue2Stats {
            percent: self.percent,
            is_buffering: self.is_buffering,
            bytes_in: self.bytes_in,
            bytes_out: self.bytes_out,
            avg_in_rate: self.rate_state.avg_in_rate,
            avg_out_rate: self.rate_state.avg_out_rate,
            current_size: self.current_bytes,
            buffers_in: self.buffers_in,
            buffers_out: self.buffers_out,
        }
    }

    /// Get downloaded ranges (download mode only).
    pub fn downloaded_ranges(&self) -> &DownloadedRanges {
        &self.downloaded_ranges
    }

    /// Whether the queue is currently in buffering mode (downstream should pause).
    pub fn is_buffering(&self) -> bool {
        self.is_buffering
    }

    /// Current buffering percentage (0-100).
    pub fn buffering_percent(&self) -> u32 {
        self.percent
    }

    // ========================================================================
    // Stream Mode
    // ========================================================================

    fn process_stream_push(&mut self, buffer: Buffer) -> BufferingAction {
        let buf_len = buffer.len();
        self.ring.push_back(buffer);
        self.current_bytes += buf_len;
        self.bytes_in += buf_len as u64;
        self.buffers_in += 1;

        // Enforce max size by dropping oldest
        while self.current_bytes > self.config.max_size_bytes && self.ring.len() > 1 {
            if let Some(old) = self.ring.pop_front() {
                self.current_bytes = self.current_bytes.saturating_sub(old.len());
            }
        }

        self.percent = self.calculate_stream_percent();
        self.update_rate_estimates();

        if self.is_buffering {
            if self.percent >= self.config.high_percent {
                self.is_buffering = false;
                self.post_buffering(100);
                BufferingAction::Resume
            } else {
                self.post_buffering(self.percent);
                BufferingAction::Hold
            }
        } else if self.percent <= self.config.low_percent && self.config.low_percent > 0 {
            self.is_buffering = true;
            self.post_buffering(self.percent);
            BufferingAction::Pause
        } else {
            BufferingAction::Forward
        }
    }

    fn stream_pop(&mut self) -> Option<Buffer> {
        let buf = self.ring.pop_front()?;
        self.current_bytes = self.current_bytes.saturating_sub(buf.len());
        self.bytes_out += buf.len() as u64;
        self.buffers_out += 1;
        self.percent = self.calculate_stream_percent();
        Some(buf)
    }

    fn calculate_stream_percent(&self) -> u32 {
        if self.config.max_size_bytes == 0 {
            return 100;
        }
        ((self.current_bytes as u64 * 100) / self.config.max_size_bytes as u64).min(100) as u32
    }

    // ========================================================================
    // Download Mode
    // ========================================================================

    fn process_download_push(&mut self, buffer: Buffer) -> Result<BufferingAction> {
        let data = buffer.as_bytes();
        let len = data.len() as u64;

        // Ensure file is open
        if self.file.is_none() {
            let path = self
                .config
                .temp_file
                .as_ref()
                .ok_or_else(|| Error::Config("download mode requires temp_file path".into()))?;
            self.file = Some(File::create(path)?);
        }

        let file = self.file.as_mut().unwrap();
        file.write_all(data)?;

        let offset = self.file_write_pos;
        self.file_write_pos += len;
        self.bytes_in += len;
        self.buffers_in += 1;
        self.downloaded_ranges.add(offset, len);

        // Also keep in memory ring for forwarding
        self.ring.push_back(buffer);
        self.current_bytes += len as usize;

        // Calculate percentage based on total size
        if let Some(total) = self.config.total_size
            && let Some(pct) = (self.file_write_pos * 100).checked_div(total)
        {
            self.percent = pct.min(100) as u32;
        }

        self.update_rate_estimates();
        self.post_buffering(self.percent);

        Ok(BufferingAction::Forward)
    }

    // ========================================================================
    // Timeshift Mode
    // ========================================================================

    fn process_timeshift_push(&mut self, buffer: Buffer) -> Result<BufferingAction> {
        let data = buffer.as_bytes();
        let len = data.len() as u64;

        // Ensure file is open
        if self.file.is_none() {
            let path =
                self.config.temp_file.as_ref().ok_or_else(|| {
                    Error::Config("timeshift mode requires temp_file path".into())
                })?;
            self.file = Some(
                File::options()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)?,
            );
        }

        let file = self.file.as_mut().unwrap();

        // Write at circular position
        let write_pos = self.file_write_pos % self.config.max_size_bytes as u64;
        file.seek(SeekFrom::Start(write_pos))?;
        file.write_all(data)?;

        self.file_write_pos += len;
        self.bytes_in += len;
        self.buffers_in += 1;

        // Also keep in memory ring for immediate forwarding
        self.ring.push_back(buffer);
        self.current_bytes += len as usize;

        // Limit in-memory size
        while self.current_bytes > self.config.max_size_bytes / 4 && self.ring.len() > 1 {
            if let Some(old) = self.ring.pop_front() {
                self.current_bytes = self.current_bytes.saturating_sub(old.len());
            }
        }

        self.update_rate_estimates();
        Ok(BufferingAction::Forward)
    }

    /// Get the available rewind duration in timeshift mode.
    pub fn available_rewind(&self) -> Duration {
        let available_bytes = self.file_write_pos.min(self.config.max_size_bytes as u64);
        if self.rate_state.avg_in_rate > 0.0 {
            Duration::from_secs_f64(available_bytes as f64 / self.rate_state.avg_in_rate)
        } else {
            Duration::ZERO
        }
    }

    // ========================================================================
    // Rate Estimation
    // ========================================================================

    fn update_rate_estimates(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.rate_state.last_time).as_secs_f64();

        // Update every 500ms
        if elapsed < 0.5 {
            return;
        }

        let bytes_in_delta = self.bytes_in - self.rate_state.last_bytes_in;
        let bytes_out_delta = self.bytes_out - self.rate_state.last_bytes_out;

        let in_rate = bytes_in_delta as f64 / elapsed;
        let out_rate = bytes_out_delta as f64 / elapsed;

        // Exponential moving average (alpha=0.3)
        const ALPHA: f64 = 0.3;
        self.rate_state.avg_in_rate = ALPHA * in_rate + (1.0 - ALPHA) * self.rate_state.avg_in_rate;
        self.rate_state.avg_out_rate =
            ALPHA * out_rate + (1.0 - ALPHA) * self.rate_state.avg_out_rate;

        self.rate_state.last_time = now;
        self.rate_state.last_bytes_in = self.bytes_in;
        self.rate_state.last_bytes_out = self.bytes_out;
    }

    // ========================================================================
    // Bus Integration
    // ========================================================================

    fn post_buffering(&self, percent: u32) {
        if let Some(ref bus) = self.bus {
            let estimated_total = if self.rate_state.avg_in_rate > 0.0 && percent < 100 {
                let remaining_bytes = self.config.max_size_bytes as f64 - self.current_bytes as f64;
                let eta_secs = remaining_bytes / self.rate_state.avg_in_rate;
                if eta_secs > 0.0 && eta_secs < 3600.0 {
                    Some(crate::clock::ClockTime::from_nanos(
                        (eta_secs * 1_000_000_000.0) as u64,
                    ))
                } else {
                    None
                }
            } else {
                None
            };

            bus.post(MessageKind::Buffering {
                percent,
                mode: self.config.mode,
                avg_in_rate: Some(self.rate_state.avg_in_rate as u64),
                avg_out_rate: Some(self.rate_state.avg_out_rate as u64),
                estimated_total,
            });
        }
    }
}

// ============================================================================
// Element Implementation
// ============================================================================

impl Element for Queue2 {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Push buffer based on mode
        let action = match self.config.mode {
            BufferingMode::Stream | BufferingMode::Live => {
                self.process_stream_push(buffer);
                // In Element mode, always try to forward
                BufferingAction::Forward
            }
            BufferingMode::Download => {
                self.process_download_push(buffer)?;
                BufferingAction::Forward
            }
            BufferingMode::Timeshift => {
                self.process_timeshift_push(buffer)?;
                BufferingAction::Forward
            }
        };

        match action {
            BufferingAction::Forward | BufferingAction::Resume => Ok(self.stream_pop()),
            BufferingAction::Hold | BufferingAction::Pause => Ok(None),
        }
    }

    fn name(&self) -> &str {
        "queue2"
    }

    fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(256, 64).unwrap())
    }

    fn make_buffer(size: usize, seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, size.min(256));
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn test_stream_mode_basic() {
        let mut q = Queue2::stream(1024);
        q.is_buffering = false; // Skip initial buffering for test

        let buf = make_buffer(100, 0);
        let result = q.process(buf).unwrap();
        assert!(result.is_some());
        assert_eq!(q.stats().buffers_in, 1);
        assert_eq!(q.stats().buffers_out, 1);
    }

    #[test]
    fn test_stream_mode_watermarks() {
        // Small buffer: 200 bytes max
        let mut q = Queue2::stream(200).with_watermarks(10, 90);

        // Start in buffering mode
        assert!(q.is_buffering());

        // Fill to ~50% (100 bytes) — still buffering
        let action = q.process_stream_push(make_buffer(100, 0));
        assert_eq!(q.buffering_percent(), 50);
        assert!(q.is_buffering()); // 50% < 90% high watermark
        assert_eq!(action, BufferingAction::Hold);

        // Fill to ~100% (another 100 bytes) — should resume
        let action = q.process_stream_push(make_buffer(100, 1));
        assert!(!q.is_buffering()); // ≥ 90% high watermark
        assert_eq!(action, BufferingAction::Resume);

        // Drain to near empty
        q.stream_pop();
        q.stream_pop();
        // Force recalculation
        assert_eq!(q.calculate_stream_percent(), 0);
    }

    #[test]
    fn test_stream_mode_enforces_max_size() {
        let mut q = Queue2::stream(300);
        q.is_buffering = false;

        // Push 5 buffers of 100 bytes each (500 > 300 max)
        for i in 0..5 {
            q.process_stream_push(make_buffer(100, i));
        }

        // Should have dropped oldest to stay within max
        assert!(q.current_bytes <= 300);
    }

    #[test]
    fn test_downloaded_ranges() {
        let mut ranges = DownloadedRanges::default();

        ranges.add(0, 100);
        ranges.add(100, 100);
        assert!(ranges.contains(0));
        assert!(ranges.contains(50));
        assert!(ranges.contains(150));
        assert!(!ranges.contains(200));

        // Merged into one range
        assert_eq!(ranges.as_slice().len(), 1);
        assert_eq!(ranges.as_slice()[0], (0, 200));
        assert_eq!(ranges.total_bytes(), 200);
    }

    #[test]
    fn test_downloaded_ranges_gap() {
        let mut ranges = DownloadedRanges::default();

        ranges.add(0, 100);
        ranges.add(200, 100);
        assert!(ranges.contains(50));
        assert!(!ranges.contains(150));
        assert!(ranges.contains(250));

        assert_eq!(ranges.as_slice().len(), 2);
    }

    #[test]
    fn test_download_mode() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let mut q = Queue2::download(temp.path(), Some(500));

        let buf = make_buffer(100, 0);
        let action = q.process_download_push(buf).unwrap();
        assert_eq!(action, BufferingAction::Forward);
        assert_eq!(q.percent, 20); // 100/500 = 20%
        assert!(q.downloaded_ranges().contains(0));
        assert_eq!(q.downloaded_ranges().total_bytes(), 100);
    }

    #[test]
    fn test_timeshift_mode() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let mut q = Queue2::timeshift(temp.path(), 1024);

        let buf = make_buffer(100, 0);
        let action = q.process_timeshift_push(buf).unwrap();
        assert_eq!(action, BufferingAction::Forward);
        assert_eq!(q.bytes_in, 100);
        assert!(q.file_write_pos > 0);
    }

    #[test]
    fn test_stats() {
        let mut q = Queue2::stream(1024).with_watermarks(0, 90);
        q.is_buffering = false;

        q.process(make_buffer(100, 0)).unwrap();
        q.process(make_buffer(100, 1)).unwrap();

        let stats = q.stats();
        assert_eq!(stats.buffers_in, 2);
        assert_eq!(stats.buffers_out, 2);
        assert!(!stats.is_buffering);
    }

    #[test]
    fn test_bus_message_on_buffering() {
        use crate::pipeline::bus::Bus;

        let (mut bus, handle) = Bus::new();
        let mut q = Queue2::stream(200).with_watermarks(10, 90);
        q.bus = Some(handle.for_element("queue2"));

        // Push buffer while buffering
        q.process_stream_push(make_buffer(100, 0));

        // Should have posted buffering message
        let msg = bus.poll();
        assert!(msg.is_some());
        if let Some(msg) = msg {
            assert!(
                matches!(msg.kind, MessageKind::Buffering { .. }),
                "Expected Buffering message, got {:?}",
                msg.kind
            );
        }
    }
}
