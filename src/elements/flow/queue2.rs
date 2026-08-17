//! Advanced buffering queue with network-oriented strategies.
//!
//! `Queue2` provides buffering modes designed for network streaming:
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::event::{Event, EventResult, SeekType, SegmentFormat};
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
    /// Download mode: a byte seek at most this far ahead of the write
    /// position is absorbed — the queue skips forward through arriving data
    /// instead of asking the source to reposition (gstqueue2's forward
    /// threshold). Default: 512 KiB.
    pub seek_forward_threshold: u64,
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
            seek_forward_threshold: 512 * 1024,
        }
    }
}

// ============================================================================
// Buffering Action
// ============================================================================

/// Action determined by the buffering state machine.
///
/// Element mode records it but always forwards (see [`Element::process`]);
/// honoring Hold/Pause is blocked on the seek/flow wiring of #163/#164.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BufferingAction {
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

    /// End of the contiguous downloaded run containing `offset`, or `None`
    /// when `offset` is a hole. "How many bytes can be served locally from
    /// here before hitting the network" — the serve-from-disk query (#188).
    pub fn contiguous_run_end(&self, offset: u64) -> Option<u64> {
        self.ranges
            .iter()
            .find(|(start, end)| offset >= *start && offset < *end)
            .map(|(_, end)| *end)
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
// Ranges Handle (download mode, #164)
// ============================================================================

/// The state a [`Queue2RangesHandle`] shares with its downloading element —
/// download-mode [`Queue2`] or `HttpCacheSrc` (#188), which reuses the same
/// handle shape.
#[derive(Debug, Default)]
pub(crate) struct RangesShared {
    /// Merged downloaded spans, in true stream offsets. Uncontended: the
    /// queue takes it per buffer, handles on demand — never across an await.
    pub(crate) ranges: Mutex<DownloadedRanges>,
    /// Total expected stream size; 0 = unknown.
    pub(crate) total: AtomicU64,
    /// Stream offset of the next byte the queue expects to receive.
    pub(crate) write_pos: AtomicU64,
}

/// Cloneable, poll-anytime view of a download-mode [`Queue2`]'s progress.
///
/// Clone it via [`Queue2::control()`](crate::control::Controllable::control)
/// *before* the pipeline starts — `Executor::start` moves the element into
/// its task. A seek-bar UI shades [`ranges`](Self::ranges) to show which
/// byte offsets are already local; the same data is posted (throttled) as
/// [`MessageKind::DownloadProgress`].
#[derive(Debug, Clone)]
pub struct Queue2RangesHandle {
    inner: Arc<RangesShared>,
}

impl Queue2RangesHandle {
    /// Handle over an element's shared ranges state (#188: also minted by
    /// `HttpCacheSrc`).
    #[cfg(feature = "http")]
    pub(crate) fn from_shared(inner: Arc<RangesShared>) -> Self {
        Self { inner }
    }

    /// Merged downloaded byte spans `(start, end)`, ascending.
    pub fn ranges(&self) -> Vec<(u64, u64)> {
        self.inner.ranges.lock().unwrap().as_slice().to_vec()
    }

    /// Total bytes downloaded so far (across all spans).
    pub fn total_bytes(&self) -> u64 {
        self.inner.ranges.lock().unwrap().total_bytes()
    }

    /// Total expected stream size, when known.
    pub fn total(&self) -> Option<u64> {
        match self.inner.total.load(Ordering::Relaxed) {
            0 => None,
            n => Some(n),
        }
    }

    /// Whether `offset` falls inside a downloaded span (an instant seek).
    pub fn contains(&self, offset: u64) -> bool {
        self.inner.ranges.lock().unwrap().contains(offset)
    }

    /// Stream offset of the next byte the queue expects to receive.
    pub fn write_pos(&self) -> u64 {
        self.inner.write_pos.load(Ordering::Relaxed)
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
/// Designed for network streaming scenarios where buffering strategies are
/// needed to handle variable network conditions. (Plain queueing needs no
/// element — the link channel is the queue.)
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
    /// The download/timeshift file's physical cursor. In download mode this
    /// tracks where the next `write_all` lands, so a stream discontinuity
    /// (post-seek data) knows to `seek` first — leaving an honest sparse
    /// hole rather than misfiled bytes.
    file_write_pos: u64,
    /// True stream offset of the next expected incoming byte (#164). Fed by
    /// each buffer's `Metadata.offset` (authoritative) and re-anchored by
    /// byte `Segment` events for unstamped sources.
    stream_pos: u64,
    /// Whether the last accepted buffer carried `Metadata.offset`. Seek
    /// absorption is gated on it: without true offsets the queue cannot
    /// know what "skip forward to byte N" means.
    offset_stamped: bool,
    /// An absorbed forward seek: record-but-discard incoming data until
    /// this stream offset arrives.
    skip_until: Option<u64>,
    /// Downloaded ranges + progress, shared with [`Queue2RangesHandle`]s.
    ranges: Arc<RangesShared>,
}

struct RateState {
    last_time: Instant,
    last_bytes_in: u64,
    last_bytes_out: u64,
    avg_in_rate: f64,
    avg_out_rate: f64,
}

impl Queue2 {
    /// Drop the in-memory ring and its byte accounting — a flushing seek
    /// abandoned that data. File-backed state (the download/timeshift file,
    /// its write position, `DownloadedRanges`) deliberately survives: what
    /// is on disk is still on disk after a seek, and #164's buffering-aware
    /// seeking builds on exactly that.
    fn clear_buffered(&mut self) {
        self.ring.clear();
        self.current_bytes = 0;
        if matches!(
            self.config.mode,
            BufferingMode::Stream | BufferingMode::Live
        ) {
            self.percent = self.calculate_stream_percent();
            if !self.is_buffering {
                // Empty after a flush: the honest state is "buffering again",
                // and the bus should say so instead of a stale 100 %.
                self.is_buffering = true;
                self.post_buffering(self.percent);
            }
        }
    }

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
        let ranges = Arc::new(RangesShared::default());
        ranges
            .total
            .store(config.total_size.unwrap_or(0), Ordering::Relaxed);
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
            stream_pos: 0,
            offset_stamped: false,
            skip_until: None,
            ranges,
        }
    }

    /// Set watermark percentages.
    pub fn with_watermarks(mut self, low: u32, high: u32) -> Self {
        self.config.low_percent = low;
        self.config.high_percent = high;
        self
    }

    /// Set the forward-seek absorption threshold (download mode, #164).
    pub fn with_seek_forward_threshold(mut self, bytes: u64) -> Self {
        self.config.seek_forward_threshold = bytes;
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
    pub fn downloaded_ranges(&self) -> DownloadedRanges {
        self.ranges.ranges.lock().unwrap().clone()
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
        // Ring fill is only what `percent` means in stream/live mode;
        // download mode's percent is downloaded-bytes-of-total and must not
        // be clobbered by the forwarding ring draining.
        if matches!(
            self.config.mode,
            BufferingMode::Stream | BufferingMode::Live
        ) {
            self.percent = self.calculate_stream_percent();
        }
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
        let len = buffer.len() as u64;

        // The buffer's stamped offset is authoritative (FileSrc/HttpSrc set
        // it, and it stays correct across seeks and the executor's
        // epoch-shed gaps); an unstamped buffer is assumed contiguous.
        let stamped = buffer.metadata().offset;
        self.offset_stamped = stamped.is_some();
        let offset = stamped.unwrap_or(self.stream_pos);
        let end = offset + len;

        // Ensure file is open. Random-access (a post-seek discontinuity
        // writes at its true offset, leaving an honest sparse hole), so the
        // physical cursor must be repositioned when the stream jumps.
        if self.file.is_none() {
            let path = self
                .config
                .temp_file
                .as_ref()
                .ok_or_else(|| Error::Config("download mode requires temp_file path".into()))?;
            self.file = Some(
                File::options()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)?,
            );
        }
        let file = self.file.as_mut().unwrap();
        if offset != self.file_write_pos {
            file.seek(SeekFrom::Start(offset))?;
        }
        file.write_all(buffer.as_bytes())?;
        self.file_write_pos = end;
        self.stream_pos = end;
        self.bytes_in += len;
        self.buffers_in += 1;

        {
            let mut ranges = self.ranges.ranges.lock().unwrap();
            ranges.add(offset, len);
            // Percentage from actually-downloaded bytes: sparse-safe where
            // the old write-cursor arithmetic would lie after a seek.
            if let Some(total) = self.config.total_size
                && total > 0
            {
                self.percent = ((ranges.total_bytes() * 100) / total).min(100) as u32;
            }
        }
        self.ranges.write_pos.store(end, Ordering::Relaxed);

        // An absorbed forward seek (#164): everything up to the target is
        // still part of the download — recorded above — but not forwarded.
        // A straddling buffer forwards its tail from the target on.
        let forward = match self.skip_until {
            Some(target) if end <= target => None,
            Some(target) => {
                self.skip_until = None;
                if offset < target {
                    Some(buffer.slice((target - offset) as usize, (end - target) as usize))
                } else {
                    // The epoch-shed window overshot the target; forward
                    // from the true offset and let downstream resync.
                    Some(buffer)
                }
            }
            None => Some(buffer),
        };
        if let Some(buf) = forward {
            self.current_bytes += buf.len();
            self.ring.push_back(buf);
        }

        if self.update_rate_estimates() {
            self.post_download_progress();
        }
        self.post_buffering(self.percent);

        Ok(BufferingAction::Forward)
    }

    /// Post the range-structured download progress (throttled by callers).
    fn post_download_progress(&self) {
        let Some(ref bus) = self.bus else { return };
        if self.config.mode != BufferingMode::Download {
            return;
        }
        let ranges = self.ranges.ranges.lock().unwrap().as_slice().to_vec();
        bus.post(MessageKind::DownloadProgress {
            ranges,
            total: self.config.total_size,
            write_pos: self.stream_pos,
        });
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

    /// Returns `true` when the 500 ms window elapsed and the estimates were
    /// refreshed — the cadence throttled bus traffic rides on.
    fn update_rate_estimates(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.rate_state.last_time).as_secs_f64();

        // Update every 500ms
        if elapsed < 0.5 {
            return false;
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
        true
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
    // #189: may forward the input buffer — the upstream producer's arena
    // budget accumulates through this element.
    fn passthrough(&self) -> bool {
        true
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Record-and-forward: the push updates stats/percent and posts
        // Buffering bus messages, but element mode never withholds data —
        // honoring the state machine's Hold/Pause needs the seek/flow
        // wiring tracked by #163/#164 pt2.
        match self.config.mode {
            BufferingMode::Stream | BufferingMode::Live => {
                let _ = self.process_stream_push(buffer);
            }
            BufferingMode::Download => {
                self.process_download_push(buffer)?;
            }
            BufferingMode::Timeshift => {
                self.process_timeshift_push(buffer)?;
            }
        }
        Ok(self.stream_pop())
    }

    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        match &event {
            // Same rule as Queue: a flushing seek drains buffering queues by
            // propagation (#162). A FlushStart also cancels an absorbed
            // skip: it means a *newer* seek was handled at or above the
            // source, superseding whatever this queue was skipping toward.
            Event::FlushStart => {
                self.clear_buffered();
                if self.skip_until.take().is_some() {
                    tracing::debug!("queue2: absorbed seek superseded by a flush from upstream");
                }
                self.post_download_progress();
            }
            // A byte Segment is the source declaring its true position —
            // the initial Bytes-from-0 segment and every post-seek one
            // (#165/#163). It re-anchors the stream cursor for unstamped
            // sources (stamped buffers override per buffer anyway), and its
            // `stop` carries the stream total when the source knows it.
            Event::Segment(seg) if seg.format == SegmentFormat::Bytes => {
                let start = seg.start.max(0) as u64;
                self.stream_pos = start;
                self.ranges.write_pos.store(start, Ordering::Relaxed);
                if seg.stop >= 0 {
                    let total = seg.stop as u64;
                    if self.config.total_size.is_none() {
                        self.config.total_size = Some(total);
                    }
                    self.ranges.total.store(total, Ordering::Relaxed);
                }
            }
            _ => {}
        }
        Some(event)
    }

    /// Absorb byte seeks the arriving stream will satisfy anyway (#164).
    ///
    /// Download mode only, and only when the input carries true offsets
    /// (`Metadata.offset`, stamped by `FileSrc`/`HttpSrc`): a `Set` seek at
    /// most `seek_forward_threshold` ahead of the write position answers
    /// `Handled` — the executor runs the flush trio downstream from this
    /// element's task and this queue discards (but still records) arriving
    /// data until the target byte. No upstream traffic, no re-download.
    ///
    /// Everything else — backward seeks (even into downloaded spans: this
    /// element cannot emit without input, so it cannot serve from disk),
    /// far-forward seeks, relative seeks, unstamped input — returns
    /// `NotHandled`, which forwards the seek toward the source. The stale
    /// in-flight backlog is the executor's flush-epoch shed; the source's
    /// post-seek byte Segment re-anchors this queue on the way back down.
    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        let Event::Seek(seek) = event else {
            return EventResult::NotHandled;
        };
        if self.config.mode != BufferingMode::Download
            || seek.format != SegmentFormat::Bytes
            || seek.start.seek_type != SeekType::Set
            || !self.offset_stamped
        {
            return EventResult::NotHandled;
        }
        let target = seek.start.position.max(0) as u64;
        if target >= self.stream_pos
            && target - self.stream_pos <= self.config.seek_forward_threshold
        {
            tracing::debug!(
                "queue2: absorbing forward byte seek to {target} ({} ahead of write pos)",
                target - self.stream_pos
            );
            self.skip_until = Some(target);
            self.clear_buffered();
            return EventResult::handled_at(target as i64);
        }
        EventResult::NotHandled
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        // EOS drain: the executor calls this until `None`. (On FlushStart
        // the output is discarded and handle_downstream_event has already
        // cleared the ring, so this is naturally empty there.)
        let buf = self.stream_pop();
        if buf.is_none() {
            self.post_download_progress();
        }
        Ok(buf)
    }

    fn name(&self) -> &str {
        "queue2"
    }

    fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }
}

impl crate::control::Controllable for Queue2 {
    type Control = Queue2RangesHandle;

    /// A poll-anytime view of download progress (ranges/total/write_pos).
    ///
    /// Clone it *before* `executor.start()` — the element is moved into its
    /// task there. Meaningful in download mode; other modes report empty
    /// ranges.
    fn control(&self) -> Queue2RangesHandle {
        Queue2RangesHandle {
            inner: Arc::clone(&self.ranges),
        }
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

    /// A buffer stamped with its stream offset, like FileSrc/HttpSrc emit.
    fn make_buffer_at(size: usize, seq: u64, offset: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, size.min(256));
        let mut metadata = Metadata::from_sequence(seq);
        metadata.offset = Some(offset);
        Buffer::new(handle, metadata)
    }

    fn download_queue(total: Option<u64>) -> (Queue2, tempfile::NamedTempFile) {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let q = Queue2::download(temp.path(), total);
        (q, temp)
    }

    fn seek_bytes(q: &mut Queue2, target: u64) -> EventResult {
        Element::handle_upstream_event(q, &Event::Seek(crate::event::SeekEvent::new_bytes(target)))
    }

    #[test]
    fn flush_start_clears_the_buffered_contents() {
        use crate::element::Element;

        let mut q = Queue2::stream(1024);
        // Fill the ring through the push half alone (Element::process is
        // 1-in-1-out and would drain as it goes).
        for seq in 0..3 {
            let _ = q.process_stream_push(make_buffer(64, seq));
        }
        assert_eq!(q.ring.len(), 3);
        assert!(q.current_bytes > 0);

        let fwd = Element::handle_downstream_event(&mut q, crate::event::Event::FlushStart);
        assert!(matches!(fwd, Some(crate::event::Event::FlushStart)));
        assert!(q.ring.is_empty(), "FlushStart must clear the ring");
        assert_eq!(q.current_bytes, 0);
    }

    #[test]
    fn flush_re_enters_buffering_with_honest_stats() {
        use crate::element::Element;

        let mut q = Queue2::stream(200).with_watermarks(10, 90);
        // Fill past the high mark so buffering completes.
        let _ = q.process_stream_push(make_buffer(100, 0));
        let _ = q.process_stream_push(make_buffer(100, 1));
        assert!(!q.is_buffering());
        assert_eq!(q.buffering_percent(), 100);

        // A flushing seek empties the ring: percent must not stay a stale
        // 100 %, and the state machine re-enters buffering.
        let _ = Element::handle_downstream_event(&mut q, crate::event::Event::FlushStart);
        assert_eq!(q.buffering_percent(), 0);
        assert!(q.is_buffering(), "empty after flush = buffering again");
    }

    #[test]
    fn flush_leaves_download_file_state_intact() {
        use crate::element::Element;

        let dir = std::env::temp_dir().join(format!("parallax_q2_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dl.tmp");
        let mut q = Queue2::download(&path, Some(1000));
        q.process(make_buffer(100, 0)).unwrap();
        assert_eq!(q.downloaded_ranges().total_bytes(), 100);
        assert_eq!(q.file_write_pos, 100);

        // The on-disk state survives a flushing seek — that is what #164's
        // buffering-aware seeking will serve locally.
        let _ = Element::handle_downstream_event(&mut q, crate::event::Event::FlushStart);
        assert_eq!(q.downloaded_ranges().total_bytes(), 100);
        assert_eq!(q.file_write_pos, 100);

        let _ = std::fs::remove_dir_all(&dir);
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
    fn forward_seek_within_threshold_is_absorbed() {
        let (mut q, _temp) = download_queue(Some(10_000));
        // A stamped buffer arms absorption (offset_stamped).
        q.process(make_buffer_at(100, 0, 0)).unwrap();
        assert_eq!(q.stream_pos, 100);

        let result = seek_bytes(&mut q, 200);
        assert!(
            matches!(
                result,
                EventResult::Handled {
                    position: Some(200)
                }
            ),
            "100 bytes ahead is well inside the 512 KiB threshold"
        );
        assert_eq!(q.skip_until, Some(200));
        assert!(q.ring.is_empty(), "absorption clears the buffered ring");
    }

    #[test]
    fn seeks_the_queue_cannot_absorb_pass_through() {
        let (mut q, _temp) = download_queue(Some(100_000_000));
        q.process(make_buffer_at(100, 0, 0)).unwrap();

        // Beyond the forward threshold.
        assert!(matches!(
            seek_bytes(&mut q, 100 + 512 * 1024 + 1),
            EventResult::NotHandled
        ));
        // Backward — even into a downloaded span: serving from disk needs
        // output-without-input, which a transform does not have.
        assert!(matches!(seek_bytes(&mut q, 0), EventResult::NotHandled));
        // TIME seeks are someone else's job (a demuxer translates them).
        let time_seek = crate::event::SeekEvent::new_time(crate::clock::ClockTime::from_nanos(1));
        assert!(matches!(
            Element::handle_upstream_event(&mut q, &Event::Seek(time_seek)),
            EventResult::NotHandled
        ));
        assert_eq!(q.skip_until, None);
    }

    #[test]
    fn unstamped_input_disables_absorption() {
        let (mut q, _temp) = download_queue(Some(10_000));
        q.process(make_buffer(100, 0)).unwrap(); // no Metadata.offset
        assert!(
            matches!(seek_bytes(&mut q, 150), EventResult::NotHandled),
            "without true offsets the queue cannot know where 'byte 150' is"
        );
    }

    #[test]
    fn stream_mode_never_absorbs() {
        let mut q = Queue2::stream(1024);
        q.offset_stamped = true; // even if offsets were somehow present
        q.stream_pos = 0;
        assert!(matches!(seek_bytes(&mut q, 10), EventResult::NotHandled));
    }

    #[test]
    fn skipped_data_is_recorded_but_not_forwarded() {
        let (mut q, _temp) = download_queue(Some(1000));
        q.process(make_buffer_at(100, 0, 0)).unwrap();
        assert!(seek_bytes(&mut q, 250).is_handled());

        // Wholly before the target: recorded, not forwarded.
        let out = q.process(make_buffer_at(100, 1, 100)).unwrap();
        assert!(out.is_none(), "byte 100..200 is before the 250 target");
        assert!(
            q.downloaded_ranges().contains(150),
            "skipped bytes are still part of the download"
        );

        // Straddler: the tail from the target on is forwarded.
        let out = q.process(make_buffer_at(100, 2, 200)).unwrap();
        let buf = out.expect("byte 200..300 straddles the 250 target");
        assert_eq!(buf.len(), 50, "only the 250..300 tail forwards");
        assert_eq!(q.skip_until, None, "skip disarms once the target arrives");
        assert_eq!(q.downloaded_ranges().total_bytes(), 300);
    }

    #[test]
    fn a_byte_segment_reanchors_an_unstamped_stream() {
        let (mut q, _temp) = download_queue(None);
        // Unstamped contiguous data advances the cursor from zero…
        q.process(make_buffer(100, 0)).unwrap();
        assert_eq!(q.stream_pos, 100);

        // …and the post-seek byte Segment re-anchors it (the source's
        // declaration of where the stream now is), carrying the total.
        let seg = crate::event::SegmentEvent::new_bytes(500, Some(2000));
        let _ = Element::handle_downstream_event(&mut q, Event::Segment(seg));
        assert_eq!(q.stream_pos, 500);
        assert_eq!(q.config.total_size, Some(2000));

        q.process(make_buffer(100, 1)).unwrap();
        assert!(
            q.downloaded_ranges().contains(550),
            "post-anchor data lands at the anchored offset"
        );
    }

    #[test]
    fn an_offset_gap_leaves_two_honest_ranges_and_a_sparse_file() {
        let (mut q, temp) = download_queue(Some(1000));
        q.process(make_buffer_at(100, 0, 0)).unwrap();
        // The source repositioned (e.g. a far seek handled upstream).
        q.process(make_buffer_at(100, 1, 600)).unwrap();

        let ranges = q.downloaded_ranges();
        assert_eq!(ranges.as_slice(), &[(0, 100), (600, 700)]);
        assert!(!ranges.contains(300), "the hole is reported honestly");
        // The file is written at true offsets: 700 bytes long, sparse gap.
        let len = std::fs::metadata(temp.path()).unwrap().len();
        assert_eq!(len, 700);
        // Percent counts downloaded bytes, not the write cursor.
        assert_eq!(q.percent, 20, "200 of 1000 bytes, not 700 of 1000");
    }

    #[test]
    fn flush_start_cancels_an_absorbed_seek() {
        let (mut q, _temp) = download_queue(Some(1000));
        q.process(make_buffer_at(100, 0, 0)).unwrap();
        assert!(seek_bytes(&mut q, 200).is_handled());
        assert!(q.skip_until.is_some());

        // A newer seek handled at the source flushes through this queue —
        // the absorbed skip must not eat the post-seek stream.
        let _ = Element::handle_downstream_event(&mut q, Event::FlushStart);
        assert_eq!(q.skip_until, None);
        // File state still survives (the long-standing contract).
        assert_eq!(q.downloaded_ranges().total_bytes(), 100);
    }

    #[test]
    fn ranges_handle_reflects_progress_live() {
        use crate::control::Controllable;

        let (mut q, _temp) = download_queue(Some(1000));
        let handle = q.control();
        assert_eq!(handle.ranges(), vec![]);
        assert_eq!(handle.total(), Some(1000));

        q.process(make_buffer_at(100, 0, 0)).unwrap();
        q.process(make_buffer_at(100, 1, 600)).unwrap();
        assert_eq!(handle.ranges(), vec![(0, 100), (600, 700)]);
        assert_eq!(handle.total_bytes(), 200);
        assert!(handle.contains(650));
        assert!(!handle.contains(300));
        assert_eq!(handle.write_pos(), 700);
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
