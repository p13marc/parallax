//! Muxer synchronization support.
//!
//! This module provides types and utilities for implementing N-to-1 muxer elements
//! with PTS-based synchronization.
//!
//! # Overview
//!
//! Muxers combine multiple input streams (e.g., video, audio, metadata) into a
//! single multiplexed output. This requires careful synchronization to ensure
//! data from different streams is interleaved correctly based on timestamps.
//!
//! # Key Types
//!
//! - [`MuxerSyncState`]: Manages PTS-based synchronization across input pads
//! - `PadState`: Per-pad state including buffer queue and EOS tracking (internal)
//! - [`SyncMode`]: Synchronization strategy (strict, loose, timed)
//! - [`MuxerSyncConfig`]: Configuration for sync behavior
//! - [`StreamType`]: Classification of stream types
//! - [`PadInfo`]: Information about a muxer input pad
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::element::muxer::{MuxerSyncState, MuxerSyncConfig, PadInfo, StreamType};
//! use parallax::element::PadId;
//!
//! // Create sync state with 40ms output interval (25fps video)
//! let mut sync = MuxerSyncState::new(MuxerSyncConfig {
//!     output_interval_ms: 40,
//!     ..Default::default()
//! });
//!
//! // Add pads
//! let video_pad = sync.add_pad(PadInfo {
//!     name: "video_0".to_string(),
//!     stream_type: StreamType::Video,
//!     required: true,
//! });
//!
//! let audio_pad = sync.add_pad(PadInfo {
//!     name: "audio_0".to_string(),
//!     stream_type: StreamType::Audio,
//!     required: true,
//! });
//!
//! // Push buffers
//! sync.push(video_pad, video_buffer)?;
//! sync.push(audio_pad, audio_buffer)?;
//!
//! // Check if ready to output
//! if sync.ready_to_output() {
//!     let buffers = sync.collect_for_output();
//!     // Mux buffers...
//!     sync.advance();
//! }
//! ```

use crate::buffer::Buffer;
use crate::clock::ClockTime;
use crate::element::PadId;
use crate::error::{Error, Result};
use crate::format::Caps;

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

// ============================================================================
// Stream Types
// ============================================================================

/// Classification of stream types for muxing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// Video stream (H.264, H.265, AV1, etc.).
    Video,
    /// Audio stream (AAC, MP3, etc.).
    Audio,
    /// Subtitle stream (text, bitmap).
    Subtitle,
    /// Data/metadata stream (KLV, SCTE-35, etc.).
    Data,
}

impl StreamType {
    /// Returns the name of this stream type.
    pub fn name(&self) -> &'static str {
        match self {
            StreamType::Video => "video",
            StreamType::Audio => "audio",
            StreamType::Subtitle => "subtitle",
            StreamType::Data => "data",
        }
    }
}

// ============================================================================
// Pad Info
// ============================================================================

/// Information about a muxer input pad.
#[derive(Debug, Clone)]
pub struct PadInfo {
    /// Name of the pad (e.g., "video_0", "audio_0").
    pub name: String,
    /// Type of stream on this pad.
    pub stream_type: StreamType,
    /// Whether this pad must have data before outputting.
    ///
    /// For strict sync mode, output waits for all required pads.
    /// Video pads are typically required; metadata pads are optional.
    pub required: bool,
}

impl PadInfo {
    /// Create a new pad info.
    pub fn new(name: impl Into<String>, stream_type: StreamType) -> Self {
        Self {
            name: name.into(),
            stream_type,
            required: stream_type == StreamType::Video,
        }
    }

    /// Mark this pad as required.
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Mark this pad as optional.
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }
}

// ============================================================================
// Sync Configuration
// ============================================================================

/// Synchronization mode for muxer output timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// Adaptive: strict for non-live, latency-bounded for live.
    ///
    /// This is the recommended mode for most use cases.
    #[default]
    Auto,

    /// Wait for all required streams before outputting.
    ///
    /// Ensures proper A/V sync but may introduce latency.
    Strict,

    /// Output when primary stream (video) is ready.
    ///
    /// Lower latency but may have brief sync issues.
    Loose,

    /// Output at fixed intervals regardless of input timing.
    ///
    /// Best for real-time streaming with known frame rate.
    Timed {
        /// Output interval in milliseconds.
        interval_ms: u64,
    },
}

/// Configuration for muxer synchronization.
#[derive(Debug, Clone)]
pub struct MuxerSyncConfig {
    /// Synchronization mode.
    pub mode: SyncMode,
    /// Output interval in milliseconds (used for Timed mode and Auto initial interval).
    ///
    /// Default: 40ms (25fps).
    pub output_interval_ms: u64,
    /// Maximum wait time for slow streams in live mode.
    ///
    /// If a required stream doesn't have data within this time, output anyway.
    /// Default: 200ms.
    pub latency: Duration,
    /// Timeout for sparse streams (e.g., metadata at 10Hz).
    ///
    /// If an optional stream doesn't send data within this time, don't wait.
    /// Default: 500ms.
    pub sparse_timeout: Duration,
    /// Whether this is a live pipeline.
    ///
    /// Live pipelines use latency-bounded sync; non-live use strict sync.
    /// Default: false.
    pub is_live: bool,
}

impl Default for MuxerSyncConfig {
    fn default() -> Self {
        Self {
            mode: SyncMode::Auto,
            output_interval_ms: 40, // 25fps
            latency: Duration::from_millis(200),
            sparse_timeout: Duration::from_millis(500),
            is_live: false,
        }
    }
}

impl MuxerSyncConfig {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sync mode.
    pub fn with_mode(mut self, mode: SyncMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the output interval in milliseconds.
    pub fn with_interval_ms(mut self, interval_ms: u64) -> Self {
        self.output_interval_ms = interval_ms;
        self
    }

    /// Set the latency bound for live mode.
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    /// Set sparse stream timeout.
    pub fn with_sparse_timeout(mut self, timeout: Duration) -> Self {
        self.sparse_timeout = timeout;
        self
    }

    /// Mark as live pipeline.
    pub fn live(mut self) -> Self {
        self.is_live = true;
        self
    }
}

// ============================================================================
// Timestamped Buffer
// ============================================================================

/// A buffer with its presentation timestamp extracted.
#[derive(Debug)]
struct TimestampedBuffer {
    /// The buffer data.
    buffer: Buffer,
    /// Presentation timestamp (extracted from buffer metadata).
    pts: ClockTime,
}

impl TimestampedBuffer {
    fn new(buffer: Buffer) -> Self {
        let pts = buffer.metadata().pts;
        Self { buffer, pts }
    }
}

// ============================================================================
// Pad State
// ============================================================================

/// Runtime state for a muxer input pad.
#[derive(Debug)]
struct PadState {
    /// Pad information.
    info: PadInfo,
    /// Buffered data for this pad.
    queue: VecDeque<TimestampedBuffer>,
    /// Last PTS seen on this pad.
    last_pts: ClockTime,
    /// Whether EOS has been received on this pad.
    eos: bool,
    /// Caps for this pad.
    caps: Caps,
}

impl PadState {
    fn new(info: PadInfo) -> Self {
        Self {
            info,
            queue: VecDeque::new(),
            last_pts: ClockTime::ZERO,
            eos: false,
            caps: Caps::any(),
        }
    }

    /// Check if this pad has data available.
    fn has_data(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Check if this pad has data at or past the target PTS.
    fn has_data_for(&self, target_pts: ClockTime) -> bool {
        // Has buffered data with PTS <= target
        let has_ready_data = self
            .queue
            .front()
            .map(|b| b.pts <= target_pts)
            .unwrap_or(false);

        // Has seen data past the target (so we won't get more data for this interval)
        // Only counts if we've actually seen data (last_pts > 0 or has data in queue)
        let has_passed_target =
            self.last_pts > target_pts || (self.last_pts == target_pts && !self.queue.is_empty());

        // Pad is at EOS (no more data coming)
        has_ready_data || has_passed_target || self.eos
    }

    /// Get the oldest buffer's PTS.
    #[allow(dead_code)]
    fn oldest_pts(&self) -> Option<ClockTime> {
        self.queue.front().map(|b| b.pts)
    }

    /// Pop buffers up to the target PTS.
    fn pop_up_to(&mut self, target_pts: ClockTime) -> Vec<Buffer> {
        let mut buffers = Vec::new();
        while let Some(front) = self.queue.front() {
            if front.pts <= target_pts {
                buffers.push(self.queue.pop_front().unwrap().buffer);
            } else {
                break;
            }
        }
        buffers
    }
}

// ============================================================================
// Collected Input
// ============================================================================

/// A collected input from a muxer pad.
#[derive(Debug)]
pub struct CollectedInput {
    /// The pad ID.
    pub pad_id: PadId,
    /// The buffer.
    pub buffer: Buffer,
    /// The stream type.
    pub stream_type: StreamType,
}

// ============================================================================
// Muxer Sync State
// ============================================================================

/// Synchronization state for a muxer element.
///
/// This manages PTS-based synchronization across multiple input pads,
/// ensuring proper interleaving of streams in the output.
#[derive(Debug)]
pub struct MuxerSyncState {
    /// Configuration.
    config: MuxerSyncConfig,
    /// Per-pad state.
    pads: HashMap<PadId, PadState>,
    /// Next pad ID to assign.
    next_pad_id: u32,
    /// Target PTS for next output.
    target_pts: ClockTime,
    /// Output interval as ClockTime.
    interval: ClockTime,
    /// Whether we've seen any data yet.
    started: bool,
}

impl MuxerSyncState {
    /// Create a new sync state with the given configuration.
    pub fn new(config: MuxerSyncConfig) -> Self {
        let interval = ClockTime::from_millis(config.output_interval_ms);
        Self {
            config,
            pads: HashMap::new(),
            next_pad_id: 0,
            target_pts: ClockTime::ZERO,
            interval,
            started: false,
        }
    }

    /// Add a new input pad.
    ///
    /// Returns the pad ID assigned to this pad.
    pub fn add_pad(&mut self, info: PadInfo) -> PadId {
        let pad_id = PadId::new(self.next_pad_id);
        self.next_pad_id += 1;
        self.pads.insert(pad_id, PadState::new(info));
        pad_id
    }

    /// Remove a pad.
    pub fn remove_pad(&mut self, pad_id: PadId) -> bool {
        self.pads.remove(&pad_id).is_some()
    }

    /// Get pad info.
    pub fn pad_info(&self, pad_id: PadId) -> Option<&PadInfo> {
        self.pads.get(&pad_id).map(|s| &s.info)
    }

    /// Get all pad infos.
    pub fn pad_infos(&self) -> Vec<PadInfo> {
        self.pads.values().map(|s| s.info.clone()).collect()
    }

    /// Set caps for a pad.
    pub fn set_pad_caps(&mut self, pad_id: PadId, caps: Caps) {
        if let Some(state) = self.pads.get_mut(&pad_id) {
            state.caps = caps;
        }
    }

    /// Get caps for a pad.
    pub fn pad_caps(&self, pad_id: PadId) -> Option<&Caps> {
        self.pads.get(&pad_id).map(|s| &s.caps)
    }

    /// Push a buffer to a pad.
    pub fn push(&mut self, pad_id: PadId, buffer: Buffer) -> Result<()> {
        let state = self
            .pads
            .get_mut(&pad_id)
            .ok_or_else(|| Error::Element(format!("Unknown pad ID: {:?}", pad_id)))?;

        if state.eos {
            return Err(Error::Element(format!("Pad {:?} already at EOS", pad_id)));
        }

        let ts_buf = TimestampedBuffer::new(buffer);

        // Update last PTS
        if ts_buf.pts > state.last_pts {
            state.last_pts = ts_buf.pts;
        }

        // Initialize target PTS from first buffer
        if !self.started {
            self.target_pts = ts_buf.pts;
            self.started = true;
        }

        state.queue.push_back(ts_buf);
        Ok(())
    }

    /// Signal EOS on a pad.
    pub fn set_eos(&mut self, pad_id: PadId) {
        if let Some(state) = self.pads.get_mut(&pad_id) {
            state.eos = true;
        }
    }

    /// Check if a pad is at EOS.
    pub fn is_eos(&self, pad_id: PadId) -> bool {
        self.pads.get(&pad_id).map(|s| s.eos).unwrap_or(true)
    }

    /// Check if all pads are at EOS.
    pub fn all_eos(&self) -> bool {
        self.pads.values().all(|s| s.eos)
    }

    /// Check if there is any buffered data.
    pub fn has_buffered_data(&self) -> bool {
        self.pads.values().any(|s| s.has_data())
    }

    /// Get the current target PTS.
    pub fn target_pts(&self) -> ClockTime {
        self.target_pts
    }

    /// Check if the muxer is ready to produce output.
    ///
    /// The behavior depends on the sync mode:
    /// - **Strict**: All required pads must have data up to target PTS
    /// - **Loose**: Any video pad has data
    /// - **Timed**: Based on elapsed time since last output
    /// - **Auto**: Strict for non-live, latency-bounded for live
    pub fn ready_to_output(&self) -> bool {
        if !self.started {
            return false;
        }

        match self.effective_mode() {
            SyncMode::Strict | SyncMode::Auto => self.strict_ready(),
            SyncMode::Loose => self.loose_ready(),
            SyncMode::Timed { .. } => self.timed_ready(),
        }
    }

    /// Get the effective sync mode (resolves Auto).
    fn effective_mode(&self) -> SyncMode {
        match self.config.mode {
            SyncMode::Auto => {
                if self.config.is_live {
                    SyncMode::Loose
                } else {
                    SyncMode::Strict
                }
            }
            other => other,
        }
    }

    /// Check if ready in strict mode.
    fn strict_ready(&self) -> bool {
        for state in self.pads.values() {
            if state.info.required && !state.eos {
                // Required pad must have data at or past target PTS
                if !state.has_data_for(self.target_pts) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if ready in loose mode.
    fn loose_ready(&self) -> bool {
        // Ready when any video pad has data
        self.pads
            .values()
            .any(|state| state.info.stream_type == StreamType::Video && state.has_data())
    }

    /// Check if ready in timed mode.
    fn timed_ready(&self) -> bool {
        // In timed mode, check if we have any data to output
        // The actual timing is managed externally by a timer
        self.has_buffered_data()
    }

    /// Collect all buffers up to the target PTS for output.
    ///
    /// Returns a vector of (PadId, Buffer, StreamType) tuples.
    pub fn collect_for_output(&mut self) -> Vec<CollectedInput> {
        let mut inputs = Vec::new();

        for (&pad_id, state) in &mut self.pads {
            let stream_type = state.info.stream_type;
            let buffers = state.pop_up_to(self.target_pts);

            for buffer in buffers {
                inputs.push(CollectedInput {
                    pad_id,
                    buffer,
                    stream_type,
                });
            }
        }

        // Sort by PTS for proper interleaving
        inputs.sort_by_key(|i| i.buffer.metadata().pts);

        inputs
    }

    /// Advance the target PTS by one interval.
    pub fn advance(&mut self) {
        self.target_pts += self.interval;
    }

    /// Advance the target PTS to a specific time.
    pub fn advance_to(&mut self, pts: ClockTime) {
        self.target_pts = pts;
    }

    /// Flush all remaining data.
    ///
    /// Returns all buffered data regardless of target PTS.
    pub fn flush(&mut self) -> Vec<CollectedInput> {
        let mut inputs = Vec::new();

        for (&pad_id, state) in &mut self.pads {
            let stream_type = state.info.stream_type;

            while let Some(ts_buf) = state.queue.pop_front() {
                inputs.push(CollectedInput {
                    pad_id,
                    buffer: ts_buf.buffer,
                    stream_type,
                });
            }
        }

        // Sort by PTS
        inputs.sort_by_key(|i| i.buffer.metadata().pts);

        inputs
    }

    /// Reset the sync state.
    pub fn reset(&mut self) {
        for state in self.pads.values_mut() {
            state.queue.clear();
            state.last_pts = ClockTime::ZERO;
            state.eos = false;
        }
        self.target_pts = ClockTime::ZERO;
        self.started = false;
    }

    /// Get statistics about the sync state.
    pub fn stats(&self) -> MuxerSyncStats {
        let mut total_queued = 0;
        let mut pads_with_data = 0;
        let mut pads_at_eos = 0;

        for state in self.pads.values() {
            total_queued += state.queue.len();
            if state.has_data() {
                pads_with_data += 1;
            }
            if state.eos {
                pads_at_eos += 1;
            }
        }

        MuxerSyncStats {
            total_pads: self.pads.len(),
            pads_with_data,
            pads_at_eos,
            total_queued_buffers: total_queued,
            target_pts: self.target_pts,
        }
    }
}

/// Statistics about the muxer sync state.
#[derive(Debug, Clone)]
pub struct MuxerSyncStats {
    /// Total number of pads.
    pub total_pads: usize,
    /// Number of pads with queued data.
    pub pads_with_data: usize,
    /// Number of pads at EOS.
    pub pads_at_eos: usize,
    /// Total number of queued buffers across all pads.
    pub total_queued_buffers: usize,
    /// Current target PTS.
    pub target_pts: ClockTime,
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

    // Use a shared arena for tests to avoid creating many arenas
    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(64, 64).unwrap())
    }

    fn make_buffer(pts_ms: u64, seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let mut metadata = Metadata::from_sequence(seq);
        metadata.pts = ClockTime::from_millis(pts_ms);
        Buffer::new(handle, metadata)
    }

    #[test]
    fn test_stream_type_name() {
        assert_eq!(StreamType::Video.name(), "video");
        assert_eq!(StreamType::Audio.name(), "audio");
        assert_eq!(StreamType::Subtitle.name(), "subtitle");
        assert_eq!(StreamType::Data.name(), "data");
    }

    #[test]
    fn test_pad_info_builder() {
        let info = PadInfo::new("video_0", StreamType::Video);
        assert_eq!(info.name, "video_0");
        assert!(info.required); // Video is required by default

        let info = PadInfo::new("data_0", StreamType::Data).required();
        assert!(info.required);

        let info = PadInfo::new("audio_0", StreamType::Audio).optional();
        assert!(!info.required);
    }

    #[test]
    fn test_sync_config_builder() {
        let config = MuxerSyncConfig::new()
            .with_mode(SyncMode::Strict)
            .with_interval_ms(33)
            .with_latency(Duration::from_millis(100))
            .live();

        assert_eq!(config.mode, SyncMode::Strict);
        assert_eq!(config.output_interval_ms, 33);
        assert_eq!(config.latency, Duration::from_millis(100));
        assert!(config.is_live);
    }

    #[test]
    fn test_muxer_sync_add_pads() {
        let mut sync = MuxerSyncState::new(MuxerSyncConfig::default());

        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video));
        let audio_pad = sync.add_pad(PadInfo::new("audio_0", StreamType::Audio));

        assert_eq!(video_pad.0, 0);
        assert_eq!(audio_pad.0, 1);

        assert_eq!(sync.pad_infos().len(), 2);
    }

    #[test]
    fn test_muxer_sync_push_buffer() {
        let mut sync = MuxerSyncState::new(MuxerSyncConfig::default());
        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video));

        let buffer = make_buffer(0, 0);
        sync.push(video_pad, buffer).unwrap();

        assert!(sync.has_buffered_data());
    }

    #[test]
    fn test_muxer_sync_push_unknown_pad() {
        let mut sync = MuxerSyncState::new(MuxerSyncConfig::default());
        let buffer = make_buffer(0, 0);

        let result = sync.push(PadId::new(999), buffer);
        assert!(result.is_err());
    }

    #[test]
    fn test_muxer_sync_eos() {
        let mut sync = MuxerSyncState::new(MuxerSyncConfig::default());
        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video));
        let audio_pad = sync.add_pad(PadInfo::new("audio_0", StreamType::Audio));

        assert!(!sync.is_eos(video_pad));
        assert!(!sync.all_eos());

        sync.set_eos(video_pad);
        assert!(sync.is_eos(video_pad));
        assert!(!sync.all_eos());

        sync.set_eos(audio_pad);
        assert!(sync.all_eos());
    }

    #[test]
    fn test_muxer_sync_strict_ready() {
        let mut sync = MuxerSyncState::new(
            MuxerSyncConfig::new()
                .with_mode(SyncMode::Strict)
                .with_interval_ms(40),
        );

        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video).required());
        let audio_pad = sync.add_pad(PadInfo::new("audio_0", StreamType::Audio).required());

        // Not ready initially (no data)
        assert!(!sync.ready_to_output());

        // Push video buffer
        sync.push(video_pad, make_buffer(0, 0)).unwrap();

        // Still not ready (missing audio)
        assert!(!sync.ready_to_output());

        // Push audio buffer
        sync.push(audio_pad, make_buffer(0, 1)).unwrap();

        // Now ready
        assert!(sync.ready_to_output());
    }

    #[test]
    fn test_muxer_sync_loose_ready() {
        let mut sync = MuxerSyncState::new(
            MuxerSyncConfig::new()
                .with_mode(SyncMode::Loose)
                .with_interval_ms(40),
        );

        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video));
        let _audio_pad = sync.add_pad(PadInfo::new("audio_0", StreamType::Audio));

        // Not ready initially
        assert!(!sync.ready_to_output());

        // Push video buffer only
        sync.push(video_pad, make_buffer(0, 0)).unwrap();

        // Ready (loose mode only needs video)
        assert!(sync.ready_to_output());
    }

    #[test]
    fn test_muxer_sync_collect_output() {
        let mut sync = MuxerSyncState::new(
            MuxerSyncConfig::new()
                .with_mode(SyncMode::Strict)
                .with_interval_ms(40),
        );

        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video));
        let audio_pad = sync.add_pad(PadInfo::new("audio_0", StreamType::Audio).optional());

        // Push buffers
        sync.push(video_pad, make_buffer(0, 0)).unwrap();
        sync.push(video_pad, make_buffer(40, 1)).unwrap();
        sync.push(audio_pad, make_buffer(20, 2)).unwrap();

        // Collect for first interval (target_pts = 0, so collect pts <= 0)
        // After first push, target_pts is set to 0
        let collected = sync.collect_for_output();
        assert_eq!(collected.len(), 1); // Only buffer at pts=0

        // Advance to next interval
        sync.advance();
        assert_eq!(sync.target_pts(), ClockTime::from_millis(40));

        // Collect again
        let collected = sync.collect_for_output();
        assert_eq!(collected.len(), 2); // Buffer at pts=20 and pts=40
    }

    #[test]
    fn test_muxer_sync_flush() {
        let mut sync = MuxerSyncState::new(MuxerSyncConfig::default());
        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video));

        sync.push(video_pad, make_buffer(0, 0)).unwrap();
        sync.push(video_pad, make_buffer(100, 1)).unwrap();
        sync.push(video_pad, make_buffer(200, 2)).unwrap();

        let flushed = sync.flush();
        assert_eq!(flushed.len(), 3);
        assert!(!sync.has_buffered_data());
    }

    #[test]
    fn test_muxer_sync_reset() {
        let mut sync = MuxerSyncState::new(MuxerSyncConfig::default());
        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video));

        sync.push(video_pad, make_buffer(100, 0)).unwrap();
        sync.advance();

        assert!(sync.has_buffered_data());
        assert!(sync.target_pts() > ClockTime::ZERO);

        sync.reset();

        assert!(!sync.has_buffered_data());
        assert_eq!(sync.target_pts(), ClockTime::ZERO);
    }

    #[test]
    fn test_muxer_sync_stats() {
        let mut sync = MuxerSyncState::new(MuxerSyncConfig::default());
        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video));
        let audio_pad = sync.add_pad(PadInfo::new("audio_0", StreamType::Audio));

        sync.push(video_pad, make_buffer(0, 0)).unwrap();
        sync.push(video_pad, make_buffer(40, 1)).unwrap();
        sync.set_eos(audio_pad);

        let stats = sync.stats();
        assert_eq!(stats.total_pads, 2);
        assert_eq!(stats.pads_with_data, 1);
        assert_eq!(stats.pads_at_eos, 1);
        assert_eq!(stats.total_queued_buffers, 2);
    }

    #[test]
    fn test_muxer_sync_eos_satisfies_required() {
        let mut sync = MuxerSyncState::new(
            MuxerSyncConfig::new()
                .with_mode(SyncMode::Strict)
                .with_interval_ms(40),
        );

        let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video).required());
        let audio_pad = sync.add_pad(PadInfo::new("audio_0", StreamType::Audio).required());

        // Push video only
        sync.push(video_pad, make_buffer(0, 0)).unwrap();

        // Not ready (audio required but no data)
        assert!(!sync.ready_to_output());

        // Mark audio as EOS
        sync.set_eos(audio_pad);

        // Now ready (EOS satisfies required)
        assert!(sync.ready_to_output());
    }

    #[test]
    fn test_muxer_sync_auto_mode() {
        // Non-live auto mode should behave like strict
        let sync = MuxerSyncState::new(MuxerSyncConfig::new().with_mode(SyncMode::Auto));
        assert_eq!(sync.effective_mode(), SyncMode::Strict);

        // Live auto mode should behave like loose
        let sync = MuxerSyncState::new(MuxerSyncConfig::new().with_mode(SyncMode::Auto).live());
        assert_eq!(sync.effective_mode(), SyncMode::Loose);
    }
}
