//! MPEG-TS muxer element wrapper.
//!
//! This module provides [`TsMuxElement`], a pipeline element that wraps the
//! low-level [`TsMux`] for use in pipelines with proper N-to-1 synchronization.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::mux::{TsMuxElement, TsMuxConfig, TsMuxTrack, TsMuxStreamType};
//! use parallax::pipeline::Pipeline;
//!
//! // Create muxer element
//! let config = TsMuxConfig::new()
//!     .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
//!     .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());
//!
//! let mux_element = TsMuxElement::new(config)?;
//!
//! // In a pipeline
//! let mut pipeline = Pipeline::new();
//! let mux = pipeline.add_node("mux", DynAsyncElement::new_box(MuxerAdapter::new(mux_element)));
//!
//! // Link multiple inputs
//! pipeline.link_pads(video_encoder, "src", mux, "video_0")?;
//! pipeline.link_pads(klv_source, "src", mux, "data_0")?;
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::muxer::{CollectedInput, MuxerSyncConfig, MuxerSyncState, PadInfo, StreamType};
use crate::element::{Muxer, MuxerInput, PadAddedCallback, PadId};
use crate::elements::mux::{TsMux, TsMuxConfig, TsMuxStreamType, TsMuxTrack};
use crate::error::{Error, Result};
use crate::format::{AudioCodec, Caps, MediaFormat, VideoCodec};
use crate::memory::{CpuSegment, MemorySegment};
use crate::metadata::Metadata;

use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// TsMuxElement
// ============================================================================

/// MPEG-TS muxer element for pipeline use.
///
/// This wraps [`TsMux`] with proper input pad management and PTS-based
/// synchronization for N-to-1 muxing.
///
/// # Input Pads
///
/// Pads are created automatically based on the configured tracks:
/// - Video tracks: `video_0`, `video_1`, etc.
/// - Audio tracks: `audio_0`, `audio_1`, etc.
/// - Data tracks: `data_0`, `data_1`, etc.
///
/// # Synchronization
///
/// The element uses [`MuxerSyncState`] to synchronize inputs:
/// - Video pads are required by default
/// - Audio/data pads are optional
/// - Output is produced when all required pads have data for the target PTS
///
/// # Example
///
/// ```rust,ignore
/// let config = TsMuxConfig::new()
///     .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
///     .add_track(TsMuxTrack::new(257, TsMuxStreamType::AacAdts).audio());
///
/// let mut mux = TsMuxElement::new(config)?;
///
/// // Push video frame
/// mux.push(MuxerInput::new(video_pad, video_buffer))?;
///
/// // Push audio frame
/// mux.push(MuxerInput::new(audio_pad, audio_buffer))?;
///
/// // Check if ready and pull
/// while mux.can_output() {
///     if let Some(output) = mux.pull()? {
///         // output.data contains TS packets
///     }
/// }
/// ```
pub struct TsMuxElement {
    /// Inner TS muxer.
    inner: TsMux,
    /// Synchronization state.
    sync: MuxerSyncState,
    /// Mapping from pad ID to track PID.
    pad_to_pid: HashMap<PadId, u16>,
    /// Mapping from track PID to pad ID.
    pid_to_pad: HashMap<u16, PadId>,
    /// Input pads with their caps.
    inputs: Vec<(PadId, Caps)>,
    /// Callback for pad added events.
    pad_added_callback: Option<PadAddedCallback>,
    /// Counters for pad naming.
    video_count: u32,
    audio_count: u32,
    data_count: u32,
}

impl TsMuxElement {
    /// Create a new TS mux element with the given configuration.
    pub fn new(config: TsMuxConfig) -> Result<Self> {
        Self::with_sync_config(config, MuxerSyncConfig::default())
    }

    /// Create a new TS mux element with custom sync configuration.
    pub fn with_sync_config(config: TsMuxConfig, sync_config: MuxerSyncConfig) -> Result<Self> {
        let mut sync = MuxerSyncState::new(sync_config);
        let mut pad_to_pid = HashMap::new();
        let mut pid_to_pad = HashMap::new();
        let mut inputs = Vec::new();

        let mut video_count = 0u32;
        let mut audio_count = 0u32;
        let mut data_count = 0u32;

        // Create pads from configured tracks
        for track in &config.tracks {
            let (name, stream_type, required) = if track.stream_type.is_video() {
                let name = format!("video_{}", video_count);
                video_count += 1;
                (name, StreamType::Video, true)
            } else if track.stream_type.is_audio() {
                let name = format!("audio_{}", audio_count);
                audio_count += 1;
                (name, StreamType::Audio, false)
            } else {
                let name = format!("data_{}", data_count);
                data_count += 1;
                (name, StreamType::Data, false)
            };

            let pad_info = PadInfo {
                name,
                stream_type,
                required,
            };

            let pad_id = sync.add_pad(pad_info);
            pad_to_pid.insert(pad_id, track.pid);
            pid_to_pad.insert(track.pid, pad_id);

            // Create caps based on stream type
            let caps = Self::caps_for_stream_type(&track.stream_type);
            inputs.push((pad_id, caps));
        }

        Ok(Self {
            inner: TsMux::new(config),
            sync,
            pad_to_pid,
            pid_to_pad,
            inputs,
            pad_added_callback: None,
            video_count,
            audio_count,
            data_count,
        })
    }

    /// Get caps for a stream type.
    fn caps_for_stream_type(stream_type: &TsMuxStreamType) -> Caps {
        match stream_type {
            TsMuxStreamType::H264 => Caps::new(MediaFormat::Video(VideoCodec::H264)),
            TsMuxStreamType::H265 => Caps::new(MediaFormat::Video(VideoCodec::H265)),
            TsMuxStreamType::Av1 => Caps::new(MediaFormat::Video(VideoCodec::Av1)),
            TsMuxStreamType::AacAdts | TsMuxStreamType::AacLatm => {
                Caps::new(MediaFormat::Audio(AudioCodec::Aac))
            }
            TsMuxStreamType::MpegAudio => Caps::new(MediaFormat::Audio(AudioCodec::Mp3)),
            _ => Caps::any(),
        }
    }

    /// Check if the muxer can produce output.
    pub fn can_output(&self) -> bool {
        self.sync.ready_to_output()
    }

    /// Push a buffer to the muxer.
    pub fn push(&mut self, input: MuxerInput) -> Result<()> {
        self.sync.push(input.pad, input.buffer)
    }

    /// Pull muxed output.
    ///
    /// Returns `None` if not ready to output.
    pub fn pull(&mut self) -> Result<Option<Buffer>> {
        if !self.can_output() {
            return Ok(None);
        }

        let inputs = self.sync.collect_for_output();
        if inputs.is_empty() {
            return Ok(None);
        }

        self.mux_inputs(inputs)
    }

    /// Mux collected inputs into TS output.
    fn mux_inputs(&mut self, inputs: Vec<CollectedInput>) -> Result<Option<Buffer>> {
        let mut output_data = Vec::new();

        // Write each input as PES
        for input in inputs {
            let pid = *self
                .pad_to_pid
                .get(&input.pad_id)
                .ok_or_else(|| Error::Element(format!("Unknown pad: {:?}", input.pad_id)))?;

            let pts = if input.buffer.metadata().pts.nanos() > 0 {
                Some(input.buffer.metadata().pts)
            } else {
                None
            };

            let dts = if input.buffer.metadata().dts.nanos() > 0 {
                Some(input.buffer.metadata().dts)
            } else {
                None
            };

            let data = input.buffer.as_bytes();
            let ts_packets = self.inner.write_pes(pid, data, pts, dts)?;
            output_data.extend(ts_packets);
        }

        // Advance sync state
        self.sync.advance();

        if output_data.is_empty() {
            return Ok(None);
        }

        // Create output buffer
        self.create_output_buffer(output_data)
    }

    /// Flush all remaining data.
    pub fn flush_all(&mut self) -> Result<Vec<Buffer>> {
        let mut outputs = Vec::new();

        // Flush sync state
        let remaining = self.sync.flush();
        if !remaining.is_empty() {
            if let Some(buffer) = self.mux_inputs(remaining)? {
                outputs.push(buffer);
            }
        }

        Ok(outputs)
    }

    /// Create an output buffer from TS data.
    fn create_output_buffer(&self, ts_data: Vec<u8>) -> Result<Option<Buffer>> {
        if ts_data.is_empty() {
            return Ok(None);
        }

        let segment = Arc::new(
            CpuSegment::new(ts_data.len())
                .map_err(|e| Error::Element(format!("Failed to allocate buffer: {}", e)))?,
        );

        let ptr = segment
            .as_mut_ptr()
            .ok_or_else(|| Error::Element("Failed to get segment pointer".into()))?;
        unsafe {
            std::ptr::copy_nonoverlapping(ts_data.as_ptr(), ptr, ts_data.len());
        }

        let handle = MemoryHandle::from_segment_with_len(segment, ts_data.len());
        let mut metadata = Metadata::new();
        metadata.pts = self.sync.target_pts();

        Ok(Some(Buffer::new(handle, metadata)))
    }

    /// Add a dynamic track.
    ///
    /// Returns the pad ID for the new track.
    pub fn add_track(&mut self, track: TsMuxTrack) -> Result<PadId> {
        let (name, stream_type, required) = if track.stream_type.is_video() {
            let name = format!("video_{}", self.video_count);
            self.video_count += 1;
            (name, StreamType::Video, true)
        } else if track.stream_type.is_audio() {
            let name = format!("audio_{}", self.audio_count);
            self.audio_count += 1;
            (name, StreamType::Audio, false)
        } else {
            let name = format!("data_{}", self.data_count);
            self.data_count += 1;
            (name, StreamType::Data, false)
        };

        let pad_info = PadInfo {
            name,
            stream_type,
            required,
        };

        let pad_id = self.sync.add_pad(pad_info);
        self.pad_to_pid.insert(pad_id, track.pid);
        self.pid_to_pad.insert(track.pid, pad_id);

        let caps = Self::caps_for_stream_type(&track.stream_type);
        self.inputs.push((pad_id, caps.clone()));

        // Notify callback
        if let Some(callback) = &mut self.pad_added_callback {
            callback(pad_id, caps);
        }

        Ok(pad_id)
    }

    /// Get the pad ID for a given PID.
    pub fn pad_for_pid(&self, pid: u16) -> Option<PadId> {
        self.pid_to_pad.get(&pid).copied()
    }

    /// Get the PID for a given pad ID.
    pub fn pid_for_pad(&self, pad_id: PadId) -> Option<u16> {
        self.pad_to_pid.get(&pad_id).copied()
    }

    /// Signal EOS on a pad.
    pub fn set_eos(&mut self, pad_id: PadId) {
        self.sync.set_eos(pad_id);
    }

    /// Check if all pads are at EOS.
    pub fn all_eos(&self) -> bool {
        self.sync.all_eos()
    }

    /// Get inner TsMux reference for advanced operations.
    pub fn inner(&self) -> &TsMux {
        &self.inner
    }

    /// Get inner TsMux mutable reference.
    pub fn inner_mut(&mut self) -> &mut TsMux {
        &mut self.inner
    }

    /// Reset the muxer state.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.sync.reset();
    }
}

impl Muxer for TsMuxElement {
    fn mux(&mut self, input: MuxerInput) -> Result<Option<Buffer>> {
        // Push the buffer
        self.push(input)?;

        // Try to pull output
        self.pull()
    }

    fn name(&self) -> &str {
        "TsMuxElement"
    }

    fn inputs(&self) -> &[(PadId, Caps)] {
        &self.inputs
    }

    fn output_caps(&self) -> Caps {
        Caps::new(MediaFormat::MpegTs)
    }

    fn on_pad_added(&mut self, callback: PadAddedCallback) {
        self.pad_added_callback = Some(callback);
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        // Return first flushed buffer; caller should call repeatedly
        let outputs = self.flush_all()?;
        Ok(outputs.into_iter().next())
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a simple video + KLV muxer element.
///
/// This creates a TS muxer with:
/// - Video track on PID 256 (H.264 or H.265 based on `h265` flag)
/// - KLV data track on PID 257
pub fn create_video_klv_muxer(h265: bool) -> Result<TsMuxElement> {
    let video_type = if h265 {
        TsMuxStreamType::H265
    } else {
        TsMuxStreamType::H264
    };

    let config = TsMuxConfig::new()
        .add_track(TsMuxTrack::new(256, video_type).video())
        .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());

    TsMuxElement::new(config)
}

/// Create a video + audio + KLV muxer element.
///
/// This creates a TS muxer with:
/// - Video track on PID 256
/// - Audio track on PID 257 (AAC)
/// - KLV data track on PID 258
pub fn create_av_klv_muxer(h265: bool) -> Result<TsMuxElement> {
    let video_type = if h265 {
        TsMuxStreamType::H265
    } else {
        TsMuxStreamType::H264
    };

    let config = TsMuxConfig::new()
        .add_track(TsMuxTrack::new(256, video_type).video())
        .add_track(TsMuxTrack::new(257, TsMuxStreamType::AacAdts).audio())
        .add_track(TsMuxTrack::new(258, TsMuxStreamType::Klv).private_data());

    TsMuxElement::new(config)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ClockTime;
    use crate::element::muxer::SyncMode;
    use crate::memory::CpuSegment;

    fn make_buffer(pts_ms: u64, data: &[u8]) -> Buffer {
        let segment = Arc::new(CpuSegment::new(data.len().max(64)).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        let handle = MemoryHandle::from_segment_with_len(segment, data.len());
        let mut metadata = Metadata::from_sequence(0);
        metadata.pts = ClockTime::from_millis(pts_ms);
        Buffer::new(handle, metadata)
    }

    #[test]
    fn test_ts_mux_element_creation() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mux = TsMuxElement::new(config).unwrap();

        assert_eq!(mux.inputs().len(), 1);
        assert_eq!(mux.name(), "TsMuxElement");
    }

    #[test]
    fn test_ts_mux_element_pads() {
        let config = TsMuxConfig::new()
            .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
            .add_track(TsMuxTrack::new(257, TsMuxStreamType::AacAdts).audio())
            .add_track(TsMuxTrack::new(258, TsMuxStreamType::Klv).private_data());

        let mux = TsMuxElement::new(config).unwrap();

        assert_eq!(mux.inputs().len(), 3);

        // Check pad-to-pid mapping
        let video_pad = mux.inputs()[0].0;
        assert_eq!(mux.pid_for_pad(video_pad), Some(256));

        let audio_pad = mux.inputs()[1].0;
        assert_eq!(mux.pid_for_pad(audio_pad), Some(257));

        let data_pad = mux.inputs()[2].0;
        assert_eq!(mux.pid_for_pad(data_pad), Some(258));
    }

    #[test]
    fn test_ts_mux_element_push_pull() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mut mux = TsMuxElement::new(config).unwrap();

        let video_pad = mux.inputs()[0].0;

        // Push a video frame (NAL unit)
        let nal_data = vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1f];
        let buffer = make_buffer(0, &nal_data);

        mux.push(MuxerInput::new(video_pad, buffer)).unwrap();

        // Should be ready (only one required pad)
        assert!(mux.can_output());

        // Pull output
        let output = mux.pull().unwrap();
        assert!(output.is_some());

        let output_buffer = output.unwrap();
        let ts_data = output_buffer.as_bytes();

        // Should have TS packets (PSI + PES)
        assert!(ts_data.len() >= 188 * 2);
        // Verify sync byte
        assert_eq!(ts_data[0], 0x47);
    }

    #[test]
    fn test_ts_mux_element_sync() {
        let config = TsMuxConfig::new()
            .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
            .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());

        let sync_config = MuxerSyncConfig::new()
            .with_mode(SyncMode::Strict)
            .with_interval_ms(40);

        let mut mux = TsMuxElement::with_sync_config(config, sync_config).unwrap();

        let video_pad = mux.inputs()[0].0;
        let data_pad = mux.inputs()[1].0;

        // Push video (required)
        let video_data = vec![0x00, 0x00, 0x00, 0x01, 0x67];
        mux.push(MuxerInput::new(video_pad, make_buffer(0, &video_data)))
            .unwrap();

        // Should be ready (data pad is optional)
        assert!(mux.can_output());

        // Push data
        let klv_data = vec![0x06, 0x0E, 0x2B];
        mux.push(MuxerInput::new(data_pad, make_buffer(0, &klv_data)))
            .unwrap();

        // Pull output (should include both)
        let output = mux.pull().unwrap();
        assert!(output.is_some());
    }

    #[test]
    fn test_ts_mux_element_eos() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mut mux = TsMuxElement::new(config).unwrap();

        let video_pad = mux.inputs()[0].0;

        assert!(!mux.all_eos());

        mux.set_eos(video_pad);

        assert!(mux.all_eos());
    }

    #[test]
    fn test_ts_mux_element_flush() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mut mux = TsMuxElement::new(config).unwrap();

        let video_pad = mux.inputs()[0].0;

        // Push multiple frames
        for i in 0..3 {
            let data = vec![0x00, 0x00, 0x00, 0x01, 0x67 + i as u8];
            mux.push(MuxerInput::new(video_pad, make_buffer(i * 40, &data)))
                .unwrap();
        }

        // Flush all
        let outputs = mux.flush_all().unwrap();

        // Should have produced output
        assert!(!outputs.is_empty());
    }

    #[test]
    fn test_ts_mux_element_reset() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mut mux = TsMuxElement::new(config).unwrap();

        let video_pad = mux.inputs()[0].0;

        // Push and pull
        let data = vec![0x00, 0x00, 0x00, 0x01, 0x67];
        mux.push(MuxerInput::new(video_pad, make_buffer(0, &data)))
            .unwrap();
        let _ = mux.pull().unwrap();

        // Reset
        mux.reset();

        // Stats should be reset
        assert_eq!(mux.inner().stats().packets_written, 0);
    }

    #[test]
    fn test_create_video_klv_muxer() {
        let mux = create_video_klv_muxer(false).unwrap();
        assert_eq!(mux.inputs().len(), 2);
    }

    #[test]
    fn test_create_av_klv_muxer() {
        let mux = create_av_klv_muxer(true).unwrap();
        assert_eq!(mux.inputs().len(), 3);
    }

    #[test]
    fn test_ts_mux_element_muxer_trait() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mut mux = TsMuxElement::new(config).unwrap();

        let video_pad = mux.inputs()[0].0;

        // Use Muxer trait method
        let data = vec![0x00, 0x00, 0x00, 0x01, 0x67];
        let result = mux.mux(MuxerInput::new(video_pad, make_buffer(0, &data)));

        assert!(result.is_ok());
        // May or may not have output depending on sync state
    }
}
