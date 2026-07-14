//! Codec traits for video encoders and decoders.
//!
//! This module defines standard traits that codecs must implement to work
//! with the pipeline element wrappers.
//!
//! # Overview
//!
//! - [`VideoEncoder`] - Trait for video encoders (produce packets from frames)
//! - [`VideoDecoder`] - Trait for video decoders (produce frames from packets)
//!
//! # Example: Implementing VideoEncoder
//!
//! ```rust,ignore
//! impl VideoEncoder for MyEncoder {
//!     type Packet = Vec<u8>;
//!
//!     fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Self::Packet>> {
//!         // Encode frame, may return 0 or more packets
//!     }
//!
//!     fn flush(&mut self) -> Result<Vec<Self::Packet>> {
//!         // Drain any buffered frames at EOS
//!     }
//! }
//! ```

use super::common::VideoFrame;
use crate::error::{Error, Result};

/// Frame type hint for encoded frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameType {
    /// Unknown frame type.
    #[default]
    Unknown,
    /// Keyframe (I-frame) - can be decoded independently.
    Key,
    /// Inter-frame (P-frame) - predicted from previous frames.
    Inter,
    /// Bidirectional (B-frame) - predicted from both previous and future frames.
    BiPred,
}

/// Trait for video encoders.
///
/// Video encoders take raw video frames and produce encoded packets.
/// Due to buffering (lookahead, B-frames), there may not be a 1:1
/// correspondence between input frames and output packets.
///
/// # Buffering Behavior
///
/// - `encode()` may return 0 packets (encoder is buffering)
/// - `encode()` may return 1 packet (typical case)
/// - `encode()` may return multiple packets (flushing lookahead)
/// - `flush()` must be called at EOS to drain all remaining packets
///
/// # Example
///
/// ```rust,ignore
/// let mut encoder = MyEncoder::new(config)?;
///
/// // Encode frames
/// for frame in frames {
///     for packet in encoder.encode(&frame)? {
///         // Process encoded packet
///     }
/// }
///
/// // Flush remaining frames at EOS
/// for packet in encoder.flush()? {
///     // Process remaining packets
/// }
/// ```
pub trait VideoEncoder: Send {
    /// Encoded packet type (usually `Vec<u8>`).
    type Packet: AsRef<[u8]> + Send;

    /// Encode a video frame.
    ///
    /// Returns zero or more encoded packets. The encoder may buffer
    /// frames internally (for B-frame reordering, lookahead, etc.),
    /// so not every input frame produces immediate output.
    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Self::Packet>>;

    /// Flush any buffered frames at end-of-stream.
    ///
    /// Must be called after all frames have been sent to drain the
    /// encoder's internal buffers.
    fn flush(&mut self) -> Result<Vec<Self::Packet>>;

    /// Get codec-specific header data (optional).
    ///
    /// For H.264: SPS/PPS NAL units
    /// For AV1: Sequence header OBU
    /// For VP9: Nothing (headers in-band)
    fn codec_data(&self) -> Option<Vec<u8>> {
        None
    }

    /// Check if encoder has buffered frames.
    fn has_pending(&self) -> bool {
        false
    }

    /// Force the next encoded frame to be a keyframe (IDR).
    ///
    /// Default: no-op, for encoders without keyframe control. Encoders that
    /// support it (H.264, hardware encoders) override this; the
    /// [`EncoderElement`](super::EncoderElement) wrapper calls it when a
    /// [`KeyframeHandle`](super::KeyframeHandle) request or an in-band
    /// [`KEYFRAME_REQUEST`](super::KEYFRAME_REQUEST) metadata flag arrives.
    fn force_keyframe(&mut self) {}

    /// Set the target bitrate in bits per second, on a possibly-running encoder.
    ///
    /// `0` means "no bitrate target": encoders that support it fall back to
    /// quality-driven rate control.
    ///
    /// Default: `Err` — the encoder cannot change its bitrate. This is
    /// deliberately not a silent no-op: a caller lowering the bitrate to fit a
    /// congested link needs to know when nothing happened.
    fn set_bitrate(&mut self, bps: u32) -> Result<()> {
        let _ = bps;
        Err(unsupported("bitrate"))
    }

    /// Set the keyframe (IDR) interval in frames. `0` lets the encoder decide.
    ///
    /// Default: `Err` — see [`set_bitrate`](Self::set_bitrate).
    fn set_keyframe_interval(&mut self, frames: u32) -> Result<()> {
        let _ = frames;
        Err(unsupported("keyframe interval"))
    }

    /// Set the target quantization parameter (0-51, lower = better quality).
    ///
    /// Default: `Err` — see [`set_bitrate`](Self::set_bitrate).
    fn set_qp(&mut self, qp: u8) -> Result<()> {
        let _ = qp;
        Err(unsupported("QP"))
    }
}

/// The error returned by the default [`VideoEncoder`] parameter setters.
fn unsupported(parameter: &str) -> Error {
    Error::Config(format!(
        "this encoder does not support changing its {parameter} at runtime"
    ))
}

/// Trait for video decoders.
///
/// Video decoders take encoded packets and produce raw video frames.
/// Similar to encoders, there may not be a 1:1 correspondence due
/// to frame reordering and buffering.
///
/// # Example
///
/// ```rust,ignore
/// let mut decoder = MyDecoder::new()?;
///
/// for packet in packets {
///     for frame in decoder.decode(&packet)? {
///         // Process decoded frame
///     }
/// }
///
/// // Flush at EOS
/// for frame in decoder.flush()? {
///     // Process remaining frames
/// }
/// ```
pub trait VideoDecoder: Send {
    /// Decode an encoded packet.
    ///
    /// Returns zero or more decoded frames. The decoder may buffer
    /// packets internally for B-frame reordering.
    fn decode(&mut self, packet: &[u8]) -> Result<Vec<VideoFrame>>;

    /// Flush any buffered frames at end-of-stream.
    fn flush(&mut self) -> Result<Vec<VideoFrame>>;

    /// Check if decoder has buffered frames.
    fn has_pending(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_type_default() {
        assert_eq!(FrameType::default(), FrameType::Unknown);
    }
}
