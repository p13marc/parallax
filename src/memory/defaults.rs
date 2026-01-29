//! Default arena sizes and configuration constants.
//!
//! This module provides recommended arena sizes for different use cases.
//! These values are based on empirical testing and typical media pipeline
//! requirements.
//!
//! # Design Rationale
//!
//! Arena sizes are chosen to balance memory usage with throughput:
//! - Too few slots → arena exhaustion, pipeline stalls
//! - Too many slots → wasted memory
//!
//! Slot sizes are based on typical data sizes:
//! - Video frames: 1080p RGBA = 8.3MB, 4K RGBA = 33MB
//! - Audio buffers: 48kHz stereo float32, 1024 samples = 8KB
//! - Network packets: typically 1500 bytes (MTU) to 64KB (UDP max)
//!
//! # Usage
//!
//! ```rust,ignore
//! use parallax::memory::{SharedArena, defaults};
//!
//! // Use recommended defaults for video capture
//! let arena = SharedArena::new(
//!     defaults::VIDEO_1080P_SLOT_SIZE,
//!     defaults::VIDEO_CAPTURE_SLOT_COUNT,
//! )?;
//! ```

// =============================================================================
// Slot Sizes (bytes)
// =============================================================================

/// Slot size for 1080p BGRA video frames (1920 * 1080 * 4 = 8,294,400 bytes).
pub const VIDEO_1080P_SLOT_SIZE: usize = 1920 * 1080 * 4;

/// Slot size for 1080p YUV420 video frames (1920 * 1080 * 3/2 = 3,110,400 bytes).
pub const VIDEO_1080P_YUV420_SLOT_SIZE: usize = 1920 * 1080 * 3 / 2;

/// Slot size for 720p BGRA video frames (1280 * 720 * 4 = 3,686,400 bytes).
pub const VIDEO_720P_SLOT_SIZE: usize = 1280 * 720 * 4;

/// Slot size for 4K BGRA video frames (3840 * 2160 * 4 = 33,177,600 bytes).
pub const VIDEO_4K_SLOT_SIZE: usize = 3840 * 2160 * 4;

/// Slot size for encoded video packets (H.264/H.265/AV1).
/// Compressed frames are typically much smaller than raw frames.
/// 256KB is generous for most encoded frames.
pub const VIDEO_ENCODED_SLOT_SIZE: usize = 256 * 1024;

/// Slot size for audio buffers (48kHz stereo float32, 1024 samples).
pub const AUDIO_SLOT_SIZE: usize = 1024 * 2 * 4;

/// Slot size for large audio buffers (48kHz stereo float32, 4096 samples).
pub const AUDIO_LARGE_SLOT_SIZE: usize = 4096 * 2 * 4;

/// Slot size for network/RTP packets.
pub const NETWORK_SLOT_SIZE: usize = 64 * 1024;

/// Slot size for small metadata buffers (KLV, SEI, etc).
pub const METADATA_SLOT_SIZE: usize = 64 * 1024;

/// Slot size for MPEG-TS muxer output (batched packets).
pub const TS_MUX_SLOT_SIZE: usize = 1024 * 1024;

/// Slot size for MPEG-TS demuxer output (PES packets can be large).
pub const TS_DEMUX_SLOT_SIZE: usize = 2 * 1024 * 1024;

/// Slot size for MP4 demuxer output (video frames can be large).
pub const MP4_DEMUX_SLOT_SIZE: usize = 4 * 1024 * 1024;

// =============================================================================
// Slot Counts
// =============================================================================

/// Default slot count for video capture sources.
/// 200 slots provides ~6 seconds buffer at 30fps, allowing downstream
/// elements (especially encoders) time to process.
pub const VIDEO_CAPTURE_SLOT_COUNT: usize = 200;

/// Slot count for video encoders.
/// 64 slots allows buffering multiple encoded frames.
pub const VIDEO_ENCODER_SLOT_COUNT: usize = 64;

/// Slot count for video decoders.
/// 64 slots for decoded frame output.
pub const VIDEO_DECODER_SLOT_COUNT: usize = 64;

/// Slot count for audio processing.
/// 64 slots provides good buffer for audio pipelines.
pub const AUDIO_SLOT_COUNT: usize = 64;

/// Slot count for network sources/sinks.
/// 32 slots is typically sufficient for network I/O.
pub const NETWORK_SLOT_COUNT: usize = 32;

/// Slot count for metadata processing.
/// 32 slots for KLV, SEI, and other metadata.
pub const METADATA_SLOT_COUNT: usize = 32;

/// Slot count for MPEG-TS muxer.
/// 32 slots for muxed output.
pub const TS_MUX_SLOT_COUNT: usize = 32;

/// Slot count for MPEG-TS demuxer.
/// 64 slots for demuxed PES packets.
pub const TS_DEMUX_SLOT_COUNT: usize = 64;

/// Slot count for MP4 demuxer.
/// 32 slots for demuxed samples.
pub const MP4_DEMUX_SLOT_COUNT: usize = 32;

/// Slot count for test arenas (small, for unit tests).
pub const TEST_SLOT_COUNT: usize = 64;

// =============================================================================
// Convenience Functions
// =============================================================================

/// Calculate slot size for raw video frames.
///
/// # Arguments
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `bytes_per_pixel` - Bytes per pixel (4 for RGBA/BGRA, 3 for RGB, 1.5 for YUV420)
///
/// # Example
/// ```rust,ignore
/// let slot_size = video_frame_size(1920, 1080, 4); // 1080p BGRA
/// ```
pub const fn video_frame_size(width: usize, height: usize, bytes_per_pixel: usize) -> usize {
    width * height * bytes_per_pixel
}

/// Calculate recommended slot count for a given duration at a framerate.
///
/// # Arguments
/// * `duration_secs` - Buffer duration in seconds
/// * `fps` - Frames per second
///
/// # Example
/// ```rust,ignore
/// let slots = slots_for_duration(5.0, 30.0); // 5 seconds at 30fps = 150 slots
/// ```
pub fn slots_for_duration(duration_secs: f64, fps: f64) -> usize {
    (duration_secs * fps).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_1080p_size() {
        assert_eq!(VIDEO_1080P_SLOT_SIZE, 8_294_400);
    }

    #[test]
    fn test_video_4k_size() {
        assert_eq!(VIDEO_4K_SLOT_SIZE, 33_177_600);
    }

    #[test]
    fn test_video_frame_size() {
        assert_eq!(video_frame_size(1920, 1080, 4), VIDEO_1080P_SLOT_SIZE);
        assert_eq!(video_frame_size(3840, 2160, 4), VIDEO_4K_SLOT_SIZE);
    }

    #[test]
    fn test_slots_for_duration() {
        assert_eq!(slots_for_duration(5.0, 30.0), 150);
        assert_eq!(slots_for_duration(1.0, 60.0), 60);
        assert_eq!(slots_for_duration(0.5, 30.0), 15);
    }
}
