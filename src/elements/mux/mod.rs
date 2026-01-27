//! Multiplexing elements.
//!
//! - [`Mp4Mux`]: MP4/MOV container muxer (requires `mp4-demux` feature)

#[cfg(feature = "mp4-demux")]
mod mp4;

#[cfg(feature = "mp4-demux")]
pub use mp4::{
    AudioCodecConfig, Mp4AudioTrackConfig, Mp4Mux, Mp4MuxConfig, Mp4MuxStats, Mp4VideoTrackConfig,
    VideoCodecConfig,
};
