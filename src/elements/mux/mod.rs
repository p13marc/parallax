//! Multiplexing elements.
//!
//! - [`Mp4Mux`]: MP4/MOV container muxer (requires `mp4-demux` feature)
//! - [`TsMux`]: MPEG Transport Stream muxer (requires `mpeg-ts` feature)
//! - [`TsMuxElement`]: Pipeline-ready TS muxer element (requires `mpeg-ts` feature)
//!
//! # Muxer Elements
//!
//! Muxer elements implement the [`Muxer`](crate::element::Muxer) trait and can be
//! used directly in pipelines. They handle N-to-1 stream synchronization automatically.
//!
//! ```rust,ignore
//! use parallax::elements::mux::{TsMuxElement, TsMuxConfig, TsMuxTrack, TsMuxStreamType};
//! use parallax::element::{MuxerAdapter, DynAsyncElement};
//!
//! let config = TsMuxConfig::new()
//!     .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
//!     .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());
//!
//! let mux_element = TsMuxElement::new(config)?;
//!
//! // Use in pipeline
//! let mux = pipeline.add_node("mux", DynAsyncElement::new_box(MuxerAdapter::new(mux_element)));
//! ```

#[cfg(feature = "mp4-demux")]
mod mp4;

#[cfg(feature = "mpeg-ts")]
mod mpegts;

#[cfg(feature = "mpeg-ts")]
mod ts_element;

#[cfg(feature = "mp4-demux")]
pub use mp4::{
    AudioCodecConfig, Mp4AudioTrackConfig, Mp4Mux, Mp4MuxConfig, Mp4MuxStats, Mp4VideoTrackConfig,
    VideoCodecConfig,
};

#[cfg(feature = "mpeg-ts")]
pub use mpegts::{TsMux, TsMuxConfig, TsMuxStats, TsMuxStreamType, TsMuxTrack};

#[cfg(feature = "mpeg-ts")]
pub use ts_element::{TsMuxElement, create_av_klv_muxer, create_video_klv_muxer};
