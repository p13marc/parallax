//! Demultiplexing elements.
//!
//! - [`StreamIdDemux`]: Demultiplex by stream ID
//! - [`TsDemux`]: MPEG Transport Stream demultiplexer (requires `mpeg-ts` feature)
//! - [`Mp4Demux`]: MP4/MOV container demultiplexer (requires `mp4-demux` feature)

mod streamid_demux;

#[cfg(feature = "mpeg-ts")]
mod mpegts;

#[cfg(feature = "mp4-demux")]
mod mp4;

pub use streamid_demux::{StreamIdDemux, StreamIdDemuxStats, StreamOutput};

#[cfg(feature = "mpeg-ts")]
pub use mpegts::{
    TS_PACKET_SIZE, TsDemux, TsDemuxStats, TsElementaryStream, TsFrame, TsProgram, TsStreamType,
};

#[cfg(feature = "mp4-demux")]
pub use mp4::{
    Mp4AudioInfo, Mp4Codec, Mp4Demux, Mp4DemuxStats, Mp4Sample, Mp4Track, Mp4TrackType,
    Mp4VideoInfo,
};
