//! Demultiplexing elements.
//!
//! - [`StreamIdDemux`]: Demultiplex by stream ID
//! - [`TsDemux`]: MPEG Transport Stream demultiplexer (requires `mpeg-ts` feature)

mod streamid_demux;

#[cfg(feature = "mpeg-ts")]
mod mpegts;

pub use streamid_demux::{StreamIdDemux, StreamIdDemuxStats, StreamOutput};

#[cfg(feature = "mpeg-ts")]
pub use mpegts::{
    TS_PACKET_SIZE, TsDemux, TsDemuxStats, TsElementaryStream, TsFrame, TsProgram, TsStreamType,
};
