//! RTP/RTCP/RTSP protocol elements.
//!
//! All elements in this module require the `rtp` or `rtsp` feature.
//!
//! ## RTP (requires `rtp` feature)
//! - `RtpSrc`, `RtpSink`: RTP packet send/receive over UDP
//! - `AsyncRtpSrc`, `AsyncRtpSink`: Async versions
//!
//! ## RTCP (requires `rtp` feature)
//! - `RtcpHandler`: RTCP sender/receiver report handling
//!
//! ## RTP Codec Payloaders/Depayloaders (requires `rtp` feature)
//! - `RtpH264Depay`, `RtpH264Pay`: H.264/AVC
//! - `RtpH265Depay`, `RtpH265Pay`: H.265/HEVC
//! - `RtpVp8Depay`, `RtpVp8Pay`: VP8
//! - `RtpVp9Depay`, `RtpVp9Pay`: VP9
//! - `RtpOpusDepay`: Opus audio
//!
//! ## Jitter Buffer (requires `rtp` feature)
//! - `RtpJitterBuffer`: Packet reordering and loss detection
//! - `AsyncJitterBuffer`: Async version with timeout-based retrieval
//!
//! ## RTSP (requires `rtsp` feature)
//! - `RtspSrc`: RTSP client source
//! - `RtspSession`: Active RTSP session

#[cfg(feature = "rtp")]
mod rtp;

#[cfg(feature = "rtp")]
mod rtcp;

#[cfg(feature = "rtp")]
mod rtp_codecs;

#[cfg(feature = "rtp")]
mod jitter_buffer;

#[cfg(feature = "rtsp")]
mod rtsp;

// RTP
#[cfg(feature = "rtp")]
pub use rtp::{AsyncRtpSink, AsyncRtpSrc, RtpSink, RtpSinkStats, RtpSrc, RtpSrcStats};

// RTCP
#[cfg(feature = "rtp")]
pub use rtcp::{
    ReceiverReportInfo, ReceptionStats, RtcpHandler, RtcpPacketInfo, RtcpStats, SenderStats,
};

// RTP Codec Payloaders/Depayloaders
#[cfg(feature = "rtp")]
pub use rtp_codecs::{
    DepayStats, PayStats, RtpH264Depay, RtpH264Pay, RtpH265Depay, RtpH265Pay, RtpOpusDepay,
    RtpVp8Depay, RtpVp8Pay, RtpVp9Depay, RtpVp9Pay,
};

// Jitter Buffer
#[cfg(feature = "rtp")]
pub use jitter_buffer::{
    AsyncJitterBuffer, JitterBufferConfig, JitterBufferStats, LossInfo, RtpJitterBuffer,
};

// RTSP
#[cfg(feature = "rtsp")]
pub use rtsp::{
    MediaType, RtspConfig, RtspCredentials, RtspFrame, RtspSession, RtspSrc, RtspStats,
    RtspTransport, StreamInfo, StreamSelection,
};
