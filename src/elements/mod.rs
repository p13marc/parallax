//! Built-in pipeline elements.
//!
//! Elements are organized into categories:
//!
//! ## [`network`] - Network Transport
//! TCP, UDP, Unix sockets, multicast, HTTP, WebSocket, Zenoh
//!
//! ## [`rtp`] - RTP/RTCP/RTSP Protocol
//! RTP packet handling, RTCP reports, codec payloaders/depayloaders, jitter buffer, RTSP client
//!
//! ## [`io`] - File and Descriptor I/O
//! File sources/sinks, raw file descriptors, console output
//!
//! ## [`testing`] - Test Sources and Sinks
//! Test pattern generators, video test sources, null elements
//!
//! ## [`flow`] - Flow Control and Routing
//! Queue, tee, funnel, selectors, concat, valve
//!
//! ## [`transform`] - Data Transformation
//! Map, filter, batch, chunk, buffer operations, metadata operations
//!
//! ## [`app`] - Application Integration
//! AppSrc, AppSink, Iced video sink
//!
//! ## [`ipc`] - Inter-Process Communication
//! IPC sources/sinks, shared memory elements
//!
//! ## [`timing`] - Timing and Rate Control
//! Delay, timeout, debounce, throttle, rate limiter
//!
//! ## [`demux`] - Demultiplexing
//! Stream ID demux, MPEG-TS demux
//!
//! ## [`util`] - Utility Elements
//! PassThrough, Identity

pub mod app;
pub mod demux;
pub mod flow;
pub mod io;
pub mod ipc;
pub mod network;
pub mod rtp;
pub mod testing;
pub mod timing;
pub mod transform;
pub mod util;

// Re-export all public types for convenience (backwards compatibility)

// Network
pub use network::{AsyncTcpSink, AsyncTcpSrc, TcpMode, TcpSink, TcpSrc};
pub use network::{AsyncUdpSink, AsyncUdpSrc, UdpSink, UdpSrc};
pub use network::{AsyncUnixSink, AsyncUnixSrc, UnixMode, UnixSink, UnixSrc};
pub use network::{UdpMulticastSink, UdpMulticastSrc, UdpMulticastStats};

#[cfg(feature = "http")]
pub use network::{HttpMethod, HttpSink, HttpSinkStats, HttpSrc, HttpStreamingSink};

#[cfg(feature = "websocket")]
pub use network::{WebSocketSink, WebSocketSrc, WebSocketStats, WsMessageType};

#[cfg(feature = "zenoh")]
pub use network::{
    ZenohCongestionControl, ZenohPriority, ZenohQuerier, ZenohQuery, ZenohQueryable, ZenohSink,
    ZenohSrc, ZenohStats,
};

// RTP/RTCP/RTSP
#[cfg(feature = "rtp")]
pub use rtp::{AsyncRtpSink, AsyncRtpSrc, RtpSink, RtpSinkStats, RtpSrc, RtpSrcStats};

#[cfg(feature = "rtp")]
pub use rtp::{
    ReceiverReportInfo, ReceptionStats, RtcpHandler, RtcpPacketInfo, RtcpStats, SenderStats,
};

#[cfg(feature = "rtp")]
pub use rtp::{
    DepayStats, PayStats, RtpH264Depay, RtpH264Pay, RtpH265Depay, RtpH265Pay, RtpOpusDepay,
    RtpVp8Depay, RtpVp8Pay, RtpVp9Depay, RtpVp9Pay,
};

#[cfg(feature = "rtp")]
pub use rtp::{
    AsyncJitterBuffer, JitterBufferConfig, JitterBufferStats, LossInfo, RtpJitterBuffer,
};

#[cfg(feature = "rtsp")]
pub use rtp::{
    MediaType, RtspConfig, RtspCredentials, RtspFrame, RtspSession, RtspSrc, RtspStats,
    RtspTransport, StreamInfo, StreamSelection,
};

// I/O
pub use io::{ConsoleFormat, ConsoleSink, FdSink, FdSrc, FileSink, FileSrc};

// Testing
pub use testing::{
    AsyncVideoTestSrc, DataSrc, NullSink, NullSource, PixelFormat, TestPattern, TestSrc,
    VideoPattern, VideoTestSrc,
};

// Flow control
pub use flow::{
    Concat, ConcatStats, ConcatStream, Funnel, FunnelInput, FunnelStats, InputSelector,
    InputSelectorStats, LeakyMode, OutputSelector, OutputSelectorStats, Queue, QueueStats,
    SelectorInput, SelectorOutput, Tee, Valve, ValveControl, ValveStats,
};

// Transform
pub use transform::{
    Batch, BatchStats, BufferConcat, BufferConcatStats, BufferJoin, BufferJoinStats, BufferPad,
    BufferPadStats, BufferSlice, BufferSplit, BufferSplitStats, BufferTrim, BufferTrimStats, Chunk,
    DuplicateFilter, DuplicateFilterStats, ExtractedMetadata, Filter, FilterMap, FilterStats,
    FlatMap, Map, MetadataExtract, MetadataFilter, MetadataInject, RangeFilter, RangeFilterStats,
    RegexFilter, RegexFilterStats, SampleFilter, SampleMode, SequenceNumber, TimestampMode,
    Timestamper, Unbatch, UnbatchStats,
};

// App integration
pub use app::{AppSink, AppSinkHandle, AppSinkStats, AppSrc, AppSrcHandle, AppSrcStats};

#[cfg(feature = "iced-sink")]
pub use app::{
    IcedVideoSink, IcedVideoSinkConfig, IcedVideoSinkHandle, IcedVideoSinkStats, InputPixelFormat,
};

// IPC
pub use ipc::{IpcSink, IpcSrc, MemorySink, MemorySinkStats, MemorySrc, SharedMemorySink};

// Timing
pub use timing::{
    AsyncDelay, Debounce, DebounceStats, Delay, DelayStats, RateLimitMode, RateLimiter, Throttle,
    ThrottleStats, Timeout, TimeoutStats,
};

// Demux
pub use demux::{StreamIdDemux, StreamIdDemuxStats, StreamOutput};

#[cfg(feature = "mpeg-ts")]
pub use demux::{
    TS_PACKET_SIZE, TsDemux, TsDemuxStats, TsElementaryStream, TsFrame, TsProgram, TsStreamType,
};

// Utility
pub use util::{Identity, IdentityStats, PassThrough};
