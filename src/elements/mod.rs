//! Built-in pipeline elements.
//!
//! This module provides common elements that are useful in most pipelines:
//!
//! ## Sources
//! - [`FileSrc`]: Reads buffers from a file
//! - [`TcpSrc`]: Reads buffers from a TCP connection
//! - [`UdpSrc`]: Reads datagrams from a UDP socket
//! - [`FdSrc`]: Reads from a raw file descriptor
//! - [`AppSrc`]: Injects buffers from application code
//! - [`DataSrc`]: Generates buffers from inline data
//! - [`TestSrc`]: Generates test pattern buffers
//! - [`VideoTestSrc`]: Generates video test pattern frames (SMPTE bars, etc.)
//! - [`AsyncVideoTestSrc`]: Async video test source with tokio timer for precise framerate
//! - [`MemorySrc`]: Reads from memory buffer/slice
//! - [`NullSource`]: Produces empty buffers (useful for testing)
//!
//! ## Sinks
//! - [`FileSink`]: Writes buffers to a file
//! - [`TcpSink`]: Writes buffers to a TCP connection
//! - [`UdpSink`]: Sends datagrams to a UDP socket
//! - [`FdSink`]: Writes to a raw file descriptor
//! - [`AppSink`]: Extracts buffers to application code
//! - [`ConsoleSink`]: Prints buffers to console for debugging
//! - [`MemorySink`]: Writes to memory buffer
//! - [`NullSink`]: Discards all buffers (useful for benchmarking)
//!
//! ## Transforms
//! - [`PassThrough`]: Passes buffers unchanged (useful for debugging/testing)
//! - [`Identity`]: Pass-through with callbacks for debugging
//! - [`RateLimiter`]: Limits buffer throughput rate
//! - [`Valve`]: Drops or passes buffers (on/off switch)
//! - [`Queue`]: Asynchronous buffer queue with backpressure
//! - [`Delay`]: Adds fixed delay to buffer flow
//! - [`Map`]: Transforms buffer contents
//! - [`FilterMap`]: Transforms and optionally filters
//! - [`Chunk`]: Splits buffers into fixed-size chunks
//! - [`Batch`]: Combines multiple buffers into one
//! - [`Unbatch`]: Splits one buffer into many
//!
//! ## Filtering
//! - [`Filter`]: Generic predicate-based filter
//! - [`SampleFilter`]: Statistical sampling (every Nth, random %)
//! - [`MetadataFilter`]: Filter by metadata values
//!
//! ## Metadata
//! - [`SequenceNumber`]: Adds sequence numbers to buffers
//! - [`Timestamper`]: Adds timestamps to buffers
//! - [`MetadataInject`]: Injects custom metadata
//!
//! ## Buffer Operations
//! - [`BufferTrim`]: Trims buffers to max size
//! - [`BufferSlice`]: Extracts slice from buffer
//! - [`BufferPad`]: Pads buffers to min size
//!
//! ## Timing
//! - [`Timeout`]: Produces fallback on timeout
//! - [`Debounce`]: Suppresses rapid buffer sequences
//! - [`Throttle`]: Drops buffers if too rapid
//!
//! ## Routing
//! - [`Tee`]: Duplicates buffers to multiple outputs (1-to-N fanout)
//! - [`Funnel`]: Merges multiple inputs into one output (N-to-1)
//! - [`InputSelector`]: Selects one of N inputs (N-to-1 switching)
//! - [`OutputSelector`]: Routes to one of N outputs (1-to-N routing)
//! - [`Concat`]: Concatenates streams sequentially
//! - [`StreamIdDemux`]: Demultiplexes by stream ID
//!
//! ## Network (Tier 3)
//! - [`UnixSrc`], [`UnixSink`]: Unix domain socket I/O
//! - [`UdpMulticastSrc`], [`UdpMulticastSink`]: UDP multicast
//! - [`HttpSrc`], [`HttpSink`]: HTTP GET/POST (requires `http` feature)
//! - [`WebSocketSrc`], [`WebSocketSink`]: WebSocket (requires `websocket` feature)
//!
//! ## Zenoh (Tier 4, requires `zenoh` feature)
//! - [`ZenohSrc`]: Subscribe to Zenoh key expression
//! - [`ZenohSink`]: Publish to Zenoh key expression
//! - [`ZenohQueryable`]: Handle Zenoh queries
//! - [`ZenohQuerier`]: Send Zenoh queries
//!
//! ## RTP/RTCP (requires `rtp` feature)
//! - [`RtpSrc`]: Receive and parse RTP packets from UDP
//! - [`RtpSink`]: Send RTP packets over UDP
//! - [`AsyncRtpSrc`]: Async version of RtpSrc
//! - [`AsyncRtpSink`]: Async version of RtpSink
//! - [`RtcpHandler`]: RTCP sender/receiver report handling
//!
//! ## RTP Codec Payloaders/Depayloaders (requires `rtp` feature)
//! - [`RtpH264Depay`], [`RtpH264Pay`]: H.264/AVC
//! - [`RtpH265Depay`], [`RtpH265Pay`]: H.265/HEVC
//! - [`RtpVp8Depay`], [`RtpVp8Pay`]: VP8
//! - [`RtpVp9Depay`], [`RtpVp9Pay`]: VP9
//! - [`RtpOpusDepay`]: Opus audio
//!
//! ## Jitter Buffer (requires `rtp` feature)
//! - [`RtpJitterBuffer`]: Packet reordering and loss detection
//! - [`AsyncJitterBuffer`]: Async version with timeout-based retrieval
//!
//! ## RTSP Client (requires `rtsp` feature)
//! - [`RtspSrc`]: RTSP client source (connects to cameras/servers)
//! - [`RtspSession`]: Active RTSP session for receiving frames
//!
//! ## MPEG-TS (requires `mpeg-ts` feature)
//! - [`TsDemux`]: MPEG Transport Stream demultiplexer
//! - [`TsFrame`]: Extracted elementary stream frame
//! - [`TsStreamType`]: Stream type classification (H.264, AAC, etc.)
//!
//! ## Iced Video Sink (requires `iced-sink` feature)
//! - [`IcedVideoSink`]: Display video frames in an Iced GUI window
//! - [`IcedVideoSinkHandle`]: Handle to run and control the video window
//!
//! ## Data Processing (Tier 5)
//! - [`FlatMap`]: One-to-many buffer transformation
//! - [`DuplicateFilter`]: Remove duplicate buffers by content hash
//! - [`RangeFilter`]: Filter by size/sequence range
//! - [`RegexFilter`]: Filter by regex pattern match
//! - [`MetadataExtract`]: Extract metadata to sideband channel
//! - [`BufferSplit`]: Split buffer at delimiter boundaries
//! - [`BufferJoin`]: Join buffers with delimiter
//! - [`BufferConcat`]: Concatenate buffer contents

mod appsink;
mod appsrc;
mod batch;
mod buffer_ops;
mod concat;
mod console;
mod datasrc;
mod delay;
mod fd;
mod file;
mod filter;
mod funnel;
mod identity;
mod ipc;
mod memory;
mod metadata_ops;
mod null;
mod passthrough;
mod queue;
mod rate_limiter;
mod selector;
mod streamid_demux;
mod tcp;
mod tee;
mod testsrc;
mod timeout;
mod transform;
mod udp;
mod unix;
mod valve;
mod videotestsrc;

// Tier 3: Network elements
mod multicast;

// Tier 5: Data processing
mod data_processing;

// Feature-gated modules
#[cfg(feature = "http")]
mod http;

#[cfg(feature = "websocket")]
mod websocket;

#[cfg(feature = "zenoh")]
mod zenoh;

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

#[cfg(feature = "mpeg-ts")]
mod mpegts;

#[cfg(feature = "iced-sink")]
mod iced_sink;

// Sources
pub use appsrc::{AppSrc, AppSrcHandle, AppSrcStats};
pub use datasrc::DataSrc;
pub use fd::FdSrc;
pub use file::FileSrc;
pub use memory::{MemorySink, MemorySinkStats, MemorySrc, SharedMemorySink};
pub use null::NullSource;
pub use tcp::{AsyncTcpSrc, TcpMode, TcpSrc};
pub use testsrc::{TestPattern, TestSrc};
pub use udp::{AsyncUdpSrc, UdpSrc};
pub use videotestsrc::{AsyncVideoTestSrc, PixelFormat, VideoPattern, VideoTestSrc};

// Sinks
pub use appsink::{AppSink, AppSinkHandle, AppSinkStats};
pub use console::{ConsoleFormat, ConsoleSink};
pub use fd::FdSink;
pub use file::FileSink;
pub use null::NullSink;
pub use tcp::{AsyncTcpSink, TcpSink};
pub use udp::{AsyncUdpSink, UdpSink};

// Transforms
pub use batch::{Batch, BatchStats, Unbatch, UnbatchStats};
pub use delay::{AsyncDelay, Delay, DelayStats};
pub use identity::{Identity, IdentityStats};
pub use passthrough::PassThrough;
pub use queue::{LeakyMode, Queue, QueueStats};
pub use rate_limiter::{RateLimitMode, RateLimiter};
pub use transform::{Chunk, FilterMap, FlatMap, Map};
pub use valve::{Valve, ValveControl, ValveStats};

// Filtering
pub use filter::{Filter, FilterStats, MetadataFilter, SampleFilter, SampleMode};

// Metadata operations
pub use metadata_ops::{MetadataInject, SequenceNumber, TimestampMode, Timestamper};

// Buffer operations
pub use buffer_ops::{BufferPad, BufferPadStats, BufferSlice, BufferTrim, BufferTrimStats};

// Timing
pub use timeout::{Debounce, DebounceStats, Throttle, ThrottleStats, Timeout, TimeoutStats};

// Routing
pub use concat::{Concat, ConcatStats, ConcatStream};
pub use funnel::{Funnel, FunnelInput, FunnelStats};
pub use selector::{
    InputSelector, InputSelectorStats, OutputSelector, OutputSelectorStats, SelectorInput,
    SelectorOutput,
};
pub use streamid_demux::{StreamIdDemux, StreamIdDemuxStats, StreamOutput};
pub use tee::Tee;

// Tier 3: Network elements
pub use multicast::{UdpMulticastSink, UdpMulticastSrc, UdpMulticastStats};
pub use unix::{AsyncUnixSink, AsyncUnixSrc, UnixMode, UnixSink, UnixSrc};

// HTTP (feature-gated)
#[cfg(feature = "http")]
pub use http::{HttpMethod, HttpSink, HttpSinkStats, HttpSrc, HttpStreamingSink};

// WebSocket (feature-gated)
#[cfg(feature = "websocket")]
pub use websocket::{WebSocketSink, WebSocketSrc, WebSocketStats, WsMessageType};

// Zenoh (feature-gated)
#[cfg(feature = "zenoh")]
pub use zenoh::{
    ZenohCongestionControl, ZenohPriority, ZenohQuerier, ZenohQuery, ZenohQueryable, ZenohSink,
    ZenohSrc, ZenohStats,
};

// RTP (feature-gated)
#[cfg(feature = "rtp")]
pub use rtp::{AsyncRtpSink, AsyncRtpSrc, RtpSink, RtpSinkStats, RtpSrc, RtpSrcStats};

// RTCP (feature-gated)
#[cfg(feature = "rtp")]
pub use rtcp::{
    ReceiverReportInfo, ReceptionStats, RtcpHandler, RtcpPacketInfo, RtcpStats, SenderStats,
};

// RTP Codec Payloaders/Depayloaders (feature-gated)
#[cfg(feature = "rtp")]
pub use rtp_codecs::{
    DepayStats, PayStats, RtpH264Depay, RtpH264Pay, RtpH265Depay, RtpH265Pay, RtpOpusDepay,
    RtpVp8Depay, RtpVp8Pay, RtpVp9Depay, RtpVp9Pay,
};

// Jitter Buffer (feature-gated)
#[cfg(feature = "rtp")]
pub use jitter_buffer::{
    AsyncJitterBuffer, JitterBufferConfig, JitterBufferStats, LossInfo, RtpJitterBuffer,
};

// RTSP (feature-gated)
#[cfg(feature = "rtsp")]
pub use rtsp::{
    MediaType, RtspConfig, RtspCredentials, RtspFrame, RtspSession, RtspSrc, RtspStats,
    RtspTransport, StreamInfo, StreamSelection,
};

// MPEG-TS (feature-gated)
#[cfg(feature = "mpeg-ts")]
pub use mpegts::{
    TS_PACKET_SIZE, TsDemux, TsDemuxStats, TsElementaryStream, TsFrame, TsProgram, TsStreamType,
};

// Iced Video Sink (feature-gated)
#[cfg(feature = "iced-sink")]
pub use iced_sink::{
    IcedVideoSink, IcedVideoSinkConfig, IcedVideoSinkHandle, IcedVideoSinkStats, InputPixelFormat,
};

// Tier 5: Data processing
pub use data_processing::{
    BufferConcat, BufferConcatStats, BufferJoin, BufferJoinStats, BufferSplit, BufferSplitStats,
    DuplicateFilter, DuplicateFilterStats, ExtractedMetadata, MetadataExtract, RangeFilter,
    RangeFilterStats, RegexFilter, RegexFilterStats,
};

// IPC elements for cross-process pipelines
pub use ipc::{IpcSink, IpcSrc};
