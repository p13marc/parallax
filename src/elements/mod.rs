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
//! Stream ID demux, MPEG-TS demux, MP4 demux
//!
//! ## [`mux`] - Multiplexing
//! MP4 muxer
//!
//! ## [`util`] - Utility Elements
//! PassThrough, Identity
//!
//! ## `codec` - Software Codecs (feature-gated)
//! AV1 decoder (dav1d), AV1 encoder (rav1e) - requires codec feature flags
//!
//! ## `device` - Device Capture (feature-gated)
//! PipeWire, libcamera, V4L2, ALSA - hardware device capture and playback

pub mod app;

#[cfg(any(
    feature = "h264",
    feature = "av1-encode",
    feature = "av1-decode",
    feature = "audio-flac",
    feature = "audio-mp3",
    feature = "audio-aac",
    feature = "audio-vorbis",
    feature = "opus",
    feature = "aac-encode",
    feature = "image-jpeg",
    feature = "image-png"
))]
pub mod codec;
pub mod demux;

#[cfg(any(
    feature = "pipewire",
    feature = "libcamera",
    feature = "v4l2",
    feature = "alsa"
))]
pub mod device;
pub mod flow;
pub mod io;
pub mod ipc;
pub mod metadata;
pub mod mux;
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
    RegexFilter, RegexFilterStats, SampleFilter, SampleMode, ScaleMode, SequenceNumber,
    TimestampMode, Timestamper, Unbatch, UnbatchStats, VideoScale,
};

// App integration
pub use app::{AppSink, AppSinkHandle, AppSinkStats, AppSrc, AppSrcHandle, AppSrcStats};

#[cfg(feature = "display")]
pub use app::AutoVideoSink;

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

#[cfg(feature = "mp4-demux")]
pub use demux::{
    Mp4AudioInfo, Mp4Codec, Mp4Demux, Mp4DemuxStats, Mp4Sample, Mp4Track, Mp4TrackType,
    Mp4VideoInfo,
};

// Mux
#[cfg(feature = "mp4-demux")]
pub use mux::{
    AudioCodecConfig, Mp4AudioTrackConfig, Mp4Mux, Mp4MuxConfig, Mp4MuxStats, Mp4VideoTrackConfig,
    VideoCodecConfig,
};

#[cfg(feature = "mpeg-ts")]
pub use mux::{TsMux, TsMuxConfig, TsMuxStats, TsMuxStreamType, TsMuxTrack};

// Metadata
pub use metadata::{KlvEncoder, KlvTag, StanagMetadataBuilder, Uls, decode_ber_length};

// Utility
pub use util::{Identity, IdentityStats, PassThrough};

// Video codecs - common types
#[cfg(any(feature = "av1-encode", feature = "av1-decode"))]
pub use codec::{PixelFormat as CodecPixelFormat, VideoFrame};

// Video codecs - H.264
#[cfg(feature = "h264")]
pub use codec::{DecodedFrame, H264Decoder, H264Encoder, H264EncoderConfig};

// Video codecs - AV1
#[cfg(feature = "av1-decode")]
pub use codec::Dav1dDecoder;

#[cfg(feature = "av1-encode")]
pub use codec::{Rav1eConfig, Rav1eEncoder};

// Audio codecs (Symphonia - FLAC, MP3, AAC, Vorbis)
#[cfg(any(
    feature = "audio-flac",
    feature = "audio-mp3",
    feature = "audio-aac",
    feature = "audio-vorbis"
))]
pub use codec::{AudioFormat, AudioFrameInfo, SampleFormat, SymphoniaDecoder};

// Audio codecs - Opus
#[cfg(feature = "opus")]
pub use codec::{OpusApplication, OpusDecoder, OpusEncoder};

// Audio codecs - AAC encoder
#[cfg(feature = "aac-encode")]
pub use codec::AacEncoder;

// Audio codec traits and wrappers (always available when any audio codec is enabled)
#[cfg(any(
    feature = "audio-flac",
    feature = "audio-mp3",
    feature = "audio-aac",
    feature = "audio-vorbis",
    feature = "opus",
    feature = "aac-encode"
))]
pub use codec::{
    AudioDecoder, AudioDecoderElement, AudioEncoder, AudioEncoderElement, AudioSampleFormat,
    AudioSamples,
};

// Image codecs - common types
#[cfg(any(feature = "image-jpeg", feature = "image-png"))]
pub use codec::{ColorType, ImageFrame};

// Image codecs - JPEG
#[cfg(feature = "image-jpeg")]
pub use codec::JpegDecoder;

// Image codecs - PNG
#[cfg(feature = "image-png")]
pub use codec::{PngDecoder, PngEncoder};

// Device capture - common types
#[cfg(any(
    feature = "pipewire",
    feature = "libcamera",
    feature = "v4l2",
    feature = "alsa"
))]
pub use device::{
    AudioCaptureDevice, CameraLocation, CaptureBackend, DeviceError, VideoCaptureDevice,
    detect_audio_backend, detect_video_backend, enumerate_audio_devices, enumerate_video_devices,
};

// Device capture - PipeWire
#[cfg(feature = "pipewire")]
pub use device::{PipeWireSink, PipeWireSrc, PipeWireTarget};

// Device capture - libcamera
#[cfg(feature = "libcamera")]
pub use device::{LibCameraConfig, LibCameraInfo, LibCameraSrc};

// Device capture - V4L2
#[cfg(feature = "v4l2")]
pub use device::{V4l2DeviceInfo, V4l2Src};

// Device capture - ALSA
#[cfg(feature = "alsa")]
pub use device::{AlsaDeviceInfo, AlsaFormat, AlsaSampleFormat, AlsaSink, AlsaSrc};
