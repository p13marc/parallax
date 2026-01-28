//! RTSP client source element.
//!
//! This module provides an RTSP source that connects to cameras and streaming
//! servers, receiving RTP streams and demuxing them into video/audio frames.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::RtspSrc;
//!
//! // Connect to an RTSP camera
//! let src = RtspSrc::new("rtsp://192.168.1.100/stream1")
//!     .with_transport(RtspTransport::TcpInterleaved)
//!     .with_credentials("admin", "password");
//!
//! // Run in async context
//! let mut src = src.connect().await?;
//!
//! // Receive frames
//! while let Some(frame) = src.next_frame().await? {
//!     match frame {
//!         RtspFrame::Video(buf) => { /* H.264/H.265 access unit */ },
//!         RtspFrame::Audio(buf) => { /* AAC/Opus frame */ },
//!     }
//! }
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use crate::metadata::{BufferFlags, Metadata, RtpMeta};

use futures::StreamExt;
use retina::client::{Demuxed, Session, SessionOptions, SetupOptions};
use retina::codec::{AudioFrame, CodecItem, VideoFrame};
use std::num::NonZeroU32;
use std::time::Duration;
use url::Url;

// ============================================================================
// Configuration Types
// ============================================================================

/// RTSP transport mode.
#[derive(Debug, Clone, Default)]
pub enum RtspTransport {
    /// TCP interleaved (RTP over RTSP connection).
    /// Most reliable, works through firewalls/NAT.
    #[default]
    TcpInterleaved,
    /// UDP transport.
    /// Lower latency but may have firewall issues.
    Udp,
}

impl From<RtspTransport> for retina::client::Transport {
    fn from(t: RtspTransport) -> Self {
        match t {
            RtspTransport::TcpInterleaved => {
                retina::client::Transport::Tcp(retina::client::TcpTransportOptions::default())
            }
            RtspTransport::Udp => {
                retina::client::Transport::Udp(retina::client::UdpTransportOptions::default())
            }
        }
    }
}

/// RTSP authentication credentials.
#[derive(Debug, Clone)]
pub struct RtspCredentials {
    /// Username for authentication.
    pub username: String,
    /// Password for authentication.
    pub password: String,
}

impl From<RtspCredentials> for retina::client::Credentials {
    fn from(c: RtspCredentials) -> Self {
        retina::client::Credentials {
            username: c.username,
            password: c.password,
        }
    }
}

/// Stream selection policy.
#[derive(Debug, Clone, Default)]
pub enum StreamSelection {
    /// Select all available streams.
    #[default]
    All,
    /// Select only video streams.
    VideoOnly,
    /// Select only audio streams.
    AudioOnly,
    /// Select specific stream indices.
    Indices(Vec<usize>),
}

/// Configuration for RtspSrc.
#[derive(Debug, Clone)]
pub struct RtspConfig {
    /// RTSP URL to connect to.
    pub url: String,
    /// Transport mode.
    pub transport: RtspTransport,
    /// Authentication credentials.
    pub credentials: Option<RtspCredentials>,
    /// Stream selection policy.
    pub stream_selection: StreamSelection,
    /// User agent string.
    pub user_agent: String,
    /// Whether to send TEARDOWN on close.
    pub teardown: retina::client::TeardownPolicy,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Maximum timestamp jump in seconds before resync.
    pub max_timestamp_jump_secs: u32,
}

impl Default for RtspConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            transport: RtspTransport::default(),
            credentials: None,
            stream_selection: StreamSelection::default(),
            user_agent: "Parallax RTSP Client".into(),
            teardown: retina::client::TeardownPolicy::Auto,
            connect_timeout: Duration::from_secs(10),
            max_timestamp_jump_secs: 10,
        }
    }
}

// ============================================================================
// Stream Information
// ============================================================================

/// Information about an RTSP stream.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Stream index.
    pub index: usize,
    /// Stream type (video, audio, application).
    pub media_type: MediaType,
    /// Codec name (e.g., "h264", "aac").
    pub codec: String,
    /// Clock rate in Hz.
    pub clock_rate: u32,
    /// For video: dimensions if known.
    pub dimensions: Option<(u32, u32)>,
    /// For audio: channels if known.
    pub channels: Option<u16>,
    /// For audio: sample rate if known.
    pub sample_rate: Option<u32>,
}

/// Media type of a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// Video stream (H.264, H.265, VP8, etc.).
    Video,
    /// Audio stream (AAC, Opus, etc.).
    Audio,
    /// Application/metadata stream (ONVIF, etc.).
    Application,
}

// ============================================================================
// Frame Types
// ============================================================================

/// A frame received from an RTSP stream.
#[derive(Debug)]
pub enum RtspFrame {
    /// Video frame (e.g., H.264 access unit).
    Video(Buffer),
    /// Audio frame (e.g., AAC frame).
    Audio(Buffer),
}

impl RtspFrame {
    /// Returns true if this is a video frame.
    pub fn is_video(&self) -> bool {
        matches!(self, RtspFrame::Video(_))
    }

    /// Returns true if this is an audio frame.
    pub fn is_audio(&self) -> bool {
        matches!(self, RtspFrame::Audio(_))
    }

    /// Get the buffer, consuming the frame.
    pub fn into_buffer(self) -> Buffer {
        match self {
            RtspFrame::Video(buf) => buf,
            RtspFrame::Audio(buf) => buf,
        }
    }

    /// Get a reference to the buffer.
    pub fn buffer(&self) -> &Buffer {
        match self {
            RtspFrame::Video(buf) => buf,
            RtspFrame::Audio(buf) => buf,
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for an RTSP source.
#[derive(Debug, Clone, Default)]
pub struct RtspStats {
    /// Total video frames received.
    pub video_frames: u64,
    /// Total audio frames received.
    pub audio_frames: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Video keyframes received.
    pub video_keyframes: u64,
    /// RTCP packets received.
    pub rtcp_packets: u64,
    /// Connection start time.
    pub connected_at: Option<std::time::Instant>,
}

// ============================================================================
// RtspSrc Builder
// ============================================================================

/// RTSP source element builder.
///
/// Use this to configure an RTSP connection before connecting.
pub struct RtspSrc {
    config: RtspConfig,
}

impl RtspSrc {
    /// Create a new RTSP source with the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            config: RtspConfig {
                url: url.into(),
                ..Default::default()
            },
        }
    }

    /// Set the transport mode.
    pub fn with_transport(mut self, transport: RtspTransport) -> Self {
        self.config.transport = transport;
        self
    }

    /// Set authentication credentials.
    pub fn with_credentials(
        mut self,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.config.credentials = Some(RtspCredentials {
            username: username.into(),
            password: password.into(),
        });
        self
    }

    /// Set stream selection policy.
    pub fn with_stream_selection(mut self, selection: StreamSelection) -> Self {
        self.config.stream_selection = selection;
        self
    }

    /// Set the user agent string.
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.config.user_agent = user_agent.into();
        self
    }

    /// Set the teardown policy.
    pub fn with_teardown(mut self, policy: retina::client::TeardownPolicy) -> Self {
        self.config.teardown = policy;
        self
    }

    /// Set the connection timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.connect_timeout = timeout;
        self
    }

    /// Set video-only stream selection.
    pub fn video_only(mut self) -> Self {
        self.config.stream_selection = StreamSelection::VideoOnly;
        self
    }

    /// Set audio-only stream selection.
    pub fn audio_only(mut self) -> Self {
        self.config.stream_selection = StreamSelection::AudioOnly;
        self
    }

    /// Get the configuration.
    pub fn config(&self) -> &RtspConfig {
        &self.config
    }

    /// Connect to the RTSP server and return an active session.
    ///
    /// This performs DESCRIBE, SETUP, and PLAY operations.
    pub async fn connect(self) -> Result<RtspSession> {
        RtspSession::connect(self.config).await
    }
}

// ============================================================================
// RtspSession (Connected)
// ============================================================================

/// An active RTSP session.
///
/// This represents a connected and playing RTSP session that can produce frames.
pub struct RtspSession {
    /// The demuxed session.
    session: Demuxed,
    /// Stream information.
    streams: Vec<StreamInfo>,
    /// Statistics.
    stats: RtspStats,
    /// Selected stream indices.
    selected_streams: Vec<usize>,
    /// Arena for output buffers.
    arena: Option<SharedArena>,
}

impl RtspSession {
    /// Connect to an RTSP server.
    async fn connect(config: RtspConfig) -> Result<Self> {
        // Parse URL
        let url = Url::parse(&config.url)
            .map_err(|e| Error::Element(format!("Invalid RTSP URL: {}", e)))?;

        // Build session options
        let mut session_opts = SessionOptions::default()
            .user_agent(config.user_agent.clone())
            .teardown(config.teardown);

        if let Some(creds) = config.credentials {
            session_opts = session_opts.creds(Some(creds.into()));
        }

        // Describe
        let mut session = Session::describe(url, session_opts)
            .await
            .map_err(|e| Error::Element(format!("RTSP DESCRIBE failed: {}", e)))?;

        // Get stream information
        let mut streams = Vec::new();
        for (i, stream) in session.streams().iter().enumerate() {
            let media_type = match stream.media() {
                "video" => MediaType::Video,
                "audio" => MediaType::Audio,
                _ => MediaType::Application,
            };

            let codec = stream.encoding_name().to_lowercase();
            let clock_rate = stream.clock_rate_hz();

            streams.push(StreamInfo {
                index: i,
                media_type,
                codec,
                clock_rate,
                dimensions: None, // Could extract from SDP if needed
                channels: None,
                sample_rate: None,
            });
        }

        // Select streams based on policy
        let selected_streams: Vec<usize> = match &config.stream_selection {
            StreamSelection::All => (0..streams.len()).collect(),
            StreamSelection::VideoOnly => streams
                .iter()
                .filter(|s| s.media_type == MediaType::Video)
                .map(|s| s.index)
                .collect(),
            StreamSelection::AudioOnly => streams
                .iter()
                .filter(|s| s.media_type == MediaType::Audio)
                .map(|s| s.index)
                .collect(),
            StreamSelection::Indices(indices) => indices.clone(),
        };

        if selected_streams.is_empty() {
            return Err(Error::Element("No streams selected".into()));
        }

        // Setup selected streams
        let transport: retina::client::Transport = config.transport.into();
        for &i in &selected_streams {
            session
                .setup(i, SetupOptions::default().transport(transport.clone()))
                .await
                .map_err(|e| {
                    Error::Element(format!("RTSP SETUP failed for stream {}: {}", i, e))
                })?;
        }

        // Play
        let play_opts = retina::client::PlayOptions::default()
            .enforce_timestamps_with_max_jump_secs(
                NonZeroU32::new(config.max_timestamp_jump_secs)
                    .unwrap_or(NonZeroU32::new(10).unwrap()),
            );

        let session = session
            .play(play_opts)
            .await
            .map_err(|e| Error::Element(format!("RTSP PLAY failed: {}", e)))?
            .demuxed()
            .map_err(|e| Error::Element(format!("Failed to demux RTSP session: {}", e)))?;

        Ok(Self {
            session,
            streams,
            stats: RtspStats {
                connected_at: Some(std::time::Instant::now()),
                ..Default::default()
            },
            selected_streams,
            arena: None,
        })
    }

    /// Get information about available streams.
    pub fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    /// Get the selected stream indices.
    pub fn selected_streams(&self) -> &[usize] {
        &self.selected_streams
    }

    /// Get current statistics.
    pub fn stats(&self) -> &RtspStats {
        &self.stats
    }

    /// Receive the next frame from the RTSP stream.
    ///
    /// Returns `None` when the stream ends.
    pub async fn next_frame(&mut self) -> Result<Option<RtspFrame>> {
        loop {
            match self.session.next().await {
                Some(Ok(item)) => {
                    match item {
                        CodecItem::VideoFrame(frame) => {
                            let buffer = self.video_frame_to_buffer(frame)?;
                            self.stats.video_frames += 1;
                            self.stats.bytes_received += buffer.len() as u64;
                            if buffer.metadata().is_keyframe() {
                                self.stats.video_keyframes += 1;
                            }
                            return Ok(Some(RtspFrame::Video(buffer)));
                        }
                        CodecItem::AudioFrame(frame) => {
                            let buffer = self.audio_frame_to_buffer(frame)?;
                            self.stats.audio_frames += 1;
                            self.stats.bytes_received += buffer.len() as u64;
                            return Ok(Some(RtspFrame::Audio(buffer)));
                        }
                        CodecItem::Rtcp(_rtcp) => {
                            self.stats.rtcp_packets += 1;
                            // Continue to next item
                        }
                        _ => {
                            // Skip other item types
                        }
                    }
                }
                Some(Err(e)) => {
                    return Err(Error::Element(format!("RTSP stream error: {}", e)));
                }
                None => {
                    return Ok(None);
                }
            }
        }
    }

    /// Convert a retina VideoFrame to a Parallax Buffer.
    fn video_frame_to_buffer(&self, frame: VideoFrame) -> Result<Buffer> {
        let data = frame.data();
        let is_keyframe = frame.is_random_access_point();
        let timestamp = frame.timestamp();

        // Get stream info for clock rate
        let stream_info = self.streams.get(frame.stream_id());
        let clock_rate = stream_info.map(|s| s.clock_rate).unwrap_or(90000) as u128;

        // Build metadata
        let ts = timestamp.timestamp();
        let nanos = if ts >= 0 {
            (ts as u128 * 1_000_000_000) / clock_rate
        } else {
            0
        };

        let mut flags = BufferFlags::NONE;
        if is_keyframe {
            flags |= BufferFlags::SYNC_POINT;
        }

        let rtp_timestamp = ts as u32;
        let metadata = Metadata::new()
            .with_pts(ClockTime::from_nanos(nanos as u64))
            .with_stream_id(frame.stream_id() as u32)
            .with_flags(flags)
            .with_rtp(RtpMeta {
                seq: 0, // Not available from demuxed frame
                ts: rtp_timestamp,
                ssrc: 0,
                pt: 0,
                marker: is_keyframe,
            });

        self.create_buffer_from_bytes_with_metadata(data, metadata)
    }

    /// Convert a retina AudioFrame to a Parallax Buffer.
    fn audio_frame_to_buffer(&self, frame: AudioFrame) -> Result<Buffer> {
        let data = frame.data();
        let timestamp = frame.timestamp();

        // Get stream info for clock rate
        let stream_info = self.streams.get(frame.stream_id());
        let clock_rate = stream_info.map(|s| s.clock_rate).unwrap_or(48000) as u128;

        // Build metadata
        let ts = timestamp.timestamp();
        let nanos = if ts >= 0 {
            (ts as u128 * 1_000_000_000) / clock_rate
        } else {
            0
        };

        let rtp_timestamp = ts as u32;
        let metadata = Metadata::new()
            .with_pts(ClockTime::from_nanos(nanos as u64))
            .with_stream_id(frame.stream_id() as u32)
            .with_rtp(RtpMeta {
                seq: 0,
                ts: rtp_timestamp,
                ssrc: 0,
                pt: 0,
                marker: false,
            });

        self.create_buffer_from_bytes_with_metadata(data, metadata)
    }

    /// Create a buffer from bytes with the given metadata.
    fn create_buffer_from_bytes_with_metadata(
        &mut self,
        data: &[u8],
        metadata: Metadata,
    ) -> Result<Buffer> {
        // Lazily initialize arena
        if self.arena.is_none() {
            self.arena = Some(
                SharedArena::new(1024 * 1024, 32)
                    .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
            );
        }
        let arena = self.arena.as_ref().unwrap();

        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
        slot.data_mut()[..data.len()].copy_from_slice(data);

        let handle = MemoryHandle::with_len(slot, data.len());
        Ok(Buffer::new(handle, metadata))
    }
}

// ============================================================================
// Async Source Implementation
// ============================================================================

impl RtspSession {
    /// Produce the next buffer (for use as an async source).
    ///
    /// This is the main interface for pipeline integration.
    pub async fn produce(&mut self) -> Result<Option<Buffer>> {
        match self.next_frame().await? {
            Some(frame) => Ok(Some(frame.into_buffer())),
            None => Ok(None),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtsp_src_builder() {
        let src = RtspSrc::new("rtsp://example.com/stream")
            .with_transport(RtspTransport::TcpInterleaved)
            .with_credentials("user", "pass")
            .with_user_agent("Test Agent")
            .with_timeout(Duration::from_secs(5))
            .video_only();

        assert_eq!(src.config().url, "rtsp://example.com/stream");
        assert!(matches!(
            src.config().transport,
            RtspTransport::TcpInterleaved
        ));
        assert!(src.config().credentials.is_some());
        assert_eq!(src.config().user_agent, "Test Agent");
        assert_eq!(src.config().connect_timeout, Duration::from_secs(5));
        assert!(matches!(
            src.config().stream_selection,
            StreamSelection::VideoOnly
        ));
    }

    #[test]
    fn test_rtsp_transport_conversion() {
        let tcp: retina::client::Transport = RtspTransport::TcpInterleaved.into();
        assert!(matches!(tcp, retina::client::Transport::Tcp(_)));

        let udp: retina::client::Transport = RtspTransport::Udp.into();
        assert!(matches!(udp, retina::client::Transport::Udp(_)));
    }

    #[test]
    fn test_stream_selection_default() {
        let selection = StreamSelection::default();
        assert!(matches!(selection, StreamSelection::All));
    }

    #[test]
    fn test_rtsp_frame_methods() {
        // We can't easily test RtspFrame without creating real buffers,
        // but we can test the type definitions compile correctly
        let _: fn(&RtspFrame) -> bool = RtspFrame::is_video;
        let _: fn(&RtspFrame) -> bool = RtspFrame::is_audio;
    }

    #[test]
    fn test_media_type_equality() {
        assert_eq!(MediaType::Video, MediaType::Video);
        assert_ne!(MediaType::Video, MediaType::Audio);
    }

    #[test]
    fn test_rtsp_stats_default() {
        let stats = RtspStats::default();
        assert_eq!(stats.video_frames, 0);
        assert_eq!(stats.audio_frames, 0);
        assert_eq!(stats.bytes_received, 0);
        assert!(stats.connected_at.is_none());
    }

    #[test]
    fn test_rtsp_config_default() {
        let config = RtspConfig::default();
        assert!(config.url.is_empty());
        assert!(matches!(config.transport, RtspTransport::TcpInterleaved));
        assert!(config.credentials.is_none());
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
    }
}
