//! RTSP client source element.
//!
//! This module provides an RTSP source that connects to cameras and streaming
//! servers, receiving RTP streams and demuxing them into video/audio frames.
//!
//! # Example: as a pipeline source
//!
//! [`RtspSession`] implements [`AsyncSource`](crate::element::AsyncSource), so
//! it is an ordinary graph node. Note it is `add_async_source`, not
//! `add_source` — RTSP is inherently async and cannot satisfy the sync
//! [`Source`](crate::element::Source) trait.
//!
//! ```rust,ignore
//! use parallax::elements::{RtspSrc, RtspTransport};
//!
//! let session = RtspSrc::new("rtsp://192.168.1.100/stream1")
//!     .with_transport(RtspTransport::TcpInterleaved)
//!     .with_credentials("admin", "password")
//!     .connect()
//!     .await?;
//!
//! let mut pipeline = Pipeline::new();
//! let src = pipeline.add_async_source("rtsp", session);
//! let sink = pipeline.add_async_sink("out", my_sink);
//! pipeline.link(src, sink)?;
//! pipeline.run().await?;
//! ```
//!
//! With the default [`RtspFrameFormat::AnnexB`], the buffers this produces feed
//! an `H264Decoder` directly — SPS/PPS ride in-band on every keyframe. Use
//! [`RtspFrameFormat::LengthPrefixed`] when muxing to MP4 instead. See
//! `examples/58_rtsp_display.rs`.
//!
//! # Geometry is not always known at connect time
//!
//! Plenty of cameras ship an SDP with no `a=framesize` and no usable
//! `sprop-parameter-sets`, so [`StreamInfo::dimensions`] is `None` when
//! `connect()` returns. The stream still announces its geometry — in the first
//! in-band SPS — and parallax adopts it as soon as it arrives, typically within
//! one keyframe.
//!
//! So a consumer that needs the frame size must be able to *wait* for it rather
//! than read once and treat `None` as permanent. Because adding the session to a
//! pipeline moves it, take an [`RtspStreamInfoHandle`] first:
//!
//! ```rust,ignore
//! let session = RtspSrc::new(url).connect().await?;
//! let info = session.stream_info_handle();   // BEFORE the move
//! pipeline.add_async_source("rtsp", session);
//!
//! let (width, height) = info.wait_for_dimensions(0).await
//!     .ok_or("session ended before the first parameter set")?;
//! ```
//!
//! # Example: owning the pump yourself
//!
//! The manual path remains available for callers who want to drive the session
//! by hand rather than hand it to an executor — see
//! `examples/57_rtsp_capture.rs`.
//!
//! ```rust,ignore
//! let mut session = RtspSrc::new("rtsp://192.168.1.100/stream1").connect().await?;
//!
//! while let Some(frame) = session.next_frame().await? {
//!     match frame {
//!         RtspFrame::Video(buf) => { /* H.264/H.265 access unit */ },
//!         RtspFrame::Audio(buf) => { /* AAC/Opus frame */ },
//!     }
//! }
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::element::ProduceResult;
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use crate::metadata::{BufferFlags, Metadata, RtpMeta};

use futures::StreamExt;
use retina::client::{Demuxed, Session, SessionOptions, SetupOptions};
use retina::codec::{AudioFrame, CodecItem, VideoFrame};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
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

/// Output framing for depacketized frames.
///
/// Controls how retina frames H.26x NAL units (and AAC audio) in the buffers
/// returned by [`RtspSession::next_frame`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RtspFrameFormat {
    /// Self-describing output: Annex-B start codes with SPS/PPS prepended to
    /// every keyframe, ADTS-wrapped AAC. This is what parallax's `H264Decoder`,
    /// typefind, and raw-bytestream file dumps expect.
    #[default]
    AnnexB,
    /// ISO-BMFF style: 4-byte length-prefixed NALs with parameter sets only in
    /// [`StreamInfo::codec_data`], raw AAC. For feeding an MP4 muxer directly.
    LengthPrefixed,
}

impl From<RtspFrameFormat> for retina::codec::FrameFormat {
    fn from(f: RtspFrameFormat) -> Self {
        match f {
            RtspFrameFormat::AnnexB => retina::codec::FrameFormat::SIMPLE,
            RtspFrameFormat::LengthPrefixed => retina::codec::FrameFormat::MP4,
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
    /// Output framing for video/audio frames.
    pub frame_format: RtspFrameFormat,
    /// User agent string.
    pub user_agent: String,
    /// Whether to send TEARDOWN on close.
    pub teardown: retina::client::TeardownPolicy,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Maximum timestamp jump in seconds before resync.
    pub max_timestamp_jump_secs: u32,
    /// Drop video frames until the first keyframe.
    ///
    /// Joining a live stream lands mid-GOP, so the first frames reference a
    /// picture the decoder never saw. Feeding them to a decoder produces either
    /// garbage or an error. Defaults to `true` — every correct consumer wants
    /// it, and every one of them was writing the same skip loop by hand.
    pub skip_until_keyframe: bool,
}

impl Default for RtspConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            transport: RtspTransport::default(),
            credentials: None,
            stream_selection: StreamSelection::default(),
            frame_format: RtspFrameFormat::default(),
            user_agent: "Parallax RTSP Client".into(),
            teardown: retina::client::TeardownPolicy::Auto,
            connect_timeout: Duration::from_secs(10),
            max_timestamp_jump_secs: 10,
            skip_until_keyframe: true,
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
    /// For video: SDP frame rate hint if present (`a=framerate`).
    pub framerate: Option<f32>,
    /// Codec initialization data if known (H.264: SPS/PPS from
    /// `sprop-parameter-sets`; AAC: AudioSpecificConfig). Lets a consumer
    /// initialize a decoder before the first in-band parameter sets arrive.
    /// For H.26x the encoding follows [`RtspConfig::frame_format`]: Annex-B
    /// start codes under [`RtspFrameFormat::AnnexB`], a decoder configuration
    /// record under [`RtspFrameFormat::LengthPrefixed`].
    pub codec_data: Option<Vec<u8>>,
}

/// Observe an [`RtspSession`]'s stream metadata while the session is running.
///
/// [`RtspSession`] implements [`AsyncSource`](crate::element::AsyncSource), so
/// adding it to a pipeline **moves** it and [`RtspSession::streams`] is no
/// longer reachable. Clone this handle out of the session *before* the move —
/// the same rule the runtime control handles follow.
///
/// This matters because geometry is not always known at connect time. Plenty of
/// cameras ship an SDP with no `a=framesize` and no usable
/// `sprop-parameter-sets`, so [`StreamInfo::dimensions`] starts out `None` and
/// is only filled in once the first in-band SPS arrives. Use
/// [`wait_for_dimensions`](Self::wait_for_dimensions) rather than reading once
/// and giving up.
///
/// # Example
///
/// ```rust,ignore
/// let session = RtspSrc::new(url).connect().await?;
/// let info = session.stream_info_handle();   // BEFORE the move
/// pipeline.add_async_source("rtsp", session);
/// let handle = executor.start(&mut pipeline)?;
///
/// // Resolves at connect time if the SDP had geometry, otherwise one
/// // keyframe later.
/// if let Some((w, h)) = info.wait_for_dimensions(0).await {
///     println!("{w}x{h}");
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RtspStreamInfoHandle {
    rx: watch::Receiver<Arc<Vec<StreamInfo>>>,
}

impl RtspStreamInfoHandle {
    /// A snapshot of what is currently known about every stream.
    pub fn streams(&self) -> Arc<Vec<StreamInfo>> {
        self.rx.borrow().clone()
    }

    /// A snapshot of one stream, by index.
    pub fn stream(&self, index: usize) -> Option<StreamInfo> {
        self.rx.borrow().get(index).cloned()
    }

    /// Wait until the dimensions of stream `index` are known.
    ///
    /// Returns immediately when they already are — including at connect time,
    /// for an SDP that carried geometry. Otherwise it resolves when the first
    /// in-band parameter set arrives, typically within one keyframe.
    ///
    /// Returns `None` if the session is dropped first, or if `index` names no
    /// stream, so a caller awaiting a dead session is not left hanging.
    pub async fn wait_for_dimensions(&self, index: usize) -> Option<(u32, u32)> {
        let mut rx = self.rx.clone();
        loop {
            {
                let streams = rx.borrow_and_update();
                // An out-of-range index can never resolve: stop rather than
                // wait forever for a stream that does not exist.
                let stream = streams.get(index)?;
                if let Some(dimensions) = stream.dimensions {
                    return Some(dimensions);
                }
            }
            // Err means every sender is gone, i.e. the session ended.
            rx.changed().await.ok()?;
        }
    }
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

    /// Set the output framing for video/audio frames.
    pub fn with_frame_format(mut self, format: RtspFrameFormat) -> Self {
        self.config.frame_format = format;
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

    /// Emit video frames from the first keyframe on, or from wherever the join
    /// happened to land.
    ///
    /// Defaults to skipping (`true`). Turn it off only if you are recording the
    /// raw stream rather than decoding it.
    pub fn skip_until_keyframe(mut self, skip: bool) -> Self {
        self.config.skip_until_keyframe = skip;
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

/// Parse an RTSP URL, lifting embedded `user:pass@` credentials out of it
/// (retina rejects URLs that contain credentials).
fn split_url_credentials(url_str: &str) -> Result<(Url, Option<RtspCredentials>)> {
    let mut url =
        Url::parse(url_str).map_err(|e| Error::Element(format!("Invalid RTSP URL: {}", e)))?;
    let mut credentials = None;
    if !url.username().is_empty() || url.password().is_some() {
        credentials = Some(RtspCredentials {
            username: url.username().to_string(),
            password: url.password().unwrap_or_default().to_string(),
        });
        let _ = url.set_username("");
        let _ = url.set_password(None);
    }
    Ok((url, credentials))
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
    /// Stream information. Authoritative copy; the hot path reads it directly
    /// and [`RtspSession::publish_streams`] mirrors it to observers.
    streams: Vec<StreamInfo>,
    /// Mirror of `streams` for [`RtspStreamInfoHandle`]s, which outlive the
    /// move of this session into a pipeline.
    stream_info_tx: watch::Sender<Arc<Vec<StreamInfo>>>,
    /// Statistics.
    stats: RtspStats,
    /// Selected stream indices.
    selected_streams: Vec<usize>,
    /// Arena for output buffers.
    arena: Option<SharedArena>,
    /// Drop video frames until the first keyframe (see
    /// [`RtspConfig::skip_until_keyframe`]).
    skip_until_keyframe: bool,
    /// Whether the first keyframe has been seen.
    saw_keyframe: bool,
}

impl RtspSession {
    /// Connect to an RTSP server.
    async fn connect(config: RtspConfig) -> Result<Self> {
        // Parse URL. retina rejects URLs with embedded credentials, so lift
        // `rtsp://user:pass@host/...` into RtspCredentials (explicit
        // `with_credentials` wins if both are present).
        let (url, url_credentials) = split_url_credentials(&config.url)?;
        let credentials = config.credentials.or(url_credentials);

        // Build session options
        let mut session_opts = SessionOptions::default()
            .user_agent(config.user_agent.clone())
            .teardown(config.teardown);

        if let Some(creds) = credentials {
            session_opts = session_opts.creds(Some(creds.into()));
        }

        // Describe (connect_timeout covers TCP connect + DESCRIBE round-trip)
        let mut session =
            tokio::time::timeout(config.connect_timeout, Session::describe(url, session_opts))
                .await
                .map_err(|_| {
                    Error::Element(format!(
                        "RTSP DESCRIBE timed out after {:?}",
                        config.connect_timeout
                    ))
                })?
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

            // Parameters parsed by retina from the SDP (sprop-parameter-sets
            // etc.). Video parameters may be absent until the first frame for
            // streams whose SDP omits them.
            let mut dimensions = None;
            let mut sample_rate = None;
            let mut codec_data = None;
            match stream.parameters() {
                Some(retina::codec::ParametersRef::Video(video)) => {
                    dimensions = Some(video.pixel_dimensions());
                    let extra = video.extra_data();
                    if !extra.is_empty() {
                        codec_data = Some(extra.to_vec());
                    }
                }
                Some(retina::codec::ParametersRef::Audio(audio)) => {
                    sample_rate = Some(audio.clock_rate());
                    let extra = audio.extra_data();
                    if !extra.is_empty() {
                        codec_data = Some(extra.to_vec());
                    }
                }
                _ => {}
            }

            streams.push(StreamInfo {
                index: i,
                media_type,
                codec,
                clock_rate,
                dimensions,
                channels: stream.channels().map(|c| c.get()),
                sample_rate,
                framerate: stream.framerate(),
                codec_data,
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
        let frame_format: retina::codec::FrameFormat = config.frame_format.into();
        for &i in &selected_streams {
            tokio::time::timeout(
                config.connect_timeout,
                session.setup(
                    i,
                    SetupOptions::default()
                        .transport(transport.clone())
                        .frame_format(frame_format),
                ),
            )
            .await
            .map_err(|_| Error::Element(format!("RTSP SETUP timed out for stream {}", i)))?
            .map_err(|e| Error::Element(format!("RTSP SETUP failed for stream {}: {}", i, e)))?;
        }

        // Play
        let play_opts = retina::client::PlayOptions::default()
            .enforce_timestamps_with_max_jump_secs(
                NonZeroU32::new(config.max_timestamp_jump_secs)
                    .unwrap_or(NonZeroU32::new(10).unwrap()),
            );

        let session = tokio::time::timeout(config.connect_timeout, session.play(play_opts))
            .await
            .map_err(|_| {
                Error::Element(format!(
                    "RTSP PLAY timed out after {:?}",
                    config.connect_timeout
                ))
            })?
            .map_err(|e| Error::Element(format!("RTSP PLAY failed: {}", e)))?
            .demuxed()
            .map_err(|e| Error::Element(format!("Failed to demux RTSP session: {}", e)))?;

        let (stream_info_tx, _) = watch::channel(Arc::new(streams.clone()));

        Ok(Self {
            session,
            streams,
            stream_info_tx,
            stats: RtspStats {
                connected_at: Some(std::time::Instant::now()),
                ..Default::default()
            },
            selected_streams,
            arena: None,
            skip_until_keyframe: config.skip_until_keyframe,
            saw_keyframe: false,
        })
    }

    /// Get information about available streams.
    ///
    /// Unreachable once the session has been moved into a pipeline — take a
    /// [`stream_info_handle`](Self::stream_info_handle) beforehand for that.
    pub fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    /// A cloneable handle that keeps reporting stream metadata after this
    /// session has been moved into a pipeline.
    ///
    /// Take it *before* `pipeline.add_async_source(session)`; see
    /// [`RtspStreamInfoHandle`].
    pub fn stream_info_handle(&self) -> RtspStreamInfoHandle {
        RtspStreamInfoHandle {
            rx: self.stream_info_tx.subscribe(),
        }
    }

    /// Mirror `self.streams` to every live [`RtspStreamInfoHandle`].
    fn publish_streams(&self) {
        self.stream_info_tx
            .send_replace(Arc::new(self.streams.clone()));
    }

    /// Adopt parameters retina parsed from the bitstream.
    ///
    /// The SDP is not always honest — no `a=framesize`, no usable
    /// `sprop-parameter-sets` — in which case `connect()` leaves
    /// [`StreamInfo::dimensions`] `None` even though the stream announces its
    /// geometry in the very first in-band SPS. retina re-parses parameter sets
    /// as they arrive on the wire, so the information is already there; this
    /// copies it across whenever a frame reports it changed.
    ///
    /// Also covers a mid-stream resolution change, which updates the same way.
    fn refresh_stream_info(&mut self, index: usize) {
        // Read everything out of `self.session` first: the borrow it hands back
        // has to be gone before we can touch `self.streams` and publish.
        let parsed =
            self.session
                .streams()
                .get(index)
                .and_then(|stream| match stream.parameters() {
                    Some(retina::codec::ParametersRef::Video(video)) => {
                        Some((video.pixel_dimensions(), video.extra_data().to_vec()))
                    }
                    _ => None,
                });

        let Some((dimensions, extra)) = parsed else {
            return;
        };
        let Some(info) = self.streams.get_mut(index) else {
            return;
        };

        let mut changed = false;
        if info.dimensions != Some(dimensions) {
            tracing::debug!(
                "rtspsrc: stream {} dimensions {:?} -> {}x{} (in-band parameter sets)",
                index,
                info.dimensions,
                dimensions.0,
                dimensions.1
            );
            info.dimensions = Some(dimensions);
            changed = true;
        }
        if !extra.is_empty() && info.codec_data.as_deref() != Some(extra.as_slice()) {
            info.codec_data = Some(extra);
            changed = true;
        }

        if changed {
            self.publish_streams();
        }
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
                            // retina re-parses in-band SPS/PPS and flags the
                            // frame that carried them. For a stream whose SDP
                            // had no geometry, this is where dimensions first
                            // become known.
                            if frame.has_new_parameters() {
                                self.refresh_stream_info(frame.stream_id());
                            }
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
    fn video_frame_to_buffer(&mut self, frame: VideoFrame) -> Result<Buffer> {
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
    fn audio_frame_to_buffer(&mut self, frame: AudioFrame) -> Result<Buffer> {
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
        let arena = self.arena.as_mut().unwrap();

        arena.reclaim();

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
    /// Pull the next buffer from the stream, or `None` at end of stream.
    ///
    /// The manual pump: use this when you want to own the loop. For a session
    /// that drives itself inside a pipeline, add it as a source — it implements
    /// [`AsyncSource`](crate::element::AsyncSource).
    ///
    /// (This was called `produce()`, which shadowed the `AsyncSource` method of
    /// the same name and so kept the session from ever *being* a source.)
    pub async fn next_buffer(&mut self) -> Result<Option<Buffer>> {
        loop {
            let Some(frame) = self.next_frame().await? else {
                return Ok(None);
            };

            // Joining a live stream lands mid-GOP: those frames reference a
            // picture the decoder never saw. Skip to the first keyframe.
            if self.skip_until_keyframe && frame.is_video() && !self.saw_keyframe {
                if !frame.buffer().metadata().is_keyframe() {
                    continue;
                }
                self.saw_keyframe = true;
            }

            return Ok(Some(frame.into_buffer()));
        }
    }
}

/// A connected RTSP session is a source like any other.
///
/// Before this, `RtspSrc` was a session API and nothing else: to get its frames
/// into a pipeline you had to spawn a task, pump `produce()` by hand, and shovel
/// the buffers through an `AppSrc`. Every caller wrote the same bridge.
///
/// ```rust,ignore
/// let session = RtspSrc::new("rtsp://camera/stream").connect().await?;
/// let src = pipeline.add_async_source("rtsp", session);
/// let dec = pipeline.add_filter("dec", H264Decoder::new()?);
/// pipeline.link(src, dec)?;
/// ```
impl crate::element::AsyncSource for RtspSession {
    async fn produce(
        &mut self,
        _ctx: &mut crate::element::ProduceContext<'_>,
    ) -> Result<ProduceResult> {
        match self.next_buffer().await? {
            Some(buffer) => Ok(ProduceResult::OwnBuffer(buffer)),
            None => Ok(ProduceResult::Eos),
        }
    }

    fn name(&self) -> &str {
        "rtspsrc"
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
        assert_eq!(config.frame_format, RtspFrameFormat::AnnexB);
    }

    #[test]
    fn test_split_url_credentials() {
        let (url, creds) =
            split_url_credentials("rtsp://demo:secret@cam.example:5541/stream?profile=1").unwrap();
        assert_eq!(url.as_str(), "rtsp://cam.example:5541/stream?profile=1");
        let creds = creds.unwrap();
        assert_eq!(creds.username, "demo");
        assert_eq!(creds.password, "secret");

        // Username without password
        let (url, creds) = split_url_credentials("rtsp://demo@cam.example/stream").unwrap();
        assert_eq!(url.as_str(), "rtsp://cam.example/stream");
        assert_eq!(creds.unwrap().password, "");

        // No credentials at all
        let (url, creds) = split_url_credentials("rtsp://cam.example/stream").unwrap();
        assert_eq!(url.as_str(), "rtsp://cam.example/stream");
        assert!(creds.is_none());

        assert!(split_url_credentials("not a url").is_err());
    }

    #[test]
    fn test_frame_format_conversion() {
        let annexb: retina::codec::FrameFormat = RtspFrameFormat::AnnexB.into();
        assert_eq!(annexb, retina::codec::FrameFormat::SIMPLE);

        let prefixed: retina::codec::FrameFormat = RtspFrameFormat::LengthPrefixed.into();
        assert_eq!(prefixed, retina::codec::FrameFormat::MP4);
    }

    // ---- RtspStreamInfoHandle -------------------------------------------
    //
    // Driving a real session needs a server (see `just rtsp-server` and
    // examples 57/58). These cover the handle itself, which is the part that
    // has to behave when the SDP was silent.

    fn video_stream(dimensions: Option<(u32, u32)>) -> StreamInfo {
        StreamInfo {
            index: 0,
            media_type: MediaType::Video,
            codec: "h264".into(),
            clock_rate: 90_000,
            dimensions,
            channels: None,
            sample_rate: None,
            framerate: None,
            codec_data: None,
        }
    }

    fn handle_over(
        streams: Vec<StreamInfo>,
    ) -> (watch::Sender<Arc<Vec<StreamInfo>>>, RtspStreamInfoHandle) {
        let (tx, rx) = watch::channel(Arc::new(streams));
        (tx, RtspStreamInfoHandle { rx })
    }

    #[tokio::test]
    async fn dimensions_from_the_sdp_resolve_immediately() {
        let (_tx, handle) = handle_over(vec![video_stream(Some((1920, 1080)))]);
        assert_eq!(handle.wait_for_dimensions(0).await, Some((1920, 1080)));
        assert_eq!(handle.stream(0).unwrap().dimensions, Some((1920, 1080)));
    }

    #[tokio::test]
    async fn a_silent_sdp_resolves_once_the_sps_arrives() {
        let (tx, handle) = handle_over(vec![video_stream(None)]);

        // Nothing known yet: the wait must not resolve.
        assert!(handle.stream(0).unwrap().dimensions.is_none());
        let waiter = tokio::spawn(async move { handle.wait_for_dimensions(0).await });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished(), "resolved before any SPS arrived");

        // ...and does once the first in-band parameter set lands.
        tx.send_replace(Arc::new(vec![video_stream(Some((1280, 720)))]));
        assert_eq!(waiter.await.unwrap(), Some((1280, 720)));
    }

    #[tokio::test]
    async fn a_dead_session_does_not_leave_a_waiter_hanging() {
        let (tx, handle) = handle_over(vec![video_stream(None)]);
        let waiter = tokio::spawn(async move { handle.wait_for_dimensions(0).await });
        tokio::task::yield_now().await;

        drop(tx);
        assert_eq!(
            waiter.await.unwrap(),
            None,
            "a dropped session must end the wait, not stall it"
        );
    }

    #[tokio::test]
    async fn an_unknown_stream_index_gives_up_rather_than_waiting() {
        let (_tx, handle) = handle_over(vec![video_stream(None)]);
        assert_eq!(handle.wait_for_dimensions(7).await, None);
        assert!(handle.stream(7).is_none());
    }

    #[test]
    fn the_handle_snapshots_every_stream() {
        let (_tx, handle) = handle_over(vec![video_stream(Some((640, 480)))]);
        let streams = handle.streams();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].dimensions, Some((640, 480)));
    }
}
