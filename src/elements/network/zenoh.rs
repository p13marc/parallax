//! Zenoh pub/sub and query elements.
//!
//! Provides Zenoh-based distributed communication for pipelines.
//!
//! - [`ZenohSrc`]: Subscribe to a Zenoh key expression (async source)
//! - [`ZenohSink`]: Publish to a Zenoh key expression (async sink)
//! - [`ZenohQueryable`]: Respond to Zenoh queries
//! - [`ZenohQuerier`]: Query Zenoh resources
//!
//! Requires the `zenoh` feature flag. The `zenoh-unstable` feature
//! additionally enables reliability control and matching listeners (zenoh's
//! own `unstable` API surface).
//!
//! # Wire format
//!
//! Samples carry the raw buffer bytes as payload and the buffer's
//! [`Metadata`] (PTS/DTS/duration/sequence/flags/format) in a versioned
//! rkyv **attachment** — see [`super::zenoh_wire`] and `docs/zenoh-wire.md`.
//! Non-parallax subscribers can consume the payload and ignore the
//! attachment; parallax subscribers reconstruct full metadata. Publishing to
//! non-parallax-aware consumers that reject attachments can be forced with
//! [`ZenohSink::without_metadata`].

#![cfg(feature = "zenoh")]

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{AsyncSink, AsyncSource, ConsumeContext, ProduceContext, ProduceResult};
use crate::error::{Error, Result};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::{BufferFlags, Metadata};
use std::time::Duration;
use zenoh::Session;
use zenoh::key_expr::KeyExpr;

use super::zenoh_wire::{KEY_EXPR_META, WireMetadata, encoding_for_format};

/// Congestion control mode for Zenoh publishing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZenohCongestionControl {
    /// Block if the network is congested.
    #[default]
    Block,
    /// Drop messages if the network is congested.
    Drop,
}

impl From<ZenohCongestionControl> for zenoh::qos::CongestionControl {
    fn from(cc: ZenohCongestionControl) -> Self {
        match cc {
            ZenohCongestionControl::Block => zenoh::qos::CongestionControl::Block,
            ZenohCongestionControl::Drop => zenoh::qos::CongestionControl::Drop,
        }
    }
}

/// Priority level for Zenoh messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZenohPriority {
    /// Real-time priority.
    RealTime,
    /// Interactive high priority.
    InteractiveHigh,
    /// Interactive low priority.
    InteractiveLow,
    /// Data high priority.
    DataHigh,
    /// Data priority (default).
    #[default]
    Data,
    /// Data low priority.
    DataLow,
    /// Background priority.
    Background,
}

impl From<ZenohPriority> for zenoh::qos::Priority {
    fn from(p: ZenohPriority) -> Self {
        match p {
            ZenohPriority::RealTime => zenoh::qos::Priority::RealTime,
            ZenohPriority::InteractiveHigh => zenoh::qos::Priority::InteractiveHigh,
            ZenohPriority::InteractiveLow => zenoh::qos::Priority::InteractiveLow,
            ZenohPriority::DataHigh => zenoh::qos::Priority::DataHigh,
            ZenohPriority::Data => zenoh::qos::Priority::Data,
            ZenohPriority::DataLow => zenoh::qos::Priority::DataLow,
            ZenohPriority::Background => zenoh::qos::Priority::Background,
        }
    }
}

/// Reliability mode for Zenoh publishing (requires `zenoh-unstable`).
#[cfg(feature = "zenoh-unstable")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZenohReliability {
    /// Deliver reliably (retransmit on loss).
    #[default]
    Reliable,
    /// Best effort (a lost sample is superseded by the next).
    BestEffort,
}

#[cfg(feature = "zenoh-unstable")]
impl From<ZenohReliability> for zenoh::qos::Reliability {
    fn from(r: ZenohReliability) -> Self {
        match r {
            ZenohReliability::Reliable => zenoh::qos::Reliability::Reliable,
            ZenohReliability::BestEffort => zenoh::qos::Reliability::BestEffort,
        }
    }
}

/// Default arena slot size for received samples (grows on demand).
/// Which wire contract an element speaks (#231, #223).
///
/// Not merely a choice of attachment codec: a profile fixes five coupled
/// decisions at once, and the combinations that cross them are nonsense.
///
/// | | `Parallax` | `Media` |
/// |---|---|---|
/// | attachment | rkyv [`WireMetadata`] | CBOR [`FrameMeta`](crate::wire::FrameMeta) |
/// | QoS default | Block / Data | best-effort · drop · interactive-high · express |
/// | sequence | forwarded from the buffer | owned by the publisher |
/// | wildcards | legal | refused |
/// | keyframe flag | `SYNC_POINT` passthrough | derived from the bytes |
///
/// `Media` implements the `@media` plane, a contract parallax does not own —
/// it is a conformant implementer, not the author. It never *builds* an
/// `@media` key (it has no origin and no producer chunk), never publishes a
/// catalogue, and never touches the control plane: it takes the key it is
/// given and carries pixels on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WireProfile {
    /// parallax↔parallax links: full metadata, parallax's own conventions.
    #[default]
    Parallax,
    /// The `@media` plane: one access unit per sample, codec on the
    /// `Encoding`, CBOR `FrameMeta` on the attachment.
    Media,
}

/// The process-wide zenoh session the pipeline-string factory uses (#222).
///
/// `zenoh::open` builds a whole transport stack and starts scouting, so a
/// pipeline like `zenohsrc key=a ! … ! zenohsink key=b` must not open two.
/// Library callers keep using `with_session` to supply their own.
///
/// `tokio::sync::OnceCell` rather than `std::sync::OnceLock` because opening
/// is async and `OnceLock::get_or_init` is not.
static SHARED_SESSION: tokio::sync::OnceCell<Session> = tokio::sync::OnceCell::const_new();

/// Environment variable naming a zenoh JSON5 config file for [`shared_session`].
pub const ZENOH_CONFIG_ENV: &str = "PARALLAX_ZENOH_CONFIG";

/// The process-wide zenoh session, opening it on first use.
///
/// Configured from the file named by `PARALLAX_ZENOH_CONFIG`, or zenoh's
/// defaults when that is unset.
pub async fn shared_session() -> Result<Session> {
    SHARED_SESSION
        .get_or_try_init(|| async {
            let config = match std::env::var(ZENOH_CONFIG_ENV) {
                Ok(path) => zenoh::Config::from_file(&path)
                    .map_err(|e| Error::Element(format!("{ZENOH_CONFIG_ENV}={path}: {e}")))?,
                Err(_) => zenoh::Config::default(),
            };
            zenoh::open(config)
                .await
                .map_err(|e| Error::Element(format!("Zenoh open error: {e}")))
        })
        .await
        .cloned()
}

const DEFAULT_ARENA_SLOT: usize = 64 * 1024;
/// Slots for a querier, which holds every reply to one query at once.
const QUERIER_SLOT_COUNT: usize = 64;

/// A Zenoh source that subscribes to a key expression.
///
/// Implements [`AsyncSource`]; add to a pipeline with
/// [`Pipeline::add_async_source`](crate::pipeline::Pipeline::add_async_source).
///
/// Metadata handling: samples published by a parallax [`ZenohSink`] carry a
/// wire attachment from which full [`Metadata`] is restored; gaps in the
/// restored sequence numbers set [`BufferFlags::DISCONT`]. Samples from
/// foreign publishers (no/unknown attachment) get fabricated metadata with a
/// local sequence counter (a warning is logged once). The concrete key
/// expression a sample arrived on is stored under the
/// [`KEY_EXPR_META`](super::zenoh_wire::KEY_EXPR_META) metadata key when it
/// differs from the subscription key expression.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::ZenohSrc;
///
/// let src = ZenohSrc::new("demo/example/**").await?;
/// let node = pipeline.add_async_source("zenoh-in", src);
/// ```
pub struct ZenohSrc {
    name: String,
    key_expr: String,
    /// Held so the session outlives the subscriber declared on it. Dropping
    /// the last `Session` clone closes the session, which silently kills the
    /// subscription and makes `produce` report EOS on its first poll — a
    /// failure mode invisible whenever the caller happens to keep a clone of
    /// its own alive, as a process-wide test session does.
    _session: Session,
    // Held to keep the subscription alive.
    _subscriber: zenoh::pubsub::Subscriber<()>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<zenoh::sample::Sample>,
    bytes_received: u64,
    samples_received: u64,
    /// Local counter for fabricated (legacy/foreign) metadata.
    sequence: u64,
    /// Last sequence number seen on the wire (for DISCONT detection).
    last_wire_sequence: Option<u64>,
    warned_foreign: bool,
    timeout: Option<Duration>,
    output: OutputArena,
    profile: WireProfile,
    /// `Media` only: what the key expression says the payload is, resolved at
    /// construction so `output_media_caps()` can answer before `prepare()`.
    shape: super::media_key::MediaKeyShape,
    /// `Media` only: drop samples until the first keyframe. A decoder fed a
    /// mid-GOP access unit produces nothing useful and complains for a whole
    /// GOP about it.
    waiting_for_keyframe: bool,
    /// `Media` only: whether the keyframe gate re-arms after a wire gap.
    resync_on_discont: bool,
    /// Samples dropped because the gate had not opened yet.
    dropped_pre_keyframe: u64,
}

impl ZenohSrc {
    /// Create a new Zenoh source with a new session.
    pub async fn new(key_expr: impl Into<String>) -> Result<Self> {
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;

        Self::with_session(session, key_expr).await
    }

    /// Create a new Zenoh source using an existing session.
    ///
    /// zenoh's `Session` is cheaply cloneable (`Arc`-backed) — clone it to
    /// share one session across multiple elements.
    pub async fn with_session(session: Session, key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        let name = format!("zenohsrc-{}", &key_expr[..key_expr.len().min(30)]);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        let subscriber = session
            .declare_subscriber(&key_expr)
            .callback(move |sample| {
                let _ = tx.send(sample);
            })
            .await
            .map_err(|e| Error::Element(format!("Zenoh subscribe error: {}", e)))?;

        Ok(Self {
            name,
            key_expr,
            _session: session,
            _subscriber: subscriber,
            receiver: rx,
            bytes_received: 0,
            samples_received: 0,
            sequence: 0,
            last_wire_sequence: None,
            warned_foreign: false,
            timeout: None,
            output: OutputArena::new(defaults::SOURCE_SLOT_COUNT).grow_to_fit(),
            profile: WireProfile::Parallax,
            shape: super::media_key::MediaKeyShape::Opaque,
            waiting_for_keyframe: false,
            resync_on_discont: false,
            dropped_pre_keyframe: 0,
        })
    }

    /// Subscribe to an `@media` tier key.
    ///
    /// The key must be concrete: the plane revoked the tier wildcard, because
    /// `…/video/h264/*` delivers every tier of a stream interleaved on one
    /// subscriber, unseparable except by re-parsing the key per sample.
    /// Exact-tier subscription *is* the quality choice.
    ///
    /// Because the codec is a literal chunk of the key, `output_media_caps()`
    /// is honest before a single sample arrives — which is what lets a graph
    /// under `ConverterPolicy::Deny` prepare deterministically.
    pub async fn media(key_expr: impl Into<String>) -> Result<Self> {
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;
        Self::media_with_session(session, key_expr).await
    }

    /// Subscribe to an `@media` tier key using an existing session.
    pub async fn media_with_session(session: Session, key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        if super::media_key::has_wildcard(&key_expr) {
            return Err(Error::Element(format!(
                "@media subscriptions must name one exact tier, got a wildcard: {key_expr}"
            )));
        }

        let shape = super::media_key::MediaKeyShape::parse(&key_expr);
        let mut src = Self::with_session(session, key_expr).await?;
        src.profile = WireProfile::Media;
        src.waiting_for_keyframe = true;
        src.shape = shape;
        Ok(src)
    }

    /// Whether to re-arm the keyframe gate after a wire discontinuity.
    ///
    /// Default `false`. Re-arming throws away a whole GOP for one lost
    /// packet, and the decoders already degrade gracefully with a rate-limited
    /// warning — opt in when you would rather freeze than smear.
    pub fn with_resync_on_discont(mut self, resync: bool) -> Self {
        self.resync_on_discont = resync;
        self
    }

    /// Which wire contract this source speaks.
    pub fn profile(&self) -> WireProfile {
        self.profile
    }

    /// Samples dropped because the keyframe gate had not opened yet.
    pub fn dropped_pre_keyframe(&self) -> u64 {
        self.dropped_pre_keyframe
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set a receive timeout. On timeout an empty buffer flagged
    /// [`BufferFlags::TIMEOUT`] is produced instead of waiting forever.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Get the key expression.
    pub fn key_expr(&self) -> &str {
        &self.key_expr
    }

    /// Get the number of bytes received.
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received
    }

    /// Get the number of samples received.
    pub fn samples_received(&self) -> u64 {
        self.samples_received
    }

    /// Get statistics.
    pub fn stats(&self) -> ZenohStats {
        ZenohStats {
            bytes_transferred: self.bytes_received,
            samples: self.samples_received,
        }
    }

    /// Acquire a slot for `len` bytes, or `None` if every slot is taken.
    ///
    /// Rounding up to a power of two is what stops growth from reallocating on
    /// every slightly-larger sample.
    fn acquire(&mut self, len: usize) -> Result<Option<crate::memory::SharedSlotRef>> {
        self.output
            .set_min_slot_size(len.next_power_of_two().max(DEFAULT_ARENA_SLOT));
        self.output.try_acquire(len, "zenohsrc")
    }

    /// Build a buffer from a received sample.
    /// `None` means no output slot was free. The sample is already off the
    /// wire, so it is lost — the same as any other zenoh congestion drop, and
    /// the receiver's DISCONT tracking already reports it.
    fn sample_to_buffer(&mut self, sample: zenoh::sample::Sample) -> Result<Option<Buffer>> {
        let data = sample.payload().to_bytes().into_owned();
        self.bytes_received += data.len() as u64;
        self.samples_received += 1;

        if self.profile == WireProfile::Media {
            return self.media_sample_to_buffer(sample, data);
        }

        let mut metadata = match sample.attachment().and_then(|a| {
            let bytes = a.to_bytes();
            WireMetadata::decode(&bytes)
        }) {
            Some(wire) => {
                let mut metadata = wire.to_metadata();
                // A gap in the published sequence means samples were lost on
                // the wire (congestion drop, late join).
                if let Some(prev) = self.last_wire_sequence
                    && wire.sequence != prev.wrapping_add(1)
                {
                    metadata.flags |= BufferFlags::DISCONT;
                }
                self.last_wire_sequence = Some(wire.sequence);
                metadata
            }
            None => {
                if !self.warned_foreign {
                    self.warned_foreign = true;
                    tracing::warn!(
                        key_expr = %self.key_expr,
                        "zenoh sample without parallax attachment; fabricating metadata \
                         (foreign or pre-v2 publisher)"
                    );
                }
                let metadata = Metadata::from_sequence(self.sequence);
                self.sequence += 1;
                metadata
            }
        };

        let key = sample.key_expr().as_str();
        if key != self.key_expr {
            metadata.set(KEY_EXPR_META, key.to_string());
        }

        let Some(mut slot) = self.acquire(data.len())? else {
            return Ok(None);
        };
        slot.data_mut()[..data.len()].copy_from_slice(&data);

        Ok(Some(Buffer::new(
            MemoryHandle::with_len(slot, data.len()),
            metadata,
        )))
    }

    /// An empty TIMEOUT-flagged buffer (produced when `with_timeout` fires).
    /// Build a buffer from an `@media` sample (#223).
    ///
    /// The plane's contract, read back: the payload is one whole access unit,
    /// the codec is on the `Encoding` (never sniffed), and per-frame metadata
    /// is a CBOR `FrameMeta` on the attachment.
    fn media_sample_to_buffer(
        &mut self,
        sample: zenoh::sample::Sample,
        data: Vec<u8>,
    ) -> Result<Option<Buffer>> {
        let frame = sample
            .attachment()
            .and_then(|a| crate::wire::FrameMeta::from_cbor(&a.to_bytes()));

        let Some(frame) = frame else {
            // A publisher on this plane that sends no FrameMeta is not
            // conformant; say so once and pass the bytes on rather than
            // dropping a stream that might still decode.
            if !self.warned_foreign {
                self.warned_foreign = true;
                tracing::warn!(
                    "zenohsrc '{}': @media sample without a FrameMeta attachment; \
                     delivering with synthesised metadata",
                    self.name
                );
            }
            let metadata = Metadata::from_sequence(self.sequence);
            self.sequence += 1;
            return self.buffer_from(data, metadata);
        };

        let mut metadata = frame.to_metadata();

        // A gap means the publisher's samples did not all arrive. So does the
        // very first sample of a subscription: joining a live stream partway
        // *is* a discontinuity, and it never used to be flagged because the
        // "previous sequence" started empty.
        let discont = match self.last_wire_sequence {
            Some(prev) => frame.sequence != prev.wrapping_add(1),
            None => true,
        };
        if discont {
            metadata.flags |= BufferFlags::DISCONT;
            if self.resync_on_discont {
                self.waiting_for_keyframe = true;
            }
        }
        self.last_wire_sequence = Some(frame.sequence);

        // Never hand a decoder a mid-GOP access unit.
        if self.waiting_for_keyframe {
            if !frame.keyframe {
                self.dropped_pre_keyframe += 1;
                if self.dropped_pre_keyframe == 1 {
                    tracing::debug!(
                        "zenohsrc '{}': waiting for a keyframe before delivering",
                        self.name
                    );
                }
                return Ok(None);
            }
            self.waiting_for_keyframe = false;
        }

        // Coded geometry, when the publisher knew it.
        if let (Some(codec), true) = (self.shape.codec(), frame.width > 0 && frame.height > 0) {
            metadata.set_coded_video_dims(codec, frame.width, frame.height);
        } else if let Some(codec) = self.shape.codec() {
            metadata.format = Some(crate::format::MediaFormat::Video(codec));
        }

        self.buffer_from(data, metadata)
    }

    /// Copy `data` into an arena slot and pair it with `metadata`.
    fn buffer_from(&mut self, data: Vec<u8>, metadata: Metadata) -> Result<Option<Buffer>> {
        let Some(mut slot) = self.acquire(data.len())? else {
            return Ok(None);
        };
        slot.data_mut()[..data.len()].copy_from_slice(&data);
        Ok(Some(Buffer::new(
            MemoryHandle::with_len(slot, data.len()),
            metadata,
        )))
    }

    fn timeout_buffer(&mut self) -> Result<Option<Buffer>> {
        let Some(slot) = self.acquire(0)? else {
            return Ok(None);
        };
        let mut metadata = Metadata::from_sequence(self.sequence);
        self.sequence += 1;
        metadata.flags |= BufferFlags::TIMEOUT;
        Ok(Some(Buffer::new(MemoryHandle::with_len(slot, 0), metadata)))
    }
}

impl AsyncSource for ZenohSrc {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    async fn produce(&mut self, _ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        let sample = if let Some(timeout) = self.timeout {
            match tokio::time::timeout(timeout, self.receiver.recv()).await {
                Ok(Some(sample)) => sample,
                Ok(None) => return Ok(ProduceResult::Eos),
                Err(_) => {
                    return Ok(match self.timeout_buffer()? {
                        Some(b) => ProduceResult::OwnBuffer(b),
                        None => ProduceResult::WouldBlock,
                    });
                }
            }
        } else {
            match self.receiver.recv().await {
                Some(sample) => sample,
                None => return Ok(ProduceResult::Eos),
            }
        };

        Ok(match self.sample_to_buffer(sample)? {
            Some(b) => ProduceResult::OwnBuffer(b),
            None => ProduceResult::WouldBlock,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Caps derived from the key expression, not from received data (#223).
    ///
    /// On `@media` the codec is a literal chunk of the key, so this is honest
    /// at `prepare()` time — before a single sample has arrived. That matters:
    /// `prepare()` is synchronous and runs first, so caps derived from the
    /// wire would be `Video(H264)` on one run and `Any` on the next depending
    /// on network timing, and the same pipeline string would pass under
    /// `ConverterPolicy::Deny` once and fail the next time.
    ///
    /// A key naming no codec parallax knows reports `Bytes` rather than `Any`:
    /// `Bytes` intersects with anything, so `Deny` still never fails on it,
    /// and it says "opaque bytes", which is true, where `Any` says "anything",
    /// which is not.
    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};

        if self.profile != WireProfile::Media {
            return ElementMediaCaps::from(crate::format::Caps::any());
        }

        let format = match self.shape.codec() {
            Some(codec) => FormatCaps::Video(codec),
            None => FormatCaps::Bytes,
        };
        ElementMediaCaps::new(vec![FormatMemoryCap::new(format, MemoryCaps::cpu_only())])
    }
}

/// A Zenoh sink that publishes to a key expression.
///
/// Implements [`AsyncSink`]; add to a pipeline with
/// [`Pipeline::add_async_sink`](crate::pipeline::Pipeline::add_async_sink).
///
/// Each buffer is published as one zenoh sample: payload = raw buffer bytes,
/// attachment = wire-serialized [`Metadata`] (unless
/// [`without_metadata`](Self::without_metadata)), sample encoding derived
/// from the buffer's media format (or overridden with
/// [`with_encoding`](Self::with_encoding)).
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::{ZenohSink, ZenohCongestionControl, ZenohPriority};
///
/// let sink = ZenohSink::new("demo/example/video").await?
///     .with_congestion_control(ZenohCongestionControl::Drop)
///     .with_priority(ZenohPriority::InteractiveHigh)
///     .with_express(true);
/// let node = pipeline.add_async_sink("zenoh-out", sink);
/// ```
pub struct ZenohSink {
    name: String,
    key_expr: String,
    session: Session,
    publisher: Option<zenoh::pubsub::Publisher<'static>>,
    congestion_control: ZenohCongestionControl,
    priority: ZenohPriority,
    express: bool,
    #[cfg(feature = "zenoh-unstable")]
    reliability: Option<ZenohReliability>,
    encoding: Option<zenoh::bytes::Encoding>,
    forward_custom_keys: Vec<&'static str>,
    attach_metadata: bool,
    profile: WireProfile,
    /// `Media` only: the publisher owns the wire sequence. `metadata.sequence`
    /// is per-element and gets rewritten mid-graph (an encoder sets it to its
    /// own frame count), so only the publisher knows what actually went out —
    /// which is what makes a subscriber's gap detection mean anything.
    wire_sequence: u64,
    /// `Media` only: keeps every published keyframe a self-contained decoder
    /// entry point.
    param_sets: Option<crate::codec::annexb::ParamSetCache>,
    /// A keyframe request waiting to be handed to the executor's upstream
    /// path, and the monotonic counter that stamps it.
    pending_keyframe: Option<crate::event::ForceKeyUnitEvent>,
    keyframe_requests: u64,
    bytes_sent: u64,
    samples_sent: u64,
}

impl ZenohSink {
    /// Create a new Zenoh sink with a new session.
    pub async fn new(key_expr: impl Into<String>) -> Result<Self> {
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;

        Self::with_session(session, key_expr).await
    }

    /// Create a new Zenoh sink using an existing session.
    ///
    /// zenoh's `Session` is cheaply cloneable (`Arc`-backed) — clone it to
    /// share one session across multiple elements.
    pub async fn with_session(session: Session, key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        let name = format!("zenohsink-{}", &key_expr[..key_expr.len().min(30)]);

        // The publisher is created lazily so QoS builder methods can run first.
        Ok(Self {
            name,
            key_expr,
            session,
            publisher: None,
            congestion_control: ZenohCongestionControl::Block,
            priority: ZenohPriority::Data,
            express: false,
            #[cfg(feature = "zenoh-unstable")]
            reliability: None,
            encoding: None,
            forward_custom_keys: Vec::new(),
            attach_metadata: true,
            profile: WireProfile::Parallax,
            wire_sequence: 0,
            param_sets: None,
            pending_keyframe: None,
            keyframe_requests: 0,
            bytes_sent: 0,
            samples_sent: 0,
        })
    }

    /// Publish on the `@media` plane.
    ///
    /// Applies the plane's `frame` QoS profile — best-effort, drop,
    /// interactive-high, express — because a stale frame is worthless and the
    /// encoder must never block. A plain declared publisher, never an
    /// AdvancedPublisher: recovering a superseded frame is anti-useful.
    ///
    /// The key must be a concrete tier key. Wildcards are refused rather than
    /// warned about: publishing under one is a lie about identity, and
    /// subscribing under one would interleave every tier of a stream on a
    /// single subscriber.
    pub async fn media(key_expr: impl Into<String>) -> Result<Self> {
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;
        Self::media_with_session(session, key_expr).await
    }

    /// Publish on the `@media` plane using an existing session.
    pub async fn media_with_session(session: Session, key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        if super::media_key::has_wildcard(&key_expr) {
            return Err(Error::Element(format!(
                "@media keys must be concrete, got a wildcard: {key_expr}"
            )));
        }

        let codec = super::media_key::MediaKeyShape::parse(&key_expr).codec();
        let mut sink = Self::with_session(session, key_expr).await?;
        sink.profile = WireProfile::Media;
        // The plane's `frame` QoS profile.
        sink.congestion_control = ZenohCongestionControl::Drop;
        sink.priority = ZenohPriority::InteractiveHigh;
        sink.express = true;
        #[cfg(feature = "zenoh-unstable")]
        {
            sink.reliability = Some(ZenohReliability::BestEffort);
        }
        sink.param_sets = match codec {
            Some(crate::format::VideoCodec::H264) => Some(
                crate::codec::annexb::ParamSetCache::new(crate::codec::annexb::NalCodec::H264),
            ),
            Some(crate::format::VideoCodec::H265) => Some(
                crate::codec::annexb::ParamSetCache::new(crate::codec::annexb::NalCodec::H265),
            ),
            // Every JPEG is self-contained; other codecs carry their headers
            // in band and have no Annex-B parameter-set concept.
            _ => None,
        };
        Ok(sink)
    }

    /// Which wire contract this sink speaks.
    pub fn profile(&self) -> WireProfile {
        self.profile
    }

    /// Ask upstream for a decoder entry point (#213, #234).
    ///
    /// Queues an [`Event::ForceKeyUnit`](crate::event::Event::ForceKeyUnit)
    /// for the executor to carry toward the sources, where the first encoder
    /// consumes it. In a passthrough graph — an RTSP camera republished
    /// verbatim — nothing can satisfy it, and that is the honest outcome:
    /// RTSP has no standard way to ask a camera for an IDR.
    ///
    /// Repeated calls before the executor collects coalesce into one request.
    /// A viewer under packet loss will ask far faster than a GOP.
    pub fn request_keyframe(&mut self) {
        if self.pending_keyframe.is_none() {
            self.keyframe_requests += 1;
            self.pending_keyframe = Some(crate::event::ForceKeyUnitEvent::new(
                self.keyframe_requests as u32,
            ));
        }
    }

    /// How many keyframe requests this sink has raised.
    pub fn keyframe_requests(&self) -> u64 {
        self.keyframe_requests
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the congestion control mode.
    pub fn with_congestion_control(mut self, cc: ZenohCongestionControl) -> Self {
        self.congestion_control = cc;
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: ZenohPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Enable/disable express mode (send immediately, don't batch).
    ///
    /// Express trades bandwidth for latency; leave it off on constrained
    /// links where batching wins.
    pub fn with_express(mut self, express: bool) -> Self {
        self.express = express;
        self
    }

    /// Set the reliability mode (requires the `zenoh-unstable` feature).
    #[cfg(feature = "zenoh-unstable")]
    pub fn with_reliability(mut self, reliability: ZenohReliability) -> Self {
        self.reliability = Some(reliability);
        self
    }

    /// Override the sample encoding. By default it is derived from each
    /// buffer's media format (e.g. `video/h264`), falling back to
    /// `zenoh/bytes`.
    pub fn with_encoding(mut self, encoding: impl Into<zenoh::bytes::Encoding>) -> Self {
        self.encoding = Some(encoding.into());
        self
    }

    /// Select byte-valued custom metadata entries (see
    /// [`Metadata::set_bytes`]) to forward on the wire, e.g. `"stanag/klv"`.
    pub fn with_forward_custom_keys(mut self, keys: &[&'static str]) -> Result<Self> {
        if self.profile == WireProfile::Media {
            return Err(Error::Element(
                "@media carries pixels and nothing else; side-band data belongs                  on the telemetry plane, not in the media attachment"
                    .to_string(),
            ));
        }
        self.forward_custom_keys = keys.to_vec();
        Ok(self)
    }

    /// Publish raw payloads only, without the metadata attachment
    /// (interop mode for consumers that must not see attachments).
    /// PTS and all other metadata are lost on the wire in this mode.
    pub fn without_metadata(mut self) -> Result<Self> {
        if self.profile == WireProfile::Media {
            return Err(Error::Element(
                "@media samples must carry a FrameMeta attachment;                  without_metadata() is not meaningful on that plane"
                    .to_string(),
            ));
        }
        self.attach_metadata = false;
        Ok(self)
    }

    /// Get the key expression.
    pub fn key_expr(&self) -> &str {
        &self.key_expr
    }

    /// Get the number of bytes sent.
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    }

    /// Get the number of samples sent.
    pub fn samples_sent(&self) -> u64 {
        self.samples_sent
    }

    /// Get statistics.
    pub fn stats(&self) -> ZenohStats {
        ZenohStats {
            bytes_transferred: self.bytes_sent,
            samples: self.samples_sent,
        }
    }

    /// Get a builder for a matching listener that fires when this publisher
    /// gains or loses matching subscribers (requires `zenoh-unstable`).
    /// Call after the first published buffer (the publisher is declared
    /// lazily) or after [`ensure_publisher`](Self::ensure_publisher).
    #[cfg(feature = "zenoh-unstable")]
    pub fn matching_listener(
        &self,
    ) -> Option<zenoh::matching::MatchingListenerBuilder<'_, zenoh::handlers::DefaultHandler>> {
        self.publisher.as_ref().map(|p| p.matching_listener())
    }

    /// Declare the publisher now (it is otherwise declared lazily on the
    /// first consumed buffer).
    pub async fn ensure_publisher(&mut self) -> Result<()> {
        if self.publisher.is_some() {
            return Ok(());
        }

        // An owned KeyExpr gives the publisher a 'static lifetime.
        let key_expr = KeyExpr::try_from(self.key_expr.clone())
            .map_err(|e| Error::Element(format!("Invalid key expression: {}", e)))?;

        let builder = self
            .session
            .declare_publisher(key_expr)
            .congestion_control(self.congestion_control.into())
            .priority(self.priority.into())
            .express(self.express);

        #[cfg(feature = "zenoh-unstable")]
        let builder = match self.reliability {
            Some(reliability) => builder.reliability(reliability.into()),
            None => builder,
        };

        let publisher = builder
            .await
            .map_err(|e| Error::Element(format!("Zenoh publisher error: {}", e)))?;

        self.publisher = Some(publisher);
        Ok(())
    }
}

impl AsyncSink for ZenohSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        self.ensure_publisher().await?;
        let publisher = self.publisher.as_ref().expect("publisher just ensured");

        let metadata = ctx.metadata();

        let encoding = self
            .encoding
            .clone()
            .unwrap_or_else(|| encoding_for_format(metadata.format.as_ref()));

        let (data, attachment) = match self.profile {
            WireProfile::Parallax => {
                let attachment = self.attach_metadata.then(|| {
                    WireMetadata::from_metadata(metadata, &self.forward_custom_keys).encode()
                });
                (ctx.input().to_vec(), attachment)
            }
            WireProfile::Media => {
                // Keep the keyframe promise at the byte level: repair a
                // keyframe that arrived without its parameter sets, then judge
                // the flag on the bytes we are actually about to send.
                let prepared = match &mut self.param_sets {
                    Some(cache) => cache.prepare(ctx.input()).into_owned(),
                    None => ctx.input().to_vec(),
                };

                let keyframe = match self.param_sets.as_ref().map(|c| c.codec()) {
                    Some(codec) => crate::codec::annexb::is_entry_point(&prepared, codec),
                    // No Annex-B notion of an entry point (JPEG, VP8/9, AV1):
                    // trust what the pipeline flagged.
                    None => metadata.flags.is_keyframe(),
                };

                let (width, height) = metadata.coded_video_dims().unwrap_or((0, 0));
                let mut frame = crate::wire::FrameMeta::from_metadata(metadata, width, height);
                frame.keyframe = keyframe;
                frame.sequence = self.wire_sequence;
                self.wire_sequence = self.wire_sequence.wrapping_add(1);

                (prepared, Some(frame.to_cbor()))
            }
        };

        let len = data.len();
        let put = publisher.put(data).encoding(encoding);
        let put = match attachment {
            Some(bytes) => put.attachment(bytes),
            None => put,
        };

        put.await
            .map_err(|e| Error::Element(format!("Zenoh put error: {}", e)))?;

        self.bytes_sent += len as u64;
        self.samples_sent += 1;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Hand a queued keyframe request to the executor's upstream path.
    ///
    /// Polled after every `consume`, so a request raised between samples goes
    /// out on the next one.
    fn take_upstream_event(&mut self) -> Option<crate::event::Event> {
        self.pending_keyframe
            .take()
            .map(crate::event::Event::ForceKeyUnit)
    }
}

/// A Zenoh queryable that responds to queries.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::ZenohQueryable;
///
/// let mut queryable = ZenohQueryable::new("demo/example/query").await?;
/// while let Some(query) = queryable.recv_query()? {
///     query.reply(b"data").await?;
/// }
/// ```
pub struct ZenohQueryable {
    name: String,
    key_expr: String,
    // Held to keep the queryable alive.
    _queryable: zenoh::query::Queryable<()>,
    receiver: tokio::sync::mpsc::UnboundedReceiver<zenoh::query::Query>,
    queries_received: u64,
}

impl ZenohQueryable {
    /// Create a new Zenoh queryable with a new session.
    pub async fn new(key_expr: impl Into<String>) -> Result<Self> {
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;

        Self::with_session(session, key_expr).await
    }

    /// Create a new Zenoh queryable using an existing session.
    pub async fn with_session(session: Session, key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        let name = format!("zenoh-queryable-{}", &key_expr[..key_expr.len().min(30)]);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        let queryable = session
            .declare_queryable(&key_expr)
            .callback(move |query| {
                let _ = tx.send(query);
            })
            .await
            .map_err(|e| Error::Element(format!("Zenoh queryable error: {}", e)))?;

        Ok(Self {
            name,
            key_expr,
            _queryable: queryable,
            receiver: rx,
            queries_received: 0,
        })
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the key expression.
    pub fn key_expr(&self) -> &str {
        &self.key_expr
    }

    /// Get the number of queries received.
    pub fn queries_received(&self) -> u64 {
        self.queries_received
    }

    /// Receive the next query (blocking).
    pub fn recv_query(&mut self) -> Result<Option<ZenohQuery>> {
        match self.receiver.blocking_recv() {
            Some(query) => {
                self.queries_received += 1;
                Ok(Some(ZenohQuery { inner: query }))
            }
            None => Ok(None),
        }
    }

    /// Receive the next query asynchronously.
    pub async fn recv_query_async(&mut self) -> Result<Option<ZenohQuery>> {
        match self.receiver.recv().await {
            Some(query) => {
                self.queries_received += 1;
                Ok(Some(ZenohQuery { inner: query }))
            }
            None => Ok(None),
        }
    }

    /// Try to receive a query without blocking.
    pub fn try_recv_query(&mut self) -> Option<ZenohQuery> {
        match self.receiver.try_recv() {
            Ok(query) => {
                self.queries_received += 1;
                Some(ZenohQuery { inner: query })
            }
            Err(_) => None,
        }
    }

    /// Get the element name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A wrapped Zenoh query.
pub struct ZenohQuery {
    inner: zenoh::query::Query,
}

impl ZenohQuery {
    /// Get the key expression of the query.
    pub fn key_expr(&self) -> &str {
        self.inner.key_expr().as_str()
    }

    /// Get the query parameters.
    pub fn parameters(&self) -> &str {
        self.inner.parameters().as_str()
    }

    /// Get the query payload, if any.
    pub fn payload(&self) -> Option<Vec<u8>> {
        self.inner.payload().map(|p| p.to_bytes().into_owned())
    }

    /// Reply to the query with data.
    pub async fn reply(self, data: &[u8]) -> Result<()> {
        self.inner
            .reply(self.inner.key_expr().clone(), data)
            .await
            .map_err(|e| Error::Element(format!("Zenoh reply error: {}", e)))
    }

    /// Reply to the query with data and buffer metadata (serialized into the
    /// reply attachment, restored by [`ZenohQuerier`] on the other side).
    pub async fn reply_with_metadata(self, data: &[u8], metadata: &Metadata) -> Result<()> {
        let wire = WireMetadata::from_metadata(metadata, &[]);
        self.inner
            .reply(self.inner.key_expr().clone(), data)
            .attachment(wire.encode())
            .await
            .map_err(|e| Error::Element(format!("Zenoh reply error: {}", e)))
    }

    /// Reply with an error.
    pub async fn reply_err(self, error: &[u8]) -> Result<()> {
        self.inner
            .reply_err(error)
            .await
            .map_err(|e| Error::Element(format!("Zenoh reply error: {}", e)))
    }
}

/// A Zenoh querier that sends queries and receives replies.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::ZenohQuerier;
///
/// let mut querier = ZenohQuerier::new().await?;
/// let replies = querier.get("demo/example/**").await?;
/// ```
pub struct ZenohQuerier {
    name: String,
    session: Session,
    timeout: Duration,
    queries_sent: u64,
    replies_received: u64,
    output: OutputArena,
}

impl ZenohQuerier {
    /// Create a new Zenoh querier with a new session.
    pub async fn new() -> Result<Self> {
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;

        Ok(Self::with_session(session))
    }

    /// Create a new Zenoh querier using an existing session.
    pub fn with_session(session: Session) -> Self {
        Self {
            name: "zenoh-querier".to_string(),
            session,
            timeout: Duration::from_secs(10),
            queries_sent: 0,
            replies_received: 0,
            // 64, not one of the `defaults` floors: a query holds every reply
            // simultaneously, and nothing sizes this from the graph because a
            // querier is not an element.
            output: OutputArena::new(QUERIER_SLOT_COUNT).grow_to_fit(),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the query timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Get the number of queries sent.
    pub fn queries_sent(&self) -> u64 {
        self.queries_sent
    }

    /// Get the number of replies received.
    pub fn replies_received(&self) -> u64 {
        self.replies_received
    }

    /// Get the element name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Send a query and collect all replies.
    ///
    /// Replies published with a parallax metadata attachment (see
    /// [`ZenohQuery::reply_with_metadata`]) get their metadata restored;
    /// other replies get sequence-only metadata.
    pub async fn get(&mut self, key_expr: &str) -> Result<Vec<Buffer>> {
        self.get_inner(key_expr, None).await
    }

    /// Send a query with a payload value and collect all replies.
    pub async fn get_with_value(&mut self, key_expr: &str, value: &[u8]) -> Result<Vec<Buffer>> {
        self.get_inner(key_expr, Some(value)).await
    }

    async fn get_inner(&mut self, key_expr: &str, value: Option<&[u8]>) -> Result<Vec<Buffer>> {
        let builder = self.session.get(key_expr).timeout(self.timeout);
        let replies = match value {
            Some(value) => builder.payload(value).await,
            None => builder.await,
        }
        .map_err(|e| Error::Element(format!("Zenoh get error: {}", e)))?;

        self.queries_sent += 1;

        let mut buffers = Vec::new();
        let mut seq = 0u64;

        while let Ok(reply) = replies.recv_async().await {
            let Ok(sample) = reply.result() else {
                continue; // Query error reply, skip
            };
            let data = sample.payload().to_bytes().into_owned();

            let metadata = sample
                .attachment()
                .and_then(|a| WireMetadata::decode(&a.to_bytes()))
                .map(|wire| wire.to_metadata())
                .unwrap_or_else(|| Metadata::from_sequence(seq));
            seq += 1;

            // `ZenohQuerier` is not an element, so no budget ever reaches it —
            // the floor is all it gets, and `acquire`'s `PoolExhausted` reaches
            // application code that can handle it.
            self.output
                .set_min_slot_size(data.len().next_power_of_two().max(DEFAULT_ARENA_SLOT));
            let mut slot = self.output.acquire(data.len(), "zenohquerier")?;
            slot.data_mut()[..data.len()].copy_from_slice(&data);

            buffers.push(Buffer::new(
                MemoryHandle::with_len(slot, data.len()),
                metadata,
            ));
            self.replies_received += 1;
        }

        Ok(buffers)
    }
}

/// Statistics for Zenoh elements.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZenohStats {
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Total samples transferred.
    pub samples: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zenoh_priority_conversion() {
        let p: zenoh::qos::Priority = ZenohPriority::RealTime.into();
        assert_eq!(p, zenoh::qos::Priority::RealTime);

        let p: zenoh::qos::Priority = ZenohPriority::Data.into();
        assert_eq!(p, zenoh::qos::Priority::Data);
    }

    #[test]
    fn test_zenoh_stats_default() {
        let stats = ZenohStats::default();
        assert_eq!(stats.bytes_transferred, 0);
        assert_eq!(stats.samples, 0);
    }

    #[test]
    fn test_zenoh_congestion_control_default() {
        let cc = ZenohCongestionControl::default();
        assert_eq!(cc, ZenohCongestionControl::Block);
    }
}
