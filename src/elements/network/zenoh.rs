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
use crate::memory::SharedArena;
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
const DEFAULT_ARENA_SLOT: usize = 64 * 1024;
const ARENA_SLOTS: usize = 32;

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
    arena: Option<SharedArena>,
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
            _subscriber: subscriber,
            receiver: rx,
            bytes_received: 0,
            samples_received: 0,
            sequence: 0,
            last_wire_sequence: None,
            warned_foreign: false,
            timeout: None,
            arena: None,
        })
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

    /// Ensure the arena exists and its slots fit `len` bytes.
    fn ensure_arena(&mut self, len: usize) -> Result<&SharedArena> {
        let needs_new = match &self.arena {
            Some(arena) => arena.slot_size() < len,
            None => true,
        };
        if needs_new {
            let slot_size = len.next_power_of_two().max(DEFAULT_ARENA_SLOT);
            self.arena = Some(
                SharedArena::new(slot_size, ARENA_SLOTS)
                    .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
            );
        }
        Ok(self.arena.as_ref().unwrap())
    }

    /// Build a buffer from a received sample.
    fn sample_to_buffer(&mut self, sample: zenoh::sample::Sample) -> Result<Buffer> {
        let data = sample.payload().to_bytes().into_owned();
        self.bytes_received += data.len() as u64;
        self.samples_received += 1;

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

        let arena = self.ensure_arena(data.len())?;
        arena.reclaim();
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
        slot.data_mut()[..data.len()].copy_from_slice(&data);

        Ok(Buffer::new(
            MemoryHandle::with_len(slot, data.len()),
            metadata,
        ))
    }

    /// An empty TIMEOUT-flagged buffer (produced when `with_timeout` fires).
    fn timeout_buffer(&mut self) -> Result<Buffer> {
        let arena = self.ensure_arena(0)?;
        arena.reclaim();
        let slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
        let mut metadata = Metadata::from_sequence(self.sequence);
        self.sequence += 1;
        metadata.flags |= BufferFlags::TIMEOUT;
        Ok(Buffer::new(MemoryHandle::with_len(slot, 0), metadata))
    }
}

impl AsyncSource for ZenohSrc {
    async fn produce(&mut self, _ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        let sample = if let Some(timeout) = self.timeout {
            match tokio::time::timeout(timeout, self.receiver.recv()).await {
                Ok(Some(sample)) => sample,
                Ok(None) => return Ok(ProduceResult::Eos),
                Err(_) => return Ok(ProduceResult::OwnBuffer(self.timeout_buffer()?)),
            }
        } else {
            match self.receiver.recv().await {
                Some(sample) => sample,
                None => return Ok(ProduceResult::Eos),
            }
        };

        Ok(ProduceResult::OwnBuffer(self.sample_to_buffer(sample)?))
    }

    fn name(&self) -> &str {
        &self.name
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
            bytes_sent: 0,
            samples_sent: 0,
        })
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
    pub fn with_forward_custom_keys(mut self, keys: &[&'static str]) -> Self {
        self.forward_custom_keys = keys.to_vec();
        self
    }

    /// Publish raw payloads only, without the metadata attachment
    /// (interop mode for consumers that must not see attachments).
    /// PTS and all other metadata are lost on the wire in this mode.
    pub fn without_metadata(mut self) -> Self {
        self.attach_metadata = false;
        self
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
        let data = ctx.input().to_vec();
        let len = data.len();

        let encoding = self
            .encoding
            .clone()
            .unwrap_or_else(|| encoding_for_format(metadata.format.as_ref()));

        let put = publisher.put(data).encoding(encoding);
        let put = if self.attach_metadata {
            let wire = WireMetadata::from_metadata(metadata, &self.forward_custom_keys);
            put.attachment(wire.encode())
        } else {
            put
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
    arena: Option<SharedArena>,
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
            arena: None,
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

            // Grow the arena if a reply doesn't fit.
            let needs_new = match &self.arena {
                Some(arena) => arena.slot_size() < data.len(),
                None => true,
            };
            if needs_new {
                let slot_size = data.len().next_power_of_two().max(DEFAULT_ARENA_SLOT);
                self.arena = Some(
                    SharedArena::new(slot_size, 64)
                        .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
                );
            }
            let arena = self.arena.as_ref().unwrap();
            arena.reclaim();
            let mut slot = arena
                .acquire()
                .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
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
