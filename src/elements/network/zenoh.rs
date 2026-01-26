//! Zenoh pub/sub and query elements.
//!
//! Provides Zenoh-based distributed communication for pipelines.
//!
//! - [`ZenohSrc`]: Subscribe to a Zenoh key expression
//! - [`ZenohSink`]: Publish to a Zenoh key expression
//! - [`ZenohQueryable`]: Respond to Zenoh queries
//! - [`ZenohQuerier`]: Query Zenoh resources
//!
//! Requires the `zenoh` feature flag.

#![cfg(feature = "zenoh")]

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Sink, Source};
use crate::error::{Error, Result};
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::sync::Arc;
use std::time::Duration;
use zenoh::Session;
use zenoh::prelude::*;

/// Congestion control mode for Zenoh publishing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZenohCongestionControl {
    /// Block if the network is congested.
    #[default]
    Block,
    /// Drop messages if the network is congested.
    Drop,
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

/// A Zenoh source that subscribes to a key expression.
///
/// Receives samples published to matching key expressions.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::ZenohSrc;
///
/// let mut src = ZenohSrc::new("demo/example/**").await?;
/// while let Some(buffer) = src.produce()? {
///     // Process received samples
/// }
/// ```
pub struct ZenohSrc {
    name: String,
    key_expr: String,
    session: Option<Arc<Session>>,
    subscriber: Option<zenoh::pubsub::Subscriber<()>>,
    receiver: Option<flume::Receiver<zenoh::sample::Sample>>,
    bytes_received: u64,
    samples_received: u64,
    sequence: u64,
    timeout: Option<Duration>,
}

impl ZenohSrc {
    /// Create a new Zenoh source with a new session.
    pub async fn new(key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;

        Self::with_session(Arc::new(session), key_expr).await
    }

    /// Create a new Zenoh source using an existing session.
    pub async fn with_session(session: Arc<Session>, key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        let name = format!("zenohsrc-{}", &key_expr[..key_expr.len().min(30)]);

        // Create a channel for receiving samples
        let (tx, rx) = flume::unbounded();

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
            session: Some(session),
            subscriber: Some(subscriber),
            receiver: Some(rx),
            bytes_received: 0,
            samples_received: 0,
            sequence: 0,
            timeout: None,
        })
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set a receive timeout.
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
}

impl Source for ZenohSrc {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        let receiver = self
            .receiver
            .as_ref()
            .ok_or_else(|| Error::Element("not subscribed".into()))?;

        let sample = if let Some(timeout) = self.timeout {
            match receiver.recv_timeout(timeout) {
                Ok(sample) => sample,
                Err(flume::RecvTimeoutError::Timeout) => {
                    // Return timeout buffer
                    let segment = Arc::new(HeapSegment::new(1)?);
                    let handle = MemoryHandle::from_segment_with_len(segment, 0);
                    let mut meta = Metadata::from_sequence(self.sequence);
                    meta.flags = meta.flags.insert(crate::metadata::BufferFlags::TIMEOUT);
                    self.sequence += 1;
                    return Ok(Some(Buffer::new(handle, meta)));
                }
                Err(flume::RecvTimeoutError::Disconnected) => return Ok(None),
            }
        } else {
            match receiver.recv() {
                Ok(sample) => sample,
                Err(_) => return Ok(None),
            }
        };

        let payload = sample.payload();
        let data: Vec<u8> = payload.to_bytes().to_vec();

        self.bytes_received += data.len() as u64;
        self.samples_received += 1;
        let seq = self.sequence;
        self.sequence += 1;

        let segment = Arc::new(HeapSegment::new(data.len().max(1))?);
        if !data.is_empty() {
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        }

        let handle = MemoryHandle::from_segment_with_len(segment, data.len());
        let mut meta = Metadata::from_sequence(seq);

        // Store key expression in metadata if different from subscription
        let key = sample.key_expr().as_str();
        if key != self.key_expr {
            // Could store in extra fields if needed
        }

        Ok(Some(Buffer::new(handle, meta)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A Zenoh sink that publishes to a key expression.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::ZenohSink;
///
/// let mut sink = ZenohSink::new("demo/example/data").await?;
/// sink.consume(buffer)?;
/// ```
pub struct ZenohSink {
    name: String,
    key_expr: String,
    session: Option<Arc<Session>>,
    publisher: Option<zenoh::pubsub::Publisher<'static>>,
    congestion_control: ZenohCongestionControl,
    priority: ZenohPriority,
    bytes_sent: u64,
    samples_sent: u64,
}

impl ZenohSink {
    /// Create a new Zenoh sink with a new session.
    pub async fn new(key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;

        Self::with_session(Arc::new(session), key_expr).await
    }

    /// Create a new Zenoh sink using an existing session.
    pub async fn with_session(session: Arc<Session>, key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        let name = format!("zenohsink-{}", &key_expr[..key_expr.len().min(30)]);

        // We'll create the publisher lazily to allow configuration
        Ok(Self {
            name,
            key_expr,
            session: Some(session),
            publisher: None,
            congestion_control: ZenohCongestionControl::Block,
            priority: ZenohPriority::Data,
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

    async fn ensure_publisher(&mut self) -> Result<()> {
        if self.publisher.is_some() {
            return Ok(());
        }

        let session = self
            .session
            .as_ref()
            .ok_or_else(|| Error::Element("no session".into()))?;

        let cc = match self.congestion_control {
            ZenohCongestionControl::Block => zenoh::qos::CongestionControl::Block,
            ZenohCongestionControl::Drop => zenoh::qos::CongestionControl::Drop,
        };

        let publisher = session
            .declare_publisher(&self.key_expr)
            .congestion_control(cc)
            .priority(self.priority.into())
            .await
            .map_err(|e| Error::Element(format!("Zenoh publisher error: {}", e)))?;

        // Store with static lifetime by leaking the session reference
        // In practice, the session outlives the publisher
        self.publisher = Some(unsafe { std::mem::transmute(publisher) });
        Ok(())
    }
}

impl Sink for ZenohSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        // We need to block on the async ensure_publisher
        // This is not ideal but maintains the sync Sink trait
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| Error::Element("no tokio runtime".into()))?;

        rt.block_on(self.ensure_publisher())?;

        let publisher = self
            .publisher
            .as_ref()
            .ok_or_else(|| Error::Element("no publisher".into()))?;

        let data = buffer.as_bytes();

        rt.block_on(async {
            publisher
                .put(data)
                .await
                .map_err(|e| Error::Element(format!("Zenoh put error: {}", e)))
        })?;

        self.bytes_sent += data.len() as u64;
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
/// let queryable = ZenohQueryable::new("demo/example/query").await?;
/// // Handle incoming queries
/// ```
pub struct ZenohQueryable {
    name: String,
    key_expr: String,
    session: Option<Arc<Session>>,
    queryable: Option<zenoh::query::Queryable<()>>,
    receiver: Option<flume::Receiver<zenoh::query::Query>>,
    queries_received: u64,
}

impl ZenohQueryable {
    /// Create a new Zenoh queryable with a new session.
    pub async fn new(key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;

        Self::with_session(Arc::new(session), key_expr).await
    }

    /// Create a new Zenoh queryable using an existing session.
    pub async fn with_session(session: Arc<Session>, key_expr: impl Into<String>) -> Result<Self> {
        let key_expr = key_expr.into();
        let name = format!("zenoh-queryable-{}", &key_expr[..key_expr.len().min(30)]);

        let (tx, rx) = flume::unbounded();

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
            session: Some(session),
            queryable: Some(queryable),
            receiver: Some(rx),
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
        let receiver = self
            .receiver
            .as_ref()
            .ok_or_else(|| Error::Element("not declared".into()))?;

        match receiver.recv() {
            Ok(query) => {
                self.queries_received += 1;
                Ok(Some(ZenohQuery { inner: query }))
            }
            Err(_) => Ok(None),
        }
    }

    /// Try to receive a query without blocking.
    pub fn try_recv_query(&mut self) -> Option<ZenohQuery> {
        let receiver = self.receiver.as_ref()?;

        match receiver.try_recv() {
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

    /// Reply to the query with data.
    pub async fn reply(self, data: &[u8]) -> Result<()> {
        self.inner
            .reply(self.inner.key_expr().clone(), data)
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
/// let querier = ZenohQuerier::new().await?;
/// let replies = querier.get("demo/example/**").await?;
/// ```
pub struct ZenohQuerier {
    name: String,
    session: Arc<Session>,
    timeout: Duration,
    queries_sent: u64,
    replies_received: u64,
}

impl ZenohQuerier {
    /// Create a new Zenoh querier with a new session.
    pub async fn new() -> Result<Self> {
        let session = zenoh::open(zenoh::Config::default())
            .await
            .map_err(|e| Error::Element(format!("Zenoh open error: {}", e)))?;

        Ok(Self::with_session(Arc::new(session)))
    }

    /// Create a new Zenoh querier using an existing session.
    pub fn with_session(session: Arc<Session>) -> Self {
        Self {
            name: "zenoh-querier".to_string(),
            session,
            timeout: Duration::from_secs(10),
            queries_sent: 0,
            replies_received: 0,
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
    pub async fn get(&mut self, key_expr: &str) -> Result<Vec<Buffer>> {
        let replies = self
            .session
            .get(key_expr)
            .timeout(self.timeout)
            .await
            .map_err(|e| Error::Element(format!("Zenoh get error: {}", e)))?;

        self.queries_sent += 1;

        let mut buffers = Vec::new();
        let mut seq = 0u64;

        while let Ok(reply) = replies.recv_async().await {
            match reply.result() {
                Ok(sample) => {
                    let payload = sample.payload();
                    let data: Vec<u8> = payload.to_bytes().to_vec();

                    let segment = Arc::new(HeapSegment::new(data.len().max(1))?);
                    if !data.is_empty() {
                        unsafe {
                            let ptr = segment.as_mut_ptr().unwrap();
                            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                        }
                    }

                    let handle = MemoryHandle::from_segment_with_len(segment, data.len());
                    buffers.push(Buffer::new(handle, Metadata::from_sequence(seq)));
                    seq += 1;
                    self.replies_received += 1;
                }
                Err(_) => {
                    // Query error, skip
                }
            }
        }

        Ok(buffers)
    }

    /// Send a query with a value and collect all replies.
    pub async fn get_with_value(&mut self, key_expr: &str, value: &[u8]) -> Result<Vec<Buffer>> {
        let replies = self
            .session
            .get(key_expr)
            .timeout(self.timeout)
            .payload(value)
            .await
            .map_err(|e| Error::Element(format!("Zenoh get error: {}", e)))?;

        self.queries_sent += 1;

        let mut buffers = Vec::new();
        let mut seq = 0u64;

        while let Ok(reply) = replies.recv_async().await {
            match reply.result() {
                Ok(sample) => {
                    let payload = sample.payload();
                    let data: Vec<u8> = payload.to_bytes().to_vec();

                    let segment = Arc::new(HeapSegment::new(data.len().max(1))?);
                    if !data.is_empty() {
                        unsafe {
                            let ptr = segment.as_mut_ptr().unwrap();
                            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                        }
                    }

                    let handle = MemoryHandle::from_segment_with_len(segment, data.len());
                    buffers.push(Buffer::new(handle, Metadata::from_sequence(seq)));
                    seq += 1;
                    self.replies_received += 1;
                }
                Err(_) => {
                    // Query error, skip
                }
            }
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
