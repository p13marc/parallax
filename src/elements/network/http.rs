//! HTTP source and sink elements.
//!
//! Provides HTTP-based data transfer using blocking I/O (`ureq`).
//!
//! - [`HttpSrc`]: Fetches data from an HTTP endpoint (GET), byte-seekable
//!   via `Range` requests when the server supports them
//! - [`HttpSink`]: Posts data to an HTTP endpoint (POST/PUT)
//!
//! `HttpSrc` probes the server at pipeline start: the executor's seekability
//! snapshot triggers the first GET (the same request `produce` would issue),
//! whose response headers answer `Accept-Ranges`/`Content-Length`. That
//! connection attempt blocks `Executor::start` for up to
//! [`HttpSrc::with_timeout`] against an unreachable server — size the
//! timeout accordingly.
//!
//! Requires the `http` feature flag.

#![cfg(feature = "http")]

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{AsyncSink, ConsumeContext, ProduceContext, ProduceResult, Source};
use crate::error::{Error, Result};
use crate::event::{Event, EventResult, SeekEvent, SeekType, SegmentFormat};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;
use crate::pipeline::seek::{DurationQuery, PositionQuery};
use std::io::Read;
use std::sync::Mutex;
use std::time::Duration;

/// HTTP method for sink operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    /// HTTP POST
    Post,
    /// HTTP PUT
    Put,
    /// HTTP PATCH
    Patch,
}

impl HttpMethod {
    #[cfg_attr(not(test), allow(dead_code))]
    fn as_str(&self) -> &'static str {
        match self {
            HttpMethod::Post => "POST",
            HttpMethod::Put => "PUT",
            HttpMethod::Patch => "PATCH",
        }
    }
}

/// An HTTP source that fetches data from a URL.
///
/// Performs an HTTP GET request and streams the response body.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::HttpSrc;
///
/// let mut src = HttpSrc::new("https://example.com/data")?;
/// while let Some(buffer) = src.produce()? {
///     // Process chunks of response body
/// }
/// ```
pub struct HttpSrc {
    name: String,
    url: String,
    chunk_size: usize,
    timeout: Option<Duration>,
    headers: Vec<(String, String)>,
    /// Interior mutability so the executor's start-time seekability snapshot
    /// (`is_seekable(&self)` / `query_duration(&self)`) can perform the
    /// probe connection. The hot path (`produce`) uses `get_mut` — no lock.
    conn: Mutex<HttpConnection>,
    bytes_read: u64,
    sequence: u64,
    output: OutputArena,
}

/// Connection state behind the probe lock. Shared with `HttpCacheSrc`
/// (#188), which reuses the ranged-GET/probe logic verbatim.
#[derive(Default)]
pub(crate) struct HttpConnection {
    pub(crate) reader: Option<Box<dyn Read + Send>>,
    /// A connection attempt happened and its header answers are recorded
    /// (also set on failure, so the pre-start probe never retries — the
    /// real error resurfaces from `produce`).
    pub(crate) probed: bool,
    pub(crate) status_code: Option<u16>,
    /// The server honors `Range` requests: it answered 206, or advertised
    /// `Accept-Ranges: bytes`.
    pub(crate) seekable: bool,
    /// Total resource length from `Content-Range` (206) or
    /// `Content-Length` (200); `None` for chunked/unknown.
    pub(crate) total_len: Option<u64>,
    /// Byte offset the next connection resumes from (set by a seek).
    pub(crate) next_offset: u64,
}

impl HttpSrc {
    /// Create a new HTTP source.
    pub fn new(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        Ok(Self {
            name: format!("httpsrc-{}", &url[..url.len().min(30)]),
            url,
            chunk_size: 64 * 1024,
            timeout: Some(Duration::from_secs(30)),
            headers: Vec::new(),
            conn: Mutex::new(HttpConnection::default()),
            bytes_read: 0,
            sequence: 0,
            output: OutputArena::new(defaults::SOURCE_SLOT_COUNT),
        })
    }

    /// Set the chunk size for reading response data.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(1);
        self
    }

    /// Set the request timeout.
    ///
    /// Also bounds the pre-start probe: the executor's seekability snapshot
    /// establishes the first connection inside `Executor::start`.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add a custom header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get the HTTP status code (after connection).
    pub fn status_code(&self) -> Option<u16> {
        self.lock_conn().status_code
    }

    /// Get the number of bytes read.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Get the element name.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn lock_conn(&self) -> std::sync::MutexGuard<'_, HttpConnection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// GET `url` from `conn.next_offset` with `Range: bytes=N-`, install the
    /// body reader, and record status / total length / range capability.
    ///
    /// An offset-0 Range doubles as the capability probe: servers that
    /// ignore it answer a harmless 200. After a real seek (`next_offset >
    /// 0`) a 200 means the server ignored the Range — that errors loudly
    /// instead of silently re-downloading from the start.
    pub(crate) fn connect(
        url: &str,
        timeout: Option<Duration>,
        headers: &[(String, String)],
        conn: &mut HttpConnection,
    ) -> Result<()> {
        conn.probed = true;

        let mut request = ureq::get(url);
        if let Some(timeout) = timeout {
            request = request.config().timeout_global(Some(timeout)).build();
        }
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }
        request = request.header("Range", &format!("bytes={}-", conn.next_offset));

        // ureq's default `http_status_as_error` maps 4xx/5xx to Err here.
        let response = request
            .call()
            .map_err(|e| Error::Element(format!("HTTP error: {}", e)))?;

        let status = response.status().as_u16();
        conn.status_code = Some(status);
        if conn.next_offset > 0 && status != 206 {
            return Err(Error::Element(format!(
                "HTTP server ignored Range request (status {status}) — cannot resume at byte {}",
                conn.next_offset
            )));
        }

        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        let accept_ranges = header("accept-ranges").is_some_and(|v| v.contains("bytes"));
        conn.seekable = status == 206 || accept_ranges;
        conn.total_len = if status == 206 {
            header("content-range")
                .as_deref()
                .and_then(parse_content_range_total)
        } else {
            response.body().content_length()
        };
        conn.reader = Some(Box::new(response.into_body().into_reader()));

        Ok(())
    }

    /// Handle a byte seek: record the new offset and drop the connection.
    ///
    /// The reconnect is lazy — the next `produce` issues the Range request —
    /// so a scrub storm of seeks coalesces into one connection, and the
    /// reported landing is exact either way (byte seeks don't snap). A
    /// server that then ignores the Range surfaces as a produce error:
    /// fatal, and rightly so — the pipeline was told this source can seek.
    fn handle_seek(&mut self, seek: &SeekEvent) -> EventResult {
        if seek.format != SegmentFormat::Bytes {
            return EventResult::NotHandled;
        }

        let conn = self.conn.get_mut().unwrap_or_else(|e| e.into_inner());
        if !conn.probed {
            // A graph-level seek can arrive before anything connected.
            let (url, timeout, headers) = (&self.url, self.timeout, &self.headers);
            if let Err(e) = Self::connect(url, timeout, headers, conn) {
                tracing::warn!("httpsrc: probe for seek failed: {e}");
                return EventResult::Error;
            }
        }
        if !conn.seekable {
            tracing::warn!("httpsrc: server does not accept byte ranges; seek refused");
            return EventResult::Error;
        }

        let target = match seek.start.seek_type {
            SeekType::Set => {
                if seek.start.position < 0 {
                    return EventResult::Error;
                }
                seek.start.position as u64
            }
            SeekType::Current => match self.bytes_read.checked_add_signed(seek.start.position) {
                Some(t) => t,
                None => return EventResult::Error,
            },
            SeekType::End => {
                // Only meaningful against a known total (chunked responses
                // have none).
                let Some(total) = conn.total_len else {
                    return EventResult::Error;
                };
                match total.checked_add_signed(seek.start.position) {
                    Some(t) => t,
                    None => return EventResult::Error,
                }
            }
            SeekType::None => return EventResult::handled(),
        };
        // Past-the-end clamps to the end; the next produce sees
        // `next_offset >= total_len` and returns a clean EOS.
        let target = match conn.total_len {
            Some(total) => target.min(total),
            None => target,
        };

        conn.reader = None;
        conn.next_offset = target;
        self.bytes_read = target;
        EventResult::handled_at(target as i64)
    }
}

/// `"bytes 100-199/1000"` → `Some(1000)`; `"bytes 0-99/*"` → `None`.
fn parse_content_range_total(value: &str) -> Option<u64> {
    value.rsplit_once('/')?.1.trim().parse().ok()
}

impl Source for HttpSrc {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        let conn = self.conn.get_mut().unwrap_or_else(|e| e.into_inner());
        if conn.reader.is_none() {
            // A seek to (or past) the end needs no connection: 416 territory.
            if let Some(total) = conn.total_len
                && conn.next_offset >= total
            {
                return Ok(ProduceResult::Eos);
            }
            // First call, post-seek reconnect, or retry after a failed
            // pre-start probe — this surfaces the real connection error.
            let (url, timeout, headers) = (&self.url, self.timeout, &self.headers);
            Self::connect(url, timeout, headers, conn)?;
        }
        let reader = conn.reader.as_mut().expect("connected above");

        let Some(mut slot) = self.output.try_acquire(self.chunk_size, "httpsrc")? else {
            return Ok(ProduceResult::WouldBlock);
        };

        match reader.read(slot.data_mut()) {
            Ok(0) => Ok(ProduceResult::Eos),
            Ok(n) => {
                // `bytes_read` is the absolute stream position (handle_seek
                // resets it to the Range target), so it is this chunk's true
                // offset — stamped for byte-aware consumers (Queue2, #164).
                let offset = self.bytes_read;
                self.bytes_read += n as u64;
                let seq = self.sequence;
                self.sequence += 1;

                let mut metadata = Metadata::from_sequence(seq);
                metadata.offset = Some(offset);
                let handle = MemoryHandle::with_len(slot, n);
                Ok(ProduceResult::OwnBuffer(Buffer::new(handle, metadata)))
            }
            Err(e) => Err(Error::Io(e)),
        }
    }

    fn is_seekable(&self) -> bool {
        let mut conn = self.lock_conn();
        if !conn.probed {
            // The executor snapshots seekability at start; the probe is the
            // same GET produce() would issue, so the connection is reused.
            if let Err(e) = Self::connect(&self.url, self.timeout, &self.headers, &mut conn) {
                tracing::warn!("httpsrc: probe failed, reporting unseekable: {e}");
                return false;
            }
        }
        conn.seekable
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        match event {
            Event::Seek(seek) => self.handle_seek(seek),
            _ => EventResult::NotHandled,
        }
    }

    fn query_position(&self) -> Option<PositionQuery> {
        Some(PositionQuery {
            format: SegmentFormat::Bytes,
            position: Some(self.bytes_read),
        })
    }

    fn query_duration(&self) -> Option<DurationQuery> {
        let mut conn = self.lock_conn();
        if !conn.probed
            && let Err(e) = Self::connect(&self.url, self.timeout, &self.headers, &mut conn)
        {
            tracing::warn!("httpsrc: probe failed, duration unknown: {e}");
        }
        Some(DurationQuery {
            format: SegmentFormat::Bytes,
            duration: conn.total_len,
        })
    }
}

/// An HTTP sink that posts data to a URL.
///
/// Each buffer is sent as a separate HTTP request.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::{HttpSink, HttpMethod};
///
/// let mut sink = HttpSink::new("https://example.com/upload", HttpMethod::Post)?;
/// sink.consume(buffer)?;
/// ```
pub struct HttpSink {
    name: String,
    url: String,
    method: HttpMethod,
    timeout: Option<Duration>,
    headers: Vec<(String, String)>,
    content_type: String,
    bytes_written: u64,
    requests_sent: u64,
    last_status: Option<u16>,
}

impl HttpSink {
    /// Create a new HTTP sink.
    pub fn new(url: impl Into<String>, method: HttpMethod) -> Result<Self> {
        let url = url.into();
        Ok(Self {
            name: format!("httpsink-{}", &url[..url.len().min(30)]),
            url,
            method,
            timeout: Some(Duration::from_secs(30)),
            headers: Vec::new(),
            content_type: "application/octet-stream".to_string(),
            bytes_written: 0,
            requests_sent: 0,
            last_status: None,
        })
    }

    /// Set the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add a custom header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Set the Content-Type header.
    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = content_type.into();
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get the HTTP method.
    pub fn method(&self) -> HttpMethod {
        self.method
    }

    /// Get the number of bytes written.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Get the number of requests sent.
    pub fn requests_sent(&self) -> u64 {
        self.requests_sent
    }

    /// Get the last HTTP status code.
    pub fn last_status(&self) -> Option<u16> {
        self.last_status
    }

    /// Get the element name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get statistics.
    pub fn stats(&self) -> HttpSinkStats {
        HttpSinkStats {
            bytes_written: self.bytes_written,
            requests_sent: self.requests_sent,
            last_status: self.last_status,
        }
    }
}

impl AsyncSink for HttpSink {
    /// One blocking ureq round-trip per buffer, run on tokio's blocking pool
    /// (#172): a slow server stalls this element's future, not a runtime
    /// worker. The 30 s default `timeout` bounds how long the pool thread is
    /// held.
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        // Owned copies for the 'static closure — an HTTP body write copies
        // the payload into the socket anyway.
        let data = ctx.input().to_vec();
        let len = data.len() as u64;
        let url = self.url.clone();
        let method = self.method;
        let timeout = self.timeout;
        let content_type = self.content_type.clone();
        let headers = self.headers.clone();

        let status = tokio::task::spawn_blocking(move || {
            let mut request = match method {
                HttpMethod::Post => ureq::post(&url),
                HttpMethod::Put => ureq::put(&url),
                HttpMethod::Patch => ureq::patch(&url),
            };

            if let Some(timeout) = timeout {
                request = request.config().timeout_global(Some(timeout)).build();
            }

            request = request.header("Content-Type", content_type.as_str());

            for (name, value) in &headers {
                request = request.header(name.as_str(), value.as_str());
            }

            request
                .send(&data[..])
                .map(|response| response.status().as_u16())
                .map_err(|e| Error::Element(format!("HTTP error: {}", e)))
        })
        .await
        .map_err(|e| Error::Element(format!("http sink task join error: {}", e)))??;

        self.last_status = Some(status);
        self.bytes_written += len;
        self.requests_sent += 1;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for HTTP sink.
#[derive(Debug, Clone, Copy, Default)]
pub struct HttpSinkStats {
    /// Total bytes written.
    pub bytes_written: u64,
    /// Total requests sent.
    pub requests_sent: u64,
    /// Last HTTP status code.
    pub last_status: Option<u16>,
}

/// Minimal blocking HTTP/1.1 test server (no dev-dependency; the same
/// TcpListener-thread pattern as the tcp.rs tests). One request per
/// connection, `Connection: close`. Shared with the `HttpCacheSrc` tests
/// (#188), which also count requests to prove serve-from-disk.
#[cfg(test)]
pub(crate) mod testserver {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    pub(crate) struct ServerOpts {
        /// Honor `Range: bytes=N-` with a 206 + Content-Range.
        pub(crate) support_range: bool,
        /// Advertise `Accept-Ranges: bytes` on 200 responses.
        pub(crate) advertise_accept_ranges: bool,
        /// Force this status on every response (with a tiny error body).
        pub(crate) status: u16,
        /// Include Content-Length (200) / Content-Range total (206).
        pub(crate) send_length: bool,
    }

    impl Default for ServerOpts {
        fn default() -> Self {
            Self {
                support_range: true,
                advertise_accept_ranges: true,
                status: 200,
                send_length: true,
            }
        }
    }

    pub(crate) fn range_server(body: Vec<u8>, opts: ServerOpts) -> SocketAddr {
        range_server_counted(body, opts).0
    }

    /// Like [`range_server`], returning a counter of requests served.
    pub(crate) fn range_server_counted(
        body: Vec<u8>,
        opts: ServerOpts,
    ) -> (SocketAddr, Arc<AtomicU64>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicU64::new(0));
        let counter = requests.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                // Read the request head.
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    match stream.read(&mut byte) {
                        Ok(1) => head.push(byte[0]),
                        _ => break,
                    }
                }
                counter.fetch_add(1, Ordering::SeqCst);
                let head = String::from_utf8_lossy(&head);
                let range_start: Option<u64> = head.lines().find_map(|l| {
                    let (name, value) = l.split_once(':')?;
                    if !name.eq_ignore_ascii_case("range") {
                        return None;
                    }
                    let spec = value.trim().strip_prefix("bytes=")?;
                    spec.split_once('-')?.0.parse().ok()
                });

                let mut response = Vec::new();
                if opts.status != 200 {
                    let msg = b"error";
                    write!(
                        response,
                        "HTTP/1.1 {} Error\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        opts.status,
                        msg.len()
                    )
                    .unwrap();
                    response.extend_from_slice(msg);
                } else if let (Some(start), true) = (range_start, opts.support_range) {
                    // 206 with the requested suffix. An out-of-range start
                    // yields an empty body (tests seek-to-end).
                    let start = start.min(body.len() as u64) as usize;
                    let part = &body[start..];
                    let total = if opts.send_length {
                        body.len().to_string()
                    } else {
                        "*".to_string()
                    };
                    write!(
                        response,
                        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/{}\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n",
                        start,
                        (start + part.len()).saturating_sub(1),
                        total,
                        part.len()
                    )
                    .unwrap();
                    response.extend_from_slice(part);
                } else {
                    // Plain 200 from the start (Range absent or unsupported).
                    write!(response, "HTTP/1.1 200 OK\r\n").unwrap();
                    if opts.advertise_accept_ranges {
                        write!(response, "Accept-Ranges: bytes\r\n").unwrap();
                    }
                    if opts.send_length {
                        write!(response, "Content-Length: {}\r\n", body.len()).unwrap();
                    }
                    write!(response, "Connection: close\r\n\r\n").unwrap();
                    response.extend_from_slice(&body);
                }
                let _ = stream.write_all(&response);
            }
        });
        (addr, requests)
    }
}

#[cfg(test)]
mod tests {
    use super::testserver::{ServerOpts, range_server};
    use super::*;
    use std::net::SocketAddr;

    fn body_of(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn src_for(addr: SocketAddr) -> HttpSrc {
        HttpSrc::new(format!("http://{addr}/data"))
            .unwrap()
            .with_chunk_size(64)
            .with_timeout(Duration::from_secs(5))
    }

    /// Drain produce() to EOS, appending payloads.
    fn drain(src: &mut HttpSrc, out: &mut Vec<u8>) {
        loop {
            let mut ctx = crate::element::ProduceContext::without_buffer();
            match src.produce(&mut ctx).unwrap() {
                ProduceResult::OwnBuffer(buffer) => out.extend_from_slice(buffer.as_bytes()),
                ProduceResult::Eos => break,
                ProduceResult::WouldBlock => std::thread::yield_now(),
                other => panic!("unexpected produce result: {other:?}"),
            }
        }
    }

    #[test]
    fn http_src_streams_body_to_eos() {
        let body = body_of(1000);
        let addr = range_server(body.clone(), ServerOpts::default());
        let mut src = src_for(addr);

        let mut out = Vec::new();
        drain(&mut src, &mut out);
        assert_eq!(out, body);
        assert_eq!(src.bytes_read(), 1000);
        assert_eq!(src.status_code(), Some(206), "offset-0 Range honored");
    }

    #[test]
    fn http_src_is_seekable_with_range_support() {
        let addr = range_server(body_of(100), ServerOpts::default());
        let src = src_for(addr);
        assert!(src.is_seekable());

        // Accept-Ranges alone (200) also qualifies.
        let addr = range_server(
            body_of(100),
            ServerOpts {
                support_range: false,
                advertise_accept_ranges: true,
                ..Default::default()
            },
        );
        let src = src_for(addr);
        assert!(src.is_seekable());
    }

    #[test]
    fn http_src_not_seekable_when_ranges_ignored() {
        let addr = range_server(
            body_of(100),
            ServerOpts {
                support_range: false,
                advertise_accept_ranges: false,
                ..Default::default()
            },
        );
        let src = src_for(addr);
        assert!(!src.is_seekable());

        // ...and a seek against it is refused.
        let mut src = src_for(addr);
        let result = src.handle_upstream_event(&Event::Seek(SeekEvent::new_bytes(10)));
        assert!(matches!(result, EventResult::Error));
    }

    #[test]
    fn http_src_query_duration_reports_length() {
        let addr = range_server(body_of(1234), ServerOpts::default());
        let src = src_for(addr);
        let q = src.query_duration().unwrap();
        assert_eq!(q.format, SegmentFormat::Bytes);
        assert_eq!(q.duration, Some(1234));

        // Unknown when the server sends no length.
        let addr = range_server(
            body_of(100),
            ServerOpts {
                support_range: false,
                advertise_accept_ranges: true,
                send_length: false,
                ..Default::default()
            },
        );
        let src = src_for(addr);
        assert_eq!(src.query_duration().unwrap().duration, None);
    }

    #[test]
    fn http_src_seek_set_resumes_at_offset() {
        let body = body_of(1000);
        let addr = range_server(body.clone(), ServerOpts::default());
        let mut src = src_for(addr);

        // Read one chunk from the head first.
        let mut ctx = crate::element::ProduceContext::without_buffer();
        let ProduceResult::OwnBuffer(first) = src.produce(&mut ctx).unwrap() else {
            panic!("expected data");
        };
        assert_eq!(first.as_bytes(), &body[..64]);
        assert_eq!(
            first.metadata().offset,
            Some(0),
            "buffers carry their absolute stream offset (#164)"
        );

        let result = src.handle_upstream_event(&Event::Seek(SeekEvent::new_bytes(700)));
        assert!(matches!(
            result,
            EventResult::Handled {
                position: Some(700)
            }
        ));
        assert_eq!(src.query_position().unwrap().position, Some(700));

        // The first post-seek buffer is stamped with the seek target.
        let ProduceResult::OwnBuffer(resumed) = src.produce(&mut ctx).unwrap() else {
            panic!("expected post-seek data");
        };
        assert_eq!(resumed.metadata().offset, Some(700), "post-seek offset");
        assert_eq!(resumed.as_bytes(), &body[700..764]);

        let mut out = Vec::new();
        drain(&mut src, &mut out);
        assert_eq!(
            out,
            &body[764..],
            "stream continues after the stamped chunk"
        );
        assert_eq!(src.bytes_read(), 1000);
    }

    #[test]
    fn http_src_seek_current_and_end() {
        let body = body_of(500);
        let addr = range_server(body.clone(), ServerOpts::default());
        let mut src = src_for(addr);

        // Current-relative from a known position.
        let mut ctx = crate::element::ProduceContext::without_buffer();
        assert!(matches!(
            src.produce(&mut ctx).unwrap(),
            ProduceResult::OwnBuffer(_)
        ));
        let seek = SeekEvent::new(
            SegmentFormat::Bytes,
            crate::event::SeekPosition::current(100),
        );
        assert!(
            matches!(
                src.handle_upstream_event(&Event::Seek(seek)),
                EventResult::Handled {
                    position: Some(164)
                }
            ),
            "64 read + 100 forward"
        );

        // End-relative needs the total.
        let seek = SeekEvent::new(SegmentFormat::Bytes, crate::event::SeekPosition::end(-100));
        assert!(matches!(
            src.handle_upstream_event(&Event::Seek(seek)),
            EventResult::Handled {
                position: Some(400)
            }
        ));
        let mut out = Vec::new();
        drain(&mut src, &mut out);
        assert_eq!(out, &body[400..]);

        // Past-the-end clamps and the next produce is a clean EOS.
        let seek = SeekEvent::new_bytes(9_999);
        assert!(matches!(
            src.handle_upstream_event(&Event::Seek(seek)),
            EventResult::Handled {
                position: Some(500)
            }
        ));
        let mut out = Vec::new();
        drain(&mut src, &mut out);
        assert!(out.is_empty());

        // End-relative against an unknown total errors.
        let addr = range_server(
            body_of(100),
            ServerOpts {
                support_range: false,
                advertise_accept_ranges: true,
                send_length: false,
                ..Default::default()
            },
        );
        let mut src = src_for(addr);
        let seek = SeekEvent::new(SegmentFormat::Bytes, crate::event::SeekPosition::end(-10));
        assert!(matches!(
            src.handle_upstream_event(&Event::Seek(seek)),
            EventResult::Error
        ));
    }

    #[test]
    fn http_src_errors_when_server_ignores_range_after_seek() {
        // Server advertises Accept-Ranges (so the seek is accepted) but
        // answers every request with 200 — the post-seek produce must fail
        // loudly instead of silently restarting from byte 0.
        let addr = range_server(
            body_of(300),
            ServerOpts {
                support_range: false,
                advertise_accept_ranges: true,
                ..Default::default()
            },
        );
        let mut src = src_for(addr);
        assert!(src.is_seekable(), "server lies about ranges");

        assert!(matches!(
            src.handle_upstream_event(&Event::Seek(SeekEvent::new_bytes(100))),
            EventResult::Handled {
                position: Some(100)
            }
        ));
        let mut ctx = crate::element::ProduceContext::without_buffer();
        let err = src.produce(&mut ctx).unwrap_err();
        assert!(
            err.to_string().contains("ignored Range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn http_src_non_2xx_is_an_error() {
        // Pins ureq's default `http_status_as_error`: a 404 body must never
        // stream as data.
        let addr = range_server(
            body_of(100),
            ServerOpts {
                status: 404,
                ..Default::default()
            },
        );
        let src = src_for(addr);
        assert!(!src.is_seekable(), "probe failure reports unseekable");

        let mut src = src_for(addr);
        let mut ctx = crate::element::ProduceContext::without_buffer();
        assert!(matches!(src.produce(&mut ctx), Err(Error::Element(_))));
    }

    #[test]
    fn parse_content_range_total_cases() {
        assert_eq!(parse_content_range_total("bytes 100-199/1000"), Some(1000));
        assert_eq!(parse_content_range_total("bytes 0-0/1"), Some(1));
        assert_eq!(parse_content_range_total("bytes 0-99/*"), None);
        assert_eq!(parse_content_range_total("garbage"), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn http_src_seeks_at_runtime() {
        use crate::elements::{AppSink, Pulled};
        use crate::pipeline::bus::MessageKind;
        use crate::pipeline::{Executor, LinkPolicy, Pipeline};

        let body = body_of(4096);
        let addr = range_server(body.clone(), ServerOpts::default());
        let src = src_for(addr).with_chunk_size(256);

        let mut pipeline = Pipeline::new();
        let sink = AppSink::with_max_buffers(2);
        let sink_handle = sink.handle();
        let s = pipeline.add_source("httpsrc", src);
        let k = pipeline.add_async_sink("appsink", sink);
        // Tight link capacity: backpressure parks the source mid-stream, so
        // the seek deterministically lands before EOS.
        pipeline
            .link_pads_full(s, "src", k, "sink", LinkPolicy::Block, Some(2))
            .unwrap();

        let executor = Executor::new();
        let mut handle = executor.start(&mut pipeline).unwrap();
        let mut bus = handle.take_bus().unwrap();

        // The start-time snapshot probed the server.
        assert!(handle.seekable());
        assert_eq!(handle.query_duration().unwrap().duration, Some(4096));

        // Pull a couple of chunks, then seek.
        for _ in 0..2 {
            assert!(matches!(sink_handle.pull_buffer().await, Pulled::Buffer(_)));
        }
        assert!(handle.seek_bytes(3000).await);

        // Everything after the seek point: the tail of the collected stream
        // must be exactly body[3000..].
        let mut collected = Vec::new();
        while let Pulled::Buffer(b) = sink_handle.pull_buffer().await {
            collected.extend_from_slice(b.as_bytes());
        }
        handle.wait().await.unwrap();
        assert!(
            collected.ends_with(&body[3000..]),
            "stream did not resume at the seek offset"
        );

        let mut seek_done = None;
        while let Some(msg) = bus.poll() {
            if let MessageKind::SeekDone {
                format, position, ..
            } = msg.kind
            {
                seek_done = Some((format, position));
            }
        }
        assert_eq!(
            seek_done,
            Some((SegmentFormat::Bytes, Some(3000))),
            "SeekDone carries the byte landing"
        );
    }

    #[test]
    fn test_http_src_creation() {
        let src = HttpSrc::new("https://example.com/data").unwrap();
        assert_eq!(src.url(), "https://example.com/data");
    }

    #[test]
    fn test_http_src_with_headers() {
        let src = HttpSrc::new("https://example.com")
            .unwrap()
            .with_header("Authorization", "Bearer token123")
            .with_header("Accept", "application/json");

        assert_eq!(src.headers.len(), 2);
    }

    #[test]
    fn test_http_sink_creation() {
        let sink = HttpSink::new("https://example.com/upload", HttpMethod::Post).unwrap();
        assert_eq!(sink.url(), "https://example.com/upload");
        assert_eq!(sink.method(), HttpMethod::Post);
    }

    #[test]
    fn test_http_sink_with_content_type() {
        let sink = HttpSink::new("https://example.com/upload", HttpMethod::Put)
            .unwrap()
            .with_content_type("application/json");

        assert_eq!(sink.content_type, "application/json");
    }

    #[test]
    fn test_http_method_as_str() {
        assert_eq!(HttpMethod::Post.as_str(), "POST");
        assert_eq!(HttpMethod::Put.as_str(), "PUT");
        assert_eq!(HttpMethod::Patch.as_str(), "PATCH");
    }

    #[test]
    fn test_http_src_with_name() {
        let src = HttpSrc::new("https://example.com")
            .unwrap()
            .with_name("my-http-source");

        assert_eq!(src.name(), "my-http-source");
    }

    #[test]
    fn test_http_sink_with_timeout() {
        let sink = HttpSink::new("https://example.com", HttpMethod::Post)
            .unwrap()
            .with_timeout(Duration::from_secs(60));

        assert_eq!(sink.timeout, Some(Duration::from_secs(60)));
    }
}
