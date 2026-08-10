//! HTTP source and sink elements.
//!
//! Provides HTTP-based data transfer using blocking I/O.
//!
//! - [`HttpSrc`]: Fetches data from an HTTP endpoint (GET)
//! - [`HttpSink`]: Posts data to an HTTP endpoint (POST/PUT)
//!
//! Note: These elements use `ureq` for simple blocking HTTP.
//! For async HTTP, consider using `reqwest` with AsyncHttpSrc/AsyncHttpSink.
//!
//! Requires the `http` feature flag.

#![cfg(feature = "http")]

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use crate::error::{Error, Result};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;
use std::io::Read;
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
    response: Option<Box<dyn Read + Send>>,
    connected: bool,
    bytes_read: u64,
    sequence: u64,
    status_code: Option<u16>,
    output: OutputArena,
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
            response: None,
            connected: false,
            bytes_read: 0,
            sequence: 0,
            status_code: None,
            output: OutputArena::new(defaults::SOURCE_SLOT_COUNT),
        })
    }

    /// Set the chunk size for reading response data.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(1);
        self
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
        self.status_code
    }

    /// Get the number of bytes read.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Get the element name.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        let mut request = ureq::get(&self.url);

        if let Some(timeout) = self.timeout {
            request = request.config().timeout_global(Some(timeout)).build();
        }

        for (name, value) in &self.headers {
            request = request.header(name.as_str(), value.as_str());
        }

        let response = request
            .call()
            .map_err(|e| Error::Element(format!("HTTP error: {}", e)))?;

        self.status_code = Some(response.status().as_u16());
        self.response = Some(Box::new(response.into_body().into_reader()));
        self.connected = true;

        Ok(())
    }
}

impl Source for HttpSrc {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        self.ensure_connected()?;

        let reader = self
            .response
            .as_mut()
            .ok_or_else(|| Error::Element("not connected".into()))?;

        let Some(mut slot) = self.output.try_acquire(self.chunk_size, "httpsrc")? else {
            return Ok(ProduceResult::WouldBlock);
        };

        match reader.read(slot.data_mut()) {
            Ok(0) => Ok(ProduceResult::Eos),
            Ok(n) => {
                self.bytes_read += n as u64;
                let seq = self.sequence;
                self.sequence += 1;

                let handle = MemoryHandle::with_len(slot, n);
                Ok(ProduceResult::OwnBuffer(Buffer::new(
                    handle,
                    Metadata::from_sequence(seq),
                )))
            }
            Err(e) => Err(Error::Io(e)),
        }
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

impl Sink for HttpSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let data = ctx.input();

        let mut request = match self.method {
            HttpMethod::Post => ureq::post(&self.url),
            HttpMethod::Put => ureq::put(&self.url),
            HttpMethod::Patch => ureq::patch(&self.url),
        };

        if let Some(timeout) = self.timeout {
            request = request.config().timeout_global(Some(timeout)).build();
        }

        request = request.header("Content-Type", self.content_type.as_str());

        for (name, value) in &self.headers {
            request = request.header(name.as_str(), value.as_str());
        }

        let response = request
            .send(data)
            .map_err(|e| Error::Element(format!("HTTP error: {}", e)))?;

        self.last_status = Some(response.status().as_u16());
        self.bytes_written += data.len() as u64;
        self.requests_sent += 1;

        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

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
