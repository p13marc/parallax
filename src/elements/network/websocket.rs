//! WebSocket source and sink elements.
//!
//! Provides WebSocket-based bidirectional communication.
//!
//! - [`WebSocketSrc`]: Receives messages from a WebSocket connection
//! - [`WebSocketSink`]: Sends messages to a WebSocket connection
//!
//! Note: These elements use `tungstenite` for WebSocket protocol support.
//!
//! Requires the `websocket` feature flag.

#![cfg(feature = "websocket")]

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Sink, Source};
use crate::error::{Error, Result};
use crate::memory::{CpuSegment, MemorySegment};
use crate::metadata::Metadata;
use std::net::TcpStream;
use std::sync::Arc;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket, connect};

/// Message type for WebSocket communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsMessageType {
    /// Binary message
    Binary,
    /// Text message
    Text,
}

/// A WebSocket source that receives messages from a WebSocket server.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::WebSocketSrc;
///
/// let mut src = WebSocketSrc::connect("ws://localhost:8080/stream")?;
/// while let Some(buffer) = src.produce()? {
///     // Process received messages
/// }
/// ```
pub struct WebSocketSrc {
    name: String,
    url: String,
    socket: Option<WebSocket<MaybeTlsStream<TcpStream>>>,
    connected: bool,
    bytes_read: u64,
    messages_received: u64,
    sequence: u64,
}

impl WebSocketSrc {
    /// Create a new WebSocket source.
    ///
    /// Does not connect immediately - connection happens on first `produce()` call.
    pub fn new(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        Ok(Self {
            name: format!("wssrc-{}", &url[..url.len().min(30)]),
            url,
            socket: None,
            connected: false,
            bytes_read: 0,
            messages_received: 0,
            sequence: 0,
        })
    }

    /// Create and immediately connect to a WebSocket server.
    pub fn connect(url: impl Into<String>) -> Result<Self> {
        let mut src = Self::new(url)?;
        src.ensure_connected()?;
        Ok(src)
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

    /// Get the number of bytes received.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Get the number of messages received.
    pub fn messages_received(&self) -> u64 {
        self.messages_received
    }

    /// Check if connected.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get statistics.
    pub fn stats(&self) -> WebSocketStats {
        WebSocketStats {
            bytes_transferred: self.bytes_read,
            messages: self.messages_received,
        }
    }

    /// Close the WebSocket connection.
    pub fn close(&mut self) -> Result<()> {
        if let Some(ref mut socket) = self.socket {
            socket
                .close(None)
                .map_err(|e| Error::Element(format!("close error: {}", e)))?;
        }
        self.connected = false;
        Ok(())
    }

    fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        let (socket, _response) = connect(&self.url)
            .map_err(|e| Error::Element(format!("WebSocket connect error: {}", e)))?;

        self.socket = Some(socket);
        self.connected = true;
        Ok(())
    }
}

impl Source for WebSocketSrc {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        self.ensure_connected()?;

        let socket = self
            .socket
            .as_mut()
            .ok_or_else(|| Error::Element("not connected".into()))?;

        loop {
            match socket.read() {
                Ok(Message::Binary(data)) => {
                    self.bytes_read += data.len() as u64;
                    self.messages_received += 1;
                    let seq = self.sequence;
                    self.sequence += 1;

                    let segment = Arc::new(CpuSegment::new(data.len().max(1))?);
                    if !data.is_empty() {
                        unsafe {
                            let ptr = segment.as_mut_ptr().unwrap();
                            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                        }
                    }

                    let handle = MemoryHandle::from_segment_with_len(segment, data.len());
                    return Ok(Some(Buffer::new(handle, Metadata::from_sequence(seq))));
                }
                Ok(Message::Text(text)) => {
                    let data = text.into_bytes();
                    self.bytes_read += data.len() as u64;
                    self.messages_received += 1;
                    let seq = self.sequence;
                    self.sequence += 1;

                    let segment = Arc::new(CpuSegment::new(data.len().max(1))?);
                    if !data.is_empty() {
                        unsafe {
                            let ptr = segment.as_mut_ptr().unwrap();
                            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                        }
                    }

                    let handle = MemoryHandle::from_segment_with_len(segment, data.len());
                    return Ok(Some(Buffer::new(handle, Metadata::from_sequence(seq))));
                }
                Ok(Message::Close(_)) => {
                    self.connected = false;
                    return Ok(None);
                }
                Ok(Message::Ping(data)) => {
                    // Respond to ping with pong
                    let _ = socket.send(Message::Pong(data));
                    continue;
                }
                Ok(Message::Pong(_)) => {
                    // Ignore pong messages
                    continue;
                }
                Ok(Message::Frame(_)) => {
                    // Raw frame, skip
                    continue;
                }
                Err(tungstenite::Error::ConnectionClosed) => {
                    self.connected = false;
                    return Ok(None);
                }
                Err(e) => {
                    return Err(Error::Element(format!("WebSocket read error: {}", e)));
                }
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for WebSocketSrc {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// A WebSocket sink that sends messages to a WebSocket server.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::{WebSocketSink, WsMessageType};
///
/// let mut sink = WebSocketSink::connect("ws://localhost:8080/stream")?;
/// sink.consume(buffer)?;
/// ```
pub struct WebSocketSink {
    name: String,
    url: String,
    socket: Option<WebSocket<MaybeTlsStream<TcpStream>>>,
    message_type: WsMessageType,
    connected: bool,
    bytes_written: u64,
    messages_sent: u64,
}

impl WebSocketSink {
    /// Create a new WebSocket sink.
    ///
    /// Does not connect immediately - connection happens on first `consume()` call.
    pub fn new(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        Ok(Self {
            name: format!("wssink-{}", &url[..url.len().min(30)]),
            url,
            socket: None,
            message_type: WsMessageType::Binary,
            connected: false,
            bytes_written: 0,
            messages_sent: 0,
        })
    }

    /// Create and immediately connect to a WebSocket server.
    pub fn connect(url: impl Into<String>) -> Result<Self> {
        let mut sink = Self::new(url)?;
        sink.ensure_connected()?;
        Ok(sink)
    }

    /// Set the message type (binary or text).
    pub fn with_message_type(mut self, msg_type: WsMessageType) -> Self {
        self.message_type = msg_type;
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

    /// Get the number of bytes sent.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Get the number of messages sent.
    pub fn messages_sent(&self) -> u64 {
        self.messages_sent
    }

    /// Check if connected.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get statistics.
    pub fn stats(&self) -> WebSocketStats {
        WebSocketStats {
            bytes_transferred: self.bytes_written,
            messages: self.messages_sent,
        }
    }

    /// Send a ping message.
    pub fn ping(&mut self) -> Result<()> {
        if let Some(ref mut socket) = self.socket {
            socket
                .send(Message::Ping(vec![]))
                .map_err(|e| Error::Element(format!("ping error: {}", e)))?;
        }
        Ok(())
    }

    /// Close the WebSocket connection.
    pub fn close(&mut self) -> Result<()> {
        if let Some(ref mut socket) = self.socket {
            socket
                .close(None)
                .map_err(|e| Error::Element(format!("close error: {}", e)))?;
        }
        self.connected = false;
        Ok(())
    }

    fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        let (socket, _response) = connect(&self.url)
            .map_err(|e| Error::Element(format!("WebSocket connect error: {}", e)))?;

        self.socket = Some(socket);
        self.connected = true;
        Ok(())
    }
}

impl Sink for WebSocketSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        self.ensure_connected()?;

        let socket = self
            .socket
            .as_mut()
            .ok_or_else(|| Error::Element("not connected".into()))?;

        let data = buffer.as_bytes();

        let message = match self.message_type {
            WsMessageType::Binary => Message::Binary(data.to_vec()),
            WsMessageType::Text => {
                let text = String::from_utf8_lossy(data).into_owned();
                Message::Text(text)
            }
        };

        socket
            .send(message)
            .map_err(|e| Error::Element(format!("WebSocket send error: {}", e)))?;

        self.bytes_written += data.len() as u64;
        self.messages_sent += 1;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for WebSocketSink {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Statistics for WebSocket elements.
#[derive(Debug, Clone, Copy, Default)]
pub struct WebSocketStats {
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Total messages transferred.
    pub messages: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_src_creation() {
        let src = WebSocketSrc::new("ws://localhost:8080").unwrap();
        assert_eq!(src.url(), "ws://localhost:8080");
        assert!(!src.is_connected());
    }

    #[test]
    fn test_websocket_sink_creation() {
        let sink = WebSocketSink::new("ws://localhost:8080").unwrap();
        assert_eq!(sink.url(), "ws://localhost:8080");
        assert!(!sink.is_connected());
    }

    #[test]
    fn test_websocket_sink_message_type() {
        let sink = WebSocketSink::new("ws://localhost:8080")
            .unwrap()
            .with_message_type(WsMessageType::Text);

        assert_eq!(sink.message_type, WsMessageType::Text);
    }

    #[test]
    fn test_websocket_src_with_name() {
        let src = WebSocketSrc::new("ws://localhost:8080")
            .unwrap()
            .with_name("my-ws-source");

        assert_eq!(src.name(), "my-ws-source");
    }

    #[test]
    fn test_websocket_stats_default() {
        let stats = WebSocketStats::default();
        assert_eq!(stats.bytes_transferred, 0);
        assert_eq!(stats.messages, 0);
    }
}
