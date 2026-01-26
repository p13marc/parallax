//! Network transport elements.
//!
//! - [`TcpSrc`], [`TcpSink`]: TCP socket I/O
//! - [`UdpSrc`], [`UdpSink`]: UDP datagram I/O
//! - [`UnixSrc`], [`UnixSink`]: Unix domain socket I/O
//! - [`UdpMulticastSrc`], [`UdpMulticastSink`]: UDP multicast
//! - [`HttpSrc`], [`HttpSink`]: HTTP GET/POST (requires `http` feature)
//! - [`WebSocketSrc`], [`WebSocketSink`]: WebSocket (requires `websocket` feature)
//! - [`ZenohSrc`], [`ZenohSink`]: Zenoh pub/sub (requires `zenoh` feature)

mod multicast;
mod tcp;
mod udp;
mod unix;

#[cfg(feature = "http")]
mod http;

#[cfg(feature = "websocket")]
mod websocket;

#[cfg(feature = "zenoh")]
mod zenoh;

// TCP
pub use tcp::{AsyncTcpSink, AsyncTcpSrc, TcpMode, TcpSink, TcpSrc};

// UDP
pub use udp::{AsyncUdpSink, AsyncUdpSrc, UdpSink, UdpSrc};

// Unix domain sockets
pub use unix::{AsyncUnixSink, AsyncUnixSrc, UnixMode, UnixSink, UnixSrc};

// Multicast
pub use multicast::{UdpMulticastSink, UdpMulticastSrc, UdpMulticastStats};

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
