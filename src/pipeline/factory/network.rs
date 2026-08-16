//! Factory registrations: network elements (TCP/UDP/Unix/multicast, plus
//! HTTP and WebSocket under their features).
//!
//! Constructors here do real I/O (connect/bind) at creation time, exactly
//! like the programmatic API they wrap — a bad address fails the parse, not
//! the run.

use super::{ElementFactory, Props};
use crate::element::{
    AsyncSinkAdapter, AsyncSourceAdapter, DynAsyncElement, SinkAdapter, SourceAdapter,
};
use crate::error::Result;

pub(super) fn register(f: &mut ElementFactory) {
    f.register("tcpsrc", create_tcpsrc);
    f.register("tcpsink", create_tcpsink);
    f.register("udpsrc", create_udpsrc);
    f.register("udpsink", create_udpsink);
    f.register("unixsrc", create_unixsrc);
    f.register("unixsink", create_unixsink);
    f.register("multicastsrc", create_multicastsrc);
    f.register("multicastsink", create_multicastsink);
    #[cfg(feature = "http")]
    {
        f.register("httpsrc", create_httpsrc);
        f.register("httpcachesrc", create_httpcachesrc);
        f.register("httpsink", create_httpsink);
    }
    #[cfg(feature = "websocket")]
    {
        f.register("wssrc", create_wssrc);
        f.register("wssink", create_wssink);
    }
}

fn create_tcpsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::AsyncTcpSrc;
    let address = props.req_str("address")?;
    let mut src = AsyncTcpSrc::connect(address.as_str())?;
    if let Some(size) = props.get_usize("buffer-size")? {
        src = src.with_buffer_size(size);
    }
    Ok(DynAsyncElement::new_box(AsyncSourceAdapter::new(src)))
}

fn create_tcpsink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::AsyncTcpSink;
    let address = props.req_str("address")?;
    let sink = AsyncTcpSink::connect(address.as_str())?;
    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(sink)))
}

// The async UDP pair's constructors are `async fn`s, which a sync factory
// cannot call — the sync elements are the registrable ones.
fn create_udpsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::UdpSrc;
    let address = props.req_str("address")?;
    let mut src = UdpSrc::bind(address.as_str())?;
    if let Some(size) = props.get_usize("buffer-size")? {
        src = src.with_buffer_size(size);
    }
    if let Some(t) = props.get_ms("timeout-ms")? {
        src = src.with_read_timeout(t)?;
    }
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

fn create_udpsink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::UdpSink;
    let address = props.req_str("address")?;
    let mut sink = UdpSink::connect(address.as_str())?;
    if let Some(t) = props.get_ms("timeout-ms")? {
        sink = sink.with_write_timeout(t)?;
    }
    Ok(DynAsyncElement::new_box(SinkAdapter::new(sink)))
}

fn create_unixsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::UnixSrc;
    let path = props.req_str("path")?;
    let listen = props.get_bool("listen")?.unwrap_or(false);
    let mut src = if listen {
        UnixSrc::listen(&path)?
    } else {
        UnixSrc::connect(&path)?
    };
    if let Some(size) = props.get_usize("buffer-size")? {
        src = src.with_buffer_size(size);
    }
    if let Some(t) = props.get_ms("timeout-ms")? {
        src = src.with_read_timeout(t);
    }
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

fn create_unixsink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::UnixSink;
    let path = props.req_str("path")?;
    let listen = props.get_bool("listen")?.unwrap_or(false);
    let mut sink = if listen {
        UnixSink::listen(&path)?
    } else {
        UnixSink::connect(&path)?
    };
    if let Some(t) = props.get_ms("timeout-ms")? {
        sink = sink.with_write_timeout(t);
    }
    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(sink)))
}

fn create_multicastsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::UdpMulticastSrc;
    let group = props.req_str("group")?;
    let port = props.req_u16("port")?;
    let mut src = UdpMulticastSrc::new(&group, port)?;
    if let Some(size) = props.get_usize("buffer-size")? {
        src = src.with_buffer_size(size);
    }
    if let Some(t) = props.get_ms("timeout-ms")? {
        src = src.with_timeout(t)?;
    }
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

fn create_multicastsink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::UdpMulticastSink;
    let group = props.req_str("group")?;
    let port = props.req_u16("port")?;
    let mut sink = UdpMulticastSink::new(&group, port)?;
    if let Some(ttl) = props.get_u32("ttl")? {
        sink = sink.with_ttl(ttl)?;
    }
    if let Some(loopback) = props.get_bool("loopback")? {
        sink = sink.with_loopback(loopback)?;
    }
    Ok(DynAsyncElement::new_box(SinkAdapter::new(sink)))
}

#[cfg(feature = "http")]
fn create_httpsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::HttpSrc;
    let location = props.req_str("location")?;
    let mut src = HttpSrc::new(location)?;
    if let Some(size) = props.get_usize("chunk-size")? {
        src = src.with_chunk_size(size);
    }
    if let Some(t) = props.get_ms("timeout-ms")? {
        src = src.with_timeout(t);
    }
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

#[cfg(feature = "http")]
fn create_httpcachesrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::HttpCacheSrc;
    let location = props.req_str("location")?;
    let mut src = HttpCacheSrc::new(location)?;
    if let Some(path) = props.get_str("cache-file") {
        src = src.with_cache_file(path)?;
    }
    if let Some(size) = props.get_usize("chunk-size")? {
        src = src.with_chunk_size(size);
    }
    if let Some(t) = props.get_ms("timeout-ms")? {
        src = src.with_timeout(t);
    }
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

#[cfg(feature = "http")]
fn create_httpsink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::{HttpMethod, HttpSink};
    let location = props.req_str("location")?;
    let method = props
        .get_enum(
            "method",
            &[("post", HttpMethod::Post), ("put", HttpMethod::Put)],
        )?
        .unwrap_or(HttpMethod::Post);
    let mut sink = HttpSink::new(location, method)?;
    if let Some(ct) = props.get_str("content-type") {
        sink = sink.with_content_type(ct);
    }
    if let Some(t) = props.get_ms("timeout-ms")? {
        sink = sink.with_timeout(t);
    }
    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(sink)))
}

#[cfg(feature = "websocket")]
fn create_wssrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::WebSocketSrc;
    let url = props.req_str("url")?;
    let src = WebSocketSrc::new(url)?;
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

#[cfg(feature = "websocket")]
fn create_wssink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::{WebSocketSink, WsMessageType};
    let url = props.req_str("url")?;
    let mut sink = WebSocketSink::new(url)?;
    if let Some(mt) = props.get_enum(
        "mode",
        &[
            ("binary", WsMessageType::Binary),
            ("text", WsMessageType::Text),
        ],
    )? {
        sink = sink.with_message_type(mt);
    }
    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(sink)))
}
