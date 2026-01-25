//! Examples demonstrating Tier 3 network elements.
//!
//! This example showcases:
//! - Unix domain socket source/sink
//! - UDP multicast source/sink
//!
//! Note: HTTP and WebSocket elements require feature flags.
//! Note: Zenoh elements require the `zenoh` feature flag.
//!
//! Run with: cargo run --example tier3_network_elements

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{Sink, Source};
use parallax::elements::{UdpMulticastSink, UdpMulticastSrc, UnixSink, UnixSrc};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

fn main() -> Result<()> {
    println!("=== Tier 3 Network Elements Examples ===\n");

    println!("1. Unix Domain Sockets");
    example_unix_sockets()?;

    println!("\n2. UDP Multicast");
    example_udp_multicast()?;

    println!("\n=== All examples completed ===");
    Ok(())
}

fn make_buffer(data: &[u8], seq: u64) -> Buffer {
    let segment = Arc::new(HeapSegment::new(data.len().max(1)).unwrap());
    if !data.is_empty() {
        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }
    let handle = MemoryHandle::from_segment_with_len(segment, data.len());
    Buffer::new(handle, Metadata::with_sequence(seq))
}

/// Demonstrates Unix domain socket communication.
fn example_unix_sockets() -> Result<()> {
    let dir = tempdir().unwrap();
    let socket_path = dir.path().join("demo.sock");

    println!("   Socket path: {:?}", socket_path);

    // Start server in a thread
    let path_clone = socket_path.clone();
    let server = thread::spawn(move || -> Result<Vec<u8>> {
        println!("   Server: listening...");
        let mut src = UnixSrc::listen(&path_clone)?
            .with_buffer_size(1024)
            .with_name("unix-server");

        let mut received = Vec::new();
        // Read until we have enough data
        while received.len() < 11 {
            if let Some(buf) = src.produce()? {
                if buf.metadata().flags.eos {
                    break;
                }
                received.extend_from_slice(buf.as_bytes());
            }
        }
        println!("   Server: received {} bytes", received.len());
        Ok(received)
    });

    // Give server time to start
    thread::sleep(Duration::from_millis(100));

    // Client sends data
    println!("   Client: connecting...");
    let mut sink = UnixSink::connect(&socket_path)?.with_name("unix-client");

    sink.consume(make_buffer(b"Hello", 0))?;
    sink.consume(make_buffer(b" World", 1))?;
    println!("   Client: sent 2 buffers");

    // Wait for server
    let received = server.join().unwrap()?;
    println!("   Result: {:?}", String::from_utf8_lossy(&received));

    Ok(())
}

/// Demonstrates UDP multicast communication.
fn example_udp_multicast() -> Result<()> {
    let multicast_addr = "239.255.0.100";
    let port = 5100;

    println!("   Multicast group: {}:{}", multicast_addr, port);

    // Start receiver in a thread
    let receiver = thread::spawn(move || -> Result<Option<Vec<u8>>> {
        println!("   Receiver: joining multicast group...");
        let mut src = UdpMulticastSrc::new(multicast_addr, port)?
            .with_timeout(Duration::from_millis(500))?
            .with_name("multicast-receiver");

        // Wait for one datagram
        match src.produce()? {
            Some(buf) if !buf.metadata().flags.timeout => {
                println!("   Receiver: got {} bytes", buf.len());
                Ok(Some(buf.as_bytes().to_vec()))
            }
            _ => {
                println!("   Receiver: timeout (no data received)");
                Ok(None)
            }
        }
    });

    // Give receiver time to join group
    thread::sleep(Duration::from_millis(100));

    // Sender
    println!("   Sender: sending to multicast group...");
    let mut sink = UdpMulticastSink::new(multicast_addr, port)?
        .with_loopback(true)? // Enable loopback so we can receive our own packets
        .with_name("multicast-sender");

    sink.consume(make_buffer(b"Hello Multicast!", 0))?;
    println!("   Sender: sent datagram");

    let stats = sink.stats();
    println!(
        "   Sender stats: {} bytes, {} datagrams",
        stats.bytes_transferred, stats.datagrams
    );

    // Wait for receiver
    match receiver.join().unwrap()? {
        Some(data) => {
            println!("   Received: {:?}", String::from_utf8_lossy(&data));
        }
        None => {
            println!("   Note: Multicast may not work in all network configurations");
        }
    }

    Ok(())
}
