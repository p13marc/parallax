//! Manual IPC with IpcSink and IpcSrc for explicit cross-process boundaries.
//!
//! Run with: cargo run --example 14_ipc_manual

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::elements::{IpcSink, IpcSrc};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

fn main() -> Result<()> {
    println!("Manual IPC: explicit IpcSink/IpcSrc connection");
    println!();

    let dir = tempdir()?;
    let socket_path = dir.path().join("pipeline.sock");
    let socket_path2 = socket_path.clone();

    // Consumer thread (IpcSrc - server, listens first)
    let consumer = thread::spawn(move || -> Result<u64> {
        let mut src = IpcSrc::new(&socket_path);
        let mut count = 0u64;
        let mut ctx = ProduceContext::without_buffer();

        loop {
            match src.produce(&mut ctx)? {
                ProduceResult::OwnBuffer(_buffer) => {
                    count += 1;
                    println!("Received buffer {}", count);
                }
                ProduceResult::Eos => break,
                _ => {}
            }
        }
        Ok(count)
    });

    // Give server time to start
    thread::sleep(Duration::from_millis(100));

    // Producer thread (IpcSink - client, connects)
    let producer = thread::spawn(move || -> Result<()> {
        let mut sink = IpcSink::connect(&socket_path2);

        for i in 0..5u64 {
            let segment = Arc::new(HeapSegment::new(8)?);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    i.to_le_bytes().as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    8,
                );
            }
            let buffer = Buffer::new(
                MemoryHandle::from_segment(segment),
                Metadata::from_sequence(i),
            );
            let ctx = ConsumeContext::new(&buffer);
            sink.consume(&ctx)?;
            println!("Sent buffer {}", i + 1);
        }
        Ok(())
    });

    producer.join().unwrap()?;

    // Give consumer time to process
    thread::sleep(Duration::from_millis(100));

    println!();
    println!("IPC transfer complete");

    Ok(())
}
