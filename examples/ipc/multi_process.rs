//! Multi-process pipeline example using shared memory IPC.
//!
//! This example demonstrates how to set up inter-process communication
//! using shared memory and Unix sockets for zero-copy buffer passing.
//!
//! The example runs in a single process but demonstrates the IPC primitives
//! that would be used for true multi-process communication.
//!
//! Run with: cargo run --example multi_process

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::error::Result;
use parallax::link::LocalLink;
use parallax::memory::{MemoryPool, MemorySegment, SharedMemorySegment};
use parallax::metadata::Metadata;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    println!("=== Multi-Process IPC Example ===\n");

    // Example 1: Local channels (in-process)
    println!("1. Local channel communication (in-process)");
    run_local_channel_example()?;

    // Example 2: Shared memory with pool
    println!("\n2. Shared memory pool example");
    run_shared_memory_example()?;

    // Example 3: Shared memory buffers across threads
    println!("\n3. Shared memory buffers between threads");
    run_shared_buffer_example()?;

    println!("\n=== All IPC examples completed ===");
    Ok(())
}

fn run_local_channel_example() -> Result<()> {
    // Create a bounded local channel
    let (sender, receiver) = LocalLink::bounded(16);

    // Spawn a producer thread
    let producer = thread::spawn(move || {
        for i in 0..5 {
            let segment =
                Arc::new(parallax::memory::HeapSegment::new(64).expect("failed to create segment"));
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(i));

            println!("   [Producer] Sending buffer {}", i);
            sender.send(buffer).expect("send failed");
            thread::sleep(Duration::from_millis(10));
        }
        println!("   [Producer] Done");
    });

    // Spawn a consumer thread
    let consumer = thread::spawn(move || {
        let mut count = 0;
        while let Some(buffer) = receiver.recv() {
            println!(
                "   [Consumer] Received buffer seq={}",
                buffer.metadata().sequence
            );
            count += 1;
            if count >= 5 {
                break;
            }
        }
        println!("   [Consumer] Done, received {} buffers", count);
    });

    producer.join().expect("producer panicked");
    consumer.join().expect("consumer panicked");

    Ok(())
}

fn run_shared_memory_example() -> Result<()> {
    // Create a shared memory segment
    let segment_name = format!("parallax-example-{}", std::process::id());
    let segment_size = 1024 * 1024; // 1MB
    let slot_size = 4096; // 4KB per buffer

    println!("   Creating shared memory segment: {}", segment_name);
    let segment = SharedMemorySegment::new(&segment_name, segment_size)?;
    println!("   Segment created, size: {} bytes", segment.len());

    // Create a memory pool from the segment
    let pool = Arc::new(MemoryPool::new(segment, slot_size)?);
    println!(
        "   Memory pool created with {} slots of {} bytes",
        pool.capacity(),
        slot_size
    );

    // Demonstrate loan/return cycle
    println!("\n   Demonstrating pool loan/return:");

    let mut slots = Vec::new();
    for i in 0..5 {
        match pool.loan() {
            Some(slot) => {
                println!("      Loaned slot {}, available: {}", i, pool.available());
                slots.push(slot);
            }
            None => {
                println!("      Pool exhausted at slot {}", i);
                break;
            }
        }
    }

    println!("   Returning slots...");
    slots.clear(); // Slots are returned on drop

    println!("   After return, available: {}", pool.available());

    Ok(())
}

fn run_shared_buffer_example() -> Result<()> {
    // Create a shared memory segment for buffers that can be passed between processes
    let segment_name = format!("parallax-buffer-{}", std::process::id());
    let segment = Arc::new(SharedMemorySegment::new(&segment_name, 64 * 1024)?);

    println!("   Created shared memory segment: {}", segment_name);
    println!("   Segment size: {} bytes", segment.len());

    // Create buffers backed by shared memory
    let handle1 = MemoryHandle::new(segment.clone(), 0, 1024);
    let handle2 = MemoryHandle::new(segment.clone(), 1024, 1024);

    let buffer1 = Buffer::<()>::new(handle1, Metadata::from_sequence(1));
    let buffer2 = Buffer::<()>::new(handle2, Metadata::from_sequence(2));

    println!("   Created 2 buffers in shared memory");

    // Use local channel to pass buffers between threads
    let (tx, rx) = LocalLink::bounded(4);

    let sender = thread::spawn(move || {
        println!("   [Sender] Sending buffer 1 (seq=1)");
        tx.send(buffer1).unwrap();
        thread::sleep(Duration::from_millis(10));

        println!("   [Sender] Sending buffer 2 (seq=2)");
        tx.send(buffer2).unwrap();

        println!("   [Sender] Done");
    });

    let receiver = thread::spawn(move || {
        let mut received = Vec::new();
        while let Some(buf) = rx.recv() {
            println!(
                "   [Receiver] Got buffer: seq={}, len={}, memory_type={:?}",
                buf.metadata().sequence,
                buf.len(),
                buf.memory().memory_type()
            );
            received.push(buf.metadata().sequence);

            if received.len() >= 2 {
                break;
            }
        }
        println!("   [Receiver] Done, received sequences: {:?}", received);
    });

    sender.join().expect("sender panicked");
    receiver.join().expect("receiver panicked");

    Ok(())
}
