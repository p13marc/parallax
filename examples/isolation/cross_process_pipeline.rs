//! Cross-process pipeline example using IpcSink and IpcSrc.
//!
//! This example demonstrates how to manually connect pipelines across
//! processes using shared memory for zero-copy data transfer.
//!
//! # Architecture
//!
//! ```text
//! Process A (Producer)              Process B (Consumer)
//! ┌──────────┐   ┌─────────┐       ┌─────────┐   ┌──────────┐
//! │ TestSrc  │──▶│ IpcSink │═══════│ IpcSrc  │──▶│ Display  │
//! └──────────┘   └─────────┘       └─────────┘   └──────────┘
//!                     │                 │
//!                     └────────┬────────┘
//!                        Unix Socket
//!                    (control messages)
//!                              +
//!                     Shared Memory Arena
//!                       (data, zero-copy)
//! ```
//!
//! # How It Works
//!
//! 1. IpcSink creates a Unix socket and shared memory arena
//! 2. IpcSrc connects to the socket and receives the arena fd via SCM_RIGHTS
//! 3. Data is written to shared memory slots (no copying)
//! 4. Control messages notify the receiver of new data
//! 5. Receiver reads directly from shared memory (zero-copy)
//!
//! Run with: cargo run --example cross_process_pipeline

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    DynAsyncElement, Element, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter,
};
use parallax::elements::{IpcSink, IpcSrc};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::{Pipeline, PipelineExecutor};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

/// A source that produces numbered buffers with payload.
struct DataSource {
    current: u64,
    max: u64,
    payload_size: usize,
}

impl DataSource {
    fn new(count: u64, payload_size: usize) -> Self {
        Self {
            current: 0,
            max: count,
            payload_size,
        }
    }
}

impl Source for DataSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.current >= self.max {
            return Ok(None);
        }

        // Create payload with recognizable pattern
        let mut data = vec![0u8; self.payload_size];
        let seq_bytes = self.current.to_le_bytes();
        data[..8].copy_from_slice(&seq_bytes);
        // Fill rest with sequence number as pattern
        for (i, byte) in data[8..].iter_mut().enumerate() {
            *byte = ((self.current as usize + i) % 256) as u8;
        }

        let segment = Arc::new(HeapSegment::new(data.len())?);
        // Write data using raw pointer
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), segment.as_mut_ptr().unwrap(), data.len());
        }
        let handle = MemoryHandle::from_segment(segment);
        let metadata = Metadata::from_sequence(self.current);
        let buffer = Buffer::new(handle, metadata);
        self.current += 1;

        Ok(Some(buffer))
    }
}

/// A sink that verifies received data.
struct VerifyingSink {
    received: Arc<AtomicU64>,
    expected_size: usize,
    name: String,
}

impl VerifyingSink {
    fn new(name: &str, expected_size: usize, counter: Arc<AtomicU64>) -> Self {
        Self {
            received: counter,
            expected_size,
            name: name.to_string(),
        }
    }
}

impl Sink for VerifyingSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let data = buffer.as_bytes();

        // Verify size
        if data.len() != self.expected_size {
            eprintln!(
                "[{}] Size mismatch: expected {}, got {}",
                self.name,
                self.expected_size,
                data.len()
            );
        }

        // Verify sequence number from payload
        if data.len() >= 8 {
            let seq = u64::from_le_bytes(data[..8].try_into().unwrap());
            let expected_seq = self.received.load(Ordering::Relaxed);
            if seq != expected_seq {
                eprintln!(
                    "[{}] Sequence mismatch: expected {}, got {}",
                    self.name, expected_seq, seq
                );
            }
        }

        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// A transform that passes data through (simulates processing).
struct DataTransform {
    name: String,
    processed: u64,
}

impl DataTransform {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            processed: 0,
        }
    }
}

impl Element for DataTransform {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.processed += 1;
        // In a real transform, we might decode/encode the data
        Ok(Some(buffer))
    }
}

fn main() -> Result<()> {
    println!("=== Cross-Process Pipeline Example ===\n");

    // Example 1: Simulated cross-process (same process, demonstrates API)
    println!("1. IPC Elements in Single Process (API demo)");
    example_single_process_ipc()?;

    // Example 2: Multi-threaded simulation
    println!("\n2. Multi-threaded IPC Simulation");
    example_threaded_ipc()?;

    // Example 3: Pipeline with IPC boundary
    println!("\n3. Pipeline with IPC Boundary");
    tokio::runtime::Runtime::new()?.block_on(example_pipeline_with_ipc())?;

    println!("\n=== Cross-Process Pipeline Example Complete ===");
    Ok(())
}

fn example_single_process_ipc() -> Result<()> {
    let dir = tempdir()?;
    let socket_path = dir.path().join("test.sock");

    println!("   Socket path: {:?}", socket_path);
    println!("   Creating IpcSink (server) and IpcSrc (client)...");

    // Create sink (server side - creates socket and arena)
    let sink = IpcSink::new(&socket_path).with_max_pending(8);

    // Create src (client side - connects to socket)
    let src = IpcSrc::new(&socket_path);

    println!("   IpcSink: binds to socket, creates shared memory arena");
    println!("   IpcSrc: connects to socket, receives arena fd via SCM_RIGHTS");
    println!("   Data flows through shared memory - zero-copy!");

    // Note: Actually using these requires concurrent execution
    // since accept() blocks until connection. See threaded example.
    drop(sink);
    drop(src);

    Ok(())
}

fn example_threaded_ipc() -> Result<()> {
    let dir = tempdir()?;
    let socket_path = dir.path().join("ipc.sock");
    let socket_path_clone = socket_path.clone();

    let buffer_count = 10;
    let payload_size = 1024;

    let received = Arc::new(AtomicU64::new(0));
    let received_clone = received.clone();

    println!(
        "   Sending {} buffers of {} bytes each",
        buffer_count, payload_size
    );

    // Producer thread (IpcSink)
    let producer = thread::spawn(move || -> Result<()> {
        // Give consumer time to start listening
        thread::sleep(Duration::from_millis(50));

        let mut sink = IpcSink::connect(&socket_path_clone).with_max_pending(4);

        let mut source = DataSource::new(buffer_count, payload_size);

        while let Some(buffer) = source.produce()? {
            sink.consume(buffer)?;
        }

        println!("   [Producer] Sent {} buffers", buffer_count);
        Ok(())
    });

    // Consumer thread (IpcSrc)
    let consumer = thread::spawn(move || -> Result<()> {
        let mut src = IpcSrc::new(&socket_path);
        let mut sink = VerifyingSink::new("consumer", payload_size, received_clone);

        loop {
            match src.produce()? {
                Some(buffer) => sink.consume(buffer)?,
                None => break,
            }
        }

        Ok(())
    });

    // Wait for both threads with timeout
    let producer_result = producer.join();

    // Give consumer a moment to finish
    thread::sleep(Duration::from_millis(100));

    // Consumer might still be waiting for more data
    // In a real scenario, we'd signal EOS through the socket

    if producer_result.is_ok() {
        println!(
            "   [Consumer] Received {} buffers",
            received.load(Ordering::Relaxed)
        );
    }

    Ok(())
}

async fn example_pipeline_with_ipc() -> Result<()> {
    let buffer_count = 50;
    let payload_size = 256;

    let received = Arc::new(AtomicU64::new(0));
    let received_clone = received.clone();

    println!("   Building pipeline: DataSource -> Transform -> Sink");
    println!("   (In production, Transform could be in a separate process)");

    // Build a pipeline that demonstrates the flow
    let mut pipeline = Pipeline::new();

    // Source
    let src = pipeline.add_node(
        "data_source",
        DynAsyncElement::new_box(SourceAdapter::new(DataSource::new(
            buffer_count,
            payload_size,
        ))),
    );

    // Transform (would be isolated in production)
    let transform = pipeline.add_node(
        "data_transform",
        DynAsyncElement::new_box(ElementAdapter::new(DataTransform::new("transform"))),
    );

    // Reverse transform (undo XOR)
    let reverse = pipeline.add_node(
        "reverse_transform",
        DynAsyncElement::new_box(ElementAdapter::new(DataTransform::new("reverse"))),
    );

    // Sink
    let sink = pipeline.add_node(
        "verify_sink",
        DynAsyncElement::new_box(SinkAdapter::new(VerifyingSink::new(
            "sink",
            payload_size,
            received_clone,
        ))),
    );

    // Link: src -> transform -> reverse -> sink
    pipeline.link(src, transform)?;
    pipeline.link(transform, reverse)?;
    pipeline.link(reverse, sink)?;

    // Run pipeline
    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await?;

    println!(
        "   Processed {} buffers through pipeline",
        received.load(Ordering::Relaxed)
    );

    // Demonstrate isolation API
    println!("\n   With isolation, the same pipeline could run as:");
    println!("   pipeline.run_isolating(vec![\"*transform*\"]).await?");
    println!("   -> transform elements would run in separate processes");
    println!("   -> IpcSink/IpcSrc injected automatically at boundaries");

    Ok(())
}
