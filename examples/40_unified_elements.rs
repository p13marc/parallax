//! Example 40: Unified Pipeline Elements
//!
//! This example demonstrates the new unified element system (Plan 05).
//! It shows how to use `SimpleSource`, `SimpleSink`, and `SimpleTransform`
//! traits with the wrapper types `Src`, `Snk`, and `Xfm`.
//!
//! # Concepts Demonstrated
//!
//! - SimpleSource trait for sources
//! - SimpleSink trait for sinks
//! - SimpleTransform trait for transforms
//! - Src, Snk, Xfm wrapper types
//! - Pipeline.add_element() method
//! - ProcessOutput unified output type
//!
//! # Running
//!
//! ```bash
//! cargo run --example 40_unified_elements
//! ```

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ProcessOutput, SimpleSink, SimpleSource, SimpleTransform, Snk, Src, Xfm};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;

// ============================================================================
// Simple Source: Counter
// ============================================================================

/// A source that produces numbered buffers.
struct CounterSource {
    count: u32,
    max: u32,
}

impl CounterSource {
    fn new(max: u32) -> Self {
        Self { count: 0, max }
    }
}

impl SimpleSource for CounterSource {
    fn produce(&mut self) -> Result<ProcessOutput> {
        if self.count >= self.max {
            return Ok(ProcessOutput::Eos);
        }

        self.count += 1;

        // Create a buffer with the count
        let segment = Arc::new(HeapSegment::new(64).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(self.count.to_le_bytes().as_ptr(), ptr, 4);
        }
        let handle = MemoryHandle::from_segment_with_len(segment, 4);
        let buffer = Buffer::new(handle, Metadata::from_sequence(self.count as u64));

        Ok(ProcessOutput::buffer(buffer))
    }

    fn name(&self) -> &str {
        "CounterSource"
    }
}

// ============================================================================
// Simple Transform: Doubler
// ============================================================================

/// A transform that doubles the value in each buffer.
struct DoublerTransform;

impl SimpleTransform for DoublerTransform {
    fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput> {
        // Read the value
        let data = buffer.as_bytes();
        if data.len() >= 4 {
            let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let doubled = value * 2;

            // Create new buffer with doubled value
            let segment = Arc::new(HeapSegment::new(64).unwrap());
            let ptr = segment.as_mut_ptr().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(doubled.to_le_bytes().as_ptr(), ptr, 4);
            }
            let handle = MemoryHandle::from_segment_with_len(segment, 4);
            let mut metadata = buffer.metadata().clone();
            metadata.sequence = doubled as u64; // Store doubled value in sequence for easy checking
            let new_buffer = Buffer::new(handle, metadata);

            Ok(ProcessOutput::buffer(new_buffer))
        } else {
            // Pass through if too small
            Ok(ProcessOutput::buffer(buffer))
        }
    }

    fn name(&self) -> &str {
        "DoublerTransform"
    }
}

// ============================================================================
// Simple Sink: Collector
// ============================================================================

/// A sink that collects received values.
struct CollectorSink {
    collected: Vec<u32>,
}

impl CollectorSink {
    fn new() -> Self {
        Self { collected: vec![] }
    }
}

impl SimpleSink for CollectorSink {
    fn consume(&mut self, buffer: &Buffer) -> Result<()> {
        let data = buffer.as_bytes();
        if data.len() >= 4 {
            let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            self.collected.push(value);
            println!(
                "  Received: {} (sequence: {})",
                value,
                buffer.metadata().sequence
            );
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "CollectorSink"
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    println!("=== Example 40: Unified Pipeline Elements ===\n");

    // ========================================================================
    // Part 1: Direct usage of wrapper types
    // ========================================================================
    println!("--- Part 1: Direct Wrapper Usage ---\n");

    // Create a source using Src wrapper
    let mut src = Src(CounterSource::new(5));

    println!("Producing from CounterSource:");
    loop {
        let output = src.0.produce()?;
        match output {
            ProcessOutput::Buffer(buf) => {
                let data = buf.as_bytes();
                let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                println!("  Produced: {}", value);
            }
            ProcessOutput::Eos => {
                println!("  End of stream");
                break;
            }
            _ => {}
        }
    }

    println!();

    // ========================================================================
    // Part 2: Using with Pipeline.add_element()
    // ========================================================================
    println!("--- Part 2: Pipeline with add_element() ---\n");

    let mut pipeline = Pipeline::new();

    // Add elements using the new add_element() method
    let src = pipeline.add_element("src", Src(CounterSource::new(5)));
    let transform = pipeline.add_element("doubler", Xfm(DoublerTransform));
    let sink = pipeline.add_element("sink", Snk(CollectorSink::new()));

    // Link them
    pipeline.link(src, transform)?;
    pipeline.link(transform, sink)?;

    println!("Pipeline structure:");
    println!(
        "  {} -> {} -> {}",
        pipeline.get_node(src).unwrap().name(),
        pipeline.get_node(transform).unwrap().name(),
        pipeline.get_node(sink).unwrap().name(),
    );
    println!();

    // Validate the pipeline
    pipeline.prepare()?;
    println!("Pipeline prepared successfully!");

    // Note: Running would require the executor, which is a more complex test
    // For now, we just demonstrate the API and structure

    println!();

    // ========================================================================
    // Part 3: Multi-output transform
    // ========================================================================
    println!("--- Part 3: Multi-Output Transform ---\n");

    /// A transform that splits one buffer into two
    struct SplitterTransform;

    impl SimpleTransform for SplitterTransform {
        fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput> {
            let data = buffer.as_bytes();
            if data.len() >= 4 {
                let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);

                // Create two buffers: value and value + 100
                let mut buffers = vec![];

                for offset in [0u32, 100] {
                    let new_value = value + offset;
                    let segment = Arc::new(HeapSegment::new(64).unwrap());
                    let ptr = segment.as_mut_ptr().unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(new_value.to_le_bytes().as_ptr(), ptr, 4);
                    }
                    let handle = MemoryHandle::from_segment_with_len(segment, 4);
                    let metadata = Metadata::from_sequence(new_value as u64);
                    buffers.push(Buffer::new(handle, metadata));
                }

                Ok(ProcessOutput::multiple(buffers))
            } else {
                Ok(ProcessOutput::buffer(buffer))
            }
        }

        fn name(&self) -> &str {
            "SplitterTransform"
        }
    }

    let mut xfm = Xfm(SplitterTransform);

    // Create an input buffer
    let segment = Arc::new(HeapSegment::new(64).unwrap());
    let ptr = segment.as_mut_ptr().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(42u32.to_le_bytes().as_ptr(), ptr, 4);
    }
    let handle = MemoryHandle::from_segment_with_len(segment, 4);
    let input = Buffer::new(handle, Metadata::from_sequence(42));

    // Transform it
    let output = xfm.0.transform(input)?;
    println!("Input: 42");
    println!("Output (multi-buffer):");
    for buf in output {
        let data = buf.as_bytes();
        let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        println!("  - {}", value);
    }

    println!();

    // ========================================================================
    // Part 4: Flush behavior
    // ========================================================================
    println!("--- Part 4: Buffering Transform with Flush ---\n");

    /// A transform that buffers one input and outputs the previous
    struct BufferingTransform {
        buffered: Option<Buffer>,
    }

    impl SimpleTransform for BufferingTransform {
        fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput> {
            let prev = self.buffered.take();
            self.buffered = Some(buffer);
            Ok(prev.into())
        }

        fn flush(&mut self) -> Result<ProcessOutput> {
            Ok(self.buffered.take().into())
        }

        fn name(&self) -> &str {
            "BufferingTransform"
        }
    }

    let mut xfm = Xfm(BufferingTransform { buffered: None });

    println!("Buffering transform demonstration:");

    // First input - no output yet
    let buf1 = {
        let segment = Arc::new(HeapSegment::new(64).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(1u32.to_le_bytes().as_ptr(), ptr, 4);
        }
        Buffer::new(
            MemoryHandle::from_segment_with_len(segment, 4),
            Metadata::from_sequence(1),
        )
    };
    let out = xfm.0.transform(buf1)?;
    println!(
        "  Input 1 -> Output: {:?}",
        if out.is_none() { "None" } else { "Buffer" }
    );

    // Second input - outputs first
    let buf2 = {
        let segment = Arc::new(HeapSegment::new(64).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(2u32.to_le_bytes().as_ptr(), ptr, 4);
        }
        Buffer::new(
            MemoryHandle::from_segment_with_len(segment, 4),
            Metadata::from_sequence(2),
        )
    };
    let out = xfm.0.transform(buf2)?;
    if let Some(buf) = out.into_buffer() {
        let value = u32::from_le_bytes([
            buf.as_bytes()[0],
            buf.as_bytes()[1],
            buf.as_bytes()[2],
            buf.as_bytes()[3],
        ]);
        println!("  Input 2 -> Output: {}", value);
    }

    // Flush - outputs second
    let out = xfm.0.flush()?;
    if let Some(buf) = out.into_buffer() {
        let value = u32::from_le_bytes([
            buf.as_bytes()[0],
            buf.as_bytes()[1],
            buf.as_bytes()[2],
            buf.as_bytes()[3],
        ]);
        println!("  Flush -> Output: {}", value);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
