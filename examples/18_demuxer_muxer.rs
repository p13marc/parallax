//! Example 18: Demuxer and Muxer Elements
//!
//! This example demonstrates how to create and use demuxer and muxer elements
//! for splitting and combining data streams.
//!
//! - **Demuxer**: Takes a single input and routes buffers to multiple outputs
//!   based on content (e.g., splitting video and audio from a container)
//!
//! - **Muxer**: Takes multiple inputs and combines them into a single output
//!   (e.g., combining video and audio into a container)
//!
//! Run with: cargo run --example 18_demuxer_muxer

use parallax::buffer::Buffer;
use parallax::element::{
    AsyncElementDyn, Demuxer, DemuxerAdapter, Muxer, MuxerAdapter, MuxerInput, PadAddedCallback,
    PadId, RoutedOutput,
};
use parallax::error::Result;
use parallax::format::Caps;
use parallax::memory::HeapSegment;
use parallax::metadata::Metadata;
use std::sync::Arc;

/// A simple demuxer that routes buffers based on sequence number.
///
/// Even sequence numbers go to pad 0, odd to pad 1.
struct EvenOddDemuxer {
    outputs: Vec<(PadId, Caps)>,
    callback: Option<PadAddedCallback>,
}

impl EvenOddDemuxer {
    fn new() -> Self {
        Self {
            outputs: vec![
                (PadId(0), Caps::any()), // Even numbers
                (PadId(1), Caps::any()), // Odd numbers
            ],
            callback: None,
        }
    }
}

impl Demuxer for EvenOddDemuxer {
    fn demux(&mut self, buffer: Buffer) -> Result<RoutedOutput> {
        let seq = buffer.metadata().sequence;
        let pad = if seq % 2 == 0 {
            PadId(0) // Even
        } else {
            PadId(1) // Odd
        };
        println!(
            "  Demuxer: routing buffer seq={} to pad {}",
            seq,
            if pad.0 == 0 { "even" } else { "odd" }
        );
        Ok(RoutedOutput::single(pad, buffer))
    }

    fn outputs(&self) -> &[(PadId, Caps)] {
        &self.outputs
    }

    fn on_pad_added(&mut self, callback: PadAddedCallback) {
        self.callback = Some(callback);
    }

    fn name(&self) -> &str {
        "EvenOddDemuxer"
    }
}

/// A simple muxer that combines buffers from multiple inputs.
///
/// It passes through buffers and tracks which pad they came from.
struct SimpleMuxer {
    inputs: Vec<(PadId, Caps)>,
    callback: Option<PadAddedCallback>,
    buffer_count: u32,
}

impl SimpleMuxer {
    fn new() -> Self {
        Self {
            inputs: vec![
                (PadId(0), Caps::any()), // Input 0
                (PadId(1), Caps::any()), // Input 1
            ],
            callback: None,
            buffer_count: 0,
        }
    }
}

impl Muxer for SimpleMuxer {
    fn mux(&mut self, input: MuxerInput) -> Result<Option<Buffer>> {
        self.buffer_count += 1;
        println!(
            "  Muxer: received buffer seq={} from pad {} (total: {})",
            input.buffer.metadata().sequence,
            input.pad.0,
            self.buffer_count
        );
        // Pass through the buffer
        Ok(Some(input.buffer))
    }

    fn inputs(&self) -> &[(PadId, Caps)] {
        &self.inputs
    }

    fn on_pad_added(&mut self, callback: PadAddedCallback) {
        self.callback = Some(callback);
    }

    fn name(&self) -> &str {
        "SimpleMuxer"
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        println!(
            "  Muxer: flushing (processed {} buffers)",
            self.buffer_count
        );
        Ok(None)
    }
}

fn main() {
    println!("=== Demuxer and Muxer Example ===\n");

    // Create test buffers
    let buffers: Vec<Buffer> = (0..6)
        .map(|i| {
            let segment = Arc::new(HeapSegment::new(8).unwrap());
            let handle = parallax::buffer::MemoryHandle::from_segment(segment);
            Buffer::new(handle, Metadata::from_sequence(i))
        })
        .collect();

    // Test Demuxer
    println!("1. Testing EvenOddDemuxer:");
    println!("   Routing buffers 0-5 based on even/odd sequence numbers\n");

    let demuxer = EvenOddDemuxer::new();
    let mut demuxer_adapter = DemuxerAdapter::new(demuxer);

    println!(
        "   Demuxer outputs: {:?}",
        demuxer_adapter.inner().outputs()
    );
    println!();

    for buffer in &buffers {
        // Create a fresh buffer for each iteration (buffers are consumed)
        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = parallax::buffer::MemoryHandle::from_segment(segment);
        let buf = Buffer::new(handle, Metadata::from_sequence(buffer.metadata().sequence));

        // Use the DynAsyncElement trait via blocking call
        let result = futures::executor::block_on(async {
            AsyncElementDyn::process(&mut demuxer_adapter, Some(buf)).await
        });

        if let Ok(Some(out_buf)) = result {
            println!("   -> Output buffer seq={}", out_buf.metadata().sequence);
        }
    }

    println!();

    // Test Muxer
    println!("2. Testing SimpleMuxer:");
    println!("   Combining buffers from multiple input pads\n");

    let muxer = SimpleMuxer::new();
    let mut muxer_adapter = MuxerAdapter::new(muxer);

    println!("   Muxer inputs: {:?}", muxer_adapter.inner().inputs());
    println!();

    for buffer in &buffers {
        // Create a fresh buffer for each iteration
        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = parallax::buffer::MemoryHandle::from_segment(segment);
        let buf = Buffer::new(handle, Metadata::from_sequence(buffer.metadata().sequence));

        let result = futures::executor::block_on(async {
            AsyncElementDyn::process(&mut muxer_adapter, Some(buf)).await
        });

        if let Ok(Some(out_buf)) = result {
            println!("   -> Output buffer seq={}", out_buf.metadata().sequence);
        }
    }

    // Flush the muxer (signal EOS)
    println!();
    let _ = futures::executor::block_on(async {
        AsyncElementDyn::process(&mut muxer_adapter, None).await
    });

    println!();
    println!("=== Example Complete ===");
    println!();
    println!("Key concepts demonstrated:");
    println!("  - Demuxer trait: single input -> multiple outputs (routed)");
    println!("  - Muxer trait: multiple inputs -> single output (combined)");
    println!("  - PadId: identifies input/output pads");
    println!("  - RoutedOutput: associates buffers with destination pads");
    println!("  - MuxerInput: associates buffers with source pads");
    println!("  - DemuxerAdapter/MuxerAdapter: integrate with pipeline executor");
}
