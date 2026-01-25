//! Examples demonstrating the new pipeline elements.
//!
//! This example showcases:
//! - AppSrc/AppSink for application integration
//! - Queue with backpressure and leaky modes
//! - Valve for flow control
//! - Funnel for merging streams
//! - InputSelector/OutputSelector for routing
//! - TestSrc for test pattern generation
//! - Concat for sequential stream concatenation
//!
//! Run with: cargo run --example new_elements

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{Element, Sink, Source};
use parallax::elements::{
    AppSink, AppSrc, Concat, Funnel, InputSelector, LeakyMode, OutputSelector, Queue, TestPattern,
    TestSrc, Valve,
};
use parallax::error::Result;
use parallax::memory::HeapSegment;
use parallax::metadata::Metadata;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() -> Result<()> {
    println!("=== New Elements Examples ===\n");

    // Example 1: AppSrc/AppSink
    println!("1. AppSrc/AppSink - Application integration");
    example_app_src_sink()?;

    // Example 2: Queue with backpressure
    println!("\n2. Queue - Buffering with backpressure");
    example_queue()?;

    // Example 3: Valve for flow control
    println!("\n3. Valve - Flow control switch");
    example_valve()?;

    // Example 4: Funnel for merging streams
    println!("\n4. Funnel - Merge multiple streams");
    example_funnel()?;

    // Example 5: Selectors for routing
    println!("\n5. InputSelector/OutputSelector - Stream routing");
    example_selectors()?;

    // Example 6: TestSrc patterns
    println!("\n6. TestSrc - Test pattern generation");
    example_testsrc()?;

    // Example 7: Concat for sequential streams
    println!("\n7. Concat - Sequential stream concatenation");
    example_concat()?;

    println!("\n=== All examples completed ===");
    Ok(())
}

fn make_buffer(seq: u64) -> Buffer {
    let segment = Arc::new(HeapSegment::new(64).unwrap());
    let handle = MemoryHandle::from_segment_with_len(segment, 64);
    Buffer::new(handle, Metadata::with_sequence(seq))
}

/// Demonstrates AppSrc for injecting data and AppSink for extracting data.
fn example_app_src_sink() -> Result<()> {
    // Create an AppSrc with a bounded queue of 10 buffers
    let mut source = AppSrc::with_max_buffers(10).with_name("app_source");
    let src_handle = source.handle();

    // Create an AppSink to receive processed buffers
    let mut sink = AppSink::with_max_buffers(10).with_name("app_sink");
    let sink_handle = sink.handle();

    // Spawn a producer thread that pushes data into the pipeline
    let producer = thread::spawn(move || {
        for i in 0..5 {
            println!("   Producer: pushing buffer {}", i);
            src_handle.push_buffer(make_buffer(i)).unwrap();
        }
        src_handle.end_stream();
        println!("   Producer: stream ended");
    });

    // Pass buffers through in the main thread (simulates pipeline processing)
    while let Some(buffer) = source.produce()? {
        sink.consume(buffer)?;
    }
    // Signal EOS to the sink so consumers know we're done
    sink.send_eos();

    // Consumer pulls data from the sink
    let consumer = thread::spawn(move || {
        let mut count = 0;
        while let Ok(Some(buffer)) = sink_handle.pull_buffer() {
            println!(
                "   Consumer: received buffer seq={}",
                buffer.metadata().sequence
            );
            count += 1;
        }
        println!("   Consumer: received {} buffers total", count);
    });

    producer.join().unwrap();
    consumer.join().unwrap();

    Ok(())
}

/// Demonstrates Queue with backpressure and leaky modes.
fn example_queue() -> Result<()> {
    // Create a queue with max 3 buffers, leaky upstream mode
    let queue = Queue::new(3)
        .leaky(LeakyMode::Upstream)
        .with_name("bounded_queue");
    let queue_clone = queue.clone();

    // Producer sends 5 buffers quickly
    let producer = thread::spawn(move || {
        for i in 0..5u64 {
            match queue_clone.push(make_buffer(i)) {
                Ok(()) => println!("   Queue: pushed buffer {}", i),
                Err(_) => println!("   Queue: buffer {} error", i),
            }
        }
        queue_clone.set_flushing(true);
    });

    // Small delay to let producer fill the queue
    thread::sleep(Duration::from_millis(10));

    // Consumer reads from queue
    let mut count = 0;
    while let Ok(Some(buffer)) = queue.pop_timeout(Some(Duration::from_millis(100))) {
        println!(
            "   Queue: consumed buffer seq={}",
            buffer.metadata().sequence
        );
        count += 1;
    }
    println!("   Queue: consumed {} buffers total", count);

    let stats = queue.stats();
    println!(
        "   Queue stats: pushed={}, popped={}, dropped={}",
        stats.total_pushed, stats.total_popped, stats.total_dropped
    );

    producer.join().unwrap();
    Ok(())
}

/// Demonstrates Valve for flow control.
fn example_valve() -> Result<()> {
    let mut valve = Valve::new().with_name("flow_valve");
    let control = valve.control();

    // Valve is open by default
    println!("   Valve open:");
    for i in 0..3 {
        let buf = make_buffer(i);
        match valve.process(buf)? {
            Some(b) => println!("      Buffer {} passed", b.metadata().sequence),
            None => println!("      Buffer {} dropped", i),
        }
    }

    // Close the valve via the control handle
    control.close();
    println!("   Valve closed:");
    for i in 3..6 {
        let buf = make_buffer(i);
        match valve.process(buf)? {
            Some(b) => println!("      Buffer {} passed", b.metadata().sequence),
            None => println!("      Buffer {} dropped", i),
        }
    }

    // Reopen via control
    control.open();
    println!("   Valve reopened:");
    let buf = make_buffer(6);
    match valve.process(buf)? {
        Some(b) => println!("      Buffer {} passed", b.metadata().sequence),
        None => println!("      Buffer 6 dropped"),
    }

    let stats = valve.stats();
    println!(
        "   Stats: passed={}, dropped={}",
        stats.passed, stats.dropped
    );

    Ok(())
}

/// Demonstrates Funnel for merging multiple streams.
fn example_funnel() -> Result<()> {
    let mut funnel = Funnel::new().with_name("merger");

    // Create input handles
    let input0 = funnel.new_input();
    let input1 = funnel.new_input();

    // Push from "different sources"
    let producer0 = thread::spawn(move || {
        for i in 0..2u64 {
            input0.push(make_buffer(i)).unwrap();
            println!("   Input 0: pushed buffer {}", i);
        }
        input0.end_stream();
    });

    let producer1 = thread::spawn(move || {
        for i in 100..102u64 {
            input1.push(make_buffer(i)).unwrap();
            println!("   Input 1: pushed buffer {}", i);
        }
        input1.end_stream();
    });

    producer0.join().unwrap();
    producer1.join().unwrap();

    // Pull merged output
    println!("   Merged output:");
    while let Some(buffer) = funnel.produce()? {
        println!("      seq={}", buffer.metadata().sequence);
    }

    let stats = funnel.stats();
    println!(
        "   Funnel: {} inputs, {} buffers merged",
        stats.input_count, stats.total_produced
    );

    Ok(())
}

/// Demonstrates InputSelector and OutputSelector for routing.
fn example_selectors() -> Result<()> {
    // InputSelector: Choose from multiple inputs (N-to-1)
    println!("   InputSelector (N-to-1):");
    {
        let selector = InputSelector::new().with_name("input_sel");

        let input0 = selector.new_input();
        let input1 = selector.new_input();

        // Push to both inputs
        input0.push(make_buffer(0))?;
        input1.push(make_buffer(100))?;

        // Default selects input 0
        if let Some(buf) = selector.pull_timeout(Some(Duration::from_millis(10)))? {
            println!("      Selected input 0: seq={}", buf.metadata().sequence);
        }

        // Switch to input 1
        selector.select(1);
        input1.push(make_buffer(101))?;

        if let Some(buf) = selector.pull_timeout(Some(Duration::from_millis(10)))? {
            println!("      Selected input 1: seq={}", buf.metadata().sequence);
        }
    }

    // OutputSelector: Route to multiple outputs (1-to-N)
    println!("   OutputSelector (1-to-N):");
    {
        let mut selector = OutputSelector::new().with_name("output_sel");

        let output0 = selector.new_output();
        let output1 = selector.new_output();

        // Send to output 0 (default)
        selector.process(make_buffer(0))?;
        if let Some(buf) = output0.try_pull() {
            println!("      Routed to output 0: seq={}", buf.metadata().sequence);
        }

        // Switch to output 1
        selector.select(1);
        selector.process(make_buffer(1))?;
        if let Some(buf) = output1.try_pull() {
            println!("      Routed to output 1: seq={}", buf.metadata().sequence);
        }
    }

    Ok(())
}

/// Demonstrates TestSrc with various patterns.
fn example_testsrc() -> Result<()> {
    let patterns = [
        (TestPattern::Zero, "Zero"),
        (TestPattern::Counter, "Counter"),
        (TestPattern::Alternating, "Alternating"),
    ];

    for (pattern, name) in patterns {
        let mut src = TestSrc::new()
            .with_pattern(pattern)
            .with_buffer_size(8)
            .with_num_buffers(3);

        print!("   Pattern '{}': ", name);
        let mut first = true;
        while let Some(buffer) = src.produce()? {
            if !first {
                print!(", ");
            }
            first = false;
            // Show first 4 bytes
            let bytes = buffer.as_bytes();
            print!("{:?}", &bytes[..4.min(bytes.len())]);
        }
        println!();
    }

    Ok(())
}

/// Demonstrates Concat for sequential stream concatenation.
fn example_concat() -> Result<()> {
    let mut concat = Concat::new().with_name("sequential");

    // Add two streams
    let stream0 = concat.add_stream();
    let stream1 = concat.add_stream();

    // Fill stream 0
    for i in 0..2u64 {
        stream0.push(make_buffer(i))?;
    }
    stream0.end_stream();
    println!("   Stream 0: 2 buffers pushed, ended");

    // Fill stream 1
    for i in 100..102u64 {
        stream1.push(make_buffer(i))?;
    }
    stream1.end_stream();
    println!("   Stream 1: 2 buffers pushed, ended");

    // Read concatenated output - stream 0 first, then stream 1
    println!("   Concat output (sequential):");
    while let Some(buffer) = concat.produce()? {
        println!("      seq={}", buffer.metadata().sequence);
    }

    let stats = concat.stats();
    println!(
        "   Concat: {} streams, {} buffers produced",
        stats.stream_count, stats.total_produced
    );

    Ok(())
}
