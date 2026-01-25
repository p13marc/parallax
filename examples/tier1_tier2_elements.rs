//! Examples demonstrating Tier 1 and Tier 2 pipeline elements.
//!
//! This example showcases:
//! - Identity for pass-through with callbacks
//! - MemorySrc/MemorySink for memory-based I/O
//! - Delay for timing control
//! - Metadata operations (SequenceNumber, Timestamper, MetadataInject)
//! - Buffer operations (BufferTrim, BufferSlice, BufferPad)
//! - Filtering (Filter, SampleFilter, MetadataFilter)
//! - Transform (Map, Chunk)
//! - Batching (Batch, Unbatch)
//! - Timing control (Timeout, Debounce, Throttle)
//!
//! Run with: cargo run --example tier1_tier2_elements

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{Element, Sink, Source};
use parallax::elements::{
    Batch, BufferPad, BufferSlice, BufferTrim, Chunk, Debounce, Delay, Filter, Identity, Map,
    MemorySink, MemorySrc, MetadataFilter, MetadataInject, SampleFilter, SequenceNumber, Throttle,
    Timeout, TimestampMode, Timestamper, Unbatch,
};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use std::sync::Arc;
use std::time::Duration;

fn main() -> Result<()> {
    println!("=== Tier 1 & Tier 2 Elements Examples ===\n");

    // Tier 1: Quick Wins
    println!("--- TIER 1: Quick Wins ---\n");

    println!("1. Identity - Pass-through with callbacks");
    example_identity()?;

    println!("\n2. MemorySrc/MemorySink - Memory-based I/O");
    example_memory_src_sink()?;

    println!("\n3. Delay - Fixed timing delay");
    example_delay()?;

    println!("\n4. SequenceNumber - Automatic sequence numbering");
    example_sequence_number()?;

    println!("\n5. Timestamper - Automatic timestamping");
    example_timestamper()?;

    println!("\n6. MetadataInject - Inject metadata fields");
    example_metadata_inject()?;

    println!("\n7. BufferTrim - Trim buffers to max size");
    example_buffer_trim()?;

    println!("\n8. BufferSlice - Extract slice from buffers");
    example_buffer_slice()?;

    println!("\n9. BufferPad - Pad buffers to min size");
    example_buffer_pad()?;

    // Tier 2: Core Functionality
    println!("\n--- TIER 2: Core Functionality ---\n");

    println!("10. Filter - Predicate-based filtering");
    example_filter()?;

    println!("\n11. SampleFilter - Sampling modes");
    example_sample_filter()?;

    println!("\n12. MetadataFilter - Filter by metadata");
    example_metadata_filter()?;

    println!("\n13. Map - Transform buffer data");
    example_map()?;

    println!("\n14. Chunk - Split into fixed chunks");
    example_chunk()?;

    println!("\n15. Batch/Unbatch - Aggregate and split");
    example_batch_unbatch()?;

    println!("\n16. Timeout - Fallback on inactivity");
    example_timeout()?;

    println!("\n17. Debounce - Suppress rapid bursts");
    example_debounce()?;

    println!("\n18. Throttle - Rate limiting");
    example_throttle()?;

    println!("\n=== All examples completed ===");
    Ok(())
}

fn make_buffer_with_data(data: &[u8], seq: u64) -> Buffer {
    let segment = Arc::new(HeapSegment::new(data.len().max(1)).unwrap());
    unsafe {
        let ptr = segment.as_mut_ptr().unwrap();
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
    let handle = MemoryHandle::from_segment_with_len(segment, data.len());
    Buffer::new(handle, Metadata::with_sequence(seq))
}

fn make_buffer(seq: u64) -> Buffer {
    make_buffer_with_data(&[1, 2, 3, 4, 5, 6, 7, 8], seq)
}

/// Demonstrates Identity element.
fn example_identity() -> Result<()> {
    let mut identity = Identity::new().with_name("debug_identity");

    for i in 0..3 {
        let buf = make_buffer(i);
        if let Some(out) = identity.process(buf)? {
            println!(
                "   Passed: buffer seq={}, len={}",
                out.metadata().sequence,
                out.len()
            );
        }
    }

    let stats = identity.stats();
    println!(
        "   Stats: {} buffers, {} bytes",
        stats.buffer_count, stats.byte_count
    );
    Ok(())
}

/// Demonstrates MemorySrc and MemorySink.
fn example_memory_src_sink() -> Result<()> {
    // Create a source from memory data
    let data = b"Hello, World! This is test data.".to_vec();
    let mut src = MemorySrc::new(data.clone())
        .with_chunk_size(10)
        .with_name("mem_source");

    // Create a sink to collect output
    let mut sink = MemorySink::new().with_name("mem_sink");

    // Process all buffers
    let mut count = 0;
    while let Some(buf) = src.produce()? {
        println!(
            "   Chunk {}: {:?}",
            count,
            String::from_utf8_lossy(buf.as_bytes())
        );
        sink.consume(buf)?;
        count += 1;
    }

    // Verify we got all data back
    let collected = sink.take_data();
    println!(
        "   Collected {} bytes: {:?}",
        collected.len(),
        String::from_utf8_lossy(&collected)
    );
    assert_eq!(collected, data);
    Ok(())
}

/// Demonstrates Delay element.
fn example_delay() -> Result<()> {
    let mut delay = Delay::new(Duration::from_millis(10)).with_name("fixed_delay");

    let start = std::time::Instant::now();
    for i in 0..3 {
        let buf = make_buffer(i);
        delay.process(buf)?;
    }
    let elapsed = start.elapsed();

    let stats = delay.stats();
    println!(
        "   Processed {} buffers with {}us total delay",
        stats.buffer_count, stats.total_delay_micros
    );
    println!("   Actual elapsed: {:?}", elapsed);
    Ok(())
}

/// Demonstrates SequenceNumber element.
fn example_sequence_number() -> Result<()> {
    let mut seq = SequenceNumber::starting_at(100)
        .with_increment(10)
        .with_name("seq_gen");

    for _ in 0..3 {
        let buf = make_buffer_with_data(b"data", 0);
        if let Some(out) = seq.process(buf)? {
            println!("   Buffer sequence: {}", out.metadata().sequence);
        }
    }

    println!("   Current counter: {}", seq.current());
    Ok(())
}

/// Demonstrates Timestamper element.
fn example_timestamper() -> Result<()> {
    let mut ts = Timestamper::new(TimestampMode::Monotonic).with_name("auto_ts");

    for i in 0..3 {
        let buf = make_buffer(i);
        if let Some(out) = ts.process(buf)? {
            let meta = out.metadata();
            println!(
                "   Buffer {}: pts={:?}, dts={:?}",
                meta.sequence, meta.pts, meta.dts
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    println!("   Timestamped {} buffers", ts.buffer_count());
    Ok(())
}

/// Demonstrates MetadataInject element.
fn example_metadata_inject() -> Result<()> {
    let mut inject = MetadataInject::new()
        .with_stream_id(42)
        .with_duration(Duration::from_millis(100))
        .with_offset(1000)
        .with_name("meta_inject");

    let buf = make_buffer(0);
    if let Some(out) = inject.process(buf)? {
        let meta = out.metadata();
        println!("   stream_id: {:?}", meta.stream_id);
        println!("   duration: {:?}", meta.duration);
        println!("   offset: {:?}", meta.offset);
    }
    Ok(())
}

/// Demonstrates BufferTrim element.
fn example_buffer_trim() -> Result<()> {
    let mut trim = BufferTrim::new(4).with_name("trimmer");

    let buf = make_buffer_with_data(b"Hello World", 0);
    println!("   Input: {} bytes", buf.len());

    if let Some(out) = trim.process(buf)? {
        println!("   Output: {} bytes -> {:?}", out.len(), out.as_bytes());
    }

    let stats = trim.stats();
    println!(
        "   Stats: {} trimmed, {} bytes removed",
        stats.trimmed_count, stats.bytes_trimmed
    );
    Ok(())
}

/// Demonstrates BufferSlice element.
fn example_buffer_slice() -> Result<()> {
    // Extract bytes from offset 2, length 4
    let mut slice = BufferSlice::new(2, 4).with_name("slicer");

    let buf = make_buffer_with_data(b"Hello World", 0);
    println!("   Input: {:?}", String::from_utf8_lossy(buf.as_bytes()));

    if let Some(out) = slice.process(buf)? {
        println!(
            "   Slice [2..6]: {:?}",
            String::from_utf8_lossy(out.as_bytes())
        );
    }
    Ok(())
}

/// Demonstrates BufferPad element.
fn example_buffer_pad() -> Result<()> {
    // Pad to minimum 8 bytes with 0x00 fill
    let mut pad = BufferPad::new(8, 0x00).with_name("padder");

    let buf = make_buffer_with_data(b"Hi", 0);
    println!("   Input: {} bytes -> {:?}", buf.len(), buf.as_bytes());

    if let Some(out) = pad.process(buf)? {
        println!("   Padded to {} bytes -> {:?}", out.len(), out.as_bytes());
    }
    Ok(())
}

/// Demonstrates Filter element.
fn example_filter() -> Result<()> {
    let mut filter = Filter::new(|buf: &Buffer| buf.len() > 5).with_name("size_filter");

    let inputs = [b"Hi".to_vec(), b"Hello World".to_vec(), b"Bye".to_vec()];
    for (i, data) in inputs.iter().enumerate() {
        let buf = make_buffer_with_data(data, i as u64);
        match filter.process(buf)? {
            Some(out) => println!("   Passed: {:?}", String::from_utf8_lossy(out.as_bytes())),
            None => println!("   Dropped: {:?}", String::from_utf8_lossy(data)),
        }
    }

    let stats = filter.stats();
    println!(
        "   Stats: {} passed, {} dropped",
        stats.passed, stats.dropped
    );
    Ok(())
}

/// Demonstrates SampleFilter modes.
fn example_sample_filter() -> Result<()> {
    println!("   EveryNth(2):");
    let mut sampler = SampleFilter::every_nth(2);
    for i in 0..6 {
        let buf = make_buffer(i);
        match sampler.process(buf)? {
            Some(out) => print!(" {} ", out.metadata().sequence),
            None => print!(" . "),
        }
    }
    println!();

    println!("   FirstN(3):");
    let mut sampler = SampleFilter::first_n(3);
    for i in 0..6 {
        let buf = make_buffer(i);
        match sampler.process(buf)? {
            Some(out) => print!(" {} ", out.metadata().sequence),
            None => print!(" . "),
        }
    }
    println!();
    Ok(())
}

/// Demonstrates MetadataFilter element.
fn example_metadata_filter() -> Result<()> {
    let mut filter = MetadataFilter::new()
        .with_sequence_range(2, 5)
        .with_name("seq_range_filter");

    print!("   Sequences 0-7, filter [2,5]: ");
    for i in 0..8 {
        let buf = make_buffer(i);
        match filter.process(buf)? {
            Some(out) => print!("{} ", out.metadata().sequence),
            None => print!(". "),
        }
    }
    println!();
    Ok(())
}

/// Demonstrates Map element.
fn example_map() -> Result<()> {
    let mut mapper = Map::new(|data: &[u8]| data.iter().map(|b| b.to_ascii_uppercase()).collect())
        .with_name("uppercase");

    let buf = make_buffer_with_data(b"hello world", 0);
    if let Some(out) = mapper.process(buf)? {
        println!("   Mapped: {:?}", String::from_utf8_lossy(out.as_bytes()));
    }

    println!("   Processed {} buffers", mapper.buffer_count());
    Ok(())
}

/// Demonstrates Chunk element.
fn example_chunk() -> Result<()> {
    let mut chunk = Chunk::new(4).with_name("chunker");

    let buf = make_buffer_with_data(b"Hello World!", 0);
    println!("   Input: {:?}", String::from_utf8_lossy(buf.as_bytes()));

    // Process returns all chunks at once using process_all
    print!("   Chunks: ");
    let chunks = chunk.process_all(buf)?;
    for c in &chunks {
        print!("{:?} ", String::from_utf8_lossy(c.as_bytes()));
    }

    // Get remaining data
    if let Ok(Some(c)) = chunk.flush() {
        print!("{:?} ", String::from_utf8_lossy(c.as_bytes()));
    }
    println!();
    Ok(())
}

/// Demonstrates Batch and Unbatch elements.
fn example_batch_unbatch() -> Result<()> {
    // Batch multiple buffers into one
    let mut batch = Batch::by_count(3).with_name("batcher");

    println!("   Batching 5 buffers (batch size 3):");
    for i in 0..5 {
        let buf = make_buffer_with_data(&[i as u8; 4], i);
        match batch.process(buf)? {
            Some(batched) => println!("   Batch ready: {} bytes", batched.len()),
            None => println!("   Buffer {} added to batch", i),
        }
    }

    // Flush remaining
    if let Ok(Some(batched)) = batch.flush() {
        println!("   Flush: {} bytes remaining", batched.len());
    }

    // Unbatch back to chunks
    println!("\n   Unbatching (chunk size 4):");
    let mut unbatch = Unbatch::new(4).with_name("unbatcher");
    let big_buf = make_buffer_with_data(b"AABBCCDD", 0);

    // Process all chunks
    let chunks = unbatch.process_all(big_buf)?;
    for chunk in chunks {
        println!("   Chunk: {:?}", String::from_utf8_lossy(chunk.as_bytes()));
    }
    Ok(())
}

/// Demonstrates Timeout element.
fn example_timeout() -> Result<()> {
    let mut timeout = Timeout::new(Duration::from_millis(50))
        .with_fallback(b"TIMEOUT".to_vec())
        .with_name("timeout_guard");

    // Process a buffer (resets timeout)
    let buf = make_buffer(0);
    if let Some(out) = timeout.process(buf)? {
        println!("   Normal buffer: seq={}", out.metadata().sequence);
    }

    // Wait for timeout
    std::thread::sleep(Duration::from_millis(60));

    // Check for timeout
    if let Some(fallback) = timeout.check_timeout()? {
        println!(
            "   Timeout triggered: {:?}",
            String::from_utf8_lossy(fallback.as_bytes())
        );
    }
    Ok(())
}

/// Demonstrates Debounce element.
fn example_debounce() -> Result<()> {
    let mut debounce = Debounce::new(Duration::from_millis(20)).with_name("debouncer");

    // Rapid bursts - only last one should pass after quiet period
    for i in 0..3 {
        let buf = make_buffer(i);
        match debounce.process(buf)? {
            Some(out) => println!("   Buffer {} passed immediately", out.metadata().sequence),
            None => println!("   Buffer {} held", i),
        }
        std::thread::sleep(Duration::from_millis(5)); // Rapid, within quiet period
    }

    // Wait for quiet period
    std::thread::sleep(Duration::from_millis(25));

    // Flush releases the held buffer
    if let Some(buf) = debounce.flush() {
        println!(
            "   After quiet period, released: seq={}",
            buf.metadata().sequence
        );
    }
    Ok(())
}

/// Demonstrates Throttle element.
fn example_throttle() -> Result<()> {
    let mut throttle = Throttle::new(Duration::from_millis(20)).with_name("throttler");

    print!("   Sending 5 buffers rapidly: ");
    for i in 0..5 {
        let buf = make_buffer(i);
        match throttle.process(buf)? {
            Some(out) => print!("{} ", out.metadata().sequence),
            None => print!(". "),
        }
    }
    println!();

    // Wait and send another
    std::thread::sleep(Duration::from_millis(25));
    let buf = make_buffer(5);
    if let Some(out) = throttle.process(buf)? {
        println!("   After waiting: {} passed", out.metadata().sequence);
    }

    let stats = throttle.stats();
    println!(
        "   Stats: {} passed, {} dropped",
        stats.passed, stats.dropped
    );
    Ok(())
}
