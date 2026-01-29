//! Flow Control and Backpressure Example
//!
//! This example demonstrates Parallax's backpressure system:
//! - Queue with water marks for flow control
//! - FlowStateHandle for cross-element signaling
//! - Source that respects backpressure and drops frames
//!
//! Run with: cargo run --example 47_flow_control

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ProduceContext, ProduceResult, Source};
use parallax::elements::flow::Queue;
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::flow::{FlowPolicy, FlowSignal, FlowStateHandle, WaterMarks};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// A simulated live video source that produces frames at a fixed rate.
/// When downstream is congested (backpressure), it drops frames to avoid lag.
struct SimulatedCameraSource {
    arena: Arc<SharedArena>,
    frame_count: u64,
    max_frames: u64,
    frame_interval: Duration,
    last_frame_time: Option<Instant>,

    // Flow control
    flow_state: Option<FlowStateHandle>,
    frames_produced: u64,
    frames_dropped: u64,
}

impl SimulatedCameraSource {
    fn new(arena: Arc<SharedArena>, max_frames: u64, fps: u32) -> Self {
        Self {
            arena,
            frame_count: 0,
            max_frames,
            frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
            last_frame_time: None,
            flow_state: None,
            frames_produced: 0,
            frames_dropped: 0,
        }
    }

    fn set_flow_state(&mut self, handle: FlowStateHandle) {
        self.flow_state = Some(handle);
    }

    fn stats(&self) -> (u64, u64) {
        (self.frames_produced, self.frames_dropped)
    }
}

impl Source for SimulatedCameraSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        // Check if we've reached the frame limit
        if self.frame_count >= self.max_frames {
            return Ok(ProduceResult::Eos);
        }

        // Simulate frame timing (wait for next frame interval)
        let now = Instant::now();
        if let Some(last) = self.last_frame_time {
            let elapsed = now.duration_since(last);
            if elapsed < self.frame_interval {
                // Not time for next frame yet
                return Ok(ProduceResult::WouldBlock);
            }
        }
        self.last_frame_time = Some(now);

        // Check backpressure BEFORE producing
        if let Some(ref flow_state) = self.flow_state {
            if !flow_state.should_produce() {
                // Downstream is congested - drop this frame
                self.frames_dropped += 1;
                self.frame_count += 1;
                flow_state.record_drop();

                // Log periodically
                if self.frames_dropped == 1 || self.frames_dropped % 10 == 0 {
                    println!(
                        "  [Camera] Dropped frame {} (backpressure, total dropped: {})",
                        self.frame_count, self.frames_dropped
                    );
                }

                return Ok(ProduceResult::WouldBlock);
            }
        }

        // Produce the frame
        self.arena.reclaim();
        let slot = self
            .arena
            .acquire()
            .ok_or_else(|| parallax::error::Error::AllocationFailed("No arena slots".into()))?;

        // Simulate frame data (just fill with frame number)
        let mut handle = MemoryHandle::with_len(slot, 1024);
        handle.as_mut_slice()[0..8].copy_from_slice(&self.frame_count.to_le_bytes());

        let mut metadata = Metadata::new();
        metadata.sequence = self.frame_count;

        self.frame_count += 1;
        self.frames_produced += 1;

        Ok(ProduceResult::OwnBuffer(Buffer::new(handle, metadata)))
    }

    fn flow_policy(&self) -> FlowPolicy {
        // Live sources should drop frames when downstream is slow
        FlowPolicy::drop_with_logging()
    }

    fn handle_flow_signal(&mut self, signal: FlowSignal) {
        if let Some(ref flow_state) = self.flow_state {
            flow_state.set_signal(signal);
        }
    }
}

/// A simulated slow encoder that takes variable time to process frames.
struct SlowEncoder {
    process_time: Duration,
    frames_processed: u64,
}

impl SlowEncoder {
    fn new(process_time_ms: u64) -> Self {
        Self {
            process_time: Duration::from_millis(process_time_ms),
            frames_processed: 0,
        }
    }

    fn process(&mut self, _buffer: Buffer) -> Buffer {
        // Simulate slow encoding
        thread::sleep(self.process_time);
        self.frames_processed += 1;
        _buffer // Pass through for this example
    }

    fn frames_processed(&self) -> u64 {
        self.frames_processed
    }
}

fn main() -> Result<()> {
    println!("=== Flow Control and Backpressure Example ===\n");

    // Create shared arena for buffers
    let arena = Arc::new(SharedArena::new(1024, 100)?);

    // Create a queue with flow control enabled
    // Water marks: 80% high (triggers Busy), 20% low (releases to Ready)
    println!("Creating queue with flow control:");
    println!("  Capacity: 20 buffers");
    println!("  High water mark: 80% (16 buffers) -> signals Busy");
    println!("  Low water mark: 20% (4 buffers) -> signals Ready\n");

    let queue = Queue::new(20).with_flow_control();
    let flow_state = queue.flow_state_handle();

    // Create camera source connected to queue's flow state
    let mut camera = SimulatedCameraSource::new(arena.clone(), 100, 30); // 30 fps, 100 frames
    camera.set_flow_state(flow_state.clone());

    // Create slow encoder (50ms per frame = ~20 fps max throughput)
    // This is slower than the 30 fps camera, causing backpressure
    let mut encoder = SlowEncoder::new(50);

    println!("Starting simulation:");
    println!("  Camera: 30 fps (33ms/frame)");
    println!("  Encoder: ~20 fps max (50ms/frame)");
    println!("  Expected: Camera will drop ~33% of frames\n");

    let start = Instant::now();
    let arena_clone = arena.clone();

    // Producer thread (camera)
    let producer_queue = queue.clone();
    let producer = thread::spawn(move || {
        let slot = arena_clone.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        let mut eos = false;

        while !eos {
            match camera.produce(&mut ctx) {
                Ok(ProduceResult::OwnBuffer(buffer)) => {
                    // Try to push to queue (non-blocking for this example)
                    if let Err(_) =
                        producer_queue.push_timeout(buffer, Some(Duration::from_millis(1)))
                    {
                        // Queue full - this shouldn't happen often with flow control
                        camera.frames_dropped += 1;
                    }
                }
                Ok(ProduceResult::Eos) => {
                    eos = true;
                }
                Ok(ProduceResult::WouldBlock) => {
                    // Either not time for frame yet, or dropped due to backpressure
                    thread::sleep(Duration::from_millis(1));
                }
                _ => {}
            }
        }

        camera.stats()
    });

    // Consumer thread (encoder)
    let consumer_queue = queue.clone();
    let consumer = thread::spawn(move || {
        loop {
            match consumer_queue.pop_timeout(Some(Duration::from_millis(100))) {
                Ok(Some(buffer)) => {
                    let _ = encoder.process(buffer);

                    // Print progress periodically
                    if encoder.frames_processed() % 20 == 0 {
                        println!(
                            "  [Encoder] Processed {} frames, queue fill: {:.0}%",
                            encoder.frames_processed(),
                            consumer_queue.fill_level()
                        );
                    }
                }
                Ok(None) => {
                    // Timeout - check if producer is done
                    if consumer_queue.is_empty() {
                        // Give producer a chance to add more
                        thread::sleep(Duration::from_millis(50));
                        if consumer_queue.is_empty() {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }

        encoder.frames_processed()
    });

    // Wait for completion
    let (produced, dropped) = producer.join().unwrap();
    let processed = consumer.join().unwrap();

    let elapsed = start.elapsed();
    let stats = queue.stats();

    println!("\n=== Results ===");
    println!("Duration: {:.2}s", elapsed.as_secs_f64());
    println!("Camera frames produced: {}", produced);
    println!("Camera frames dropped: {}", dropped);
    println!("Encoder frames processed: {}", processed);
    println!(
        "Drop rate: {:.1}%",
        (dropped as f64 / (produced + dropped) as f64) * 100.0
    );
    println!("\nQueue statistics:");
    println!("  Total pushed: {}", stats.total_pushed);
    println!("  Total popped: {}", stats.total_popped);
    println!("  High water events: {}", stats.high_water_events);

    println!("\nFlow state statistics:");
    println!(
        "  Backpressure events: {}",
        flow_state.backpressure_events()
    );
    println!(
        "  Frames dropped (recorded): {}",
        flow_state.frames_dropped()
    );

    // Demonstrate custom water marks
    println!("\n=== Custom Water Marks Demo ===");

    let custom_wm = WaterMarks::with_percentages(100, 50, 10); // 50% high, 10% low
    let aggressive_queue = Queue::new(100).with_water_marks(custom_wm);
    let aggressive_flow = aggressive_queue.flow_state_handle();

    println!("Queue with aggressive water marks (50% high, 10% low):");

    // Fill to 49% - should stay Ready
    for i in 0..49 {
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 64);
        let buffer = Buffer::new(handle, Metadata::from_sequence(i));
        aggressive_queue.push(buffer)?;
    }
    println!("  At 49%: signal = {:?}", aggressive_flow.signal());

    // Fill to 50% - should become Busy
    let slot = arena.acquire().unwrap();
    let handle = MemoryHandle::with_len(slot, 64);
    let buffer = Buffer::new(handle, Metadata::from_sequence(49));
    aggressive_queue.push(buffer)?;
    println!("  At 50%: signal = {:?}", aggressive_flow.signal());

    // Drain to 11% - should still be Busy
    while aggressive_queue.len() > 11 {
        aggressive_queue.pop_timeout(Some(Duration::from_millis(1)))?;
    }
    println!("  At 11%: signal = {:?}", aggressive_flow.signal());

    // Drain to 10% - should become Ready
    aggressive_queue.pop_timeout(Some(Duration::from_millis(1)))?;
    println!("  At 10%: signal = {:?}", aggressive_flow.signal());

    println!("\n=== Flow Control Demo Complete ===");

    Ok(())
}
