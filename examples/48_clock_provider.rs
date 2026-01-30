//! Clock Provider Example
//!
//! This example demonstrates how to use custom clocks in Parallax pipelines
//! for accurate timing and A/V synchronization.
//!
//! # Concepts
//!
//! - **Clock**: A time source that provides monotonic timestamps
//! - **ClockProvider**: An element that can provide a clock (e.g., audio sink)
//! - **PipelineClock**: The pipeline's time reference with base time tracking
//! - **ClockTime**: Nanosecond-precision timestamp type
//!
//! # Clock Selection
//!
//! In professional A/V systems, the audio clock is typically used as the master
//! because audio devices maintain very precise sample rates. Video is then
//! synchronized to the audio clock to prevent drift.
//!
//! Priority levels:
//! - 0-99: Software clocks (system monotonic) - default
//! - 100-199: Hardware clocks (audio devices)
//! - 200-299: Network clocks (NTP)
//! - 300+: Precision clocks (PTP)
//!
//! Run with: cargo run --example 48_clock_provider

use parallax::clock::{Clock, ClockFlags, ClockTime};
use parallax::elements::{NullSink, NullSource};
use parallax::pipeline::{Executor, Pipeline};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// A custom clock implementation for demonstration.
///
/// In a real application, this would wrap hardware timing from:
/// - ALSA audio device (AlsaClock)
/// - PipeWire stream timing
/// - Network time protocol (NTP/PTP)
struct DemoClock {
    start: Instant,
    offset_nanos: AtomicU64,
    name: String,
}

impl DemoClock {
    fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            offset_nanos: AtomicU64::new(0),
            name: name.to_string(),
        }
    }

    /// Add an offset to simulate clock drift or adjustment.
    fn add_offset(&self, nanos: u64) {
        self.offset_nanos.fetch_add(nanos, Ordering::SeqCst);
    }
}

impl Clock for DemoClock {
    fn now(&self) -> ClockTime {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        let offset = self.offset_nanos.load(Ordering::SeqCst);
        ClockTime::from_nanos(elapsed + offset)
    }

    fn flags(&self) -> ClockFlags {
        // This demo clock can be master and is based on hardware (Instant)
        ClockFlags::CAN_BE_MASTER | ClockFlags::HARDWARE
    }

    fn resolution(&self) -> u64 {
        // Nanosecond resolution
        1
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Clock Provider Example ===\n");

    // Example 1: Default system clock
    println!("1. Using default system clock:");
    {
        let pipeline = Pipeline::new();
        let clock = pipeline.clock();
        println!("   Clock name: {}", clock.clock().name());
        println!("   Running time: {:?}", clock.running_time());
        println!("   Started: {}", clock.is_started());
    }

    // Example 2: Custom clock
    println!("\n2. Using custom clock:");
    {
        let custom_clock = Arc::new(DemoClock::new("demo-audio-clock"));
        let clock_ref = custom_clock.clone();

        println!("   Initial time: {:?}", custom_clock.now());
        println!("   Clock flags: {:?}", custom_clock.flags());
        println!("   Resolution: {} ns", custom_clock.resolution());

        // Simulate some time passing
        std::thread::sleep(std::time::Duration::from_millis(10));
        println!("   After 10ms: {:?}", custom_clock.now());

        // Add an offset (simulating clock adjustment)
        custom_clock.add_offset(1_000_000); // +1ms
        println!("   After +1ms offset: {:?}", custom_clock.now());

        // Use in pipeline
        let mut pipeline = Pipeline::with_clock(clock_ref);
        let src = pipeline.add_source("src", NullSource::new(10));
        let sink = pipeline.add_sink("sink", NullSink::new());
        pipeline.link(src, sink)?;

        // Start pipeline - this starts the clock
        let executor = Executor::new();
        executor.run(&mut pipeline).await?;

        println!("   Final clock time: {:?}", custom_clock.now());
    }

    // Example 3: Clock time arithmetic
    println!("\n3. ClockTime arithmetic:");
    {
        let t1 = ClockTime::from_secs(1);
        let t2 = ClockTime::from_millis(500);

        println!("   t1 = {:?} (1 second)", t1);
        println!("   t2 = {:?} (500ms)", t2);
        println!("   t1 + t2 = {:?}", t1 + t2);
        println!("   t1 - t2 = {:?}", t1 - t2);
        println!("   t1 > t2: {}", t1 > t2);

        // NONE sentinel
        let none = ClockTime::NONE;
        println!("   NONE.is_none(): {}", none.is_none());
        println!("   t1.is_some(): {}", t1.is_some());
    }

    // Example 4: ClockFlags
    println!("\n4. ClockFlags:");
    {
        let software = ClockFlags::CAN_BE_MASTER;
        let hardware = ClockFlags::CAN_BE_MASTER | ClockFlags::HARDWARE;
        let network = ClockFlags::CAN_BE_MASTER | ClockFlags::NETWORK;
        let realtime = ClockFlags::CAN_BE_MASTER | ClockFlags::HARDWARE | ClockFlags::REALTIME;

        println!("   Software clock: {:?}", software);
        println!("   Hardware clock: {:?}", hardware);
        println!("   Network clock: {:?}", network);
        println!("   Realtime clock: {:?}", realtime);

        println!(
            "   hardware.contains(HARDWARE): {}",
            hardware.contains(ClockFlags::HARDWARE)
        );
        println!(
            "   software.contains(HARDWARE): {}",
            software.contains(ClockFlags::HARDWARE)
        );
    }

    // Example 5: Running time calculation
    println!("\n5. Running time (time since pipeline start):");
    {
        let clock = Arc::new(DemoClock::new("timing-demo"));
        let mut pipeline = Pipeline::with_clock(clock);

        let src = pipeline.add_source("src", NullSource::new(5));
        let sink = pipeline.add_sink("sink", NullSink::new());
        pipeline.link(src, sink)?;

        println!(
            "   Before start - running time: {:?}",
            pipeline.running_time()
        );
        println!("   Clock started: {}", pipeline.clock().is_started());

        // The executor starts the clock automatically
        let executor = Executor::new();
        let handle = executor.start(&mut pipeline)?;

        // Give it a moment to start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        println!(
            "   During run - running time: {:?}",
            pipeline.running_time()
        );
        println!("   Clock started: {}", pipeline.clock().is_started());

        handle.wait().await?;

        println!(
            "   After completion - running time: {:?}",
            pipeline.running_time()
        );
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
