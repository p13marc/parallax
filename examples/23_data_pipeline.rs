//! Real-time data processing pipeline example.
//!
//! Demonstrates a streaming data pipeline that processes sensor data:
//! - Generates simulated sensor readings
//! - Filters out-of-range values
//! - Computes rolling statistics
//! - Outputs alerts for anomalies
//!
//! Run with: cargo run --example 23_data_pipeline

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    ConsumeContext, DynAsyncElement, Element, ElementAdapter, ProduceContext, ProduceResult, Sink,
    SinkAdapter, Source, SourceAdapter,
};
use parallax::error::Result;
use parallax::memory::{CpuArena, HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Simulated sensor reading.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct SensorReading {
    timestamp: u64,
    sensor_id: u32,
    value: f32,
    flags: u32,
}

impl SensorReading {
    fn to_bytes(&self) -> [u8; 20] {
        let mut bytes = [0u8; 20];
        bytes[0..8].copy_from_slice(&self.timestamp.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.sensor_id.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.value.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.flags.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 20 {
            return None;
        }
        Some(Self {
            timestamp: u64::from_le_bytes(bytes[0..8].try_into().ok()?),
            sensor_id: u32::from_le_bytes(bytes[8..12].try_into().ok()?),
            value: f32::from_le_bytes(bytes[12..16].try_into().ok()?),
            flags: u32::from_le_bytes(bytes[16..20].try_into().ok()?),
        })
    }
}

/// Source that generates simulated sensor data.
struct SensorSource {
    count: u64,
    max: u64,
    rng_state: u64,
}

impl SensorSource {
    fn new(count: u64) -> Self {
        Self {
            count: 0,
            max: count,
            rng_state: 12345,
        }
    }

    // Simple PRNG for reproducible "random" values (returns 0.0 to 1.0)
    fn next_random(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_state >> 16) & 0xFFFF) as f32 / 65536.0
    }
}

impl Source for SensorSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }

        // Generate a reading with occasional outliers
        let base_value = 50.0 + 10.0 * (self.count as f32 * 0.1).sin();
        let noise = (self.next_random() - 0.5) * 5.0;
        let value = if self.next_random() > 0.95 {
            // 5% chance of anomaly
            base_value + noise + (self.next_random() - 0.5) * 100.0
        } else {
            base_value + noise
        };

        let reading = SensorReading {
            timestamp: self.count * 100,        // 100ms intervals
            sensor_id: (self.count % 4) as u32, // 4 sensors
            value,
            flags: 0,
        };

        let data = reading.to_bytes();

        if ctx.has_buffer() {
            let output = ctx.output();
            output[..data.len()].copy_from_slice(&data);
            ctx.set_sequence(self.count);
            self.count += 1;
            Ok(ProduceResult::Produced(data.len()))
        } else {
            let segment = Arc::new(HeapSegment::new(data.len())?);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    data.len(),
                );
            }
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(self.count));
            self.count += 1;
            Ok(ProduceResult::OwnBuffer(buffer))
        }
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(20) // Size of SensorReading
    }
}

/// Filter that removes out-of-range readings.
struct RangeFilter {
    min: f32,
    max: f32,
    filtered_count: u64,
}

impl RangeFilter {
    fn new(min: f32, max: f32) -> Self {
        Self {
            min,
            max,
            filtered_count: 0,
        }
    }
}

impl Element for RangeFilter {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        if let Some(reading) = SensorReading::from_bytes(buffer.as_bytes()) {
            if reading.value >= self.min && reading.value <= self.max {
                Ok(Some(buffer))
            } else {
                self.filtered_count += 1;
                Ok(None) // Filter out this reading
            }
        } else {
            Ok(Some(buffer)) // Pass through non-sensor data
        }
    }
}

/// Transform that computes rolling statistics.
struct StatsCompute {
    window: Vec<f32>,
    window_size: usize,
    sum: f32,
}

impl StatsCompute {
    fn new(window_size: usize) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size,
            sum: 0.0,
        }
    }

    fn add_sample(&mut self, value: f32) -> (f32, f32) {
        if self.window.len() >= self.window_size {
            let old = self.window.remove(0);
            self.sum -= old;
        }
        self.window.push(value);
        self.sum += value;

        let mean = self.sum / self.window.len() as f32;
        let variance =
            self.window.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / self.window.len() as f32;
        let stddev = variance.sqrt();

        (mean, stddev)
    }
}

impl Element for StatsCompute {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        if let Some(reading) = SensorReading::from_bytes(buffer.as_bytes()) {
            let (mean, stddev) = self.add_sample(reading.value);

            // Mark as anomaly if more than 2 stddev from mean
            let is_anomaly = (reading.value - mean).abs() > 2.0 * stddev && self.window.len() > 5;

            // Create new reading with flags updated
            let new_reading = SensorReading {
                flags: if is_anomaly { 1 } else { 0 },
                ..reading
            };

            let data = new_reading.to_bytes();
            let segment = Arc::new(HeapSegment::new(data.len())?);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    data.len(),
                );
            }
            let new_buffer = Buffer::<()>::new(
                MemoryHandle::from_segment(segment),
                buffer.metadata().clone(),
            );
            Ok(Some(new_buffer))
        } else {
            Ok(Some(buffer))
        }
    }
}

/// Sink that displays readings and alerts on anomalies.
struct DisplaySink {
    total_readings: Arc<AtomicU64>,
    anomaly_count: Arc<AtomicU64>,
    show_all: bool,
}

impl Sink for DisplaySink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.total_readings.fetch_add(1, Ordering::Relaxed);

        if let Some(reading) = SensorReading::from_bytes(ctx.input()) {
            let is_anomaly = reading.flags == 1;

            if is_anomaly {
                self.anomaly_count.fetch_add(1, Ordering::Relaxed);
                println!(
                    "ALERT! Sensor {} at t={}: value={:.2} (ANOMALY)",
                    reading.sensor_id, reading.timestamp, reading.value
                );
            } else if self.show_all {
                println!(
                    "Sensor {} at t={}: value={:.2}",
                    reading.sensor_id, reading.timestamp, reading.value
                );
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Real-Time Data Processing Pipeline");
    println!("===================================");
    println!();

    let total_readings = Arc::new(AtomicU64::new(0));
    let anomaly_count = Arc::new(AtomicU64::new(0));

    let arena = CpuArena::new(64, 32)?;

    let mut pipeline = Pipeline::new();

    // Source: generate 1000 sensor readings
    let src = pipeline.add_node(
        "sensor_source",
        DynAsyncElement::new_box(SourceAdapter::with_arena(SensorSource::new(1000), arena)),
    );

    // Filter: remove readings outside -50 to 150 range (generous to allow most anomalies through)
    let filter = pipeline.add_node(
        "range_filter",
        DynAsyncElement::new_box(ElementAdapter::new(RangeFilter::new(-50.0, 150.0))),
    );

    // Stats: compute rolling statistics (window of 10)
    let stats = pipeline.add_node(
        "stats_compute",
        DynAsyncElement::new_box(ElementAdapter::new(StatsCompute::new(10))),
    );

    // Sink: display results
    let sink = pipeline.add_node(
        "display_sink",
        DynAsyncElement::new_box(SinkAdapter::new(DisplaySink {
            total_readings: total_readings.clone(),
            anomaly_count: anomaly_count.clone(),
            show_all: false, // Only show anomalies
        })),
    );

    pipeline.link(src, filter)?;
    pipeline.link(filter, stats)?;
    pipeline.link(stats, sink)?;

    println!("Pipeline: sensor_source -> range_filter -> stats_compute -> display_sink");
    println!();
    println!("Processing 1000 sensor readings...");
    println!("(Only anomalies are displayed)");
    println!();

    pipeline.run().await?;

    println!();
    println!("-----------------------------------");
    println!("Summary:");
    println!(
        "  Total readings processed: {}",
        total_readings.load(Ordering::Relaxed)
    );
    println!(
        "  Anomalies detected: {}",
        anomaly_count.load(Ordering::Relaxed)
    );
    println!();
    println!("This pipeline could be enhanced with:");
    println!("  - Network source (UDP/TCP for real sensor data)");
    println!("  - Database sink (store readings)");
    println!("  - Tee element (branch to multiple sinks)");
    println!("  - Process isolation (sandbox untrusted transforms)");

    Ok(())
}
