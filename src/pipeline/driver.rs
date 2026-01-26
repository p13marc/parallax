//! Driver node support for timer-based pipeline scheduling.
//!
//! Drivers are special nodes that initiate processing cycles in the pipeline.
//! They're inspired by PipeWire's driver concept where a sink (like an audio
//! device) drives the processing graph at a regular interval (quantum).
//!
//! # Driver Types
//!
//! - **TimerDriver**: Software timer-based driver for regular intervals
//! - **SinkDriver**: Hardware-paced driver (audio device callback, display vsync)
//! - **ManualDriver**: Externally triggered driver for testing/control
//!
//! # Quantum and Period
//!
//! - **Quantum**: Number of samples to process per cycle
//! - **Period**: Time duration of one cycle
//!
//! For audio at 48kHz:
//! - Quantum 64 = 1.33ms period (pro audio)
//! - Quantum 256 = 5.33ms period (low latency)
//! - Quantum 1024 = 21.33ms period (standard)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::driver::{TimerDriver, DriverConfig};
//! use std::time::Duration;
//!
//! // Create a 10ms timer driver
//! let driver = TimerDriver::new(DriverConfig {
//!     period: Duration::from_millis(10),
//!     quantum: 480,  // 48kHz * 10ms
//! });
//!
//! // Start the driver
//! let handle = driver.start()?;
//!
//! // Wait for cycle trigger
//! handle.wait_cycle().await;
//! ```

use crate::error::{Error, Result};
use crate::pipeline::rt_bridge::EventFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a driver.
#[derive(Debug, Clone)]
pub struct DriverConfig {
    /// Period (time duration) of one processing cycle.
    pub period: Duration,

    /// Quantum (samples per cycle) for audio processing.
    /// This is informational and passed to elements.
    pub quantum: u32,

    /// Sample rate in Hz (for audio).
    pub sample_rate: u32,

    /// Maximum allowed jitter before logging a warning.
    pub max_jitter: Duration,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            period: Duration::from_millis(10), // 10ms default
            quantum: 480,                      // 48kHz * 10ms
            sample_rate: 48000,
            max_jitter: Duration::from_millis(2),
        }
    }
}

impl DriverConfig {
    /// Create a driver config for low-latency audio.
    ///
    /// Uses 64 samples at 48kHz = 1.33ms period.
    pub fn low_latency_audio() -> Self {
        Self {
            period: Duration::from_micros(1333),
            quantum: 64,
            sample_rate: 48000,
            max_jitter: Duration::from_micros(500),
        }
    }

    /// Create a driver config for standard audio.
    ///
    /// Uses 1024 samples at 48kHz = 21.33ms period.
    pub fn standard_audio() -> Self {
        Self {
            period: Duration::from_micros(21333),
            quantum: 1024,
            sample_rate: 48000,
            max_jitter: Duration::from_millis(5),
        }
    }

    /// Create a driver config for video at a given framerate.
    pub fn video(fps: u32) -> Self {
        let period_us = 1_000_000 / fps as u64;
        Self {
            period: Duration::from_micros(period_us),
            quantum: 1, // One frame per cycle
            sample_rate: fps,
            max_jitter: Duration::from_millis(2),
        }
    }

    /// Create a custom driver config.
    pub fn custom(period: Duration, quantum: u32) -> Self {
        Self {
            period,
            quantum,
            sample_rate: (quantum as f64 / period.as_secs_f64()) as u32,
            max_jitter: Duration::from_millis(2),
        }
    }
}

// ============================================================================
// Driver Statistics
// ============================================================================

/// Statistics about driver timing.
#[derive(Debug, Clone, Default)]
pub struct DriverStats {
    /// Total number of cycles completed.
    pub cycles: u64,
    /// Number of cycles that exceeded max jitter.
    pub late_cycles: u64,
    /// Minimum observed cycle time.
    pub min_cycle_time: Duration,
    /// Maximum observed cycle time.
    pub max_cycle_time: Duration,
    /// Average cycle time (exponential moving average).
    pub avg_cycle_time: Duration,
    /// Last cycle's processing time.
    pub last_process_time: Duration,
}

/// Atomic statistics for lock-free updates.
struct AtomicStats {
    cycles: AtomicU64,
    late_cycles: AtomicU64,
    min_cycle_ns: AtomicU64,
    max_cycle_ns: AtomicU64,
    avg_cycle_ns: AtomicU64,
    last_process_ns: AtomicU64,
}

impl AtomicStats {
    fn new() -> Self {
        Self {
            cycles: AtomicU64::new(0),
            late_cycles: AtomicU64::new(0),
            min_cycle_ns: AtomicU64::new(u64::MAX),
            max_cycle_ns: AtomicU64::new(0),
            avg_cycle_ns: AtomicU64::new(0),
            last_process_ns: AtomicU64::new(0),
        }
    }

    fn record_cycle(&self, cycle_time: Duration, max_jitter: Duration) {
        let ns = cycle_time.as_nanos() as u64;

        self.cycles.fetch_add(1, Ordering::Relaxed);

        // Update min
        let mut current = self.min_cycle_ns.load(Ordering::Relaxed);
        while ns < current {
            match self.min_cycle_ns.compare_exchange_weak(
                current,
                ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }

        // Update max
        current = self.max_cycle_ns.load(Ordering::Relaxed);
        while ns > current {
            match self.max_cycle_ns.compare_exchange_weak(
                current,
                ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }

        // Update average (EMA with alpha = 0.1)
        let old_avg = self.avg_cycle_ns.load(Ordering::Relaxed);
        let new_avg = if old_avg == 0 {
            ns
        } else {
            (old_avg * 9 + ns) / 10
        };
        self.avg_cycle_ns.store(new_avg, Ordering::Relaxed);

        // Check for late cycle
        let expected_ns = max_jitter.as_nanos() as u64;
        if ns > expected_ns {
            self.late_cycles.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_process_time(&self, process_time: Duration) {
        self.last_process_ns
            .store(process_time.as_nanos() as u64, Ordering::Relaxed);
    }

    fn snapshot(&self) -> DriverStats {
        let min_ns = self.min_cycle_ns.load(Ordering::Relaxed);
        DriverStats {
            cycles: self.cycles.load(Ordering::Relaxed),
            late_cycles: self.late_cycles.load(Ordering::Relaxed),
            min_cycle_time: if min_ns == u64::MAX {
                Duration::ZERO
            } else {
                Duration::from_nanos(min_ns)
            },
            max_cycle_time: Duration::from_nanos(self.max_cycle_ns.load(Ordering::Relaxed)),
            avg_cycle_time: Duration::from_nanos(self.avg_cycle_ns.load(Ordering::Relaxed)),
            last_process_time: Duration::from_nanos(self.last_process_ns.load(Ordering::Relaxed)),
        }
    }
}

// ============================================================================
// Timer Driver
// ============================================================================

/// A software timer-based driver for regular processing cycles.
///
/// Uses tokio's sleep for timing (async) or std::thread::sleep (sync).
pub struct TimerDriver {
    config: DriverConfig,
    stats: Arc<AtomicStats>,
    stop: Arc<AtomicBool>,
}

impl TimerDriver {
    /// Create a new timer driver with the given configuration.
    pub fn new(config: DriverConfig) -> Self {
        Self {
            config,
            stats: Arc::new(AtomicStats::new()),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get the driver configuration.
    pub fn config(&self) -> &DriverConfig {
        &self.config
    }

    /// Get current statistics.
    pub fn stats(&self) -> DriverStats {
        self.stats.snapshot()
    }

    /// Start the driver and return a handle for receiving cycle triggers.
    ///
    /// This spawns a background task that triggers at the configured period.
    pub fn start_async(self) -> TimerDriverHandle {
        let trigger = Arc::new(EventFd::new().expect("failed to create eventfd"));
        let config = self.config.clone();
        let stats = self.stats.clone();
        let stop = self.stop.clone();
        let trigger_clone = trigger.clone();

        let task = tokio::spawn(async move {
            let mut next_cycle = Instant::now();

            while !stop.load(Ordering::Relaxed) {
                // Calculate sleep duration
                let now = Instant::now();
                let sleep_duration = next_cycle.saturating_duration_since(now);

                if !sleep_duration.is_zero() {
                    tokio::time::sleep(sleep_duration).await;
                }

                // Record cycle timing
                let actual_time = now.elapsed();
                stats.record_cycle(
                    config.period + actual_time,
                    config.period + config.max_jitter,
                );

                // Trigger the cycle
                if let Err(e) = trigger_clone.notify() {
                    tracing::error!("driver trigger failed: {}", e);
                    break;
                }

                // Schedule next cycle
                next_cycle += config.period;

                // If we've fallen behind, catch up
                if next_cycle < Instant::now() {
                    tracing::warn!("driver falling behind, skipping to next cycle");
                    next_cycle = Instant::now() + config.period;
                }
            }
        });

        TimerDriverHandle {
            trigger,
            stats: self.stats,
            stop: self.stop,
            task: Some(task),
            config: self.config,
        }
    }

    /// Start the driver in a dedicated thread (for RT scheduling).
    pub fn start_rt(self) -> RtTimerDriverHandle {
        let trigger = Arc::new(EventFd::new().expect("failed to create eventfd"));
        let config = self.config.clone();
        let stats = self.stats.clone();
        let stop = self.stop.clone();
        let trigger_clone = trigger.clone();

        let thread = std::thread::Builder::new()
            .name("parallax-driver".to_string())
            .spawn(move || {
                let mut next_cycle = Instant::now();

                while !stop.load(Ordering::Relaxed) {
                    // Calculate sleep duration
                    let now = Instant::now();
                    let sleep_duration = next_cycle.saturating_duration_since(now);

                    if !sleep_duration.is_zero() {
                        // Use spin-sleep for better precision on short durations
                        if sleep_duration < Duration::from_millis(1) {
                            spin_sleep(sleep_duration);
                        } else {
                            std::thread::sleep(sleep_duration);
                        }
                    }

                    // Record cycle timing
                    let actual_time = now.elapsed();
                    stats.record_cycle(
                        config.period + actual_time,
                        config.period + config.max_jitter,
                    );

                    // Trigger the cycle
                    if let Err(e) = trigger_clone.notify() {
                        tracing::error!("driver trigger failed: {}", e);
                        break;
                    }

                    // Schedule next cycle
                    next_cycle += config.period;

                    // If we've fallen behind, catch up
                    if next_cycle < Instant::now() {
                        tracing::warn!("driver falling behind, skipping to next cycle");
                        next_cycle = Instant::now() + config.period;
                    }
                }
            })
            .expect("failed to spawn driver thread");

        RtTimerDriverHandle {
            trigger,
            stats: self.stats,
            stop: self.stop,
            thread: Some(thread),
            config: self.config,
        }
    }
}

/// Handle to a running async timer driver.
pub struct TimerDriverHandle {
    trigger: Arc<EventFd>,
    stats: Arc<AtomicStats>,
    stop: Arc<AtomicBool>,
    task: Option<tokio::task::JoinHandle<()>>,
    config: DriverConfig,
}

impl TimerDriverHandle {
    /// Wait for the next cycle trigger.
    pub async fn wait_cycle(&self) -> Result<()> {
        self.trigger.wait_async().await
    }

    /// Try to get a cycle trigger without waiting.
    pub fn try_cycle(&self) -> Result<bool> {
        self.trigger.try_wait()
    }

    /// Get the eventfd for direct integration.
    pub fn trigger(&self) -> &Arc<EventFd> {
        &self.trigger
    }

    /// Get current statistics.
    pub fn stats(&self) -> DriverStats {
        self.stats.snapshot()
    }

    /// Record processing time for this cycle.
    pub fn record_process_time(&self, duration: Duration) {
        self.stats.record_process_time(duration);
    }

    /// Get the driver configuration.
    pub fn config(&self) -> &DriverConfig {
        &self.config
    }

    /// Stop the driver.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    /// Check if the driver is running.
    pub fn is_running(&self) -> bool {
        self.task.as_ref().map_or(false, |t| !t.is_finished())
    }
}

impl Drop for TimerDriverHandle {
    fn drop(&mut self) {
        self.stop();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

/// Handle to a running RT timer driver.
pub struct RtTimerDriverHandle {
    trigger: Arc<EventFd>,
    stats: Arc<AtomicStats>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    config: DriverConfig,
}

impl RtTimerDriverHandle {
    /// Get a cycle trigger (non-blocking poll from RT context).
    pub fn try_cycle(&self) -> Result<bool> {
        self.trigger.try_wait()
    }

    /// Get the eventfd for direct integration.
    pub fn trigger(&self) -> &Arc<EventFd> {
        &self.trigger
    }

    /// Get current statistics.
    pub fn stats(&self) -> DriverStats {
        self.stats.snapshot()
    }

    /// Record processing time for this cycle.
    pub fn record_process_time(&self, duration: Duration) {
        self.stats.record_process_time(duration);
    }

    /// Get the driver configuration.
    pub fn config(&self) -> &DriverConfig {
        &self.config
    }

    /// Stop the driver.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    /// Join the driver thread.
    pub fn join(mut self) -> Result<()> {
        self.stop();
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| Error::InvalidSegment("driver thread panicked".into()))?;
        }
        Ok(())
    }
}

impl Drop for RtTimerDriverHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================================
// Manual Driver
// ============================================================================

/// A manually-triggered driver for testing and external control.
pub struct ManualDriver {
    trigger: Arc<EventFd>,
    stats: Arc<AtomicStats>,
    config: DriverConfig,
}

impl ManualDriver {
    /// Create a new manual driver.
    pub fn new(config: DriverConfig) -> Result<Self> {
        Ok(Self {
            trigger: Arc::new(EventFd::new()?),
            stats: Arc::new(AtomicStats::new()),
            config,
        })
    }

    /// Trigger a processing cycle.
    pub fn trigger(&self) -> Result<()> {
        self.stats.cycles.fetch_add(1, Ordering::Relaxed);
        self.trigger.notify()
    }

    /// Get the eventfd for integration.
    pub fn eventfd(&self) -> &Arc<EventFd> {
        &self.trigger
    }

    /// Get current statistics.
    pub fn stats(&self) -> DriverStats {
        self.stats.snapshot()
    }

    /// Get the configuration.
    pub fn config(&self) -> &DriverConfig {
        &self.config
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Spin-sleep for short durations (better precision than thread::sleep).
fn spin_sleep(duration: Duration) {
    let target = Instant::now() + duration;
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_driver_config_defaults() {
        let config = DriverConfig::default();
        assert_eq!(config.period, Duration::from_millis(10));
        assert_eq!(config.quantum, 480);
        assert_eq!(config.sample_rate, 48000);
    }

    #[test]
    fn test_driver_config_low_latency() {
        let config = DriverConfig::low_latency_audio();
        assert_eq!(config.quantum, 64);
        assert!(config.period < Duration::from_millis(2));
    }

    #[test]
    fn test_driver_config_video() {
        let config = DriverConfig::video(60);
        assert!(config.period > Duration::from_millis(16));
        assert!(config.period < Duration::from_millis(17));
    }

    #[test]
    fn test_atomic_stats() {
        let stats = AtomicStats::new();

        stats.record_cycle(Duration::from_millis(10), Duration::from_millis(15));
        stats.record_cycle(Duration::from_millis(12), Duration::from_millis(15));
        stats.record_cycle(Duration::from_millis(8), Duration::from_millis(15));

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.cycles, 3);
        assert_eq!(snapshot.late_cycles, 0);
        assert_eq!(snapshot.min_cycle_time, Duration::from_millis(8));
        assert_eq!(snapshot.max_cycle_time, Duration::from_millis(12));
    }

    #[test]
    fn test_manual_driver() {
        let driver = ManualDriver::new(DriverConfig::default()).unwrap();

        // Trigger a few cycles
        driver.trigger().unwrap();
        driver.trigger().unwrap();
        driver.trigger().unwrap();

        let stats = driver.stats();
        assert_eq!(stats.cycles, 3);
    }

    #[tokio::test]
    async fn test_timer_driver_async() {
        let config = DriverConfig {
            period: Duration::from_millis(5),
            quantum: 240,
            sample_rate: 48000,
            max_jitter: Duration::from_millis(2),
        };

        let driver = TimerDriver::new(config);
        let handle = driver.start_async();

        // Wait for a few cycles
        for _ in 0..3 {
            handle.wait_cycle().await.unwrap();
        }

        let stats = handle.stats();
        assert!(stats.cycles >= 3);

        handle.stop();
    }

    #[test]
    fn test_timer_driver_rt() {
        let config = DriverConfig {
            period: Duration::from_millis(5),
            quantum: 240,
            sample_rate: 48000,
            max_jitter: Duration::from_millis(2),
        };

        let driver = TimerDriver::new(config);
        let handle = driver.start_rt();

        // Wait for a few cycles using polling
        let mut cycles = 0;
        let start = Instant::now();
        while cycles < 3 && start.elapsed() < Duration::from_millis(100) {
            if handle.try_cycle().unwrap() {
                cycles += 1;
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        assert!(cycles >= 3);

        handle.stop();
    }
}
