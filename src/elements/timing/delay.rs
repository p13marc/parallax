//! Delay element for adding fixed delays to buffer flow.
//!
//! Introduces a configurable delay before passing buffers downstream.

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A delay element that introduces a fixed delay to buffer flow.
///
/// Each buffer is held for the specified duration before being passed
/// to the next element. This is useful for testing, simulation, or
/// timing adjustments.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Delay;
/// use std::time::Duration;
///
/// let delay = Delay::new(Duration::from_millis(100));
/// ```
pub struct Delay {
    name: String,
    delay: Duration,
    count: AtomicU64,
    total_delay: AtomicU64, // in microseconds
}

impl Delay {
    /// Create a new delay element with the specified duration.
    pub fn new(delay: Duration) -> Self {
        Self {
            name: "delay".to_string(),
            delay,
            count: AtomicU64::new(0),
            total_delay: AtomicU64::new(0),
        }
    }

    /// Create a delay element from milliseconds.
    pub fn from_millis(millis: u64) -> Self {
        Self::new(Duration::from_millis(millis))
    }

    /// Create a delay element from microseconds.
    pub fn from_micros(micros: u64) -> Self {
        Self::new(Duration::from_micros(micros))
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the configured delay duration.
    pub fn delay(&self) -> Duration {
        self.delay
    }

    /// Set a new delay duration.
    pub fn set_delay(&mut self, delay: Duration) {
        self.delay = delay;
    }

    /// Get the number of buffers processed.
    pub fn buffer_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the total delay applied (in microseconds).
    pub fn total_delay_micros(&self) -> u64 {
        self.total_delay.load(Ordering::Relaxed)
    }

    /// Get statistics.
    pub fn stats(&self) -> DelayStats {
        DelayStats {
            buffer_count: self.count.load(Ordering::Relaxed),
            configured_delay: self.delay,
            total_delay_micros: self.total_delay.load(Ordering::Relaxed),
        }
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_delay.store(0, Ordering::Relaxed);
    }
}

impl Element for Delay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        if !self.delay.is_zero() {
            let start = Instant::now();
            std::thread::sleep(self.delay);
            let actual = start.elapsed();
            self.total_delay
                .fetch_add(actual.as_micros() as u64, Ordering::Relaxed);
        }

        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Delay element.
#[derive(Debug, Clone, Copy)]
pub struct DelayStats {
    /// Number of buffers processed.
    pub buffer_count: u64,
    /// Configured delay duration.
    pub configured_delay: Duration,
    /// Total actual delay applied in microseconds.
    pub total_delay_micros: u64,
}

/// An async-compatible delay element.
///
/// Note: Requires tokio runtime with `time` feature enabled.
pub struct AsyncDelay {
    name: String,
    delay: Duration,
    count: AtomicU64,
}

impl AsyncDelay {
    /// Create a new async delay element.
    pub fn new(delay: Duration) -> Self {
        Self {
            name: "async-delay".to_string(),
            delay,
            count: AtomicU64::new(0),
        }
    }

    /// Create from milliseconds.
    pub fn from_millis(millis: u64) -> Self {
        Self::new(Duration::from_millis(millis))
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the configured delay.
    pub fn delay(&self) -> Duration {
        self.delay
    }

    /// Set a new delay duration.
    pub fn set_delay(&mut self, delay: Duration) {
        self.delay = delay;
    }

    /// Process a buffer with async delay using std thread sleep.
    /// For true async delay, use tokio::time::sleep directly in your code.
    pub fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(Some(buffer))
    }

    /// Get buffer count.
    pub fn buffer_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the element name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::CpuSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    fn create_test_buffer(seq: u64) -> Buffer {
        let segment = Arc::new(CpuSegment::new(64).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn test_delay_passthrough() {
        let mut delay = Delay::new(Duration::ZERO);

        let buffer = create_test_buffer(42);
        let result = delay.process(buffer).unwrap();

        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    #[test]
    fn test_delay_timing() {
        let mut delay = Delay::from_millis(50);

        let start = Instant::now();
        delay.process(create_test_buffer(0)).unwrap();
        let elapsed = start.elapsed();

        assert!(elapsed >= Duration::from_millis(45)); // Allow some margin
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_delay_stats() {
        let mut delay = Delay::from_millis(10);

        delay.process(create_test_buffer(0)).unwrap();
        delay.process(create_test_buffer(1)).unwrap();

        let stats = delay.stats();
        assert_eq!(stats.buffer_count, 2);
        assert!(stats.total_delay_micros >= 20_000); // At least 20ms total
    }

    #[test]
    fn test_delay_reset_stats() {
        let mut delay = Delay::new(Duration::ZERO);

        delay.process(create_test_buffer(0)).unwrap();
        assert_eq!(delay.buffer_count(), 1);

        delay.reset_stats();
        assert_eq!(delay.buffer_count(), 0);
    }

    #[test]
    fn test_delay_set_delay() {
        let mut delay = Delay::from_millis(100);
        assert_eq!(delay.delay(), Duration::from_millis(100));

        delay.set_delay(Duration::from_millis(50));
        assert_eq!(delay.delay(), Duration::from_millis(50));
    }

    #[test]
    fn test_delay_with_name() {
        let delay = Delay::from_millis(10).with_name("my-delay");
        assert_eq!(delay.name(), "my-delay");
    }

    #[test]
    fn test_async_delay() {
        let mut delay = AsyncDelay::from_millis(50);

        let start = Instant::now();
        let result = delay.process(create_test_buffer(0)).unwrap();
        let elapsed = start.elapsed();

        assert!(result.is_some());
        assert!(elapsed >= Duration::from_millis(45));
    }
}
