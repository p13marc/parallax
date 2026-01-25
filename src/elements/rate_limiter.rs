//! Rate limiter element for controlling throughput.
//!
//! Provides a transform element that limits the rate of buffer flow through
//! a pipeline, useful for testing, simulation, or controlling resource usage.

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;
use std::time::{Duration, Instant};

/// Rate limiting strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateLimitMode {
    /// Limit by number of buffers per second.
    BuffersPerSecond(f64),
    /// Limit by bytes per second.
    BytesPerSecond(u64),
    /// Fixed delay between buffers.
    FixedDelay(Duration),
}

/// A transform element that limits the rate of buffer flow.
///
/// Useful for:
/// - Testing pipelines at controlled rates
/// - Simulating slower sources
/// - Preventing downstream overload
/// - Bandwidth throttling
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RateLimiter;
/// use std::time::Duration;
///
/// // Limit to 100 buffers per second
/// let limiter = RateLimiter::buffers_per_second(100.0);
///
/// // Limit to 1 MB/s
/// let limiter = RateLimiter::bytes_per_second(1_000_000);
///
/// // Fixed 10ms delay between buffers
/// let limiter = RateLimiter::fixed_delay(Duration::from_millis(10));
/// ```
pub struct RateLimiter {
    name: String,
    mode: RateLimitMode,
    last_time: Option<Instant>,
    bytes_this_second: u64,
    second_start: Option<Instant>,
    total_buffers: u64,
    total_bytes: u64,
    total_delay: Duration,
}

impl RateLimiter {
    /// Create a rate limiter with a maximum number of buffers per second.
    pub fn buffers_per_second(rate: f64) -> Self {
        Self {
            name: format!("rate-limiter-{:.0}buf/s", rate),
            mode: RateLimitMode::BuffersPerSecond(rate),
            last_time: None,
            bytes_this_second: 0,
            second_start: None,
            total_buffers: 0,
            total_bytes: 0,
            total_delay: Duration::ZERO,
        }
    }

    /// Create a rate limiter with a maximum number of bytes per second.
    pub fn bytes_per_second(rate: u64) -> Self {
        Self {
            name: format!("rate-limiter-{}B/s", rate),
            mode: RateLimitMode::BytesPerSecond(rate),
            last_time: None,
            bytes_this_second: 0,
            second_start: None,
            total_buffers: 0,
            total_bytes: 0,
            total_delay: Duration::ZERO,
        }
    }

    /// Create a rate limiter with a fixed delay between buffers.
    pub fn fixed_delay(delay: Duration) -> Self {
        Self {
            name: format!("rate-limiter-{}ms", delay.as_millis()),
            mode: RateLimitMode::FixedDelay(delay),
            last_time: None,
            bytes_this_second: 0,
            second_start: None,
            total_buffers: 0,
            total_bytes: 0,
            total_delay: Duration::ZERO,
        }
    }

    /// Set a custom name for this element.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the total number of buffers processed.
    pub fn total_buffers(&self) -> u64 {
        self.total_buffers
    }

    /// Get the total number of bytes processed.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Get the total delay introduced by this rate limiter.
    pub fn total_delay(&self) -> Duration {
        self.total_delay
    }

    /// Get the current rate limiting mode.
    pub fn mode(&self) -> RateLimitMode {
        self.mode
    }

    fn apply_rate_limit(&mut self, buffer_len: usize) {
        let now = Instant::now();

        match self.mode {
            RateLimitMode::BuffersPerSecond(rate) => {
                if rate <= 0.0 {
                    return;
                }
                let interval = Duration::from_secs_f64(1.0 / rate);

                if let Some(last) = self.last_time {
                    let elapsed = now.duration_since(last);
                    if elapsed < interval {
                        let sleep_time = interval - elapsed;
                        std::thread::sleep(sleep_time);
                        self.total_delay += sleep_time;
                    }
                }
            }
            RateLimitMode::BytesPerSecond(rate) => {
                if rate == 0 {
                    return;
                }

                // Reset counters every second
                if let Some(start) = self.second_start {
                    let elapsed = now.duration_since(start);
                    if elapsed >= Duration::from_secs(1) {
                        self.second_start = Some(now);
                        self.bytes_this_second = 0;
                    }
                } else {
                    self.second_start = Some(now);
                }

                // Check if we need to wait
                let new_total = self.bytes_this_second + buffer_len as u64;
                if new_total > rate {
                    // Calculate how long to wait
                    let excess = new_total - rate;
                    let wait_fraction = excess as f64 / rate as f64;
                    let sleep_time = Duration::from_secs_f64(wait_fraction);
                    std::thread::sleep(sleep_time);
                    self.total_delay += sleep_time;

                    // Reset for new second
                    self.second_start = Some(Instant::now());
                    self.bytes_this_second = buffer_len as u64;
                } else {
                    self.bytes_this_second = new_total;
                }
            }
            RateLimitMode::FixedDelay(delay) => {
                if let Some(last) = self.last_time {
                    let elapsed = now.duration_since(last);
                    if elapsed < delay {
                        let sleep_time = delay - elapsed;
                        std::thread::sleep(sleep_time);
                        self.total_delay += sleep_time;
                    }
                }
            }
        }

        self.last_time = Some(Instant::now());
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::buffers_per_second(1000.0)
    }
}

impl Element for RateLimiter {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let len = buffer.len();

        self.apply_rate_limit(len);

        self.total_buffers += 1;
        self.total_bytes += len as u64;

        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    fn create_test_buffer(size: usize) -> Buffer {
        let segment = Arc::new(HeapSegment::new(size).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::default())
    }

    #[test]
    fn test_rate_limiter_creation() {
        let limiter = RateLimiter::buffers_per_second(100.0);
        assert_eq!(limiter.total_buffers(), 0);
        assert_eq!(limiter.total_bytes(), 0);
    }

    #[test]
    fn test_rate_limiter_modes() {
        let limiter = RateLimiter::buffers_per_second(50.0);
        assert!(matches!(limiter.mode(), RateLimitMode::BuffersPerSecond(r) if r == 50.0));

        let limiter = RateLimiter::bytes_per_second(1000);
        assert!(matches!(
            limiter.mode(),
            RateLimitMode::BytesPerSecond(1000)
        ));

        let limiter = RateLimiter::fixed_delay(Duration::from_millis(10));
        assert!(
            matches!(limiter.mode(), RateLimitMode::FixedDelay(d) if d == Duration::from_millis(10))
        );
    }

    #[test]
    fn test_rate_limiter_process() {
        let mut limiter = RateLimiter::buffers_per_second(10000.0); // High rate for fast test
        let buffer = create_test_buffer(100);

        let result = limiter.process(buffer).unwrap();
        assert!(result.is_some());
        assert_eq!(limiter.total_buffers(), 1);
        assert_eq!(limiter.total_bytes(), 100);
    }

    #[test]
    fn test_rate_limiter_fixed_delay() {
        let mut limiter = RateLimiter::fixed_delay(Duration::from_millis(10));

        let start = Instant::now();

        // Process 3 buffers
        for _ in 0..3 {
            let buffer = create_test_buffer(10);
            limiter.process(buffer).unwrap();
        }

        let elapsed = start.elapsed();

        // Should have waited at least 20ms (2 delays, first buffer has no delay)
        assert!(
            elapsed >= Duration::from_millis(15),
            "Expected at least 15ms, got {:?}",
            elapsed
        );
        assert_eq!(limiter.total_buffers(), 3);
    }

    #[test]
    fn test_rate_limiter_buffers_per_second() {
        let mut limiter = RateLimiter::buffers_per_second(100.0); // 10ms between buffers

        let start = Instant::now();

        // Process 3 buffers
        for _ in 0..3 {
            let buffer = create_test_buffer(10);
            limiter.process(buffer).unwrap();
        }

        let elapsed = start.elapsed();

        // Should have waited at least 20ms (2 intervals)
        assert!(
            elapsed >= Duration::from_millis(15),
            "Expected at least 15ms, got {:?}",
            elapsed
        );
    }

    #[test]
    fn test_rate_limiter_with_name() {
        let limiter = RateLimiter::buffers_per_second(100.0).with_name("my-limiter");
        assert_eq!(limiter.name(), "my-limiter");
    }

    #[test]
    fn test_rate_limiter_default() {
        let limiter = RateLimiter::default();
        assert!(matches!(limiter.mode(), RateLimitMode::BuffersPerSecond(_)));
    }

    #[test]
    fn test_rate_limiter_total_delay() {
        let mut limiter = RateLimiter::fixed_delay(Duration::from_millis(20));

        // Process 3 buffers
        for _ in 0..3 {
            let buffer = create_test_buffer(10);
            limiter.process(buffer).unwrap();
        }

        // Total delay should be at least 40ms (2 waits)
        assert!(
            limiter.total_delay() >= Duration::from_millis(30),
            "Expected at least 30ms delay, got {:?}",
            limiter.total_delay()
        );
    }
}
