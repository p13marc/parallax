//! Timeout and debounce elements for timing control.
//!
//! Elements that interact with timing constraints.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Element;
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use crate::metadata::Metadata;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Get the global timeout arena (lazily initialized).
fn timeout_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    // 4KB slots should be sufficient for fallback data, 32 slots for concurrency
    ARENA.get_or_init(|| SharedArena::new(4096, 32).expect("Failed to create timeout arena"))
}

/// A timeout element that produces a fallback buffer if no input arrives in time.
///
/// This is useful for heartbeat/keepalive scenarios where you need to ensure
/// some output even when input stops flowing.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Timeout;
/// use std::time::Duration;
///
/// // Produce empty fallback if no input for 1 second
/// let timeout = Timeout::new(Duration::from_secs(1));
///
/// // Or with custom fallback data
/// let timeout = Timeout::new(Duration::from_secs(1))
///     .with_fallback(b"timeout".to_vec());
/// ```
pub struct Timeout {
    name: String,
    timeout: Duration,
    last_buffer: Option<Instant>,
    fallback_data: Vec<u8>,
    timeouts_triggered: AtomicU64,
    buffers_passed: AtomicU64,
}

impl Timeout {
    /// Create a new timeout element.
    pub fn new(timeout: Duration) -> Self {
        Self {
            name: "timeout".to_string(),
            timeout,
            last_buffer: None,
            fallback_data: Vec::new(),
            timeouts_triggered: AtomicU64::new(0),
            buffers_passed: AtomicU64::new(0),
        }
    }

    /// Create from milliseconds.
    pub fn from_millis(millis: u64) -> Self {
        Self::new(Duration::from_millis(millis))
    }

    /// Set custom fallback data.
    pub fn with_fallback(mut self, data: Vec<u8>) -> Self {
        self.fallback_data = data;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Check if timeout has occurred and return fallback if so.
    pub fn check_timeout(&mut self) -> Result<Option<Buffer>> {
        if let Some(last) = self.last_buffer {
            if last.elapsed() >= self.timeout {
                self.timeouts_triggered.fetch_add(1, Ordering::Relaxed);
                self.last_buffer = Some(Instant::now());
                return self.create_fallback();
            }
        }
        Ok(None)
    }

    fn create_fallback(&self) -> Result<Option<Buffer>> {
        let len = self.fallback_data.len();
        let arena = timeout_arena();
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("arena exhausted".into()))?;

        if !self.fallback_data.is_empty() {
            slot.data_mut()[..len].copy_from_slice(&self.fallback_data);
        }

        let handle = MemoryHandle::with_len(slot, len);
        let mut metadata = Metadata::new();
        metadata.flags = metadata.flags.insert(crate::metadata::BufferFlags::TIMEOUT);

        Ok(Some(Buffer::new(handle, metadata)))
    }

    /// Get statistics.
    pub fn stats(&self) -> TimeoutStats {
        TimeoutStats {
            buffers_passed: self.buffers_passed.load(Ordering::Relaxed),
            timeouts_triggered: self.timeouts_triggered.load(Ordering::Relaxed),
        }
    }
}

impl Element for Timeout {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.last_buffer = Some(Instant::now());
        self.buffers_passed.fetch_add(1, Ordering::Relaxed);
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Timeout element.
#[derive(Debug, Clone, Copy)]
pub struct TimeoutStats {
    /// Buffers that passed through normally.
    pub buffers_passed: u64,
    /// Number of timeout events triggered.
    pub timeouts_triggered: u64,
}

/// A debounce element that suppresses rapid buffer sequences.
///
/// Only passes buffers that arrive after a quiet period. Useful for
/// rate limiting or suppressing bursts.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Debounce;
/// use std::time::Duration;
///
/// // Only pass buffers after 100ms of quiet
/// let debounce = Debounce::new(Duration::from_millis(100));
/// ```
pub struct Debounce {
    name: String,
    quiet_period: Duration,
    last_buffer_time: Option<Instant>,
    last_buffer: Option<Buffer>,
    passed: AtomicU64,
    suppressed: AtomicU64,
}

impl Debounce {
    /// Create a new debounce element.
    pub fn new(quiet_period: Duration) -> Self {
        Self {
            name: "debounce".to_string(),
            quiet_period,
            last_buffer_time: None,
            last_buffer: None,
            passed: AtomicU64::new(0),
            suppressed: AtomicU64::new(0),
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

    /// Check if quiet period has passed and flush held buffer.
    pub fn check_quiet(&mut self) -> Option<Buffer> {
        if let Some(time) = self.last_buffer_time {
            if time.elapsed() >= self.quiet_period {
                self.passed.fetch_add(1, Ordering::Relaxed);
                self.last_buffer_time = None;
                return self.last_buffer.take();
            }
        }
        None
    }

    /// Flush any held buffer immediately.
    pub fn flush(&mut self) -> Option<Buffer> {
        self.last_buffer_time = None;
        let buf = self.last_buffer.take();
        if buf.is_some() {
            self.passed.fetch_add(1, Ordering::Relaxed);
        }
        buf
    }

    /// Get statistics.
    pub fn stats(&self) -> DebounceStats {
        DebounceStats {
            passed: self.passed.load(Ordering::Relaxed),
            suppressed: self.suppressed.load(Ordering::Relaxed),
        }
    }
}

impl Element for Debounce {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // If we have a held buffer, it gets suppressed
        if self.last_buffer.is_some() {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
        }

        self.last_buffer = Some(buffer);
        self.last_buffer_time = Some(Instant::now());

        // Never return immediately - caller should use check_quiet()
        Ok(None)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Debounce element.
#[derive(Debug, Clone, Copy)]
pub struct DebounceStats {
    /// Buffers that were passed after quiet period.
    pub passed: u64,
    /// Buffers that were suppressed (replaced before quiet period).
    pub suppressed: u64,
}

/// A throttle element that limits the rate of buffer flow.
///
/// Drops buffers if they arrive too quickly. Different from RateLimiter
/// which delays rather than drops.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Throttle;
/// use std::time::Duration;
///
/// // Allow at most 1 buffer per 100ms
/// let throttle = Throttle::new(Duration::from_millis(100));
/// ```
pub struct Throttle {
    name: String,
    min_interval: Duration,
    last_passed: Option<Instant>,
    passed: AtomicU64,
    dropped: AtomicU64,
}

impl Throttle {
    /// Create a new throttle element.
    pub fn new(min_interval: Duration) -> Self {
        Self {
            name: "throttle".to_string(),
            min_interval,
            last_passed: None,
            passed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Create from milliseconds.
    pub fn from_millis(millis: u64) -> Self {
        Self::new(Duration::from_millis(millis))
    }

    /// Create limiting to a specific rate (buffers per second).
    pub fn rate(buffers_per_second: f64) -> Self {
        let interval = Duration::from_secs_f64(1.0 / buffers_per_second);
        Self::new(interval)
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> ThrottleStats {
        ThrottleStats {
            passed: self.passed.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}

impl Element for Throttle {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let now = Instant::now();

        let should_pass = match self.last_passed {
            None => true,
            Some(last) => now.duration_since(last) >= self.min_interval,
        };

        if should_pass {
            self.last_passed = Some(now);
            self.passed.fetch_add(1, Ordering::Relaxed);
            Ok(Some(buffer))
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Throttle element.
#[derive(Debug, Clone, Copy)]
pub struct ThrottleStats {
    /// Buffers that passed through.
    pub passed: u64,
    /// Buffers that were dropped due to rate limiting.
    pub dropped: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(64, 64).unwrap())
    }

    fn create_test_buffer(seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    // Timeout tests

    #[test]
    fn test_timeout_passthrough() {
        let mut timeout = Timeout::from_millis(100);

        let buffer = create_test_buffer(42);
        let result = timeout.process(buffer).unwrap();

        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    #[test]
    fn test_timeout_triggers() {
        let mut timeout = Timeout::from_millis(50);

        // Process a buffer
        timeout.process(create_test_buffer(0)).unwrap();

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(60));

        // Check should trigger
        let fallback = timeout.check_timeout().unwrap();
        assert!(fallback.is_some());
        assert!(fallback.unwrap().metadata().flags.is_timeout());
    }

    #[test]
    fn test_timeout_no_trigger_when_active() {
        let mut timeout = Timeout::from_millis(100);

        timeout.process(create_test_buffer(0)).unwrap();

        // Check immediately - should not trigger
        let result = timeout.check_timeout().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_timeout_with_fallback_data() {
        let mut timeout = Timeout::from_millis(10).with_fallback(b"fallback".to_vec());

        timeout.process(create_test_buffer(0)).unwrap();
        std::thread::sleep(Duration::from_millis(20));

        let fallback = timeout.check_timeout().unwrap().unwrap();
        assert_eq!(fallback.as_bytes(), b"fallback");
    }

    // Debounce tests

    #[test]
    fn test_debounce_holds_buffer() {
        let mut debounce = Debounce::from_millis(50);

        // Process should never return immediately
        let result = debounce.process(create_test_buffer(0)).unwrap();
        assert!(result.is_none());

        // Check immediately - not enough quiet time
        assert!(debounce.check_quiet().is_none());
    }

    #[test]
    fn test_debounce_releases_after_quiet() {
        let mut debounce = Debounce::from_millis(30);

        debounce.process(create_test_buffer(42)).unwrap();

        // Wait for quiet period
        std::thread::sleep(Duration::from_millis(40));

        let result = debounce.check_quiet();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    #[test]
    fn test_debounce_suppresses() {
        let mut debounce = Debounce::from_millis(100);

        debounce.process(create_test_buffer(0)).unwrap();
        debounce.process(create_test_buffer(1)).unwrap();
        debounce.process(create_test_buffer(2)).unwrap();

        let stats = debounce.stats();
        assert_eq!(stats.suppressed, 2); // First two were suppressed
    }

    #[test]
    fn test_debounce_flush() {
        let mut debounce = Debounce::from_millis(1000);

        debounce.process(create_test_buffer(42)).unwrap();

        // Flush immediately without waiting
        let result = debounce.flush();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    // Throttle tests

    #[test]
    fn test_throttle_first_passes() {
        let mut throttle = Throttle::from_millis(100);

        let result = throttle.process(create_test_buffer(0)).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_throttle_drops_rapid() {
        let mut throttle = Throttle::from_millis(100);

        // First passes
        let r1 = throttle.process(create_test_buffer(0)).unwrap();
        assert!(r1.is_some());

        // Immediate second should be dropped
        let r2 = throttle.process(create_test_buffer(1)).unwrap();
        assert!(r2.is_none());

        let stats = throttle.stats();
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.dropped, 1);
    }

    #[test]
    fn test_throttle_passes_after_interval() {
        let mut throttle = Throttle::from_millis(30);

        throttle.process(create_test_buffer(0)).unwrap();

        std::thread::sleep(Duration::from_millis(40));

        let result = throttle.process(create_test_buffer(1)).unwrap();
        assert!(result.is_some());

        assert_eq!(throttle.stats().passed, 2);
    }

    #[test]
    fn test_throttle_rate() {
        let throttle = Throttle::rate(10.0); // 10 per second = 100ms interval
        assert!(throttle.min_interval >= Duration::from_millis(99));
        assert!(throttle.min_interval <= Duration::from_millis(101));
    }
}
