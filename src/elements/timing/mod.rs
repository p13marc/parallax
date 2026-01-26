//! Timing and rate control elements.
//!
//! - [`Delay`], [`AsyncDelay`]: Add fixed delay to buffer flow
//! - [`Timeout`]: Produce fallback on timeout
//! - [`Debounce`]: Suppress rapid buffer sequences
//! - [`Throttle`]: Drop buffers if too rapid
//! - [`RateLimiter`]: Limit buffer throughput rate

mod delay;
mod rate_limiter;
mod timeout;

pub use delay::{AsyncDelay, Delay, DelayStats};
pub use rate_limiter::{RateLimitMode, RateLimiter};
pub use timeout::{Debounce, DebounceStats, Throttle, ThrottleStats, Timeout, TimeoutStats};
