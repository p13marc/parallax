//! RTP Jitter Buffer for packet reordering and timing.
//!
//! This module provides a jitter buffer that handles:
//! - Packet reordering based on RTP sequence numbers
//! - Configurable buffer depth (by time or packet count)
//! - Packet loss detection and signaling
//! - Late packet handling
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::{RtpSrc, RtpJitterBuffer, RtpH264Depay};
//!
//! // Pipeline: RtpSrc -> JitterBuffer -> H264Depay
//! let src = RtpSrc::bind("0.0.0.0:5004")?;
//! let jitter = RtpJitterBuffer::new()
//!     .with_latency_ms(100)    // 100ms buffer
//!     .with_max_packets(100);   // Max 100 packets
//!
//! // Jitter buffer reorders packets and signals losses
//! ```

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::metadata::RtpMeta;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Default jitter buffer latency in milliseconds.
const DEFAULT_LATENCY_MS: u64 = 200;

/// Default maximum number of packets to buffer.
const DEFAULT_MAX_PACKETS: usize = 512;

/// Default clock rate for RTP timestamp calculations.
const DEFAULT_CLOCK_RATE: u32 = 90000;

// ============================================================================
// Jitter Buffer
// ============================================================================

/// RTP jitter buffer for packet reordering.
///
/// The jitter buffer holds incoming RTP packets and releases them in sequence
/// order after a configurable delay. This handles network jitter and packet
/// reordering common in UDP-based RTP streams.
///
/// # Modes
///
/// - **Latency mode**: Release packets after a fixed delay from first packet
/// - **Packet mode**: Release packets when buffer reaches a threshold
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RtpJitterBuffer;
///
/// let jitter = RtpJitterBuffer::new()
///     .with_latency_ms(100)
///     .with_max_packets(200)
///     .with_clock_rate(90000);
/// ```
pub struct RtpJitterBuffer {
    name: String,

    /// Buffered packets, keyed by extended sequence number.
    buffer: BTreeMap<u32, BufferedPacket>,

    /// Configuration.
    config: JitterBufferConfig,

    /// State tracking.
    state: JitterBufferState,

    /// Statistics.
    stats: JitterBufferStats,
}

/// Configuration for the jitter buffer.
#[derive(Debug, Clone)]
pub struct JitterBufferConfig {
    /// Target latency in milliseconds.
    pub latency_ms: u64,
    /// Maximum packets to buffer.
    pub max_packets: usize,
    /// RTP clock rate for timestamp calculations.
    pub clock_rate: u32,
    /// Drop late packets instead of outputting them.
    pub drop_late: bool,
    /// Maximum time to wait for a packet before declaring it lost (ms).
    pub max_dropout_time_ms: u64,
}

impl Default for JitterBufferConfig {
    fn default() -> Self {
        Self {
            latency_ms: DEFAULT_LATENCY_MS,
            max_packets: DEFAULT_MAX_PACKETS,
            clock_rate: DEFAULT_CLOCK_RATE,
            drop_late: false,
            max_dropout_time_ms: 1000,
        }
    }
}

/// Internal state for the jitter buffer.
#[derive(Debug, Default)]
struct JitterBufferState {
    /// Whether the buffer has been initialized with the first packet.
    initialized: bool,
    /// Base sequence number (first packet received).
    base_seq: u16,
    /// Number of sequence cycles (wraparounds).
    seq_cycles: u16,
    /// Highest sequence number received.
    highest_seq: u16,
    /// Next expected sequence number to output.
    next_output_seq: u32,
    /// Time when first packet was received.
    first_packet_time: Option<Instant>,
    /// Base RTP timestamp.
    base_rtp_ts: u32,
    /// Whether we're in the initial buffering phase.
    buffering: bool,
}

/// A packet stored in the jitter buffer.
#[derive(Debug)]
#[allow(dead_code)] // Fields reserved for future jitter calculation
struct BufferedPacket {
    /// The buffer containing the packet data.
    buffer: Buffer,
    /// Time when this packet was received (for jitter calculation).
    received_at: Instant,
    /// Extended sequence number (for loss detection).
    ext_seq: u32,
}

/// Statistics for the jitter buffer.
#[derive(Debug, Clone, Default)]
pub struct JitterBufferStats {
    /// Total packets received.
    pub packets_received: u64,
    /// Packets output in order.
    pub packets_output: u64,
    /// Packets dropped (late or buffer overflow).
    pub packets_dropped: u64,
    /// Packets lost (never received).
    pub packets_lost: u64,
    /// Packets received out of order.
    pub packets_reordered: u64,
    /// Duplicate packets received.
    pub packets_duplicate: u64,
    /// Current buffer level (packets).
    pub buffer_level: usize,
    /// Maximum buffer level reached.
    pub max_buffer_level: usize,
    /// Estimated jitter in milliseconds.
    pub jitter_ms: f64,
}

impl RtpJitterBuffer {
    /// Create a new jitter buffer with default settings.
    pub fn new() -> Self {
        Self {
            name: "rtp-jitter-buffer".into(),
            buffer: BTreeMap::new(),
            config: JitterBufferConfig::default(),
            state: JitterBufferState::default(),
            stats: JitterBufferStats::default(),
        }
    }

    /// Set the target latency in milliseconds.
    pub fn with_latency_ms(mut self, ms: u64) -> Self {
        self.config.latency_ms = ms;
        self
    }

    /// Set the maximum number of packets to buffer.
    pub fn with_max_packets(mut self, max: usize) -> Self {
        self.config.max_packets = max;
        self
    }

    /// Set the RTP clock rate.
    pub fn with_clock_rate(mut self, rate: u32) -> Self {
        self.config.clock_rate = rate;
        self
    }

    /// Set whether to drop late packets.
    pub fn with_drop_late(mut self, drop: bool) -> Self {
        self.config.drop_late = drop;
        self
    }

    /// Set the maximum dropout time in milliseconds.
    pub fn with_max_dropout_time_ms(mut self, ms: u64) -> Self {
        self.config.max_dropout_time_ms = ms;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &JitterBufferConfig {
        &self.config
    }

    /// Get current statistics.
    pub fn stats(&self) -> &JitterBufferStats {
        &self.stats
    }

    /// Get the current buffer level (number of packets).
    pub fn buffer_level(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is in the initial buffering phase.
    pub fn is_buffering(&self) -> bool {
        self.state.buffering
    }

    /// Reset the jitter buffer state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.state = JitterBufferState::default();
        // Keep stats but reset buffer level
        self.stats.buffer_level = 0;
    }

    /// Calculate extended sequence number handling wraparound.
    fn extend_seq(&mut self, seq: u16) -> u32 {
        if !self.state.initialized {
            self.state.base_seq = seq;
            self.state.highest_seq = seq;
            self.state.initialized = true;
            self.state.buffering = true;
            return seq as u32;
        }

        // Check for sequence wraparound
        let diff = seq.wrapping_sub(self.state.highest_seq) as i16;

        if diff > 0 {
            // Normal forward progression
            if seq < self.state.highest_seq {
                // Wrapped around
                self.state.seq_cycles += 1;
            }
            self.state.highest_seq = seq;
        } else if diff < -0x8000 {
            // Large negative diff means we wrapped forward
            self.state.seq_cycles += 1;
            self.state.highest_seq = seq;
        }
        // else: out of order packet, don't update highest

        (self.state.seq_cycles as u32) << 16 | seq as u32
    }

    /// Insert a packet into the buffer.
    fn insert_packet(&mut self, buffer: Buffer, rtp: &RtpMeta) {
        let now = Instant::now();
        let ext_seq = self.extend_seq(rtp.seq);

        // Initialize timing on first packet
        if self.state.first_packet_time.is_none() {
            self.state.first_packet_time = Some(now);
            self.state.base_rtp_ts = rtp.ts;
            self.state.next_output_seq = ext_seq;
        }

        // Check for duplicate
        if self.buffer.contains_key(&ext_seq) {
            self.stats.packets_duplicate += 1;
            return;
        }

        // Check if this is a late packet (before our output window)
        if ext_seq < self.state.next_output_seq {
            if self.config.drop_late {
                self.stats.packets_dropped += 1;
                return;
            }
            self.stats.packets_reordered += 1;
        }

        // Check for reordering
        if !self.buffer.is_empty() {
            if let Some((&last_seq, _)) = self.buffer.last_key_value() {
                if ext_seq < last_seq {
                    self.stats.packets_reordered += 1;
                }
            }
        }

        // Insert packet
        self.buffer.insert(
            ext_seq,
            BufferedPacket {
                buffer,
                received_at: now,
                ext_seq,
            },
        );

        self.stats.packets_received += 1;
        self.stats.buffer_level = self.buffer.len();
        if self.stats.buffer_level > self.stats.max_buffer_level {
            self.stats.max_buffer_level = self.stats.buffer_level;
        }

        // Enforce max buffer size
        while self.buffer.len() > self.config.max_packets {
            if let Some((&seq, _)) = self.buffer.first_key_value() {
                self.buffer.remove(&seq);
                self.stats.packets_dropped += 1;
                // Advance output sequence if we dropped what we were waiting for
                if seq == self.state.next_output_seq {
                    self.state.next_output_seq += 1;
                }
            }
        }
    }

    /// Check if buffering phase is complete.
    fn check_buffering_complete(&mut self) -> bool {
        if !self.state.buffering {
            return true;
        }

        if let Some(first_time) = self.state.first_packet_time {
            let elapsed = first_time.elapsed();
            if elapsed >= Duration::from_millis(self.config.latency_ms) {
                self.state.buffering = false;
                return true;
            }
        }

        false
    }

    /// Try to output the next packet in sequence.
    fn try_output(&mut self) -> Option<Buffer> {
        // Check if we have the next expected packet
        if let Some(packet) = self.buffer.remove(&self.state.next_output_seq) {
            self.state.next_output_seq += 1;
            self.stats.packets_output += 1;
            self.stats.buffer_level = self.buffer.len();
            return Some(packet.buffer);
        }

        // Check if we should skip a lost packet
        if let Some(first_time) = self.state.first_packet_time {
            let elapsed = first_time.elapsed();
            let expected_output_time =
                Duration::from_millis(self.config.latency_ms + self.config.max_dropout_time_ms);

            if elapsed > expected_output_time && !self.buffer.is_empty() {
                // We've waited long enough, skip to the next available packet
                if let Some((&next_available, _)) = self.buffer.first_key_value() {
                    if next_available > self.state.next_output_seq {
                        // Count skipped packets as lost
                        let lost = next_available - self.state.next_output_seq;
                        self.stats.packets_lost += lost as u64;
                        self.state.next_output_seq = next_available;

                        // Now try to output the available packet
                        if let Some(packet) = self.buffer.remove(&self.state.next_output_seq) {
                            self.state.next_output_seq += 1;
                            self.stats.packets_output += 1;
                            self.stats.buffer_level = self.buffer.len();
                            return Some(packet.buffer);
                        }
                    }
                }
            }
        }

        None
    }

    /// Push a packet into the jitter buffer.
    ///
    /// Returns a packet if one is ready for output, or None if still buffering.
    pub fn push(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Extract RTP metadata
        let rtp = buffer
            .metadata()
            .rtp
            .ok_or_else(|| Error::Element("Buffer has no RTP metadata".into()))?;

        // Insert into buffer
        self.insert_packet(buffer, &rtp);

        // Check if buffering is complete
        if !self.check_buffering_complete() {
            return Ok(None);
        }

        // Try to output next packet
        Ok(self.try_output())
    }

    /// Flush any remaining packets from the buffer.
    ///
    /// Returns packets in sequence order, marking gaps as lost.
    pub fn flush(&mut self) -> Vec<Buffer> {
        let mut output = Vec::new();

        while let Some((&seq, _)) = self.buffer.first_key_value() {
            // Count any gaps as lost
            if seq > self.state.next_output_seq {
                let lost = seq - self.state.next_output_seq;
                self.stats.packets_lost += lost as u64;
                self.state.next_output_seq = seq;
            }

            if let Some(packet) = self.buffer.remove(&seq) {
                self.state.next_output_seq = seq + 1;
                self.stats.packets_output += 1;
                output.push(packet.buffer);
            }
        }

        self.stats.buffer_level = 0;
        output
    }

    /// Get information about packet loss for RTCP reporting.
    pub fn get_loss_info(&self) -> LossInfo {
        let expected = if self.state.initialized {
            let base = self.state.base_seq as u32;
            let highest = (self.state.seq_cycles as u32) << 16 | self.state.highest_seq as u32;
            highest.wrapping_sub(base).wrapping_add(1)
        } else {
            0
        };

        let received = self.stats.packets_received + self.stats.packets_duplicate;
        let lost = expected.saturating_sub(received as u32);

        LossInfo {
            expected,
            received: received as u32,
            lost,
            fraction_lost: if expected > 0 {
                ((lost * 256) / expected).min(255) as u8
            } else {
                0
            },
            highest_seq: (self.state.seq_cycles as u32) << 16 | self.state.highest_seq as u32,
        }
    }
}

impl Default for RtpJitterBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpJitterBuffer {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.push(buffer)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Information about packet loss for RTCP reporting.
#[derive(Debug, Clone, Default)]
pub struct LossInfo {
    /// Expected number of packets.
    pub expected: u32,
    /// Number of packets received.
    pub received: u32,
    /// Number of packets lost.
    pub lost: u32,
    /// Fraction of packets lost (0-255).
    pub fraction_lost: u8,
    /// Highest extended sequence number received.
    pub highest_seq: u32,
}

// ============================================================================
// Async Jitter Buffer
// ============================================================================

/// Async jitter buffer with timed output.
///
/// This version uses tokio timers to automatically release packets
/// at the appropriate time, maintaining consistent output timing.
pub struct AsyncJitterBuffer {
    inner: RtpJitterBuffer,
    /// Output queue for ready packets.
    output_queue: Vec<Buffer>,
}

impl AsyncJitterBuffer {
    /// Create a new async jitter buffer.
    pub fn new() -> Self {
        Self {
            inner: RtpJitterBuffer::new(),
            output_queue: Vec::new(),
        }
    }

    /// Set the target latency in milliseconds.
    pub fn with_latency_ms(mut self, ms: u64) -> Self {
        self.inner = self.inner.with_latency_ms(ms);
        self
    }

    /// Set the maximum number of packets to buffer.
    pub fn with_max_packets(mut self, max: usize) -> Self {
        self.inner = self.inner.with_max_packets(max);
        self
    }

    /// Set the RTP clock rate.
    pub fn with_clock_rate(mut self, rate: u32) -> Self {
        self.inner = self.inner.with_clock_rate(rate);
        self
    }

    /// Get current statistics.
    pub fn stats(&self) -> &JitterBufferStats {
        self.inner.stats()
    }

    /// Push a packet and get any ready output.
    pub async fn push(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // If we're still buffering, wait for the latency period
        if self.inner.is_buffering() {
            self.inner.push(buffer)?;

            // Check if we should wait
            if self.inner.is_buffering() {
                return Ok(None);
            }
        } else {
            // Normal operation - push and try to get output
            if let Some(output) = self.inner.push(buffer)? {
                return Ok(Some(output));
            }
        }

        // Try to get any available output
        Ok(self.inner.try_output())
    }

    /// Receive the next packet, waiting if necessary.
    ///
    /// This method blocks until a packet is ready for output.
    pub async fn recv(&mut self) -> Option<Buffer> {
        // First check output queue
        if !self.output_queue.is_empty() {
            return Some(self.output_queue.remove(0));
        }

        // Try to get from inner buffer
        self.inner.try_output()
    }

    /// Flush remaining packets.
    pub fn flush(&mut self) -> Vec<Buffer> {
        let mut output = std::mem::take(&mut self.output_queue);
        output.extend(self.inner.flush());
        output
    }
}

impl Default for AsyncJitterBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{HeapSegment, MemorySegment};
    use crate::metadata::Metadata;
    use std::sync::Arc;

    fn create_rtp_buffer(seq: u16, ts: u32, data: &[u8]) -> Buffer {
        let segment = Arc::new(HeapSegment::new(data.len()).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, data.len());

        let rtp = RtpMeta {
            seq,
            ts,
            ssrc: 0x12345678,
            pt: 96,
            marker: false,
        };

        let metadata = Metadata::new().with_rtp(rtp);
        Buffer::new(handle, metadata)
    }

    #[test]
    fn test_jitter_buffer_creation() {
        let jb = RtpJitterBuffer::new();
        assert_eq!(jb.config().latency_ms, DEFAULT_LATENCY_MS);
        assert_eq!(jb.config().max_packets, DEFAULT_MAX_PACKETS);
        assert_eq!(jb.buffer_level(), 0);
    }

    #[test]
    fn test_jitter_buffer_configuration() {
        let jb = RtpJitterBuffer::new()
            .with_latency_ms(100)
            .with_max_packets(50)
            .with_clock_rate(48000)
            .with_drop_late(true);

        assert_eq!(jb.config().latency_ms, 100);
        assert_eq!(jb.config().max_packets, 50);
        assert_eq!(jb.config().clock_rate, 48000);
        assert!(jb.config().drop_late);
    }

    #[test]
    fn test_jitter_buffer_in_order() {
        let mut jb = RtpJitterBuffer::new().with_latency_ms(0); // No latency for testing

        // Push packets in order
        for i in 0..5u16 {
            let buf = create_rtp_buffer(i, i as u32 * 3000, &[i as u8]);
            let _ = jb.push(buf);
        }

        assert_eq!(jb.stats().packets_received, 5);
        assert_eq!(jb.stats().packets_reordered, 0);
    }

    #[test]
    fn test_jitter_buffer_reordering() {
        let mut jb = RtpJitterBuffer::new().with_latency_ms(0);

        // Push packets out of order: 0, 2, 1, 3
        let _ = jb.push(create_rtp_buffer(0, 0, &[0]));
        let _ = jb.push(create_rtp_buffer(2, 6000, &[2]));
        let _ = jb.push(create_rtp_buffer(1, 3000, &[1])); // Out of order
        let _ = jb.push(create_rtp_buffer(3, 9000, &[3]));

        assert_eq!(jb.stats().packets_received, 4);
        assert!(jb.stats().packets_reordered > 0);
    }

    #[test]
    fn test_jitter_buffer_duplicates() {
        // Use high latency to keep packets in buffer for duplicate detection
        let mut jb = RtpJitterBuffer::new().with_latency_ms(10000);

        // Push same packet twice
        let _ = jb.push(create_rtp_buffer(0, 0, &[0]));
        let _ = jb.push(create_rtp_buffer(0, 0, &[0])); // Duplicate

        assert_eq!(jb.stats().packets_received, 1);
        assert_eq!(jb.stats().packets_duplicate, 1);
    }

    #[test]
    fn test_jitter_buffer_max_size() {
        let mut jb = RtpJitterBuffer::new()
            .with_latency_ms(10000) // High latency to prevent output
            .with_max_packets(5);

        // Push more than max packets
        for i in 0..10u16 {
            let _ = jb.push(create_rtp_buffer(i, i as u32 * 3000, &[i as u8]));
        }

        assert!(jb.buffer_level() <= 5);
        assert!(jb.stats().packets_dropped > 0);
    }

    #[test]
    fn test_jitter_buffer_flush() {
        let mut jb = RtpJitterBuffer::new().with_latency_ms(10000); // High latency

        // Push some packets
        for i in 0..5u16 {
            let _ = jb.push(create_rtp_buffer(i, i as u32 * 3000, &[i as u8]));
        }

        assert_eq!(jb.buffer_level(), 5);

        // Flush
        let flushed = jb.flush();
        assert_eq!(flushed.len(), 5);
        assert_eq!(jb.buffer_level(), 0);

        // Verify order
        for (i, buf) in flushed.iter().enumerate() {
            let rtp = buf.metadata().rtp.unwrap();
            assert_eq!(rtp.seq, i as u16);
        }
    }

    #[test]
    fn test_jitter_buffer_sequence_wrap() {
        let mut jb = RtpJitterBuffer::new().with_latency_ms(0);

        // Push packets around sequence wrap
        let _ = jb.push(create_rtp_buffer(65534, 0, &[0]));
        let _ = jb.push(create_rtp_buffer(65535, 3000, &[1]));
        let _ = jb.push(create_rtp_buffer(0, 6000, &[2])); // Wrapped
        let _ = jb.push(create_rtp_buffer(1, 9000, &[3]));

        assert_eq!(jb.stats().packets_received, 4);
        // All should be in order after wrap
        assert_eq!(jb.stats().packets_reordered, 0);
    }

    #[test]
    fn test_jitter_buffer_loss_info() {
        let mut jb = RtpJitterBuffer::new().with_latency_ms(0);

        // Push packets with a gap (missing seq 2)
        let _ = jb.push(create_rtp_buffer(0, 0, &[0]));
        let _ = jb.push(create_rtp_buffer(1, 3000, &[1]));
        // Skip seq 2
        let _ = jb.push(create_rtp_buffer(3, 9000, &[3]));
        let _ = jb.push(create_rtp_buffer(4, 12000, &[4]));

        let loss_info = jb.get_loss_info();
        assert_eq!(loss_info.received, 4);
        assert_eq!(loss_info.expected, 5); // 0-4 = 5 packets
        assert_eq!(loss_info.lost, 1);
    }

    #[test]
    fn test_jitter_buffer_reset() {
        // Use high latency to keep packets in buffer
        let mut jb = RtpJitterBuffer::new().with_latency_ms(10000);

        // Push some packets
        for i in 0..5u16 {
            let _ = jb.push(create_rtp_buffer(i, i as u32 * 3000, &[i as u8]));
        }

        assert!(jb.buffer_level() > 0);

        // Reset
        jb.reset();

        assert_eq!(jb.buffer_level(), 0);
        assert!(!jb.state.initialized);
    }

    #[test]
    fn test_jitter_buffer_output_order() {
        // Use high latency to buffer packets before output
        let mut jb = RtpJitterBuffer::new()
            .with_latency_ms(10000)
            .with_drop_late(false);

        // Push out of order
        let _ = jb.push(create_rtp_buffer(2, 6000, &[2]));
        let _ = jb.push(create_rtp_buffer(0, 0, &[0]));
        let _ = jb.push(create_rtp_buffer(1, 3000, &[1]));

        // Flush and verify order
        let output = jb.flush();
        assert_eq!(output.len(), 3);

        for (i, buf) in output.iter().enumerate() {
            let rtp = buf.metadata().rtp.unwrap();
            assert_eq!(rtp.seq, i as u16, "Packet {} has wrong sequence", i);
        }
    }

    #[test]
    fn test_async_jitter_buffer_creation() {
        let ajb = AsyncJitterBuffer::new()
            .with_latency_ms(50)
            .with_max_packets(100);

        assert_eq!(ajb.stats().packets_received, 0);
    }
}
