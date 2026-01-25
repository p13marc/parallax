//! RTCP (RTP Control Protocol) elements.
//!
//! Provides RTCP sender and receiver report handling for RTP streams.
//!
//! - [`RtcpSender`]: Sends RTCP Sender Reports (SR) for RTP senders
//! - [`RtcpReceiver`]: Sends RTCP Receiver Reports (RR) for RTP receivers
//! - [`RtcpHandler`]: Combined RTCP handler that tracks statistics and generates reports
//!
//! # RTCP Overview
//!
//! RTCP provides feedback on RTP stream quality:
//! - **Sender Reports (SR)**: Sent by RTP senders with transmission statistics
//! - **Receiver Reports (RR)**: Sent by RTP receivers with reception quality metrics
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::{RtpSrc, RtcpHandler};
//!
//! // Create RTP receiver with RTCP feedback
//! let mut rtp_src = RtpSrc::bind("0.0.0.0:5004")?;
//! let mut rtcp = RtcpHandler::new(0x12345678)
//!     .with_rtcp_port(5005);
//!
//! // Process RTP packets and periodically send RTCP reports
//! loop {
//!     if let Some(buffer) = rtp_src.produce()? {
//!         rtcp.update_reception_stats(&buffer);
//!     }
//!     if rtcp.should_send_report() {
//!         rtcp.send_receiver_report()?;
//!     }
//! }
//! ```

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::metadata::RtpMeta;

use bytes::Bytes;
use rtcp::receiver_report::ReceiverReport;
use rtcp::reception_report::ReceptionReport;
use rtcp::sender_report::SenderReport;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// Use webrtc-util 0.10 for RTCP (matches rtcp crate's dependency)
use webrtc_util_0_10::marshal::{Marshal, Unmarshal};

/// Default RTCP report interval (5 seconds per RFC 3550).
const DEFAULT_RTCP_INTERVAL: Duration = Duration::from_secs(5);

/// Minimum RTCP report interval.
const MIN_RTCP_INTERVAL: Duration = Duration::from_millis(500);

// ============================================================================
// Reception Statistics
// ============================================================================

/// Statistics for a single RTP source being received.
#[derive(Debug, Clone, Default)]
pub struct ReceptionStats {
    /// SSRC of the source being tracked.
    pub ssrc: u32,
    /// Total packets received from this source.
    pub packets_received: u64,
    /// Total bytes received (payload only).
    pub bytes_received: u64,
    /// Total packets lost (estimated).
    pub packets_lost: u32,
    /// Highest sequence number received (extended).
    pub highest_seq: u32,
    /// Sequence number cycles (wraparounds).
    pub seq_cycles: u16,
    /// Base sequence number (first received).
    pub base_seq: Option<u16>,
    /// Last sequence number received.
    pub last_seq: u16,
    /// Interarrival jitter estimate (in timestamp units).
    pub jitter: u32,
    /// Last SR NTP timestamp received (middle 32 bits).
    pub last_sr_ntp: u32,
    /// Time when last SR was received.
    pub last_sr_time: Option<Instant>,
    /// Previous packet arrival time (for jitter calculation).
    prev_arrival: Option<Instant>,
    /// Previous RTP timestamp (for jitter calculation).
    prev_rtp_ts: Option<u32>,
    /// Clock rate for timestamp calculations.
    clock_rate: u32,
}

impl ReceptionStats {
    /// Create new reception stats for a source.
    pub fn new(ssrc: u32, clock_rate: u32) -> Self {
        Self {
            ssrc,
            clock_rate,
            ..Default::default()
        }
    }

    /// Update statistics with a received packet.
    pub fn update(&mut self, rtp: &RtpMeta, arrival_time: Instant) {
        self.packets_received += 1;

        // Initialize base sequence on first packet
        if self.base_seq.is_none() {
            self.base_seq = Some(rtp.seq);
            self.highest_seq = rtp.seq as u32;
        }

        // Track sequence number and detect wraparound
        let seq = rtp.seq;
        let _expected_seq = self.last_seq.wrapping_add(1);

        if seq < self.last_seq && self.last_seq > 0xF000 && seq < 0x1000 {
            // Sequence wrapped around
            self.seq_cycles += 1;
        }

        // Update highest sequence (extended)
        let extended_seq = (self.seq_cycles as u32) << 16 | seq as u32;
        if extended_seq > self.highest_seq {
            self.highest_seq = extended_seq;
        }

        self.last_seq = seq;

        // Calculate jitter (RFC 3550 Appendix A.8)
        if let (Some(prev_arrival), Some(prev_rtp_ts)) = (self.prev_arrival, self.prev_rtp_ts) {
            let arrival_diff = arrival_time.duration_since(prev_arrival);
            let arrival_diff_ts =
                (arrival_diff.as_nanos() as u64 * self.clock_rate as u64 / 1_000_000_000) as i64;

            let rtp_diff = rtp.ts.wrapping_sub(prev_rtp_ts) as i64;
            let d = (arrival_diff_ts - rtp_diff).unsigned_abs() as u32;

            // Jitter = Jitter + (|D| - Jitter) / 16
            self.jitter = self.jitter.wrapping_add(d.wrapping_sub(self.jitter) / 16);
        }

        self.prev_arrival = Some(arrival_time);
        self.prev_rtp_ts = Some(rtp.ts);
    }

    /// Calculate packets lost.
    pub fn calculate_lost(&self) -> u32 {
        let base = self.base_seq.unwrap_or(0) as u32;
        let expected = self.highest_seq.wrapping_sub(base).wrapping_add(1);
        expected.saturating_sub(self.packets_received as u32)
    }

    /// Calculate fraction lost (since last report).
    pub fn fraction_lost(&self, prev_expected: u32, prev_received: u64) -> u8 {
        let base = self.base_seq.unwrap_or(0) as u32;
        let expected_now = self.highest_seq.wrapping_sub(base).wrapping_add(1);
        let expected_interval = expected_now.saturating_sub(prev_expected);
        let received_interval = (self.packets_received - prev_received) as u32;

        if expected_interval == 0 {
            return 0;
        }

        let lost_interval = expected_interval.saturating_sub(received_interval);
        ((lost_interval * 256) / expected_interval).min(255) as u8
    }

    /// Build a reception report for this source.
    pub fn to_reception_report(&self) -> ReceptionReport {
        let delay = if let Some(sr_time) = self.last_sr_time {
            // Delay in 1/65536 seconds
            let elapsed = sr_time.elapsed();
            ((elapsed.as_secs_f64() * 65536.0) as u32).min(u32::MAX)
        } else {
            0
        };

        ReceptionReport {
            ssrc: self.ssrc,
            fraction_lost: 0, // Will be calculated by caller with interval data
            total_lost: self.calculate_lost(),
            last_sequence_number: self.highest_seq,
            jitter: self.jitter,
            last_sender_report: self.last_sr_ntp,
            delay,
        }
    }

    /// Record receipt of a Sender Report.
    pub fn record_sr(&mut self, ntp_time: u64) {
        // Store middle 32 bits of NTP timestamp
        self.last_sr_ntp = ((ntp_time >> 16) & 0xFFFFFFFF) as u32;
        self.last_sr_time = Some(Instant::now());
    }
}

// ============================================================================
// Sender Statistics
// ============================================================================

/// Statistics for an RTP sender.
#[derive(Debug, Clone, Default)]
pub struct SenderStats {
    /// SSRC of this sender.
    pub ssrc: u32,
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total bytes sent (payload only).
    pub bytes_sent: u64,
    /// Last RTP timestamp sent.
    pub last_rtp_ts: u32,
    /// Clock rate for timestamp calculations.
    pub clock_rate: u32,
}

impl SenderStats {
    /// Create new sender stats.
    pub fn new(ssrc: u32, clock_rate: u32) -> Self {
        Self {
            ssrc,
            clock_rate,
            ..Default::default()
        }
    }

    /// Update statistics after sending a packet.
    pub fn update(&mut self, rtp: &RtpMeta, payload_size: usize) {
        self.packets_sent += 1;
        self.bytes_sent += payload_size as u64;
        self.last_rtp_ts = rtp.ts;
    }

    /// Build a sender report.
    pub fn to_sender_report(&self, reception_reports: Vec<ReceptionReport>) -> SenderReport {
        let ntp_time = get_ntp_time();
        let rtp_time = self.last_rtp_ts;

        SenderReport {
            ssrc: self.ssrc,
            ntp_time,
            rtp_time,
            packet_count: self.packets_sent as u32,
            octet_count: self.bytes_sent as u32,
            reports: reception_reports,
            profile_extensions: Bytes::new(),
        }
    }
}

// ============================================================================
// RTCP Handler
// ============================================================================

/// Combined RTCP handler for RTP streams.
///
/// Tracks reception and transmission statistics and generates RTCP reports.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RtcpHandler;
///
/// let mut rtcp = RtcpHandler::new(0x12345678)
///     .with_remote_addr("192.168.1.100:5005")?
///     .with_report_interval(Duration::from_secs(5));
///
/// // Update stats as packets are received/sent
/// rtcp.on_rtp_received(&rtp_meta);
/// rtcp.on_rtp_sent(&rtp_meta, payload_len);
///
/// // Periodically check and send reports
/// if rtcp.should_send_report() {
///     rtcp.send_report()?;
/// }
/// ```
pub struct RtcpHandler {
    /// Our SSRC.
    ssrc: u32,
    /// Clock rate for RTP timestamps.
    clock_rate: u32,
    /// UDP socket for sending RTCP.
    socket: Option<UdpSocket>,
    /// Remote RTCP address.
    remote_addr: Option<SocketAddr>,
    /// Report interval.
    report_interval: Duration,
    /// Last report time.
    last_report_time: Option<Instant>,
    /// Sender statistics (if we're sending).
    sender_stats: Option<SenderStats>,
    /// Reception statistics per source.
    reception_stats: std::collections::HashMap<u32, ReceptionStats>,
    /// Previous expected packets (for fraction lost calculation).
    prev_expected: std::collections::HashMap<u32, u32>,
    /// Previous received packets (for fraction lost calculation).
    prev_received: std::collections::HashMap<u32, u64>,
    /// Statistics for reporting.
    stats: RtcpStats,
}

/// RTCP handler statistics.
#[derive(Debug, Clone, Default)]
pub struct RtcpStats {
    /// Sender reports sent.
    pub sr_sent: u64,
    /// Receiver reports sent.
    pub rr_sent: u64,
    /// Sender reports received.
    pub sr_received: u64,
    /// Receiver reports received.
    pub rr_received: u64,
    /// Bytes sent.
    pub bytes_sent: u64,
    /// Bytes received.
    pub bytes_received: u64,
}

impl RtcpHandler {
    /// Create a new RTCP handler.
    pub fn new(ssrc: u32) -> Self {
        Self {
            ssrc,
            clock_rate: 90000, // Default video clock rate
            socket: None,
            remote_addr: None,
            report_interval: DEFAULT_RTCP_INTERVAL,
            last_report_time: None,
            sender_stats: None,
            reception_stats: std::collections::HashMap::new(),
            prev_expected: std::collections::HashMap::new(),
            prev_received: std::collections::HashMap::new(),
            stats: RtcpStats::default(),
        }
    }

    /// Set the clock rate.
    pub fn with_clock_rate(mut self, rate: u32) -> Self {
        self.clock_rate = rate;
        self
    }

    /// Set the remote RTCP address.
    pub fn with_remote_addr<A: ToSocketAddrs>(mut self, addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;
        self.remote_addr = Some(addr);

        // Create socket if not already created
        if self.socket.is_none() {
            self.socket = Some(UdpSocket::bind("0.0.0.0:0")?);
        }

        Ok(self)
    }

    /// Bind to a specific local port for receiving RTCP.
    pub fn bind<A: ToSocketAddrs>(mut self, addr: A) -> Result<Self> {
        let socket = UdpSocket::bind(addr)?;
        socket.set_nonblocking(true)?;
        self.socket = Some(socket);
        Ok(self)
    }

    /// Set the report interval.
    pub fn with_report_interval(mut self, interval: Duration) -> Self {
        self.report_interval = interval.max(MIN_RTCP_INTERVAL);
        self
    }

    /// Enable sender mode (we're sending RTP).
    pub fn as_sender(mut self) -> Self {
        self.sender_stats = Some(SenderStats::new(self.ssrc, self.clock_rate));
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &RtcpStats {
        &self.stats
    }

    /// Get reception stats for a specific source.
    pub fn reception_stats(&self, ssrc: u32) -> Option<&ReceptionStats> {
        self.reception_stats.get(&ssrc)
    }

    /// Get all reception stats.
    pub fn all_reception_stats(&self) -> &std::collections::HashMap<u32, ReceptionStats> {
        &self.reception_stats
    }

    /// Get sender stats.
    pub fn sender_stats(&self) -> Option<&SenderStats> {
        self.sender_stats.as_ref()
    }

    /// Update reception statistics for a received RTP packet.
    pub fn on_rtp_received(&mut self, rtp: &RtpMeta) {
        let now = Instant::now();
        let stats = self
            .reception_stats
            .entry(rtp.ssrc)
            .or_insert_with(|| ReceptionStats::new(rtp.ssrc, self.clock_rate));
        stats.update(rtp, now);
    }

    /// Update reception statistics from a buffer.
    pub fn on_buffer_received(&mut self, buffer: &Buffer) {
        if let Some(rtp) = &buffer.metadata().rtp {
            self.on_rtp_received(rtp);
        }
    }

    /// Update sender statistics for a sent RTP packet.
    pub fn on_rtp_sent(&mut self, rtp: &RtpMeta, payload_size: usize) {
        if let Some(stats) = &mut self.sender_stats {
            stats.update(rtp, payload_size);
        }
    }

    /// Check if it's time to send an RTCP report.
    pub fn should_send_report(&self) -> bool {
        match self.last_report_time {
            None => true,
            Some(last) => last.elapsed() >= self.report_interval,
        }
    }

    /// Send an RTCP report (SR or RR depending on mode).
    pub fn send_report(&mut self) -> Result<()> {
        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| Error::Config("RTCP socket not configured".into()))?;
        let remote = self
            .remote_addr
            .ok_or_else(|| Error::Config("RTCP remote address not set".into()))?;

        // Build reception reports for all sources we're receiving from
        let mut reception_reports = Vec::new();
        for (ssrc, stats) in &self.reception_stats {
            let prev_expected = *self.prev_expected.get(ssrc).unwrap_or(&0);
            let prev_received = *self.prev_received.get(ssrc).unwrap_or(&0);

            let mut report = stats.to_reception_report();
            report.fraction_lost = stats.fraction_lost(prev_expected, prev_received);
            reception_reports.push(report);

            // Update prev values for next interval
            let base = stats.base_seq.unwrap_or(0) as u32;
            self.prev_expected
                .insert(*ssrc, stats.highest_seq.wrapping_sub(base).wrapping_add(1));
            self.prev_received.insert(*ssrc, stats.packets_received);
        }

        // Send SR if we're a sender, otherwise RR
        let buf: Bytes = if let Some(sender_stats) = &self.sender_stats {
            let sr = sender_stats.to_sender_report(reception_reports);
            let bytes = sr
                .marshal()
                .map_err(|e| Error::Element(format!("RTCP marshal error: {}", e)))?;
            self.stats.sr_sent += 1;
            bytes
        } else {
            let rr = ReceiverReport {
                ssrc: self.ssrc,
                reports: reception_reports,
                profile_extensions: Bytes::new(),
            };
            let bytes = rr
                .marshal()
                .map_err(|e| Error::Element(format!("RTCP marshal error: {}", e)))?;
            self.stats.rr_sent += 1;
            bytes
        };

        socket.send_to(&buf, remote)?;
        self.stats.bytes_sent += buf.len() as u64;
        self.last_report_time = Some(Instant::now());

        Ok(())
    }

    /// Receive and process an RTCP packet.
    pub fn receive(&mut self) -> Result<Option<RtcpPacketInfo>> {
        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| Error::Config("RTCP socket not configured".into()))?;

        let mut buf = [0u8; 1500];
        match socket.recv_from(&mut buf) {
            Ok((n, _sender)) => {
                self.stats.bytes_received += n as u64;
                self.process_rtcp_packet(&buf[..n])
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(Error::Io(e)),
        }
    }

    /// Process a received RTCP packet.
    fn process_rtcp_packet(&mut self, data: &[u8]) -> Result<Option<RtcpPacketInfo>> {
        if data.len() < 4 {
            return Ok(None);
        }

        // Check packet type (second byte, lower 7 bits after removing padding bit from first)
        let pt = data[1];

        match pt {
            200 => {
                // Sender Report
                let mut buf = data;
                let sr = SenderReport::unmarshal(&mut buf)
                    .map_err(|e| Error::Element(format!("RTCP SR parse error: {}", e)))?;

                self.stats.sr_received += 1;

                // Record SR timestamp for DLSR calculation
                if let Some(stats) = self.reception_stats.get_mut(&sr.ssrc) {
                    stats.record_sr(sr.ntp_time);
                }

                Ok(Some(RtcpPacketInfo::SenderReport {
                    ssrc: sr.ssrc,
                    ntp_time: sr.ntp_time,
                    rtp_time: sr.rtp_time,
                    packet_count: sr.packet_count,
                    octet_count: sr.octet_count,
                }))
            }
            201 => {
                // Receiver Report
                let mut buf = data;
                let rr = ReceiverReport::unmarshal(&mut buf)
                    .map_err(|e| Error::Element(format!("RTCP RR parse error: {}", e)))?;

                self.stats.rr_received += 1;

                Ok(Some(RtcpPacketInfo::ReceiverReport {
                    ssrc: rr.ssrc,
                    reports: rr
                        .reports
                        .into_iter()
                        .map(|r| ReceiverReportInfo {
                            ssrc: r.ssrc,
                            fraction_lost: r.fraction_lost,
                            total_lost: r.total_lost,
                            highest_seq: r.last_sequence_number,
                            jitter: r.jitter,
                        })
                        .collect(),
                }))
            }
            _ => {
                // Unknown or unhandled packet type
                Ok(None)
            }
        }
    }
}

/// Information about a received RTCP packet.
#[derive(Debug, Clone)]
pub enum RtcpPacketInfo {
    /// Sender Report received.
    SenderReport {
        /// SSRC of the sender.
        ssrc: u32,
        /// NTP timestamp.
        ntp_time: u64,
        /// RTP timestamp corresponding to NTP time.
        rtp_time: u32,
        /// Total packets sent.
        packet_count: u32,
        /// Total bytes sent.
        octet_count: u32,
    },
    /// Receiver Report received.
    ReceiverReport {
        /// SSRC of the reporter.
        ssrc: u32,
        /// Reception reports for sources.
        reports: Vec<ReceiverReportInfo>,
    },
}

/// Information from a reception report block.
#[derive(Debug, Clone)]
pub struct ReceiverReportInfo {
    /// SSRC being reported on.
    pub ssrc: u32,
    /// Fraction of packets lost.
    pub fraction_lost: u8,
    /// Total packets lost.
    pub total_lost: u32,
    /// Highest sequence number received.
    pub highest_seq: u32,
    /// Interarrival jitter.
    pub jitter: u32,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get current NTP timestamp (64-bit).
fn get_ntp_time() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    // NTP epoch is 1900, Unix epoch is 1970
    // Difference is 2208988800 seconds
    const NTP_EPOCH_OFFSET: u64 = 2208988800;

    let ntp_secs = now.as_secs() + NTP_EPOCH_OFFSET;
    let ntp_frac = ((now.subsec_nanos() as u64) << 32) / 1_000_000_000;

    (ntp_secs << 32) | ntp_frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reception_stats_new() {
        let stats = ReceptionStats::new(0x12345678, 90000);
        assert_eq!(stats.ssrc, 0x12345678);
        assert_eq!(stats.clock_rate, 90000);
        assert_eq!(stats.packets_received, 0);
    }

    #[test]
    fn test_reception_stats_update() {
        let mut stats = ReceptionStats::new(0x12345678, 90000);

        let rtp1 = RtpMeta {
            seq: 1000,
            ts: 0,
            ssrc: 0x12345678,
            pt: 96,
            marker: false,
        };
        stats.update(&rtp1, Instant::now());

        assert_eq!(stats.packets_received, 1);
        assert_eq!(stats.base_seq, Some(1000));
        assert_eq!(stats.last_seq, 1000);

        let rtp2 = RtpMeta {
            seq: 1001,
            ts: 3000,
            ssrc: 0x12345678,
            pt: 96,
            marker: false,
        };
        std::thread::sleep(Duration::from_millis(10));
        stats.update(&rtp2, Instant::now());

        assert_eq!(stats.packets_received, 2);
        assert_eq!(stats.last_seq, 1001);
        assert_eq!(stats.highest_seq, 1001);
    }

    #[test]
    fn test_reception_stats_sequence_wrap() {
        let mut stats = ReceptionStats::new(0x12345678, 90000);

        // Start near max sequence
        let rtp1 = RtpMeta {
            seq: 65534,
            ts: 0,
            ssrc: 0x12345678,
            pt: 96,
            marker: false,
        };
        stats.update(&rtp1, Instant::now());

        let rtp2 = RtpMeta {
            seq: 65535,
            ts: 3000,
            ssrc: 0x12345678,
            pt: 96,
            marker: false,
        };
        stats.update(&rtp2, Instant::now());

        // Wrap around to 0
        let rtp3 = RtpMeta {
            seq: 0,
            ts: 6000,
            ssrc: 0x12345678,
            pt: 96,
            marker: false,
        };
        stats.update(&rtp3, Instant::now());

        assert_eq!(stats.seq_cycles, 1);
        // Extended seq should be 0x10000 (65536)
        assert_eq!(stats.highest_seq, 0x10000);
    }

    #[test]
    fn test_reception_stats_lost_calculation() {
        let mut stats = ReceptionStats::new(0x12345678, 90000);

        // Receive packets 100, 101, 103, 104 (missing 102)
        for seq in [100u16, 101, 103, 104] {
            let rtp = RtpMeta {
                seq,
                ts: seq as u32 * 3000,
                ssrc: 0x12345678,
                pt: 96,
                marker: false,
            };
            stats.update(&rtp, Instant::now());
        }

        assert_eq!(stats.packets_received, 4);
        // Expected: 104 - 100 + 1 = 5, received = 4, lost = 1
        assert_eq!(stats.calculate_lost(), 1);
    }

    #[test]
    fn test_sender_stats() {
        let mut stats = SenderStats::new(0xABCDEF01, 90000);

        let rtp = RtpMeta {
            seq: 1000,
            ts: 90000,
            ssrc: 0xABCDEF01,
            pt: 96,
            marker: true,
        };
        stats.update(&rtp, 1000);

        assert_eq!(stats.packets_sent, 1);
        assert_eq!(stats.bytes_sent, 1000);
        assert_eq!(stats.last_rtp_ts, 90000);
    }

    #[test]
    fn test_rtcp_handler_creation() {
        let handler = RtcpHandler::new(0x12345678)
            .with_clock_rate(48000)
            .with_report_interval(Duration::from_secs(2));

        assert_eq!(handler.ssrc, 0x12345678);
        assert_eq!(handler.clock_rate, 48000);
        assert_eq!(handler.report_interval, Duration::from_secs(2));
    }

    #[test]
    fn test_rtcp_handler_should_send_report() {
        let handler = RtcpHandler::new(0x12345678).with_report_interval(Duration::from_millis(100));

        // Should send immediately (no previous report)
        assert!(handler.should_send_report());
    }

    #[test]
    fn test_rtcp_handler_reception_tracking() {
        let mut handler = RtcpHandler::new(0x12345678);

        let rtp = RtpMeta {
            seq: 1000,
            ts: 0,
            ssrc: 0xAABBCCDD,
            pt: 96,
            marker: false,
        };
        handler.on_rtp_received(&rtp);

        let stats = handler.reception_stats(0xAABBCCDD).unwrap();
        assert_eq!(stats.packets_received, 1);
        assert_eq!(stats.ssrc, 0xAABBCCDD);
    }

    #[test]
    fn test_ntp_time() {
        let ntp = get_ntp_time();
        // NTP time should be non-zero and have reasonable value
        assert!(ntp > 0);
        // High 32 bits should be > 3_900_000_000 (roughly 2024 in NTP seconds)
        let ntp_secs = ntp >> 32;
        assert!(ntp_secs > 3_900_000_000);
    }

    #[test]
    fn test_reception_report_generation() {
        let mut stats = ReceptionStats::new(0x12345678, 90000);

        for seq in 0u16..10 {
            let rtp = RtpMeta {
                seq,
                ts: seq as u32 * 3000,
                ssrc: 0x12345678,
                pt: 96,
                marker: false,
            };
            stats.update(&rtp, Instant::now());
        }

        let report = stats.to_reception_report();
        assert_eq!(report.ssrc, 0x12345678);
        assert_eq!(report.last_sequence_number, 9);
        assert_eq!(report.total_lost, 0);
    }

    #[test]
    fn test_sender_report_generation() {
        let mut stats = SenderStats::new(0xABCDEF01, 90000);

        for i in 0..10 {
            let rtp = RtpMeta {
                seq: i,
                ts: i as u32 * 3000,
                ssrc: 0xABCDEF01,
                pt: 96,
                marker: i == 9,
            };
            stats.update(&rtp, 100);
        }

        let sr = stats.to_sender_report(vec![]);
        assert_eq!(sr.ssrc, 0xABCDEF01);
        assert_eq!(sr.packet_count, 10);
        assert_eq!(sr.octet_count, 1000);
        assert!(sr.ntp_time > 0);
    }
}
