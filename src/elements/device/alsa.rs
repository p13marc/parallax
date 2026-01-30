//! ALSA audio capture and playback (fallback).
//!
//! ALSA (Advanced Linux Sound Architecture) provides direct access to audio
//! hardware. This is a fallback for when PipeWire is not available.
//!
//! ## Example
//!
//! ```rust,ignore
//! use parallax::elements::device::alsa::{AlsaSrc, AlsaSink, AlsaFormat};
//!
//! // List available devices
//! let devices = AlsaSrc::enumerate_devices()?;
//! for dev in &devices {
//!     println!("{}: {}", dev.name, dev.description);
//! }
//!
//! // Capture from default device
//! let mic = AlsaSrc::new("default", AlsaFormat::default())?;
//!
//! // Playback to default device
//! let speaker = AlsaSink::new("default", AlsaFormat::default())?;
//! ```

use std::ffi::CString;
use std::sync::atomic::{AtomicI64, Ordering};

use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, PollDescriptors, ValueOr};
use tokio::io::unix::AsyncFd;

use crate::clock::ClockTime;
use crate::element::{
    Affinity, AsyncSink, AsyncSource, ConsumeContext, ExecutionHints, ProduceContext, ProduceResult,
};
use crate::error::Result;
use crate::pipeline::flow::{FlowPolicy, FlowSignal, FlowStateHandle};

use super::DeviceError;

/// Check if ALSA is available on this system.
pub fn is_available() -> bool {
    // Try to open the default device
    PCM::new("default", Direction::Capture, false).is_ok()
        || PCM::new("default", Direction::Playback, false).is_ok()
}

/// Information about an ALSA device.
#[derive(Debug, Clone)]
pub struct AlsaDeviceInfo {
    /// Device name (e.g., "hw:0,0" or "default").
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this device supports capture.
    pub is_capture: bool,
    /// Whether this device supports playback.
    pub is_playback: bool,
}

/// Enumerate ALSA devices.
pub fn enumerate_devices() -> Result<Vec<AlsaDeviceInfo>> {
    let mut devices = Vec::new();

    // Add default devices
    devices.push(AlsaDeviceInfo {
        name: "default".to_string(),
        description: "Default Audio Device".to_string(),
        is_capture: true,
        is_playback: true,
    });

    // Enumerate hardware devices using hints
    let pcm_cstr = CString::new("pcm").unwrap();
    if let Ok(hints) = alsa::device_name::HintIter::new(None, &pcm_cstr) {
        for hint in hints {
            if let Some(name) = hint.name {
                // Skip null device
                if name == "null" {
                    continue;
                }

                let description = hint.desc.unwrap_or_else(|| name.clone());

                // Determine capabilities from name/description
                let is_capture = !name.contains("playback");
                let is_playback = !name.contains("capture");

                devices.push(AlsaDeviceInfo {
                    name,
                    description,
                    is_capture,
                    is_playback,
                });
            }
        }
    }

    Ok(devices)
}

/// ALSA audio format configuration.
#[derive(Debug, Clone)]
pub struct AlsaFormat {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u32,
    /// Sample format.
    pub format: AlsaSampleFormat,
    /// Buffer size in frames.
    pub buffer_frames: u32,
    /// Period size in frames.
    pub period_frames: u32,
}

impl Default for AlsaFormat {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            format: AlsaSampleFormat::S16LE,
            buffer_frames: 4096,
            period_frames: 1024,
        }
    }
}

/// ALSA sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlsaSampleFormat {
    /// Signed 16-bit little-endian.
    S16LE,
    /// Signed 32-bit little-endian.
    S32LE,
    /// 32-bit float little-endian.
    F32LE,
    /// Unsigned 8-bit.
    U8,
}

impl AlsaSampleFormat {
    /// Convert to ALSA format enum.
    fn to_alsa(self) -> Format {
        match self {
            AlsaSampleFormat::S16LE => Format::s16(),
            AlsaSampleFormat::S32LE => Format::s32(),
            AlsaSampleFormat::F32LE => Format::float(),
            AlsaSampleFormat::U8 => Format::U8,
        }
    }

    /// Get bytes per sample.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            AlsaSampleFormat::S16LE => 2,
            AlsaSampleFormat::S32LE => 4,
            AlsaSampleFormat::F32LE => 4,
            AlsaSampleFormat::U8 => 1,
        }
    }
}

/// ALSA audio capture source.
pub struct AlsaSrc {
    /// PCM device.
    pcm: PCM,
    /// Audio format.
    format: AlsaFormat,
    /// Frame size in bytes.
    frame_size: usize,
    /// Flow state handle for downstream backpressure monitoring.
    flow_state: Option<FlowStateHandle>,
    /// Samples dropped due to backpressure.
    samples_dropped: u64,
    /// First ALSA timestamp in nanoseconds (for relative PTS calculation).
    /// i64::MIN indicates "not yet set".
    first_timestamp_nanos: AtomicI64,
    /// Number of frames produced (for calculating PTS from sample count).
    frames_produced: u64,
}

impl AlsaSrc {
    /// Create a capture source for the given device.
    pub fn new(device: &str, format: AlsaFormat) -> Result<Self> {
        let pcm = PCM::new(device, Direction::Capture, false).map_err(|e| {
            if e.errno() == libc::ENOENT {
                DeviceError::NotFound(device.to_string())
            } else if e.errno() == libc::EACCES {
                DeviceError::PermissionDenied(device.to_string())
            } else if e.errno() == libc::EBUSY {
                DeviceError::Busy(device.to_string())
            } else {
                DeviceError::Alsa(e.to_string())
            }
        })?;

        // Configure hardware parameters
        {
            let hwp = HwParams::any(&pcm).map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_access(Access::RWInterleaved)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_format(format.format.to_alsa())
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_channels(format.channels)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_rate(format.sample_rate, ValueOr::Nearest)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_buffer_size(format.buffer_frames as i64)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_period_size(format.period_frames as i64, ValueOr::Nearest)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            pcm.hw_params(&hwp)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;
        }

        // Start the capture
        pcm.prepare()
            .map_err(|e| DeviceError::Alsa(e.to_string()))?;

        let frame_size = format.format.bytes_per_sample() * format.channels as usize;

        Ok(Self {
            pcm,
            format,
            frame_size,
            flow_state: None,
            samples_dropped: 0,
            first_timestamp_nanos: AtomicI64::new(i64::MIN),
            frames_produced: 0,
        })
    }

    /// Get the audio format.
    pub fn format(&self) -> &AlsaFormat {
        &self.format
    }

    /// Enumerate available capture devices.
    pub fn enumerate_devices() -> Result<Vec<AlsaDeviceInfo>> {
        let all = enumerate_devices()?;
        Ok(all.into_iter().filter(|d| d.is_capture).collect())
    }

    /// Set the flow state handle for downstream backpressure monitoring.
    ///
    /// When set, the source will check this handle before producing data.
    /// If downstream signals backpressure (Busy), audio samples will be dropped
    /// to prevent lag buildup.
    pub fn set_flow_state(&mut self, handle: FlowStateHandle) {
        self.flow_state = Some(handle);
    }

    /// Get the number of samples dropped due to backpressure.
    pub fn samples_dropped(&self) -> u64 {
        self.samples_dropped
    }

    /// Calculate relative PTS from ALSA timestamp or sample count.
    ///
    /// Tries to use hardware timestamps from ALSA status. Falls back to
    /// calculating PTS from sample count if hardware timestamps aren't available.
    fn calculate_pts(&self, frames_read: usize) -> ClockTime {
        // Try to get hardware timestamp from ALSA status
        if let Ok(status) = self.pcm.status() {
            let htstamp = status.get_htstamp();
            let current_nanos = htstamp.tv_sec as i64 * 1_000_000_000 + htstamp.tv_nsec as i64;

            // Only use hardware timestamp if it's valid (non-zero)
            if current_nanos > 0 {
                // Try to set the first timestamp atomically
                let _ = self.first_timestamp_nanos.compare_exchange(
                    i64::MIN,
                    current_nanos,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );

                let first_nanos = self.first_timestamp_nanos.load(Ordering::SeqCst);
                let relative_nanos = (current_nanos - first_nanos).max(0) as u64;
                return ClockTime::from_nanos(relative_nanos);
            }
        }

        // Fallback: calculate PTS from sample count
        // PTS = (frames_produced * 1_000_000_000) / sample_rate
        let total_frames = self.frames_produced + frames_read as u64;
        let nanos = total_frames * 1_000_000_000 / self.format.sample_rate as u64;
        ClockTime::from_nanos(nanos)
    }

    /// Get poll descriptors for async waiting.
    fn poll_descriptors(&self) -> Result<Vec<libc::pollfd>> {
        let count = PollDescriptors::count(&self.pcm);

        let mut fds = vec![
            libc::pollfd {
                fd: 0,
                events: 0,
                revents: 0
            };
            count
        ];

        PollDescriptors::fill(&self.pcm, &mut fds).map_err(|e| DeviceError::Alsa(e.to_string()))?;

        Ok(fds)
    }
}

impl AsyncSource for AlsaSrc {
    async fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        // Calculate frames that fit in output buffer
        let max_frames = ctx.output().len() / self.frame_size;
        if max_frames == 0 {
            return Ok(ProduceResult::WouldBlock);
        }

        // Check for downstream backpressure before reading
        // ALSA is a live source - dropping samples is better than accumulating lag
        if let Some(ref flow_state) = self.flow_state {
            if !flow_state.should_produce() {
                // Drop audio samples due to backpressure
                self.samples_dropped += 1;
                flow_state.record_drop();

                if self.samples_dropped == 1 || self.samples_dropped % 100 == 0 {
                    tracing::warn!(
                        "ALSA: dropping audio due to backpressure (total dropped: {})",
                        self.samples_dropped
                    );
                }

                // Still need to drain the ALSA buffer to prevent overrun
                // Read and discard
                let io = self
                    .pcm
                    .io_i16()
                    .map_err(|e| DeviceError::Alsa(e.to_string()))?;
                let mut discard_buf = vec![0i16; max_frames * self.format.channels as usize];
                let _ = io.readi(&mut discard_buf);

                return Ok(ProduceResult::WouldBlock);
            }
        }

        // Wait for data using poll
        let fds = self.poll_descriptors()?;
        if !fds.is_empty() {
            // Use AsyncFd to wait
            let fd = fds[0].fd;
            if let Ok(async_fd) = AsyncFd::new(fd) {
                let _ = async_fd.readable().await;
            }
        }

        // Read available frames
        let io = self
            .pcm
            .io_i16()
            .map_err(|e| DeviceError::Alsa(e.to_string()))?;

        // Interpret output buffer as i16 slice
        let output = ctx.output();
        let samples = output.len() / 2; // i16 = 2 bytes
        let _frames = samples / self.format.channels as usize;

        // Create a mutable i16 slice from the output buffer
        let output_ptr = output.as_mut_ptr() as *mut i16;
        let output_i16 = unsafe { std::slice::from_raw_parts_mut(output_ptr, samples) };

        match io.readi(output_i16) {
            Ok(frames_read) => {
                // Calculate PTS from ALSA timestamp or sample count
                let pts = self.calculate_pts(frames_read);
                ctx.set_pts(pts);

                // Update frames produced count
                self.frames_produced += frames_read as u64;

                let bytes = frames_read * self.frame_size;
                Ok(ProduceResult::Produced(bytes))
            }
            Err(e) => {
                // Handle underrun
                if e.errno() == libc::EPIPE {
                    self.pcm
                        .prepare()
                        .map_err(|e| DeviceError::Alsa(e.to_string()))?;
                    Ok(ProduceResult::WouldBlock)
                } else {
                    Err(DeviceError::Alsa(e.to_string()).into())
                }
            }
        }
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(self.format.period_frames as usize * self.frame_size)
    }

    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    fn is_rt_safe(&self) -> bool {
        false
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }

    fn handle_flow_signal(&mut self, signal: FlowSignal) {
        // Update our internal state based on downstream signal
        if let Some(ref flow_state) = self.flow_state {
            flow_state.set_signal(signal);
        }
    }

    fn flow_policy(&self) -> FlowPolicy {
        // ALSA is a live source - always use Drop policy to prevent lag
        FlowPolicy::Drop {
            log_drops: true,
            max_consecutive: None,
        }
    }
}

/// ALSA audio playback sink.
pub struct AlsaSink {
    /// PCM device.
    pcm: PCM,
    /// Audio format.
    format: AlsaFormat,
    /// Frame size in bytes.
    frame_size: usize,
}

impl AlsaSink {
    /// Create a playback sink for the given device.
    pub fn new(device: &str, format: AlsaFormat) -> Result<Self> {
        let pcm = PCM::new(device, Direction::Playback, false).map_err(|e| {
            if e.errno() == libc::ENOENT {
                DeviceError::NotFound(device.to_string())
            } else if e.errno() == libc::EACCES {
                DeviceError::PermissionDenied(device.to_string())
            } else if e.errno() == libc::EBUSY {
                DeviceError::Busy(device.to_string())
            } else {
                DeviceError::Alsa(e.to_string())
            }
        })?;

        // Configure hardware parameters
        {
            let hwp = HwParams::any(&pcm).map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_access(Access::RWInterleaved)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_format(format.format.to_alsa())
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_channels(format.channels)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_rate(format.sample_rate, ValueOr::Nearest)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_buffer_size(format.buffer_frames as i64)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            hwp.set_period_size(format.period_frames as i64, ValueOr::Nearest)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;

            pcm.hw_params(&hwp)
                .map_err(|e| DeviceError::Alsa(e.to_string()))?;
        }

        // Prepare for playback
        pcm.prepare()
            .map_err(|e| DeviceError::Alsa(e.to_string()))?;

        let frame_size = format.format.bytes_per_sample() * format.channels as usize;

        Ok(Self {
            pcm,
            format,
            frame_size,
        })
    }

    /// Get the audio format.
    pub fn format(&self) -> &AlsaFormat {
        &self.format
    }

    /// Enumerate available playback devices.
    pub fn enumerate_devices() -> Result<Vec<AlsaDeviceInfo>> {
        let all = enumerate_devices()?;
        Ok(all.into_iter().filter(|d| d.is_playback).collect())
    }

    /// Get poll descriptors for async waiting.
    fn poll_descriptors(&self) -> Result<Vec<libc::pollfd>> {
        let count = PollDescriptors::count(&self.pcm);

        let mut fds = vec![
            libc::pollfd {
                fd: 0,
                events: 0,
                revents: 0
            };
            count
        ];

        PollDescriptors::fill(&self.pcm, &mut fds).map_err(|e| DeviceError::Alsa(e.to_string()))?;

        Ok(fds)
    }
}

impl AsyncSink for AlsaSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let data = ctx.input();
        let frames = data.len() / self.frame_size;

        if frames == 0 {
            return Ok(());
        }

        // Wait for space using poll
        let fds = self.poll_descriptors()?;
        if !fds.is_empty() {
            let fd = fds[0].fd;
            if let Ok(async_fd) = AsyncFd::new(fd) {
                let _ = async_fd.writable().await;
            }
        }

        // Write frames
        let io = self
            .pcm
            .io_i16()
            .map_err(|e| DeviceError::Alsa(e.to_string()))?;

        // Interpret input as i16 slice
        let samples = data.len() / 2;
        let input_ptr = data.as_ptr() as *const i16;
        let input_i16 = unsafe { std::slice::from_raw_parts(input_ptr, samples) };

        match io.writei(input_i16) {
            Ok(_) => Ok(()),
            Err(e) => {
                // Handle underrun
                if e.errno() == libc::EPIPE {
                    self.pcm
                        .prepare()
                        .map_err(|e| DeviceError::Alsa(e.to_string()))?;
                    // Retry write
                    let _ = io.writei(input_i16);
                    Ok(())
                } else {
                    Err(DeviceError::Alsa(e.to_string()).into())
                }
            }
        }
    }

    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    fn is_rt_safe(&self) -> bool {
        false
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        let available = is_available();
        println!("ALSA available: {}", available);
    }

    #[test]
    fn test_enumerate_devices() {
        match enumerate_devices() {
            Ok(devices) => {
                println!("Found {} ALSA devices:", devices.len());
                for dev in &devices {
                    println!(
                        "  {} - {} (capture: {}, playback: {})",
                        dev.name, dev.description, dev.is_capture, dev.is_playback
                    );
                }
            }
            Err(e) => {
                println!("Failed to enumerate: {}", e);
            }
        }
    }
}
