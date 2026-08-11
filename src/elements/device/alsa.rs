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
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, PollDescriptors, ValueOr};
use tokio::io::unix::AsyncFd;

use crate::clock::{Clock, ClockFlags, ClockProvider, ClockTime};
use crate::element::{
    AsyncSink, AsyncSource, ConsumeContext, ExecutionHints, ProduceContext, ProduceResult,
};
use crate::error::Result;
use crate::format::{
    AudioFormatCaps, CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps,
};
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

/// How many underruns one buffer may recover from before giving up.
///
/// A device that keeps underrunning on the same buffer is not going to start
/// working; without a bound the write loop would spin on it forever.
const MAX_UNDERRUN_RETRIES: u32 = 8;

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

    /// The pipeline caps sample format this maps onto.
    ///
    /// Parallax's [`SampleFormat`] carries no endianness — everything in the
    /// tree is native-endian, and ALSA is configured little-endian, so on the
    /// platforms this crate supports the two agree.
    ///
    /// [`SampleFormat`]: crate::format::SampleFormat
    pub fn to_caps_format(self) -> crate::format::SampleFormat {
        use crate::format::SampleFormat;
        match self {
            AlsaSampleFormat::S16LE => SampleFormat::S16,
            AlsaSampleFormat::S32LE => SampleFormat::S32,
            AlsaSampleFormat::F32LE => SampleFormat::F32,
            AlsaSampleFormat::U8 => SampleFormat::U8,
        }
    }
}

/// The caps an ALSA device advertises: exactly what it was configured for.
///
/// The device is opened in one fixed rate/channel-count/sample format, so
/// pinning all three is honest — and it is what lets the negotiation solver
/// insert an `audioconvert`/`audioresample` in front of a mismatched upstream
/// (or fail `prepare()` with an actionable message under the default
/// `ConverterPolicy::Deny`) instead of the device rendering noise.
fn alsa_media_caps(format: &AlsaFormat) -> ElementMediaCaps {
    let caps = AudioFormatCaps {
        sample_rate: CapsValue::Fixed(format.sample_rate),
        channels: CapsValue::Fixed(format.channels as u16),
        sample_format: CapsValue::Fixed(format.format.to_caps_format()),
    };

    ElementMediaCaps::new(vec![FormatMemoryCap::new(
        FormatCaps::AudioRaw(caps),
        MemoryCaps::cpu_only(),
    )])
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
            let current_nanos = htstamp.tv_sec * 1_000_000_000 + htstamp.tv_nsec;

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
        if let Some(ref flow_state) = self.flow_state
            && !flow_state.should_produce()
        {
            // Drop audio samples due to backpressure
            self.samples_dropped += 1;
            flow_state.record_drop();

            if self.samples_dropped == 1 || self.samples_dropped.is_multiple_of(100) {
                tracing::warn!(
                    "ALSA: dropping audio due to backpressure (total dropped: {})",
                    self.samples_dropped
                );
            }

            // Still need to drain the ALSA buffer to prevent overrun
            // Read and discard
            let io = self.pcm.io_bytes();
            let mut discard_buf = vec![0u8; max_frames * self.frame_size];
            let _ = io.readi(&mut discard_buf);

            return Ok(ProduceResult::WouldBlock);
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

        // Read available frames. Bytes, not `i16`: the device may well be
        // configured for S32/F32/U8, and `io_bytes` converts the byte count to
        // frames through whatever format it actually has (#70).
        let io = self.pcm.io_bytes();

        let output = ctx.output();
        // Whole frames only — a partial frame at the tail is not readable.
        let readable = max_frames * self.frame_size;

        match io.readi(&mut output[..readable]) {
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

    fn output_media_caps(&self) -> ElementMediaCaps {
        alsa_media_caps(&self.format)
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
    /// Clock provider for automatic pipeline clock selection.
    clock_provider: AlsaSinkClockProvider,
    /// Frames handed to the device so far.
    written_frames: u64,
    /// Device playback position, shared with the clock.
    position: Arc<AlsaPosition>,
}

/// Clock provider wrapper for AlsaSink.
///
/// This is needed because `alsa::PCM` contains raw pointers and is `!Sync`,
/// so AlsaSink can't implement `ClockProvider` directly. Instead, we create
/// an `AlsaClock` at construction time (which only needs the sample rate)
/// and wrap it in this `Sync`-safe provider.
struct AlsaSinkClockProvider {
    clock: Arc<AlsaClock>,
}

impl ClockProvider for AlsaSinkClockProvider {
    fn provide_clock(&self) -> Option<Arc<dyn Clock>> {
        Some(self.clock.clone())
    }

    fn clock_priority(&self) -> u32 {
        150 // Hardware audio clock
    }
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

        let position = Arc::new(AlsaPosition::new(format.sample_rate));
        let clock_provider = AlsaSinkClockProvider {
            clock: Arc::new(AlsaClock::from_position(Arc::clone(&position))),
        };

        Ok(Self {
            pcm,
            format,
            frame_size,
            clock_provider,
            written_frames: 0,
            position,
        })
    }

    /// The device playback position this sink publishes.
    ///
    /// Clone it before `start()` if you want to observe playback progress from
    /// outside the pipeline — the sink itself moves into its executor task.
    pub fn position(&self) -> Arc<AlsaPosition> {
        Arc::clone(&self.position)
    }

    /// Publish how far the device has actually got.
    ///
    /// `snd_pcm_delay` is the frames still queued ahead of the write pointer,
    /// so written-minus-delay is what has been *played*. A failed query leaves
    /// the last position standing rather than guessing.
    fn publish_position(&self) {
        let Ok(delay) = self.pcm.delay() else {
            return;
        };
        let queued = delay.max(0) as u64;
        self.position.update(
            self.written_frames.saturating_sub(queued),
            std::time::Instant::now(),
        );
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

    /// Create a clock reading this device's playback position.
    ///
    /// The same clock `select_clock` picks up automatically; this is for
    /// callers that want to read it directly.
    pub fn create_clock(&self) -> AlsaClock {
        AlsaClock::from_position(Arc::clone(&self.position))
    }
}

/// How far the clock may extrapolate past its last position report.
///
/// Between two `writei` calls nothing updates the position, so the clock
/// interpolates with elapsed wall time to stay smooth. That is only honest for
/// about a period; beyond it the device may have stopped, and a clock that
/// kept running would drift away from the audio actually being heard.
const MAX_INTERPOLATION: Duration = Duration::from_millis(100);

/// The device's playback position, shared between the sink and its clock.
///
/// `alsa::PCM` holds raw pointers and is `!Sync`, so the clock cannot hold the
/// handle (that is why [`AlsaSinkClockProvider`] exists at all). Instead the
/// sink publishes how many frames the device has actually consumed each time it
/// writes, and the clock reads that — a pair of atomics across the gap rather
/// than a shared handle.
///
/// All arithmetic here is plain integers over an injected `Instant`, so the
/// behaviour under underrun and stall is unit-testable without a sound card.
#[derive(Debug)]
pub struct AlsaPosition {
    /// Frames the device reports as played.
    frames_played: AtomicU64,
    /// Nanoseconds since `epoch` when that count was published, plus one.
    ///
    /// Zero means "no report yet", which is not the same as "reported zero at
    /// the epoch": without the distinction a device that has never played a
    /// sample would interpolate a full window off its start and hand
    /// `PipelineClock::start` a non-zero base time.
    updated_at_nanos: AtomicU64,
    /// Highest time ever reported, so the clock cannot run backwards.
    last_reported_nanos: AtomicU64,
    /// Reference instant for the two nanosecond fields above.
    epoch: std::time::Instant,
    /// Configured sample rate, for frames → time.
    sample_rate: u32,
    /// Nanoseconds since `epoch` when the stream ended, plus one (0 = live).
    ///
    /// After release the clock continues at wall rate from where the device
    /// left off. The stream *ending* is not a stall: paced sinks on other
    /// branches still need time to advance to present their tail, and a
    /// frozen master clock would park them forever.
    released_at_nanos: AtomicU64,
    /// The derived playback time captured at release.
    released_base_nanos: AtomicU64,
}

impl AlsaPosition {
    /// A position that has not seen any playback yet.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            frames_played: AtomicU64::new(0),
            updated_at_nanos: AtomicU64::new(0),
            last_reported_nanos: AtomicU64::new(0),
            epoch: std::time::Instant::now(),
            sample_rate: sample_rate.max(1),
            released_at_nanos: AtomicU64::new(0),
            released_base_nanos: AtomicU64::new(0),
        }
    }

    /// End-of-stream: freeze the device-derived time base and continue at
    /// wall rate from it. Idempotent — the first release wins.
    pub fn release(&self, at: std::time::Instant) {
        // Capture the base *before* claiming the flag, so the winning value
        // is computed from live-device time. The claim decides idempotency;
        // a reader racing between claim and base-store sees base 0, and the
        // monotonic fetch_max in `now_at` absorbs it — `now_at` here has
        // already pushed `last_reported_nanos` to at least `base`.
        let base = self.now_at(at).nanos();
        let at_nanos =
            (at.saturating_duration_since(self.epoch).as_nanos() as u64).saturating_add(1);
        if self
            .released_at_nanos
            .compare_exchange(0, at_nanos, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.released_base_nanos.store(base, Ordering::Release);
        }
    }

    /// Publish a new device position, observed at `at`.
    pub fn update(&self, frames_played: u64, at: std::time::Instant) {
        self.frames_played.store(frames_played, Ordering::Release);
        self.updated_at_nanos.store(
            (at.saturating_duration_since(self.epoch).as_nanos() as u64).saturating_add(1),
            Ordering::Release,
        );
    }

    /// The device's playback time as of `at`.
    ///
    /// Between reports this interpolates with elapsed wall time, capped at
    /// [`MAX_INTERPOLATION`] so a stalled device stops the clock rather than
    /// letting it run away. The result never decreases: an underrun makes the
    /// reported position jump, and a clock that went backwards would send every
    /// PTS-paced sink into an unrecoverable drop loop.
    pub fn now_at(&self, at: std::time::Instant) -> ClockTime {
        // A released clock runs at wall rate from the point playback ended.
        if let Some(released_at) = self
            .released_at_nanos
            .load(Ordering::Acquire)
            .checked_sub(1)
        {
            let base = self.released_base_nanos.load(Ordering::Acquire);
            let now_nanos = at.saturating_duration_since(self.epoch).as_nanos() as u64;
            let time = base.saturating_add(now_nanos.saturating_sub(released_at));
            let previous = self.last_reported_nanos.fetch_max(time, Ordering::AcqRel);
            return ClockTime::from_nanos(time.max(previous));
        }

        let frames = self.frames_played.load(Ordering::Acquire);
        let Some(updated_at) = self.updated_at_nanos.load(Ordering::Acquire).checked_sub(1) else {
            // Nothing has played yet. Report zero rather than interpolating
            // off the epoch — see `updated_at_nanos`.
            return ClockTime::ZERO;
        };

        let played_nanos = frames
            .saturating_mul(1_000_000_000)
            .checked_div(self.sample_rate as u64)
            .unwrap_or(0);

        let now_nanos = at.saturating_duration_since(self.epoch).as_nanos() as u64;
        let since_update = now_nanos.saturating_sub(updated_at);
        let interpolated =
            played_nanos.saturating_add(since_update.min(MAX_INTERPOLATION.as_nanos() as u64));

        // fetch_max, not a load-then-store: two threads may read the clock at
        // once, and the loser must not publish a stale maximum.
        let previous = self
            .last_reported_nanos
            .fetch_max(interpolated, Ordering::AcqRel);
        ClockTime::from_nanos(interpolated.max(previous))
    }

    /// Frames the device has reported playing.
    pub fn frames_played(&self) -> u64 {
        self.frames_played.load(Ordering::Acquire)
    }
}

/// Clock implementation based on ALSA audio hardware timing.
///
/// The time base is what the *device* has consumed — frames written minus
/// `snd_pcm_delay`, converted at the configured sample rate — not wall time.
/// That is what makes it usable as the A/V master clock: when the sound card
/// runs slightly fast or slow, or stalls, video paced against this clock
/// follows the audio rather than drifting away from it.
///
/// Until #67 this was `Instant::elapsed()` wearing a `HARDWARE` badge.
///
/// Priority: 150 (hardware audio clock, preferred over software clocks)
pub struct AlsaClock {
    position: Arc<AlsaPosition>,
}

impl AlsaClock {
    /// Create a clock with its own, unbound position.
    ///
    /// Nothing updates the returned clock, so it reads zero forever — useful
    /// for tests and for callers that only want the trait object's shape.
    /// [`AlsaSink`] builds a bound one, which is what `select_clock` picks up.
    pub fn new(sample_rate: u32) -> Self {
        Self::from_position(Arc::new(AlsaPosition::new(sample_rate)))
    }

    /// Create a clock reading the given device position.
    pub fn from_position(position: Arc<AlsaPosition>) -> Self {
        Self { position }
    }

    /// The position cell this clock reads.
    pub fn position(&self) -> &Arc<AlsaPosition> {
        &self.position
    }
}

impl Clock for AlsaClock {
    fn now(&self) -> ClockTime {
        self.position.now_at(std::time::Instant::now())
    }

    fn flags(&self) -> ClockFlags {
        // ALSA provides a hardware-based clock that can be master
        ClockFlags::CAN_BE_MASTER | ClockFlags::HARDWARE
    }

    fn resolution(&self) -> u64 {
        // Resolution is one sample period: 1/sample_rate seconds
        // At 48kHz, this is ~20.8 microseconds (20833 nanoseconds)
        1_000_000_000u64 / self.position.sample_rate as u64
    }

    fn name(&self) -> &str {
        "alsa-audio-clock"
    }
}

// Note: AlsaSink provides a clock automatically via `as_clock_provider()`.
// The pipeline's `select_clock()` will discover it and use it as the master
// clock. Manual clock setting is still possible via `pipeline.set_clock()`.

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

        // Bytes, not `i16`. `io_bytes()` converts the byte count to frames
        // through the PCM's *configured* format, so one path serves S16, S32,
        // F32 and U8 alike — and there is no `from_raw_parts` on a buffer slice
        // whose alignment we do not control. `io_i16()` used to be hard-coded
        // here, which meant any other configured format simply errored on
        // every buffer (#70).
        let io = self.pcm.io_bytes();

        // `writei` can write fewer frames than asked for. The remainder used to
        // be silently discarded, which drops audio and makes the frame
        // accounting the clock depends on wrong.
        let wanted = frames * self.frame_size;
        let mut written = 0usize;
        let mut underruns = 0u32;

        while written < wanted {
            match io.writei(&data[written..wanted]) {
                Ok(0) => break,
                Ok(frames_written) => {
                    written += frames_written * self.frame_size;
                    self.written_frames += frames_written as u64;
                    // Republish after every accepted chunk: the clock is the
                    // pipeline's master, so it should not go stale inside a
                    // long buffer.
                    self.publish_position();
                }
                Err(e) if e.errno() == libc::EPIPE => {
                    // Underrun: re-prepare and carry on from where we stopped.
                    // Bounded, so a device that will not recover surfaces as an
                    // error instead of spinning forever.
                    underruns += 1;
                    if underruns > MAX_UNDERRUN_RETRIES {
                        return Err(DeviceError::Alsa(format!(
                            "device underran {underruns} times without recovering"
                        ))
                        .into());
                    }
                    tracing::debug!("alsasink: underrun, re-preparing the device");
                    self.pcm
                        .prepare()
                        .map_err(|e| DeviceError::Alsa(e.to_string()))?;
                }
                Err(e) => return Err(DeviceError::Alsa(e.to_string()).into()),
            }
        }

        Ok(())
    }

    fn input_media_caps(&self) -> ElementMediaCaps {
        alsa_media_caps(&self.format)
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }

    fn handle_downstream_event(
        &mut self,
        event: crate::event::Event,
    ) -> Option<crate::event::Event> {
        if matches!(
            event,
            crate::event::Event::Eos | crate::event::Event::Error(_)
        ) {
            // Play out what the device already holds, then release the clock:
            // this stream ending is not a stall, and other paced branches
            // (video presenting its buffered tail) still need time to move.
            let _ = self.pcm.drain();
            self.position().release(std::time::Instant::now());
        }
        Some(event)
    }

    fn as_clock_provider(&self) -> Option<&dyn ClockProvider> {
        Some(&self.clock_provider)
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

    // ------------------------------------------------------------------
    // Configured sample format is honoured (#70)
    // ------------------------------------------------------------------

    #[test]
    fn every_alsa_format_maps_onto_a_caps_format() {
        use crate::format::SampleFormat;

        // The mapping must be total: a format with no caps equivalent would
        // mean a device that cannot advertise what it plays.
        for (alsa, expected, bytes) in [
            (AlsaSampleFormat::S16LE, SampleFormat::S16, 2),
            (AlsaSampleFormat::S32LE, SampleFormat::S32, 4),
            (AlsaSampleFormat::F32LE, SampleFormat::F32, 4),
            (AlsaSampleFormat::U8, SampleFormat::U8, 1),
        ] {
            assert_eq!(alsa.to_caps_format(), expected, "{alsa:?}");
            assert_eq!(alsa.bytes_per_sample(), bytes, "{alsa:?}");
        }
    }

    #[test]
    fn caps_pin_the_configured_format_not_just_any_audio() {
        use crate::format::SampleFormat;

        let format = AlsaFormat {
            sample_rate: 44_100,
            channels: 1,
            format: AlsaSampleFormat::F32LE,
            ..AlsaFormat::default()
        };

        let caps = alsa_media_caps(&format);
        let pair = caps.preferred().expect("one format/memory pair");

        let FormatCaps::AudioRaw(audio) = &pair.format else {
            panic!("ALSA must advertise raw audio, got {:?}", pair.format);
        };

        // All three pinned: this is what makes the solver insert a converter
        // rather than letting `Any ∩ Fixed` pass a mismatch through.
        assert_eq!(audio.sample_rate, CapsValue::Fixed(44_100));
        assert_eq!(audio.channels, CapsValue::Fixed(1));
        assert_eq!(
            audio.sample_format,
            CapsValue::Fixed(SampleFormat::F32),
            "an f32 device must not advertise itself as accepting anything"
        );
    }

    // ------------------------------------------------------------------
    // The clock follows the device, not a stopwatch (#67)
    // ------------------------------------------------------------------

    /// A fixed reference instant, so tests inject time instead of sleeping.
    fn at(position: &AlsaPosition, millis: u64) -> std::time::Instant {
        position.epoch + Duration::from_millis(millis)
    }

    #[test]
    fn a_silent_device_leaves_the_clock_at_zero() {
        // The old clock was `Instant::elapsed()`, so it ran whether or not a
        // single sample had been played.
        let position = AlsaPosition::new(48_000);
        assert_eq!(position.now_at(at(&position, 5_000)), ClockTime::ZERO);
    }

    #[test]
    fn the_clock_advances_with_frames_the_device_consumed() {
        let position = AlsaPosition::new(48_000);

        // Half a second of audio played, reported at t=500ms.
        position.update(24_000, at(&position, 500));
        assert_eq!(
            position.now_at(at(&position, 500)),
            ClockTime::from_nanos(500_000_000)
        );

        // A device running 10% slow: twice the wall time, only 1.5x the audio.
        position.update(36_000, at(&position, 1_000));
        assert_eq!(
            position.now_at(at(&position, 1_000)),
            ClockTime::from_nanos(750_000_000),
            "the clock must follow the device, not the wall"
        );
    }

    #[test]
    fn between_reports_the_clock_interpolates() {
        let position = AlsaPosition::new(48_000);
        position.update(48_000, at(&position, 1_000));

        // 20 ms after the report, with no new one: smooth, not frozen.
        assert_eq!(
            position.now_at(at(&position, 1_020)),
            ClockTime::from_nanos(1_020_000_000)
        );
    }

    #[test]
    fn a_stalled_device_stops_the_clock_rather_than_running_away() {
        let position = AlsaPosition::new(48_000);
        position.update(48_000, at(&position, 1_000));

        // Ten seconds with no further report — the device is gone. The clock
        // may extrapolate one interpolation window and no further, otherwise a
        // paced sink would drop every remaining frame as hopelessly late.
        let stalled = position.now_at(at(&position, 11_000));
        assert_eq!(
            stalled,
            ClockTime::from_nanos(1_000_000_000 + MAX_INTERPOLATION.as_nanos() as u64)
        );
    }

    #[test]
    fn a_released_clock_keeps_running_at_wall_rate() {
        // End of stream is not a stall: the device played its last frame, and
        // other paced branches (video presenting its buffered tail) still
        // need time to advance — a frozen master clock parks them forever.
        let position = AlsaPosition::new(48_000);
        position.update(48_000, at(&position, 1_000));
        position.release(at(&position, 1_000));

        assert_eq!(
            position.now_at(at(&position, 1_500)),
            ClockTime::from_nanos(1_500_000_000),
            "wall rate from the release point"
        );
        assert_eq!(
            position.now_at(at(&position, 11_000)),
            ClockTime::from_nanos(11_000_000_000),
            "no interpolation cap after release"
        );
    }

    #[test]
    fn release_is_idempotent_and_monotonic() {
        let position = AlsaPosition::new(48_000);
        position.update(48_000, at(&position, 1_000));
        let before = position.now_at(at(&position, 1_000));

        position.release(at(&position, 1_000));
        position.release(at(&position, 1_050)); // second release: no effect

        // Still anchored at the first release: wall rate from t=1000, not a
        // re-anchor at t=1050. (Queries only ever move forward — the
        // monotonic guard makes an out-of-order query sticky by design.)
        let after = position.now_at(at(&position, 1_100));
        assert!(after >= before, "release rewound the clock");
        assert_eq!(after, ClockTime::from_nanos(1_100_000_000));
    }

    #[test]
    fn an_underrun_does_not_rewind_the_clock() {
        let position = AlsaPosition::new(48_000);
        position.update(48_000, at(&position, 1_000));
        let before = position.now_at(at(&position, 1_000));

        // An underrun drains the queue, so written-minus-delay can jump about;
        // here the recovery reports a *smaller* played count than before.
        position.update(40_000, at(&position, 1_010));
        let after = position.now_at(at(&position, 1_010));

        assert!(
            after >= before,
            "clock went backwards across an underrun: {before:?} then {after:?}"
        );
    }

    #[test]
    fn playback_resumes_forward_after_an_underrun() {
        let position = AlsaPosition::new(48_000);
        position.update(48_000, at(&position, 1_000));
        position.update(40_000, at(&position, 1_010)); // underrun, clamped
        position.update(96_000, at(&position, 2_000)); // recovered and ahead

        assert_eq!(
            position.now_at(at(&position, 2_000)),
            ClockTime::from_nanos(2_000_000_000),
            "once the device is genuinely ahead again the clamp must let go"
        );
    }

    #[test]
    fn the_clock_reports_the_position_it_was_built_from() {
        let position = Arc::new(AlsaPosition::new(48_000));
        let clock = AlsaClock::from_position(Arc::clone(&position));

        position.update(48_000, std::time::Instant::now());

        assert!(clock.now() >= ClockTime::from_nanos(1_000_000_000));
        assert_eq!(clock.name(), "alsa-audio-clock");
        assert!(clock.flags().contains(ClockFlags::HARDWARE));
        assert!(clock.flags().contains(ClockFlags::CAN_BE_MASTER));
        // One sample at 48 kHz.
        assert_eq!(clock.resolution(), 20_833);
        assert_eq!(position.frames_played(), 48_000);
    }

    #[test]
    fn frame_size_follows_the_configured_format() {
        // The write loop converts frames to bytes with this, so a wrong value
        // here silently truncates or over-reads every buffer.
        for (format, channels, expected) in [
            (AlsaSampleFormat::S16LE, 2, 4),
            (AlsaSampleFormat::S32LE, 2, 8),
            (AlsaSampleFormat::F32LE, 1, 4),
            (AlsaSampleFormat::U8, 2, 2),
        ] {
            assert_eq!(
                format.bytes_per_sample() * channels,
                expected,
                "{format:?} x{channels}"
            );
        }
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
