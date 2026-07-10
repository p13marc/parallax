//! V4L2 memory-to-memory (M2M) stateful hardware H.264 encoder.
//!
//! Drives a kernel stateful encoder (Raspberry Pi `bcm2835-codec`, i.MX
//! VPU, Rockchip, and many other SoCs) through the V4L2 M2M API using the
//! [`v4l2r`] crate. Raw frames are queued on the OUTPUT queue, encoded
//! packets are drained from the CAPTURE queue.
//!
//! [`V4l2M2mH264Encoder`] implements [`VideoEncoder`], so it drops into
//! the standard [`EncoderElement`](super::EncoderElement) wrapper:
//!
//! ```rust,ignore
//! let device = find_m2m_encoder(b"H264").ok_or("no hardware encoder")?;
//! let config = V4l2M2mEncoderConfig::new(1280, 720).bitrate(4_000_000);
//! let encoder = V4l2M2mH264Encoder::new(&device, config)?;
//! let element = EncoderElement::new(encoder, VideoFormat {
//!     width: 1280, height: 720,
//!     pixel_format: PixelFormat::Nv12,
//!     framerate: Framerate { num: 30, den: 1 },
//! })?;
//! ```
//!
//! # Memory model
//!
//! Version 1 uses MMAP buffers on both queues and copies frame data in and
//! packet data out. A DMABUF zero-copy path (importing `V4l2Src`'s exported
//! buffers directly) is a planned follow-up.
//!
//! # Testing without hardware
//!
//! The `vicodec` virtual driver (`modprobe vicodec`) implements the same
//! stateful-encoder state machine but only produces FWHT, not H.264.
//! [`V4l2CodedFormat::Fwht`] exists purely so tests can exercise the full
//! queue/drain/re-arm mechanics against it; production use is H.264.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use v4l2r::controls::SafeExtControl;
use v4l2r::controls::codec::{
    VideoBitrate, VideoBitrateMode, VideoGopSize, VideoH264IPeriod, VideoH264Profile,
    VideoPrependSpsPpsToIdr,
};
use v4l2r::device::poller::{DeviceEvent, Poller};
use v4l2r::device::queue::direction::{Capture, Output};
use v4l2r::device::queue::{BuffersAllocated, GetFreeCaptureBuffer, GetFreeOutputBuffer, Queue};
use v4l2r::device::{AllocatedQueue, Device, DeviceConfig, Stream, TryDequeue};
use v4l2r::ioctl::{self, BufferFlags, CtrlWhich, DqBufIoctlError, EncoderCommand, FormatFlags};
use v4l2r::memory::MmapHandle;
use v4l2r::nix::sys::time::{TimeVal, TimeValLike};
use v4l2r::{Format, QueueType};

use super::common::{PixelFormat, VideoFrame};
use super::traits::VideoEncoder;
use crate::error::{Error, Result};

/// Coded (CAPTURE-side) format produced by the M2M device.
///
/// [`Fwht`](Self::Fwht) targets the `vicodec` virtual driver so the queue
/// mechanics can be tested without encoder hardware — it is not a
/// production codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum V4l2CodedFormat {
    /// H.264 / AVC (hardware encoders).
    #[default]
    H264,
    /// FWHT (the `vicodec` virtual test driver).
    Fwht,
}

impl V4l2CodedFormat {
    fn fourcc(self) -> &'static [u8; 4] {
        match self {
            Self::H264 => b"H264",
            Self::Fwht => b"FWHT",
        }
    }
}

/// H.264 profile requested from the hardware encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V4l2H264Profile {
    /// Baseline profile.
    Baseline,
    /// Constrained baseline profile.
    ConstrainedBaseline,
    /// Main profile.
    Main,
    /// High profile.
    High,
}

impl V4l2H264Profile {
    fn to_control(self) -> VideoH264Profile {
        match self {
            Self::Baseline => VideoH264Profile::Baseline,
            Self::ConstrainedBaseline => VideoH264Profile::ConstrainedBaseline,
            Self::Main => VideoH264Profile::Main,
            Self::High => VideoH264Profile::High,
        }
    }
}

/// Configuration for [`V4l2M2mH264Encoder`].
#[derive(Debug, Clone)]
pub struct V4l2M2mEncoderConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Raw input pixel format for the OUTPUT queue. Only [`PixelFormat::Nv12`]
    /// (most hardware) and [`PixelFormat::I420`] are supported.
    pub pixel_format: PixelFormat,
    /// Target bitrate in bits per second (0 = driver default).
    pub bitrate_bps: u32,
    /// GOP length in frames: an IDR every N frames (0 = driver default).
    /// Applied via `V4L2_CID_MPEG_VIDEO_H264_I_PERIOD`, falling back to
    /// `V4L2_CID_MPEG_VIDEO_GOP_SIZE`.
    pub gop_size: u32,
    /// H.264 profile (None = driver default). Ignored for FWHT.
    pub profile: Option<V4l2H264Profile>,
    /// Frame rate as (numerator, denominator), fed to the driver's rate
    /// control via `VIDIOC_S_PARM`.
    pub framerate: (u32, u32),
    /// Number of OUTPUT (raw frame) buffers to allocate.
    pub num_output_buffers: u32,
    /// Number of CAPTURE (encoded packet) buffers to allocate.
    pub num_capture_buffers: u32,
    /// Coded format the CAPTURE queue is set to.
    pub coded_format: V4l2CodedFormat,
}

impl V4l2M2mEncoderConfig {
    /// Create a config with the given dimensions and defaults: NV12 input,
    /// H.264 output, 30 fps, driver-default bitrate and GOP.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixel_format: PixelFormat::Nv12,
            bitrate_bps: 0,
            gop_size: 0,
            profile: None,
            framerate: (30, 1),
            num_output_buffers: 4,
            num_capture_buffers: 4,
            coded_format: V4l2CodedFormat::H264,
        }
    }

    /// Set the raw input pixel format (NV12 or I420).
    pub fn pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Set the target bitrate in bits per second.
    pub fn bitrate(mut self, bps: u32) -> Self {
        self.bitrate_bps = bps;
        self
    }

    /// Set the GOP length (IDR interval) in frames.
    pub fn gop_size(mut self, frames: u32) -> Self {
        self.gop_size = frames;
        self
    }

    /// Set the H.264 profile.
    pub fn profile(mut self, profile: V4l2H264Profile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Set the frame rate as a fraction (numerator, denominator).
    pub fn framerate(mut self, num: u32, den: u32) -> Self {
        self.framerate = (num, den);
        self
    }

    /// Set the coded format ([`V4l2CodedFormat::Fwht`] is test-only).
    pub fn coded_format(mut self, format: V4l2CodedFormat) -> Self {
        self.coded_format = format;
        self
    }
}

/// How long to wait for the hardware before declaring it stalled.
const POLL_TIMEOUT: Duration = Duration::from_secs(2);

/// V4L2 M2M stateful hardware H.264 encoder.
///
/// See the [module docs](self) for usage. Not `Sync`: one pipeline element
/// owns the device.
pub struct V4l2M2mH264Encoder {
    device: Arc<Device>,
    output_queue: Queue<Output, BuffersAllocated<Vec<MmapHandle>>>,
    capture_queue: Queue<Capture, BuffersAllocated<Vec<MmapHandle>>>,
    poller: Poller,
    /// OUTPUT format as applied by the driver (strides may differ from ours).
    output_format: Format,
    config: V4l2M2mEncoderConfig,
    /// Cached SPS+PPS (Annex-B) from the first keyframe packet.
    codec_data: Option<Vec<u8>>,
    /// Set after a flush; the next encode() re-arms with `V4L2_ENC_CMD_START`.
    drained: bool,
    frames_queued: u64,
}

impl V4l2M2mH264Encoder {
    /// Open and configure an M2M encoder device.
    ///
    /// `device` must be a V4L2 M2M node whose CAPTURE side produces the
    /// configured coded format — use [`find_m2m_encoder`] to locate one.
    pub fn new(device: impl AsRef<Path>, config: V4l2M2mEncoderConfig) -> Result<Self> {
        let path = device.as_ref();
        let device = Device::open(path, DeviceConfig::new().non_blocking_dqbuf())
            .map_err(|e| Error::Config(format!("V4L2 M2M: open {}: {e}", path.display())))?;
        let device = Arc::new(device);

        // Single-planar first (vicodec, bcm2835-codec), multi-planar fallback
        // (most SoC encoders expose only the mplane API).
        let (mut output_queue, mut capture_queue) =
            match Queue::get_output_queue(Arc::clone(&device)) {
                Ok(output) => {
                    let capture = Queue::get_capture_queue(Arc::clone(&device)).map_err(|e| {
                        Error::Config(format!("V4L2 M2M: create capture queue: {e}"))
                    })?;
                    (output, capture)
                }
                Err(_) => {
                    let output =
                        Queue::get_output_mplane_queue(Arc::clone(&device)).map_err(|e| {
                            Error::Config(format!("V4L2 M2M: create output queue: {e}"))
                        })?;
                    let capture =
                        Queue::get_capture_mplane_queue(Arc::clone(&device)).map_err(|e| {
                            Error::Config(format!("V4L2 M2M: create capture queue: {e}"))
                        })?;
                    (output, capture)
                }
            };

        // Per the stateful-encoder spec, the coded (CAPTURE) format is set
        // first — pixelformat ONLY. Frame dimensions belong to the raw
        // OUTPUT format and propagate to the coded side; setting a size
        // here instead makes drivers (e.g. vicodec) ignore it and clamp
        // the OUTPUT to their current coded size.
        let coded_fourcc = config.coded_format.fourcc();
        let capture_format: Format = capture_queue
            .change_format()
            .map_err(|e| Error::Config(format!("V4L2 M2M: get capture format: {e}")))?
            .set_pixelformat(coded_fourcc)
            .apply()
            .map_err(|e| Error::Config(format!("V4L2 M2M: set capture format: {e}")))?;
        if capture_format.pixelformat != coded_fourcc.into() {
            return Err(Error::Config(format!(
                "V4L2 M2M: device does not encode {} (offered {})",
                std::str::from_utf8(coded_fourcc).unwrap_or("?"),
                capture_format.pixelformat,
            )));
        }

        let raw_fourcc: &[u8; 4] = match config.pixel_format {
            PixelFormat::Nv12 => b"NV12",
            PixelFormat::I420 => b"YU12",
            other => {
                return Err(Error::Config(format!(
                    "V4L2 M2M: unsupported input format {other:?} (use Nv12 or I420)"
                )));
            }
        };
        let output_format: Format = output_queue
            .change_format()
            .map_err(|e| Error::Config(format!("V4L2 M2M: get output format: {e}")))?
            .set_size(config.width as usize, config.height as usize)
            .set_pixelformat(raw_fourcc)
            .apply()
            .map_err(|e| Error::Config(format!("V4L2 M2M: set output format: {e}")))?;
        if output_format.pixelformat != raw_fourcc.into() {
            return Err(Error::Config(format!(
                "V4L2 M2M: device does not accept {} input (offered {})",
                std::str::from_utf8(raw_fourcc).unwrap_or("?"),
                output_format.pixelformat,
            )));
        }
        if output_format.width != config.width || output_format.height != config.height {
            return Err(Error::Config(format!(
                "V4L2 M2M: driver adjusted {}x{} to {}x{} (alignment constraints); \
                 configure the adjusted size",
                config.width, config.height, output_format.width, output_format.height,
            )));
        }

        Self::apply_controls(&device, &config);

        // Frame rate for the driver's rate control (best-effort; vicodec
        // doesn't implement S_PARM).
        let mut parm = v4l2r::bindings::v4l2_streamparm {
            type_: output_queue.get_type() as u32,
            ..Default::default()
        };
        parm.parm.output.timeperframe = v4l2r::bindings::v4l2_fract {
            // timeperframe is the reciprocal of the frame rate.
            numerator: config.framerate.1,
            denominator: config.framerate.0,
        };
        if let Err(e) = ioctl::s_parm::<_, v4l2r::bindings::v4l2_streamparm>(&*device, parm) {
            tracing::debug!("V4L2 M2M: S_PARM not supported: {e}");
        }

        let output_queue = output_queue
            .request_buffers::<Vec<MmapHandle>>(config.num_output_buffers)
            .map_err(|e| Error::Config(format!("V4L2 M2M: allocate output buffers: {e}")))?;
        let capture_queue = capture_queue
            .request_buffers::<Vec<MmapHandle>>(config.num_capture_buffers)
            .map_err(|e| Error::Config(format!("V4L2 M2M: allocate capture buffers: {e}")))?;

        output_queue
            .stream_on()
            .map_err(|e| Error::Config(format!("V4L2 M2M: stream on output: {e}")))?;
        capture_queue
            .stream_on()
            .map_err(|e| Error::Config(format!("V4L2 M2M: stream on capture: {e}")))?;

        // MMAP capture buffers carry no information: queue them all up front.
        while let Ok(buffer) = capture_queue.try_get_free_buffer() {
            buffer
                .queue()
                .map_err(|e| Error::Config(format!("V4L2 M2M: queue capture buffer: {e}")))?;
        }

        let mut poller = Poller::new(Arc::clone(&device))
            .map_err(|e| Error::Config(format!("V4L2 M2M: create poller: {e}")))?;
        poller
            .enable_event(DeviceEvent::CaptureReady)
            .and_then(|_| poller.enable_event(DeviceEvent::OutputReady))
            .map_err(|e| Error::Config(format!("V4L2 M2M: enable poll events: {e}")))?;

        Ok(Self {
            device,
            output_queue,
            capture_queue,
            poller,
            output_format,
            config,
            codec_data: None,
            drained: false,
            frames_queued: 0,
        })
    }

    /// Apply codec controls. All are best-effort: drivers advertise wildly
    /// different control sets (vicodec supports none of the H.264 CIDs), and
    /// an unsupported knob should not prevent encoding.
    fn apply_controls(device: &Device, config: &V4l2M2mEncoderConfig) {
        if config.bitrate_bps > 0 {
            let mut bitrate = SafeExtControl::<VideoBitrate>::from_value(config.bitrate_bps as i32);
            if let Err(e) = ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut bitrate) {
                tracing::debug!("V4L2 M2M: bitrate control unsupported: {e}");
            }
            let mut mode = SafeExtControl::<VideoBitrateMode>::from_value(
                v4l2r::controls::codec::VideoBitrateMode::ConstantBitrate as i32,
            );
            if let Err(e) = ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut mode) {
                tracing::debug!("V4L2 M2M: bitrate mode control unsupported: {e}");
            }
        }

        if config.gop_size > 0 {
            let mut i_period =
                SafeExtControl::<VideoH264IPeriod>::from_value(config.gop_size as i32);
            if ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut i_period).is_err() {
                let mut gop = SafeExtControl::<VideoGopSize>::from_value(config.gop_size as i32);
                if let Err(e) = ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut gop) {
                    tracing::debug!("V4L2 M2M: GOP size control unsupported: {e}");
                }
            }
        }

        if let Some(profile) = config.profile {
            let mut ctrl =
                SafeExtControl::<VideoH264Profile>::from_value(profile.to_control() as i32);
            if let Err(e) = ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut ctrl) {
                tracing::debug!("V4L2 M2M: H.264 profile control unsupported: {e}");
            }
        }

        // In-band SPS/PPS on every IDR makes streams late-joinable and lets
        // codec_data() cache the headers from the first packet.
        let mut prepend = SafeExtControl::<VideoPrependSpsPpsToIdr>::from_value(1);
        if let Err(e) = ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut prepend) {
            tracing::debug!("V4L2 M2M: prepend SPS/PPS control unsupported: {e}");
        }
    }

    /// Return finished OUTPUT buffers to the free pool.
    fn reclaim_output_buffers(&self) {
        while let Ok(dqbuf) = self.output_queue.try_dequeue() {
            // MMAP handles have nothing to take; dropping frees the slot.
            drop(dqbuf);
        }
    }

    /// Drain every CAPTURE packet the hardware has ready, without blocking.
    fn drain_capture(&mut self) -> Result<Vec<Vec<u8>>> {
        let mut packets = Vec::new();
        loop {
            match self.capture_queue.try_dequeue() {
                Ok(dqbuf) => {
                    let flags = dqbuf.data.flags();
                    if !flags.contains(BufferFlags::ERROR) {
                        let mapping = dqbuf.get_plane_mapping(0).ok_or_else(|| {
                            Error::Element("V4L2 M2M: map capture plane".to_string())
                        })?;
                        if !mapping.is_empty() {
                            let data = mapping.to_vec();
                            self.maybe_cache_codec_data(&data, flags);
                            packets.push(data);
                        }
                    }
                    drop(dqbuf);
                    // Immediately hand the slot back to the hardware.
                    if let Ok(buffer) = self.capture_queue.try_get_free_buffer() {
                        buffer.queue().map_err(|e| {
                            Error::Element(format!("V4L2 M2M: requeue capture buffer: {e}"))
                        })?;
                    }
                }
                Err(v4l2r::ioctl::IoctlConvertError::IoctlError(DqBufIoctlError::NotReady)) => {
                    break;
                }
                Err(e) => {
                    return Err(Error::Element(format!("V4L2 M2M: dequeue capture: {e}")));
                }
            }
        }
        Ok(packets)
    }

    /// Cache SPS/PPS from the first keyframe packet (H.264 only).
    fn maybe_cache_codec_data(&mut self, data: &[u8], flags: BufferFlags) {
        if self.codec_data.is_some()
            || self.config.coded_format != V4l2CodedFormat::H264
            || !flags.contains(BufferFlags::KEYFRAME)
        {
            return;
        }
        let headers = extract_sps_pps(data);
        if !headers.is_empty() {
            self.codec_data = Some(headers);
        }
    }

    /// Wait for a device event, mapping poll failures/timeouts to errors.
    fn wait_for_event(&mut self, context: &str) -> Result<()> {
        let events = self
            .poller
            .poll(Some(POLL_TIMEOUT))
            .map_err(|e| Error::Element(format!("V4L2 M2M: poll ({context}): {e}")))?;
        if events.count() == 0 {
            return Err(Error::Element(format!(
                "V4L2 M2M: encoder stalled waiting for {context} \
                 (no event within {POLL_TIMEOUT:?})"
            )));
        }
        Ok(())
    }
}

impl VideoEncoder for V4l2M2mH264Encoder {
    type Packet = Vec<u8>;

    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Vec<u8>>> {
        if frame.width != self.config.width
            || frame.height != self.config.height
            || frame.format != self.config.pixel_format
        {
            return Err(Error::Element(format!(
                "V4L2 M2M: frame is {}x{} {:?}, encoder configured for {}x{} {:?}",
                frame.width,
                frame.height,
                frame.format,
                self.config.width,
                self.config.height,
                self.config.pixel_format,
            )));
        }

        // Resume after a flush(): the spec keeps both queues streaming, a
        // START command re-arms the encoder.
        if self.drained {
            ioctl::encoder_cmd::<_, ()>(&*self.device, &EncoderCommand::Start)
                .map_err(|e| Error::Element(format!("V4L2 M2M: ENC_CMD_START: {e}")))?;
            while let Ok(buffer) = self.capture_queue.try_get_free_buffer() {
                buffer.queue().map_err(|e| {
                    Error::Element(format!("V4L2 M2M: requeue capture buffer: {e}"))
                })?;
            }
            self.drained = false;
        }

        self.reclaim_output_buffers();
        let qbuf = loop {
            match self.output_queue.try_get_free_buffer() {
                Ok(buffer) => break buffer,
                Err(_) => {
                    // All raw-frame slots are in the hardware; wait for one.
                    self.wait_for_event("a free output buffer")?;
                    self.reclaim_output_buffers();
                }
            }
        };

        let mut mapping = qbuf
            .get_plane_mapping(0)
            .ok_or_else(|| Error::Element("V4L2 M2M: map output plane".to_string()))?;
        let bytes_used = fill_output_plane(&mut mapping, &self.output_format, frame)?;
        drop(mapping);

        let timestamp = TimeVal::microseconds(frame.pts / 1_000);
        qbuf.set_timestamp(timestamp)
            .queue(&[bytes_used])
            .map_err(|e| Error::Element(format!("V4L2 M2M: queue output buffer: {e}")))?;
        self.frames_queued += 1;

        // Non-blocking: stateful encoders have pipeline latency, so early
        // frames legitimately produce nothing (EncoderElement buffers).
        self.drain_capture()
    }

    fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        if self.drained || self.frames_queued == 0 {
            return Ok(Vec::new());
        }
        ioctl::encoder_cmd::<_, ()>(&*self.device, &EncoderCommand::Stop(false))
            .map_err(|e| Error::Element(format!("V4L2 M2M: ENC_CMD_STOP: {e}")))?;

        let mut packets = Vec::new();
        let deadline = Instant::now() + POLL_TIMEOUT;
        'drain: loop {
            match self.capture_queue.try_dequeue() {
                Ok(dqbuf) => {
                    let flags = dqbuf.data.flags();
                    let is_last = dqbuf.data.is_last();
                    if !flags.contains(BufferFlags::ERROR) {
                        let mapping = dqbuf.get_plane_mapping(0).ok_or_else(|| {
                            Error::Element("V4L2 M2M: map capture plane".to_string())
                        })?;
                        if !mapping.is_empty() {
                            let data = mapping.to_vec();
                            self.maybe_cache_codec_data(&data, flags);
                            packets.push(data);
                        }
                    }
                    // Do NOT requeue during a drain: the queue must empty out.
                    drop(dqbuf);
                    if is_last {
                        break 'drain;
                    }
                }
                Err(v4l2r::ioctl::IoctlConvertError::IoctlError(DqBufIoctlError::NotReady)) => {
                    if self.capture_queue.num_queued_buffers() == 0 {
                        // Nothing left for the driver to fill (defensive; the
                        // LAST flag is the normal exit).
                        break 'drain;
                    }
                    if Instant::now() >= deadline {
                        return Err(Error::Element(
                            "V4L2 M2M: drain timed out waiting for the LAST buffer".to_string(),
                        ));
                    }
                    self.wait_for_event("drain completion")?;
                }
                Err(e) => {
                    return Err(Error::Element(format!("V4L2 M2M: dequeue capture: {e}")));
                }
            }
        }

        self.reclaim_output_buffers();
        self.drained = true;
        Ok(packets)
    }

    fn codec_data(&self) -> Option<Vec<u8>> {
        self.codec_data.clone()
    }

    fn force_keyframe(&mut self) {
        // Button control on the same fd as the M2M context. Best-effort:
        // not all drivers implement it.
        if let Err(e) = ioctl::s_ctrl(
            &*self.device,
            v4l2r::bindings::V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME,
            1,
        ) {
            tracing::debug!("V4L2 M2M: FORCE_KEY_FRAME unsupported: {e}");
        }
    }
}

impl std::fmt::Debug for V4l2M2mH264Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V4l2M2mH264Encoder")
            .field("config", &self.config)
            .field("drained", &self.drained)
            .field("frames_queued", &self.frames_queued)
            .finish()
    }
}

impl Drop for V4l2M2mH264Encoder {
    fn drop(&mut self) {
        let _ = self.capture_queue.stream_off();
        let _ = self.output_queue.stream_off();
    }
}

/// Copy a [`VideoFrame`] into a single-plane OUTPUT buffer, honoring the
/// driver's row stride (`bytesperline`). Returns the bytes used.
fn fill_output_plane(dst: &mut [u8], format: &Format, frame: &VideoFrame) -> Result<usize> {
    let plane = format.plane_fmt.first().ok_or_else(|| {
        Error::Element("V4L2 M2M: driver reported a format without planes".to_string())
    })?;
    let dst_stride = plane.bytesperline as usize;
    let sizeimage = plane.sizeimage as usize;
    if dst.len() < sizeimage {
        return Err(Error::Element(format!(
            "V4L2 M2M: output mapping is {} bytes, driver wants {sizeimage}",
            dst.len(),
        )));
    }

    let height = frame.height as usize;
    let row_bytes = frame.stride_y.min(dst_stride);
    copy_plane(
        dst,
        dst_stride,
        frame.y_plane(),
        frame.stride_y,
        height,
        row_bytes,
    );
    let y_size = dst_stride * height;

    match frame.format {
        PixelFormat::Nv12 => {
            // One interleaved UV plane, full row width, half height.
            copy_plane(
                &mut dst[y_size..],
                dst_stride,
                frame.u_plane(),
                frame.stride_u,
                height / 2,
                row_bytes,
            );
        }
        PixelFormat::I420 => {
            // Chroma rows are half the luma stride (per the V4L2 YU12 spec).
            let dst_c_stride = dst_stride / 2;
            let c_rows = height / 2;
            let c_bytes = frame.stride_u.min(dst_c_stride);
            copy_plane(
                &mut dst[y_size..],
                dst_c_stride,
                frame.u_plane(),
                frame.stride_u,
                c_rows,
                c_bytes,
            );
            let u_size = dst_c_stride * c_rows;
            copy_plane(
                &mut dst[y_size + u_size..],
                dst_c_stride,
                frame.v_plane(),
                frame.stride_v,
                c_rows,
                c_bytes,
            );
        }
        other => {
            return Err(Error::Element(format!(
                "V4L2 M2M: unsupported frame format {other:?}"
            )));
        }
    }

    Ok(sizeimage)
}

/// Row-by-row copy between planes with different strides.
fn copy_plane(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    rows: usize,
    row_bytes: usize,
) {
    for row in 0..rows {
        let s = row * src_stride;
        let d = row * dst_stride;
        dst[d..d + row_bytes].copy_from_slice(&src[s..s + row_bytes]);
    }
}

/// Extract SPS (NAL type 7) and PPS (NAL type 8) units from an Annex-B
/// stream, concatenated with 4-byte start codes. Empty if none found.
fn extract_sps_pps(data: &[u8]) -> Vec<u8> {
    let mut headers = Vec::new();
    for nal in annex_b_nals(data) {
        let nal_type = nal.first().map(|b| b & 0x1F);
        if nal_type == Some(7) || nal_type == Some(8) {
            headers.extend_from_slice(&[0, 0, 0, 1]);
            headers.extend_from_slice(nal);
        }
    }
    headers
}

/// Split an Annex-B stream into NAL unit payloads (start codes stripped).
fn annex_b_nals(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i..].starts_with(&[0, 0, 0, 1]) {
            starts.push(i + 4);
            i += 4;
        } else if data[i..].starts_with(&[0, 0, 1]) {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(n, &start)| {
            let end = starts
                .get(n + 1)
                .map(|&next| {
                    // Back off over the next NAL's start code.
                    let code_len = if next >= 4 && data[next - 4..next] == [0, 0, 0, 1] {
                        4
                    } else {
                        3
                    };
                    next - code_len
                })
                .unwrap_or(data.len());
            &data[start..end]
        })
        .collect()
}

/// Find the first `/dev/video*` node that is an M2M device producing the
/// given coded fourcc (e.g. `b"H264"`) on its CAPTURE queue.
///
/// Returns `None` when no hardware encoder is present.
pub fn find_m2m_encoder(fourcc: &[u8; 4]) -> Option<PathBuf> {
    let target = v4l2r::PixelFormat::from_fourcc(fourcc);
    for index in 0..64 {
        let path = PathBuf::from(format!("/dev/video{index}"));
        if !path.exists() {
            continue;
        }
        let Ok(device) = Device::open(&path, DeviceConfig::new()) else {
            continue;
        };
        let caps = device.caps().device_caps();
        let (m2m, mplane) = (
            caps.contains(ioctl::Capabilities::VIDEO_M2M),
            caps.contains(ioctl::Capabilities::VIDEO_M2M_MPLANE),
        );
        if !m2m && !mplane {
            continue;
        }
        let queue_type = if mplane {
            QueueType::VideoCaptureMplane
        } else {
            QueueType::VideoCapture
        };
        if ioctl::FormatIterator::new(&device, queue_type)
            .any(|desc| desc.pixelformat == target && desc.flags.contains(FormatFlags::COMPRESSED))
        {
            return Some(path);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = V4l2M2mEncoderConfig::new(1280, 720);
        assert_eq!(config.pixel_format, PixelFormat::Nv12);
        assert_eq!(config.coded_format, V4l2CodedFormat::H264);
        assert_eq!(config.framerate, (30, 1));
        assert_eq!(config.bitrate_bps, 0);
    }

    #[test]
    fn config_builders() {
        let config = V4l2M2mEncoderConfig::new(640, 480)
            .pixel_format(PixelFormat::I420)
            .bitrate(2_000_000)
            .gop_size(30)
            .profile(V4l2H264Profile::Main)
            .framerate(15, 1)
            .coded_format(V4l2CodedFormat::Fwht);
        assert_eq!(config.pixel_format, PixelFormat::I420);
        assert_eq!(config.bitrate_bps, 2_000_000);
        assert_eq!(config.gop_size, 30);
        assert_eq!(config.profile, Some(V4l2H264Profile::Main));
        assert_eq!(config.framerate, (15, 1));
        assert_eq!(config.coded_format, V4l2CodedFormat::Fwht);
    }

    #[test]
    fn sps_pps_extraction_four_byte_codes() {
        // SPS (0x67), PPS (0x68), IDR (0x65) with 4-byte start codes.
        let stream = [
            0, 0, 0, 1, 0x67, 0xAA, 0xBB, //
            0, 0, 0, 1, 0x68, 0xCC, //
            0, 0, 0, 1, 0x65, 0x11, 0x22, 0x33,
        ];
        let headers = extract_sps_pps(&stream);
        assert_eq!(
            headers,
            vec![0, 0, 0, 1, 0x67, 0xAA, 0xBB, 0, 0, 0, 1, 0x68, 0xCC]
        );
    }

    #[test]
    fn sps_pps_extraction_three_byte_codes() {
        let stream = [
            0, 0, 1, 0x67, 0x01, //
            0, 0, 1, 0x68, 0x02, //
            0, 0, 1, 0x41, 0x03, // non-IDR slice: excluded
        ];
        let headers = extract_sps_pps(&stream);
        assert_eq!(
            headers,
            vec![0, 0, 0, 1, 0x67, 0x01, 0, 0, 0, 1, 0x68, 0x02]
        );
    }

    #[test]
    fn sps_pps_extraction_none_present() {
        let stream = [0, 0, 0, 1, 0x65, 0x11, 0x22];
        assert!(extract_sps_pps(&stream).is_empty());
        assert!(extract_sps_pps(&[]).is_empty());
    }

    #[test]
    fn plane_copy_respects_strides() {
        // 4x2 luma at src stride 4 into dst stride 6.
        let src = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut dst = [0u8; 12];
        copy_plane(&mut dst, 6, &src, 4, 2, 4);
        assert_eq!(dst, [1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0]);
    }

    #[test]
    fn find_m2m_encoder_does_not_panic() {
        // Result depends on the machine; just exercise the probe path.
        let h264 = find_m2m_encoder(b"H264");
        println!("H264 M2M encoder: {h264:?}");
    }

    /// A synthetic frame with a moving luma gradient.
    fn test_frame(width: u32, height: u32, format: PixelFormat, seq: u64) -> VideoFrame {
        let mut frame = VideoFrame::new(width, height, format);
        frame.pts = (seq * 33_000_000) as i64;
        for y in 0..height as usize {
            for x in 0..width as usize {
                frame.data[y * frame.stride_y + x] = ((x + y) as u8).wrapping_add(seq as u8 * 8);
            }
        }
        frame
    }

    fn open_with_any_input(device: &str, coded: V4l2CodedFormat) -> V4l2M2mH264Encoder {
        let mut failures = Vec::new();
        for pf in [PixelFormat::Nv12, PixelFormat::I420] {
            let config = V4l2M2mEncoderConfig::new(320, 240)
                .pixel_format(pf)
                .coded_format(coded);
            match V4l2M2mH264Encoder::new(device, config) {
                Ok(encoder) => return encoder,
                Err(e) => failures.push(format!("{pf:?}: {e}")),
            }
        }
        panic!("device accepts neither NV12 nor I420 input: {failures:?}");
    }

    /// Hardware-gated: `modprobe vicodec`, then set
    /// `PARALLAX_VICODEC_TEST_DEVICE` to the FWHT *encoder* node, or to
    /// `auto` to locate it via [`find_m2m_encoder`]. Exercises the full
    /// queue/drain/re-arm state machine without encoder hardware.
    #[test]
    fn vicodec_queue_mechanics() {
        let Ok(device) = std::env::var("PARALLAX_VICODEC_TEST_DEVICE") else {
            println!("PARALLAX_VICODEC_TEST_DEVICE not set, skipping");
            return;
        };
        let device = if device == "auto" {
            let found = find_m2m_encoder(b"FWHT").expect("no FWHT M2M device (vicodec loaded?)");
            found.to_string_lossy().into_owned()
        } else {
            device
        };
        println!("using vicodec encoder at {device}");

        let mut encoder = open_with_any_input(&device, V4l2CodedFormat::Fwht);
        let input_format = encoder.config.pixel_format;

        let mut packets = 0usize;
        for seq in 0..10 {
            let frame = test_frame(320, 240, input_format, seq);
            packets += encoder.encode(&frame).expect("encode").len();
        }
        packets += encoder.flush().expect("flush").len();
        assert_eq!(packets, 10, "every queued frame must come back encoded");

        // Re-arm after drain: the encoder must accept another stream.
        let mut packets = 0usize;
        for seq in 0..3 {
            let frame = test_frame(320, 240, input_format, seq);
            packets += encoder.encode(&frame).expect("encode after re-arm").len();
        }
        packets += encoder.flush().expect("second flush").len();
        assert_eq!(packets, 3, "encoder must restart cleanly after a drain");

        // Repeated flush without new input is a no-op.
        assert!(encoder.flush().expect("idempotent flush").is_empty());
    }

    /// Hardware-gated: set `PARALLAX_V4L2_M2M_TEST_DEVICE=/dev/videoN` to a
    /// real H.264 M2M encoder (e.g. bcm2835-codec on a Raspberry Pi).
    #[test]
    fn hw_h264_encode_roundtrip() {
        let Ok(device) = std::env::var("PARALLAX_V4L2_M2M_TEST_DEVICE") else {
            println!("PARALLAX_V4L2_M2M_TEST_DEVICE not set, skipping");
            return;
        };

        let mut encoder = open_with_any_input(&device, V4l2CodedFormat::H264);
        let input_format = encoder.config.pixel_format;

        let mut all_packets = Vec::new();
        for seq in 0..30 {
            if seq == 15 {
                encoder.force_keyframe();
            }
            let frame = test_frame(320, 240, input_format, seq);
            all_packets.extend(encoder.encode(&frame).expect("encode"));
        }
        all_packets.extend(encoder.flush().expect("flush"));

        assert!(!all_packets.is_empty(), "hardware produced no packets");
        let first = &all_packets[0];
        assert!(
            !extract_sps_pps(first).is_empty(),
            "first packet must carry SPS/PPS (in-band headers)"
        );
        assert!(
            encoder.codec_data().is_some(),
            "codec_data must be cached from the first keyframe"
        );
        println!(
            "{} packets, {} total bytes",
            all_packets.len(),
            all_packets.iter().map(Vec::len).sum::<usize>()
        );
    }
}
