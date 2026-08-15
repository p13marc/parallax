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
//! let config = V4l2M2mEncoderConfig::new().bitrate(4_000_000);
//! let encoder = V4l2M2mH264Encoder::new(&device, config)?;
//! let element = EncoderElement::new(encoder);
//! ```
//!
//! Neither the config nor the wrapper takes dimensions: geometry travels
//! in-band in each buffer's `Metadata`, and the driver queues are configured
//! from the first frame (and reconfigured if it changes).
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

use super::common::{PixelFormat, VideoFrameRef};
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
///
/// There is **no width or height**: geometry travels in-band, in each buffer's
/// [`Metadata`](crate::metadata::Metadata). The driver queues are configured
/// from the first frame and reconfigured if it changes (#38).
pub struct V4l2M2mEncoderConfig {
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
    /// Create a config with defaults: NV12 input, H.264 output, 30 fps,
    /// driver-default bitrate and GOP.
    ///
    /// Frame dimensions come from the buffers, not from here.
    pub fn new() -> Self {
        Self {
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

impl Default for V4l2M2mEncoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// How long to wait for the hardware before declaring it stalled.
const POLL_TIMEOUT: Duration = Duration::from_secs(2);

/// V4L2 M2M stateful hardware H.264 encoder.
///
/// See the [module docs](self) for usage. Not `Sync`: one pipeline element
/// owns the device.
///
/// # Runtime control, and how far to trust it
///
/// [`set_bitrate`](VideoEncoder::set_bitrate),
/// [`set_keyframe_interval`](VideoEncoder::set_keyframe_interval) and
/// [`force_keyframe`](VideoEncoder::force_keyframe) work on a *streaming*
/// encoder: V4L2's stateful encoder uAPI says the client "is allowed to use
/// `VIDIOC_S_CTRL()` to change encoder parameters at any time".
///
/// What a driver *does* with them is another matter. Availability is
/// driver-specific; a driver that will not take a change while streaming
/// returns `-EBUSY` (which surfaces here as an `Err`). Worse, several drivers
/// **accept the ioctl and then ignore it** mid-stream — GOP size is the usual
/// casualty, bitrate is more widely honoured — and userspace cannot detect
/// that, because the ioctl succeeded. Treat a live GOP change on hardware as
/// best-effort, and verify against the driver you actually ship on.
pub struct V4l2M2mH264Encoder {
    device: Arc<Device>,
    config: V4l2M2mEncoderConfig,
    /// Queue state, configured from the first frame's geometry.
    ///
    /// V4L2 wants `S_FMT` before buffers are allocated and the queues stream
    /// on, which is why this used to force dimensions into the constructor.
    /// Deferring it to the first frame is what lets geometry stay in-band
    /// (#38); a resize drops this and rebuilds it.
    streaming: Option<Streaming>,
    /// Cached SPS+PPS (Annex-B) from the first keyframe packet.
    codec_data: Option<Vec<u8>>,
}

/// The queues, poller and negotiated format for one geometry.
struct Streaming {
    output_queue: Queue<Output, BuffersAllocated<Vec<MmapHandle>>>,
    capture_queue: Queue<Capture, BuffersAllocated<Vec<MmapHandle>>>,
    poller: Poller,
    /// OUTPUT format as applied by the driver (strides may differ from ours).
    output_format: Format,
    /// The geometry these queues were configured for.
    dims: (u32, u32),
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
            .map_err(|e| classify_open_failure(path, &e))?;
        let device = Arc::new(device);

        // Reject an unencodable input format now: it is a property of the
        // config, not of any frame, so there is no reason to wait.
        raw_fourcc_for(config.pixel_format)?;

        Ok(Self {
            device,
            config,
            streaming: None,
            codec_data: None,
        })
    }

    /// Ensure the queues are configured and streaming at `width`x`height`.
    ///
    /// On a geometry change the previous `Streaming` is dropped, which streams
    /// the queues off and frees their buffers, and a fresh one is built.
    fn ensure_streaming(&mut self, width: u32, height: u32) -> Result<()> {
        if let Some(s) = &self.streaming
            && s.dims == (width, height)
        {
            return Ok(());
        }

        if self.streaming.is_some() {
            tracing::info!("V4L2 M2M: input resized to {width}x{height}, reconfiguring queues");
            self.streaming = None;
        }

        self.streaming = Some(Self::start_streaming(
            Arc::clone(&self.device),
            &self.config,
            width,
            height,
        )?);
        Ok(())
    }

    /// Configure both queues for one geometry and stream them on.
    fn start_streaming(
        device: Arc<Device>,
        config: &V4l2M2mEncoderConfig,
        width: u32,
        height: u32,
    ) -> Result<Streaming> {
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

        let raw_fourcc = raw_fourcc_for(config.pixel_format)?;
        let output_format: Format = output_queue
            .change_format()
            .map_err(|e| Error::Config(format!("V4L2 M2M: get output format: {e}")))?
            .set_size(width as usize, height as usize)
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
        if output_format.width != width || output_format.height != height {
            return Err(Error::Config(format!(
                "V4L2 M2M: driver adjusted {width}x{height} to {}x{} (alignment \
                 constraints); scale to the adjusted size upstream",
                output_format.width, output_format.height,
            )));
        }

        Self::apply_controls(&device, config);

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

        Ok(Streaming {
            output_queue,
            capture_queue,
            poller,
            output_format,
            dims: (width, height),
            drained: false,
            frames_queued: 0,
        })
    }

    /// Apply codec controls. All are best-effort: drivers advertise wildly
    /// different control sets (vicodec supports none of the H.264 CIDs), and
    /// an unsupported knob should not prevent encoding.
    fn apply_controls(device: &Device, config: &V4l2M2mEncoderConfig) {
        if config.bitrate_bps > 0 {
            // Ignore failures here: at open time a missing control is a
            // property of the driver, not a user error.
            let _ = set_bitrate_control(device, config.bitrate_bps);
        }

        if config.gop_size > 0 {
            let _ = set_gop_control(device, config.gop_size);
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

    /// The streaming state, or an error naming why there isn't one.
    fn streaming_mut(&mut self) -> Result<&mut Streaming> {
        self.streaming.as_mut().ok_or_else(|| {
            Error::Element(
                "V4L2 M2M: no frame has been encoded yet, so the queues are not configured".into(),
            )
        })
    }

    /// Cache SPS/PPS from the first keyframe packet (H.264 only).
    fn cache_codec_data(&mut self, packets: &[(Vec<u8>, BufferFlags)]) {
        if self.codec_data.is_some() || self.config.coded_format != V4l2CodedFormat::H264 {
            return;
        }
        for (data, flags) in packets {
            if !flags.contains(BufferFlags::KEYFRAME) {
                continue;
            }
            let headers = extract_sps_pps(data);
            if !headers.is_empty() {
                self.codec_data = Some(headers);
                return;
            }
        }
    }
}

impl Streaming {
    /// Return finished OUTPUT buffers to the free pool.
    fn reclaim_output_buffers(&self) {
        while let Ok(dqbuf) = self.output_queue.try_dequeue() {
            // MMAP handles have nothing to take; dropping frees the slot.
            drop(dqbuf);
        }
    }

    /// Drain every CAPTURE packet the hardware has ready, without blocking.
    ///
    /// Returns each packet with its buffer flags so the caller can pick out
    /// the keyframe that carries SPS/PPS.
    fn drain_capture(&mut self) -> Result<Vec<(Vec<u8>, BufferFlags)>> {
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
                            packets.push((mapping.to_vec(), flags));
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
                // NotReady: nothing encoded yet. Eos (EPIPE): the stream is
                // stopped after a drain and produces nothing until the
                // re-arm takes effect — same answer for a non-blocking poll.
                Err(v4l2r::ioctl::IoctlConvertError::IoctlError(
                    DqBufIoctlError::NotReady | DqBufIoctlError::Eos,
                )) => {
                    break;
                }
                Err(e) => {
                    return Err(Error::Element(format!("V4L2 M2M: dequeue capture: {e}")));
                }
            }
        }
        Ok(packets)
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

/// Turn a device-open failure into a named error where we can.
///
/// `EBUSY` is *the* characteristic V4L2 failure — a second open of a node
/// another pipeline or process already holds — and it used to land in a
/// stringly-typed `Error::Config` catch-all here, undiagnosable without
/// matching on the message (#47). `V4l2Src` names it; so should this.
///
/// v4l2r's `DeviceOpenError` does not expose the errno, so the classification
/// re-probes with `std::fs`. That only runs on the failure path.
fn classify_open_failure(path: &Path, original: &dyn std::fmt::Display) -> Error {
    use crate::elements::device::DeviceError;

    let display = path.display().to_string();
    let fallback = || Error::Config(format!("V4L2 M2M: open {display}: {original}"));

    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
    {
        Err(io) => match io.kind() {
            std::io::ErrorKind::ResourceBusy => DeviceError::Busy(display).into(),
            std::io::ErrorKind::NotFound => DeviceError::NotFound(display).into(),
            std::io::ErrorKind::PermissionDenied => DeviceError::PermissionDenied(display).into(),
            _ if io.raw_os_error() == Some(libc::EBUSY) => DeviceError::Busy(display).into(),
            _ => fallback(),
        },
        // The node itself opens, so the failure was later (QUERYCAP).
        Ok(_) => fallback(),
    }
}

/// The V4L2 fourcc for a raw input pixel format.
fn raw_fourcc_for(format: PixelFormat) -> Result<&'static [u8; 4]> {
    match format {
        PixelFormat::Nv12 => Ok(b"NV12"),
        PixelFormat::I420 => Ok(b"YU12"),
        other => Err(Error::Config(format!(
            "V4L2 M2M: unsupported input format {other:?} (use Nv12 or I420)"
        ))),
    }
}

impl VideoEncoder for V4l2M2mH264Encoder {
    type Packet = Vec<u8>;

    fn encode(&mut self, frame: VideoFrameRef<'_>) -> Result<Vec<Vec<u8>>> {
        if frame.format != self.config.pixel_format {
            return Err(Error::Element(format!(
                "V4L2 M2M: frame is {:?}, encoder configured for {:?}",
                frame.format, self.config.pixel_format,
            )));
        }

        // Geometry comes from the frame. The queues are configured for it on
        // the first frame, and reconfigured if it changes (#38).
        self.ensure_streaming(frame.width, frame.height)?;

        let device = Arc::clone(&self.device);
        let streaming = self.streaming_mut()?;

        // Resume after a flush(). ENC_CMD_START clears the mem2mem stopped
        // state, but vb2's last_buffer_dequeued flag — the source of the
        // EPIPE on DQBUF — is only reliably reset by restarting the CAPTURE
        // queue (vicodec, for one, never clears it on START because the
        // mem2mem helper clears has_stopped before the driver checks it).
        // The spec allows CAPTURE STREAMOFF/STREAMON as an equivalent
        // resume path, so do both.
        if streaming.drained {
            ioctl::encoder_cmd::<_, ()>(&*device, &EncoderCommand::Start)
                .map_err(|e| Error::Element(format!("V4L2 M2M: ENC_CMD_START: {e}")))?;
            streaming
                .capture_queue
                .stream_off()
                .map_err(|e| Error::Element(format!("V4L2 M2M: capture stream off: {e}")))?;
            streaming
                .capture_queue
                .stream_on()
                .map_err(|e| Error::Element(format!("V4L2 M2M: capture stream on: {e}")))?;
            while let Ok(buffer) = streaming.capture_queue.try_get_free_buffer() {
                buffer.queue().map_err(|e| {
                    Error::Element(format!("V4L2 M2M: requeue capture buffer: {e}"))
                })?;
            }
            streaming.drained = false;
        }

        streaming.reclaim_output_buffers();
        let qbuf = loop {
            match streaming.output_queue.try_get_free_buffer() {
                Ok(buffer) => break buffer,
                Err(_) => {
                    // All raw-frame slots are in the hardware; wait for one.
                    streaming.wait_for_event("a free output buffer")?;
                    streaming.reclaim_output_buffers();
                }
            }
        };

        let mut mapping = qbuf
            .get_plane_mapping(0)
            .ok_or_else(|| Error::Element("V4L2 M2M: map output plane".to_string()))?;
        let bytes_used = fill_output_plane(&mut mapping, &streaming.output_format, frame)?;
        drop(mapping);

        let timestamp = TimeVal::microseconds(frame.pts / 1_000);
        qbuf.set_timestamp(timestamp)
            .queue(&[bytes_used])
            .map_err(|e| Error::Element(format!("V4L2 M2M: queue output buffer: {e}")))?;
        streaming.frames_queued += 1;

        // Non-blocking: stateful encoders have pipeline latency, so early
        // frames legitimately produce nothing (EncoderElement buffers).
        let packets = streaming.drain_capture()?;
        self.cache_codec_data(&packets);
        Ok(packets.into_iter().map(|(data, _)| data).collect())
    }

    fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        // No streaming state means no frame was ever queued, so there is
        // nothing buffered in the hardware to drain.
        let Some(streaming) = self.streaming.as_mut() else {
            return Ok(Vec::new());
        };
        if streaming.drained || streaming.frames_queued == 0 {
            return Ok(Vec::new());
        }
        ioctl::encoder_cmd::<_, ()>(&*self.device, &EncoderCommand::Stop(false))
            .map_err(|e| Error::Element(format!("V4L2 M2M: ENC_CMD_STOP: {e}")))?;

        let mut packets = Vec::new();
        let deadline = Instant::now() + POLL_TIMEOUT;
        'drain: loop {
            match streaming.capture_queue.try_dequeue() {
                Ok(dqbuf) => {
                    let flags = dqbuf.data.flags();
                    let is_last = dqbuf.data.is_last();
                    if !flags.contains(BufferFlags::ERROR) {
                        let mapping = dqbuf.get_plane_mapping(0).ok_or_else(|| {
                            Error::Element("V4L2 M2M: map capture plane".to_string())
                        })?;
                        if !mapping.is_empty() {
                            packets.push((mapping.to_vec(), flags));
                        }
                    }
                    // Do NOT requeue during a drain: the queue must empty out.
                    drop(dqbuf);
                    if is_last {
                        break 'drain;
                    }
                }
                // Eos (EPIPE): the driver already considers the stream
                // stopped — drain complete.
                Err(v4l2r::ioctl::IoctlConvertError::IoctlError(DqBufIoctlError::Eos)) => {
                    break 'drain;
                }
                Err(v4l2r::ioctl::IoctlConvertError::IoctlError(DqBufIoctlError::NotReady)) => {
                    if streaming.capture_queue.num_queued_buffers() == 0 {
                        // Nothing left for the driver to fill (defensive; the
                        // LAST flag is the normal exit).
                        break 'drain;
                    }
                    if Instant::now() >= deadline {
                        return Err(Error::Element(
                            "V4L2 M2M: drain timed out waiting for the LAST buffer".to_string(),
                        ));
                    }
                    streaming.wait_for_event("drain completion")?;
                }
                Err(e) => {
                    return Err(Error::Element(format!("V4L2 M2M: dequeue capture: {e}")));
                }
            }
        }

        streaming.reclaim_output_buffers();
        streaming.drained = true;
        self.cache_codec_data(&packets);
        Ok(packets.into_iter().map(|(data, _)| data).collect())
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

    fn set_bitrate(&mut self, bps: u32) -> Result<()> {
        set_bitrate_control(&self.device, bps)?;
        self.config.bitrate_bps = bps;
        tracing::info!("V4L2 M2M: bitrate set to {bps} bps");
        Ok(())
    }

    fn set_keyframe_interval(&mut self, frames: u32) -> Result<()> {
        set_gop_control(&self.device, frames)?;
        self.config.gop_size = frames;
        tracing::info!("V4L2 M2M: GOP size set to {frames} frames");
        Ok(())
    }
}

/// Set the target bitrate on a (possibly streaming) encoder context.
///
/// V4L2 permits control changes at any time — "The client is allowed to use
/// `VIDIOC_S_CTRL()` to change encoder parameters at any time" — but which
/// controls a driver *honours* mid-stream is driver-specific, and one that
/// refuses returns `-EBUSY`. Shared by open-time configuration and the runtime
/// setter so the two cannot drift apart.
fn set_bitrate_control(device: &Device, bps: u32) -> Result<()> {
    let mut bitrate = SafeExtControl::<VideoBitrate>::from_value(bps as i32);
    ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut bitrate).map_err(|e| {
        Error::Config(format!(
            "V4L2 M2M: driver rejected a bitrate of {bps} bps: {e}"
        ))
    })?;

    // CBR: without this the driver may treat the bitrate as an upper bound.
    let mut mode = SafeExtControl::<VideoBitrateMode>::from_value(
        v4l2r::controls::codec::VideoBitrateMode::ConstantBitrate as i32,
    );
    if let Err(e) = ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut mode) {
        tracing::debug!("V4L2 M2M: bitrate mode control unsupported: {e}");
    }
    Ok(())
}

/// Set the keyframe interval, preferring `H264_I_PERIOD` and falling back to
/// the generic `GOP_SIZE`.
///
/// Worth knowing: several drivers accept this ioctl and then ignore it while
/// streaming (GOP is the usual casualty; bitrate is more widely honoured). We
/// cannot detect that from userspace — the ioctl succeeds — so treat a live GOP
/// change on hardware as best-effort.
fn set_gop_control(device: &Device, frames: u32) -> Result<()> {
    let mut i_period = SafeExtControl::<VideoH264IPeriod>::from_value(frames as i32);
    if ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut i_period).is_ok() {
        return Ok(());
    }

    let mut gop = SafeExtControl::<VideoGopSize>::from_value(frames as i32);
    ioctl::s_ext_ctrls(device, CtrlWhich::Current, &mut gop).map_err(|e| {
        Error::Config(format!(
            "V4L2 M2M: driver rejected a GOP size of {frames}: {e}"
        ))
    })?;
    Ok(())
}

impl std::fmt::Debug for V4l2M2mH264Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("V4l2M2mH264Encoder")
            .field("config", &self.config)
            .field("dims", &self.streaming.as_ref().map(|s| s.dims))
            .field("drained", &self.streaming.as_ref().map(|s| s.drained))
            .field(
                "frames_queued",
                &self.streaming.as_ref().map_or(0, |s| s.frames_queued),
            )
            .finish()
    }
}

impl Drop for Streaming {
    fn drop(&mut self) {
        // Also runs when a resize replaces this state, which is what makes
        // reconfiguring the queues safe.
        let _ = self.capture_queue.stream_off();
        let _ = self.output_queue.stream_off();
    }
}

/// Copy a [`VideoFrameRef`] into a single-plane OUTPUT buffer, honoring the
/// driver's row stride (`bytesperline`). Returns the bytes used.
fn fill_output_plane(dst: &mut [u8], format: &Format, frame: VideoFrameRef<'_>) -> Result<usize> {
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
            // One interleaved UV plane: full-width rows spaced like the luma
            // plane. Addressed straight from the data layout — NV12 stride_u
            // conventions differ between frame producers (EncoderElement uses
            // the full row width, VideoFrame::new the planar half-width), but
            // the bytes are laid out identically.
            let src_uv = &frame.data[frame.stride_y * height..];
            copy_plane(
                &mut dst[y_size..],
                dst_stride,
                src_uv,
                frame.stride_y,
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

/// Extract SPS (NAL type 7) and PPS (NAL type 8) units from an Annex-B stream,
/// concatenated with 4-byte start codes. Empty if none found.
///
/// The hardware encoder emits parameter sets once, so they are cached and
/// re-prepended to every keyframe — hence the start-code-carrying form here,
/// unlike [`annexb::extract_param_sets`](crate::codec::annexb::extract_param_sets)
/// which returns them raw.
fn extract_sps_pps(data: &[u8]) -> Vec<u8> {
    use crate::codec::annexb::{NAL_PPS, NAL_SPS, nal_units};

    let mut headers = Vec::new();
    for nal in nal_units(data) {
        if matches!(nal.nal_type(), NAL_SPS | NAL_PPS) {
            headers.extend_from_slice(&[0, 0, 0, 1]);
            headers.extend_from_slice(nal.data);
        }
    }
    headers
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
    use crate::elements::codec::common::VideoFrame;

    /// #38: the config carries no geometry at all — it cannot lie about a
    /// resolution it was never told.
    #[test]
    fn config_has_no_dimensions() {
        let config = V4l2M2mEncoderConfig::new();
        // Compile-time proof by construction: there is no width/height to
        // read. What the config *does* carry is codec policy.
        assert_eq!(config.pixel_format, PixelFormat::Nv12);
        assert_eq!(config.num_output_buffers, 4);
        // And Default agrees with new().
        assert_eq!(
            format!("{:?}", V4l2M2mEncoderConfig::default()),
            format!("{config:?}")
        );
    }

    #[test]
    fn config_defaults() {
        let config = V4l2M2mEncoderConfig::new();
        assert_eq!(config.pixel_format, PixelFormat::Nv12);
        assert_eq!(config.coded_format, V4l2CodedFormat::H264);
        assert_eq!(config.framerate, (30, 1));
        assert_eq!(config.bitrate_bps, 0);
    }

    #[test]
    fn config_builders() {
        let config = V4l2M2mEncoderConfig::new()
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

    fn driver_format(width: u32, height: u32, bytesperline: u32, sizeimage: u32) -> Format {
        Format {
            width,
            height,
            pixelformat: v4l2r::PixelFormat::from_fourcc(b"NV12"),
            plane_fmt: vec![v4l2r::PlaneLayout {
                sizeimage,
                bytesperline,
            }],
        }
    }

    /// NV12 stride_u conventions differ between frame producers
    /// (EncoderElement: full row width; VideoFrame::new: planar half-width).
    /// fill_output_plane must copy the same bytes under both.
    #[test]
    fn nv12_fill_handles_both_stride_conventions() {
        const W: usize = 16;
        const H: usize = 8;
        let format = driver_format(W as u32, H as u32, W as u32, (W * H * 3 / 2) as u32);

        // VideoFrame::new convention: stride_u = stride_y / 2.
        let mut frame = VideoFrame::new(W as u32, H as u32, PixelFormat::Nv12);
        for (i, b) in frame.data.iter_mut().enumerate() {
            *b = i as u8;
        }
        let mut dst_a = vec![0u8; W * H * 3 / 2];
        let used = fill_output_plane(&mut dst_a, &format, frame.as_view()).unwrap();
        assert_eq!(used, W * H * 3 / 2);

        // EncoderElement convention: stride_u = stride_y, stride_v = 0.
        frame.stride_u = frame.stride_y;
        frame.stride_v = 0;
        let mut dst_b = vec![0u8; W * H * 3 / 2];
        fill_output_plane(&mut dst_b, &format, frame.as_view()).unwrap();

        assert_eq!(dst_a, dst_b, "same bytes regardless of stride convention");
        assert_eq!(&dst_a[..], &frame.data[..], "tight NV12 copies verbatim");
    }

    /// The driver stride can exceed the frame's: rows must land at
    /// bytesperline offsets with padding between them.
    #[test]
    fn nv12_fill_honors_driver_stride() {
        const W: usize = 8;
        const H: usize = 4;
        const DST_STRIDE: usize = 12;
        let sizeimage = DST_STRIDE * H * 3 / 2;
        let format = driver_format(W as u32, H as u32, DST_STRIDE as u32, sizeimage as u32);

        let mut frame = VideoFrame::new(W as u32, H as u32, PixelFormat::Nv12);
        frame.data.fill(0xAA);
        let mut dst = vec![0u8; sizeimage];
        fill_output_plane(&mut dst, &format, frame.as_view()).unwrap();

        // First luma row: 8 payload bytes then 4 padding bytes.
        assert_eq!(&dst[..W], &[0xAA; W]);
        assert_eq!(&dst[W..DST_STRIDE], &[0; DST_STRIDE - W]);
        // First UV row sits at bytesperline * height.
        let uv = DST_STRIDE * H;
        assert_eq!(&dst[uv..uv + W], &[0xAA; W]);
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
            let config = V4l2M2mEncoderConfig::new()
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
            let frame = test_frame(640, 480, input_format, seq);
            packets += encoder.encode(frame.as_view()).expect("encode").len();
        }
        packets += encoder.flush().expect("flush").len();
        assert_eq!(packets, 10, "every queued frame must come back encoded");

        // Re-arm after drain: the encoder must accept another stream.
        let mut packets = 0usize;
        for seq in 0..3 {
            let frame = test_frame(640, 480, input_format, seq);
            packets += encoder
                .encode(frame.as_view())
                .expect("encode after re-arm")
                .len();
        }
        packets += encoder.flush().expect("second flush").len();
        assert_eq!(packets, 3, "encoder must restart cleanly after a drain");

        // Repeated flush without new input is a no-op.
        assert!(encoder.flush().expect("idempotent flush").is_empty());
    }

    /// Hardware-gated: live bitrate and GOP changes on a *streaming* encoder.
    ///
    /// V4L2 permits control changes at any time, but honouring them is
    /// driver-specific — this asserts the ioctls are accepted and encoding
    /// survives them, not that the driver actually re-rates the stream (which
    /// userspace cannot detect: the ioctl succeeds either way).
    #[test]
    fn hw_live_control_changes() {
        use super::super::traits::VideoEncoder;

        let Ok(device) = std::env::var("PARALLAX_V4L2_M2M_TEST_DEVICE") else {
            println!("PARALLAX_V4L2_M2M_TEST_DEVICE not set, skipping");
            return;
        };

        let mut encoder = open_with_any_input(&device, V4l2CodedFormat::H264);
        let input_format = encoder.config.pixel_format;

        for seq in 0..5 {
            encoder
                .encode(test_frame(640, 480, input_format, seq).as_view())
                .expect("encode");
        }

        // Mid-stream, on a live context.
        match encoder.set_bitrate(500_000) {
            Ok(()) => assert_eq!(encoder.config.bitrate_bps, 500_000),
            Err(e) => println!("driver rejected a live bitrate change: {e}"),
        }
        match encoder.set_keyframe_interval(15) {
            Ok(()) => assert_eq!(encoder.config.gop_size, 15),
            Err(e) => println!("driver rejected a live GOP change: {e}"),
        }

        // Encoding must survive either outcome.
        let mut packets = 0;
        for seq in 5..10 {
            packets += encoder
                .encode(test_frame(640, 480, input_format, seq).as_view())
                .expect("encoding must continue after a control change")
                .len();
        }
        packets += encoder.flush().expect("flush").len();
        assert!(
            packets > 0,
            "encoder stopped producing after a control change"
        );
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
            let frame = test_frame(640, 480, input_format, seq);
            all_packets.extend(encoder.encode(frame.as_view()).expect("encode"));
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
