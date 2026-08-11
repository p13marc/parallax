//! AV1 software decoder using dav1d.
//!
//! dav1d is a fast, cross-platform AV1 decoder developed by VideoLAN.
//! It's used in Firefox, VLC, and many other applications.
//!
//! # System Dependencies
//!
//! Requires the dav1d library to be installed:
//!
//! - **Fedora/RHEL**: `sudo dnf install libdav1d-devel`
//! - **Debian/Ubuntu**: `sudo apt install libdav1d-dev`
//! - **Arch**: `sudo pacman -S dav1d`
//! - **macOS**: `brew install dav1d`
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::Dav1dDecoder;
//!
//! let decoder = Dav1dDecoder::new()?;
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::element::{Element, ExecutionHints};
use crate::error::{Error, Result};
use crate::memory::{OutputArena, OutputBudget, defaults};

use super::common::{PixelFormat, VideoFrame};

/// AV1 software decoder using dav1d.
///
/// # Input
///
/// Expects OBU (Open Bitstream Unit) formatted AV1 data.
///
/// # Output
///
/// Produces raw video frames in I420 or I420p10 format.
///
/// # Example
///
/// ```rust,ignore
/// let decoder = Dav1dDecoder::new()?;
/// pipeline.add_node("av1dec", DynAsyncElement::new_box(ElementAdapter::new(decoder)));
/// ```
pub struct Dav1dDecoder {
    decoder: dav1d::Decoder,
    frame_count: u64,
    /// Arena for output buffer allocation.
    output: OutputArena,
    /// Decoded frames waiting to be emitted (dav1d can release several
    /// pictures after one send once its frame-delay pipeline fills).
    ready: std::collections::VecDeque<VideoFrame>,
    /// Input metadata awaiting its decoded frame, matched by the pts that
    /// rides through dav1d as the packet timestamp.
    pending_meta: std::collections::VecDeque<crate::metadata::Metadata>,
    last_dims: Option<(u32, u32)>,
    frames_out: u64,
}

impl Dav1dDecoder {
    /// Create a new dav1d decoder with default settings.
    pub fn new() -> Result<Self> {
        let settings = dav1d::Settings::new();
        let decoder = dav1d::Decoder::with_settings(&settings)
            .map_err(|e| Error::Config(format!("Failed to create dav1d decoder: {:?}", e)))?;

        Ok(Self::from_decoder(decoder))
    }

    /// Create a decoder with custom settings.
    pub fn with_settings(settings: &dav1d::Settings) -> Result<Self> {
        let decoder = dav1d::Decoder::with_settings(settings)
            .map_err(|e| Error::Config(format!("Failed to create dav1d decoder: {:?}", e)))?;

        Ok(Self::from_decoder(decoder))
    }

    fn from_decoder(decoder: dav1d::Decoder) -> Self {
        Self {
            decoder,
            frame_count: 0,
            output: OutputArena::new(defaults::VIDEO_DECODER_SLOT_COUNT),
            ready: std::collections::VecDeque::new(),
            pending_meta: std::collections::VecDeque::new(),
            last_dims: None,
            frames_out: 0,
        }
    }

    /// Get the number of frames decoded.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Drain every picture dav1d has ready into `self.ready`.
    fn drain_pictures(&mut self) -> Result<()> {
        loop {
            match self.decoder.get_picture() {
                Ok(picture) => {
                    let frame = self.picture_to_frame(&picture)?;
                    self.frame_count += 1;
                    self.ready.push_back(frame);
                }
                Err(dav1d::Error::Again) => return Ok(()), // nothing more yet
                Err(e) => {
                    return Err(Error::InvalidSegment(format!(
                        "dav1d decode failed: {:?}",
                        e
                    )));
                }
            }
        }
    }

    /// Feed one temporal unit, honoring dav1d's flow control.
    ///
    /// `Again` from `send_data` is not an error: the decoder wants its
    /// ready pictures consumed before accepting more input. Drain, then
    /// resubmit the retained data until it is accepted.
    fn send_frame(&mut self, input: &[u8], pts_ns: i64) -> Result<()> {
        let mut result = self
            .decoder
            .send_data(input.to_vec(), None, Some(pts_ns), None);
        loop {
            match result {
                Ok(()) => return Ok(()),
                Err(dav1d::Error::Again) => {
                    self.drain_pictures()?;
                    result = self.decoder.send_pending_data();
                }
                Err(e) => {
                    return Err(Error::InvalidSegment(format!(
                        "dav1d send_data failed: {:?}",
                        e
                    )));
                }
            }
        }
    }

    /// Build the output buffer for the oldest ready frame, claiming the
    /// input metadata whose pts rode through dav1d as the timestamp.
    fn emit_ready(&mut self) -> Result<Option<Buffer>> {
        let Some(frame) = self.ready.pop_front() else {
            return Ok(None);
        };

        let dims = (frame.width, frame.height);
        if self.last_dims.is_some_and(|last| last != dims) {
            tracing::info!(
                "dav1ddecoder: resolution changed to {}x{}, rebuilding the output arena",
                dims.0,
                dims.1
            );
            self.output.reset();
        }
        self.last_dims = Some(dims);

        let mut slot = self.output.acquire(frame.data.len(), "dav1ddecoder")?;
        slot.data_mut()[..frame.data.len()].copy_from_slice(&frame.data);

        // The pts attached at send time comes back on the picture, so match
        // the originating input's metadata by it (FIFO fallback for
        // untimestamped streams).
        let claimed = self
            .pending_meta
            .iter()
            .position(|m| m.pts.nanos() as i64 == frame.pts)
            .and_then(|i| self.pending_meta.remove(i))
            .or_else(|| self.pending_meta.pop_front());
        let mut metadata = claimed.unwrap_or_default();
        metadata.pts = ClockTime::from_nanos(frame.pts as u64);
        metadata.sequence = self.frames_out;
        self.frames_out += 1;
        metadata.set_video_dims(dims.0, dims.1, frame.format.into());

        Ok(Some(Buffer::new(
            MemoryHandle::with_len(slot, frame.data.len()),
            metadata,
        )))
    }

    /// Convert dav1d Picture to our VideoFrame.
    fn picture_to_frame(&self, picture: &dav1d::Picture) -> Result<VideoFrame> {
        let width = picture.width();
        let height = picture.height();
        let bit_depth = picture.bit_depth();

        let format = match (picture.pixel_layout(), bit_depth) {
            (dav1d::PixelLayout::I420, 8) => PixelFormat::I420,
            (dav1d::PixelLayout::I420, 10) => PixelFormat::I420p10,
            (dav1d::PixelLayout::I422, 8) => PixelFormat::I422,
            (dav1d::PixelLayout::I444, 8) => PixelFormat::I444,
            _ => {
                return Err(Error::InvalidSegment(format!(
                    "Unsupported pixel format: {:?} {}bit",
                    picture.pixel_layout(),
                    bit_depth
                )));
            }
        };

        // Downstream consumes packed planes (VideoConvert & friends assume
        // tightly-packed I420), so strip dav1d's row padding while copying.
        let bytes_per_sample = if bit_depth > 8 { 2 } else { 1 };
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = match format {
            PixelFormat::I444 => (w, h),
            PixelFormat::I422 => (w.div_ceil(2), h),
            _ => (w.div_ceil(2), h.div_ceil(2)),
        };

        let mut frame = VideoFrame::new(width, height, format);
        frame.pts = picture.timestamp().unwrap_or(0);
        frame.stride_y = w * bytes_per_sample;
        frame.stride_u = cw * bytes_per_sample;
        frame.stride_v = cw * bytes_per_sample;

        let y_size = frame.stride_y * h;
        let c_size = frame.stride_u * ch;
        frame.data = vec![0u8; y_size + 2 * c_size];

        for (component, rows, row_bytes, offset) in [
            (dav1d::PlanarImageComponent::Y, h, frame.stride_y, 0),
            (dav1d::PlanarImageComponent::U, ch, frame.stride_u, y_size),
            (
                dav1d::PlanarImageComponent::V,
                ch,
                frame.stride_v,
                y_size + c_size,
            ),
        ] {
            let plane = picture.plane(component);
            let src_stride = picture.stride(component) as usize;
            for row in 0..rows {
                let src = &plane[row * src_stride..row * src_stride + row_bytes];
                frame.data[offset + row * row_bytes..offset + (row + 1) * row_bytes]
                    .copy_from_slice(src);
            }
        }

        Ok(frame)
    }
}

impl Element for Dav1dDecoder {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    // Decoders never skip an input: it would be a reference frame the
    // decoder never sees. Shed the output copy instead.
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        const MAX_PENDING_METADATA: usize = 64;
        if self.pending_meta.len() >= MAX_PENDING_METADATA {
            self.pending_meta.pop_front();
        }
        self.pending_meta.push_back(buffer.metadata().clone());

        let pts = buffer.metadata().pts.nanos() as i64;
        self.send_frame(buffer.as_bytes(), pts)?;
        self.drain_pictures()?;
        self.emit_ready()
    }

    /// Drain the pictures dav1d still holds (its frame-delay pipeline) at
    /// EOS, one per call until empty.
    fn flush(&mut self) -> Result<Option<Buffer>> {
        self.drain_pictures()?;
        self.emit_ready()
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::native() // Native code (FFI), might crash on bad input
    }
}
