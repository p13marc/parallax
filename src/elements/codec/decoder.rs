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
    /// ACCURATE-seek clipping (#165): decoded frames below the current Time
    /// segment's start are dropped after decoding.
    clip: super::common::SegmentClip,
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
            clip: Default::default(),
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
        // the originating input's metadata by it.
        let mut metadata = self.claim_meta_for(frame.pts);
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

        let stride_y = w * bytes_per_sample;
        let stride_c = cw * bytes_per_sample;
        let y_size = stride_y * h;
        let c_size = stride_c * ch;

        // Single allocation, no zero-fill: rows are appended in output
        // order, so the Vec is exactly full when the copy finishes.
        let mut data = Vec::with_capacity(y_size + 2 * c_size);
        Self::append_packed_planes(picture, &mut data, h, ch, stride_y, stride_c);

        Ok(VideoFrame {
            width,
            height,
            format,
            pts: picture.timestamp().unwrap_or(0),
            data,
            stride_y,
            stride_u: stride_c,
            stride_v: stride_c,
        })
    }

    /// Append the picture's planes, de-strided, onto `out`.
    fn append_packed_planes(
        picture: &dav1d::Picture,
        out: &mut Vec<u8>,
        h: usize,
        ch: usize,
        stride_y: usize,
        stride_c: usize,
    ) {
        for (component, rows, row_bytes) in [
            (dav1d::PlanarImageComponent::Y, h, stride_y),
            (dav1d::PlanarImageComponent::U, ch, stride_c),
            (dav1d::PlanarImageComponent::V, ch, stride_c),
        ] {
            let plane = picture.plane(component);
            let src_stride = picture.stride(component) as usize;
            for row in 0..rows {
                out.extend_from_slice(&plane[row * src_stride..row * src_stride + row_bytes]);
            }
        }
    }

    /// Geometry of a picture in packed-output terms:
    /// `(format, h, ch, stride_y, stride_c, total_bytes)`.
    fn packed_geometry(
        picture: &dav1d::Picture,
    ) -> Result<(PixelFormat, usize, usize, usize, usize, usize)> {
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
        let bytes_per_sample = if bit_depth > 8 { 2 } else { 1 };
        let (w, h) = (picture.width() as usize, picture.height() as usize);
        let (cw, ch) = match format {
            PixelFormat::I444 => (w, h),
            PixelFormat::I422 => (w.div_ceil(2), h),
            _ => (w.div_ceil(2), h.div_ceil(2)),
        };
        let stride_y = w * bytes_per_sample;
        let stride_c = cw * bytes_per_sample;
        Ok((
            format,
            h,
            ch,
            stride_y,
            stride_c,
            stride_y * h + 2 * stride_c * ch,
        ))
    }

    /// Single-copy hot path: de-stride the picture's planes straight into
    /// an arena slot (#139). Used when no earlier frame is queued, which is
    /// the steady state — one picture out per temporal unit in.
    fn picture_to_slot(&mut self, picture: &dav1d::Picture) -> Result<Buffer> {
        let (format, h, ch, stride_y, stride_c, total) = Self::packed_geometry(picture)?;
        let dims = (picture.width(), picture.height());
        if self.last_dims.is_some_and(|last| last != dims) {
            tracing::info!(
                "dav1ddecoder: resolution changed to {}x{}, rebuilding the output arena",
                dims.0,
                dims.1
            );
            self.output.reset();
        }
        self.last_dims = Some(dims);

        let mut slot = self.output.acquire(total, "dav1ddecoder")?;
        {
            let dst = &mut slot.data_mut()[..total];
            let mut off = 0;
            for (component, rows, row_bytes) in [
                (dav1d::PlanarImageComponent::Y, h, stride_y),
                (dav1d::PlanarImageComponent::U, ch, stride_c),
                (dav1d::PlanarImageComponent::V, ch, stride_c),
            ] {
                let plane = picture.plane(component);
                let src_stride = picture.stride(component) as usize;
                for row in 0..rows {
                    dst[off..off + row_bytes]
                        .copy_from_slice(&plane[row * src_stride..row * src_stride + row_bytes]);
                    off += row_bytes;
                }
            }
        }
        self.frame_count += 1;

        let pts = picture.timestamp().unwrap_or(0);
        let mut metadata = self.claim_meta_for(pts);
        metadata.pts = ClockTime::from_nanos(pts as u64);
        metadata.sequence = self.frames_out;
        self.frames_out += 1;
        metadata.set_video_dims(dims.0, dims.1, format.into());

        Ok(Buffer::new(MemoryHandle::with_len(slot, total), metadata))
    }

    /// Claim the input metadata whose pts rode through dav1d as the packet
    /// timestamp (FIFO fallback for untimestamped streams).
    fn claim_meta_for(&mut self, pts: i64) -> crate::metadata::Metadata {
        self.pending_meta
            .iter()
            .position(|m| m.pts.nanos() as i64 == pts)
            .and_then(|i| self.pending_meta.remove(i))
            .or_else(|| self.pending_meta.pop_front())
            .unwrap_or_default()
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

        // Steady state — nothing queued: the fresh picture de-strides
        // straight into an arena slot, no intermediate frame (#139). Owned
        // copies only happen when pictures burst faster than they're
        // emitted (send_frame's Again drain, or >1 picture per send).
        if self.ready.is_empty() {
            match self.decoder.get_picture() {
                Ok(picture) => {
                    let out = self.picture_to_slot(&picture)?;
                    self.drain_pictures()?; // surplus, if any → owned queue
                    // ACCURATE clipping (#165): decoded, out-of-segment.
                    if self.clip.clips(out.metadata().pts) {
                        return Ok(None);
                    }
                    return Ok(Some(out));
                }
                Err(dav1d::Error::Again) => return Ok(None),
                Err(e) => {
                    return Err(Error::InvalidSegment(format!(
                        "dav1d decode failed: {:?}",
                        e
                    )));
                }
            }
        }
        // Frames already queued: keep display order — append the fresh
        // pictures, emit the oldest (skipping any that clip out-of-segment).
        self.drain_pictures()?;
        loop {
            match self.emit_ready()? {
                Some(b) if self.clip.clips(b.metadata().pts) => continue,
                other => return Ok(other),
            }
        }
    }

    fn handle_downstream_event(
        &mut self,
        event: crate::event::Event,
    ) -> Option<crate::event::Event> {
        self.clip.observe(&event);
        Some(event)
    }

    /// Drain the pictures dav1d still holds (its frame-delay pipeline) at
    /// EOS, one per call until empty.
    fn flush(&mut self) -> Result<Option<Buffer>> {
        // Loop, not a single step: a clipped frame must not end the drain —
        // the executor stops calling flush() at the first None.
        loop {
            if let Some(buf) = self.emit_ready()? {
                if self.clip.clips(buf.metadata().pts) {
                    continue;
                }
                return Ok(Some(buf));
            }
            match self.decoder.get_picture() {
                Ok(picture) => {
                    let out = self.picture_to_slot(&picture)?;
                    if self.clip.clips(out.metadata().pts) {
                        continue;
                    }
                    return Ok(Some(out));
                }
                Err(dav1d::Error::Again) => return Ok(None),
                Err(e) => {
                    return Err(Error::InvalidSegment(format!(
                        "dav1d decode failed: {:?}",
                        e
                    )));
                }
            }
        }
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::native() // Native code (FFI), might crash on bad input
    }
}
