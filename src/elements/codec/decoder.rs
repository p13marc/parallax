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

use super::common::PixelFormat;

/// Zero-copy input wrapper for `dav1d::Decoder::send_data` (#195).
///
/// dav1d boxes this and reads the payload through `AsRef<[u8]>`, calling the
/// drop from its data-release callback — possibly on one of its worker
/// threads (`Buffer` is `Send`). Until then the upstream arena slot stays
/// pinned: two atomics per frame instead of a full compressed-TU heap copy.
/// The pin is bounded by dav1d's small input queue, tiny next to the
/// demuxers' 32-slot output arenas.
struct InputData(Buffer);

impl AsRef<[u8]> for InputData {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

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
    /// The settings the decoder was built with, kept so the `with_*`
    /// builders can rebuild it (#189). dav1d only takes settings at
    /// construction.
    settings: dav1d::Settings,
    frame_count: u64,
    /// Arena for output buffer allocation.
    output: OutputArena,
    /// Decoded pictures waiting to be emitted (dav1d can release several
    /// after one send once its frame-delay pipeline fills). Held as
    /// `dav1d::Picture`s — refcounted views over dav1d-owned memory — so
    /// queueing costs nothing and the single de-stride copy into an arena
    /// slot happens at emit time (#195; the old owned-`VideoFrame` queue
    /// paid a second full-frame copy on every queued picture).
    ready: std::collections::VecDeque<dav1d::Picture>,
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
    /// Create a new dav1d decoder with default settings (auto threads, auto
    /// frame delay, film-grain synthesis on).
    pub fn new() -> Result<Self> {
        Self::build(dav1d::Settings::new())
    }

    /// Create a decoder with custom settings.
    ///
    /// Prefer the `with_*` builders for the common knobs — they keep the
    /// caller off the `dav1d` crate.
    pub fn with_settings(settings: &dav1d::Settings) -> Result<Self> {
        // dav1d::Settings is not Clone; round-trip the values we expose.
        let mut own = dav1d::Settings::new();
        own.set_n_threads(settings.get_n_threads());
        own.set_max_frame_delay(settings.get_max_frame_delay());
        own.set_apply_grain(settings.get_apply_grain());
        own.set_operating_point(settings.get_operating_point());
        own.set_all_layers(settings.get_all_layers());
        own.set_frame_size_limit(settings.get_frame_size_limit());
        Self::build(own)
    }

    fn build(settings: dav1d::Settings) -> Result<Self> {
        let decoder = dav1d::Decoder::with_settings(&settings)
            .map_err(|e| Error::Config(format!("Failed to create dav1d decoder: {:?}", e)))?;
        Ok(Self {
            decoder,
            settings,
            frame_count: 0,
            output: OutputArena::new(defaults::VIDEO_DECODER_SLOT_COUNT),
            ready: std::collections::VecDeque::new(),
            pending_meta: std::collections::VecDeque::new(),
            last_dims: None,
            frames_out: 0,
            clip: Default::default(),
        })
    }

    /// Rebuild the decoder from `self.settings`. Builders only — dav1d takes
    /// settings at construction, and a rebuild discards decode state, so this
    /// must run before any data is fed.
    fn reconfigure(mut self) -> Result<Self> {
        self.decoder = dav1d::Decoder::with_settings(&self.settings)
            .map_err(|e| Error::Config(format!("Failed to create dav1d decoder: {:?}", e)))?;
        Ok(self)
    }

    /// Decoder worker threads. `0` (the default) lets dav1d use every
    /// logical core.
    ///
    /// At paced (realtime) rates, worker CPU scales with thread count
    /// almost independently of the work (#192: 1080p60 on an 8-thread
    /// machine measured ~0.45 cores at 2 threads vs ~2.3 at 8, identical
    /// output) — idle workers churn on dav1d's task queues. A player
    /// should size this to the stream (2 is ample for 1080p60, which
    /// batch-decodes at ~3.7× realtime on 2 threads); a throughput
    /// transcode wants the default.
    pub fn with_threads(mut self, n_threads: u32) -> Result<Self> {
        self.settings.set_n_threads(n_threads);
        self.reconfigure()
    }

    /// Cap dav1d's internal frame-delay pipeline (#189).
    ///
    /// `0` (the default) derives the delay from the thread count — on an
    /// 8-thread machine that holds ~8 internal pictures (~25-50 MB at
    /// 1080p) and adds ~8 frames of latency. A player wants a small bound
    /// (2 is comfortable for 1080p60 on a desktop); a throughput transcode
    /// wants the default. (#189 measured delay=2 raising decode CPU ~50%
    /// at 8 threads; the #192 re-measure found it CPU-neutral — treat the
    /// penalty as configuration-dependent and re-measure before relying
    /// on either number.)
    pub fn with_max_frame_delay(mut self, max_frame_delay: u32) -> Result<Self> {
        self.settings.set_max_frame_delay(max_frame_delay);
        self.reconfigure()
    }

    /// Toggle film-grain synthesis (default on). Turning it off trades
    /// fidelity on grainy streams for decode time.
    pub fn with_apply_grain(mut self, apply_grain: bool) -> Result<Self> {
        self.settings.set_apply_grain(apply_grain);
        self.reconfigure()
    }

    /// Get the number of frames decoded.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Pull one picture off dav1d's output into `self.ready`.
    ///
    /// Returns whether a picture was obtained (`false` = `Again`, nothing
    /// ready yet).
    fn harvest_one(&mut self) -> Result<bool> {
        match self.decoder.get_picture() {
            Ok(picture) => {
                self.ready.push_back(picture);
                Ok(true)
            }
            Err(dav1d::Error::Again) => Ok(false),
            Err(e) => Err(Error::InvalidSegment(format!(
                "dav1d decode failed: {:?}",
                e
            ))),
        }
    }

    /// Drain every picture dav1d has ready into `self.ready`.
    fn drain_pictures(&mut self) -> Result<()> {
        while self.harvest_one()? {}
        Ok(())
    }

    /// Feed one temporal unit, honoring dav1d's flow control.
    ///
    /// `Again` from `send_data` is not an error: the decoder wants its
    /// ready pictures consumed before accepting more input. Drain, then
    /// resubmit the retained data until it is accepted.
    fn send_frame(&mut self, input: Buffer, pts_ns: i64) -> Result<()> {
        let mut result = self
            .decoder
            .send_data(InputData(input), None, Some(pts_ns), None);
        // Bounded: each retry either lands the input or consumes output;
        // hitting the bound means dav1d is refusing input while claiming no
        // output exists, which is a decoder bug, not flow control.
        for _ in 0..1024 {
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
        Err(Error::InvalidSegment(
            "dav1d refused input without releasing output".into(),
        ))
    }

    /// Build the output buffer for the oldest ready picture — the same
    /// single de-stride copy as the steady-state path, just deferred to
    /// emit time.
    fn emit_ready(&mut self) -> Result<Option<Buffer>> {
        match self.ready.pop_front() {
            Some(picture) => self.picture_to_slot(&picture).map(Some),
            None => Ok(None),
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
        self.send_frame(buffer, pts)?;

        // Harvest whatever dav1d has finished; `Again` while its frame
        // pipeline fills is the natural priming phase. (#192 measured the
        // feeding pattern to be CPU-neutral — worker-thread count is the
        // lever — so this stays the simple shape.)
        self.drain_pictures()?;

        // Emit the oldest ready picture, skipping any that clip
        // out-of-segment (#165). Empty while priming → None.
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
        // the caller (the adapter's drain loop) stops at the first None.
        loop {
            if let Some(buf) = self.emit_ready()? {
                if self.clip.clips(buf.metadata().pts) {
                    continue;
                }
                return Ok(Some(buf));
            }
            if !self.harvest_one()? {
                // `Again` with no more input = the pipeline is empty.
                return Ok(None);
            }
        }
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::native() // Native code (FFI), might crash on bad input
    }
}
