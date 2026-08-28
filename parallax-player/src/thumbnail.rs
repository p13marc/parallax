//! Decoding one frame at an arbitrary time, off the playing pipeline (#282).
//!
//! What the seek bar wants is a picture at the instant under the cursor, on a
//! second decode context, without disturbing playback.
//!
//! The issue proposed a second short-lived *pipeline* — `Mp4Demux` → decoder →
//! `AppSink` → `JpegEncoder` — built and dropped per hover. It is right that
//! nothing here decodes a frame at T, and wrong that a pipeline is needed:
//! every video decoder implements [`Element`] directly (#160), and `process`
//! and `flush` are ordinary synchronous methods. Driving a demuxer straight
//! into a decoder off any graph is already how `tests/mp4_av1_tests.rs`,
//! `tests/codec_tests.rs` and `benches/media_path.rs` work.
//!
//! So there is no `Pipeline`, no `Executor`, no `AppSink`, no seek event, no
//! flush epoch, and nothing to tear down but a `File` — the same argument
//! `snapshot.rs` makes for not putting a still capture in the graph.
//!
//! What that buys, beyond less code: the demuxer and the decoder stay open
//! between requests. An MP4 index parse is 100 ms and more, so rebuilding one
//! per hover tick would be the whole latency budget. Opening it once and
//! seeking it repeatedly is what makes a preview feel attached to the cursor.
//!
//! # Two things it will not do, and says so
//!
//! **Cue-less Matroska.** Such a file still seeks — the fallback rewinds and
//! scans forward from the start — so a thumbnailer built on it would appear to
//! work and then read gigabytes per hover. [`MkvDemux::has_cues`] is the
//! question to ask first.
//!
//! **HEVC, and anything else with no decoder.** Already refused a step
//! earlier: `engine::probe` will not open such a file at all, so in practice
//! this arm is unreachable from the player and exists so the module is honest
//! on its own.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

use parallax::clock::ClockTime;
use parallax::converters::{PixelFormat as ConvertFormat, ScaleEngine};
use parallax::element::{Demuxer, DemuxerProduce, Element};
use parallax::event::{Event, EventResult, SeekEvent, SeekFlags, SeekSnap};
use parallax::format::PixelFormat;
use parallax::memory::OutputBudget;
use parallax::video::{FrameView, VideoFrame};

use crate::engine::{self, ProbedSource};
use crate::snapshot;

/// How many decoded frames a single request will walk before giving up on
/// reaching the requested instant and showing what it has.
///
/// A seek lands on the keyframe at-or-before the target, so reaching the
/// target means decoding the rest of that GOP. Most are well under a second;
/// a pathological one is not worth a scrubber freezing over, and the frame
/// this returns instead is still from the right neighbourhood.
const MAX_DECODE_FRAMES: usize = 300;

/// A decoded, downscaled preview frame: packed 8-bit RGB.
#[derive(Clone)]
pub struct Thumb {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Presentation time of the frame actually decoded, which is at or after
    /// the requested one — or before it, when the walk hit
    /// [`MAX_DECODE_FRAMES`].
    pub pts_ns: u64,
    /// `width * height * 4` bytes of packed RGBA — what Iced's
    /// `image::Handle::from_rgba` takes.
    pub rgba: Vec<u8>,
}

/// Hand-written: an Iced `Message` must be `Debug`, and a derived one would
/// print forty thousand bytes into the log for every preview.
impl std::fmt::Debug for Thumb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Thumb")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pts_ns", &self.pts_ns)
            .field("bytes", &self.rgba.len())
            .finish()
    }
}

/// An open file, a decoder, and the ability to ask for a frame at a time.
///
/// Blocking throughout: hold it behind a mutex and call [`frame_at`] on
/// `spawn_blocking`, never on the interface's thread.
///
/// [`frame_at`]: Self::frame_at
pub struct Thumbnailer {
    source: Source,
    decoder: Box<dyn Element + Send>,
    /// Longest edge of the preview, in pixels.
    max_edge: u32,
}

enum Source {
    Mp4 {
        demux: Box<engine::ThumbMp4>,
        track: u32,
    },
    Mkv {
        demux: Box<engine::ThumbMkv>,
    },
}

impl Thumbnailer {
    /// Open `path` a second time and get ready to decode previews from it.
    ///
    /// `Err` is a refusal to be shown as "no preview", not a failure to
    /// report: the seek bar should keep working without one.
    pub fn open(path: &Path, max_edge: u32) -> Result<Self> {
        let probed = engine::probe(path).with_context(|| "reopening for previews")?;
        let (source, description) = probed.into_parts();

        let video = description
            .video
            .as_ref()
            .ok_or_else(|| anyhow!("no video track"))?;
        if description.duration.is_none() {
            bail!("the container declares no duration, so there is nothing to scrub");
        }

        let decoder = engine::software_decoder(&video.codec, video.height)?;

        let source = match source {
            ProbedSource::Mp4(demux) => {
                let track = demux
                    .video_track_id()
                    .ok_or_else(|| anyhow!("no video track"))?;
                Source::Mp4 { demux, track }
            }
            ProbedSource::Mkv(demux) => {
                if !demux.has_cues() {
                    bail!(
                        "no cue table — every seek would be a linear scan from the start, \
                         which is not something to do per hover"
                    );
                }
                // Nothing here wants sound, and an unrouted audio pad would
                // hand us buffers to throw away on every produce().
                Source::Mkv {
                    demux: Box::new(demux.video_only()),
                }
            }
        };

        let mut this = Self {
            source,
            decoder,
            max_edge,
        };
        this.size_arenas();
        Ok(this)
    }

    /// One frame in, one frame out, so the static floors are the wrong size.
    ///
    /// Without this the decoder maps its full `VIDEO_DECODER_SLOT_COUNT` of
    /// megabyte-plus slots — hundreds of megabytes for a preview — because
    /// `OutputBudget::resolve` falls back to the floor when no executor has
    /// told it otherwise.
    fn size_arenas(&mut self) {
        let budget = OutputBudget::new(1, 2);
        self.decoder.set_output_budget(budget);
        match &mut self.source {
            Source::Mp4 { demux, .. } => demux.set_output_budget(budget),
            Source::Mkv { demux } => demux.set_output_budget(budget),
        }
    }

    /// Decode a preview at `at_ns`.
    pub fn frame_at(&mut self, at_ns: u64) -> Result<Thumb> {
        self.seek(at_ns)?;

        let mut newest: Option<VideoFrame> = None;
        let mut walked = 0usize;

        while walked < MAX_DECODE_FRAMES {
            let Some(input) = self.next_access_unit()? else {
                break;
            };
            let Some(out) = self
                .decoder
                .process(input)
                .context("decoding a preview frame")?
            else {
                continue;
            };
            walked += 1;

            let pts = out.metadata().pts.nanos();
            let frame = VideoFrame::from_buffer(&out).context("reading the decoded frame")?;
            let reached = pts >= at_ns;
            newest = Some(frame);
            if reached {
                break;
            }
        }

        // A target past the last keyframe legitimately decodes nothing until
        // the codec is drained: openh264 and dav1d hold frames back for
        // reorder, so the picture we want can still be inside the decoder.
        if newest.is_none() {
            while let Some(out) = self.decoder.flush().context("draining the decoder")? {
                newest = Some(VideoFrame::from_buffer(&out).context("reading the decoded frame")?);
            }
        }

        let frame = newest.ok_or_else(|| anyhow!("no frame decoded at {at_ns} ns"))?;
        self.to_thumb(&frame)
    }

    /// Reposition both the container and the decoder.
    fn seek(&mut self, at_ns: u64) -> Result<()> {
        match &mut self.source {
            Source::Mp4 { demux, .. } => {
                demux
                    .seek_all_to_time(at_ns, SeekSnap::Before)
                    .with_context(|| format!("seeking to {at_ns} ns"))?;
            }
            Source::Mkv { demux } => {
                // `SNAP_BEFORE` on top of the always-on KEY_UNIT: the walk
                // below decodes forward to the target, so landing at-or-before
                // it is what makes reaching it possible at all. MKV's default
                // is forward, which would overshoot by a whole GOP.
                let seek = SeekEvent::new_time(ClockTime::from_nanos(at_ns))
                    .with_flags(SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::SNAP_BEFORE);
                match demux.handle_upstream_event(&Event::Seek(seek)) {
                    EventResult::Error => bail!("the demuxer refused a seek to {at_ns} ns"),
                    EventResult::NotHandled => bail!("the demuxer does not seek in time"),
                    _ => {}
                }
            }
        }

        // The decoder is mid-GOP as far as it knows. `FlushStart` is what the
        // executor sends across a seek, and what makes `H264Decoder` rebuild
        // its openh264 instance rather than refuse everything until the next
        // IDR; the output is discarded exactly as the executor discards it.
        let _ = self.decoder.handle_downstream_event(Event::FlushStart);
        while self.decoder.flush().unwrap_or(None).is_some() {}
        Ok(())
    }

    /// The next encoded video access unit, or `None` at end of file.
    fn next_access_unit(&mut self) -> Result<Option<parallax::buffer::Buffer>> {
        match &mut self.source {
            Source::Mp4 { demux, track } => Ok(demux
                .read_sample(*track)
                .context("reading a sample")?
                .map(|s| s.buffer)),
            Source::Mkv { demux } => loop {
                match demux.produce().context("demuxing")? {
                    DemuxerProduce::Routed(routed) => {
                        // Video is `PadId(0)`; `video_only()` should mean it is
                        // the only one, and an empty `Routed` is how the
                        // demuxer yields while discarding a post-seek skip.
                        if let Some((_, buf)) = routed.into_iter().find(|(pad, _)| pad.0 == 0) {
                            return Ok(Some(buf));
                        }
                    }
                    DemuxerProduce::Eos => return Ok(None),
                    other => bail!("unexpected demuxer output: {other:?}"),
                }
            },
        }
    }

    /// Downscale, then convert. Scale first because that is the cheap order
    /// for a downscale — the colour conversion then runs over a few thousand
    /// pixels instead of two million — and it is the order the converter
    /// registry picks for the same reason.
    fn to_thumb(&self, frame: &VideoFrame) -> Result<Thumb> {
        let view = frame.view();
        if view.width == 0 || view.height == 0 {
            bail!("the decoded frame has no dimensions");
        }

        let (target_w, target_h) = target_size(&view, self.max_edge);
        let scale_format: ConvertFormat = view
            .format
            .try_into()
            .map_err(|_| anyhow!("cannot scale {:?}", view.format))?;
        let engine = ScaleEngine::new(view.width, view.height, target_w, target_h, scale_format)
            .context("building the scaler")?;

        let mut scaled = vec![0u8; engine.output_size()];
        let layout = view.layout.unwrap_or_else(|| engine.packed_input_layout());
        engine
            .scale(frame.data.as_bytes(), layout, &mut scaled)
            .context("scaling the frame")?;

        let rgba = snapshot::convert_to(
            ConvertFormat::Rgba,
            &scaled,
            PixelFormat::from(scale_format),
            None,
            view.color,
            target_w,
            target_h,
        )?;

        Ok(Thumb {
            width: target_w,
            height: target_h,
            pts_ns: frame.data.metadata().pts.nanos(),
            rgba,
        })
    }
}

/// The preview's pixel dimensions: the frame's *display* aspect, longest edge
/// capped at `max_edge`, both even.
///
/// Even because [`ScaleEngine`] refuses odd dimensions for a YUV format, which
/// every decoder here emits — a chroma plane at half resolution has no
/// half-pixel.
///
/// Display aspect, so an anamorphic file previews at the shape it plays at
/// (#266). Rotation is deliberately *not* applied: rotating pixels needs a
/// filter this has no reason to own, and `CpuBackend` already draws rotated
/// content upright, so a preview that matches the picture on screen is the
/// more useful answer than one that does not.
fn target_size(view: &FrameView, max_edge: u32) -> (u32, u32) {
    let (mut w, mut h) = (view.width, view.height);
    if let Some(display) = view.display {
        let upright = parallax::format::DisplayGeometry {
            rotation: parallax::format::Rotation::None,
            ..display
        };
        let (dw, dh) = upright.display_size(w, h);
        if dw > 0 && dh > 0 {
            (w, h) = (dw, dh);
        }
    }

    let longest = w.max(h);
    if longest > max_edge {
        let scale = f64::from(max_edge) / f64::from(longest);
        w = ((f64::from(w) * scale).round() as u32).max(2);
        h = ((f64::from(h) * scale).round() as u32).max(2);
    }
    // Round down to even, but never to zero.
    ((w & !1).max(2), (h & !1).max(2))
}

/// How many preview frames to keep. A slow drag across a two-hour film asks
/// for a few hundred distinct buckets; sixty of them at 160x90 is under a
/// megabyte, and the ones a user goes back to are the ones they just left.
const CACHE_ENTRIES: usize = 60;

/// Consecutive decode failures after which previews are given up on.
///
/// One failure is a frame; three in a row is the file.
const MAX_FAILURES: usize = 3;

/// The longest edge of a preview, in pixels.
pub const MAX_EDGE: u32 = 160;

/// Whether previews are possible for the file that is open.
enum Availability {
    /// Nobody has hovered yet, so nothing has been opened.
    Unknown,
    /// Open and working.
    Ready,
    /// Refused, with the reason — a cue-less MKV, no decoder, no duration.
    Unavailable(String),
}

/// The seek bar's preview state: what is on screen, what is being decoded,
/// and what has been asked for since (#282).
///
/// Coalescing is a **single slot**, not a queue. An Iced `Task` cannot be
/// cancelled and aborting one would not stop a `spawn_blocking` closure that
/// has already started, so "cancel on hover-move" is really "let the answer
/// arrive and decide it is stale" — the same generation-counter shape the
/// probe path uses for the same reason. One request is in flight at a time and
/// at most one waits behind it, so a fast drag across the bar decodes the
/// instant the cursor started on, then the one it ended on, and nothing in
/// between.
pub struct Previews {
    /// The decoder, opened on the first hover and kept.
    ///
    /// Behind a mutex because the blocking closure needs it by value and the
    /// interface keeps it between requests. Contention is impossible by
    /// construction — `in_flight` allows only one holder — but the lock is
    /// what lets the closure own it for the duration.
    slot: std::sync::Arc<std::sync::Mutex<Option<Thumbnailer>>>,
    availability: Availability,
    /// The file `slot` was opened on, so a playlist advance rebuilds it.
    path: Option<std::path::PathBuf>,
    /// Width of the scrubber in pixels, learned from the widget itself.
    ///
    /// The scrubber is `Length::Fill` and `MouseArea::on_move` reports a
    /// widget-local point with no size, so without this an x offset cannot be
    /// turned into a fraction of the file.
    bar_width: f32,
    /// Where the cursor is over the bar, as a fraction, while it is there.
    hovered: Option<f32>,
    /// The bucket currently being decoded.
    in_flight: Option<u64>,
    /// The newest bucket asked for while one was in flight.
    pending: Option<u64>,
    /// Bumped whenever the open file changes, so an answer for the previous
    /// one is discarded rather than drawn over the new one.
    generation: u64,
    /// Decoded previews, ready to draw. Cached as Iced handles, not as
    /// pixels: `Handle::from_rgba` mints a fresh texture id every call, so
    /// building one per `view()` would re-upload the image on every redraw —
    /// and a video player redraws at the frame rate.
    cache: lru::LruCache<u64, Preview>,
    /// Decode failures since the last success.
    failures: usize,
}

impl Default for Previews {
    fn default() -> Self {
        Self {
            slot: Default::default(),
            availability: Availability::Unknown,
            path: None,
            bar_width: 0.0,
            hovered: None,
            in_flight: None,
            pending: None,
            generation: 0,
            cache: lru::LruCache::new(
                std::num::NonZeroUsize::new(CACHE_ENTRIES).expect("non-zero"),
            ),
            failures: 0,
        }
    }
}

/// What [`Previews::hover`] decided should happen.
pub enum Request {
    /// Nothing: no duration, previews unavailable, or the answer is cached.
    Idle,
    /// Decode this instant, tagging the answer with this generation.
    Decode {
        /// Where in the file, rounded to a cache bucket.
        at_ns: u64,
        /// The generation to tag the answer with.
        generation: u64,
    },
}

impl Previews {
    /// Point at a new file. Everything decoded for the last one is wrong now.
    pub fn open(&mut self, path: std::path::PathBuf) {
        self.slot = Default::default();
        self.availability = Availability::Unknown;
        self.path = Some(path);
        self.hovered = None;
        self.in_flight = None;
        self.pending = None;
        self.generation = self.generation.wrapping_add(1);
        self.cache.clear();
        self.failures = 0;
    }

    /// Forget everything: no file is open.
    pub fn close(&mut self) {
        self.open(std::path::PathBuf::new());
        self.path = None;
    }

    /// The file previews are being decoded from, if any.
    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
    }

    /// The scrubber's measured width, from the `Sensor` around it.
    pub fn set_bar_width(&mut self, width: f32) {
        self.bar_width = width;
    }

    /// Whether previews are worth offering at all — false once the file has
    /// refused, so the interface can draw a bar with nothing above it rather
    /// than a spinner that never resolves.
    pub fn unavailable(&self) -> bool {
        matches!(self.availability, Availability::Unavailable(_))
    }

    /// Why previews are not on offer, once that has been decided.
    pub fn refusal(&self) -> Option<&str> {
        match &self.availability {
            Availability::Unavailable(why) => Some(why),
            _ => None,
        }
    }

    /// The cursor moved to `x` pixels along the bar, over a file of
    /// `duration_ns`.
    ///
    /// Returns what to do about it.
    pub fn hover(&mut self, x: f32, duration_ns: u64) -> Request {
        if self.bar_width <= 0.0 || duration_ns == 0 || self.unavailable() {
            return Request::Idle;
        }
        let fraction = (x / self.bar_width).clamp(0.0, 1.0);
        self.hovered = Some(fraction);

        let at_ns = bucket(fraction, duration_ns);
        if self.cache.contains(&at_ns) {
            return Request::Idle;
        }
        if self.in_flight == Some(at_ns) || self.pending == Some(at_ns) {
            return Request::Idle;
        }
        if self.in_flight.is_some() {
            // The single slot: this replaces whatever was waiting, so a drag
            // never builds a backlog of instants nobody is looking at any more.
            self.pending = Some(at_ns);
            return Request::Idle;
        }

        self.in_flight = Some(at_ns);
        Request::Decode {
            at_ns,
            generation: self.generation,
        }
    }

    /// The cursor left the bar.
    pub fn leave(&mut self) {
        self.hovered = None;
    }

    /// A decode finished. Returns the next request, if one was waiting.
    pub fn finished(
        &mut self,
        generation: u64,
        at_ns: u64,
        result: Result<Thumb, String>,
        opened: bool,
    ) -> Request {
        if generation != self.generation {
            // A different file is open now; the answer describes the old one.
            return Request::Idle;
        }
        self.in_flight = None;

        match result {
            Ok(thumb) => {
                self.failures = 0;
                self.availability = Availability::Ready;
                self.cache.put(
                    at_ns,
                    Preview {
                        width: thumb.width as f32,
                        height: thumb.height as f32,
                        handle: iced::widget::image::Handle::from_rgba(
                            thumb.width,
                            thumb.height,
                            thumb.rgba,
                        ),
                        pts_ns: thumb.pts_ns,
                    },
                );
            }
            // The very first request also opens the file, so a failure there
            // is the file refusing previews rather than one frame going wrong.
            Err(why) if !opened => {
                self.availability = Availability::Unavailable(why);
                self.pending = None;
                return Request::Idle;
            }
            Err(why) => {
                self.failures += 1;
                if self.failures >= MAX_FAILURES {
                    self.availability = Availability::Unavailable(why);
                    self.pending = None;
                    return Request::Idle;
                }
            }
        }

        match self.pending.take() {
            Some(next) => {
                self.in_flight = Some(next);
                Request::Decode {
                    at_ns: next,
                    generation: self.generation,
                }
            }
            None => Request::Idle,
        }
    }

    /// The preview to draw right now, with the fraction along the bar to draw
    /// it at — `None` when the cursor is elsewhere or nothing is decoded yet.
    pub fn showing(&self, duration_ns: u64) -> Option<(f32, &Preview)> {
        let fraction = self.hovered?;
        if duration_ns == 0 {
            return None;
        }
        let preview = self.cache.peek(&bucket(fraction, duration_ns))?;
        Some((fraction, preview))
    }

    /// The decoder, for the blocking side.
    pub fn decoder_slot(&self) -> std::sync::Arc<std::sync::Mutex<Option<Thumbnailer>>> {
        std::sync::Arc::clone(&self.slot)
    }
}

/// One decoded preview, ready to draw.
#[derive(Debug, Clone)]
pub struct Preview {
    /// Pixel width, for the layout.
    pub width: f32,
    /// Pixel height, for the layout.
    pub height: f32,
    /// The frame this actually shows, which is the keyframe-snapped instant
    /// rather than the one under the cursor.
    pub pts_ns: u64,
    /// The texture, minted once when the decode landed.
    pub handle: iced::widget::image::Handle,
}

/// The number of distinct instants a file is previewed at.
///
/// Fewer than the pixels along the bar, deliberately: neighbouring pixels of a
/// two-hour film are seconds apart and decoding each would be a decode per
/// mouse event. Two hundred buckets is finer than a scrubber can be aimed and
/// coarse enough that a slow drag mostly hits the cache.
const BUCKETS: u64 = 200;

/// Round a position along the bar to the instant that will be decoded for it.
fn bucket(fraction: f32, duration_ns: u64) -> u64 {
    let step = (duration_ns / BUCKETS).max(1);
    let at = (f64::from(fraction) * duration_ns as f64) as u64;
    (at / step * step).min(duration_ns.saturating_sub(1))
}

/// Open the file if it is not open yet, and decode one preview.
///
/// Blocking: the caller runs it on `spawn_blocking`. The `bool` says whether
/// the thumbnailer was already open, which is how the caller tells "this file
/// has no previews" from "this one frame failed".
pub fn decode(
    slot: &std::sync::Mutex<Option<Thumbnailer>>,
    path: &Path,
    at_ns: u64,
) -> (bool, Result<Thumb, String>) {
    let mut guard = match slot.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let opened = guard.is_some();
    if !opened {
        match Thumbnailer::open(path, MAX_EDGE) {
            Ok(t) => *guard = Some(t),
            Err(e) => return (false, Err(format!("{e:#}"))),
        }
    }
    let result = guard
        .as_mut()
        .expect("just opened")
        .frame_at(at_ns)
        .map_err(|e| format!("{e:#}"));
    (opened, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(
        width: u32,
        height: u32,
        display: Option<parallax::format::DisplayGeometry>,
    ) -> FrameView {
        FrameView {
            width,
            height,
            format: PixelFormat::I420,
            layout: None,
            color: None,
            display,
        }
    }

    #[test]
    fn a_preview_fits_its_longest_edge() {
        assert_eq!(target_size(&view(1920, 1080, None), 160), (160, 90));
        assert_eq!(target_size(&view(1080, 1920, None), 160), (90, 160));
    }

    /// Under the cap it is left alone — a preview is never upscaled.
    #[test]
    fn a_small_frame_is_not_enlarged() {
        assert_eq!(target_size(&view(64, 64, None), 160), (64, 64));
    }

    /// `ScaleEngine` refuses odd dimensions for YUV, which is every format a
    /// decoder in this player emits.
    #[test]
    fn both_edges_are_even_and_never_zero() {
        for (w, h) in [(1919u32, 1079u32), (721, 481), (3, 3), (1, 1)] {
            let (tw, th) = target_size(&view(w, h, None), 160);
            assert_eq!(tw % 2, 0, "{w}x{h} gave an odd width");
            assert_eq!(th % 2, 0, "{w}x{h} gave an odd height");
            assert!(tw >= 2 && th >= 2, "{w}x{h} collapsed to nothing");
        }
    }

    /// #266: an anamorphic file previews at the shape it plays at, not at its
    /// coded shape.
    #[test]
    fn a_non_square_pixel_ratio_is_honoured() {
        let par = parallax::format::DisplayGeometry {
            par: (16, 11),
            rotation: parallax::format::Rotation::None,
        };
        // 720x576 PAL at 16:11 displays as 1047x576 — widescreen, not 5:4.
        let (w, h) = target_size(&view(720, 576, Some(par)), 160);
        assert!(w > h, "an anamorphic PAL frame is wider than it is tall");
        assert_eq!((w, h), (160, 88));
    }

    /// The first end-to-end media test in this crate, and only feasible
    /// because there is no pipeline: the reason there is no end-to-end seek
    /// test here is that the demuxer-to-decoder link is 24 deep and the only
    /// fixture is 20 frames, so EOS beats the seek. Nothing of that applies
    /// to a demuxer driven by hand.
    #[test]
    fn a_frame_comes_back_from_an_mp4() {
        let path = std::path::Path::new("../tests/fixtures/tiny_av1_opus.mp4");
        let mut t = Thumbnailer::open(path, 160).expect("opening the AV1 fixture");

        let thumb = t.frame_at(0).expect("a preview at the start");
        assert_eq!(
            (thumb.width, thumb.height),
            (64, 64),
            "the fixture is 64x64"
        );
        assert_eq!(thumb.rgba.len(), 64 * 64 * 4, "packed RGBA");
        assert!(
            thumb.rgba.iter().any(|&b| b != thumb.rgba[0]),
            "a uniform image means nothing was actually decoded"
        );
    }

    /// The demuxer and decoder stay open, so a second ask must land somewhere
    /// else rather than replay the first answer.
    #[test]
    fn asking_twice_moves() {
        let path = std::path::Path::new("../tests/fixtures/tiny_av1_opus.mp4");
        let mut t = Thumbnailer::open(path, 160).expect("opening the AV1 fixture");

        let first = t.frame_at(0).expect("a preview at the start");
        // The fixture is ten frames at 30 fps; 200 ms is past the middle of it.
        let later = t.frame_at(200_000_000).expect("a later preview");
        assert!(
            later.pts_ns > first.pts_ns,
            "a later request came back at {} ns, not after {} ns",
            later.pts_ns,
            first.pts_ns
        );
    }

    /// Seeking backwards is the case a scrubber does constantly, and the one
    /// a decoder driven by hand gets wrong if nothing tells it about the
    /// flush.
    #[test]
    fn seeking_backwards_still_decodes() {
        let path = std::path::Path::new("../tests/fixtures/tiny_av1_opus.mp4");
        let mut t = Thumbnailer::open(path, 160).expect("opening the AV1 fixture");

        let _ = t.frame_at(200_000_000).expect("a late preview");
        let back = t.frame_at(0).expect("back to the start");
        assert!(
            back.rgba.iter().any(|&b| b != back.rgba[0]),
            "the decoder came back empty after a backward seek"
        );
    }

    fn thumb(pts_ns: u64) -> Thumb {
        Thumb {
            width: 2,
            height: 2,
            pts_ns,
            rgba: vec![0; 2 * 2 * 4],
        }
    }

    /// A file one minute long, with the bar 600 px wide.
    fn scrubbing() -> (Previews, u64) {
        let mut p = Previews::default();
        p.open(std::path::PathBuf::from("/dev/null"));
        p.set_bar_width(600.0);
        (p, 60_000_000_000)
    }

    fn decoding(request: &Request) -> Option<u64> {
        match request {
            Request::Decode { at_ns, .. } => Some(*at_ns),
            Request::Idle => None,
        }
    }

    /// "Rapid hover movement cancels cleanly and never queues work" (#282):
    /// one request in flight, at most one waiting, and the one that runs next
    /// is where the cursor *ended up* — not the twenty places it passed over.
    #[test]
    fn a_fast_drag_decodes_the_ends_and_nothing_in_between() {
        let (mut p, duration) = scrubbing();

        let first = decoding(&p.hover(0.0, duration)).expect("the first hover decodes");

        // Everything in between only replaces the single pending slot.
        for x in 1..=500 {
            assert!(
                decoding(&p.hover(x as f32, duration)).is_none(),
                "a hover during a decode must not start a second one"
            );
        }

        let next = decoding(&p.finished(p.generation, first, Ok(thumb(first)), true))
            .expect("the waiting request runs");
        assert_eq!(
            next,
            bucket(500.0 / 600.0, duration),
            "the cursor's last position is what gets decoded, not its first step"
        );
    }

    /// The cache is what makes a slow drag back over ground already covered
    /// cost nothing.
    #[test]
    fn a_decoded_instant_is_not_decoded_again() {
        let (mut p, duration) = scrubbing();
        let at = decoding(&p.hover(300.0, duration)).expect("decodes");
        let _ = p.finished(p.generation, at, Ok(thumb(at)), true);

        assert!(
            decoding(&p.hover(300.0, duration)).is_none(),
            "the same instant must come from the cache"
        );
        assert!(p.showing(duration).is_some(), "and must be drawable");
    }

    /// An answer for the file that was open before is not an answer about
    /// this one — the reason a generation counter exists at all, since an
    /// Iced task cannot be cancelled.
    #[test]
    fn an_answer_for_the_previous_file_is_discarded() {
        let (mut p, duration) = scrubbing();
        let at = decoding(&p.hover(300.0, duration)).expect("decodes");
        let stale = p.generation;

        p.open(std::path::PathBuf::from("/dev/zero"));
        let _ = p.finished(stale, at, Ok(thumb(at)), true);

        assert!(p.showing(duration).is_none(), "the old frame was drawn");
        assert!(
            !p.unavailable(),
            "and it must not have poisoned the new file"
        );
    }

    /// The first request also opens the file, so a failure there is the file
    /// refusing previews — not one frame going wrong.
    #[test]
    fn a_failure_to_open_gives_up_immediately() {
        let (mut p, duration) = scrubbing();
        let at = decoding(&p.hover(300.0, duration)).expect("decodes");

        let _ = p.finished(p.generation, at, Err("no cue table".into()), false);

        assert!(p.unavailable());
        assert_eq!(p.refusal(), Some("no cue table"));
        assert!(
            decoding(&p.hover(400.0, duration)).is_none(),
            "a refused file must not keep being asked"
        );
    }

    /// One bad frame is not a bad file. Three in a row is.
    #[test]
    fn a_run_of_decode_failures_gives_up_but_a_single_one_does_not() {
        let (mut p, duration) = scrubbing();

        for step in 1..MAX_FAILURES {
            let at = decoding(&p.hover(100.0 * step as f32, duration)).expect("decodes");
            let _ = p.finished(p.generation, at, Err("one bad frame".into()), true);
            assert!(!p.unavailable(), "gave up after {step} failure(s)");
        }

        let at = decoding(&p.hover(500.0, duration)).expect("decodes");
        let _ = p.finished(p.generation, at, Err("one bad frame".into()), true);
        assert!(
            p.unavailable(),
            "a run of failures is the file, not the frame"
        );
    }

    /// Without a measured bar there is no way to turn x into a fraction, and
    /// guessing would decode the wrong instant.
    #[test]
    fn nothing_is_decoded_before_the_bar_has_been_measured() {
        let mut p = Previews::default();
        p.open(std::path::PathBuf::from("/dev/null"));
        assert!(decoding(&p.hover(300.0, 60_000_000_000)).is_none());
    }

    /// A live stream has nothing to scrub against.
    #[test]
    fn a_file_of_unknown_length_is_never_previewed() {
        let (mut p, _) = scrubbing();
        assert!(decoding(&p.hover(300.0, 0)).is_none());
    }

    /// The MKV path: a different demuxer, a different seek mechanism, and the
    /// cue check that decides whether previews are offered at all.
    #[test]
    fn the_matroska_path_either_previews_or_says_why_not() {
        let path = std::path::Path::new("../tests/fixtures/tiny_h264_aac.mkv");
        match Thumbnailer::open(path, 160) {
            Ok(mut t) => {
                let thumb = t.frame_at(0).expect("a preview at the start");
                assert!(thumb.width >= 2 && thumb.height >= 2);
                assert_eq!(thumb.rgba.len() as u32, thumb.width * thumb.height * 4);
                assert!(
                    thumb.rgba.iter().any(|&b| b != thumb.rgba[0]),
                    "a uniform image means nothing was actually decoded"
                );
            }
            Err(e) => {
                let why = format!("{e:#}");
                assert!(
                    why.contains("cue"),
                    "the only acceptable Matroska refusal here is a missing cue table, got: {why}"
                );
            }
        }
    }

    /// #282's "degrade to no preview": a container with no duration has
    /// nothing to scrub, and a cue-less MKV would linear-scan per hover.
    /// Neither is an error to report — the seek bar just goes without.
    #[test]
    fn a_file_with_no_decoder_is_refused_rather_than_opened() {
        let path = std::path::Path::new("../tests/fixtures/tiny_hevc.mp4");
        assert!(
            Thumbnailer::open(path, 160).is_err(),
            "HEVC has no decoder in this tree; a preview must decline"
        );
    }

    /// Rotation is not applied, so it must not change the shape either: a
    /// preview that swapped axes without rotating pixels would be squashed.
    #[test]
    fn rotation_does_not_swap_the_preview_axes() {
        let turned = parallax::format::DisplayGeometry {
            par: (1, 1),
            rotation: parallax::format::Rotation::Cw90,
        };
        assert_eq!(
            target_size(&view(1920, 1080, Some(turned)), 160),
            target_size(&view(1920, 1080, None), 160),
        );
    }
}
