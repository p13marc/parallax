//! Timed text: cues, where they wait, and how to read them (#279).
//!
//! Unfeatured and toolkit-agnostic, beside [`crate::video`] and for the same
//! reason: the engine's job is to deliver *timed text*, and drawing it is the
//! host's. A `TextOverlay` element would mean rasterising glyphs into video
//! frames — losing crispness at every scale, forcing a CPU pass over frames
//! that are otherwise untouched, and duplicating a font stack the host
//! already has.
//!
//! ```text
//! demux ── subtitle pad ──▶ AppSubtitleSink ⇄ CueTrack ⇄ the host's text layer
//! ```
//!
//! What arrives from a container is already a cue: Matroska strips the timing
//! into the block (`BlockDuration` *is* the out-time) and, for ASS, the
//! `Dialogue:` prefix as well. So there is no subtitle *parser* on the
//! playback path at all — only [`text::ass_dialogue`] to take the plain words
//! out of an event body. [`srt`] exists for sidecar `.srt` files, which is
//! where a downloaded subtitle actually lives.

mod cue;
pub mod srt;
mod store;
pub mod text;

pub use cue::{Cue, DEFAULT_DURATION};
pub use store::CueTrack;
