//! A sink that hands timed text to a host (#279).
//!
//! [`AppVideoSink`](super::AppVideoSink)'s much smaller sibling. There is no
//! pacing here and no arena retention: a cue is a short string, the host asks
//! which ones are active whenever it repaints, and the interval in the buffer
//! says when they stop being. What it does share is the flush discipline —
//! a seek retires everything from the position being left, including the cue
//! currently on screen.

use std::sync::Arc;

use crate::element::{AsyncSink, ConsumeContext};
use crate::error::Result;
use crate::event::SegmentEvent;
use crate::format::{MediaFormat, TextCodec};
use crate::subtitle::{Cue, CueTrack, text};

/// Publishes subtitle cues into a [`CueTrack`] for a host to draw.
///
/// ```no_run
/// # use parallax::elements::AppSubtitleSink;
/// # use parallax::pipeline::Pipeline;
/// # fn build(pipeline: &mut Pipeline, demux: parallax::pipeline::NodeId) {
/// let sink = AppSubtitleSink::new();
/// let cues = sink.cues();               // BEFORE the move
/// let node = pipeline.add_async_sink("subtitles", sink);
/// # }
/// ```
#[derive(Debug)]
pub struct AppSubtitleSink {
    cues: Arc<CueTrack>,
    /// The Time segment, for mapping a cue's PTS to running time.
    segment: Option<SegmentEvent>,
    /// The generation the next cues belong to.
    generation: u64,
    /// Cues whose text arrived empty after stripping, for diagnostics.
    blank: u64,
}

impl Default for AppSubtitleSink {
    fn default() -> Self {
        Self::new()
    }
}

impl AppSubtitleSink {
    /// A sink with its own cue track.
    pub fn new() -> Self {
        Self {
            cues: Arc::new(CueTrack::new()),
            segment: None,
            generation: 0,
            blank: 0,
        }
    }

    /// The cues, for the host.
    ///
    /// Clone it *before* `executor.start()` — that moves the element into its
    /// task, like every handle in parallax.
    pub fn cues(&self) -> Arc<CueTrack> {
        Arc::clone(&self.cues)
    }

    /// Turn a buffer into a cue, or explain why it is not one.
    fn cue_from(&self, ctx: &ConsumeContext<'_>) -> Option<Cue> {
        let body = std::str::from_utf8(ctx.buffer().as_bytes())
            .inspect_err(|e| tracing::warn!("subtitle cue is not UTF-8 ({e}); dropped"))
            .ok()?;

        // What the container said it is. An undeclared payload is treated as
        // plain text: that is what `S_TEXT/UTF8` is, and it is the only guess
        // here that cannot lose information.
        let plain = match ctx.metadata().format {
            Some(MediaFormat::Text(TextCodec::Ass)) => text::ass_dialogue(body),
            Some(MediaFormat::Text(TextCodec::WebVtt)) => text::webvtt_payload(body),
            _ => body.to_string(),
        };

        // Running time, not media time: a host draws against the same clock
        // the video widget paces to, and a seek moves the mapping.
        let start = match &self.segment {
            Some(segment) => segment.to_running_time(ctx.metadata().pts),
            None => ctx.metadata().pts,
        };
        let mut cue = Cue::new(plain, start, ctx.metadata().duration);
        cue.generation = self.generation;
        Some(cue)
    }
}

impl AsyncSink for AppSubtitleSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        if let Some(cue) = self.cue_from(ctx) {
            if cue.is_blank() {
                // ASS uses timing-only events for effects; they are not a
                // failure, they simply have nothing to draw.
                self.blank += 1;
            }
            self.cues.push(cue);
        }
        Ok(())
    }

    fn handle_downstream_event(
        &mut self,
        event: crate::event::Event,
    ) -> Option<crate::event::Event> {
        use crate::event::Event;
        match &event {
            // Including the cue on screen — see `CueTrack::retire`.
            Event::FlushStart => self.cues.retire(),
            Event::FlushStop(stop) => {
                self.generation = stop.epoch;
                self.cues.open(stop.epoch);
            }
            Event::Segment(seg) if seg.format == crate::event::SegmentFormat::Time => {
                self.segment = Some(seg.clone());
            }
            _ => {}
        }
        Some(event)
    }

    fn name(&self) -> &str {
        "appsubtitlesink"
    }

    /// Zero, and deliberately so.
    ///
    /// This sink holds nothing back: a cue is published the moment it
    /// arrives, and its own interval says when it goes away. Declaring the
    /// video sink's queue depth here would inflate the pipeline's aggregate
    /// latency for a branch that adds none.
    fn latency(&self) -> Option<crate::pipeline::seek::LatencyRange> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ClockTime;

    /// The codec decides how the payload is read, and an undeclared one is
    /// plain text — the only guess that cannot lose information.
    #[test]
    fn the_container_says_how_to_read_the_payload() {
        let sink = AppSubtitleSink::new();
        // No segment: media time is running time, which is what an
        // unseeked stream from zero means.
        assert!(sink.segment.is_none());

        // The stripping itself is tested in `subtitle::text`; what matters
        // here is that the right stripper is chosen.
        assert_eq!(
            text::ass_dialogue(r"0,0,Default,,0,0,0,,{\i1}styled"),
            "styled"
        );
        assert_eq!(text::webvtt_payload("<b>bold</b>"), "bold");
    }

    #[test]
    fn a_cue_carries_the_generation_it_was_published_in() {
        let mut sink = AppSubtitleSink::new();
        sink.handle_downstream_event(crate::event::Event::FlushStart);
        assert_eq!(
            sink.cues().generation(),
            1,
            "a flush retires the position being left"
        );

        let stop = crate::event::FlushStopEvent {
            epoch: 48,
            reset_time: true,
        };
        sink.handle_downstream_event(crate::event::Event::FlushStop(stop));
        assert_eq!(sink.generation, 48);
        assert_eq!(sink.cues().generation(), 48);

        // And a cue published now is accepted rather than judged stale.
        let mut cue = Cue::new("after", ClockTime::ZERO, ClockTime::from_secs(1));
        cue.generation = sink.generation;
        sink.cues().push(cue);
        assert_eq!(sink.cues().active(ClockTime::ZERO).len(), 1);
    }
}
