//! The widget: a [`shader::Program`] that draws the frame that is due, and
//! asks to be woken when the next one is.

use std::sync::Arc;
use std::time::Duration;

use iced::widget::shader;
use iced::{Event, Rectangle, mouse, window};

use parallax::clock::ClockTime;
use parallax::pipeline::{Pace, StallWatch, schedule};
use parallax::video::{FrameMailbox, PacedFrame};

use crate::render::VideoPrimitive;

/// Default lateness past which a frame is not worth showing.
///
/// One frame at 25 fps, the same budget `AutoVideoSink` uses.
///
/// Public because a host that draws no window still has to pace, and two
/// hosts pacing to different budgets would report different frame counts for
/// the same file — which is exactly what the headless run is asked to prove.
pub const DEFAULT_MAX_LATENESS: Duration = Duration::from_millis(40);

/// Longest gap between redraws while a stream is live but nothing is queued.
///
/// The doorbell subscription is what normally wakes an idle widget; this is
/// the belt to its braces. A missed wake-up then costs a fifth of a second
/// rather than freezing the picture until some unrelated event arrives.
const IDLE_POLL: Duration = Duration::from_millis(200);

/// What the widget carries between redraws.
///
/// Iced rebuilds the `VideoView` on every `view()`, so anything that must
/// survive a rebuild lives here.
#[derive(Debug, Default)]
pub struct State {
    /// The frame on screen. Held so a redraw with no new frame — a resize, a
    /// UI change, a compositor repaint — draws the picture again rather than
    /// nothing.
    current: Option<PacedFrame>,
    /// Watches for a clock that has stopped for reasons no pause explains.
    stall: StallWatch,
    /// Whether the current stall has already been logged.
    warned_stall: bool,
}

impl State {
    /// The frame currently on screen, if any.
    pub fn current(&self) -> Option<&PacedFrame> {
        self.current.as_ref()
    }
}

/// Draws the video a [`FrameMailbox`] is being filled with.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use parallax::video::FrameMailbox;
/// # use parallax_iced::VideoView;
/// # fn view(mailbox: &Arc<FrameMailbox>) -> iced::Element<'_, ()> {
/// iced::widget::shader(VideoView::new(Arc::clone(mailbox)))
///     .width(iced::Fill)
///     .height(iced::Fill)
///     .into()
/// # }
/// ```
///
/// # How it is driven
///
/// A `RedrawRequested` is the tick. The widget takes whatever is due, then
/// asks Iced to redraw *at* the next frame's presentation time — expressed as
/// a duration from the instant that event carried, so pipeline running time
/// and `Instant` never have to be reconciled. In steady state that is the
/// only traffic: no messages, no `view()` rebuild per frame.
///
/// It never publishes a message from that update. Iced rebuilds the whole
/// interface when a redraw produces one — up to three times, and then it
/// warns.
#[derive(Debug)]
pub struct VideoView {
    mailbox: Arc<FrameMailbox>,
    max_lateness: Duration,
    srgb_target: bool,
}

impl VideoView {
    /// Draw the frames arriving in `mailbox`.
    pub fn new(mailbox: Arc<FrameMailbox>) -> Self {
        Self {
            mailbox,
            max_lateness: DEFAULT_MAX_LATENESS,
            // Iced picks an sRGB surface, so the shader has to hand it linear
            // values or the transfer function is applied twice.
            srgb_target: true,
        }
    }

    /// How late a frame may be and still be worth drawing.
    pub fn with_max_lateness(mut self, max_lateness: Duration) -> Self {
        self.max_lateness = max_lateness;
        self
    }

    /// Override the target's colour space. Only useful for tests: a real
    /// Iced surface is always sRGB.
    pub fn with_srgb_target(mut self, srgb: bool) -> Self {
        self.srgb_target = srgb;
        self
    }

    /// Take everything due, keeping the last of them.
    ///
    /// Draining rather than taking one is deliberate: if the host fell behind,
    /// the frames between the last drawn one and now are already superseded,
    /// and showing them in sequence at redraw speed would play the video fast.
    fn advance(&self, state: &mut State) {
        let Some(clock) = self.mailbox.clock() else {
            // No clock yet — an element only sees one through its consume
            // context, so there is nothing to pace against until the first
            // buffer arrives. Draw on arrival.
            self.drain(state, ClockTime::NONE);
            return;
        };

        let now = clock.now();
        // A clock that stops for reasons no pause explains means the branch
        // driving it is starved — possibly by this very widget holding
        // frames. Draw anyway: that drains the mailbox and unblocks it.
        if state.stall.stalled(now, clock.is_paused()) {
            if !state.warned_stall {
                state.warned_stall = true;
                tracing::warn!(
                    "parallax-iced: the pipeline clock has not moved for {:?}; \
                     drawing on arrival instead of waiting on it",
                    parallax::pipeline::STALL_LIMIT
                );
            }
            self.drain(state, ClockTime::NONE);
            return;
        }
        state.warned_stall = false;
        self.drain(state, now);
    }

    fn drain(&self, state: &mut State, now: ClockTime) {
        while let Some(paced) = self.mailbox.take_due(now, self.max_lateness) {
            state.current = Some(paced);
        }
    }
}

impl<Message> shader::Program<Message> for VideoView {
    type State = State;
    type Primitive = VideoPrimitive;

    fn update(
        &self,
        state: &mut Self::State,
        event: &Event,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Option<shader::Action<Message>> {
        // A redraw is the tick: the only event that arrives at the rate the
        // screen actually refreshes.
        let Event::Window(window::Event::RedrawRequested(instant)) = event else {
            return None;
        };

        self.advance(state);

        let Some(clock) = self.mailbox.clock() else {
            // Nothing to pace against yet; come back on the next frame.
            return Some(shader::Action::request_redraw());
        };

        // A paused clock is supposed to have stopped, so park — returning no
        // action is how a widget says "wait". The last frame stays on screen,
        // and a resume always arrives through the application's own message,
        // which redraws anyway.
        if clock.is_paused() {
            return None;
        }

        match self.mailbox.next_due() {
            // Ask `schedule` when to come back, not the raw target — it is
            // the same function `take_due` uses to decide whether a frame is
            // due, so the two cannot disagree. They did: a target more than
            // `MAX_WAIT` ahead is due *now* by the PTS-jump escape hatch,
            // while sleeping until it stalled the picture for as long as the
            // jump was.
            Some(target) => match schedule(target, clock.now(), self.max_lateness) {
                Pace::Wait(delay) => Some(shader::Action::request_redraw_at(*instant + delay)),
                Pace::Present | Pace::DropLate { .. } => Some(shader::Action::request_redraw()),
            },
            // Nothing queued. The doorbell subscription is what normally wakes
            // us; this is the fallback so a missed one costs a fifth of a
            // second rather than the picture.
            None if self.mailbox.end_reason().is_none() => {
                Some(shader::Action::request_redraw_at(*instant + IDLE_POLL))
            }
            None => None,
        }
    }

    fn draw(
        &self,
        state: &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        VideoPrimitive {
            id: self.mailbox.id(),
            frame: state.current.as_ref().map(|paced| paced.frame.clone()),
            srgb_target: self.srgb_target,
        }
    }
}
