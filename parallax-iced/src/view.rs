//! The widget: a [`shader::Program`] that draws whatever frame is due.

use std::sync::Arc;
use std::time::Duration;

use iced::widget::shader;
use iced::{Event, Rectangle, mouse, window};

use parallax::video::{FrameMailbox, PacedFrame};

use crate::render::VideoPrimitive;

/// Default lateness past which a frame is not worth showing.
///
/// One frame at 25 fps, the same budget `AutoVideoSink` uses.
const DEFAULT_MAX_LATENESS: Duration = Duration::from_millis(40);

/// What the widget carries between redraws.
///
/// Iced builds the `VideoView` fresh on every `view()`, so anything that must
/// survive a rebuild lives here.
#[derive(Debug, Default)]
pub struct State {
    /// The frame on screen. Held so a redraw that has no new frame — a resize,
    /// a UI change, a compositor repaint — draws the picture again rather than
    /// black.
    current: Option<PacedFrame>,
}

impl State {
    /// The frame currently on screen, if any.
    pub fn current(&self) -> Option<&PacedFrame> {
        self.current.as_ref()
    }
}

/// Draws the video a [`FrameMailbox`] is being filled with.
///
/// Put it in a `shader` widget:
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

    /// Take everything that is due, keeping the last of them.
    ///
    /// Draining rather than taking one is deliberate: if the host fell behind,
    /// the frames between the last drawn one and now are already superseded,
    /// and showing them in sequence at redraw speed would play the video fast.
    fn advance(&self, state: &mut State) -> bool {
        let now = self
            .mailbox
            .clock()
            .map(|c| c.now())
            .unwrap_or(parallax::clock::ClockTime::NONE);

        let mut advanced = false;
        while let Some(paced) = self.mailbox.take_due(now, self.max_lateness) {
            state.current = Some(paced);
            advanced = true;
        }
        advanced
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
        // A redraw is the tick: it is the only event that arrives at the rate
        // the screen actually refreshes.
        let Event::Window(window::Event::RedrawRequested(_)) = event else {
            return None;
        };

        self.advance(state);

        // Deliberately not `Action::publish`: Iced rebuilds the whole
        // interface when a redraw produces a message — up to three times,
        // and then it warns. Ask for the next frame and nothing else.
        Some(shader::Action::request_redraw())
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
