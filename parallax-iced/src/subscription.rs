//! Waking the interface when a frame lands on an idle mailbox.
//!
//! In steady state this does nothing. The widget asks Iced to redraw at the
//! next frame's presentation time and reads the mailbox itself, so there is
//! no message traffic per frame at all — no channel, no `view()` rebuild used
//! as a doorbell.
//!
//! What it covers is the idle case: startup, a resume, the far side of a
//! seek, a stream that went quiet. There the widget has nothing queued, so it
//! parks, and something has to tell the runtime a frame arrived.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use iced::Subscription;
use iced::futures::Stream;
use iced::futures::stream;

use parallax::video::FrameMailbox;

/// Identifies one mailbox to Iced's subscription bookkeeping.
///
/// Hashing **only the id** is the point. Iced keys a running subscription by
/// its recipe hash, so rebuilding the pipeline — which makes a new mailbox
/// with a new id — tears the old stream down and starts a new one. Frames
/// from a previous file cannot reach the new interface, because the thing
/// that would deliver them no longer exists.
#[derive(Clone)]
pub struct FrameSignal {
    id: u64,
    mailbox: Arc<FrameMailbox>,
}

impl std::fmt::Debug for FrameSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameSignal").field("id", &self.id).finish()
    }
}

impl FrameSignal {
    /// Watch this mailbox.
    pub fn new(mailbox: Arc<FrameMailbox>) -> Self {
        Self {
            id: mailbox.id(),
            mailbox,
        }
    }
}

impl Hash for FrameSignal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// A [`Subscription`] that yields `()` whenever the mailbox goes from empty to
/// occupied, and once more when the stream ends.
///
/// Map it to whatever message wakes your interface:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use parallax::video::FrameMailbox;
/// # #[derive(Debug, Clone)] enum Message { Frame }
/// # fn subscription(mailbox: &Arc<FrameMailbox>) -> iced::Subscription<Message> {
/// parallax_iced::frames(Arc::clone(mailbox)).map(|_| Message::Frame)
/// # }
/// ```
pub fn frames(mailbox: Arc<FrameMailbox>) -> Subscription<()> {
    // A non-capturing `fn` pointer, as `run_with` requires.
    Subscription::run_with(FrameSignal::new(mailbox), |signal| {
        doorbell(Arc::clone(&signal.mailbox))
    })
}

fn doorbell(mailbox: Arc<FrameMailbox>) -> impl Stream<Item = ()> {
    stream::unfold(mailbox, |mailbox| async move {
        // Ended and drained: nothing more will arrive, so end the stream
        // rather than leave a task parked on a doorbell nobody will ring.
        if mailbox.end_reason().is_some() && !mailbox.has_frame() {
            return None;
        }
        // The sink rings this only on the empty-to-occupied edge and at
        // end-of-stream, which is exactly the set of moments a parked
        // interface needs to hear about.
        mailbox.doorbell().notified().await;
        Some(((), mailbox))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(signal: &FrameSignal) -> u64 {
        let mut h = DefaultHasher::new();
        signal.hash(&mut h);
        h.finish()
    }

    #[test]
    fn a_rebuilt_pipeline_is_a_different_subscription() {
        let a = FrameSignal::new(Arc::new(FrameMailbox::new(3)));
        let b = FrameSignal::new(Arc::new(FrameMailbox::new(3)));
        assert_ne!(
            hash_of(&a),
            hash_of(&b),
            "a new mailbox must not reuse the old stream"
        );
    }

    #[test]
    fn the_same_mailbox_is_the_same_subscription() {
        let mailbox = Arc::new(FrameMailbox::new(3));
        let a = FrameSignal::new(Arc::clone(&mailbox));
        let b = FrameSignal::new(mailbox);
        assert_eq!(
            hash_of(&a),
            hash_of(&b),
            "rebuilding view() must not restart the stream"
        );
    }
}
