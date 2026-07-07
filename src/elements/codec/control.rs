//! Runtime control handles for codec elements.
//!
//! Once a pipeline is started, its elements are moved into their executor
//! tasks and can no longer be reached through
//! [`Pipeline::get_element_mut`](crate::pipeline::Pipeline::get_element_mut).
//! Control handles bridge that gap: they are cloned from an element *before*
//! `executor.start()` and remain valid while the pipeline runs, exactly like
//! [`AppSinkHandle`](crate::elements::app::AppSinkHandle).
//!
//! # Example
//!
//! ```rust,ignore
//! let encoder = H264Encoder::new(config)?;
//! let keyframes = encoder.keyframe_handle();
//! pipeline.add_filter("enc", encoder);
//!
//! let handle = executor.start(&mut pipeline)?;
//! // ... later, when a new viewer subscribes:
//! keyframes.request(); // next encoded frame is an IDR
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Buffer-metadata key requesting a keyframe from a downstream encoder.
///
/// Elements that inject buffers (e.g. via
/// [`AppSrcHandle`](crate::elements::app::AppSrcHandle)) can set this key to
/// `true` on a buffer's metadata; encoder elements honor it by forcing the
/// frame carrying it (or the next one) to be encoded as a keyframe.
pub const KEYFRAME_REQUEST: &str = "video/keyframe_request";

/// Cloneable handle to request a keyframe (IDR) from a running encoder.
///
/// A request is *sticky*: it stays pending until the encoder processes its
/// next frame, then it is consumed. Multiple requests arriving before that
/// frame coalesce into a single keyframe — the correct semantics for
/// "N subscribers joined, give them a decodable picture".
///
/// The handle is lock-free and allocation-free, so it is safe to check from
/// real-time encoder paths and to call [`request`](Self::request) from any
/// thread or async task.
#[derive(Clone, Debug, Default)]
pub struct KeyframeHandle(Arc<AtomicBool>);

impl KeyframeHandle {
    /// Create a new handle with no pending request.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request that the next encoded frame be a keyframe.
    pub fn request(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether a request is pending (not yet consumed by the encoder).
    pub fn is_pending(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Consume a pending request, returning whether one was pending.
    ///
    /// Called by encoder elements at the top of their processing path.
    /// Custom encoder elements outside this crate may use it the same way.
    pub fn take(&self) -> bool {
        self.0.swap(false, Ordering::AcqRel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_is_sticky_and_consumed_by_take() {
        let handle = KeyframeHandle::new();
        assert!(!handle.is_pending());
        assert!(!handle.take());

        handle.request();
        assert!(handle.is_pending());
        assert!(handle.take(), "take must observe the request");
        assert!(!handle.is_pending(), "take must consume the request");
        assert!(!handle.take());
    }

    #[test]
    fn multiple_requests_coalesce() {
        let handle = KeyframeHandle::new();
        handle.request();
        handle.request();
        handle.request();
        assert!(handle.take());
        assert!(!handle.take(), "requests coalesce into one keyframe");
    }

    #[test]
    fn clones_share_state() {
        let a = KeyframeHandle::new();
        let b = a.clone();
        b.request();
        assert!(a.take());
        assert!(!b.is_pending());
    }
}
