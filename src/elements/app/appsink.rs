//! AppSink element for extracting data to application code.
//!
//! Allows applications to pull buffers from a pipeline programmatically.

use crate::buffer::Buffer;
use crate::element::{ConsumeContext, Sink};
use crate::error::{Error, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

/// A sink element that allows applications to extract buffers from a pipeline.
///
/// AppSink provides a way for application code to pull data from a pipeline.
/// Buffers are queued internally and can be retrieved via a handle.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::AppSink;
///
/// let app_sink = AppSink::new();
/// let handle = app_sink.handle();
///
/// // Pipeline pushes data to sink...
///
/// // In application code (async-first; from a plain thread use
/// // `pull_buffer_blocking()`):
/// while let Some(buffer) = handle.pull_buffer().await? {
///     // Process buffer
/// }
/// ```
pub struct AppSink {
    name: String,
    inner: Arc<AppSinkInner>,
}

struct AppSinkInner {
    state: Mutex<AppSinkState>,
    data_available: Condvar,
    space_available: Condvar,
    /// Async counterparts of the two condvars.
    ///
    /// The queue is a `Mutex<VecDeque>` + `Condvar`, not a channel, so there is
    /// no `recv_async()` to reach for — an async consumer needs its own wakeup
    /// path. `Notify` is signalled everywhere the condvars are.
    data_available_async: Notify,
    space_available_async: Notify,
}

impl AppSinkInner {
    /// Lock the state, ignoring poison.
    ///
    /// Poison would turn one panic into a cascade on the *application's* side
    /// of the boundary: `consume` waits on a condvar holding this lock, so a
    /// panicking element task poisons it, and every subsequent `pull_buffer`,
    /// `stats` or `queue_len` would then panic too — the opposite of telling
    /// the application what went wrong.
    ///
    /// Ignoring it is sound here because there is no invariant to break.
    /// `AppSinkState` is a `VecDeque` plus counters, and every mutation is a
    /// single push/pop and a counter bump; no observer can see a half-finished
    /// multi-step update, so the poison flag carries no information.
    fn lock(&self) -> std::sync::MutexGuard<'_, AppSinkState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Unwrap a condvar wait, ignoring poison — see [`AppSinkInner::lock`].
///
/// These are the calls that matter: a waiter parked here is *holding* the lock
/// a panicking task poisoned, so honouring poison would make the wakeup itself
/// panic.
fn wait_ok<T>(result: std::sync::LockResult<T>) -> T {
    result.unwrap_or_else(|e| e.into_inner())
}

struct AppSinkState {
    queue: VecDeque<Buffer>,
    max_buffers: usize,
    eos: bool,
    flushing: bool,
    drop_on_full: bool,
    total_received: u64,
    total_pulled: u64,
    total_dropped: u64,
}

/// Handle for pulling data from an AppSink.
///
/// This handle can be cloned and sent to other threads.
#[derive(Clone)]
pub struct AppSinkHandle {
    inner: Arc<AppSinkInner>,
}

impl AppSink {
    /// Create a new AppSink with default settings.
    pub fn new() -> Self {
        Self::with_max_buffers(64)
    }

    /// Create a new AppSink with a specific queue size.
    pub fn with_max_buffers(max_buffers: usize) -> Self {
        Self {
            name: "appsink".to_string(),
            inner: Arc::new(AppSinkInner {
                state: Mutex::new(AppSinkState {
                    queue: VecDeque::with_capacity(max_buffers.min(256)),
                    max_buffers,
                    eos: false,
                    flushing: false,
                    drop_on_full: false,
                    total_received: 0,
                    total_pulled: 0,
                    total_dropped: 0,
                }),
                data_available: Condvar::new(),
                space_available: Condvar::new(),
                data_available_async: Notify::new(),
                space_available_async: Notify::new(),
            }),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set whether to drop buffers when the queue is full.
    ///
    /// If false (default), the sink will block when full.
    pub fn drop_on_full(self, drop: bool) -> Self {
        self.inner.lock().drop_on_full = drop;
        self
    }

    /// Get a handle for pulling data from this sink.
    pub fn handle(&self) -> AppSinkHandle {
        AppSinkHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Get the current queue length.
    pub fn queue_len(&self) -> usize {
        self.inner.lock().queue.len()
    }

    /// Check if end-of-stream has been received.
    pub fn is_eos(&self) -> bool {
        self.inner.lock().eos
    }

    /// Get statistics.
    pub fn stats(&self) -> AppSinkStats {
        let state = self.inner.lock();
        AppSinkStats {
            queued_buffers: state.queue.len(),
            total_received: state.total_received,
            total_pulled: state.total_pulled,
            total_dropped: state.total_dropped,
            eos: state.eos,
        }
    }

    /// Signal end of stream from the pipeline side.
    pub fn send_eos(&self) {
        let mut state = self.inner.lock();
        state.eos = true;
        self.inner.data_available.notify_all();
        self.inner.data_available_async.notify_waiters();
    }
}

impl Default for AppSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for AppSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let mut state = self.inner.lock();

        if state.flushing {
            return Err(Error::Element("appsink is flushing".into()));
        }

        // Handle full queue.
        //
        // NOTE: this condvar wait blocks the executor task's thread — a tokio
        // worker — until the application pulls. That is the *designed*
        // back-pressure path (AppSrc::produce deliberately refuses to do the
        // same and returns WouldBlock instead), but it is worth knowing about:
        // an application that stops pulling stalls a runtime worker. Use
        // `drop_on_full(true)` when the consumer is allowed to fall behind.
        while state.queue.len() >= state.max_buffers && !state.flushing {
            if state.drop_on_full {
                state.total_dropped += 1;
                return Ok(());
            }
            state = wait_ok(self.inner.space_available.wait(state));
        }

        if state.flushing {
            return Err(Error::Element("appsink is flushing".into()));
        }

        // Clone the buffer from the context to store it
        state.queue.push_back(ctx.buffer().clone());
        state.total_received += 1;

        self.inner.data_available.notify_one();
        self.inner.data_available_async.notify_one();
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn handle_downstream_event(
        &mut self,
        event: crate::event::Event,
    ) -> Option<crate::event::Event> {
        // Pipeline EOS (including a source/transform error upstream, which the
        // executor converts to EOS) must reach consumers blocked on
        // `pull_buffer` — flip the handle-visible EOS flag.
        if matches!(event, crate::event::Event::Eos) {
            self.send_eos();
        }
        Some(event)
    }
}

impl AppSinkHandle {
    /// Pull a buffer, awaiting one if the queue is empty.
    ///
    /// Async-first: this is the primary form, because the application boundary
    /// of a tokio-first crate is very often a tokio task — it yields instead of
    /// parking the thread, so a consumer can `select!` over it. From a plain
    /// thread, use [`pull_buffer_blocking`](Self::pull_buffer_blocking).
    ///
    /// Returns `Ok(None)` at EOS with an empty queue.
    pub async fn pull_buffer(&self) -> Result<Option<Buffer>> {
        loop {
            // Register for the wakeup *before* looking at the queue, or a push
            // landing between the check and the await would be missed.
            let notified = self.inner.data_available_async.notified();

            {
                let mut state = self.inner.lock();
                if state.flushing {
                    return Err(Error::Element("appsink is flushing".into()));
                }
                if let Some(buffer) = state.queue.pop_front() {
                    state.total_pulled += 1;
                    self.inner.space_available.notify_one();
                    self.inner.space_available_async.notify_one();
                    return Ok(Some(buffer));
                }
                if state.eos {
                    return Ok(None);
                }
            }

            notified.await;
        }
    }

    /// Pull a buffer, giving up after `timeout`.
    ///
    /// Returns `Ok(None)` on timeout or at EOS.
    pub async fn pull_buffer_timeout(&self, timeout: Duration) -> Result<Option<Buffer>> {
        match tokio::time::timeout(timeout, self.pull_buffer()).await {
            Ok(result) => result,
            Err(_) => Ok(None),
        }
    }

    /// Pull a buffer, parking the calling thread until one arrives.
    ///
    /// The blocking twin of [`pull_buffer`](Self::pull_buffer), for consumers
    /// that live on a plain thread. Never call it from inside a tokio runtime —
    /// it parks the worker.
    ///
    /// Returns `Ok(None)` when EOS is reached and no more buffers are available.
    pub fn pull_buffer_blocking(&self) -> Result<Option<Buffer>> {
        self.pull_wait(None)
    }

    /// Pull a buffer on a plain thread, giving up after `timeout`.
    ///
    /// Returns `Ok(None)` on timeout or EOS.
    pub fn pull_buffer_timeout_blocking(&self, timeout: Duration) -> Result<Option<Buffer>> {
        self.pull_wait(Some(timeout))
    }

    /// The one blocking wait both `_blocking` forms share.
    fn pull_wait(&self, timeout: Option<Duration>) -> Result<Option<Buffer>> {
        let mut state = self.inner.lock();

        // Wait for data
        while state.queue.is_empty() && !state.eos && !state.flushing {
            state = if let Some(t) = timeout {
                let (s, result) = wait_ok(self.inner.data_available.wait_timeout(state, t));
                if result.timed_out() {
                    return Ok(None);
                }
                s
            } else {
                wait_ok(self.inner.data_available.wait(state))
            };
        }

        if state.flushing {
            return Err(Error::Element("appsink is flushing".into()));
        }

        if let Some(buffer) = state.queue.pop_front() {
            state.total_pulled += 1;
            self.inner.space_available.notify_one();
            self.inner.space_available_async.notify_one();
            Ok(Some(buffer))
        } else {
            Ok(None)
        }
    }

    /// Statistics for this sink.
    ///
    /// The element itself is moved into its executor task at `start()`, so
    /// `AppSink::stats()` cannot be called on a running pipeline. This can —
    /// which is what makes `total_dropped`, the number a live consumer actually
    /// cares about, readable at all.
    pub fn stats(&self) -> AppSinkStats {
        let state = self.inner.lock();
        AppSinkStats {
            queued_buffers: state.queue.len(),
            total_received: state.total_received,
            total_pulled: state.total_pulled,
            total_dropped: state.total_dropped,
            eos: state.eos,
        }
    }

    /// Try to pull a buffer without blocking.
    pub fn try_pull_buffer(&self) -> Option<Buffer> {
        let mut state = self.inner.lock();

        if let Some(buffer) = state.queue.pop_front() {
            state.total_pulled += 1;
            self.inner.space_available.notify_one();
            self.inner.space_available_async.notify_one();
            Some(buffer)
        } else {
            None
        }
    }

    /// Set flushing mode.
    pub fn set_flushing(&self, flushing: bool) {
        let mut state = self.inner.lock();
        state.flushing = flushing;
        if flushing {
            self.inner.data_available.notify_all();
            self.inner.space_available.notify_all();
            self.inner.data_available_async.notify_waiters();
            self.inner.space_available_async.notify_waiters();
        }
    }

    /// Clear the queue.
    pub fn clear(&self) {
        let mut state = self.inner.lock();
        state.queue.clear();
        self.inner.space_available.notify_all();
        self.inner.space_available_async.notify_waiters();
    }

    /// Get the current queue length.
    pub fn queue_len(&self) -> usize {
        self.inner.lock().queue.len()
    }

    /// Check if EOS has been reached.
    pub fn is_eos(&self) -> bool {
        self.inner.lock().eos
    }

    /// Check if there are buffers available.
    pub fn has_buffer(&self) -> bool {
        !self.inner.lock().queue.is_empty()
    }
}

/// Statistics about AppSink operation.
#[derive(Debug, Clone, Copy)]
pub struct AppSinkStats {
    /// Number of buffers currently queued.
    pub queued_buffers: usize,
    /// Total buffers received from the pipeline.
    pub total_received: u64,
    /// Total buffers pulled by the application.
    pub total_pulled: u64,
    /// Total buffers dropped (when drop_on_full is enabled).
    pub total_dropped: u64,
    /// Whether EOS has been received.
    pub eos: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;
    use std::thread;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(128, 128).unwrap())
    }

    fn create_test_buffer(seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 100);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn test_appsink_creation() {
        let sink = AppSink::new();
        assert_eq!(sink.queue_len(), 0);
        assert!(!sink.is_eos());
    }

    #[test]
    fn test_appsink_consume_pull() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).unwrap();

        let buf1 = create_test_buffer(1);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1).unwrap();

        assert_eq!(handle.queue_len(), 2);

        let buf = handle
            .pull_buffer_timeout_blocking(Duration::from_millis(100))
            .unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 0);

        let buf = handle
            .pull_buffer_timeout_blocking(Duration::from_millis(100))
            .unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 1);
    }

    #[test]
    fn test_appsink_eos() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).unwrap();
        sink.send_eos();

        assert!(sink.is_eos());

        // Should still get the buffered data
        let buf = handle.pull_buffer_blocking().unwrap();
        assert!(buf.is_some());

        // Now should get None for EOS
        let buf = handle.pull_buffer_blocking().unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn test_appsink_try_pull() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        // No data - should return None immediately
        assert!(handle.try_pull_buffer().is_none());

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).unwrap();

        // Now should get data
        let buf = handle.try_pull_buffer();
        assert!(buf.is_some());
    }

    #[test]
    fn test_appsink_drop_on_full() {
        let mut sink = AppSink::with_max_buffers(2).drop_on_full(true);

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).unwrap();

        let buf1 = create_test_buffer(1);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1).unwrap();

        let buf2 = create_test_buffer(2);
        let ctx2 = ConsumeContext::new(&buf2);
        sink.consume(&ctx2).unwrap(); // Should be dropped

        assert_eq!(sink.queue_len(), 2);
        assert_eq!(sink.stats().total_dropped, 1);
    }

    #[test]
    fn test_appsink_multithreaded() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let producer = thread::spawn(move || {
            for i in 0..10 {
                let buf = create_test_buffer(i);
                let ctx = ConsumeContext::new(&buf);
                sink.consume(&ctx).unwrap();
            }
            sink.send_eos();
        });

        let mut received = Vec::new();
        while let Ok(Some(buf)) = handle.pull_buffer_blocking() {
            received.push(buf.metadata().sequence);
        }

        producer.join().unwrap();
        assert_eq!(received.len(), 10);
    }

    #[test]
    fn test_appsink_clear() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).unwrap();

        let buf1 = create_test_buffer(1);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1).unwrap();

        handle.clear();

        assert_eq!(handle.queue_len(), 0);
    }

    #[test]
    fn test_appsink_stats() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).unwrap();

        let buf1 = create_test_buffer(1);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1).unwrap();

        handle.try_pull_buffer();

        let stats = sink.stats();
        assert_eq!(stats.total_received, 2);
        assert_eq!(stats.total_pulled, 1);
        assert_eq!(stats.queued_buffers, 1);
    }

    #[tokio::test]
    async fn pull_buffer_wakes_when_a_buffer_arrives() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        // Nothing queued yet: the pull must wait, not spin or return None.
        let pull = tokio::spawn(async move { handle.pull_buffer().await });

        tokio::task::yield_now().await;

        let buf = create_test_buffer(7);
        let ctx = ConsumeContext::new(&buf);
        sink.consume(&ctx).unwrap();

        let pulled = pull.await.unwrap().unwrap().expect("a buffer");
        assert_eq!(pulled.metadata().sequence, 7);
    }

    #[tokio::test]
    async fn pull_buffer_returns_none_at_eos() {
        let sink = AppSink::new();
        let handle = sink.handle();

        let pull = tokio::spawn(async move { handle.pull_buffer().await });
        tokio::task::yield_now().await;

        sink.send_eos();
        assert!(pull.await.unwrap().unwrap().is_none());
    }

    #[tokio::test]
    async fn pull_buffer_timeout_gives_up() {
        let sink = AppSink::new();
        let handle = sink.handle();

        let result = handle
            .pull_buffer_timeout(Duration::from_millis(20))
            .await
            .unwrap();
        assert!(
            result.is_none(),
            "no data and no EOS: time out, do not hang"
        );
    }

    #[tokio::test]
    async fn the_handle_exposes_stats_including_drops() {
        let mut sink = AppSink::with_max_buffers(1).drop_on_full(true);
        let handle = sink.handle();

        for seq in 0..3 {
            let buf = create_test_buffer(seq);
            let ctx = ConsumeContext::new(&buf);
            sink.consume(&ctx).unwrap();
        }

        // AppSink::stats() is unreachable on a running pipeline (the element is
        // moved into its task); the handle's is not.
        let stats = handle.stats();
        assert_eq!(stats.total_received, 1);
        assert_eq!(stats.total_dropped, 2);
    }

    #[test]
    fn a_panic_while_holding_the_lock_does_not_cascade() {
        // `consume` waits on a condvar *holding* this mutex, so a panicking
        // element task poisons it. If that poison were honoured, every call on
        // the application's side — the side trying to find out what went wrong
        // — would panic instead of answering.
        let sink = AppSink::new();
        let handle = sink.handle();

        let inner = Arc::clone(&sink.inner);
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = inner.lock();
            panic!("element task died mid-update");
        }));
        assert!(poisoned.is_err(), "the test's own panic never happened");
        assert!(
            inner.state.lock().is_err(),
            "the mutex should really be poisoned, or this proves nothing"
        );

        // Every one of these used to panic.
        assert_eq!(handle.queue_len(), 0);
        assert!(!handle.is_eos());
        assert!(handle.try_pull_buffer().is_none());
        let _ = handle.stats();
        handle.set_flushing(true);
        handle.set_flushing(false);
        handle.clear();
    }
}
