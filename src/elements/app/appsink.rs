//! AppSink element for extracting data to application code.
//!
//! Allows applications to pull buffers from a pipeline programmatically.

use crate::buffer::Buffer;
use crate::element::{AsyncSink, ConsumeContext};
use crate::error::Result;
use crate::pipeline::{EndReason, StreamError};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

/// The outcome of a pull. Every way one can end, named.
///
/// This exists because `Ok(None)` used to mean three different things — timed
/// out, ended cleanly, and died — and a consumer could not tell them apart
/// without polling a side channel.
#[derive(Debug)]
// Intentional, and the same trade the executor's `Message` enum makes: boxing
// the buffer would put an allocation on the pull path, which every delivered
// frame goes through, to shrink the three outcomes nobody returns in a loop.
#[allow(clippy::large_enum_variant)]
pub enum Pulled {
    /// A buffer was dequeued.
    Buffer(Buffer),
    /// Nothing available *yet*; the stream is still live.
    ///
    /// [`try_pull_buffer`](AppSinkHandle::try_pull_buffer) returns this
    /// immediately on an empty queue; the `_timeout` forms return it when the
    /// timeout expired. The blocking and async forms without a timeout never do.
    Empty,
    /// The handle is flushing. Not terminal — turn flushing off and pull again.
    Flushing,
    /// The stream is over *and* the queue is drained. Sticky: every later pull
    /// returns the same reason.
    Ended(EndReason),
}

impl Pulled {
    /// The buffer, if there was one.
    pub fn buffer(self) -> Option<Buffer> {
        match self {
            Pulled::Buffer(b) => Some(b),
            _ => None,
        }
    }

    /// Whether this outcome is terminal.
    pub fn is_ended(&self) -> bool {
        matches!(self, Pulled::Ended(_))
    }
}

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

    /// Try to enqueue without waiting.
    ///
    /// Deliberately **not** async: keeping the lock inside a plain fn is what
    /// guarantees no `MutexGuard` is ever alive across an await — which would
    /// both deadlock and fail `AsyncSink::consume`'s `Send` bound.
    fn try_enqueue(&self, buffer: &Buffer) -> Enqueued {
        let mut state = self.lock();
        // Flushing: the app is discarding this window of the stream. Drop the
        // buffer rather than erroring — the executor treats any Err from a
        // sink as fatal, so returning one here would turn every seek into a
        // teardown (the same reasoning as `AppSrc::produce`).
        if state.flushing {
            state.total_dropped += 1;
            return Enqueued::Accepted;
        }
        if state.queue.len() >= state.max_buffers {
            if state.drop_on_full {
                state.total_dropped += 1;
                return Enqueued::Accepted;
            }
            return Enqueued::Full;
        }
        state.queue.push_back(buffer.clone());
        state.total_received += 1;
        drop(state);
        self.data_available.notify_one();
        self.data_available_async.notify_one();
        Enqueued::Accepted
    }

    /// Enqueue, awaiting space when the queue is full.
    ///
    /// This is the whole point of #168: a full queue parks this *task*, never
    /// the worker thread it happens to be running on.
    async fn enqueue(&self, buffer: &Buffer) {
        loop {
            let mut wait = std::pin::pin!(self.space_available_async.notified());
            // Register BEFORE the check, and eagerly: `notify_one` (a pull
            // freeing space) leaves a permit behind for a later waiter, but
            // `notify_waiters` (set_flushing / clear / FlushStart) wakes only
            // those already registered and leaves nothing. Without `enable()`
            // a flush landing between the check and the first poll would be
            // lost, and this sink would wait forever on a queue nobody is
            // going to drain.
            wait.as_mut().enable();
            match self.try_enqueue(buffer) {
                Enqueued::Accepted => return,
                Enqueued::Full => wait.await,
            }
        }
    }

    /// Enqueue from a plain thread, parking it until there is room.
    fn enqueue_blocking(&self, buffer: &Buffer) {
        let mut state = self.lock();
        loop {
            if state.flushing {
                state.total_dropped += 1;
                return;
            }
            if state.queue.len() < state.max_buffers {
                break;
            }
            if state.drop_on_full {
                state.total_dropped += 1;
                return;
            }
            state = wait_ok(self.space_available.wait(state));
        }
        state.queue.push_back(buffer.clone());
        state.total_received += 1;
        drop(state);
        self.data_available.notify_one();
        self.data_available_async.notify_one();
    }

    /// Drop everything queued and wake every waiter, sync and async.
    ///
    /// Shared by `clear()`, `set_flushing(true)` and the FlushStart event so
    /// the four notifications cannot drift apart again — the FlushStart arm
    /// used to signal only the sync condvar, which an async waiter never sees.
    fn clear_queue(&self, count_as_dropped: bool) {
        let mut state = self.lock();
        let dropped = state.queue.len();
        state.queue.clear();
        if count_as_dropped {
            state.total_dropped += dropped as u64;
        }
        drop(state);
        self.space_available.notify_all();
        self.space_available_async.notify_waiters();
        self.data_available.notify_all();
        self.data_available_async.notify_waiters();
    }
}

/// What a non-blocking enqueue attempt did.
enum Enqueued {
    /// Handled — queued, or deliberately discarded (`drop_on_full`, flushing).
    Accepted,
    /// Full with `drop_on_full` off: the caller must wait for space.
    Full,
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
    /// Why the stream ended, once it has. First writer wins.
    end: Option<EndReason>,
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
                    end: None,
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

    /// Whether the stream has ended, for any reason.
    pub fn is_ended(&self) -> bool {
        self.inner.lock().end.is_some()
    }

    /// Why the stream ended, or `None` while it is still live.
    pub fn end_reason(&self) -> Option<EndReason> {
        self.inner.lock().end.clone()
    }

    /// Get statistics.
    pub fn stats(&self) -> AppSinkStats {
        let state = self.inner.lock();
        AppSinkStats {
            queued_buffers: state.queue.len(),
            total_received: state.total_received,
            total_pulled: state.total_pulled,
            total_dropped: state.total_dropped,
            ended: state.end.is_some(),
        }
    }

    /// Signal a clean end of stream from the pipeline side.
    pub fn send_eos(&self) {
        self.end(EndReason::Eos);
    }

    /// Signal that the pipeline failed.
    ///
    /// Consumers see [`Pulled::Ended(EndReason::Error(..))`](Pulled::Ended)
    /// once the queue drains — buffers already delivered are still valid and
    /// are handed over first.
    pub fn send_error(&self, error: StreamError) {
        self.end(EndReason::Error(error));
    }

    /// Record the terminal reason. The first one wins: an EOS arriving after a
    /// failure must not overwrite why it really stopped.
    fn end(&self, reason: EndReason) {
        let mut state = self.inner.lock();
        if state.end.is_none() {
            state.end = Some(reason);
        }
        drop(state);
        self.inner.data_available.notify_all();
        self.inner.data_available_async.notify_waiters();
    }
}

impl Default for AppSink {
    fn default() -> Self {
        Self::new()
    }
}

impl AppSink {
    /// Queue a buffer from a plain thread, parking it until there is room.
    ///
    /// The element-side twin of [`AppSinkHandle::pull_buffer_blocking`], for
    /// feeders that are not tokio tasks. Never call it from inside a runtime:
    /// it parks the worker, which is exactly what #168 was about.
    pub fn push_blocking(&mut self, buffer: &Buffer) {
        self.inner.enqueue_blocking(buffer);
    }
}

impl AsyncSink for AppSink {
    /// The pull queue holds up to `max_buffers` arena-backed clones the
    /// application hasn't collected yet (#189) — each pins a producer slot.
    fn retained_buffers(&self) -> usize {
        wait_ok(self.inner.state.lock()).max_buffers
    }

    /// Queue the buffer, awaiting space when the queue is full.
    ///
    /// `AsyncSink` rather than `Sink` on purpose (#168). The old sync impl
    /// waited on a condvar, which parks the tokio worker — and because the
    /// push before it wakes the application's puller into that same worker's
    /// non-stealable LIFO slot, the puller was stranded on a thread that
    /// would never run it again: total deadlock, not even interruptible by
    /// `abort()`. Awaiting parks the task instead, so the worker stays free
    /// and the puller runs.
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        self.inner.enqueue(ctx.buffer()).await;
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
        match &event {
            crate::event::Event::Eos => self.send_eos(),
            crate::event::Event::Error(err) => self.send_error(err.clone()),
            // Queued-but-unpulled buffers are stale the moment a flush
            // starts; drop them so the application cannot pull pre-seek data
            // after the flush sequence.
            crate::event::Event::FlushStart => self.inner.clear_queue(true),
            _ => {}
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
    /// Never returns [`Pulled::Empty`]: with no timeout there is nothing to
    /// give up on.
    pub async fn pull_buffer(&self) -> Pulled {
        loop {
            // Register for the wakeup *before* looking at the queue, or a push
            // landing between the check and the await would be missed.
            //
            // `enable()` is what makes that true for both signals: a push uses
            // `notify_one`, which leaves a permit for a waiter that registers
            // later, but `end()` and `clear()` use `notify_waiters`, which
            // wakes only those already registered and leaves nothing behind.
            // With a bare `notified()` — which registers on first poll, not at
            // creation — an EOS landing between the check and the await is
            // lost, and this pull waits forever on a stream that has ended.
            let mut notified = std::pin::pin!(self.inner.data_available_async.notified());
            notified.as_mut().enable();

            if let Some(outcome) = self.take_or_end() {
                return outcome;
            }

            notified.await;
        }
    }

    /// Pull a buffer, giving up after `timeout`.
    ///
    /// [`Pulled::Empty`] means the timeout expired and the stream is still
    /// live; [`Pulled::Ended`] means it is over. Telling those two apart is the
    /// whole point — they used to both be `Ok(None)`.
    pub async fn pull_buffer_timeout(&self, timeout: Duration) -> Pulled {
        match tokio::time::timeout(timeout, self.pull_buffer()).await {
            Ok(outcome) => outcome,
            Err(_) => Pulled::Empty,
        }
    }

    /// Pull a buffer, parking the calling thread until one arrives.
    ///
    /// The blocking twin of [`pull_buffer`](Self::pull_buffer), for consumers
    /// that live on a plain thread. Never call it from inside a tokio runtime —
    /// it parks the worker.
    pub fn pull_buffer_blocking(&self) -> Pulled {
        self.pull_wait(None)
    }

    /// Pull a buffer on a plain thread, giving up after `timeout`.
    pub fn pull_buffer_timeout_blocking(&self, timeout: Duration) -> Pulled {
        self.pull_wait(Some(timeout))
    }

    /// Take the head of the queue, or report a terminal state.
    ///
    /// `None` means "nothing yet, but still live" — the caller should wait.
    /// Note the ordering: queued buffers are handed over *before* the end is
    /// reported, because buffers that arrived before an element died are still
    /// perfectly good.
    fn take_or_end(&self) -> Option<Pulled> {
        let mut state = self.inner.lock();
        if state.flushing {
            return Some(Pulled::Flushing);
        }
        if let Some(buffer) = state.queue.pop_front() {
            state.total_pulled += 1;
            let drained = state.queue.is_empty() && state.end.is_some();
            drop(state);
            self.inner.space_available.notify_one();
            self.inner.space_available_async.notify_one();
            if drained {
                // That was the last one: wake anyone waiting on `ended()`.
                self.inner.data_available.notify_all();
                self.inner.data_available_async.notify_waiters();
            }
            return Some(Pulled::Buffer(buffer));
        }
        state.end.clone().map(Pulled::Ended)
    }

    /// The one blocking wait both `_blocking` forms share.
    fn pull_wait(&self, timeout: Option<Duration>) -> Pulled {
        let deadline = timeout.map(|t| std::time::Instant::now() + t);

        loop {
            {
                let state = self.inner.lock();
                if !state.flushing && state.queue.is_empty() && state.end.is_none() {
                    // Nothing to report yet — wait below, still holding nothing.
                    let remaining = match deadline {
                        Some(d) => match d.checked_duration_since(std::time::Instant::now()) {
                            Some(r) if !r.is_zero() => Some(r),
                            _ => return Pulled::Empty,
                        },
                        None => None,
                    };
                    let state = match remaining {
                        Some(r) => {
                            let (s, result) =
                                wait_ok(self.inner.data_available.wait_timeout(state, r));
                            if result.timed_out()
                                && s.queue.is_empty()
                                && s.end.is_none()
                                && !s.flushing
                            {
                                return Pulled::Empty;
                            }
                            s
                        }
                        None => wait_ok(self.inner.data_available.wait(state)),
                    };
                    drop(state);
                    continue;
                }
            }

            if let Some(outcome) = self.take_or_end() {
                return outcome;
            }
        }
    }

    /// Wait until the stream has ended **and** the queue is drained.
    ///
    /// Resolves exactly when a pull would return [`Pulled::Ended`], which is
    /// what makes this safe to `select!` against [`pull_buffer`](Self::pull_buffer)
    /// without losing buffers:
    ///
    /// ```rust,ignore
    /// loop {
    ///     tokio::select! {
    ///         Pulled::Buffer(b) = sink.pull_buffer() => handle(b),
    ///         reason = sink.ended() => break reason,
    ///         _ = shutdown.cancelled() => break EndReason::Aborted,
    ///     }
    /// }
    /// ```
    pub async fn ended(&self) -> EndReason {
        loop {
            // Registered eagerly for the same reason as `pull_buffer`: `end()`
            // signals with `notify_waiters`, which leaves no permit behind.
            let mut notified = std::pin::pin!(self.inner.data_available_async.notified());
            notified.as_mut().enable();
            {
                let state = self.inner.lock();
                if state.queue.is_empty()
                    && let Some(reason) = state.end.clone()
                {
                    return reason;
                }
            }
            notified.await;
        }
    }

    /// The blocking twin of [`ended`](Self::ended).
    pub fn ended_blocking(&self) -> EndReason {
        let mut state = self.inner.lock();
        loop {
            if state.queue.is_empty()
                && let Some(reason) = state.end.clone()
            {
                return reason;
            }
            state = wait_ok(self.inner.data_available.wait(state));
        }
    }

    /// Why the stream ended, without waiting for the queue to drain.
    ///
    /// `Some` as soon as the reason is known, even with buffers still queued —
    /// use it to react early; use [`ended`](Self::ended) to wait for the drain.
    pub fn end_reason(&self) -> Option<EndReason> {
        self.inner.lock().end.clone()
    }

    /// Whether the terminal reason is known yet.
    pub fn is_ended(&self) -> bool {
        self.inner.lock().end.is_some()
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
            ended: state.end.is_some(),
        }
    }

    /// Take a buffer if one is queued, without waiting.
    ///
    /// Returns [`Pulled::Empty`] when the queue is empty but the stream is
    /// still live, and [`Pulled::Ended`] when it is over — a distinction the
    /// old bare `Option<Buffer>` could not express at all.
    pub fn try_pull_buffer(&self) -> Pulled {
        self.take_or_end().unwrap_or(Pulled::Empty)
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
    ///
    /// Wakes every waiter, sync and async — a clear can be what drains the
    /// last buffer, so `ended()` waiters need the same nudge a final pull
    /// would have given them.
    pub fn clear(&self) {
        self.inner.clear_queue(false);
    }

    /// Get the current queue length.
    pub fn queue_len(&self) -> usize {
        self.inner.lock().queue.len()
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
    /// Whether the stream has ended, for any reason. The reason itself lives
    /// on [`AppSinkHandle::end_reason`].
    pub ended: bool,
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
        assert!(!sink.is_ended());
    }

    #[tokio::test]
    async fn test_appsink_consume_pull() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).await.unwrap();

        let buf1 = create_test_buffer(1);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1).await.unwrap();

        assert_eq!(handle.queue_len(), 2);

        let buf = handle
            .pull_buffer_timeout_blocking(Duration::from_millis(100))
            .buffer()
            .expect("first buffer");
        assert_eq!(buf.metadata().sequence, 0);

        let buf = handle
            .pull_buffer_timeout_blocking(Duration::from_millis(100))
            .buffer()
            .expect("second buffer");
        assert_eq!(buf.metadata().sequence, 1);
    }

    #[tokio::test]
    async fn test_appsink_eos() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).await.unwrap();
        sink.send_eos();

        assert!(sink.is_ended());

        // Buffers queued before the end are still valid and come first.
        assert!(matches!(handle.pull_buffer_blocking(), Pulled::Buffer(_)));

        // Only once drained does the end get reported.
        assert!(matches!(
            handle.pull_buffer_blocking(),
            Pulled::Ended(EndReason::Eos)
        ));
    }

    #[tokio::test]
    async fn test_appsink_try_pull() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        // Empty but live — distinct from ended, which is the whole point.
        assert!(matches!(handle.try_pull_buffer(), Pulled::Empty));

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).await.unwrap();

        assert!(matches!(handle.try_pull_buffer(), Pulled::Buffer(_)));
    }

    #[tokio::test]
    async fn test_appsink_drop_on_full() {
        let mut sink = AppSink::with_max_buffers(2).drop_on_full(true);

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).await.unwrap();

        let buf1 = create_test_buffer(1);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1).await.unwrap();

        let buf2 = create_test_buffer(2);
        let ctx2 = ConsumeContext::new(&buf2);
        sink.consume(&ctx2).await.unwrap(); // Should be dropped

        assert_eq!(sink.queue_len(), 2);
        assert_eq!(sink.stats().total_dropped, 1);
    }

    /// #168: a full sink parks its *task*, not its worker — and resumes as
    /// soon as the application makes room.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_full_appsink_awaits_space_instead_of_blocking() {
        let mut sink = AppSink::with_max_buffers(1);
        let handle = sink.handle();

        let first = create_test_buffer(0);
        sink.consume(&ConsumeContext::new(&first)).await.unwrap();

        // Spawned, not polled locally: if this regresses to a condvar it parks
        // worker A, and the timeout below still runs on worker B — a failure
        // rather than a hung suite.
        let second = create_test_buffer(1);
        let task = tokio::spawn(async move {
            sink.consume(&ConsumeContext::new(&second)).await.unwrap();
            sink
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!task.is_finished(), "a full sink must wait for space");

        assert!(matches!(handle.try_pull_buffer(), Pulled::Buffer(_)));
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("freeing space must wake the waiting consume")
            .unwrap();
        assert_eq!(handle.queue_len(), 1);
    }

    /// The reason the await registers with `enable()`: `set_flushing` signals
    /// with `notify_waiters`, which leaves no permit for a late registrant.
    /// With a bare `notified()` this test hangs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flushing_releases_a_parked_consume() {
        let mut sink = AppSink::with_max_buffers(1);
        let handle = sink.handle();
        let first = create_test_buffer(0);
        sink.consume(&ConsumeContext::new(&first)).await.unwrap();

        let second = create_test_buffer(1);
        let task = tokio::spawn(async move {
            sink.consume(&ConsumeContext::new(&second)).await.unwrap();
            sink
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!task.is_finished());

        handle.set_flushing(true);
        let sink = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("set_flushing must wake a parked consume")
            .unwrap();

        // Flushing discards rather than erroring: an Err here would be fatal
        // to the sink task, turning a seek into a teardown.
        assert_eq!(sink.stats().total_dropped, 1);
    }

    /// `clear()` frees space, so a parked consume completes and its buffer is
    /// queued (unlike the flushing case, which discards).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn clear_releases_a_parked_consume() {
        let mut sink = AppSink::with_max_buffers(1);
        let handle = sink.handle();
        let first = create_test_buffer(0);
        sink.consume(&ConsumeContext::new(&first)).await.unwrap();

        let second = create_test_buffer(1);
        let task = tokio::spawn(async move {
            sink.consume(&ConsumeContext::new(&second)).await.unwrap();
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!task.is_finished());

        handle.clear();
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("clear must wake a parked consume")
            .unwrap();
        assert_eq!(handle.queue_len(), 1);
    }

    #[test]
    fn test_appsink_multithreaded() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let producer = thread::spawn(move || {
            for i in 0..10 {
                let buf = create_test_buffer(i);
                let ctx = ConsumeContext::new(&buf);
                sink.push_blocking(ctx.buffer());
            }
            sink.send_eos();
        });

        let mut received = Vec::new();
        while let Pulled::Buffer(buf) = handle.pull_buffer_blocking() {
            received.push(buf.metadata().sequence);
        }

        producer.join().unwrap();
        assert_eq!(received.len(), 10);
    }

    #[tokio::test]
    async fn test_appsink_clear() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).await.unwrap();

        let buf1 = create_test_buffer(1);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1).await.unwrap();

        handle.clear();

        assert_eq!(handle.queue_len(), 0);
    }

    #[tokio::test]
    async fn test_appsink_stats() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let buf0 = create_test_buffer(0);
        let ctx0 = ConsumeContext::new(&buf0);
        sink.consume(&ctx0).await.unwrap();

        let buf1 = create_test_buffer(1);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1).await.unwrap();

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
        sink.consume(&ctx).await.unwrap();

        let pulled = pull.await.unwrap().buffer().expect("a buffer");
        assert_eq!(pulled.metadata().sequence, 7);
    }

    #[tokio::test]
    async fn pull_buffer_reports_eos_as_a_reason() {
        let sink = AppSink::new();
        let handle = sink.handle();

        let pull = tokio::spawn(async move { handle.pull_buffer().await });
        tokio::task::yield_now().await;

        sink.send_eos();
        assert!(matches!(pull.await.unwrap(), Pulled::Ended(EndReason::Eos)));
    }

    #[tokio::test]
    async fn pull_buffer_timeout_gives_up() {
        let sink = AppSink::new();
        let handle = sink.handle();

        let result = handle.pull_buffer_timeout(Duration::from_millis(20)).await;
        assert!(
            matches!(result, Pulled::Empty),
            "a timeout is not an end of stream: got {result:?}"
        );
    }

    #[tokio::test]
    async fn the_handle_exposes_stats_including_drops() {
        let mut sink = AppSink::with_max_buffers(1).drop_on_full(true);
        let handle = sink.handle();

        for seq in 0..3 {
            let buf = create_test_buffer(seq);
            let ctx = ConsumeContext::new(&buf);
            sink.consume(&ctx).await.unwrap();
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
        assert!(!handle.is_ended());
        assert!(matches!(handle.try_pull_buffer(), Pulled::Empty));
        let _ = handle.stats();
        handle.set_flushing(true);
        handle.set_flushing(false);
        handle.clear();
    }
}
