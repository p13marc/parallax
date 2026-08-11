//! Unified pipeline executor with automatic execution strategy.
//!
//! A single executor that handles all execution modes:
//! - **Auto** (default): Automatically determines optimal strategy per element
//! - Async: All elements run as Tokio tasks
//! - Hybrid: Mix of async tasks and RT threads
//!
//! # Automatic Mode
//!
//! In automatic mode (default), the executor analyzes each element's
//! [`ExecutionHints`] to determine the best execution strategy:
//!
//! - **Low-latency elements** (audio processing) → RT threads
//! - **I/O-bound elements** (network, file) → Async tasks
//! - **CPU-bound elements** → Dedicated threads or RT threads
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::{Pipeline, Executor};
//!
//! let mut pipeline = Pipeline::parse("filesrc ! h264dec ! display")?;
//!
//! // Automatic mode (default) - the executor will:
//! // - Run filesrc as async (I/O-bound)
//! // - Run h264dec based on its hints
//! // - Run display as async or RT based on latency needs
//! pipeline.run().await?;
//! ```

use crate::buffer::Buffer;
use crate::clock::{Clock, ClockTime};
use crate::element::{
    AsyncElementDyn, DemuxResult, DynAsyncElement, ElementType, ExecutionHints, LatencyHint,
    Output, ProcessingHint, SourceResult,
};
use crate::error::{Error, Result};
use crate::event::{Event, EventResult, FlushStopEvent, SeekEvent, SeekType, SegmentEvent};
use crate::memory::{OutputBudget, defaults};
use crate::pipeline::bus::{Bus, BusHandle};
use crate::pipeline::probe::{ProbeRegistry, ProbeReturn};
use crate::pipeline::rt_bridge::AsyncRtBridge;
use crate::pipeline::rt_scheduler::{
    BoundaryDirection, GraphPartition, RtConfig, RtScheduler, SchedulingMode,
};
use crate::pipeline::tracer::TracerRegistry;
use crate::pipeline::{
    DriverConfig, EndReason, EventReceiver, EventSender, LinkPolicy, NodeId, Pipeline,
    PipelineEvent, PipelineState, StreamError, TimerDriver,
};
use futures::FutureExt;
use kanal::{AsyncReceiver, AsyncSender, bounded_async};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::task::{Context, Poll};

use tokio::sync::watch;
use tokio::task::JoinHandle;

// ============================================================================
// Configuration
// ============================================================================

/// Unified executor configuration.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Scheduling mode (Auto, Async, Hybrid, RealTime).
    /// Auto (default) analyzes element hints to determine the best strategy.
    pub scheduling: SchedulingMode,

    /// Channel buffer size between elements.
    pub channel_capacity: usize,

    /// RT scheduling configuration (for Hybrid/RealTime modes).
    pub rt: RtConfig,

    /// Driver configuration (for timed execution).
    pub driver: Option<DriverConfig>,

    /// Enable automatic strategy detection from element hints.
    /// When true (default), the executor analyzes ExecutionHints to determine
    /// optimal scheduling per element.
    pub auto_strategy: bool,

    /// Fail the pipeline after this many buffers shed back-to-back.
    ///
    /// An element whose output arena is exhausted sheds the buffer and carries
    /// on (see [`Error::PoolExhausted`]), which is what a live capture wants:
    /// a dropped frame beats a dead pipeline. `None`, the default, means it
    /// never gives up.
    ///
    /// Set it for work where silent degradation is the worse outcome — a batch
    /// transcode producing a file with gaps in it should stop and say so rather
    /// than finish and look successful.
    pub shed_fatal_after: Option<u64>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            scheduling: SchedulingMode::Async,
            channel_capacity: 16,
            rt: RtConfig::default(),
            driver: None,
            auto_strategy: true, // Enable automatic by default
            shed_fatal_after: None,
        }
    }
}

impl ExecutorConfig {
    /// Create config for automatic strategy detection (default).
    ///
    /// The executor will analyze each element's `ExecutionHints` to determine:
    /// - Which elements should run in RT threads (low-latency)
    /// - Which elements are I/O-bound (async tasks)
    pub fn auto() -> Self {
        Self::default()
    }

    /// Create config for pure async execution (no automatic detection).
    pub fn async_only() -> Self {
        Self {
            auto_strategy: false,
            ..Default::default()
        }
    }

    /// Create config for hybrid async + RT execution.
    pub fn hybrid() -> Self {
        Self {
            scheduling: SchedulingMode::Hybrid,
            rt: RtConfig::hybrid(),
            auto_strategy: false,
            ..Default::default()
        }
    }

    /// Create config for low-latency audio.
    pub fn low_latency_audio() -> Self {
        Self {
            scheduling: SchedulingMode::Hybrid,
            rt: RtConfig::low_latency_audio(),
            driver: Some(DriverConfig::low_latency_audio()),
            auto_strategy: false,
            ..Default::default()
        }
    }

    /// Create config for video processing.
    pub fn video(fps: u32) -> Self {
        Self {
            scheduling: SchedulingMode::Hybrid,
            rt: RtConfig::hybrid(),
            driver: Some(DriverConfig::video(fps)),
            auto_strategy: false,
            ..Default::default()
        }
    }

    /// Set scheduling mode.
    pub fn with_scheduling(mut self, mode: SchedulingMode) -> Self {
        self.scheduling = mode;
        self
    }

    /// Set channel capacity.
    pub fn with_channel_capacity(mut self, capacity: usize) -> Self {
        self.channel_capacity = capacity;
        self
    }

    /// Fail the pipeline after `limit` buffers shed back-to-back.
    ///
    /// See [`shed_fatal_after`](Self::shed_fatal_after). The default is to shed
    /// indefinitely, which is right for live media and wrong for batch work.
    pub fn with_shed_fatal_after(mut self, limit: u64) -> Self {
        self.shed_fatal_after = Some(limit);
        self
    }

    /// Set RT priority (requires CAP_SYS_NICE).
    pub fn with_rt_priority(mut self, priority: i32) -> Self {
        self.rt.rt_priority = Some(priority);
        self
    }

    /// Set quantum (samples per cycle).
    pub fn with_quantum(mut self, quantum: u32) -> Self {
        self.rt.quantum = quantum;
        self
    }

    /// Set driver configuration.
    pub fn with_driver(mut self, driver: DriverConfig) -> Self {
        self.driver = Some(driver);
        self
    }

    /// Disable automatic strategy detection.
    pub fn without_auto_strategy(mut self) -> Self {
        self.auto_strategy = false;
        self
    }
}

// ============================================================================
// Execution Strategy (per-element)
// ============================================================================

/// Execution strategy for a single element, determined by analyzing hints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElementStrategy {
    /// Run in Tokio async runtime.
    Async,
    /// Run in dedicated RT thread.
    RealTime,
}

/// Analyzed execution plan for the entire pipeline.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Strategy per node.
    pub strategies: HashMap<NodeId, ElementStrategy>,
    /// Nodes that need RT scheduling.
    pub rt_nodes: HashSet<NodeId>,
    /// Nodes that can run async.
    pub async_nodes: HashSet<NodeId>,
    /// Whether any RT scheduling is needed.
    pub needs_rt: bool,
}

impl ExecutionPlan {
    /// Create an empty plan (all async).
    pub fn all_async() -> Self {
        Self {
            strategies: HashMap::new(),
            rt_nodes: HashSet::new(),
            async_nodes: HashSet::new(),
            needs_rt: false,
        }
    }
}

/// Analyze a pipeline and determine the optimal execution strategy for each element.
fn analyze_pipeline(pipeline: &Pipeline) -> ExecutionPlan {
    let mut plan = ExecutionPlan::all_async();

    for node_id in pipeline.node_ids() {
        if let Some(node) = pipeline.get_node(node_id) {
            // Get hints from node (which delegates to element if present)
            let hints = node.execution_hints();
            let strategy = determine_element_strategy(&hints);

            match strategy {
                ElementStrategy::RealTime => {
                    plan.rt_nodes.insert(node_id);
                    plan.needs_rt = true;
                }
                ElementStrategy::Async => {
                    plan.async_nodes.insert(node_id);
                }
            }

            plan.strategies.insert(node_id, strategy);
        }
    }

    tracing::debug!(
        "Execution plan: {} async, {} RT",
        plan.async_nodes.len(),
        plan.rt_nodes.len(),
    );

    plan
}

/// Determine the execution strategy for a single element based on its hints.
///
/// Elements declare facts about their capabilities (rt_safe, processing type,
/// latency, etc.) and the executor derives the optimal strategy.
///
/// Priority order: RT (rt_safe + low latency) > I/O-bound > default async.
fn determine_element_strategy(hints: &ExecutionHints) -> ElementStrategy {
    // Rule 1: RT-safe + low latency → RT thread
    if hints.rt_safe && matches!(hints.latency, LatencyHint::UltraLow | LatencyHint::Low) {
        return ElementStrategy::RealTime;
    }

    // Rule 2: I/O-bound → async
    if hints.processing == ProcessingHint::IoBound {
        return ElementStrategy::Async;
    }

    // Default: async is safest
    ElementStrategy::Async
}

// ============================================================================
// Terminal outcome
// ============================================================================

/// How the run ended, decided once and remembered.
///
/// The outcome used to be reachable only through [`PipelineHandle::wait`], which
/// consumes the handle — so the caller who keeps the handle to control the
/// pipeline could never learn why it stopped. The event channel could not fill
/// the gap either: it is a `broadcast`, so a terminal event sent before you
/// subscribed is gone.
///
/// Hence [`watch`]: it *retains* its value, so an observer that arrives after
/// the pipeline is long finished still sees the answer.
///
/// Two rules hold everything together:
///
/// - **Sticky.** The first writer wins, so an error can never be papered over by
///   a sibling's later clean EOS. [`watch::Sender::send_if_modified`] runs the
///   closure under the write lock, which makes that an atomic compare-and-set,
///   and its `bool` result is the "did I win" signal that gates the bus post.
/// - **Counted.** `live` is one per spawned task plus a seed, and the *last*
///   task out declares EOS. Without the seed, a source that finishes before its
///   siblings are even spawned would declare EOS for the whole graph.
struct TerminalOutcome {
    tx: watch::Sender<Option<EndReason>>,
    live: AtomicUsize,
    bus: BusHandle,
}

impl TerminalOutcome {
    fn new(bus: BusHandle) -> Arc<Self> {
        let (tx, _) = watch::channel(None);
        Arc::new(Self {
            tx,
            live: AtomicUsize::new(0),
            bus,
        })
    }

    /// Record the outcome if nothing has been recorded yet.
    ///
    /// Returns whether this call is the one that decided it.
    fn record(&self, reason: EndReason) -> bool {
        self.tx.send_if_modified(|slot| {
            if slot.is_some() {
                return false;
            }
            *slot = Some(reason);
            true
        })
    }

    /// The pipeline failed. Reports it on the bus, attributed to the element.
    fn fail_with(&self, err: StreamError) {
        // The bus message is gated on winning, so a run posts exactly one
        // terminal message: a graph that loses three elements to the same cause
        // is one failure, not three, and `Bus::wait_for_eos_or_error` would
        // report whichever raced to the front anyway.
        if self.record(EndReason::Error(err.clone())) {
            match err.node() {
                Some(node) => self.bus.for_element(node).post_error(err.message(), None),
                None => self.bus.post_error(err.message(), None),
            }
        }
    }

    fn fail(&self, node: &str, err: &Error) {
        self.fail_with(StreamError::new(node, err.to_string()));
    }

    /// Every task is done and none of them failed.
    fn finish(&self) {
        if self.record(EndReason::Eos) {
            self.bus.post_eos();
        }
    }

    /// The pipeline was torn down. Deliberately silent on the bus: there is no
    /// `MessageKind::Aborted`, and an `Eos` here would claim the stream ran out
    /// when the caller cut it off.
    fn abort(&self) {
        self.record(EndReason::Aborted);
    }

    fn peek(&self) -> Option<EndReason> {
        self.tx.borrow().clone()
    }
}

/// One task's share of the live count, released on drop.
///
/// Drop, and not an explicit call at the end of each task, because the error
/// arms `return Err` early and an aborted task never reaches any arm at all.
/// Drop is the only exit every path goes through.
struct TaskGuard {
    outcome: Arc<TerminalOutcome>,
    node: Arc<str>,
}

impl TaskGuard {
    /// Claim a share. Called on the spawning thread, *before* `tokio::spawn`:
    /// spawn returns before the future is first polled, so a guard built inside
    /// the task body would leave the count at the seed and let `start()` declare
    /// EOS before a single element had run.
    fn new(outcome: &Arc<TerminalOutcome>, node: &str) -> Self {
        outcome.live.fetch_add(1, Ordering::Relaxed);
        Self {
            outcome: outcome.clone(),
            node: node.into(),
        }
    }

    fn fail(&self, err: &Error) {
        self.outcome.fail(&self.node, err);
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        // A panic that escaped `guard()` — from a probe, a tracer, or the
        // plumbing between element calls — unwinds past every error arm, so
        // without this the task would silently count as a clean finish while
        // `wait()` reported `Error::Panic`. Two observers, two answers.
        //
        // False positives are structurally excluded: unwinding has already
        // stopped by the time `guard()`'s `catch_unwind` returns, and task
        // cancellation is not a panic.
        if std::thread::panicking() {
            self.outcome.fail_with(StreamError::new(
                &*self.node,
                "task panicked outside the element call",
            ));
        }

        if self.outcome.live.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.outcome.finish();
        }
    }
}

/// Run a task body, recording its failure before the guard releases the count.
///
/// Wrapping the whole body rather than editing each error arm is deliberate:
/// the arms disagree about what they report — the two `shed_fatal_after` paths
/// return `Err` without telling anyone at all — and a new arm added later would
/// have to remember. Here the ordering that makes "error beats EOS" work
/// (record, *then* release the count) is a property of the code shape.
async fn reporting(share: TaskGuard, body: impl Future<Output = Result<()>>) -> Result<()> {
    let result = body.await;
    if let Err(e) = &result {
        share.fail(e);
    }
    result
}

/// The pipeline's terminal outcome, awaitable.
///
/// Owned rather than borrowed, so it can be created before
/// [`PipelineHandle::wait`] consumes the handle and awaited afterwards, or
/// raced against `wait()` in a `select!`:
///
/// ```rust,ignore
/// let ended = handle.ended();
/// tokio::select! {
///     _ = tokio::signal::ctrl_c() => handle.abort(),
///     reason = ended => println!("pipeline ended: {reason:?}"),
/// }
/// ```
#[must_use = "Ended is a future — await it to learn how the pipeline finished"]
pub struct Ended {
    inner: Pin<Box<dyn Future<Output = EndReason> + Send>>,
}

impl Future for Ended {
    type Output = EndReason;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<EndReason> {
        self.inner.as_mut().poll(cx)
    }
}

impl std::fmt::Debug for Ended {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ended").finish_non_exhaustive()
    }
}

/// Cooperative stop, detached from the handle.
///
/// [`PipelineHandle::wait`] takes the handle by value, so a caller that awaits
/// it can no longer reach [`PipelineHandle::stop`]. Take one of these first and
/// the two are independent.
#[derive(Clone, Debug)]
pub struct Stopper {
    stop: Arc<AtomicBool>,
}

impl Stopper {
    /// Ask every source to stop producing. See [`PipelineHandle::stop`].
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
}

// ============================================================================
// Handle
// ============================================================================

/// Control-channel sender for one running source task.
///
/// Upstream events cannot ride the data channels (those point downstream), so
/// every async-spawned source gets a small out-of-band channel, drained at the
/// top of its produce loop. Sources scheduled on RT threads have none — the RT
/// path carries no events at all yet.
struct SourceControl {
    name: String,
    tx: AsyncSender<Event>,
}

/// Runtime-control state assembled while tasks spawn, then moved onto the
/// [`PipelineHandle`]: the source control channels, the pause gate the source
/// loops watch, and the last-presented-PTS cell the sink loops write.
struct RuntimeControls {
    controls: Vec<SourceControl>,
    pause_rx: watch::Receiver<bool>,
    position: Arc<AtomicU64>,
}

/// Handle to a running pipeline.
pub struct PipelineHandle {
    /// Tokio task handles.
    tasks: Vec<JoinHandle<Result<()>>>,
    /// RT thread handles (if any).
    rt_handles: Vec<crate::pipeline::rt_scheduler::DataThreadHandle>,
    /// Event sender.
    events: EventSender,
    /// Bridges (kept alive).
    #[allow(dead_code)]
    bridges: Vec<Arc<AsyncRtBridge>>,
    /// Driver handle (if any).
    #[allow(dead_code)]
    driver: Option<crate::pipeline::TimerDriverHandle>,
    /// RT driver task (periodic trigger for RT data thread).
    /// Stored separately because it runs indefinitely and must be
    /// aborted when the pipeline finishes.
    rt_driver_task: Option<JoinHandle<Result<()>>>,
    /// Pipeline message bus (moved from Pipeline on start).
    bus: Option<Bus>,
    /// Bus handle for posting pipeline-level messages.
    bus_handle: Option<BusHandle>,
    /// Cooperative stop flag, checked by source tasks between `produce()`
    /// calls (see [`PipelineHandle::stop`]).
    stop: Arc<AtomicBool>,
    /// How the run ended, once it has (see [`PipelineHandle::ended`]).
    outcome: Arc<TerminalOutcome>,
    /// The seed share of the live count, retained only when RT threads are
    /// running. They loop until told to stop, so they cannot be counted like
    /// tasks — instead the seed holds EOS back until `wait`/`abort` joins them.
    seed: Option<TaskGuard>,
    /// Control-channel senders into the running source tasks (see
    /// [`send_event_upstream`](Self::send_event_upstream)).
    controls: Vec<SourceControl>,
    /// Pause gate watched by the source loops (see [`pause`](Self::pause)).
    pause_tx: watch::Sender<bool>,
    /// The shared clock wrapper, when the pipeline has a started clock.
    pausable: Option<Arc<crate::clock::PausableClock>>,
    /// Base time distributed to the elements, for the running-time fallback in
    /// [`position`](Self::position). `NONE` when the pipeline has no clock.
    base_time: ClockTime,
    /// Last-presented PTS in nanoseconds, written by the sink tasks;
    /// `u64::MAX` until a sink has presented (or after a flush reset it).
    position: Arc<AtomicU64>,
}

impl PipelineHandle {
    /// Wait for the pipeline to complete.
    pub async fn wait(mut self) -> Result<()> {
        let mut first_error = None;

        // Wait for all Tokio tasks
        for task in self.tasks {
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    self.events.send_error(e.to_string(), None);
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
                Err(e) => {
                    // Belt and braces. Tasks catch their own panics and turn
                    // them into Error::Panic with the element's name attached,
                    // so reaching here means the task was cancelled or the
                    // catch itself failed — no node name is available.
                    let err = if e.is_panic() {
                        Error::Panic {
                            node: "<unknown>".into(),
                            message: crate::error::panic_message(e.into_panic().as_ref()),
                        }
                    } else {
                        Error::Pipeline(format!("element task did not finish: {e}"))
                    };
                    self.events.send_error(err.to_string(), None);
                    // The task never ran its own epilogue, so nothing has told
                    // the outcome. (The `Ok(Err(e))` arm above needs no such
                    // call — that task reported before it returned.)
                    self.outcome.fail("<unknown>", &err);
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }

        // Abort the RT driver task (it runs indefinitely)
        if let Some(task) = self.rt_driver_task.take() {
            task.abort();
            let _ = task.await; // Ignore JoinError from abort
        }

        // Signal RT threads to stop and join (via spawn_blocking to avoid
        // blocking the async executor)
        for handle in self.rt_handles.drain(..) {
            handle.signal_stop();
            let join_result = tokio::task::spawn_blocking(move || handle.join()).await;
            match join_result {
                Ok(Err(e)) => {
                    // An RT thread is not a task and has no guard, so this is
                    // the only place its failure can reach the outcome.
                    self.outcome.fail("<rt-thread>", &e);
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
                Err(e) => {
                    let err = if e.is_panic() {
                        Error::Panic {
                            node: "<rt-join>".into(),
                            message: crate::error::panic_message(e.into_panic().as_ref()),
                        }
                    } else {
                        Error::Pipeline(format!("RT thread join did not finish: {e}"))
                    };
                    self.outcome.fail("<rt-join>", &err);
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
                Ok(Ok(())) => {}
            }
        }

        // Every RT thread is joined, so the seed can go — this is what lets a
        // hybrid pipeline reach EOS at all. Async-only runs released it back in
        // `start()` and this is a no-op.
        drop(self.seed.take());

        if first_error.is_none() {
            self.events.send_eos();
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// How the pipeline ended, awaitable — and, unlike [`wait`](Self::wait),
    /// without consuming the handle.
    ///
    /// Resolves once for the run and then keeps resolving: the outcome is
    /// retained, so an observer that arrives after the pipeline finished still
    /// gets the answer instead of waiting forever. That is the difference from
    /// [`subscribe`](Self::subscribe), whose broadcast channel drops anything
    /// sent before you subscribed.
    ///
    /// The returned [`Ended`] is owned, so the usual shape works:
    ///
    /// ```rust,ignore
    /// let ended = handle.ended();
    /// handle.wait().await?;
    /// assert_eq!(ended.await, EndReason::Eos);
    /// ```
    ///
    /// In hybrid mode the answer waits for [`wait`](Self::wait) or
    /// [`abort`](Self::abort) to join the RT threads, which never end on their
    /// own.
    pub fn ended(&self) -> Ended {
        // Moving the Arc into the future keeps the sender alive for as long as
        // anyone is waiting, so `wait_for` cannot fail on a dropped handle.
        // Subscribing inside is safe: `wait_for` inspects the current value
        // before it waits, which is exactly the late-observer guarantee.
        let outcome = self.outcome.clone();
        Ended {
            inner: Box::pin(async move {
                let mut rx = outcome.tx.subscribe();
                match rx.wait_for(|slot| slot.is_some()).await {
                    Ok(slot) => slot.clone().unwrap_or(EndReason::Aborted),
                    Err(_) => {
                        debug_assert!(false, "the outcome sender outlives every Ended");
                        EndReason::Aborted
                    }
                }
            }),
        }
    }

    /// How the pipeline ended, or `None` if it is still running.
    ///
    /// The non-blocking twin of [`ended`](Self::ended).
    pub fn end_reason(&self) -> Option<EndReason> {
        self.outcome.peek()
    }

    /// A [`Stopper`] that outlives this handle.
    ///
    /// [`wait`](Self::wait) takes the handle by value; take one of these before
    /// awaiting it if you still need to stop the pipeline.
    pub fn stopper(&self) -> Stopper {
        Stopper {
            stop: self.stop.clone(),
        }
    }

    /// Signal all sources to stop cooperatively.
    ///
    /// Sources end their produce loop at the next iteration and propagate
    /// EOS downstream exactly like a natural end-of-stream, so sinks drain
    /// and [`wait`](Self::wait) returns `Ok(())`. This is the graceful way
    /// to stop a pipeline with live (infinite) sources — [`abort`](Self::abort)
    /// alone cannot end a source loop that never reaches an await point.
    ///
    /// Limitation: a source blocked *inside* a synchronous `produce()` call
    /// (e.g. waiting on a hardware frame with no timeout) only observes the
    /// flag once that call returns.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    /// Abort all pipeline tasks.
    ///
    /// Reported as [`EndReason::Aborted`], distinct from the EOS that
    /// [`stop`](Self::stop) produces: the stream did not run out, it was cut.
    pub fn abort(mut self) {
        // Best effort for live sources: tasks blocked in a synchronous
        // produce() are never re-polled by abort(), so also raise the
        // cooperative stop flag — they exit at the next loop iteration.
        self.stop.store(true, Ordering::Release);
        // Before the aborts, not after. Cancellation is asynchronous: the tasks
        // drop their guards on a worker thread moments later, and whichever got
        // there first would otherwise report a clean EOS for a torn-down run.
        self.outcome.abort();
        if let Some(task) = self.rt_driver_task.take() {
            task.abort();
        }
        for task in self.tasks {
            task.abort();
        }

        for handle in self.rt_handles.drain(..) {
            handle.signal_stop();
            let _ = handle.join();
        }
        drop(self.seed.take());

        self.events.send(PipelineEvent::Stopped);
    }

    /// Send an upstream event to every running source task.
    ///
    /// This is the runtime counterpart of [`Pipeline::send_event_upstream`],
    /// which only works before `start()` moves the elements into their tasks.
    /// The event is handled asynchronously by each source's produce loop:
    /// for a flushing [`Event::Seek`] the source repositions, broadcasts
    /// FlushStart → FlushStop → Segment in-band to its downstream graph, and
    /// posts [`MessageKind::SeekDone`](crate::pipeline::bus::MessageKind::SeekDone) on the bus when it is done.
    ///
    /// Returns `false` if no source accepted the event — the pipeline has no
    /// async-spawned sources (RT-scheduled sources carry no control channel),
    /// or every source task has already finished.
    pub async fn send_event_upstream(&self, event: Event) -> bool {
        let mut delivered = false;
        for control in &self.controls {
            if control.tx.send(event.clone()).await.is_ok() {
                delivered = true;
            } else {
                tracing::debug!(
                    "source '{}' is gone; upstream event '{}' not delivered",
                    control.name,
                    event.name()
                );
            }
        }
        delivered
    }

    /// Seek a running pipeline.
    ///
    /// Delivers the seek to every source; sources that are not seekable (live
    /// capture, AppSrc without an app-side seek story) report it and continue.
    /// Watch the bus for [`MessageKind::SeekDone`](crate::pipeline::bus::MessageKind::SeekDone) to learn when the flush
    /// sequence has run.
    pub async fn seek(&self, seek: SeekEvent) -> bool {
        self.send_event_upstream(Event::Seek(seek)).await
    }

    /// Seek to a time position (flushing, keyframe-snapped). See [`Self::seek`].
    pub async fn seek_time(&self, position: ClockTime) -> bool {
        self.seek(SeekEvent::new_time(position)).await
    }

    /// Seek to a byte offset (flushing). See [`Self::seek`].
    pub async fn seek_bytes(&self, position: u64) -> bool {
        self.seek(SeekEvent::new_bytes(position)).await
    }

    /// Pause the running pipeline. Idempotent.
    ///
    /// Freezes the shared [`PausableClock`](crate::clock::PausableClock) —
    /// sinks pacing presentation against running time stall on the spot — and
    /// gates every source's produce loop, so unclocked pipelines pause too.
    /// In-flight buffers drain into the sinks' pacing waits rather than being
    /// dropped. Posts `StateChanged{Running → Idle}` on the bus.
    ///
    /// Limits: an `AlsaSink` plays out what its device buffer already holds
    /// (a crisp `snd_pcm_pause` control is a follow-up), and a source blocked
    /// *inside* `process_source` observes the gate only when that call
    /// returns — the same caveat as [`stop`](Self::stop).
    pub fn pause(&self) {
        if *self.pause_tx.borrow() {
            return;
        }
        // Clock first, then the gate: from the freeze on, nothing is presented,
        // so the few buffers a not-yet-gated source still produces just queue.
        if let Some(clock) = &self.pausable {
            clock.pause();
        }
        self.pause_tx.send_replace(true);
        self.events
            .send_state_changed(PipelineState::Running, PipelineState::Idle);
        if let Some(bus) = &self.bus_handle {
            bus.post_state_changed(PipelineState::Running, PipelineState::Idle);
        }
    }

    /// Resume a paused pipeline. Idempotent.
    ///
    /// Un-gates the sources and resumes the clock gap-free: running time
    /// continues from where it froze, so sinks pick up on the very next frame
    /// with no burst of late frames. Posts `StateChanged{Idle → Running}`.
    pub fn resume(&self) {
        if !*self.pause_tx.borrow() {
            return;
        }
        self.pause_tx.send_replace(false);
        if let Some(clock) = &self.pausable {
            clock.resume();
        }
        self.events
            .send_state_changed(PipelineState::Idle, PipelineState::Running);
        if let Some(bus) = &self.bus_handle {
            bus.post_state_changed(PipelineState::Idle, PipelineState::Running);
        }
    }

    /// Whether the pipeline is currently paused by [`pause`](Self::pause).
    pub fn is_paused(&self) -> bool {
        *self.pause_tx.borrow()
    }

    /// Current stream position.
    ///
    /// The PTS of the last buffer any sink presented — monotonic between
    /// flushes, frozen while paused (nothing is presented), and re-anchored by
    /// the `Segment` of a runtime seek. Before the first presentation it falls
    /// back to running time (which matches the stream position only for
    /// streams that start at zero), and is `ClockTime::NONE` for a clock-less
    /// pipeline that has not presented anything.
    pub fn position(&self) -> ClockTime {
        let pts = self.position.load(Ordering::Acquire);
        if pts != u64::MAX {
            return ClockTime::from_nanos(pts);
        }
        match &self.pausable {
            Some(clock) if self.base_time.is_some() => clock.now().saturating_sub(self.base_time),
            _ => ClockTime::NONE,
        }
    }

    /// Subscribe to pipeline events.
    pub fn subscribe(&self) -> EventReceiver {
        self.events.subscribe()
    }

    /// Get the event sender.
    pub fn event_sender(&self) -> &EventSender {
        &self.events
    }

    /// Get a mutable reference to the pipeline bus for polling messages.
    ///
    /// Returns `None` if the bus was already taken via [`take_bus`](Self::take_bus).
    pub fn bus_mut(&mut self) -> Option<&mut Bus> {
        self.bus.as_mut()
    }

    /// Take the bus out of the handle (for moving to another task).
    pub fn take_bus(&mut self) -> Option<Bus> {
        self.bus.take()
    }

    /// Get the bus handle for posting pipeline-level messages.
    pub fn bus_handle(&self) -> Option<&BusHandle> {
        self.bus_handle.as_ref()
    }
}

// ============================================================================
// Executor
// ============================================================================

/// Unified pipeline executor.
///
/// Handles all execution modes through a single interface.
pub struct Executor {
    config: ExecutorConfig,
}

impl Executor {
    /// Create a new executor with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
        }
    }

    /// Create an executor with custom configuration.
    pub fn with_config(config: ExecutorConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Run the pipeline to completion.
    pub async fn run(&self, pipeline: &mut Pipeline) -> Result<()> {
        let handle = self.start(pipeline)?;
        handle.wait().await
    }

    /// Start the pipeline and return a handle.
    pub fn start(&self, pipeline: &mut Pipeline) -> Result<PipelineHandle> {
        // Create event sender
        let events = EventSender::new(256);

        // Cooperative stop flag shared with every source task (see
        // PipelineHandle::stop).
        let stop = Arc::new(AtomicBool::new(false));

        // Report what the element hints suggest. This is advisory only — see
        // the note on `effective_scheduling` below for why the suggestion is
        // not acted on automatically.
        if self.config.auto_strategy {
            let plan = analyze_pipeline(pipeline);
            if plan.needs_rt && self.config.scheduling == SchedulingMode::Async {
                tracing::info!(
                    "{} of {} elements report RT-safe, low-latency hints. The pipeline is \
                     running async; pass ExecutorConfig::with_scheduling(SchedulingMode::Hybrid) \
                     to put them on RT threads.",
                    plan.rt_nodes.len(),
                    plan.rt_nodes.len() + plan.async_nodes.len(),
                );
            }
        }

        // State transitions
        let old_state = pipeline.state();
        let bus_handle = pipeline.bus_handle().clone();
        let outcome = TerminalOutcome::new(bus_handle.clone());
        // The seed share, held for the whole of `start()`. Without it, a source
        // that runs dry before its siblings are even spawned would take the live
        // count to zero and declare EOS for the whole graph.
        let seed = TaskGuard::new(&outcome, "<pipeline>");
        if old_state == PipelineState::Suspended {
            pipeline.prepare()?;
            events.send_state_changed(old_state, PipelineState::Idle);
            bus_handle.post_state_changed(old_state, PipelineState::Idle);
        }

        // The configured mode is the effective mode.
        //
        // `auto_strategy` used to recompute this from the plan, promoting a
        // pipeline to `Hybrid` whenever any element looked RT-worthy — and
        // then discarding the result, because the partitioner reads
        // `RtConfig::mode`, which `auto_strategy` never touched. So the
        // promotion has never taken effect, and honouring it now would move
        // every pipeline containing an rt_safe low-latency element onto RT
        // threads by default: a large behavioural change, onto the code path
        // that still has no probes or tracers (#43). That is its own decision,
        // not a side effect of fixing #6, so auto-promotion stays off and the
        // plan is used for reporting only.
        //
        // What #6 needed is below: an explicitly configured mode is now
        // actually applied, instead of being silently overridden here and then
        // ignored by the partitioner.
        let effective_scheduling = self.config.scheduling;

        // Partition graph for hybrid scheduling.
        //
        // `ExecutorConfig::scheduling` is authoritative: `RtConfig::mode` is
        // derived from it here rather than trusted. Without this,
        // `ExecutorConfig::default().with_scheduling(RealTime)` left `rt.mode`
        // at its `Async` default, `should_run_rt` rejected every node, and the
        // pipeline *always* fell back to async no matter how many RT-safe
        // elements it contained — the root cause behind #6, one level below
        // the empty-partition fallback the issue described.
        let mut rt_config = self.config.rt.clone();
        rt_config.mode = effective_scheduling;
        let mut scheduler = RtScheduler::new(rt_config);
        let partition = if effective_scheduling != SchedulingMode::Async {
            scheduler.partition_graph(pipeline)?
        } else {
            // All async - create empty partition
            GraphPartition {
                async_nodes: pipeline.node_ids(),
                rt_nodes: Vec::new(),
                boundary_edges: Vec::new(),
            }
        };

        tracing::debug!(
            "Graph partition: {} async, {} RT, {} boundaries (mode: {:?})",
            partition.async_nodes.len(),
            partition.rt_nodes.len(),
            partition.boundary_edges.len(),
            effective_scheduling
        );

        // Auto-select the best clock from pipeline elements, then start it.
        //
        // The selected clock is wrapped in a `PausableClock` *before* it is
        // distributed: elements receive the raw `Arc<dyn Clock>` plus a copied
        // base time, so this shared wrapper is the only lever that still
        // reaches them after start — it is what makes `PipelineHandle::pause`
        // freeze presentation everywhere at once.
        pipeline.select_clock();
        pipeline.start_clock();
        let pipeline_clock = pipeline.clock();
        let (clock_info, pausable): (
            Option<(Arc<dyn Clock>, ClockTime)>,
            Option<Arc<crate::clock::PausableClock>>,
        ) = if pipeline_clock.is_started() {
            let wrapped = Arc::new(crate::clock::PausableClock::new(pipeline_clock.clock()));
            (
                Some((
                    wrapped.clone() as Arc<dyn Clock>,
                    pipeline_clock.base_time(),
                )),
                Some(wrapped),
            )
        } else {
            (None, None)
        };

        // Runtime-control state, filled in as the tasks spawn.
        let (pause_tx, pause_rx) = watch::channel(false);
        let mut runtime = RuntimeControls {
            controls: Vec::new(),
            pause_rx,
            position: Arc::new(AtomicU64::new(u64::MAX)),
        };

        // Execute based on scheduling mode
        let (tasks, rt_handles, bridges, rt_driver_task) = match effective_scheduling {
            SchedulingMode::Async => {
                let tasks = self.run_async(
                    pipeline,
                    clock_info.as_ref(),
                    &events,
                    &stop,
                    &outcome,
                    &mut runtime,
                )?;
                (tasks, Vec::new(), Vec::new(), None)
            }
            SchedulingMode::Hybrid | SchedulingMode::RealTime => {
                if partition.rt_nodes.is_empty() {
                    // No node qualified, so there is nothing to schedule on an
                    // RT thread and the whole graph runs async. That is a
                    // legitimate outcome — but it silently discards the latency
                    // guarantee the caller asked for, so say so (#6).
                    //
                    // `RealTime` gets the louder message: `Hybrid` is a
                    // best-effort request by construction, whereas `RealTime`
                    // is chosen precisely when RT execution is the point.
                    let names = partition
                        .async_nodes
                        .iter()
                        .filter_map(|&id| pipeline.get_node(id).map(|n| n.name().to_string()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    if effective_scheduling == SchedulingMode::RealTime {
                        tracing::warn!(
                            "SchedulingMode::RealTime requested, but no element reported \
                             ExecutionHints::rt_safe — running fully async. \
                             Nodes: [{names}]. Implement SyncElement and return \
                             ExecutionHints::rt_safe() from execution_hints() to opt in."
                        );
                    } else {
                        tracing::warn!(
                            "SchedulingMode::Hybrid requested, but no element is both rt_safe \
                             and low-latency — running fully async. Nodes: [{names}]."
                        );
                    }
                    let tasks = self.run_async(
                        pipeline,
                        clock_info.as_ref(),
                        &events,
                        &stop,
                        &outcome,
                        &mut runtime,
                    )?;
                    (tasks, Vec::new(), Vec::new(), None)
                } else {
                    self.run_hybrid(
                        pipeline,
                        &partition,
                        &mut scheduler,
                        clock_info.as_ref(),
                        &events,
                        &stop,
                        &outcome,
                        &mut runtime,
                    )?
                }
            }
        };

        // Start driver if configured
        let driver = self.config.driver.as_ref().map(|config| {
            let driver = TimerDriver::new(config.clone());
            driver.start_async()
        });

        // Activate (Idle → Running)
        let idle_state = pipeline.state();
        if let Err(e) = pipeline.activate() {
            // The tasks are already spawned. Say what happened before returning,
            // or they drop their guards into a spurious EOS.
            outcome.fail("<pipeline>", &e);
            return Err(e);
        }
        events.send_state_changed(idle_state, PipelineState::Running);
        bus_handle.post_state_changed(idle_state, PipelineState::Running);
        events.send(PipelineEvent::Started);
        pipeline.tracer_registry().notify_start();

        // Take the bus from the pipeline and store it on the handle.
        let bus = pipeline.take_bus();
        let bus_handle = Some(pipeline.bus_handle().clone());

        // Release the seed share of the live count, now that every task holds
        // its own — but only for an all-async run. RT threads loop until told to
        // stop, so they cannot be counted; instead the seed rides on the handle
        // and is released once `wait`/`abort` has joined them.
        //
        // Last, after the state change is announced: while the seed is held, a
        // pipeline that finishes instantly cannot post EOS ahead of
        // `StateChanged{Idle → Running}`.
        let seed = if rt_handles.is_empty() {
            drop(seed);
            None
        } else {
            Some(seed)
        };

        Ok(PipelineHandle {
            tasks,
            rt_handles,
            events,
            bridges,
            driver,
            rt_driver_task,
            bus,
            bus_handle,
            stop,
            outcome,
            seed,
            controls: runtime.controls,
            pause_tx,
            pausable,
            base_time: clock_info.map(|(_, b)| b).unwrap_or(ClockTime::NONE),
            position: runtime.position,
        })
    }

    /// Run all nodes as async Tokio tasks.
    #[allow(clippy::too_many_arguments)]
    fn run_async(
        &self,
        pipeline: &mut Pipeline,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
        outcome: &Arc<TerminalOutcome>,
        runtime: &mut RuntimeControls,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut channels = ChannelNetwork::new();

        // Build channels
        for src_id in pipeline.sources() {
            self.build_channels(pipeline, src_id, &mut channels);
        }

        // Spawn tasks
        self.spawn_tasks(
            pipeline, channels, clock_info, events, stop, outcome, runtime,
        )
    }

    /// Run with hybrid async + RT execution.
    #[allow(clippy::too_many_arguments)]
    fn run_hybrid(
        &self,
        pipeline: &mut Pipeline,
        partition: &GraphPartition,
        scheduler: &mut RtScheduler,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
        outcome: &Arc<TerminalOutcome>,
        runtime: &mut RuntimeControls,
    ) -> Result<(
        Vec<JoinHandle<Result<()>>>,
        Vec<crate::pipeline::rt_scheduler::DataThreadHandle>,
        Vec<Arc<AsyncRtBridge>>,
        Option<JoinHandle<Result<()>>>,
    )> {
        use crate::pipeline::rt_bridge::EventFd;
        use crate::pipeline::rt_scheduler::{BoundaryDirection, spawn_data_thread};

        // Create bridges at boundaries
        scheduler.create_bridges(partition)?;
        scheduler.setup_activations(partition)?;
        scheduler.compute_processing_order(partition, pipeline)?;
        scheduler.setup_dependencies(partition, pipeline)?;
        scheduler.select_driver(partition, pipeline);

        // Build downstream map for dependency signaling
        let downstream_map = scheduler.build_downstream_map(partition, pipeline);

        // Build channels for async nodes
        let mut channels = ChannelNetwork::new();
        let async_set: std::collections::HashSet<_> =
            partition.async_nodes.iter().copied().collect();

        for src_id in pipeline.sources() {
            if async_set.contains(&src_id) {
                self.build_channels_for_async(
                    pipeline,
                    src_id,
                    &async_set,
                    partition,
                    scheduler,
                    &mut channels,
                );
            }
        }

        // Spawn async tasks for the async portion of the graph
        let tasks = self.spawn_tasks_for_partition(
            pipeline, partition, channels, scheduler, clock_info, events, stop, outcome, runtime,
        )?;

        // Collect bridges (keep alive)
        let bridges: Vec<_> = partition
            .boundary_edges
            .iter()
            .filter_map(|e| scheduler.get_bridge(e.source, e.sink))
            .collect();

        // --- Spawn RT data thread ---

        // Extract RT elements from the pipeline graph.
        //
        // These bypass `spawn_node_task_with_bridges`, so the setters it
        // applies have to be repeated here — until this was noticed, an element
        // scheduled onto an RT thread silently received no clock, no bus and no
        // arena budget, and behaved differently from the same element in an
        // async graph.
        //
        // Budgets are computed up front because `children()` borrows the
        // pipeline shared while `get_node_mut` needs it mutable.
        let rt_budgets: HashMap<NodeId, OutputBudget> = partition
            .rt_nodes
            .iter()
            .map(|&node_id| {
                let bridged = partition.boundary_edges.iter().any(|e| {
                    e.source == node_id && matches!(e.direction, BoundaryDirection::RtToAsync)
                });
                (
                    node_id,
                    self.output_slot_budget(pipeline, node_id, usize::from(bridged)),
                )
            })
            .collect();

        let mut rt_elements: HashMap<NodeId, Box<DynAsyncElement<'static>>> = HashMap::new();
        for &node_id in &partition.rt_nodes {
            let node_name = pipeline
                .get_node(node_id)
                .map(|n| n.name().to_string())
                .unwrap_or_default();
            let bus = pipeline.bus_handle().for_element(&node_name);

            if let Some(node) = pipeline.get_node_mut(node_id)
                && let Some(mut element) = node.take_element()
            {
                if let Some((clock, base_time)) = clock_info {
                    element.set_clock(clock.clone(), *base_time);
                }
                element.set_bus(bus);
                if let Some(budget) = rt_budgets.get(&node_id) {
                    element.set_output_budget(*budget);
                }
                rt_elements.insert(node_id, element);
            }
        }

        // Build input/output bridge maps for the RT data thread
        let mut input_bridges: HashMap<NodeId, Arc<AsyncRtBridge>> = HashMap::new();
        let mut output_bridges: HashMap<NodeId, Arc<AsyncRtBridge>> = HashMap::new();

        for edge in &partition.boundary_edges {
            if let Some(bridge) = scheduler.get_bridge(edge.source, edge.sink) {
                match edge.direction {
                    BoundaryDirection::AsyncToRt => {
                        input_bridges.insert(edge.sink, bridge);
                    }
                    BoundaryDirection::RtToAsync => {
                        output_bridges.insert(edge.source, bridge);
                    }
                }
            }
        }

        // Create driver trigger (the data thread waits on this each cycle)
        let driver_trigger = Arc::new(EventFd::new()?);

        // Spawn the data thread
        let rt_handle = spawn_data_thread(
            "parallax-rt-0".to_string(),
            self.config.rt.clone(),
            scheduler.processing_order().to_vec(),
            scheduler.activations().clone(),
            rt_elements,
            input_bridges,
            output_bridges,
            downstream_map,
            driver_trigger.clone(),
        )?;

        // Spawn a driver task that triggers the RT thread periodically
        let driver_period = self
            .config
            .driver
            .as_ref()
            .map(|d| d.period)
            .unwrap_or(std::time::Duration::from_millis(5));

        let driver_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(driver_period);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                if let Err(e) = driver_trigger.notify() {
                    tracing::error!("driver trigger notify error: {}", e);
                    return Err(Error::Io(std::io::Error::other(format!(
                        "driver trigger: {e}"
                    ))));
                }
            }
            #[allow(unreachable_code)]
            Ok(())
        });
        // NOTE: driver_task is NOT pushed to `tasks` because it runs indefinitely.
        // It is stored in PipelineHandle.rt_driver_task and aborted on shutdown.

        let rt_handles = vec![rt_handle];

        Ok((tasks, rt_handles, bridges, Some(driver_task)))
    }

    /// How many buffers the graph downstream of `node_id` can hold at once.
    ///
    /// Elements that allocate their own output buffers need an arena at least
    /// this big, or they run out of slots the moment a consumer falls behind —
    /// which used to kill the pipeline. Only the executor knows the number:
    /// link capacity is its configuration.
    ///
    /// **Per pad it is the maximum, not the sum.** Fan-out clones a `Buffer`,
    /// and a clone is a refcount bump on the same slot, so three branches each
    /// holding the same buffer pin one slot between them. The producer stalls on
    /// whichever `Block` branch fills first, so the deepest link bounds the pad.
    /// Summing would over-allocate by up to N×.
    ///
    /// **Across pads it is the sum**, because separate src pads (a demuxer's)
    /// carry genuinely different buffers.
    ///
    /// A bridged edge contributes `RtConfig::bridge_capacity` the same way — the
    /// bridge is a queue like any other.
    fn output_slot_budget(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        output_bridges: usize,
    ) -> OutputBudget {
        let mut per_pad: HashMap<String, usize> = HashMap::new();

        for (_child_id, link) in pipeline.children(node_id) {
            let capacity = link.capacity.unwrap_or(self.config.channel_capacity);
            let deepest = per_pad.entry(link.src_pad.clone()).or_insert(0);
            *deepest = (*deepest).max(capacity);
        }

        let mut downstream_capacity: usize = per_pad.values().sum();
        if output_bridges > 0 {
            downstream_capacity =
                downstream_capacity.saturating_add(self.config.rt.bridge_capacity);
        }

        OutputBudget::new(downstream_capacity, defaults::IN_FLIGHT_MARGIN)
    }

    /// Build channel network recursively.
    fn build_channels(&self, pipeline: &Pipeline, node_id: NodeId, network: &mut ChannelNetwork) {
        for (child_id, link) in pipeline.children(node_id) {
            if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                let capacity = link.capacity.unwrap_or(self.config.channel_capacity);
                let (tx, rx) = bounded_async::<Message>(capacity);
                network.add_channel(
                    node_id,
                    link.src_pad.clone(),
                    child_id,
                    link.sink_pad.clone(),
                    tx,
                    rx,
                    link.policy,
                    node_name(pipeline, child_id),
                );
            }
            self.build_channels(pipeline, child_id, network);
        }
    }

    /// Build channels for async portion only.
    fn build_channels_for_async(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        async_set: &std::collections::HashSet<NodeId>,
        partition: &GraphPartition,
        _scheduler: &RtScheduler,
        network: &mut ChannelNetwork,
    ) {
        for (child_id, link) in pipeline.children(node_id) {
            let is_boundary = partition
                .boundary_edges
                .iter()
                .any(|e| e.source == node_id && e.sink == child_id);

            if is_boundary {
                continue; // Bridge handles this
            }

            if async_set.contains(&child_id) {
                if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                    let capacity = link.capacity.unwrap_or(self.config.channel_capacity);
                    let (tx, rx) = bounded_async::<Message>(capacity);
                    network.add_channel(
                        node_id,
                        link.src_pad.clone(),
                        child_id,
                        link.sink_pad.clone(),
                        tx,
                        rx,
                        link.policy,
                        node_name(pipeline, child_id),
                    );
                }
                self.build_channels_for_async(
                    pipeline, child_id, async_set, partition, _scheduler, network,
                );
            }
        }
    }

    /// Spawn tasks for all nodes.
    #[allow(clippy::too_many_arguments)]
    fn spawn_tasks(
        &self,
        pipeline: &mut Pipeline,
        mut channels: ChannelNetwork,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
        outcome: &Arc<TerminalOutcome>,
        runtime: &mut RuntimeControls,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut tasks = Vec::new();

        let node_ids: Vec<NodeId> = pipeline
            .sources()
            .into_iter()
            .chain(self.collect_reachable(pipeline))
            .collect();

        let mut seen = std::collections::HashSet::new();
        let node_ids: Vec<NodeId> = node_ids.into_iter().filter(|id| seen.insert(*id)).collect();

        for node_id in node_ids {
            let task = self.spawn_node_task(
                pipeline,
                node_id,
                &mut channels,
                clock_info,
                events,
                stop,
                outcome,
                runtime,
            )?;
            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Spawn tasks for async partition only.
    #[allow(clippy::too_many_arguments)]
    fn spawn_tasks_for_partition(
        &self,
        pipeline: &mut Pipeline,
        partition: &GraphPartition,
        mut channels: ChannelNetwork,
        scheduler: &RtScheduler,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
        outcome: &Arc<TerminalOutcome>,
        runtime: &mut RuntimeControls,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut tasks = Vec::new();

        for &node_id in &partition.async_nodes {
            let output_bridges: Vec<_> = partition
                .boundary_edges
                .iter()
                .filter(|e| e.source == node_id && e.direction == BoundaryDirection::AsyncToRt)
                .filter_map(|e| scheduler.get_bridge(e.source, e.sink))
                .collect();

            let input_bridges: Vec<_> = partition
                .boundary_edges
                .iter()
                .filter(|e| e.sink == node_id && e.direction == BoundaryDirection::RtToAsync)
                .filter_map(|e| scheduler.get_bridge(e.source, e.sink))
                .collect();

            let task = self.spawn_node_task_with_bridges(
                pipeline,
                node_id,
                &mut channels,
                output_bridges,
                input_bridges,
                clock_info,
                events,
                stop,
                outcome,
                runtime,
            )?;
            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Spawn a task for a single node.
    #[allow(clippy::too_many_arguments)]
    fn spawn_node_task(
        &self,
        pipeline: &mut Pipeline,
        node_id: NodeId,
        channels: &mut ChannelNetwork,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
        outcome: &Arc<TerminalOutcome>,
        runtime: &mut RuntimeControls,
    ) -> Result<JoinHandle<Result<()>>> {
        self.spawn_node_task_with_bridges(
            pipeline,
            node_id,
            channels,
            Vec::new(),
            Vec::new(),
            clock_info,
            events,
            stop,
            outcome,
            runtime,
        )
    }

    /// Spawn a task with optional bridges.
    #[allow(clippy::too_many_arguments)]
    fn spawn_node_task_with_bridges(
        &self,
        pipeline: &mut Pipeline,
        node_id: NodeId,
        channels: &mut ChannelNetwork,
        output_bridges: Vec<Arc<AsyncRtBridge>>,
        input_bridges: Vec<Arc<AsyncRtBridge>>,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
        outcome: &Arc<TerminalOutcome>,
        runtime: &mut RuntimeControls,
    ) -> Result<JoinHandle<Result<()>>> {
        // Before `get_node_mut` borrows the pipeline mutably: `children()` needs
        // it shared.
        let budget = self.output_slot_budget(pipeline, node_id, output_bridges.len());

        let node = pipeline
            .get_node_mut(node_id)
            .ok_or_else(|| Error::InvalidSegment("node not found".into()))?;

        let element_type = node.element_type();
        let node_name = node.name().to_string();

        let mut element = node.take_element().ok_or_else(|| {
            Error::InvalidSegment(format!("element '{}' already taken", node_name))
        })?;

        // Hand every element the pipeline clock, exactly as `set_bus` below
        // does. This used to be gated on `ElementType::Source`, which meant a
        // sink could not see the clock at all — and so could not schedule
        // presentation against running time (#65). The trait method defaults
        // to a no-op, so element types that ignore the clock are unaffected.
        if let Some((clock, base_time)) = clock_info {
            element.set_clock(clock.clone(), *base_time);
        }

        // Set bus handle so elements can post messages
        element.set_bus(pipeline.bus_handle().for_element(&node_name));

        // Tell the element how much the graph below it can hold, so it can size
        // its output arena before the first frame builds it.
        element.set_output_budget(budget);

        // Take the channels this element type actually needs, inside the match.
        //
        // These used to be hoisted out of the match, which quietly broke both
        // N-ary element types: `take_inputs`/`take_outputs` *drain* the maps
        // (they are `take_*_by_pad().into_values().flatten()`), so by the time
        // the Muxer arm called `take_inputs_by_pad` it got an empty map and the
        // muxer received no inputs at all — likewise the Demuxer arm's
        // `take_outputs_by_pad`. Muxers and demuxers added through
        // `add_muxer`/`add_demuxer` therefore never moved a single buffer.
        let events_clone = events.clone();
        let probes = pipeline.probe_registry().clone();
        let tracers = pipeline.tracer_registry().clone();
        // Claimed here, on this thread, and *not* inside the task body:
        // `tokio::spawn` returns before the future is first polled, so a guard
        // built in there would leave the live count at the seed and let
        // `start()` declare EOS before any element had run.
        let share = TaskGuard::new(outcome, &node_name);

        let task = match element_type {
            ElementType::Source => {
                // Out-of-band control channel for upstream events (seek). Small
                // and bounded: control events are rare and handled promptly.
                let (control_tx, control_rx) = bounded_async::<Event>(8);
                let bus = pipeline.bus_handle().for_element(&node_name);
                runtime.controls.push(SourceControl {
                    name: node_name.clone(),
                    tx: control_tx,
                });
                spawn_source_task(
                    node_name,
                    node_id,
                    element,
                    channels.take_outputs(node_id),
                    output_bridges,
                    events_clone,
                    probes,
                    tracers,
                    stop.clone(),
                    control_rx,
                    runtime.pause_rx.clone(),
                    bus,
                    share,
                )
            }
            ElementType::Sink => spawn_sink_task(
                node_name,
                node_id,
                element,
                channels.take_inputs(node_id),
                input_bridges,
                events_clone,
                probes,
                tracers,
                runtime.position.clone(),
                share,
            ),
            ElementType::Transform => spawn_transform_task(
                node_name,
                node_id,
                element,
                channels.take_inputs(node_id),
                channels.take_outputs(node_id),
                input_bridges,
                output_bridges,
                events_clone,
                probes,
                tracers,
                ShedTracker::new(self.config.shed_fatal_after),
                share,
            ),
            ElementType::Demuxer => {
                // Inputs flattened (one sink pad), outputs kept per pad — that
                // is the whole point of a demuxer.
                let inputs = channels.take_inputs(node_id);
                let outputs_by_pad = channels.take_outputs_by_pad(node_id);
                // A source-style demuxer (no input links) is a source in every
                // sense, including runtime control: it gets a control channel
                // so PipelineHandle::seek reaches Demuxer::handle_upstream_event.
                let control_rx = if inputs.is_empty() {
                    let (control_tx, control_rx) = bounded_async::<Event>(8);
                    runtime.controls.push(SourceControl {
                        name: node_name.clone(),
                        tx: control_tx,
                    });
                    Some(control_rx)
                } else {
                    None
                };
                let bus = pipeline.bus_handle().for_element(&node_name);
                spawn_demuxer_task(
                    node_name,
                    node_id,
                    element,
                    inputs,
                    outputs_by_pad,
                    events_clone,
                    probes,
                    tracers,
                    stop.clone(),
                    control_rx,
                    bus,
                    share,
                )
            }
            ElementType::Muxer => {
                // Mirror image: inputs per pad, outputs flattened.
                let inputs_by_pad = channels.take_inputs_by_pad(node_id);
                let outputs = channels.take_outputs(node_id);
                spawn_muxer_task(
                    node_name,
                    node_id,
                    element,
                    inputs_by_pad,
                    outputs,
                    events_clone,
                    probes,
                    tracers,
                    share,
                )
            }
        };

        Ok(task)
    }

    /// Collect reachable nodes from sources.
    fn collect_reachable(&self, pipeline: &Pipeline) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for src in pipeline.sources() {
            self.collect_from(pipeline, src, &mut result, &mut visited);
        }

        result
    }

    fn collect_from(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        result: &mut Vec<NodeId>,
        visited: &mut std::collections::HashSet<NodeId>,
    ) {
        if !visited.insert(node_id) {
            return;
        }

        for (child_id, _) in pipeline.children(node_id) {
            result.push(child_id);
            self.collect_from(pipeline, child_id, result, visited);
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Internal Types
// ============================================================================

/// Bookkeeping for buffers shed because an element's output arena was full.
///
/// [`Error::PoolExhausted`] is the one error the executor does not treat as
/// fatal. An arena runs dry when downstream is holding more buffers than the
/// budget anticipated — an application sitting on everything it pulled, a deep
/// queue — and for a live pipeline the right answer is to drop a buffer and keep
/// going, not to tear the graph down. Frames are recoverable; a dead capture
/// session is not.
///
/// It still has to be *visible*, or a pipeline silently delivering half its
/// frames looks healthy. Every shed increments the drop tracer and the
/// `parallax_buffers_dropped` metric; the log is rate-limited to the 1st, 10th,
/// 100th... consecutive shed, because the failure mode is high-frequency by
/// nature and a per-frame warning would bury everything else.
struct ShedTracker {
    consecutive: u64,
    total: u64,
    fatal_after: Option<u64>,
}

impl ShedTracker {
    fn new(fatal_after: Option<u64>) -> Self {
        Self {
            consecutive: 0,
            total: 0,
            fatal_after,
        }
    }

    /// Record one shed buffer. `Err` means the configured limit was reached and
    /// the caller should fail the task after all.
    fn record(&mut self, name: &str, tracers: &TracerRegistry) -> Result<()> {
        self.consecutive += 1;
        self.total += 1;

        tracers.notify_drop(name);
        crate::observability::record_buffer_dropped("pipeline", name);

        if is_power_of_ten(self.consecutive) {
            tracing::warn!(
                "{name}: output arena exhausted, shedding buffers ({} in a row, {} total) — \
                 downstream is holding more than the link capacity accounts for",
                self.consecutive,
                self.total,
            );
        }

        match self.fatal_after {
            Some(limit) if self.consecutive >= limit => Err(Error::Element(format!(
                "{name}: shed {} buffers in a row (shed_fatal_after = {limit})",
                self.consecutive
            ))),
            _ => Ok(()),
        }
    }

    /// A buffer got through: the burst is over.
    fn reset(&mut self) {
        self.consecutive = 0;
    }
}

/// `1, 10, 100, ...` — the rate-limit ladder for shed warnings.
fn is_power_of_ten(mut n: u64) -> bool {
    if n == 0 {
        return false;
    }
    while n.is_multiple_of(10) {
        n /= 10;
    }
    n == 1
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // Intentional: avoid heap allocation on hot path
enum Message {
    Buffer(Buffer),
    Eos,
    /// An upstream element failed. Terminal, and never accompanied by an `Eos`
    /// — sending both invites the receiver to record whichever arrives second
    /// as the reason the stream ended.
    Error(StreamError),
    /// A non-terminal in-band event (FlushStart/FlushStop/Segment/...).
    ///
    /// `Eos` and `Error` keep their own variants — they are terminal and every
    /// receive loop exits on them, which an `Event` arm must never do. Events
    /// occupy channel slots between buffers, so their ordering relative to
    /// buffers on a link is FIFO-exact.
    Event(Event),
}

type ChannelKey = (NodeId, String, NodeId, String);

/// One downstream branch of a src-pad, with the policy of the link that made it.
///
/// The old code flattened a pad's outputs to a bare `Vec<AsyncSender>`, which
/// threw away *which branch is which* — and with it any chance of treating a
/// slow branch differently from a fast one.
#[derive(Clone)]
struct OutputBranch {
    tx: AsyncSender<Message>,
    policy: LinkPolicy,
    /// Name of the element on the far end, for drop reporting.
    sink_name: String,
}

impl OutputBranch {
    /// Send a buffer, honouring the link policy.
    ///
    /// kanal's `try_send` returns `Ok(false)` when the channel is **full** (not
    /// an error) and `Err` only when it is closed — easy to get backwards.
    /// Returns `false` once the branch's receiver is gone.
    ///
    /// The result used to be discarded, and kanal reports a closed channel
    /// *immediately* rather than blocking — so a source feeding a sink that had
    /// died spun at 100% CPU producing into a closed channel until the handle
    /// was dropped. Callers use this to stop.
    async fn send_buffer(&self, buffer: Buffer, tracers: &TracerRegistry) -> bool {
        match self.policy {
            LinkPolicy::Block => self.tx.send(Message::Buffer(buffer)).await.is_ok(),
            LinkPolicy::Drop => match self.tx.try_send(Message::Buffer(buffer)) {
                Ok(true) => true,
                Ok(false) => {
                    tracers.notify_drop(&self.sink_name);
                    true
                }
                Err(_) => false, // closed
            },
        }
    }
}

/// Broadcast one buffer to every branch of a pad.
///
/// Blocking branches are awaited **concurrently**, not one after another: a
/// sequential loop makes each branch wait for the ones before it, which shows up
/// as latency skew between equal-speed branches. The single-output case — the
/// overwhelmingly common one — takes a direct await and allocates nothing.
/// Returns `false` when every branch's receiver has gone away, so the producer
/// can stop rather than spin. A pad with no branches at all counts as live.
async fn broadcast(branches: &[OutputBranch], buffer: Buffer, tracers: &TracerRegistry) -> bool {
    match branches {
        [] => true,
        [only] => only.send_buffer(buffer, tracers).await,
        many => futures::future::join_all(
            many.iter()
                .map(|branch| branch.send_buffer(buffer.clone(), tracers)),
        )
        .await
        .into_iter()
        .any(|connected| connected),
    }
}

/// Send EOS to every branch, **always blocking**.
///
/// A dropped EOS would leave the branch's sink waiting forever, so `Drop` does
/// not apply to it. Only buffers are ever dropped.
async fn broadcast_eos(branches: &[OutputBranch]) {
    for branch in branches {
        let _ = branch.tx.send(Message::Eos).await;
    }
}

/// Send an in-band event to every branch, **always blocking**.
///
/// Events are control flow, not payload: a dropped FlushStop would leave the
/// branch's subtree discarding buffers forever, so `LinkPolicy::Drop` does not
/// apply — same reasoning as [`broadcast_eos`].
async fn broadcast_event(branches: &[OutputBranch], event: &Event) {
    for branch in branches {
        let _ = branch.tx.send(Message::Event(event.clone())).await;
    }
}

/// Run an element call, converting a panic into an ordinary element error.
///
/// A panicking task used to be invisible: it never reached its own error arm,
/// so it never told anything downstream, and every sink below it waited
/// forever. Turning the panic into an `Err` at the call site means the arms
/// that already know how to report and propagate a failure handle it — the
/// pipeline reports `Error::Panic` naming the element, and consumers see
/// `EndReason::Error` like any other failure.
///
/// `AssertUnwindSafe` is required because the element is `&mut`-borrowed across
/// await points, and is sound here for a specific reason: every caller treats
/// the `Err` as terminal, so the element is dropped rather than re-entered. The
/// hazard `UnwindSafe` guards against — observing state left half-updated — is
/// structurally excluded.
///
/// The default panic hook still runs, so the backtrace is printed as usual.
async fn guard<T>(node: &str, call: impl std::future::Future<Output = Result<T>>) -> Result<T> {
    match AssertUnwindSafe(call).catch_unwind().await {
        Ok(result) => result,
        Err(payload) => Err(Error::Panic {
            node: node.to_owned(),
            message: crate::error::panic_message(payload.as_ref()),
        }),
    }
}

/// Send a terminal error to every branch, **always blocking**.
///
/// Same reasoning as [`broadcast_eos`]: a dropped terminal message wedges the
/// branch's sink. This is what makes a failed pipeline distinguishable from a
/// finished one at the app boundary — the sink used to receive a plain EOS and
/// report a clean end of stream.
async fn broadcast_error(branches: &[OutputBranch], error: &StreamError) {
    for branch in branches {
        let _ = branch.tx.send(Message::Error(error.clone())).await;
    }
}

/// Name of a node, for drop reporting on the link that feeds it.
fn node_name(pipeline: &Pipeline, node_id: NodeId) -> String {
    pipeline
        .get_node(node_id)
        .map(|n| n.name().to_string())
        .unwrap_or_default()
}

struct ChannelNetwork {
    channels: HashMap<ChannelKey, (AsyncSender<Message>, AsyncReceiver<Message>)>,
    outputs: HashMap<(NodeId, String), Vec<OutputBranch>>,
    inputs: HashMap<(NodeId, String), Vec<AsyncReceiver<Message>>>,
}

impl ChannelNetwork {
    fn new() -> Self {
        Self {
            channels: HashMap::new(),
            outputs: HashMap::new(),
            inputs: HashMap::new(),
        }
    }

    fn has_channel(&self, src: NodeId, src_pad: &str, sink: NodeId, sink_pad: &str) -> bool {
        self.channels
            .contains_key(&(src, src_pad.to_string(), sink, sink_pad.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn add_channel(
        &mut self,
        src: NodeId,
        src_pad: String,
        sink: NodeId,
        sink_pad: String,
        tx: AsyncSender<Message>,
        rx: AsyncReceiver<Message>,
        policy: LinkPolicy,
        sink_name: String,
    ) {
        self.channels.insert(
            (src, src_pad.clone(), sink, sink_pad.clone()),
            (tx.clone(), rx.clone()),
        );
        self.outputs
            .entry((src, src_pad))
            .or_default()
            .push(OutputBranch {
                tx,
                policy,
                sink_name,
            });
        self.inputs.entry((sink, sink_pad)).or_default().push(rx);
    }

    fn take_outputs_by_pad(&mut self, node: NodeId) -> HashMap<String, Vec<OutputBranch>> {
        let mut result = HashMap::new();
        let keys: Vec<_> = self
            .outputs
            .keys()
            .filter(|(n, _)| *n == node)
            .cloned()
            .collect();
        for (n, pad) in keys {
            if let Some(senders) = self.outputs.remove(&(n, pad.clone())) {
                result.insert(pad, senders);
            }
        }
        result
    }

    fn take_inputs_by_pad(&mut self, node: NodeId) -> HashMap<String, Vec<AsyncReceiver<Message>>> {
        let mut result = HashMap::new();
        let keys: Vec<_> = self
            .inputs
            .keys()
            .filter(|(n, _)| *n == node)
            .cloned()
            .collect();
        for (n, pad) in keys {
            if let Some(receivers) = self.inputs.remove(&(n, pad.clone())) {
                result.insert(pad, receivers);
            }
        }
        result
    }

    fn take_outputs(&mut self, node: NodeId) -> Vec<OutputBranch> {
        self.take_outputs_by_pad(node)
            .into_values()
            .flatten()
            .collect()
    }

    fn take_inputs(&mut self, node: NodeId) -> Vec<AsyncReceiver<Message>> {
        self.take_inputs_by_pad(node)
            .into_values()
            .flatten()
            .collect()
    }
}

// ============================================================================
// Task Spawning
// ============================================================================

#[allow(clippy::too_many_arguments)]
/// Handle one upstream event delivered to a source task's control channel.
///
/// For a handled flushing seek this runs the flush sequence in-band on every
/// output branch — FlushStart → FlushStop → Segment, the order downstream
/// elements rely on to discard stale data and re-anchor their timeline — and
/// posts [`MessageKind::SeekDone`](crate::pipeline::bus::MessageKind::SeekDone) on the bus. The segment is synthesized here
/// from the seek's target: asking the element for its actual landing position
/// would need a new `AsyncElementDyn` method, which is an ABI bump.
async fn handle_source_control_event(
    name: &str,
    element: &mut Box<DynAsyncElement<'static>>,
    event: &Event,
    outputs: &[OutputBranch],
    src_pad: &crate::pipeline::probe::PadRef,
    probe_registry: &ProbeRegistry,
    bus: &BusHandle,
) {
    match probe_registry.invoke_event(src_pad, event, false) {
        ProbeReturn::Drop | ProbeReturn::Handled => return,
        _ => {}
    }

    let result = element.handle_upstream_event(event);
    let Event::Seek(seek) = event else {
        return;
    };

    match result {
        EventResult::Handled => {
            // Src-pad probes see each emitted event, mirroring how buffers are
            // probed on the pad that emits them; Drop/Handled suppresses it.
            let emit = |event: Event| match probe_registry.invoke_event(src_pad, &event, true) {
                ProbeReturn::Drop | ProbeReturn::Handled => None,
                _ => Some(event),
            };
            if seek.flags.contains(crate::event::SeekFlags::FLUSH) {
                if let Some(ev) = emit(Event::FlushStart) {
                    broadcast_event(outputs, &ev).await;
                }
                if let Some(ev) = emit(Event::FlushStop(FlushStopEvent::new(true))) {
                    broadcast_event(outputs, &ev).await;
                }
            }

            // Segment start = the requested target. `base` stays 0: nothing
            // downstream consumes running time from segments yet (#71 will).
            let target = match seek.start.seek_type {
                SeekType::Set => seek.start.position.max(0),
                _ => 0,
            };
            let segment = match seek.format {
                crate::event::SegmentFormat::Bytes => SegmentEvent::new_bytes(target as u64, None),
                _ => SegmentEvent::new_time(ClockTime::from_nanos(target as u64), None),
            };
            if let Some(ev) = emit(Event::Segment(segment)) {
                broadcast_event(outputs, &ev).await;
            }

            bus.post(crate::pipeline::bus::MessageKind::SeekDone {
                position: ClockTime::from_nanos(target as u64),
            });
            tracing::info!("source '{name}': seek handled, segment starts at {target}");
        }
        EventResult::NotHandled => {
            bus.post_warning(format!("source '{name}' cannot seek; seek ignored"), None);
        }
        EventResult::Error => {
            bus.post_warning(format!("source '{name}': seek failed"), None);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_source_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    outputs: Vec<OutputBranch>,
    output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    stop: Arc<AtomicBool>,
    control_rx: AsyncReceiver<Event>,
    pause_rx: watch::Receiver<bool>,
    bus: BusHandle,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    tokio::spawn(reporting(share, async move {
        tracing::debug!("source '{}' started", name);
        events.send_node_started(&name);

        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;
        let mut would_block_count: u64 = 0;

        loop {
            // Cooperative stop (PipelineHandle::stop/abort): end the loop
            // like a natural EOS so live sources release their device and
            // downstream drains cleanly.
            if stop.load(Ordering::Acquire) {
                tracing::info!("source '{}': stopped after {} buffers", name, count);
                broadcast_eos(&outputs).await;
                for bridge in &output_bridges {
                    bridge.signal_eos();
                }
                break;
            }

            // Drain control events (seek) between produce calls. Polled rather
            // than select!ed against `process_source`: cancelling a produce
            // future mid-await could lose the buffer it was about to return.
            // The cost is that a source blocked inside `process_source` sees
            // the event only when that call returns — same caveat as `stop`.
            while let Ok(Some(event)) = control_rx.try_recv() {
                handle_source_control_event(
                    &name,
                    &mut element,
                    &event,
                    &outputs,
                    &src_pad,
                    &probe_registry,
                    &bus,
                )
                .await;
            }

            // Runtime pause (PipelineHandle::pause): stop producing until
            // resumed. Control events still drain so a seek can land while
            // paused, and stop still wins.
            while *pause_rx.borrow() && !stop.load(Ordering::Acquire) {
                while let Ok(Some(event)) = control_rx.try_recv() {
                    handle_source_control_event(
                        &name,
                        &mut element,
                        &event,
                        &outputs,
                        &src_pad,
                        &probe_registry,
                        &bus,
                    )
                    .await;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            tracing::trace!("source '{}': calling process_source", name);
            match guard(&name, element.process_source()).await {
                Ok(SourceResult::Buffer(buffer)) => {
                    let buffer = *buffer;
                    count += 1;
                    would_block_count = 0; // Reset

                    // Invoke buffer probes
                    match probe_registry.invoke_buffer(&src_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }

                    // Notify tracers
                    tracers.notify_buffer(&name, &buffer);

                    tracing::debug!(
                        "source '{}': produced buffer {} ({} bytes)",
                        name,
                        count,
                        buffer.len()
                    );
                    let connected = broadcast(&outputs, buffer.clone(), &tracers).await;
                    for bridge in &output_bridges {
                        let _ = bridge.push_async(buffer.clone()).await;
                    }
                    if !connected && output_bridges.is_empty() {
                        // Every downstream receiver is gone (a sink failed, or
                        // the pipeline is being torn down). Producing further
                        // buffers would be a busy loop into a closed channel.
                        tracing::info!(
                            "source '{}': downstream is gone after {} buffers, stopping",
                            name,
                            count
                        );
                        break;
                    }
                }
                Ok(SourceResult::WouldBlock) => {
                    would_block_count += 1;
                    if would_block_count == 1 || would_block_count.is_multiple_of(1000) {
                        tracing::debug!(
                            "source '{}': WouldBlock (count: {})",
                            name,
                            would_block_count
                        );
                    }
                    // No data available yet, sleep briefly and retry
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                Ok(SourceResult::Eos) => {
                    tracing::info!("source '{}': EOS after {} buffers", name, count);
                    broadcast_eos(&outputs).await;
                    for bridge in &output_bridges {
                        bridge.signal_eos();
                    }
                    break;
                }
                Err(e) => {
                    tracing::error!("source '{}': error: {}", name, e);
                    events.send_error(e.to_string(), Some(name.clone()));
                    // A failed source will never produce again. Downstream gets
                    // the reason: this used to send a plain EOS, which a sink
                    // could not tell apart from the stream simply ending.
                    let err = StreamError::new(&name, e.to_string());
                    broadcast_error(&outputs, &err).await;
                    for bridge in &output_bridges {
                        bridge.signal_error(err.clone());
                    }
                    return Err(e);
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    }))
}

#[allow(clippy::too_many_arguments)]
fn spawn_sink_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    input_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    position: Arc<AtomicU64>,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    // Advance the shared last-presented-PTS cell. `max` keeps it monotonic
    // against decoder reordering; the FlushStop reset is what lets a backwards
    // seek move it backwards.
    fn present(position: &AtomicU64, pts: ClockTime) {
        let Some(pts) = pts.to_option() else { return };
        let _ = position.fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
            (cur == u64::MAX || pts.nanos() > cur).then_some(pts.nanos())
        });
    }

    tokio::spawn(reporting(share, async move {
        tracing::debug!("sink '{}' started", name);
        events.send_node_started(&name);

        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let mut count: u64 = 0;

        let n_inputs = inputs.len();
        tracing::debug!("sink '{}': {} inputs", name, n_inputs);
        if let Some(rx) = inputs.into_iter().next() {
            // Between FlushStart and FlushStop arriving buffers are stale.
            let mut flushing = false;
            // Standard path: read from kanal channel
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        count += 1;
                        tracing::debug!("sink '{}': received buffer {}", name, count);
                        if flushing {
                            tracers.notify_drop(&name);
                            continue;
                        }
                        match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }
                        tracers.notify_buffer(&name, &buffer);
                        let pts = buffer.metadata().pts;
                        if let Err(e) = guard(&name, element.process(Some(buffer))).await {
                            events.send_error(e.to_string(), Some(name.clone()));
                            return Err(e);
                        }
                        tracers.notify_buffer_processed(&name);
                        present(&position, pts);
                    }
                    Ok(Message::Event(event)) => {
                        match probe_registry.invoke_event(&sink_pad, &event, true) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }
                        match &event {
                            Event::FlushStart => flushing = true,
                            Event::FlushStop(_) => {
                                flushing = false;
                                // Forget the pre-seek position; the Segment
                                // that follows re-anchors it, and `present`'s
                                // max() starts fresh from there.
                                position.store(u64::MAX, Ordering::Release);
                            }
                            Event::Segment(seg)
                                if seg.format == crate::event::SegmentFormat::Time =>
                            {
                                position.store(seg.start.max(0) as u64, Ordering::Release);
                            }
                            _ => {}
                        }
                        // No `flush()` here: a sink's flush *commits* buffered
                        // output (an Mp4FileSink finalizes the file) — the
                        // element decides what a flush-seek means for it.
                        let _ = element.handle_downstream_event(event);
                    }
                    Ok(Message::Eos) => {
                        tracing::debug!("sink '{}': EOS after {}", name, count);
                        // Deliver EOS to the element so sinks with external
                        // consumers (AppSink) can unblock them.
                        let _ = element.handle_downstream_event(crate::event::Event::Eos);
                        break;
                    }
                    Ok(Message::Error(err)) => {
                        // The whole point of #85: the consumer learns the
                        // pipeline *failed*, not that the stream ended.
                        tracing::error!("sink '{}': upstream failed: {}", name, err);
                        let _ = element.handle_downstream_event(crate::event::Event::Error(err));
                        break;
                    }
                    Err(e) => {
                        // Upstream now sends a terminal message before dropping
                        // its sender, so a bare close means the pipeline was
                        // aborted rather than that it ended.
                        tracing::debug!("sink '{}': channel closed after {}: {}", name, count, e);
                        let _ = element.handle_downstream_event(crate::event::Event::Eos);
                        break;
                    }
                }
            }
        } else if let Some(bridge) = input_bridges.into_iter().next() {
            // Bridge path: read from RT→Async bridge. Observability mirrors the
            // channel path above; it used to be missing here entirely.
            loop {
                // Drain all available buffers
                while let Some(buffer) = bridge.try_pop() {
                    count += 1;
                    match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }
                    tracers.notify_buffer(&name, &buffer);
                    let pts = buffer.metadata().pts;
                    let result = guard(&name, element.process(Some(buffer))).await;
                    tracers.notify_buffer_processed(&name);
                    if let Err(e) = result {
                        events.send_error(e.to_string(), Some(name.clone()));
                        return Err(e);
                    }
                    present(&position, pts);
                }
                // Check if we're done (EOS + empty)
                if bridge.is_done() {
                    // Same distinction as the channel path: an RT producer that
                    // failed left a reason on the bridge, and reporting a clean
                    // EOS instead would re-hide it for hybrid pipelines.
                    let event = match bridge.take_error() {
                        Some(err) => {
                            tracing::error!("sink '{}': upstream RT failed: {}", name, err);
                            crate::event::Event::Error(err)
                        }
                        None => {
                            tracing::info!("sink '{}': bridge EOS after {} buffers", name, count);
                            crate::event::Event::Eos
                        }
                    };
                    let _ = element.handle_downstream_event(event);
                    break;
                }
                // Wait for more data or EOS signal
                match bridge.data_eventfd().wait_async().await {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::error!("sink '{}': bridge eventfd error: {}", name, e);
                        break;
                    }
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    }))
}

#[allow(clippy::too_many_arguments)]
fn spawn_transform_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    outputs: Vec<OutputBranch>,
    input_bridges: Vec<Arc<AsyncRtBridge>>,
    output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    mut shed: ShedTracker,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    tokio::spawn(reporting(share, async move {
        tracing::debug!("transform '{}' started", name);
        events.send_node_started(&name);

        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;

        /// Helper to send output buffer to all downstream channels and bridges.
        async fn send_output(
            buffer: Buffer,
            outputs: &[OutputBranch],
            output_bridges: &[Arc<AsyncRtBridge>],
            tracers: &TracerRegistry,
        ) {
            // Move the buffer into the common single-consumer case instead
            // of cloning for it and dropping the original (#142) — the
            // clone was a metadata copy plus six refcount atomics per
            // buffer per element for nothing.
            if output_bridges.is_empty() {
                broadcast(outputs, buffer, tracers).await;
                return;
            }
            broadcast(outputs, buffer.clone(), tracers).await;
            let (last, rest) = output_bridges
                .split_last()
                .expect("output_bridges checked non-empty");
            for bridge in rest {
                let _ = bridge.push_async(buffer.clone()).await;
            }
            let _ = last.push_async(buffer).await;
        }

        /// Helper to send EOS to all downstream channels and bridges.
        async fn send_eos(outputs: &[OutputBranch], output_bridges: &[Arc<AsyncRtBridge>]) {
            broadcast_eos(outputs).await;
            for bridge in output_bridges {
                bridge.signal_eos();
            }
        }

        /// Helper to send a terminal error downstream, in place of EOS.
        async fn send_error(
            outputs: &[OutputBranch],
            output_bridges: &[Arc<AsyncRtBridge>],
            error: &StreamError,
        ) {
            broadcast_error(outputs, error).await;
            for bridge in output_bridges {
                bridge.signal_error(error.clone());
            }
        }

        if let Some(rx) = inputs.into_iter().next() {
            // Between FlushStart and FlushStop every arriving buffer is stale
            // by definition — drop it instead of processing.
            let mut flushing = false;
            // Standard path: read from kanal channel
            loop {
                tracing::trace!("transform '{}': waiting for input", name);
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        count += 1;
                        tracing::debug!(
                            "transform '{}': received buffer {} ({} bytes)",
                            name,
                            count,
                            buffer.len()
                        );

                        if flushing {
                            tracers.notify_drop(&name);
                            continue;
                        }

                        // Sink-pad probes see the input before the element does.
                        match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }

                        // LatencyTracer pairs notify_buffer with
                        // notify_buffer_processed, so the second call must land
                        // BEFORE send_output — otherwise a downstream
                        // send().await sits between them and back-pressure gets
                        // billed as this element's processing time.
                        tracers.notify_buffer(&name, &buffer);
                        let result = guard(&name, element.process(Some(buffer))).await;
                        tracers.notify_buffer_processed(&name);

                        match result {
                            Ok(Some(out)) => {
                                tracing::debug!(
                                    "transform '{}': produced output ({} bytes)",
                                    name,
                                    out.len()
                                );
                                match probe_registry.invoke_buffer(&src_pad, &out) {
                                    ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                    _ => {}
                                }
                                shed.reset();
                                send_output(out, &outputs, &output_bridges, &tracers).await;
                            }
                            Ok(None) => {
                                tracing::debug!(
                                    "transform '{}': no output for buffer {}",
                                    name,
                                    count
                                );
                                shed.reset();
                            }
                            // Arena exhaustion is flow control, not failure: the
                            // element had nowhere to put this buffer because
                            // downstream is still holding the last ones. Shed it
                            // and keep the pipeline alive.
                            Err(Error::PoolExhausted) => {
                                shed.record(&name, &tracers)?;
                            }
                            Err(e) => {
                                tracing::error!("transform '{}': error: {}", name, e);
                                events.send_error(e.to_string(), Some(name.clone()));
                                // A failed transform stops processing. Downstream
                                // gets the *reason*, not a bare EOS that would
                                // read as a clean end of stream.
                                let err = StreamError::new(&name, e.to_string());
                                send_error(&outputs, &output_bridges, &err).await;
                                return Err(e);
                            }
                        }
                    }
                    Ok(Message::Event(event)) => {
                        match probe_registry.invoke_event(&sink_pad, &event, true) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }
                        match &event {
                            Event::FlushStart => {
                                flushing = true;
                                // Drain the element's internal state and
                                // *discard* it: pending reordered frames from
                                // before the seek must not surface after it.
                                if let Err(e) = guard(&name, element.flush()).await {
                                    tracing::warn!("flush error in '{}': {}", name, e);
                                }
                                shed.reset();
                            }
                            Event::FlushStop(_) => flushing = false,
                            _ => {}
                        }
                        // The element sees every event and decides what goes
                        // on: `Some` forwards (the default), `None` consumes.
                        if let Some(fwd) = element.handle_downstream_event(event) {
                            match probe_registry.invoke_event(&src_pad, &fwd, true) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            broadcast_event(&outputs, &fwd).await;
                        }
                    }
                    Ok(Message::Error(err)) => {
                        // Terminal: pass the reason on rather than flushing and
                        // reporting a clean end.
                        tracing::error!("transform '{}': upstream failed: {}", name, err);
                        send_error(&outputs, &output_bridges, &err).await;
                        break;
                    }
                    Ok(Message::Eos) => {
                        tracing::info!(
                            "transform '{}': received EOS after {} buffers, flushing",
                            name,
                            count
                        );
                        // Flush any buffered data before propagating EOS
                        match guard(&name, element.flush()).await {
                            Ok(output) => {
                                let buffers = match output {
                                    Output::None => vec![],
                                    Output::Single(b) => vec![b],
                                    Output::Multiple(v) => v,
                                };
                                tracing::info!(
                                    "transform '{}': flush produced {} buffers",
                                    name,
                                    buffers.len()
                                );
                                for buffer in buffers {
                                    send_output(buffer, &outputs, &output_bridges, &tracers).await;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("flush error in '{}': {}", name, e);
                            }
                        }
                        send_eos(&outputs, &output_bridges).await;
                        break;
                    }
                    Err(_) => {
                        send_eos(&outputs, &output_bridges).await;
                        break;
                    }
                }
            }
        } else if let Some(bridge) = input_bridges.into_iter().next() {
            // Bridge path: read from RT→Async bridge.
            //
            // Observability here mirrors the channel path above, including the
            // notify_buffer_processed-before-send_output ordering. It used to
            // be absent entirely, so putting an element on an RT thread — the
            // elements most worth measuring — made it invisible to probes and
            // to LatencyTracer.
            loop {
                while let Some(buffer) = bridge.try_pop() {
                    count += 1;

                    match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }

                    tracers.notify_buffer(&name, &buffer);
                    let result = guard(&name, element.process(Some(buffer))).await;
                    tracers.notify_buffer_processed(&name);

                    match result {
                        Ok(Some(out)) => {
                            match probe_registry.invoke_buffer(&src_pad, &out) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            shed.reset();
                            send_output(out, &outputs, &output_bridges, &tracers).await;
                        }
                        Ok(None) => shed.reset(),
                        // Shed rather than die — see the channel path above.
                        Err(Error::PoolExhausted) => {
                            shed.record(&name, &tracers)?;
                        }
                        Err(e) => {
                            events.send_error(e.to_string(), Some(name.clone()));
                            // This arm used to return without telling anyone
                            // downstream, so every sink below it hung.
                            let err = StreamError::new(&name, e.to_string());
                            send_error(&outputs, &output_bridges, &err).await;
                            return Err(e);
                        }
                    }
                }
                if bridge.is_done() {
                    // Flush
                    match guard(&name, element.flush()).await {
                        Ok(output) => {
                            let buffers = match output {
                                Output::None => vec![],
                                Output::Single(b) => vec![b],
                                Output::Multiple(v) => v,
                            };
                            for buffer in buffers {
                                send_output(buffer, &outputs, &output_bridges, &tracers).await;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("flush error in '{}': {}", name, e);
                        }
                    }
                    match bridge.take_error() {
                        Some(err) => send_error(&outputs, &output_bridges, &err).await,
                        None => send_eos(&outputs, &output_bridges).await,
                    }
                    break;
                }
                match bridge.data_eventfd().wait_async().await {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::error!("transform '{}': bridge eventfd error: {}", name, e);
                        break;
                    }
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    }))
}

#[allow(clippy::too_many_arguments)]
/// Deliver one routed demuxer buffer to its pad's links — and only those.
///
/// This is the #76 fix: buffers used to be broadcast to *every* pad, so a
/// two-branch A/V demuxer sent audio down the video branch and vice versa.
/// A buffer routed to a pad with no links is dropped (rate-limited warn); an
/// empty pad name is the legacy escape hatch and broadcasts to all pads.
async fn route_demux_buffer(
    name: &str,
    pad: &str,
    buffer: Buffer,
    outputs_by_pad: &HashMap<String, Vec<OutputBranch>>,
    tracers: &TracerRegistry,
    unrouted: &mut u64,
) {
    match outputs_by_pad.get(pad) {
        Some(branches) => {
            broadcast(branches, buffer, tracers).await;
        }
        None if pad.is_empty() => {
            for branches in outputs_by_pad.values() {
                broadcast(branches, buffer.clone(), tracers).await;
            }
        }
        None => {
            tracers.notify_drop(name);
            *unrouted += 1;
            if is_power_of_ten(*unrouted) {
                tracing::warn!(
                    "demuxer '{name}': no links on pad '{pad}', dropping its buffers \
                     ({unrouted} so far) — link it with link_pads(demux, \"{pad}\", ..)"
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_demuxer_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    outputs_by_pad: HashMap<String, Vec<OutputBranch>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    stop: Arc<AtomicBool>,
    control_rx: Option<AsyncReceiver<Event>>,
    bus: BusHandle,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    tokio::spawn(reporting(share, async move {
        tracing::debug!("demuxer '{}' started", name);
        events.send_node_started(&name);

        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;
        let mut unrouted: u64 = 0;
        // Control events (seek) broadcast their flush sequence to every pad.
        let all_branches: Vec<OutputBranch> = outputs_by_pad.values().flatten().cloned().collect();

        if let Some(rx) = inputs.into_iter().next() {
            // Between FlushStart and FlushStop arriving buffers are stale.
            let mut flushing = false;
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        count += 1;

                        if flushing {
                            tracers.notify_drop(&name);
                            continue;
                        }

                        match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }

                        // Same ordering rule as spawn_transform_task: the
                        // processed notification must land before the
                        // broadcast, or downstream back-pressure is billed as
                        // demux time.
                        tracers.notify_buffer(&name, &buffer);
                        let result = guard(&name, element.process_demux(Some(buffer))).await;
                        tracers.notify_buffer_processed(&name);

                        match result {
                            Ok(DemuxResult::Routed(routed)) => {
                                for (pad, out) in routed {
                                    match probe_registry.invoke_buffer(&src_pad, &out) {
                                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                        _ => {}
                                    }
                                    route_demux_buffer(
                                        &name,
                                        &pad,
                                        out,
                                        &outputs_by_pad,
                                        &tracers,
                                        &mut unrouted,
                                    )
                                    .await;
                                }
                            }
                            // Input-driven demuxers do not signal these; treat
                            // them as "nothing to emit" rather than inventing
                            // an early end of stream.
                            Ok(DemuxResult::WouldBlock | DemuxResult::Eos) => {}
                            Err(e) => {
                                events.send_error(e.to_string(), Some(name.clone()));
                                // Used to return without telling any of the
                                // output pads, so every branch below hung.
                                let err = StreamError::new(&name, e.to_string());
                                for branches in outputs_by_pad.values() {
                                    broadcast_error(branches, &err).await;
                                }
                                return Err(e);
                            }
                        }
                    }
                    Ok(Message::Event(event)) => {
                        match probe_registry.invoke_event(&sink_pad, &event, true) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }
                        match &event {
                            Event::FlushStart => {
                                flushing = true;
                                if let Err(e) = guard(&name, element.flush()).await {
                                    tracing::warn!("flush error in '{}': {}", name, e);
                                }
                            }
                            Event::FlushStop(_) => flushing = false,
                            _ => {}
                        }
                        // Events go to every output pad, like EOS does.
                        if let Some(fwd) = element.handle_downstream_event(event) {
                            match probe_registry.invoke_event(&src_pad, &fwd, true) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            for branches in outputs_by_pad.values() {
                                broadcast_event(branches, &fwd).await;
                            }
                        }
                    }
                    Ok(Message::Error(err)) => {
                        tracing::error!("demuxer '{}': upstream failed: {}", name, err);
                        for branches in outputs_by_pad.values() {
                            broadcast_error(branches, &err).await;
                        }
                        break;
                    }
                    Ok(Message::Eos) | Err(_) => {
                        // Drain the demuxer through the routed path
                        // (process_demux(None) → Demuxer::produce), so a
                        // trailing partial frame keeps its pad — the old
                        // flush() broadcast sent it down every branch.
                        loop {
                            match guard(&name, element.process_demux(None)).await {
                                Ok(DemuxResult::Routed(routed)) if !routed.is_empty() => {
                                    for (pad, out) in routed {
                                        route_demux_buffer(
                                            &name,
                                            &pad,
                                            out,
                                            &outputs_by_pad,
                                            &tracers,
                                            &mut unrouted,
                                        )
                                        .await;
                                    }
                                }
                                Ok(_) => break,
                                Err(e) => {
                                    tracing::warn!("EOS drain error in '{}': {}", name, e);
                                    break;
                                }
                            }
                        }
                        for branches in outputs_by_pad.values() {
                            broadcast_eos(branches).await;
                        }
                        break;
                    }
                }
            }
        } else {
            // Source-style demuxer: no input links, the element owns its
            // reader (Demuxer::produce). Drive it like a source, routing
            // each produced buffer to its pad's links.
            loop {
                // Cooperative stop, exactly like spawn_source_task.
                if stop.load(Ordering::Acquire) {
                    tracing::info!("demuxer '{}': stopped after {} buffers", name, count);
                    for branches in outputs_by_pad.values() {
                        broadcast_eos(branches).await;
                    }
                    break;
                }

                // Runtime control (seek), drained between produce calls —
                // same polling contract as spawn_source_task.
                if let Some(control_rx) = &control_rx {
                    while let Ok(Some(event)) = control_rx.try_recv() {
                        handle_source_control_event(
                            &name,
                            &mut element,
                            &event,
                            &all_branches,
                            &src_pad,
                            &probe_registry,
                            &bus,
                        )
                        .await;
                    }
                }

                match guard(&name, element.process_demux(None)).await {
                    Ok(DemuxResult::Routed(routed)) => {
                        for (pad, out) in routed {
                            count += 1;
                            match probe_registry.invoke_buffer(&src_pad, &out) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            tracers.notify_buffer(&name, &out);
                            route_demux_buffer(
                                &name,
                                &pad,
                                out,
                                &outputs_by_pad,
                                &tracers,
                                &mut unrouted,
                            )
                            .await;
                        }
                    }
                    Ok(DemuxResult::WouldBlock) => {
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    }
                    // Arena exhaustion is flow control here exactly as it is
                    // for transforms: downstream still holds every slot. A
                    // source-style demuxer has nothing to shed but time.
                    Err(Error::PoolExhausted) => {
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    }
                    Ok(DemuxResult::Eos) => {
                        tracing::info!("demuxer '{}': EOS after {} buffers", name, count);
                        for branches in outputs_by_pad.values() {
                            broadcast_eos(branches).await;
                        }
                        break;
                    }
                    Err(e) => {
                        tracing::error!("demuxer '{}': error: {}", name, e);
                        events.send_error(e.to_string(), Some(name.clone()));
                        let err = StreamError::new(&name, e.to_string());
                        for branches in outputs_by_pad.values() {
                            broadcast_error(branches, &err).await;
                        }
                        return Err(e);
                    }
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    }))
}

#[allow(clippy::too_many_arguments)]
fn spawn_muxer_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    inputs_by_pad: HashMap<String, Vec<AsyncReceiver<Message>>>,
    outputs: Vec<OutputBranch>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    use futures::stream::{FuturesUnordered, StreamExt};

    tokio::spawn(reporting(share, async move {
        tracing::debug!("muxer '{}' started", name);
        events.send_node_started(&name);

        // A muxer has several sink pads, but probes are registered per node
        // rather than per named pad here, so all inputs share one PadRef —
        // consistent with how a transform's single sink pad is handled.
        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;

        // One pending receive per input, re-armed after each message.
        //
        // `FuturesUnordered` drops a future once it resolves, so the previous
        // version — which collected one `rx.recv()` future per input and never
        // pushed another — delivered exactly *one* buffer per input pad and
        // then fell out of the loop. A two-input muxer muxed two buffers,
        // whatever the stream length.
        async fn recv_one(
            pad: String,
            rx: AsyncReceiver<Message>,
        ) -> (String, AsyncReceiver<Message>, Option<Message>) {
            let msg = rx.recv().await.ok();
            (pad, rx, msg)
        }

        let mut receivers: FuturesUnordered<_> = inputs_by_pad
            .into_iter()
            .flat_map(|(pad, rxs)| {
                rxs.into_iter()
                    .map(move |rx| recv_one(pad.clone(), rx))
                    .collect::<Vec<_>>()
            })
            .collect();

        let total = receivers.len();
        let mut eos_count = 0;
        // Flushes are counted across inputs like EOS: FlushStart forwards on
        // the first arrival, FlushStop once every started flush has stopped —
        // forwarding each input's pair verbatim would flush the output twice.
        let mut active_flushes: usize = 0;

        while let Some((pad, rx, msg)) = receivers.next().await {
            // Keep listening on this input unless it just ended; the EOS and
            // error arms deliberately let it drop.
            if matches!(msg, Some(Message::Buffer(_) | Message::Event(_))) {
                receivers.push(recv_one(pad, rx));
            }
            let msg = match msg {
                Some(m) => Ok(m),
                None => Err(()),
            };
            match msg {
                Ok(Message::Buffer(buffer)) => {
                    count += 1;

                    if active_flushes > 0 {
                        tracers.notify_drop(&name);
                        continue;
                    }

                    match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }

                    // Muxers buffer and interleave, so most inputs produce no
                    // output — LatencyTracer sees the pair regardless, which is
                    // what makes "how long is this muxer taking" answerable at
                    // all. It previously had no instrumentation whatsoever.
                    tracers.notify_buffer(&name, &buffer);
                    let result = guard(&name, element.process(Some(buffer))).await;
                    tracers.notify_buffer_processed(&name);

                    match result {
                        Ok(Some(out)) => {
                            match probe_registry.invoke_buffer(&src_pad, &out) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            broadcast(&outputs, out, &tracers).await;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            events.send_error(e.to_string(), Some(name.clone()));
                            // Used to return without telling the output, so the
                            // sink below hung.
                            let err = StreamError::new(&name, e.to_string());
                            broadcast_error(&outputs, &err).await;
                            return Err(e);
                        }
                    }
                }
                Ok(Message::Event(event)) => {
                    match probe_registry.invoke_event(&sink_pad, &event, true) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }
                    let forward = match &event {
                        Event::FlushStart => {
                            active_flushes += 1;
                            active_flushes == 1
                        }
                        Event::FlushStop(_) => {
                            active_flushes = active_flushes.saturating_sub(1);
                            active_flushes == 0
                        }
                        _ => true,
                    };
                    match element.handle_downstream_event(event) {
                        Some(fwd) if forward => {
                            match probe_registry.invoke_event(&src_pad, &fwd, true) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            broadcast_event(&outputs, &fwd).await;
                        }
                        _ => {}
                    }
                }
                Ok(Message::Error(err)) => {
                    // One failed input dooms the muxed stream: the output would
                    // be missing a track from here on. Pass the reason on
                    // immediately rather than waiting for the other inputs.
                    tracing::error!("muxer '{}': an input failed: {}", name, err);
                    broadcast_error(&outputs, &err).await;
                    break;
                }
                Ok(Message::Eos) | Err(_) => {
                    eos_count += 1;
                    if eos_count >= total {
                        // Flush any remaining data from final processing
                        if let Ok(Some(out)) = guard(&name, element.process(None)).await {
                            broadcast(&outputs, out, &tracers).await;
                        }
                        // Flush any buffered data before propagating EOS
                        match guard(&name, element.flush()).await {
                            Ok(output) => {
                                let buffers = match output {
                                    Output::None => vec![],
                                    Output::Single(b) => vec![b],
                                    Output::Multiple(v) => v,
                                };
                                for buffer in buffers {
                                    broadcast(&outputs, buffer, &tracers).await;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("flush error in '{}': {}", name, e);
                            }
                        }
                        broadcast_eos(&outputs).await;
                        break;
                    }
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    }))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::element::{
        ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
        SourceAdapter,
    };
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(64, 64).unwrap())
    }

    struct CountingSource {
        count: u64,
        max: u64,
    }

    impl Source for CountingSource {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
            if self.count >= self.max {
                return Ok(ProduceResult::Eos);
            }
            let arena = test_arena();
            let slot = arena.acquire().unwrap();
            let handle = MemoryHandle::new(slot);
            let buffer = Buffer::new(handle, Metadata::from_sequence(self.count));
            self.count += 1;
            Ok(ProduceResult::OwnBuffer(buffer))
        }
    }

    struct CountingSink {
        received: Arc<AtomicU64>,
    }

    impl Sink for CountingSink {
        fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
            self.received.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn test_executor_config_defaults() {
        let config = ExecutorConfig::default();
        assert_eq!(config.scheduling, SchedulingMode::Async);
        assert_eq!(config.channel_capacity, 16);
        assert!(config.driver.is_none());
    }

    #[test]
    fn test_executor_config_presets() {
        let config = ExecutorConfig::hybrid();
        assert_eq!(config.scheduling, SchedulingMode::Hybrid);

        let config = ExecutorConfig::low_latency_audio();
        assert_eq!(config.scheduling, SchedulingMode::Hybrid);
        assert!(config.driver.is_some());

        let config = ExecutorConfig::video(60);
        assert!(config.driver.is_some());
    }

    #[tokio::test]
    async fn test_unified_executor() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 5 })),
        );
        let received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
                received: received.clone(),
            })),
        );
        pipeline.link(src, sink).unwrap();

        let executor = Executor::new();
        executor.run(&mut pipeline).await.unwrap();

        assert_eq!(received.load(Ordering::Relaxed), 5);
        assert_eq!(pipeline.state(), PipelineState::Running);
    }

    #[tokio::test]
    async fn test_unified_executor_with_config() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 3 })),
        );
        let received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
                received: received.clone(),
            })),
        );
        pipeline.link(src, sink).unwrap();

        let config = ExecutorConfig::default().with_channel_capacity(32);
        let executor = Executor::with_config(config);
        executor.run(&mut pipeline).await.unwrap();

        assert_eq!(received.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_determine_element_strategy_defaults() {
        let hints = ExecutionHints::default();

        // Default hints -> Async
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_determine_element_strategy_rt_safe() {
        // RT-safe profile (rt_safe + low latency) -> RealTime
        let hints = ExecutionHints::rt_safe();
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::RealTime);

        // RT-safe but normal latency -> Async (RT needs low latency)
        let hints = ExecutionHints::default().with_rt_safe(true);
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_determine_element_strategy_low_latency() {
        // Low latency + RT-safe -> RealTime
        let hints = ExecutionHints::low_latency().with_rt_safe(true);
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::RealTime);

        // Low latency but NOT RT-safe -> Async
        let hints = ExecutionHints::low_latency().with_rt_safe(false);
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_determine_element_strategy_io_bound() {
        let hints = ExecutionHints::io_bound();

        // I/O-bound -> always Async
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_execution_plan_analysis() {
        let mut pipeline = Pipeline::new();

        // Add a simple source and sink
        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 5 })),
        );
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
                received: Arc::new(AtomicU64::new(0)),
            })),
        );
        pipeline.link(src, sink).unwrap();

        // Analyze the pipeline
        let plan = analyze_pipeline(&pipeline);

        // Default elements should be async
        assert!(!plan.needs_rt);
        assert_eq!(plan.async_nodes.len(), 2);
        assert_eq!(plan.rt_nodes.len(), 0);
    }

    #[test]
    fn test_executor_config_auto_strategy() {
        // Default config should have auto_strategy enabled
        let config = ExecutorConfig::default();
        assert!(config.auto_strategy);

        // Preset configs should disable auto_strategy
        let config = ExecutorConfig::async_only();
        assert!(!config.auto_strategy);

        let config = ExecutorConfig::hybrid();
        assert!(!config.auto_strategy);

        // without_auto_strategy should disable it
        let config = ExecutorConfig::default().without_auto_strategy();
        assert!(!config.auto_strategy);
    }

    // ---- ShedTracker ------------------------------------------------------

    #[test]
    fn the_warning_ladder_is_powers_of_ten() {
        let on: Vec<u64> = (0..=1000).filter(|&n| is_power_of_ten(n)).collect();
        assert_eq!(on, vec![1, 10, 100, 1000]);
        assert!(!is_power_of_ten(0), "zero sheds is not a shed");
    }

    #[test]
    fn shedding_is_not_fatal_by_default() {
        let tracers = TracerRegistry::new();
        let mut shed = ShedTracker::new(None);
        for _ in 0..10_000 {
            shed.record("t", &tracers)
                .expect("shedding must never fail by default");
        }
        assert_eq!(shed.total, 10_000);
    }

    #[test]
    fn shed_fatal_after_fires_on_consecutive_sheds_only() {
        let tracers = TracerRegistry::new();
        let mut shed = ShedTracker::new(Some(3));

        assert!(shed.record("t", &tracers).is_ok());
        assert!(shed.record("t", &tracers).is_ok());
        // A buffer got through: the burst is over and the count restarts.
        shed.reset();

        assert!(shed.record("t", &tracers).is_ok());
        assert!(shed.record("t", &tracers).is_ok());
        let err = shed
            .record("t", &tracers)
            .expect_err("3 in a row should trip the limit");
        assert!(format!("{err}").contains("shed"), "unhelpful: {err}");
        assert_eq!(
            shed.total, 5,
            "the total counts every shed, not just the run"
        );
    }

    // ---- output_slot_budget ----------------------------------------------

    /// A source, and `branches` sinks hanging off its single src pad with the
    /// given per-link capacities.
    fn fan_out(capacities: &[Option<usize>]) -> (Pipeline, NodeId) {
        let mut pipeline = Pipeline::new();
        let src = pipeline.add_source("src", CountingSource { count: 0, max: 0 });
        for (i, capacity) in capacities.iter().enumerate() {
            let sink = pipeline.add_sink(
                format!("sink{i}"),
                CountingSink {
                    received: Arc::new(AtomicU64::new(0)),
                },
            );
            pipeline
                .link_pads_full(src, "src", sink, "sink", LinkPolicy::Block, *capacity)
                .unwrap();
        }
        (pipeline, src)
    }

    #[test]
    fn a_sink_gets_no_downstream_capacity() {
        let (pipeline, src) = fan_out(&[None]);
        let executor = Executor::new();
        let sink = pipeline.node_ids().into_iter().find(|&n| n != src).unwrap();

        let budget = executor.output_slot_budget(&pipeline, sink, 0);
        assert_eq!(budget.downstream_capacity, 0);
        assert_eq!(budget.in_flight_margin, defaults::IN_FLIGHT_MARGIN);
    }

    #[test]
    fn one_link_takes_the_configured_channel_capacity() {
        let (pipeline, src) = fan_out(&[None]);
        let executor = Executor::with_config(ExecutorConfig::default().with_channel_capacity(64));

        assert_eq!(
            executor.output_slot_budget(&pipeline, src, 0).slots(),
            64 + defaults::IN_FLIGHT_MARGIN
        );
    }

    #[test]
    fn a_per_link_capacity_override_is_honoured() {
        let (pipeline, src) = fan_out(&[Some(128)]);
        let executor = Executor::new();

        assert_eq!(
            executor.output_slot_budget(&pipeline, src, 0).slots(),
            128 + defaults::IN_FLIGHT_MARGIN
        );
    }

    #[test]
    fn fan_out_on_one_pad_takes_the_max_not_the_sum() {
        // Three branches off the same src pad share each buffer — a clone is a
        // refcount bump on the same slot — so the deepest link bounds the pad.
        // Summing would ask for 8 + 64 + 16 = 88 slots for 64 slots of work.
        let (pipeline, src) = fan_out(&[Some(8), Some(64), Some(16)]);
        let executor = Executor::new();

        let budget = executor.output_slot_budget(&pipeline, src, 0);
        assert_eq!(budget.downstream_capacity, 64);
        assert_eq!(budget.slots(), 64 + defaults::IN_FLIGHT_MARGIN);
    }

    #[test]
    fn a_bridged_edge_adds_the_bridge_capacity() {
        let (pipeline, src) = fan_out(&[Some(8)]);
        let executor = Executor::new();
        let bridge_capacity = executor.config.rt.bridge_capacity;

        let budget = executor.output_slot_budget(&pipeline, src, 1);
        assert_eq!(budget.downstream_capacity, 8 + bridge_capacity);
    }

    #[test]
    fn a_deep_link_outgrows_the_codec_floor() {
        // The point of the whole exercise: the arena must track the channel, so
        // a consumer holding a full channel cannot starve the producer.
        let (pipeline, src) = fan_out(&[None]);
        let executor = Executor::with_config(ExecutorConfig::default().with_channel_capacity(256));

        let slots = executor
            .output_slot_budget(&pipeline, src, 0)
            .resolve(defaults::MIN_OUTPUT_SLOT_COUNT, 4096);
        assert!(
            slots > 256,
            "{slots} slots cannot outlive a 256-deep channel"
        );
    }
}
