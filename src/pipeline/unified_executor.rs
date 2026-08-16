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
use crate::event::{
    Event, EventResult, FlushStopEvent, SeekEvent, SeekType, SegmentEvent, SegmentFormat,
    StreamStartEvent,
};
use crate::memory::{OutputBudget, defaults};
use crate::pipeline::bus::{Bus, BusHandle};
use crate::pipeline::flow::{LinkFlowMonitor, WaterMarks};
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
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::task::{Context, Poll};
use tokio::sync::mpsc::error::{TryRecvError, TrySendError};
use tokio::sync::mpsc::{Receiver as MsgReceiver, Sender as MsgSender, channel as message_channel};

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
/// the arms disagree about what they report — the `shed_fatal_after` paths
/// (transform and sink) return `Err` without telling anyone at all — and a new
/// arm added later would
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
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
    /// Whether the element declared seek support, snapshotted in
    /// `spawn_element_task` before the element moved into its task. Gates
    /// `PipelineHandle::seek*`: seeks are only dispatched to elements that
    /// declared support (GStreamer's `GST_QUERY_SEEKING` discipline).
    seekable: bool,
    /// The element's duration answer at start (`None` = unknown/live).
    duration: Option<crate::pipeline::seek::DurationQuery>,
}

/// Runtime-control state assembled while tasks spawn, then moved onto the
/// [`PipelineHandle`]: the source control channels, the pause gate the source
/// loops watch, and the last-presented-PTS cell the sink loops write.
struct RuntimeControls {
    controls: Vec<SourceControl>,
    /// Sink-node inboxes (#163): where `PipelineHandle::seek` and
    /// `send_event_upstream` enter the graph in fully-async pipelines.
    /// Empty in hybrid mode — dispatch falls back to the source controls.
    upstream_entries: Vec<(String, tokio::sync::mpsc::UnboundedSender<Event>)>,
    pause_rx: watch::Receiver<bool>,
    position: Arc<AtomicU64>,
    /// Seek format conversions declared by mid-graph elements (#163), from
    /// every node as it spawns — a fed demuxer is not a source and so gets
    /// no `SourceControl` to hang this on.
    translations: Vec<crate::pipeline::seek::SeekTranslation>,
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
    /// Sink-node inboxes (#163): the upstream-event entry points in a
    /// fully-async pipeline. Empty in hybrid mode (legacy source fan-out).
    upstream_entries: Vec<(String, tokio::sync::mpsc::UnboundedSender<Event>)>,
    /// Seek format conversions declared by mid-graph elements, snapshotted
    /// at start; see [`query_seekable`](Self::query_seekable).
    translations: Vec<crate::pipeline::seek::SeekTranslation>,
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
    /// Aggregate declared latency (#184), snapshotted at start; `None`
    /// when no element declared one.
    latency: Option<crate::pipeline::seek::LatencyRange>,
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

    /// Send an upstream event to every running source task, **ungated**.
    ///
    /// This is the runtime counterpart of [`Pipeline::send_event_upstream`],
    /// which only works before `start()` moves the elements into their tasks.
    /// The event is handled asynchronously by each source's produce loop.
    ///
    /// Unlike [`seek`](Self::seek) this fans out to *every* source,
    /// seekable or not — the escape hatch for custom upstream events a
    /// source may handle regardless of seekability. A `Seek` sent this way
    /// bypasses the seekability gate and falls back to the per-source
    /// "cannot seek" bus warning.
    ///
    /// Returns `false` if no source accepted the event — the pipeline has no
    /// async-spawned sources (RT-scheduled sources carry no control channel),
    /// or every source task has already finished.
    pub async fn send_event_upstream(&self, event: Event) -> bool {
        // #163: enter at the sinks and travel hop-by-hop toward the sources
        // (mid-graph elements get their chance to handle or translate);
        // hybrid pipelines keep the legacy direct source fan-out.
        if !self.upstream_entries.is_empty() {
            let mut delivered = false;
            for (name, tx) in &self.upstream_entries {
                if tx.send(event.clone()).is_ok() {
                    delivered = true;
                } else {
                    tracing::debug!(
                        "sink '{name}' is gone; upstream event '{}' not delivered",
                        event.name()
                    );
                }
            }
            return delivered;
        }
        let mut delivered = false;
        for control in &self.controls {
            if control.tx.send(event.clone()).is_ok() {
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
    /// **Gated on seekability** (GStreamer's discipline: an unhandled seek
    /// does nothing): the seek is dispatched only to sources that declared
    /// `is_seekable()` at start, and if none did this returns `false`
    /// immediately — nothing is dispatched, nothing is flushed, nothing is
    /// posted on the bus. Check [`seekable`](Self::seekable) /
    /// [`query_seekable`](Self::query_seekable) first to know in advance.
    /// Watch the bus for [`MessageKind::SeekDone`](crate::pipeline::bus::MessageKind::SeekDone) to learn when the flush
    /// sequence has run.
    pub async fn seek(&self, seek: SeekEvent) -> bool {
        // The gate is unchanged: no source declared seekability → nothing
        // is dispatched at all.
        if !self.controls.iter().any(|c| c.seekable) {
            return false;
        }
        // #163: dispatch via the sinks so the seek travels hop-by-hop and
        // mid-graph elements can handle it; hybrid pipelines (no sink
        // inboxes) keep the legacy direct fan-out to seekable sources.
        if !self.upstream_entries.is_empty() {
            let mut delivered = false;
            for (_, tx) in &self.upstream_entries {
                if tx.send(Event::Seek(seek.clone())).is_ok() {
                    delivered = true;
                }
            }
            return delivered;
        }
        let mut delivered = false;
        for control in self.controls.iter().filter(|c| c.seekable) {
            if control.tx.send(Event::Seek(seek.clone())).is_ok() {
                delivered = true;
            }
        }
        delivered
    }

    /// Whether any source in the running pipeline declared seek support.
    ///
    /// Snapshotted at start from `is_seekable()` on every async-spawned
    /// source and source-style demuxer (RT-scheduled sources carry no
    /// control channel and count as unseekable).
    pub fn seekable(&self) -> bool {
        self.controls.iter().any(|c| c.seekable)
    }

    /// The seekable range of the running pipeline.
    ///
    /// First seekable source wins, matching `Pipeline::query_seekable`.
    pub fn query_seekable(&self) -> crate::pipeline::seek::SeekableQuery {
        crate::pipeline::seek::aggregate_seekable(
            self.controls
                .iter()
                .map(|c| (c.seekable, c.duration.as_ref())),
            &self.translations,
        )
    }

    /// Total stream duration, as the sources reported it at start.
    ///
    /// The best `SegmentFormat::Time` answer across sources;
    /// [`ClockTime::NONE`] when no source knows (live capture, streamed
    /// WebM without a Segment Duration). This is the runtime counterpart of
    /// the pre-start `Pipeline::query_duration` — applications no longer
    /// need to reach around the framework to a demuxer object.
    pub fn duration(&self) -> ClockTime {
        self.controls
            .iter()
            .filter_map(|c| c.duration.as_ref())
            .filter(|d| d.format == crate::event::SegmentFormat::Time)
            .filter_map(|d| d.duration)
            .max()
            .map(ClockTime::from_nanos)
            .unwrap_or(ClockTime::NONE)
    }

    /// The first duration answer any source gave at start, `Time` preferred.
    pub fn query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        let durations: Vec<_> = self
            .controls
            .iter()
            .filter_map(|c| c.duration.clone())
            .collect();
        durations
            .iter()
            .find(|d| d.format == crate::event::SegmentFormat::Time)
            .cloned()
            .or_else(|| durations.into_iter().next())
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
    /// gates the producer loops (sources and source-style demuxers) *and* the
    /// sink loops. A gated sink delivers [`Event::Pause`] to its element
    /// (`AlsaSink` pauses or silences its device buffer, so audio stops
    /// within a period instead of draining seconds of queued stream) and
    /// holds its first fresh buffer for resume — pause is not flush, nothing
    /// is dropped. Transforms are deliberately not gated: they park on
    /// backpressure, and staying live is what lets a flushing seek propagate
    /// while paused. Posts `StateChanged{Running → Idle}` on the bus.
    ///
    /// Limits: an element blocked *inside* `process`/`process_source`
    /// observes the gate only when that call returns — the same caveat as
    /// [`stop`](Self::stop) — and RT-scheduled nodes see no gate at all.
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
    /// Un-gates the producers and sinks and resumes the clock gap-free:
    /// running time continues from where it froze, so sinks pick up on the
    /// very next frame with no burst of late frames. Each sink delivers
    /// [`Event::Resume`] to its element and replays the buffer it held
    /// during the pause. Posts `StateChanged{Idle → Running}`.
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

    /// The pipeline's aggregate declared latency (#184), snapshotted at
    /// start: each element's declared `latency()` summed along every
    /// source→sink path, worst path reported. `None` when no element
    /// declares one. Also posted once at start as
    /// [`MessageKind::LatencyChanged`](crate::pipeline::bus::MessageKind::LatencyChanged).
    pub fn latency(&self) -> Option<crate::pipeline::seek::LatencyRange> {
        self.latency
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
        // Aggregate declared latency (#184), while the elements are still in
        // the graph. Posted once — the variant is honest now: it appears
        // exactly when a computed value exists.
        let latency = pipeline.query_latency();
        if let Some(l) = latency {
            bus_handle.post(crate::pipeline::bus::MessageKind::LatencyChanged {
                min: l.min,
                max: l.max,
            });
        }
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
            upstream_entries: Vec::new(),
            pause_rx,
            position: Arc::new(AtomicU64::new(u64::MAX)),
            translations: Vec::new(),
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
            upstream_entries: runtime.upstream_entries,
            translations: runtime.translations,
            pause_tx,
            pausable,
            base_time: clock_info.map(|(_, b)| b).unwrap_or(ClockTime::NONE),
            position: runtime.position,
            latency,
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
        use crate::memory::EventFd;
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
                if let Some(memory) = pipeline.node_output_memory_type(node_id) {
                    element.set_negotiated_memory(memory);
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
    /// Materialize one link into a channel pair + optional flow monitor.
    ///
    /// `DropOldest` links get the leaky (evicting) channel; everything else
    /// uses tokio mpsc. This pairing is what `send_buffer`'s dispatch relies
    /// on.
    fn make_link_channel(
        &self,
        link: &crate::pipeline::Link,
    ) -> (BranchTx, BranchRx, Option<Arc<LinkFlowMonitor>>) {
        // `.max(1)`: tokio panics on a zero-capacity channel where
        // kanal made a rendezvous channel, and both
        // `link_pads_full(.., Some(0))`
        // and `with_channel_capacity(0)` are public. Clamping here also
        // keeps `WaterMarks::from_capacity` off a degenerate 0/0 band.
        let capacity = link.capacity.unwrap_or(self.config.channel_capacity).max(1);
        let (tx, rx) = if link.policy == LinkPolicy::DropOldest {
            let (tx, rx) = crate::pipeline::leaky::channel::<Message>(capacity);
            (BranchTx::Leaky(tx), BranchRx::Leaky(rx))
        } else {
            let (tx, rx) = message_channel::<Message>(capacity);
            (BranchTx::Mpsc(tx), BranchRx::Mpsc(rx))
        };
        let flow = link.flow_state.as_ref().map(|state| {
            let marks = link
                .watermarks
                .unwrap_or_else(|| WaterMarks::from_capacity(capacity));
            Arc::new(LinkFlowMonitor::new(marks, state.clone()))
        });
        (tx, rx, flow)
    }

    fn build_channels(&self, pipeline: &Pipeline, node_id: NodeId, network: &mut ChannelNetwork) {
        for (child_id, link) in pipeline.children(node_id) {
            if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                let (tx, rx, flow) = self.make_link_channel(link);
                network.add_channel(
                    node_id,
                    link.src_pad.clone(),
                    child_id,
                    link.sink_pad.clone(),
                    tx,
                    rx,
                    link.policy,
                    node_name(pipeline, child_id),
                    flow,
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
                // Bridge handles this. The bridge is a bare SPSC ring with no
                // policy hook, so a non-default LinkPolicy is silently a Block
                // here — worth a trace when someone asked for lossy.
                if link.policy != LinkPolicy::Block {
                    tracing::debug!(
                        "boundary edge {:?} -> {:?} is bridged; LinkPolicy::{:?} does not apply",
                        node_id,
                        child_id,
                        link.policy
                    );
                }
                continue;
            }

            if async_set.contains(&child_id) {
                if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                    let (tx, rx, flow) = self.make_link_channel(link);
                    network.add_channel(
                        node_id,
                        link.src_pad.clone(),
                        child_id,
                        link.sink_pad.clone(),
                        tx,
                        rx,
                        link.policy,
                        node_name(pipeline, child_id),
                        flow,
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

        // #163: one unbounded upstream inbox per node, created up front so
        // each task can be handed its parents' senders at spawn.
        let mut inbox_tx: HashMap<NodeId, tokio::sync::mpsc::UnboundedSender<Event>> =
            HashMap::new();
        let mut inbox_rx: HashMap<NodeId, tokio::sync::mpsc::UnboundedReceiver<Event>> =
            HashMap::new();
        for &node_id in &node_ids {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
            inbox_tx.insert(node_id, tx);
            inbox_rx.insert(node_id, rx);
        }

        for node_id in node_ids {
            let parents: Vec<(String, tokio::sync::mpsc::UnboundedSender<Event>)> = pipeline
                .parents(node_id)
                .into_iter()
                .filter_map(|(pid, _)| {
                    inbox_tx
                        .get(&pid)
                        .map(|tx| (node_name(pipeline, pid), tx.clone()))
                })
                .collect();
            let upstream = Some((
                inbox_tx[&node_id].clone(),
                UpstreamHop {
                    rx: inbox_rx.remove(&node_id).expect("inbox created above"),
                    parents,
                },
            ));
            let task = self.spawn_node_task(
                pipeline,
                node_id,
                &mut channels,
                upstream,
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

            // Hybrid mode keeps the legacy direct-to-source dispatch: RT
            // segments carry no events, so hop routing cannot traverse them.
            let task = self.spawn_node_task_with_bridges(
                pipeline,
                node_id,
                None,
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
        upstream: Option<(tokio::sync::mpsc::UnboundedSender<Event>, UpstreamHop)>,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
        outcome: &Arc<TerminalOutcome>,
        runtime: &mut RuntimeControls,
    ) -> Result<JoinHandle<Result<()>>> {
        self.spawn_node_task_with_bridges(
            pipeline,
            node_id,
            upstream,
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
        upstream: Option<(tokio::sync::mpsc::UnboundedSender<Event>, UpstreamHop)>,
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

        // #145: what memory the downstream link negotiated, so a
        // dmabuf-capable source only emits dmabuf when the consumer wants
        // it. None (no negotiation ran) leaves the safe CPU default.
        if let Some(memory) = pipeline.node_output_memory_type(node_id) {
            element.set_negotiated_memory(memory);
        }

        // Snapshotted here for the same reason `is_seekable` is: after spawn
        // the element has moved into its task. Collected from every node, not
        // just sources — the point is exactly the mid-graph translator.
        runtime.translations.extend(element.seek_translations());

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
                // The source's upstream inbox doubles as its control
                // channel; hybrid mode (no hop) falls back to a private
                // bounded channel with no parents — identical semantics.
                let (own_tx, hop) = match upstream {
                    Some((tx, hop)) => (tx, hop),
                    None => {
                        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
                        (
                            tx,
                            UpstreamHop {
                                rx,
                                parents: Vec::new(),
                            },
                        )
                    }
                };
                let bus = pipeline.bus_handle().for_element(&node_name);
                // Snapshotted here, while the element is still in hand —
                // after spawn it has moved into its task.
                let seekable = element.is_seekable();
                runtime.controls.push(SourceControl {
                    name: node_name.clone(),
                    tx: own_tx,
                    seekable,
                    duration: element.source_query_duration(),
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
                    hop,
                    runtime.pause_rx.clone(),
                    channels.epoch_cell(node_id),
                    bus,
                    seekable,
                    share,
                )
            }
            ElementType::Sink => {
                // #163: sinks are the upstream-event entry points — register
                // this sink's inbox with the handle.
                let hop = upstream.map(|(own_tx, hop)| {
                    runtime.upstream_entries.push((node_name.clone(), own_tx));
                    hop
                });
                let bus = pipeline.bus_handle().for_element(&node_name);
                spawn_sink_task(
                    node_name,
                    node_id,
                    element,
                    channels.take_inputs(node_id),
                    input_bridges,
                    events_clone,
                    probes,
                    tracers,
                    runtime.position.clone(),
                    channels.epoch_cell(node_id),
                    runtime.pause_rx.clone(),
                    hop,
                    bus,
                    ShedTracker::new(self.config.shed_fatal_after),
                    share,
                )
            }
            ElementType::Transform => {
                let bus = pipeline.bus_handle().for_element(&node_name);
                spawn_transform_task(
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
                    channels.epoch_cell(node_id),
                    upstream.map(|(_, hop)| hop),
                    bus,
                    share,
                )
            }
            ElementType::Demuxer => {
                // Inputs flattened (one sink pad), outputs kept per pad — that
                // is the whole point of a demuxer.
                let inputs = channels.take_inputs(node_id);
                let outputs_by_pad = channels.take_outputs_by_pad(node_id);
                // A source-style demuxer (no input links) is a source in every
                // sense, including runtime control: its inbox is registered as
                // a SourceControl so PipelineHandle::seek reaches
                // Demuxer::handle_upstream_event. A fed demuxer keeps its
                // inbox as a mid-graph hop (#163).
                let source_style = inputs.is_empty();
                let seekable = element.is_seekable();
                let hop = if source_style {
                    let (own_tx, hop) = match upstream {
                        Some((tx, hop)) => (tx, hop),
                        None => {
                            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
                            (
                                tx,
                                UpstreamHop {
                                    rx,
                                    parents: Vec::new(),
                                },
                            )
                        }
                    };
                    runtime.controls.push(SourceControl {
                        name: node_name.clone(),
                        tx: own_tx,
                        seekable,
                        duration: element.source_query_duration(),
                    });
                    Some(hop)
                } else {
                    upstream.map(|(_, hop)| hop)
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
                    hop,
                    channels.epoch_cell(node_id),
                    runtime.pause_rx.clone(),
                    bus,
                    source_style && seekable,
                    share,
                )
            }
            ElementType::Muxer => {
                // Mirror image: inputs per pad, outputs flattened.
                let inputs_by_pad = channels.take_inputs_by_pad(node_id);
                let outputs = channels.take_outputs(node_id);
                let bus = pipeline.bus_handle().for_element(&node_name);
                spawn_muxer_task(
                    node_name,
                    node_id,
                    element,
                    inputs_by_pad,
                    outputs,
                    events_clone,
                    probes,
                    tracers,
                    channels.epoch_cell(node_id),
                    upstream.map(|(_, hop)| hop),
                    bus,
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
    /// A buffer, stamped with its producer's flush epoch at production time.
    ///
    /// The epoch is what makes a flushing seek actually flush (#157): after
    /// the element repositions, the seek handler raises **its own node's**
    /// epoch cell to the seek's epoch (`fetch_max` — idempotent across the
    /// sources handling one seek, #162; per-node since #163 phase C), so
    /// every buffer that producer stamped before the seek tests stale.
    /// Consumers judge staleness against the cell of the branch the buffer
    /// arrived on and fold the observed epoch into their own cell
    /// ([`InputBranch::is_stale`]), so the shed propagates hop-by-hop at
    /// receive speed strictly along the seek's path — the queued backlog
    /// that used to play out as ~2.5 s of pre-seek audio drains in
    /// microseconds, while an innocent sibling chain's in-flight buffers
    /// are left alone. FIFO ordering between buffers and events is
    /// untouched.
    Buffer(Buffer, u64),
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

/// The sending half of a link channel: tokio mpsc for `Block`/`DropNewest`
/// links, the drop-oldest [`leaky`] channel for `DropOldest` ones (tokio's
/// `Sender` cannot evict a queued message, so that policy needs its own
/// primitive). Paired with [`BranchRx`] by construction in `build_channels`.
#[derive(Clone)]
enum BranchTx {
    Mpsc(MsgSender<Message>),
    Leaky(crate::pipeline::leaky::LeakySender<Message>),
}

/// The receiving half matching [`BranchTx`].
enum BranchRx {
    Mpsc(MsgReceiver<Message>),
    Leaky(crate::pipeline::leaky::LeakyReceiver<Message>),
}

impl BranchRx {
    /// Queued messages, for flow-monitor sampling.
    fn len(&self) -> usize {
        match self {
            BranchRx::Mpsc(rx) => rx.len(),
            BranchRx::Leaky(rx) => rx.len(),
        }
    }
}

/// One downstream branch of a src-pad, with the policy of the link that made it.
///
/// The old code flattened a pad's outputs to a bare `Vec<AsyncSender>`, which
/// threw away *which branch is which* — and with it any chance of treating a
/// slow branch differently from a fast one.
#[derive(Clone)]
struct OutputBranch {
    tx: BranchTx,
    policy: LinkPolicy,
    /// Name of the element on the far end, for drop reporting.
    sink_name: String,
    /// Flow monitor for this link (`Pipeline::monitor_link`), fed the
    /// channel occupancy after every send. The receive side samples too —
    /// a gated source stops sending, so sender-only sampling would never
    /// observe the drain that flips the signal back to Ready.
    flow: Option<Arc<LinkFlowMonitor>>,
}

impl OutputBranch {
    /// Queued messages on this link.
    ///
    /// tokio's `Sender` has no `len()`; `max_capacity - capacity` is exact
    /// here because nothing in this file calls `reserve()`, and it can never
    /// exceed the capacity. The leaky sender tracks its length directly.
    #[inline]
    fn occupancy(&self) -> usize {
        match &self.tx {
            BranchTx::Mpsc(tx) => tx.max_capacity() - tx.capacity(),
            BranchTx::Leaky(tx) => tx.len(),
        }
    }

    /// Send a buffer, honouring the link policy.
    ///
    /// Returns `false` once the branch's receiver is gone, which is what lets
    /// a producer stop instead of spinning: a source feeding a sink that had
    /// died used to run at 100% CPU into a closed channel (#85,
    /// `tests/no_hang_on_error.rs`). A **full** channel is not that — under
    /// the lossy policies the buffer (incoming or oldest-queued) is shed and
    /// the branch stays live.
    ///
    /// NOTE: `send().await` must never appear as a `select!` branch. tokio
    /// guarantees the message was not *sent* if the future is cancelled, but
    /// the message is *dropped*. Nothing here sends inside a select; keep it
    /// that way.
    async fn send_buffer(&self, buffer: Buffer, epoch: u64, tracers: &TracerRegistry) -> bool {
        let msg = Message::Buffer(buffer, epoch);
        let sent = match (&self.tx, self.policy) {
            (BranchTx::Mpsc(tx), LinkPolicy::Block) => tx.send(msg).await.is_ok(),
            (BranchTx::Mpsc(tx), LinkPolicy::DropNewest) => match tx.try_send(msg) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    self.record_drop(tracers);
                    true
                }
                Err(TrySendError::Closed(_)) => false,
            },
            (BranchTx::Leaky(tx), _) => {
                use crate::pipeline::leaky::LossyPush;
                match tx.send_lossy(msg) {
                    LossyPush::Sent => true,
                    LossyPush::SentEvictedOldest(_old) => {
                        self.record_drop(tracers);
                        true
                    }
                    LossyPush::Closed(_) => false,
                }
            }
            (BranchTx::Mpsc(_), LinkPolicy::DropOldest) => {
                unreachable!("build_channels pairs DropOldest links with a leaky channel")
            }
        };
        if let Some(flow) = &self.flow {
            flow.update(self.occupancy());
        }
        // The lossy paths never suspend: try_send and send_lossy are
        // synchronous, so a producer whose buffers are always accepted (or
        // shed) would never yield — and a consumer task woken by these sends
        // lands in this worker's non-stealable LIFO slot, where a
        // never-yielding producer starves it forever (the AppSink lesson,
        // gotcha 15). Block links get this for free: tokio's mpsc send
        // participates in the coop budget even when it returns Ready.
        // `consume_budget` charges the same budget here, forcing a yield
        // every ~128 sends instead of every send.
        if self.policy != LinkPolicy::Block {
            tokio::task::coop::consume_budget().await;
        }
        sent
    }

    /// Report one shed buffer on this link.
    ///
    /// Policy drops are expected steady-state behaviour of a deliberately
    /// lossy branch — a tracer notification and a metric, not a warning
    /// ladder like the arena-exhaustion sheds.
    fn record_drop(&self, tracers: &TracerRegistry) {
        tracers.notify_drop(&self.sink_name);
        crate::observability::record_buffer_dropped("pipeline", &self.sink_name);
    }

    /// Send a control message (EOS / in-band event / terminal error),
    /// **always delivered** whatever the link policy.
    ///
    /// Mpsc links block for room; the leaky channel enqueues past capacity
    /// without blocking (control is rare and bounded). Either way a lossy
    /// policy never sheds control — a dropped FlushStop or EOS wedges the
    /// subtree below it forever.
    async fn send_control(&self, msg: Message) -> bool {
        match &self.tx {
            BranchTx::Mpsc(tx) => tx.send(msg).await.is_ok(),
            BranchTx::Leaky(tx) => tx.send_control(msg),
        }
    }
}

/// One upstream branch of a sink-pad: the receiving end of a link's channel,
/// plus that link's flow monitor. `recv`/`try_recv` sample occupancy after
/// every receive, which is the half of the monitoring that observes drains.
struct InputBranch {
    rx: BranchRx,
    flow: Option<Arc<LinkFlowMonitor>>,
    /// The producing node's flush-epoch cell (#163 phase C): staleness is
    /// judged against the producer that stamped the buffer, not a
    /// pipeline-wide counter — a seek on one chain cannot shed an innocent
    /// sibling chain's in-flight buffers.
    producer_epoch: Arc<AtomicU64>,
}

impl InputBranch {
    /// Whether a buffer stamped `stamp` predates its producer's current
    /// flush epoch (#157, scoped per branch by #163 phase C).
    ///
    /// Folds the producer's epoch into the consumer's own cell (`own`), so
    /// a mid-stream bump propagates downstream edge-by-edge at receive
    /// speed — that is what sheds the *next* hop's backlog before the
    /// in-band flush trio can get there. `Acquire` pairs with the seek
    /// handler's `fetch_max(AcqRel)`.
    fn is_stale(&self, stamp: u64, own: &AtomicU64) -> bool {
        let current = self.producer_epoch.load(Ordering::Acquire);
        if current > own.load(Ordering::Relaxed) {
            own.fetch_max(current, Ordering::AcqRel);
        }
        stamp < current
    }

    /// Poll for one message; `Ready(None)` once every sender is gone.
    ///
    /// **Cancel-safe, and it must stay that way.** Both channel kinds
    /// guarantee that an abandoned poll consumed no message (tokio documents
    /// `poll_recv`; the leaky receiver pops synchronously in the returning
    /// poll), which is what lets the consuming loops below race this against
    /// their upstream inbox in a plain `select!`, and the muxer drive many
    /// branches through one fair-poll future (#181). Anything stashed across
    /// polls would be lost when a select loser is dropped.
    ///
    /// Samples the flow monitor on every `Ready`, the receive-side half of
    /// link monitoring (the half that observes drains).
    fn poll_recv(&mut self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Message>> {
        let msg = match &mut self.rx {
            BranchRx::Mpsc(rx) => std::task::ready!(rx.poll_recv(cx)),
            BranchRx::Leaky(rx) => std::task::ready!(rx.poll_recv(cx)),
        };
        if let Some(flow) = &self.flow {
            flow.update(self.rx.len());
        }
        std::task::Poll::Ready(msg)
    }

    /// Receive one message; `None` once every sender is gone. Cancel-safe:
    /// see [`poll_recv`](Self::poll_recv), which this wraps.
    async fn recv(&mut self) -> Option<Message> {
        std::future::poll_fn(|cx| self.poll_recv(cx)).await
    }

    /// Non-blocking receive, for the sink's paused drain.
    fn try_recv(&mut self) -> std::result::Result<Message, TryRecvError> {
        let msg = match &mut self.rx {
            BranchRx::Mpsc(rx) => rx.try_recv(),
            BranchRx::Leaky(rx) => rx.try_recv(),
        };
        if let Some(flow) = &self.flow {
            flow.update(self.rx.len());
        }
        msg
    }
}

/// The muxer's input set: a rotating fair poll over every input branch.
///
/// Replaces the `FuturesUnordered` + re-armed `recv_one` scheme, whose
/// re-push allocated one `Arc<Task>` per received message per branch — the
/// last per-buffer heap allocation on the executor's data path (#181).
/// `next_msg` polls the branches round-robin starting one past the last
/// winner (so a chatty input cannot starve the others) and allocates
/// nothing in steady state.
///
/// Cancel-safe as a `select!` branch: a message is consumed only in the
/// poll that returns it, and the receivers live *here* — owned by the loop,
/// not the dropped future — so losing the select loses nothing. A waker
/// left registered by an abandoned poll can only re-wake this same task.
struct MuxInputs {
    /// Live branches with their pad names. Retired (swap_remove) on
    /// EOS/error/close; order across branches carries no guarantee, only
    /// per-branch FIFO does.
    branches: Vec<(String, InputBranch)>,
    /// Fairness cursor: polling starts here.
    next: usize,
}

impl MuxInputs {
    fn new(inputs_by_pad: HashMap<String, Vec<InputBranch>>) -> Self {
        let branches = inputs_by_pad
            .into_iter()
            .flat_map(|(pad, rxs)| rxs.into_iter().map(move |rx| (pad.clone(), rx)))
            .collect();
        Self { branches, next: 0 }
    }

    fn len(&self) -> usize {
        self.branches.len()
    }

    /// `Ready(Some((idx, msg)))` for one branch's message (`msg == None` =
    /// that branch closed); `Ready(None)` once every branch is retired.
    fn poll_next(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<(usize, Option<Message>)>> {
        if self.branches.is_empty() {
            return std::task::Poll::Ready(None);
        }
        let n = self.branches.len();
        for i in 0..n {
            let idx = (self.next + i) % n;
            if let std::task::Poll::Ready(msg) = self.branches[idx].1.poll_recv(cx) {
                self.next = (idx + 1) % n;
                return std::task::Poll::Ready(Some((idx, msg)));
            }
        }
        std::task::Poll::Pending
    }

    /// Per-branch staleness (#163 phase C): each input branch carries its
    /// own producer's epoch cell, so one input's seek cannot shed a quiet
    /// sibling's buffers. `idx` is valid at check time — only terminal
    /// messages retire a branch, never buffers.
    fn is_stale(&self, idx: usize, stamp: u64, own: &AtomicU64) -> bool {
        self.branches[idx].1.is_stale(stamp, own)
    }

    async fn next_msg(&mut self) -> Option<(usize, Option<Message>)> {
        std::future::poll_fn(|cx| self.poll_next(cx)).await
    }

    /// Retire a branch that ended (EOS, error, or closed channel).
    fn retire(&mut self, idx: usize) {
        self.branches.swap_remove(idx);
        if self.next >= self.branches.len() {
            self.next = 0;
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
async fn broadcast(
    branches: &[OutputBranch],
    buffer: Buffer,
    epoch: u64,
    tracers: &TracerRegistry,
) -> bool {
    match branches {
        [] => true,
        [only] => only.send_buffer(buffer, epoch, tracers).await,
        many => futures::future::join_all(
            many.iter()
                .map(|branch| branch.send_buffer(buffer.clone(), epoch, tracers)),
        )
        .await
        .into_iter()
        .any(|connected| connected),
    }
}

/// Send EOS to every branch, **always delivered**.
///
/// A dropped EOS would leave the branch's sink waiting forever, so the lossy
/// policies do not apply to it. Only buffers are ever dropped.
async fn broadcast_eos(branches: &[OutputBranch]) {
    for branch in branches {
        let _ = branch.send_control(Message::Eos).await;
    }
}

/// Send an in-band event to every branch, **always delivered**.
///
/// Events are control flow, not payload: a dropped FlushStop would leave the
/// branch's subtree discarding buffers forever, so the lossy policies do not
/// apply — same reasoning as [`broadcast_eos`].
async fn broadcast_event(branches: &[OutputBranch], event: &Event) {
    for branch in branches {
        let _ = branch.send_control(Message::Event(event.clone())).await;
    }
}

/// Initial segment for a producing pad (#165): Time anchored at the first
/// buffer's PTS — so `position()` is honest for streams that start at t > 0
/// and the mapping matches the post-seek segment convention — or Bytes from
/// zero when the stream is untimestamped.
fn initial_segment_for(buffer: &Buffer) -> SegmentEvent {
    match buffer.metadata().pts.to_option() {
        Some(pts) => SegmentEvent::new_time(pts, None),
        None => SegmentEvent::new_bytes(0, None),
    }
}

/// Shape of the Time segments a fed demuxer re-anchors its pads with after
/// a seek it translated upstream (#165). Without it every re-anchor was a
/// bare `initial_segment_for` — rate 1.0, base 0 — so a fed demuxer dropped
/// trick-play rate entirely and non-flushing seeks lost their accumulated
/// base at the translation boundary. The shape is sticky: it describes the
/// output timeline until the next translated seek replaces it, so a later
/// unrelated re-anchor still carries the trick-play rate.
struct PadReanchor {
    rate: f64,
    base: i64,
    stop: Option<ClockTime>,
}

/// The segment a pad anchors with at its first routed buffer: the plain
/// lazy initial segment normally, or the translated seek's shape (rate,
/// base, stop) around this pad's own first post-seek PTS — the honest
/// per-pad landing, same convention as the translated SeekDone.
fn anchor_segment_for(buffer: &Buffer, reanchor: Option<&PadReanchor>) -> SegmentEvent {
    match (reanchor, buffer.metadata().pts.to_option()) {
        (Some(r), Some(pts)) => SegmentEvent::new_time(pts, r.stop)
            .with_rate(r.rate)
            .with_base(r.base),
        _ => initial_segment_for(buffer),
    }
}

/// Producer-side view of the outgoing segment (#165): the Time segment last
/// put on the wire plus how far playback advanced under it. This is what a
/// *non-flushing* seek's successor segment needs — its `base` is the running
/// time already consumed, so running time stays monotonic across the queued
/// boundary (GStreamer's `gst_segment_do_seek` non-flush rule).
///
/// One tracker per producing task. A source-style demuxer keeps a single
/// tracker fed by the max PTS across its pads, so an A/V pair's accumulated
/// base can skew by about one frame between streams — a documented
/// approximation, revisit if multi-sink sync ever depends on it.
#[derive(Default)]
struct SegmentTracker {
    /// The Time segment currently on the wire; `None` until one is emitted.
    current: Option<SegmentEvent>,
    /// Highest PTS stamped since `current` was installed.
    last_pts: ClockTime,
}

impl SegmentTracker {
    /// Record a produced buffer's PTS under the current segment.
    fn observe(&mut self, pts: ClockTime) {
        if pts != ClockTime::NONE && (self.last_pts == ClockTime::NONE || pts > self.last_pts) {
            self.last_pts = pts;
        }
    }

    /// A new segment went on the wire: it becomes current, nothing played
    /// under it yet. Bytes segments are ignored — base is a Time concept.
    fn installed(&mut self, segment: &SegmentEvent) {
        if segment.format == crate::event::SegmentFormat::Time {
            self.current = Some(segment.clone());
            self.last_pts = ClockTime::NONE;
        }
    }

    /// Base for a successor non-flushing segment: the running time of the
    /// furthest position played under the outgoing segment. Falls back to
    /// the outgoing segment's own base when nothing played under it (two
    /// queued seeks back to back), and 0 when no segment exists yet.
    fn accumulated_base(&self) -> i64 {
        let Some(current) = &self.current else {
            return 0;
        };
        if self.last_pts == ClockTime::NONE {
            return current.base;
        }
        match current.to_running_time(self.last_pts).to_option() {
            Some(rt) => rt.nanos() as i64,
            None => current.base,
        }
    }
}

/// What one [`handle_upstream_hop`] call did, for the caller's bookkeeping.
struct HopOutcome {
    /// A Segment was emitted — the caller suppresses its lazy initial one.
    segment_emitted: bool,
    /// This task ran a seek to completion (the `Handled` arm).
    handled_seek: Option<HandledSeek>,
}

impl HopOutcome {
    const NONE: Self = Self {
        segment_emitted: false,
        handled_seek: None,
    };
}

/// The facts of a seek this task just handled, kept for segment bookkeeping.
struct HandledSeek {
    seqnum: u64,
    format: crate::event::SegmentFormat,
    flags: crate::event::SeekFlags,
    stop: Option<u64>,
}

/// A SEGMENT-flagged seek this producing task is playing out (#165).
///
/// When produce() reaches Eos while one is active, the task posts
/// [`MessageKind::SegmentDone`](crate::pipeline::bus::MessageKind::SegmentDone)
/// once instead of broadcasting EOS, then idles (control still drains) so
/// the application's follow-up seek — non-flushing SEGMENT back to the
/// start for a gapless loop — finds a live producer. `PipelineHandle::stop`
/// still tears down cleanly via the loop's stop check.
struct ActiveSegmentSeek {
    seqnum: u64,
    format: crate::event::SegmentFormat,
    stop: Option<u64>,
    done_posted: bool,
}

/// Fold a hop's outcome into the task's active-SEGMENT-seek state: any
/// newly handled seek replaces the previous one (SEGMENT keeps the segment
/// discipline armed, a plain seek disarms it).
fn track_segment_seek(outcome: &HopOutcome, active: &mut Option<ActiveSegmentSeek>) {
    if let Some(hs) = &outcome.handled_seek {
        *active = hs
            .flags
            .contains(crate::event::SeekFlags::SEGMENT)
            .then_some(ActiveSegmentSeek {
                seqnum: hs.seqnum,
                format: hs.format,
                stop: hs.stop,
                done_posted: false,
            });
    }
}

/// Probe-then-broadcast for an event emitted on a src pad: src-pad probes see
/// each emitted event, and Drop/Handled suppresses the emission.
async fn emit_event(
    outputs: &[OutputBranch],
    src_pad: &crate::pipeline::probe::PadRef,
    probes: &ProbeRegistry,
    event: Event,
) {
    match probes.invoke_event(src_pad, &event, true) {
        ProbeReturn::Drop | ProbeReturn::Handled => {}
        _ => broadcast_event(outputs, &event).await,
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

// Hot-path dispatch (#175): an element whose adapter wraps a sync author
// trait declares `dispatches_inline()`, and the task calls the `*_inline`
// form — same body, no boxed future, zero allocations. Genuinely async
// elements take the erased `async` path, whose future dynosaur boxes per
// call. Each task samples the flag once, before its loop.
//
// These helpers are plain `async fn`s so `guard`'s catch_unwind wraps the
// inline body exactly like the async one — a panic inside either surfaces
// as `Error::Panic` naming the element.

async fn hybrid_process(
    mut element: &mut DynAsyncElement<'static>,
    inline: bool,
    input: Option<Buffer>,
) -> Result<Option<Buffer>> {
    if inline {
        element.process_inline(input)
    } else {
        element.process(input).await
    }
}

async fn hybrid_process_source(
    mut element: &mut DynAsyncElement<'static>,
    inline: bool,
) -> Result<crate::element::SourceResult> {
    if inline {
        element.process_source_inline()
    } else {
        element.process_source().await
    }
}

async fn hybrid_process_demux(
    mut element: &mut DynAsyncElement<'static>,
    inline: bool,
    input: Option<Buffer>,
) -> Result<crate::element::DemuxResult> {
    if inline {
        element.process_demux_inline(input)
    } else {
        element.process_demux(input).await
    }
}

/// Send a terminal error to every branch, **always delivered**.
///
/// Same reasoning as [`broadcast_eos`]: a dropped terminal message wedges the
/// branch's sink. This is what makes a failed pipeline distinguishable from a
/// finished one at the app boundary — the sink used to receive a plain EOS and
/// report a clean end of stream.
async fn broadcast_error(branches: &[OutputBranch], error: &StreamError) {
    for branch in branches {
        let _ = branch.send_control(Message::Error(error.clone())).await;
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
    /// Dedup key set for `has_channel` — one channel per graph edge.
    ///
    /// This used to hold a clone of both halves. tokio's receiver is not
    /// `Clone`, and keeping either half here would defer closed-detection
    /// until `spawn_tasks` returned; the key alone is all `has_channel`
    /// ever read.
    channels: HashSet<ChannelKey>,
    outputs: HashMap<(NodeId, String), Vec<OutputBranch>>,
    inputs: HashMap<(NodeId, String), Vec<InputBranch>>,
    /// One flush-epoch cell per node (#163 phase C). A consumer judges a
    /// buffer's staleness against the cell of the node that *produced* it
    /// (paired into `InputBranch` here), and each task receives its own
    /// cell at spawn — a seek handler bumps only that, which scopes the
    /// shed to the seek's actual path instead of the whole pipeline.
    epochs: HashMap<NodeId, Arc<AtomicU64>>,
}

impl ChannelNetwork {
    fn new() -> Self {
        Self {
            channels: HashSet::new(),
            outputs: HashMap::new(),
            inputs: HashMap::new(),
            epochs: HashMap::new(),
        }
    }

    /// The node's flush-epoch cell, created on first use.
    fn epoch_cell(&mut self, node: NodeId) -> Arc<AtomicU64> {
        self.epochs.entry(node).or_default().clone()
    }

    fn has_channel(&self, src: NodeId, src_pad: &str, sink: NodeId, sink_pad: &str) -> bool {
        self.channels
            .contains(&(src, src_pad.to_string(), sink, sink_pad.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn add_channel(
        &mut self,
        src: NodeId,
        src_pad: String,
        sink: NodeId,
        sink_pad: String,
        tx: BranchTx,
        rx: BranchRx,
        policy: LinkPolicy,
        sink_name: String,
        flow: Option<Arc<LinkFlowMonitor>>,
    ) {
        self.channels
            .insert((src, src_pad.clone(), sink, sink_pad.clone()));
        let producer_epoch = self.epoch_cell(src);
        self.outputs
            .entry((src, src_pad))
            .or_default()
            .push(OutputBranch {
                tx,
                policy,
                sink_name,
                flow: flow.clone(),
            });
        self.inputs
            .entry((sink, sink_pad))
            .or_default()
            .push(InputBranch {
                rx,
                flow,
                producer_epoch,
            });
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

    fn take_inputs_by_pad(&mut self, node: NodeId) -> HashMap<String, Vec<InputBranch>> {
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

    fn take_inputs(&mut self, node: NodeId) -> Vec<InputBranch> {
        self.take_inputs_by_pad(node)
            .into_values()
            .flatten()
            .collect()
    }
}

// ============================================================================
// Task Spawning
// ============================================================================

/// One node's upstream-event connectivity (#163): the receiving end of its
/// unbounded inbox plus its parents' inbox senders. Upstream events enter
/// the graph at the sinks and travel hop-by-hop toward the sources; a node
/// that does not handle an event forwards it to every parent.
///
/// The inboxes are **unbounded** as a correctness choice, not convenience:
/// an upstream send from task A to its parent B can coincide with B being
/// parked in `send().await` into A's full data channel. A bounded upstream
/// inbox would close that cycle into a deadlock; unbounded sends never
/// await, so the cycle cannot form. Control events are rare and small.
struct UpstreamHop {
    rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    /// `(parent name, parent inbox)` — the async-spawned upstream peers.
    parents: Vec<(String, tokio::sync::mpsc::UnboundedSender<Event>)>,
}

/// A seek this task converted into another format and forwarded upstream
/// (#163 phase B), held until the upstream flush comes back down so the
/// completion can be reported in the format the application asked in.
struct PendingTranslation {
    seqnum: u64,
    /// The format the application seeked in (the outward one).
    format: SegmentFormat,
    /// The original seek's rate — the fed demuxer's re-anchored pad
    /// segments must carry it or trick play dies at the translation (#165).
    rate: f64,
    /// Whether the original seek flushes. A non-flushing translation gets
    /// no FlushStop back from the source, so the fed demuxer clears its
    /// pad-anchor set itself and accumulates `base` (#165).
    flushing: bool,
    /// The original seek's absolute stop, when it set one.
    stop: Option<u64>,
}

/// Which side of a consuming task's data-vs-upstream select fired.
///
/// Returning the winner instead of acting inside the `select!` arm is what
/// keeps the `&mut up_rx` borrow scoped to the select itself, so the
/// inbox-closed arm can reassign it and the handler arms can borrow
/// `up_parents` freely.
// Not boxed on purpose: this is a per-iteration stack value that exists only
// to carry the select's winner out to the arms. `Message` is already moved by
// value through the channel itself, so the enum costs nothing extra — whereas
// boxing would put an allocation back on the per-buffer path, which is the
// thing this migration removes.
#[allow(clippy::large_enum_variant)]
enum Incoming {
    Upstream(Option<Event>),
    Data(Option<Message>),
}

/// Forward an event to every parent inbox (unbounded: never blocks).
fn forward_to_parents(
    parents: &[(String, tokio::sync::mpsc::UnboundedSender<Event>)],
    event: &Event,
) {
    for (parent, tx) in parents {
        if tx.send(event.clone()).is_err() {
            tracing::debug!(
                "parent '{parent}' is gone; upstream {} dropped",
                event.name()
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Handle one upstream event delivered to a node's inbox (#163).
///
/// For a handled flushing seek this bumps **this node's own** flush-epoch
/// cell (per-node since phase C — the shed's scope equals the flush trio's
/// scope) and runs the flush sequence in-band on every
/// output branch — FlushStart → FlushStop → Segment, the order downstream
/// elements rely on to discard stale data and re-anchor their timeline — and
/// posts [`MessageKind::SeekDone`](crate::pipeline::bus::MessageKind::SeekDone) on the bus. The segment is synthesized here
/// from the seek's target and reported landing.
///
/// Returns `true` when a Segment was emitted, so the caller can suppress its
/// lazy initial segment (#165) — a seek that lands before the first buffer
/// already established the mapping.
///
/// This is one hop of the upstream route (#163): the same function runs in
/// a source's control drain (empty `parents` — the route ends here) and in
/// mid-graph tasks (transform/demuxer/muxer), where `NotHandled` forwards
/// the event to every parent inbox. A handling task runs the flush trio
/// from its own loop, between its process calls, so the epoch discipline
/// holds wherever the seek terminates. `last_seek_epoch` dedups multi-path
/// delivery (a diamond delivers the same seek along every branch); a seek
/// with a seqnum at or below the last seen one is dropped — newer wins.
///
/// `tracker` is the producing task's [`SegmentTracker`] (#165): a handled
/// non-flushing Time seek's segment gets `base = accumulated running time`
/// from it, and every emitted Time segment is installed back into it.
/// Mid-graph tasks that emit no producer segments pass `None` — behavior is
/// then exactly the pre-tracker one (base 0).
async fn handle_upstream_hop(
    name: &str,
    element: &mut Box<DynAsyncElement<'static>>,
    event: &Event,
    outputs: &[OutputBranch],
    src_pad: &crate::pipeline::probe::PadRef,
    probe_registry: &ProbeRegistry,
    own_epoch: &AtomicU64,
    bus: &BusHandle,
    parents: &[(String, tokio::sync::mpsc::UnboundedSender<Event>)],
    last_seek_epoch: &mut u64,
    warn_unhandled_seek: bool,
    pending_translation: &mut Option<PendingTranslation>,
    tracker: Option<&mut SegmentTracker>,
) -> HopOutcome {
    if let Event::Seek(seek) = event {
        // Ordered by epoch, not seqnum (#173): a refinement round shares its
        // seek's seqnum but must pass this guard, while a diamond's duplicate
        // delivery of the same round must not.
        if seek.epoch() <= *last_seek_epoch {
            tracing::debug!(
                "'{name}': seek {} already seen (multi-path delivery), dropped",
                seek.seqnum()
            );
            return HopOutcome::NONE;
        }
        *last_seek_epoch = seek.epoch();
    }

    match probe_registry.invoke_event(src_pad, event, false) {
        ProbeReturn::Drop | ProbeReturn::Handled => return HopOutcome::NONE,
        _ => {}
    }

    let result = element.handle_upstream_event(event);
    let Event::Seek(seek) = event else {
        match &result {
            EventResult::NotHandled => forward_to_parents(parents, event),
            EventResult::Forward(translated) => forward_to_parents(parents, translated),
            _ => {}
        }
        return HopOutcome::NONE;
    };

    match result {
        EventResult::Handled { position: landing } => {
            // Src-pad probes see each emitted event, mirroring how buffers are
            // probed on the pad that emits them; Drop/Handled suppresses it.
            let emit = |event: Event| match probe_registry.invoke_event(src_pad, &event, true) {
                ProbeReturn::Drop | ProbeReturn::Handled => None,
                _ => Some(event),
            };
            if seek.flags.contains(crate::event::SeekFlags::FLUSH) {
                // The element repositioned in `handle_upstream_event` above and
                // this runs inside the producer's own task, between produce
                // calls — so every buffer already stamped is pre-seek and every
                // buffer stamped after this line is post-seek, with no cross-
                // task handshake. Transitioning *before* FlushStart means
                // consumers drain the stale backlog at receive speed and the
                // flush trio arrives promptly instead of queueing behind
                // seconds of data.
                //
                // The epoch BECOMES the seek's epoch ordinal — seqnum and
                // refinement round folded into one monotonic value (#173);
                // for an ordinary seek that is just the seqnum scaled.
                // fetch_max is what makes a multi-source seek transition
                // exactly once: every handling source calls it with the SAME
                // value (Clone shares it), the second call is a no-op, and
                // each caller's own AcqRel RMW guarantees its later loads see
                // ≥ it — so no source's post-seek buffers can be stamped
                // stale by a sibling handling the same seek.
                own_epoch.fetch_max(seek.epoch(), Ordering::AcqRel);
                if let Some(ev) = emit(Event::FlushStart) {
                    broadcast_event(outputs, &ev).await;
                }
                if let Some(ev) = emit(Event::FlushStop(FlushStopEvent::new(true))) {
                    broadcast_event(outputs, &ev).await;
                }
            }

            // Segment start: the element's actual landing position when it
            // reported one (a keyframe-snapping demuxer lands off-target),
            // else the requested position. Current/End-relative requests
            // have no absolute target — only the element knows where they
            // ended up (FileSrc reports it). The seek's rate and stop are
            // carried so the segment's shape is right for trick play.
            // A flushing seek restarts running time (`base` 0); a
            // non-flushing seek accumulates the running time consumed under
            // the outgoing segment into `base` (#165) so downstream running
            // time stays monotonic across the queued boundary.
            // `applied_rate` stays 1.0 until server-side trick modes exist.
            let requested = match seek.start.seek_type {
                SeekType::Set => Some(seek.start.position.max(0)),
                _ => None,
            };
            let reverse = seek.rate < 0.0;
            // Reverse (#165): the segment covers [start, stop] and playback
            // begins at the TOP — the element's reported landing is where
            // decoding starts (near stop), NOT the segment's start. Forward
            // keeps the landing-as-start rule (keyframe snap) — except under
            // ACCURATE, where the segment starts at the REQUESTED time even
            // though data starts at the snapped keyframe: everything between
            // is out-of-segment, which is exactly what decoder/sink clipping
            // keys on so the first shown frame is the request itself (#165).
            // ACCURATE outranks the always-on KEY_UNIT here, or it would be
            // unreachable. SeekDone still reports the honest landing.
            let accurate = seek.flags.contains(crate::event::SeekFlags::ACCURATE);
            let start = if reverse {
                requested.or(Some(0))
            } else if accurate {
                requested.or_else(|| landing.map(|p| p.max(0)))
            } else {
                landing.map(|p| p.max(0)).or(requested)
            };
            let segment_start = start.unwrap_or_else(|| {
                tracing::warn!(
                    "source '{name}': relative seek with no reported landing; \
                     segment restarts at 0"
                );
                0
            });
            let stop = match seek.stop.seek_type {
                SeekType::Set => Some(seek.stop.position.max(0) as u64),
                // No explicit stop: fall back to the element's total in the
                // seek's own format. A fed demuxer translating TIME→BYTES
                // has no other way to learn the file size, and without it
                // its byte estimate cannot be clamped to the last byte.
                // For a reverse seek the element's landing IS the top of
                // the range, and beats a possibly-unknown duration.
                _ => {
                    let duration = element
                        .source_query_duration()
                        .filter(|d| d.format == seek.format)
                        .and_then(|d| d.duration);
                    if reverse {
                        landing.map(|p| p.max(0) as u64).or(duration)
                    } else {
                        duration
                    }
                }
            };
            if reverse && stop.is_none() {
                // An unmappable reverse segment would make every PTS
                // out-of-segment; better to say so once than to emit it.
                bus.post_warning(
                    format!(
                        "'{name}': reverse seek {} has no resolvable stop; \
                         segment will be unmappable",
                        seek.seqnum()
                    ),
                    None,
                );
            }
            let flushing = seek.flags.contains(crate::event::SeekFlags::FLUSH);
            let base = match (&tracker, flushing) {
                (Some(t), false) => t.accumulated_base(),
                _ => 0,
            };
            let segment = match seek.format {
                crate::event::SegmentFormat::Bytes => {
                    SegmentEvent::new_bytes(segment_start as u64, stop)
                }
                _ => SegmentEvent::new_time(
                    ClockTime::from_nanos(segment_start as u64),
                    stop.map(ClockTime::from_nanos),
                ),
            }
            .with_rate(seek.rate)
            .with_base(base);
            if let Some(t) = tracker {
                t.installed(&segment);
            }
            let segment_emitted = match emit(Event::Segment(segment)) {
                Some(ev) => {
                    broadcast_event(outputs, &ev).await;
                    true
                }
                None => false,
            };

            bus.post(crate::pipeline::bus::MessageKind::SeekDone {
                seqnum: seek.seqnum(),
                source: name.to_string(),
                format: seek.format,
                // Reverse playback starts at the range's top; forward
                // reports the element's landing (== segment start except
                // under ACCURATE, whose segment starts at the request).
                position: if reverse {
                    stop
                } else {
                    landing.map(|p| p.max(0) as u64).or(start.map(|p| p as u64))
                },
            });
            tracing::info!(
                "source '{name}': seek {} handled, segment starts at {segment_start}",
                seek.seqnum()
            );
            HopOutcome {
                segment_emitted,
                handled_seek: Some(HandledSeek {
                    seqnum: seek.seqnum(),
                    format: seek.format,
                    flags: seek.flags,
                    stop,
                }),
            }
        }
        EventResult::Forward(translated) => {
            if parents.is_empty() {
                bus.post_warning(
                    format!("'{name}' translated a seek but has no upstream peer"),
                    None,
                );
                return HopOutcome::NONE;
            }
            // The invariant `SeekEvent::derive` exists to preserve. A
            // replacement that renamed the seek would break flush-epoch
            // idempotence and SeekDone correlation, so it is a bug in the
            // element rather than a policy to honour.
            match &*translated {
                Event::Seek(t) if t.seqnum() == seek.seqnum() => {}
                other => {
                    bus.post_warning(
                        format!(
                            "'{name}' replaced seek {} with an unrelated {} — dropped",
                            seek.seqnum(),
                            other.name()
                        ),
                        None,
                    );
                    return HopOutcome::NONE;
                }
            }
            *pending_translation = Some(PendingTranslation {
                seqnum: seek.seqnum(),
                format: seek.format,
                rate: seek.rate,
                flushing: seek.flags.contains(crate::event::SeekFlags::FLUSH),
                stop: match seek.stop.seek_type {
                    SeekType::Set => Some(seek.stop.position.max(0) as u64),
                    _ => None,
                },
            });
            tracing::debug!(
                "'{name}': translated seek {} to {:?}, forwarding upstream",
                seek.seqnum(),
                translated.name()
            );
            // No epoch bump, no flush trio, no SeekDone here: the upstream
            // source runs all of that when it handles the derived seek, and
            // because `derive` kept the seqnum, `fetch_max` reaches exactly
            // the same value it otherwise would.
            forward_to_parents(parents, &translated);
            HopOutcome::NONE
        }
        EventResult::NotHandled if !parents.is_empty() => {
            // Not ours — keep it travelling toward the sources.
            forward_to_parents(parents, event);
            HopOutcome::NONE
        }
        EventResult::NotHandled => {
            // End of the route. Only warn when this element claimed
            // seekability at start — an unseekable source in a mixed graph
            // legitimately declines.
            if warn_unhandled_seek {
                bus.post_warning(format!("source '{name}' cannot seek; seek ignored"), None);
            } else {
                tracing::debug!("'{name}': seek {} reached an unseekable end", seek.seqnum());
            }
            HopOutcome::NONE
        }
        EventResult::Error => {
            bus.post_warning(format!("source '{name}': seek failed"), None);
            HopOutcome::NONE
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
    mut upstream: UpstreamHop,
    pause_rx: watch::Receiver<bool>,
    own_epoch: Arc<AtomicU64>,
    bus: BusHandle,
    // Whether this source claimed seekability at start — an unhandled seek
    // only warns then.
    warn_unhandled_seek: bool,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    tokio::spawn(reporting(share, async move {
        let inline_dispatch = element.dispatches_inline();
        tracing::debug!("source '{}' started", name);
        events.send_node_started(&name);

        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        // StreamStart precedes everything on the wire — including a seek's
        // flush trio, when one lands before the first buffer (#165).
        emit_event(
            &outputs,
            &src_pad,
            &probe_registry,
            Event::StreamStart(StreamStartEvent::new(&name)),
        )
        .await;
        // The initial Segment is lazy: it anchors at the FIRST buffer's PTS,
        // which is only known once produced. A pre-first-buffer seek emits
        // the segment instead and suppresses this one. Byte-oriented sources
        // (FileSrc, HttpSrc — they answer duration queries in Bytes) get a
        // Bytes segment instead: their buffers carry a meaningless PTS of
        // zero, not a timeline (`Metadata::default()` is ZERO, not NONE).
        // The total rides along as the segment's `stop`: a fed demuxer
        // translating a TIME seek into a BYTES one has no other way to learn
        // the stream size, and without it no byte estimate can be clamped.
        let byte_total = element
            .source_query_duration()
            .filter(|q| q.format == crate::event::SegmentFormat::Bytes)
            .map(|q| q.duration);
        let bytes_native = byte_total.is_some();
        let byte_total = byte_total.flatten();
        let mut segment_sent = false;
        let mut segment_tracker = SegmentTracker::default();
        let mut segment_seek: Option<ActiveSegmentSeek> = None;
        let mut last_seek_epoch: u64 = 0;
        // #163 phase B: set when this element converts a seek into another
        // format and forwards it upstream; cleared when the completion is
        // reported in the format the application asked in.
        let mut pending_translation: Option<PendingTranslation> = None;
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
            while let Ok(event) = upstream.rx.try_recv() {
                let outcome = handle_upstream_hop(
                    &name,
                    &mut element,
                    &event,
                    &outputs,
                    &src_pad,
                    &probe_registry,
                    &own_epoch,
                    &bus,
                    &upstream.parents,
                    &mut last_seek_epoch,
                    warn_unhandled_seek,
                    &mut pending_translation,
                    Some(&mut segment_tracker),
                )
                .await;
                segment_sent |= outcome.segment_emitted;
                track_segment_seek(&outcome, &mut segment_seek);
            }

            // Runtime pause (PipelineHandle::pause): stop producing until
            // resumed. Control events still drain so a seek can land while
            // paused, and stop still wins.
            while *pause_rx.borrow() && !stop.load(Ordering::Acquire) {
                while let Ok(event) = upstream.rx.try_recv() {
                    let outcome = handle_upstream_hop(
                        &name,
                        &mut element,
                        &event,
                        &outputs,
                        &src_pad,
                        &probe_registry,
                        &own_epoch,
                        &bus,
                        &upstream.parents,
                        &mut last_seek_epoch,
                        warn_unhandled_seek,
                        &mut pending_translation,
                        Some(&mut segment_tracker),
                    )
                    .await;
                    segment_sent |= outcome.segment_emitted;
                    track_segment_seek(&outcome, &mut segment_seek);
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            tracing::trace!("source '{}': calling process_source", name);
            match guard(&name, hybrid_process_source(&mut element, inline_dispatch)).await {
                Ok(SourceResult::Buffer(buffer)) => {
                    count += 1;
                    would_block_count = 0; // Reset

                    // Lazy initial segment, anchored at this first buffer's
                    // PTS (#165) — emitted before the buffer's own probes so
                    // pad logs see segment-then-buffer, matching the wire.
                    if !segment_sent {
                        segment_sent = true;
                        let segment = if bytes_native {
                            SegmentEvent::new_bytes(0, byte_total)
                        } else {
                            initial_segment_for(&buffer)
                        };
                        segment_tracker.installed(&segment);
                        emit_event(&outputs, &src_pad, &probe_registry, Event::Segment(segment))
                            .await;
                    }
                    segment_tracker.observe(buffer.metadata().pts);

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
                    // Stamped at send time: a seek handled in this iteration's
                    // control drain already bumped the epoch, so this buffer —
                    // produced after the reposition — carries the new one.
                    let epoch = own_epoch.load(Ordering::Acquire);
                    let connected = broadcast(&outputs, buffer.clone(), epoch, &tracers).await;
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
                    // SEGMENT seek (#165): the segment ran out, not the
                    // pipeline. Post SegmentDone once and idle awaiting the
                    // app's follow-up seek; stop still tears down above.
                    if let Some(ss) = segment_seek.as_mut() {
                        if !ss.done_posted {
                            ss.done_posted = true;
                            bus.post(crate::pipeline::bus::MessageKind::SegmentDone {
                                seqnum: ss.seqnum,
                                source: name.clone(),
                                format: ss.format,
                                position: ss.stop,
                            });
                            tracing::info!("source '{}': segment done (seek {})", name, ss.seqnum);
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                        continue;
                    }
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
    inputs: Vec<InputBranch>,
    input_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    position: Arc<AtomicU64>,
    own_epoch: Arc<AtomicU64>,
    pause_rx: watch::Receiver<bool>,
    upstream: Option<UpstreamHop>,
    bus: BusHandle,
    mut shed: ShedTracker,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    // Advance the shared last-presented-PTS cell. `max` keeps it monotonic
    // against decoder reordering; the FlushStop reset is what lets a backwards
    // seek move it backwards.
    fn present(position: &AtomicU64, pts: ClockTime, reverse: bool) {
        let Some(pts) = pts.to_option() else { return };
        if reverse {
            // Reverse playback (#165): PTS legitimately decrease; max-only
            // would freeze the reported position at the first frame.
            position.store(pts.nanos(), Ordering::Release);
            return;
        }
        let _ = position.fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
            (cur == u64::MAX || pts.nanos() > cur).then_some(pts.nanos())
        });
    }

    /// Deliver one in-band event to the sink element, with the probe and
    /// position bookkeeping the event arms rely on. Shared by the running
    /// loop and the paused drain, so a flushing seek acts identically in
    /// both states.
    ///
    /// No `flush()` call here: a sink's flush *commits* buffered output (an
    /// Mp4FileSink finalizes the file) — the element decides what a
    /// flush-seek means for it.
    fn deliver_sink_event(
        element: &mut Box<DynAsyncElement<'static>>,
        event: Event,
        probe_registry: &ProbeRegistry,
        sink_pad: &crate::pipeline::probe::PadRef,
        position: &AtomicU64,
        reverse: &mut bool,
    ) {
        match probe_registry.invoke_event(sink_pad, &event, true) {
            ProbeReturn::Drop | ProbeReturn::Handled => return,
            _ => {}
        }
        match &event {
            Event::FlushStop(_) => {
                // Forget the pre-seek position; the Segment that follows
                // re-anchors it, and `present`'s max() starts fresh there.
                position.store(u64::MAX, Ordering::Release);
            }
            Event::Segment(seg) if seg.format == crate::event::SegmentFormat::Time => {
                // Reverse playback (#165) starts at the range's top.
                *reverse = seg.rate < 0.0;
                let anchor = if *reverse && seg.stop >= 0 {
                    seg.stop as u64
                } else {
                    seg.start.max(0) as u64
                };
                position.store(anchor, Ordering::Release);
            }
            _ => {}
        }
        let _ = element.handle_downstream_event(event);
    }

    tokio::spawn(reporting(share, async move {
        let inline_dispatch = element.dispatches_inline();
        tracing::debug!("sink '{}' started", name);
        events.send_node_started(&name);

        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let mut count: u64 = 0;

        let n_inputs = inputs.len();
        tracing::debug!("sink '{}': {} inputs", name, n_inputs);
        if let Some(mut rx) = inputs.into_iter().next() {
            // Split the hop: tokio recv needs `&mut rx` while the parents
            // list stays shared with the handler calls.
            let (mut up_rx, up_parents) = match upstream {
                Some(hop) => (Some(hop.rx), hop.parents),
                None => (None, Vec::new()),
            };
            let mut last_seek_epoch: u64 = 0;
            // Reverse playback (#165), from the current segment's rate.
            let mut reverse = false;
            // #163 phase B: set when this element converts a seek into another
            // format and forwards it upstream; cleared when the completion is
            // reported in the format the application asked in.
            let mut pending_translation: Option<PendingTranslation> = None;
            // Standard path: read from the link channel
            loop {
                // Runtime pause (#156). Gating only the producers leaves the
                // queued stream ahead of this sink playing out (~2.6 s of
                // audio on a real pipeline), so the sink holds too: tell the
                // element once (AlsaSink silences its device), then drain
                // non-blockingly until resume — stale-epoch buffers drop (a
                // seek can land while paused), events act immediately (its
                // FlushStart must silence the device now, not at resume), and
                // the first fresh buffer is *stashed*, because pause is not
                // flush: it replays right after resume. Eos/Error end the
                // hold — a run that has ended stays ended, paused or not.
                //
                // A sink parked in `recv()` on an idle channel sees the gate
                // only with the next message; during playback the channel is
                // never idle, and an idle sink has nothing to silence.
                let msg = if *pause_rx.borrow() {
                    let _ = element.handle_downstream_event(Event::Pause);
                    let mut stashed: Option<Message> = None;
                    while *pause_rx.borrow() {
                        // Upstream events must keep moving while paused — a
                        // seek lands during pause by design (#71), and since
                        // #163 its path runs through this sink.
                        if let Some(rx_up) = up_rx.as_mut() {
                            while let Ok(event) = rx_up.try_recv() {
                                let _ = handle_upstream_hop(
                                    &name,
                                    &mut element,
                                    &event,
                                    &[],
                                    &sink_pad,
                                    &probe_registry,
                                    &own_epoch,
                                    &bus,
                                    &up_parents,
                                    &mut last_seek_epoch,
                                    false,
                                    &mut pending_translation,
                                    None,
                                )
                                .await;
                            }
                        }
                        // A run that has ended stays ended, paused or not.
                        if matches!(stashed, Some(Message::Eos | Message::Error(_))) {
                            break;
                        }
                        // Holding a fresh buffer: stop draining (its
                        // successors keep FIFO order in the channel) and
                        // just wait for resume.
                        if stashed.is_some() {
                            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                            continue;
                        }
                        match rx.try_recv() {
                            Ok(Message::Buffer(buffer, epoch)) => {
                                if rx.is_stale(epoch, &own_epoch) {
                                    count += 1;
                                    tracers.notify_drop(&name);
                                } else {
                                    stashed = Some(Message::Buffer(buffer, epoch));
                                }
                            }
                            Ok(Message::Event(event)) => {
                                deliver_sink_event(
                                    &mut element,
                                    event,
                                    &probe_registry,
                                    &sink_pad,
                                    &position,
                                    &mut reverse,
                                );
                            }
                            Ok(terminal) => stashed = Some(terminal),
                            Err(TryRecvError::Empty) => {
                                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                            }
                            // Disconnected only fires once the queue is empty
                            // AND every sender is gone, so buffered messages
                            // still drain first. The recv below reports the
                            // close to the normal teardown arm.
                            Err(TryRecvError::Disconnected) => break,
                        }
                    }
                    let _ = element.handle_downstream_event(Event::Resume);
                    match stashed {
                        Some(m) => Some(m),
                        None => rx.recv().await,
                    }
                } else if up_rx.is_some() {
                    // Upstream events enter the graph here (#163): the sink
                    // is the dispatch point, offering each event to its
                    // element and forwarding unhandled ones toward the
                    // sources. `biased` polls the inbox first so control
                    // outruns queued data; both receives are cancel-safe, so
                    // the losing branch simply resumes next iteration.
                    let incoming = {
                        let rx_up = up_rx.as_mut().expect("checked above");
                        tokio::select! {
                            biased;
                            ev = rx_up.recv() => Incoming::Upstream(ev),
                            msg = rx.recv() => Incoming::Data(msg),
                        }
                    };
                    match incoming {
                        Incoming::Upstream(Some(event)) => {
                            let _ = handle_upstream_hop(
                                &name,
                                &mut element,
                                &event,
                                &[],
                                &sink_pad,
                                &probe_registry,
                                &own_epoch,
                                &bus,
                                &up_parents,
                                &mut last_seek_epoch,
                                false,
                                &mut pending_translation,
                                None,
                            )
                            .await;
                            continue;
                        }
                        Incoming::Upstream(None) => {
                            // Inbox closed (handle dropped): stop selecting.
                            up_rx = None;
                            continue;
                        }
                        Incoming::Data(msg) => msg,
                    }
                } else {
                    rx.recv().await
                };
                match msg {
                    Some(Message::Buffer(buffer, epoch)) => {
                        count += 1;
                        tracing::debug!("sink '{}': received buffer {}", name, count);
                        // Pre-seek data: queued ahead of the flush trio, shed
                        // at receive speed instead of presented (#157).
                        if rx.is_stale(epoch, &own_epoch) {
                            tracers.notify_drop(&name);
                            continue;
                        }
                        match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }
                        tracers.notify_buffer(&name, &buffer);
                        let pts = buffer.metadata().pts;
                        match guard(
                            &name,
                            hybrid_process(&mut element, inline_dispatch, Some(buffer)),
                        )
                        .await
                        {
                            Ok(_) => {
                                shed.reset();
                                tracers.notify_buffer_processed(&name);
                                present(&position, pts, reverse);
                            }
                            // Same rule as the transform arm (gotcha 13): a sink
                            // that allocates — an encoder-backed sink, a sink
                            // with its own output arena — sheds the frame
                            // instead of killing a live pipeline.
                            Err(Error::PoolExhausted) => {
                                shed.record(&name, &tracers)?;
                            }
                            Err(e) => {
                                events.send_error(e.to_string(), Some(name.clone()));
                                return Err(e);
                            }
                        }
                        // #184: a sink is the natural QoS origin — poll for
                        // an event it wants to send upstream. Routed toward
                        // the sources on the #163 transport; QoS is also
                        // mirrored onto the bus here, the single point where
                        // sink-originated traffic surfaces.
                        while let Some(event) = element.take_upstream_event() {
                            if let Event::Qos(qos) = &event {
                                bus.post_qos(qos);
                            }
                            if up_parents.is_empty() {
                                tracing::debug!(
                                    "sink '{name}': originated {} with no upstream peer",
                                    event.name()
                                );
                            } else {
                                forward_to_parents(&up_parents, &event);
                            }
                        }
                    }
                    Some(Message::Event(event)) => {
                        deliver_sink_event(
                            &mut element,
                            event,
                            &probe_registry,
                            &sink_pad,
                            &position,
                            &mut reverse,
                        );
                    }
                    Some(Message::Eos) => {
                        tracing::debug!("sink '{}': EOS after {}", name, count);
                        // Deliver EOS to the element so sinks with external
                        // consumers (AppSink) can unblock them.
                        let _ = element.handle_downstream_event(crate::event::Event::Eos);
                        break;
                    }
                    Some(Message::Error(err)) => {
                        // The whole point of #85: the consumer learns the
                        // pipeline *failed*, not that the stream ended.
                        tracing::error!("sink '{}': upstream failed: {}", name, err);
                        let _ = element.handle_downstream_event(crate::event::Event::Error(err));
                        break;
                    }
                    None => {
                        // Upstream now sends a terminal message before dropping
                        // its sender, so a bare close means the pipeline was
                        // aborted rather than that it ended.
                        tracing::debug!("sink '{}': channel closed after {}", name, count);
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
                    let result = guard(
                        &name,
                        hybrid_process(&mut element, inline_dispatch, Some(buffer)),
                    )
                    .await;
                    tracers.notify_buffer_processed(&name);
                    match result {
                        Ok(_) => {
                            shed.reset();
                            // RT bridges carry no events, so no segment can
                            // flip this path into reverse.
                            present(&position, pts, false);
                        }
                        // Mirrors the channel path: arena exhaustion is flow
                        // control, not failure.
                        Err(Error::PoolExhausted) => {
                            shed.record(&name, &tracers)?;
                        }
                        Err(e) => {
                            events.send_error(e.to_string(), Some(name.clone()));
                            return Err(e);
                        }
                    }
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
    inputs: Vec<InputBranch>,
    outputs: Vec<OutputBranch>,
    input_bridges: Vec<Arc<AsyncRtBridge>>,
    output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    mut shed: ShedTracker,
    own_epoch: Arc<AtomicU64>,
    upstream: Option<UpstreamHop>,
    bus: BusHandle,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    tokio::spawn(reporting(share, async move {
        let inline_dispatch = element.dispatches_inline();
        tracing::debug!("transform '{}' started", name);
        events.send_node_started(&name);

        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;

        /// Helper to send output buffer to all downstream channels and bridges.
        async fn send_output(
            buffer: Buffer,
            epoch: u64,
            outputs: &[OutputBranch],
            output_bridges: &[Arc<AsyncRtBridge>],
            tracers: &TracerRegistry,
        ) {
            // Move the buffer into the common single-consumer case instead
            // of cloning for it and dropping the original (#142) — the
            // clone was a metadata copy plus six refcount atomics per
            // buffer per element for nothing.
            if output_bridges.is_empty() {
                broadcast(outputs, buffer, epoch, tracers).await;
                return;
            }
            broadcast(outputs, buffer.clone(), epoch, tracers).await;
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

        if let Some(mut rx) = inputs.into_iter().next() {
            // Epoch of the last accepted input. Outputs are stamped with this,
            // NOT with the global counter at send time: if a flush lands while
            // `process()` is running, its output came from pre-seek input and
            // must test stale downstream — a send-time global stamp would
            // launder it fresh.
            let mut in_epoch: u64 = 0;
            // Split the hop: tokio recv needs `&mut rx` while the parents
            // list stays shared with the handler calls.
            let (mut up_rx, up_parents) = match upstream {
                Some(hop) => (Some(hop.rx), hop.parents),
                None => (None, Vec::new()),
            };
            let mut last_seek_epoch: u64 = 0;
            // #163 phase B: set when this element converts a seek into another
            // format and forwards it upstream; cleared when the completion is
            // reported in the format the application asked in.
            let mut pending_translation: Option<PendingTranslation> = None;
            // Standard path: read from the link channel
            loop {
                tracing::trace!("transform '{}': waiting for input", name);
                // Upstream events (seek and friends, #163) take priority:
                // this element may handle them (running the flush trio from
                // its own loop, preserving the epoch discipline) or forward
                // them toward the sources.
                let data = if up_rx.is_some() {
                    // `biased` polls the inbox first so control outruns queued
                    // data; both receives are cancel-safe, so the losing
                    // branch simply resumes next iteration.
                    let incoming = {
                        let rx_up = up_rx.as_mut().expect("checked above");
                        tokio::select! {
                            biased;
                            ev = rx_up.recv() => Incoming::Upstream(ev),
                            msg = rx.recv() => Incoming::Data(msg),
                        }
                    };
                    match incoming {
                        Incoming::Upstream(Some(event)) => {
                            let _ = handle_upstream_hop(
                                &name,
                                &mut element,
                                &event,
                                &outputs,
                                &src_pad,
                                &probe_registry,
                                &own_epoch,
                                &bus,
                                &up_parents,
                                &mut last_seek_epoch,
                                false,
                                &mut pending_translation,
                                None,
                            )
                            .await;
                            continue;
                        }
                        Incoming::Upstream(None) => {
                            // Inbox closed (handle dropped): stop selecting.
                            up_rx = None;
                            continue;
                        }
                        Incoming::Data(msg) => msg,
                    }
                } else {
                    rx.recv().await
                };
                match data {
                    Some(Message::Buffer(buffer, epoch)) => {
                        count += 1;
                        tracing::debug!(
                            "transform '{}': received buffer {} ({} bytes)",
                            name,
                            count,
                            buffer.len()
                        );

                        // Pre-seek data — shed at receive speed (#157).
                        if rx.is_stale(epoch, &own_epoch) {
                            tracers.notify_drop(&name);
                            continue;
                        }
                        // Outputs must stamp >= our own cell: a mid-graph
                        // seek handler (Queue2) bumps it while upstream
                        // never does (#163 phase C).
                        in_epoch = epoch.max(own_epoch.load(Ordering::Acquire));

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
                        let result = guard(
                            &name,
                            hybrid_process(&mut element, inline_dispatch, Some(buffer)),
                        )
                        .await;
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
                                send_output(out, in_epoch, &outputs, &output_bridges, &tracers)
                                    .await;
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
                    Some(Message::Event(event)) => {
                        match probe_registry.invoke_event(&sink_pad, &event, true) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }
                        if let Event::FlushStart = &event {
                            // Drain the element's internal state and *discard*
                            // it: pending reordered frames from before the
                            // seek must not surface after it. The stale
                            // backlog ahead of this event was epoch-dropped,
                            // so this runs before any fresh data arrives.
                            if let Err(e) = guard(&name, element.flush()).await {
                                tracing::warn!("flush error in '{}': {}", name, e);
                            }
                            shed.reset();
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
                    Some(Message::Error(err)) => {
                        // Terminal: pass the reason on rather than flushing and
                        // reporting a clean end.
                        tracing::error!("transform '{}': upstream failed: {}", name, err);
                        send_error(&outputs, &output_bridges, &err).await;
                        break;
                    }
                    Some(Message::Eos) => {
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
                                    send_output(
                                        buffer,
                                        in_epoch,
                                        &outputs,
                                        &output_bridges,
                                        &tracers,
                                    )
                                    .await;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("flush error in '{}': {}", name, e);
                            }
                        }
                        send_eos(&outputs, &output_bridges).await;
                        break;
                    }
                    None => {
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
            //
            // Flush/pause do not cross bridges: the RT path carries no events,
            // so bridge buffers have no epoch of their own. Outputs are
            // stamped with the current global — RT-fed subtrees cannot seek,
            // so nothing they emit is ever pre-seek.
            loop {
                while let Some(buffer) = bridge.try_pop() {
                    count += 1;

                    match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }

                    tracers.notify_buffer(&name, &buffer);
                    let result = guard(
                        &name,
                        hybrid_process(&mut element, inline_dispatch, Some(buffer)),
                    )
                    .await;
                    tracers.notify_buffer_processed(&name);

                    match result {
                        Ok(Some(out)) => {
                            match probe_registry.invoke_buffer(&src_pad, &out) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            shed.reset();
                            let epoch = own_epoch.load(Ordering::Acquire);
                            send_output(out, epoch, &outputs, &output_bridges, &tracers).await;
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
                                let epoch = own_epoch.load(Ordering::Acquire);
                                send_output(buffer, epoch, &outputs, &output_bridges, &tracers)
                                    .await;
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
    epoch: u64,
    outputs_by_pad: &HashMap<String, Vec<OutputBranch>>,
    tracers: &TracerRegistry,
    unrouted: &mut u64,
    segment_pads: &mut HashSet<String>,
    reanchor: Option<&PadReanchor>,
    src_pad: &crate::pipeline::probe::PadRef,
    probes: &ProbeRegistry,
) -> Option<SegmentEvent> {
    let mut anchored = None;
    match outputs_by_pad.get(pad) {
        Some(branches) => {
            // Per-pad lazy initial segment, anchored at this pad's first
            // buffer (#165): each elementary stream carries its own PTS
            // domain, so pads anchor independently.
            if segment_pads.insert(pad.to_string()) {
                let seg = anchor_segment_for(&buffer, reanchor);
                emit_event(branches, src_pad, probes, Event::Segment(seg.clone())).await;
                anchored = Some(seg);
            }
            broadcast(branches, buffer, epoch, tracers).await;
        }
        None if pad.is_empty() => {
            // Legacy broadcast pad: one shared segment for every branch.
            if segment_pads.insert(String::new()) {
                let seg = anchor_segment_for(&buffer, reanchor);
                for branches in outputs_by_pad.values() {
                    emit_event(branches, src_pad, probes, Event::Segment(seg.clone())).await;
                }
                anchored = Some(seg);
            }
            for branches in outputs_by_pad.values() {
                broadcast(branches, buffer.clone(), epoch, tracers).await;
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
    anchored
}

#[allow(clippy::too_many_arguments)]
fn spawn_demuxer_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<InputBranch>,
    outputs_by_pad: HashMap<String, Vec<OutputBranch>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    stop: Arc<AtomicBool>,
    mut upstream: Option<UpstreamHop>,
    own_epoch: Arc<AtomicU64>,
    pause_rx: watch::Receiver<bool>,
    bus: BusHandle,
    warn_unhandled_seek: bool,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    tokio::spawn(reporting(share, async move {
        let inline_dispatch = element.dispatches_inline();
        tracing::debug!("demuxer '{}' started", name);
        events.send_node_started(&name);

        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;
        let mut unrouted: u64 = 0;
        // Control events (seek) broadcast their flush sequence to every pad.
        let all_branches: Vec<OutputBranch> = outputs_by_pad.values().flatten().cloned().collect();

        // #165: this node OWNS its pads' StreamStart/Segment — eager
        // per-pad StreamStart now, lazy per-pad Segment at each pad's first
        // buffer. Upstream StreamStart/Segment (a fed demuxer's byte-domain
        // input) is offered to the element and swallowed below.
        for (pad, branches) in &outputs_by_pad {
            emit_event(
                branches,
                &src_pad,
                &probe_registry,
                Event::StreamStart(StreamStartEvent::new(format!("{name}/{pad}"))),
            )
            .await;
        }
        let mut segment_pads: HashSet<String> = HashSet::new();
        // #165: source-style demuxers accumulate non-flushing-seek base like
        // sources. One tracker for the node (max PTS across pads -- the seek
        // segment is broadcast to all pads anyway); ~one frame of A/V skew
        // in base is a documented approximation.
        let mut segment_tracker = SegmentTracker::default();
        // Fed branch only (#165): the translated seek's segment shape for
        // pad re-anchors, and whether the next pad anchor should (re)install
        // into the tracker — true at start and after every un-anchor, so
        // "one node, one base" survives re-anchor cycles.
        let mut pad_reanchor: Option<PadReanchor> = None;
        let mut tracker_needs_install = true;
        let mut segment_seek: Option<ActiveSegmentSeek> = None;
        let mut last_seek_epoch: u64 = 0;
        // #163 phase B: set when this element converts a seek into another
        // format and forwards it upstream; cleared when the completion is
        // reported in the format the application asked in.
        let mut pending_translation: Option<PendingTranslation> = None;

        if let Some(mut rx) = inputs.into_iter().next() {
            // Same input-epoch rule as spawn_transform_task: outputs carry the
            // epoch of the input they came from.
            let mut in_epoch: u64 = 0;
            let (mut up_rx, up_parents) = match upstream {
                Some(hop) => (Some(hop.rx), hop.parents),
                None => (None, Vec::new()),
            };
            loop {
                // Upstream events from downstream take priority (#163):
                // handle or forward them between data messages.
                let data = if up_rx.is_some() {
                    // `biased` polls the inbox first so control outruns queued
                    // data; both receives are cancel-safe, so the losing
                    // branch simply resumes next iteration.
                    let incoming = {
                        let rx_up = up_rx.as_mut().expect("checked above");
                        tokio::select! {
                            biased;
                            ev = rx_up.recv() => Incoming::Upstream(ev),
                            msg = rx.recv() => Incoming::Data(msg),
                        }
                    };
                    match incoming {
                        Incoming::Upstream(Some(event)) => {
                            let _ = handle_upstream_hop(
                                &name,
                                &mut element,
                                &event,
                                &all_branches,
                                &src_pad,
                                &probe_registry,
                                &own_epoch,
                                &bus,
                                &up_parents,
                                &mut last_seek_epoch,
                                warn_unhandled_seek,
                                &mut pending_translation,
                                Some(&mut segment_tracker),
                            )
                            .await;
                            continue;
                        }
                        Incoming::Upstream(None) => {
                            // Inbox closed (handle dropped): stop selecting.
                            up_rx = None;
                            continue;
                        }
                        Incoming::Data(msg) => msg,
                    }
                } else {
                    rx.recv().await
                };
                match data {
                    Some(Message::Buffer(buffer, epoch)) => {
                        count += 1;

                        // Pre-seek data — shed at receive speed (#157).
                        if rx.is_stale(epoch, &own_epoch) {
                            tracers.notify_drop(&name);
                            continue;
                        }
                        // Outputs must stamp >= our own cell: a mid-graph
                        // seek handler (Queue2) bumps it while upstream
                        // never does (#163 phase C).
                        in_epoch = epoch.max(own_epoch.load(Ordering::Acquire));

                        match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }

                        // Same ordering rule as spawn_transform_task: the
                        // processed notification must land before the
                        // broadcast, or downstream back-pressure is billed as
                        // demux time.
                        tracers.notify_buffer(&name, &buffer);
                        let result = guard(
                            &name,
                            hybrid_process_demux(&mut element, inline_dispatch, Some(buffer)),
                        )
                        .await;
                        tracers.notify_buffer_processed(&name);

                        // ACCURATE refinement (#173): the element saw the
                        // first post-flush PTS inside `process_demux` and may
                        // have staged a corrected seek. Collect it BEFORE the
                        // routing loop — a pending correction means this
                        // batch is the mis-landed round, whose SeekDone must
                        // be held (the next round re-arms the same
                        // `pending_translation`).
                        let refining = match element.take_upstream_event() {
                            Some(event) if !up_parents.is_empty() => {
                                tracing::debug!(
                                    "demuxer '{}': forwarding refined {} upstream",
                                    name,
                                    event.name()
                                );
                                forward_to_parents(&up_parents, &event);
                                true
                            }
                            Some(event) => {
                                bus.post_warning(
                                    format!(
                                        "'{name}' staged an upstream {} but has no \
                                         upstream peer — dropped",
                                        event.name()
                                    ),
                                    None,
                                );
                                false
                            }
                            None => false,
                        };

                        match result {
                            Ok(DemuxResult::Routed(routed)) => {
                                for (pad, out) in routed {
                                    match probe_registry.invoke_buffer(&src_pad, &out) {
                                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                        _ => {}
                                    }
                                    let pts = out.metadata().pts;
                                    let anchored = route_demux_buffer(
                                        &name,
                                        &pad,
                                        out,
                                        in_epoch,
                                        &outputs_by_pad,
                                        &tracers,
                                        &mut unrouted,
                                        &mut segment_pads,
                                        pad_reanchor.as_ref(),
                                        &src_pad,
                                        &probe_registry,
                                    )
                                    .await;
                                    // Same "one node, one base" rule as the
                                    // source-style branch: the first pad
                                    // anchoring after an un-anchor installs
                                    // the node's tracker segment (#165).
                                    if let Some(seg) = &anchored
                                        && tracker_needs_install
                                    {
                                        segment_tracker.installed(seg);
                                        tracker_needs_install = false;
                                    }
                                    segment_tracker.observe(pts);
                                    // #163 phase B: a seek this demuxer
                                    // translated completes here, not at the
                                    // source. The source answered in BYTES,
                                    // which is not the question the
                                    // application asked; the first post-flush
                                    // buffer's PTS is the honest landing in
                                    // the format it did ask in. A seek still
                                    // being refined does not complete (#173).
                                    if anchored.is_some()
                                        && !refining
                                        && let Some(pt) = pending_translation.take()
                                    {
                                        bus.post(crate::pipeline::bus::MessageKind::SeekDone {
                                            seqnum: pt.seqnum,
                                            source: name.clone(),
                                            format: pt.format,
                                            position: pts.to_option().map(|p| p.nanos()),
                                        });
                                    }
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
                    Some(Message::Event(event)) => {
                        match probe_registry.invoke_event(&sink_pad, &event, true) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }
                        if let Event::FlushStart = &event {
                            // Discard buffered state; stale inputs ahead of
                            // this event were already epoch-dropped.
                            if let Err(e) = guard(&name, element.flush_demux()).await {
                                tracing::warn!("flush error in '{}': {}", name, e);
                            }
                        }
                        // #165: an in-band Segment from upstream marks the
                        // exact boundary where the input timeline changed.
                        // While a translated seek is pending, that boundary
                        // is where the pads re-anchor with the original
                        // seek's shape — rate, stop, and (for a NON-flushing
                        // seek, whose only boundary marker this is: no
                        // FlushStop ever comes back) the base accumulated
                        // through the queued drain, taken here so running
                        // time stays monotonic across the FIFO boundary.
                        if let Event::Segment(_) = &event
                            && let Some(pt) = &pending_translation
                        {
                            pad_reanchor = Some(PadReanchor {
                                rate: pt.rate,
                                base: if pt.flushing {
                                    0
                                } else {
                                    segment_tracker.accumulated_base()
                                },
                                stop: pt.stop.map(ClockTime::from_nanos),
                            });
                            segment_pads.clear();
                            tracker_needs_install = true;
                        }
                        // Events go to every output pad, like EOS does.
                        if let Some(fwd) = element.handle_downstream_event(event) {
                            match &fwd {
                                // This node owns its pads' stream identity:
                                // the input-side StreamStart/Segment describe
                                // the upstream (byte) domain, and forwarding
                                // them would mislabel the elementary streams.
                                // The element saw them above — that is the
                                // hook a TIME→BYTES-translating demuxer will
                                // use (#163 phase B). Swallowed before the
                                // src-pad probes: they never reach the wire.
                                Event::StreamStart(_) | Event::Segment(_) => {
                                    tracing::debug!(
                                        "demuxer '{name}': swallowing upstream {}",
                                        fwd.name()
                                    );
                                    continue;
                                }
                                // After an upstream flush (e.g. a byte seek on
                                // the source below a fed demuxer), each pad
                                // re-anchors with a fresh Time segment at its
                                // first post-seek buffer.
                                Event::FlushStop(_) => {
                                    segment_pads.clear();
                                    tracker_needs_install = true;
                                }
                                _ => {}
                            }
                            match probe_registry.invoke_event(&src_pad, &fwd, true) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            for branches in outputs_by_pad.values() {
                                broadcast_event(branches, &fwd).await;
                            }
                        }
                    }
                    Some(Message::Error(err)) => {
                        tracing::error!("demuxer '{}': upstream failed: {}", name, err);
                        for branches in outputs_by_pad.values() {
                            broadcast_error(branches, &err).await;
                        }
                        break;
                    }
                    Some(Message::Eos) | None => {
                        // Drain the demuxer through the routed path
                        // (process_demux(None) → Demuxer::produce), so a
                        // trailing partial frame keeps its pad — the old
                        // flush() broadcast sent it down every branch.
                        loop {
                            match guard(
                                &name,
                                hybrid_process_demux(&mut element, inline_dispatch, None),
                            )
                            .await
                            {
                                Ok(DemuxResult::Routed(routed)) if !routed.is_empty() => {
                                    for (pad, out) in routed {
                                        route_demux_buffer(
                                            &name,
                                            &pad,
                                            out,
                                            in_epoch,
                                            &outputs_by_pad,
                                            &tracers,
                                            &mut unrouted,
                                            &mut segment_pads,
                                            pad_reanchor.as_ref(),
                                            &src_pad,
                                            &probe_registry,
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
                        // A *fed* demuxer's `produce` is never called, so the
                        // loop above drained nothing; whatever its parser was
                        // still assembling comes out here, on its own pad.
                        match guard(&name, element.flush_demux()).await {
                            Ok(routed) => {
                                for (pad, out) in routed {
                                    route_demux_buffer(
                                        &name,
                                        &pad,
                                        out,
                                        in_epoch,
                                        &outputs_by_pad,
                                        &tracers,
                                        &mut unrouted,
                                        &mut segment_pads,
                                        pad_reanchor.as_ref(),
                                        &src_pad,
                                        &probe_registry,
                                    )
                                    .await;
                                }
                            }
                            Err(e) => tracing::warn!("EOS flush error in '{}': {}", name, e),
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
                if let Some(hop) = upstream.as_mut() {
                    while let Ok(event) = hop.rx.try_recv() {
                        let outcome = handle_upstream_hop(
                            &name,
                            &mut element,
                            &event,
                            &all_branches,
                            &src_pad,
                            &probe_registry,
                            &own_epoch,
                            &bus,
                            &hop.parents,
                            &mut last_seek_epoch,
                            warn_unhandled_seek,
                            &mut pending_translation,
                            Some(&mut segment_tracker),
                        )
                        .await;
                        if outcome.segment_emitted {
                            // The seek's segment went to every pad.
                            segment_pads.extend(outputs_by_pad.keys().cloned());
                        }
                        track_segment_seek(&outcome, &mut segment_seek);
                    }
                }

                // Runtime pause: a source-style demuxer is the pipeline's
                // producer, so it takes the same gate as spawn_source_task —
                // #156's root cause was that it didn't, leaving pause with
                // nothing to gate in demuxer-rooted pipelines. Control events
                // still drain so a seek can land while paused. (The fed
                // branch above is deliberately ungated, like transforms: it
                // must keep moving for a flush to propagate, and backpressure
                // parks it once the producer stops.)
                while *pause_rx.borrow() && !stop.load(Ordering::Acquire) {
                    if let Some(hop) = upstream.as_mut() {
                        while let Ok(event) = hop.rx.try_recv() {
                            let outcome = handle_upstream_hop(
                                &name,
                                &mut element,
                                &event,
                                &all_branches,
                                &src_pad,
                                &probe_registry,
                                &own_epoch,
                                &bus,
                                &hop.parents,
                                &mut last_seek_epoch,
                                warn_unhandled_seek,
                                &mut pending_translation,
                                Some(&mut segment_tracker),
                            )
                            .await;
                            if outcome.segment_emitted {
                                segment_pads.extend(outputs_by_pad.keys().cloned());
                            }
                            track_segment_seek(&outcome, &mut segment_seek);
                        }
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }

                match guard(
                    &name,
                    hybrid_process_demux(&mut element, inline_dispatch, None),
                )
                .await
                {
                    Ok(DemuxResult::Routed(routed)) => {
                        // Same stamp-at-send rule as spawn_source_task: a seek
                        // handled in this iteration's control drain already
                        // bumped the epoch.
                        let epoch = own_epoch.load(Ordering::Acquire);
                        for (pad, out) in routed {
                            count += 1;
                            match probe_registry.invoke_buffer(&src_pad, &out) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            tracers.notify_buffer(&name, &out);
                            let pts = out.metadata().pts;
                            let anchored = route_demux_buffer(
                                &name,
                                &pad,
                                out,
                                epoch,
                                &outputs_by_pad,
                                &tracers,
                                &mut unrouted,
                                &mut segment_pads,
                                None,
                                &src_pad,
                                &probe_registry,
                            )
                            .await;
                            // The first pad's lazy segment is the node's
                            // running-time anchor (#165); later pads anchor
                            // on the wire but not in the tracker -- one
                            // node, one base.
                            if let Some(seg) = &anchored
                                && segment_tracker.current.is_none()
                            {
                                segment_tracker.installed(seg);
                            }
                            segment_tracker.observe(pts);
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
                        // SEGMENT seek (#165): post SegmentDone once and idle
                        // awaiting the follow-up seek — same discipline as
                        // spawn_source_task's Eos arm.
                        if let Some(ss) = segment_seek.as_mut() {
                            if !ss.done_posted {
                                ss.done_posted = true;
                                bus.post(crate::pipeline::bus::MessageKind::SegmentDone {
                                    seqnum: ss.seqnum,
                                    source: name.clone(),
                                    format: ss.format,
                                    position: ss.stop,
                                });
                                tracing::info!(
                                    "demuxer '{}': segment done (seek {})",
                                    name,
                                    ss.seqnum
                                );
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                            continue;
                        }
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
    inputs_by_pad: HashMap<String, Vec<InputBranch>>,
    outputs: Vec<OutputBranch>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    own_epoch: Arc<AtomicU64>,
    upstream: Option<UpstreamHop>,
    bus: BusHandle,
    share: TaskGuard,
) -> JoinHandle<Result<()>> {
    tokio::spawn(reporting(share, async move {
        let inline_dispatch = element.dispatches_inline();
        tracing::debug!("muxer '{}' started", name);
        events.send_node_started(&name);

        // A muxer has several sink pads, but probes are registered per node
        // rather than per named pad here, so all inputs share one PadRef —
        // consistent with how a transform's single sink pad is handled.
        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;

        // #165: the muxed output is a new stream — this node owns its
        // StreamStart/Segment, and the per-input ones are swallowed below
        // (forwarding N inputs' segments would mislabel the one output).
        emit_event(
            &outputs,
            &src_pad,
            &probe_registry,
            Event::StreamStart(StreamStartEvent::new(&name)),
        )
        .await;
        let mut segment_sent = false;

        // All input branches behind one rotating fair poll (#181). Its
        // predecessor re-armed a `recv_one` future into a `FuturesUnordered`
        // per received message — one `Arc<Task>` allocation per buffer per
        // branch, the last per-buffer allocation on the data path.
        let mut inputs = MuxInputs::new(inputs_by_pad);

        let total = inputs.len();
        let mut eos_count = 0;
        // Flushes are counted across inputs like EOS: FlushStart forwards on
        // the first arrival, FlushStop once every started flush has stopped —
        // forwarding each input's pair verbatim would flush the output twice.
        let mut active_flushes: usize = 0;
        // Outputs carry the newest epoch seen across inputs — a muxer's output
        // interleaves all of them.
        let mut in_epoch: u64 = 0;

        let (mut up_rx, up_parents) = match upstream {
            Some(hop) => (Some(hop.rx), hop.parents),
            None => (None, Vec::new()),
        };
        let mut last_seek_epoch: u64 = 0;
        // #163 phase B: set when this element converts a seek into another
        // format and forwards it upstream; cleared when the completion is
        // reported in the format the application asked in.
        let mut pending_translation: Option<PendingTranslation> = None;
        loop {
            // Upstream events (#163) take priority over data; a muxer that
            // does not handle one forwards it to every input branch's
            // parent. The inbox is tokio mpsc (cancel-safe), and
            // `MuxInputs::next_msg` consumes a message only in the poll
            // that returns it, with the receivers owned outside the future
            // — both branches are safe to lose.
            let item = match up_rx.as_mut() {
                Some(rx_up) => {
                    tokio::select! {
                        biased;
                        ev = rx_up.recv() => Err(ev),
                        item = inputs.next_msg() => Ok(item),
                    }
                }
                None => Ok(inputs.next_msg().await),
            };
            let item = match item {
                Err(Some(event)) => {
                    let _ = handle_upstream_hop(
                        &name,
                        &mut element,
                        &event,
                        &outputs,
                        &src_pad,
                        &probe_registry,
                        &own_epoch,
                        &bus,
                        &up_parents,
                        &mut last_seek_epoch,
                        false,
                        &mut pending_translation,
                        None,
                    )
                    .await;
                    continue;
                }
                Err(None) => {
                    // Inbox closed (handle dropped): stop selecting on it.
                    up_rx = None;
                    continue;
                }
                Ok(i) => i,
            };
            let Some((idx, msg)) = item else {
                // Every branch retired: same terminal condition as the
                // drained FuturesUnordered before it.
                break;
            };
            // A branch that just ended (EOS, error, closed) stops being
            // polled; live branches simply stay in the set.
            if !matches!(msg, Some(Message::Buffer(..) | Message::Event(_))) {
                inputs.retire(idx);
            }
            match msg {
                Some(Message::Buffer(buffer, epoch)) => {
                    count += 1;

                    // Pre-seek data — shed at receive speed (#157).
                    if inputs.is_stale(idx, epoch, &own_epoch) {
                        tracers.notify_drop(&name);
                        continue;
                    }
                    in_epoch = in_epoch.max(epoch).max(own_epoch.load(Ordering::Acquire));

                    match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }

                    // Muxers buffer and interleave, so most inputs produce no
                    // output — LatencyTracer sees the pair regardless, which is
                    // what makes "how long is this muxer taking" answerable at
                    // all. It previously had no instrumentation whatsoever.
                    tracers.notify_buffer(&name, &buffer);
                    let result = guard(
                        &name,
                        hybrid_process(&mut element, inline_dispatch, Some(buffer)),
                    )
                    .await;
                    tracers.notify_buffer_processed(&name);

                    match result {
                        Ok(Some(out)) => {
                            match probe_registry.invoke_buffer(&src_pad, &out) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            if !segment_sent {
                                segment_sent = true;
                                emit_event(
                                    &outputs,
                                    &src_pad,
                                    &probe_registry,
                                    Event::Segment(initial_segment_for(&out)),
                                )
                                .await;
                            }
                            broadcast(&outputs, out, in_epoch, &tracers).await;
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
                Some(Message::Event(event)) => {
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
                            match &fwd {
                                // The muxed output has its own stream
                                // identity; per-input StreamStart/Segment
                                // describe the input tracks (#165), swallowed
                                // before the src-pad probes.
                                Event::StreamStart(_) | Event::Segment(_) => continue,
                                // Post-flush, the output re-anchors at its
                                // first fresh buffer.
                                Event::FlushStop(_) => segment_sent = false,
                                _ => {}
                            }
                            match probe_registry.invoke_event(&src_pad, &fwd, true) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            broadcast_event(&outputs, &fwd).await;
                        }
                        _ => {}
                    }
                }
                Some(Message::Error(err)) => {
                    // One failed input dooms the muxed stream: the output would
                    // be missing a track from here on. Pass the reason on
                    // immediately rather than waiting for the other inputs.
                    tracing::error!("muxer '{}': an input failed: {}", name, err);
                    broadcast_error(&outputs, &err).await;
                    break;
                }
                Some(Message::Eos) | None => {
                    eos_count += 1;
                    if eos_count >= total {
                        // Flush any remaining data from final processing
                        if let Ok(Some(out)) =
                            guard(&name, hybrid_process(&mut element, inline_dispatch, None)).await
                        {
                            if !segment_sent {
                                segment_sent = true;
                                emit_event(
                                    &outputs,
                                    &src_pad,
                                    &probe_registry,
                                    Event::Segment(initial_segment_for(&out)),
                                )
                                .await;
                            }
                            broadcast(&outputs, out, in_epoch, &tracers).await;
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
                                    if !segment_sent {
                                        segment_sent = true;
                                        emit_event(
                                            &outputs,
                                            &src_pad,
                                            &probe_registry,
                                            Event::Segment(initial_segment_for(&buffer)),
                                        )
                                        .await;
                                    }
                                    broadcast(&outputs, buffer, in_epoch, &tracers).await;
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

    mod segment_tracker {
        use super::super::SegmentTracker;
        use crate::clock::ClockTime;
        use crate::event::SegmentEvent;

        #[test]
        fn no_segment_means_base_zero() {
            let t = SegmentTracker::default();
            assert_eq!(t.accumulated_base(), 0);
        }

        #[test]
        fn accumulates_running_time_of_last_pts() {
            let mut t = SegmentTracker::default();
            t.installed(&SegmentEvent::new_time(ClockTime::from_nanos(1_000), None));
            t.observe(ClockTime::from_nanos(4_000));
            // (4000 - 1000) / 1.0 + 0
            assert_eq!(t.accumulated_base(), 3_000);
        }

        #[test]
        fn nothing_played_falls_back_to_current_base() {
            let mut t = SegmentTracker::default();
            t.installed(&SegmentEvent::new_time(ClockTime::from_nanos(0), None).with_base(7_000));
            // Two queued seeks back to back: the second inherits the
            // first's base rather than resetting to 0.
            assert_eq!(t.accumulated_base(), 7_000);
        }

        #[test]
        fn rate_scales_the_accumulation() {
            let mut t = SegmentTracker::default();
            t.installed(&SegmentEvent::new_time(ClockTime::from_nanos(0), None).with_rate(2.0));
            t.observe(ClockTime::from_nanos(10_000));
            // 10_000 elapsed at 2x = 5_000 of running time.
            assert_eq!(t.accumulated_base(), 5_000);
        }

        #[test]
        fn observe_keeps_the_max() {
            let mut t = SegmentTracker::default();
            t.installed(&SegmentEvent::new_time(ClockTime::ZERO, None));
            t.observe(ClockTime::from_nanos(5_000));
            t.observe(ClockTime::from_nanos(2_000)); // B-frame style regression
            assert_eq!(t.accumulated_base(), 5_000);
        }

        #[test]
        fn bytes_segments_are_ignored() {
            let mut t = SegmentTracker::default();
            t.installed(&SegmentEvent::new_bytes(0, None));
            assert!(t.current.is_none());
            assert_eq!(t.accumulated_base(), 0);
        }

        #[test]
        fn install_resets_last_pts() {
            let mut t = SegmentTracker::default();
            t.installed(&SegmentEvent::new_time(ClockTime::ZERO, None));
            t.observe(ClockTime::from_nanos(9_000));
            t.installed(
                &SegmentEvent::new_time(ClockTime::from_nanos(20_000), None).with_base(9_000),
            );
            // Nothing observed under the new segment yet.
            assert_eq!(t.accumulated_base(), 9_000);
        }
    }
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
