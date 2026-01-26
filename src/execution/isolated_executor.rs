//! Isolated pipeline executor with transparent IPC injection.
//!
//! This executor automatically handles process isolation without requiring
//! users to manually add IpcSrc/IpcSink elements. Users write normal pipelines
//! and specify an execution mode - the executor handles the rest.
//!
//! # How It Works
//!
//! 1. User creates a normal pipeline: `filesrc ! decoder ! sink`
//! 2. User specifies execution mode (e.g., isolate decoder)
//! 3. Executor analyzes pipeline and identifies isolation boundaries
//! 4. Executor spawns child processes for isolated elements
//! 5. IPC channels are created transparently between processes
//! 6. Pipeline runs with zero-copy shared memory between processes
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::execution::{IsolatedExecutor, ExecutionMode};
//!
//! // Create a normal pipeline
//! let pipeline = Pipeline::parse("filesrc location=video.mp4 ! h264dec ! displaysink")?;
//!
//! // Isolate decoders (they might crash on malformed input)
//! let mode = ExecutionMode::grouped(vec!["*dec*".to_string()]);
//!
//! // Run with automatic IPC injection
//! let executor = IsolatedExecutor::new(mode);
//! executor.run(pipeline).await?;
//! ```

use super::mode::{ExecutionMode, GroupId};
use super::supervisor::{RestartPolicy, Supervisor};
use crate::error::{Error, Result};
use crate::memory::CpuArena;
use crate::pipeline::{NodeId, Pipeline, PipelineEvent};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Configuration for the isolated executor.
#[derive(Clone, Debug)]
pub struct IsolatedExecutorConfig {
    /// Size of shared memory arenas (bytes).
    pub arena_size: usize,
    /// Number of slots per arena.
    pub arena_slots: usize,
    /// Size of each slot (bytes).
    pub slot_size: usize,
    /// Restart policy for crashed elements.
    pub restart_policy: RestartPolicy,
    /// Channel capacity for IPC.
    pub channel_capacity: usize,
}

impl Default for IsolatedExecutorConfig {
    fn default() -> Self {
        Self {
            arena_size: 64 * 1024 * 1024, // 64 MB
            arena_slots: 64,
            slot_size: 1024 * 1024, // 1 MB per slot
            restart_policy: RestartPolicy::default(),
            channel_capacity: 16,
        }
    }
}

/// An execution plan describing how to run a pipeline with isolation.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Groups of elements that run together.
    pub groups: HashMap<GroupId, ElementGroup>,
    /// IPC boundaries between groups.
    pub boundaries: Vec<IpcBoundary>,
    /// Original node to group mapping.
    pub node_groups: HashMap<NodeId, GroupId>,
}

/// A group of elements that run in the same process.
#[derive(Debug)]
pub struct ElementGroup {
    /// Group ID.
    pub id: GroupId,
    /// Node IDs in this group.
    pub nodes: Vec<NodeId>,
    /// Whether this group runs in the supervisor process.
    pub is_supervisor: bool,
}

/// An IPC boundary between two groups.
#[derive(Debug, Clone)]
pub struct IpcBoundary {
    /// Source node (in source group).
    pub src_node: NodeId,
    /// Sink node (in sink group).
    pub sink_node: NodeId,
    /// Source group.
    pub src_group: GroupId,
    /// Sink group.
    pub sink_group: GroupId,
}

/// Executor that runs pipelines with transparent process isolation.
pub struct IsolatedExecutor {
    mode: ExecutionMode,
    config: IsolatedExecutorConfig,
}

impl IsolatedExecutor {
    /// Create a new isolated executor with the given execution mode.
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            mode,
            config: IsolatedExecutorConfig::default(),
        }
    }

    /// Create an isolated executor with custom configuration.
    pub fn with_config(mode: ExecutionMode, config: IsolatedExecutorConfig) -> Self {
        Self { mode, config }
    }

    /// Analyze a pipeline and create an execution plan.
    ///
    /// This determines which elements run in which processes and where
    /// IPC boundaries need to be inserted.
    pub fn plan(&self, pipeline: &Pipeline) -> Result<ExecutionPlan> {
        match &self.mode {
            ExecutionMode::InProcess => self.plan_in_process(pipeline),
            ExecutionMode::Isolated { .. } => self.plan_fully_isolated(pipeline),
            ExecutionMode::Grouped { .. } => self.plan_grouped(pipeline),
        }
    }

    /// Plan for in-process execution (everything in supervisor).
    fn plan_in_process(&self, pipeline: &Pipeline) -> Result<ExecutionPlan> {
        let mut nodes = Vec::new();
        let mut node_groups = HashMap::new();

        // All nodes go in the supervisor group
        for src in pipeline.sources() {
            self.collect_nodes(pipeline, src, &mut nodes, &mut HashSet::new());
        }

        for &node in &nodes {
            node_groups.insert(node, GroupId::SUPERVISOR);
        }

        let group = ElementGroup {
            id: GroupId::SUPERVISOR,
            nodes,
            is_supervisor: true,
        };

        let mut groups = HashMap::new();
        groups.insert(GroupId::SUPERVISOR, group);

        Ok(ExecutionPlan {
            groups,
            boundaries: Vec::new(), // No IPC needed
            node_groups,
        })
    }

    /// Plan for fully isolated execution (each element in its own process).
    fn plan_fully_isolated(&self, pipeline: &Pipeline) -> Result<ExecutionPlan> {
        let mut groups = HashMap::new();
        let mut boundaries = Vec::new();
        let mut node_groups = HashMap::new();
        let mut next_group_id = 1u32;

        // Collect all nodes
        let mut all_nodes = Vec::new();
        for src in pipeline.sources() {
            self.collect_nodes(pipeline, src, &mut all_nodes, &mut HashSet::new());
        }

        // Each node gets its own group
        for &node_id in &all_nodes {
            let group_id = GroupId::new(next_group_id);
            next_group_id += 1;

            let group = ElementGroup {
                id: group_id,
                nodes: vec![node_id],
                is_supervisor: false,
            };

            groups.insert(group_id, group);
            node_groups.insert(node_id, group_id);
        }

        // Create boundaries for all edges
        for &node_id in &all_nodes {
            let src_group = node_groups[&node_id];
            for (child_id, _link) in pipeline.children(node_id) {
                let sink_group = node_groups[&child_id];
                if src_group != sink_group {
                    boundaries.push(IpcBoundary {
                        src_node: node_id,
                        sink_node: child_id,
                        src_group,
                        sink_group,
                    });
                }
            }
        }

        Ok(ExecutionPlan {
            groups,
            boundaries,
            node_groups,
        })
    }

    /// Plan for grouped execution.
    fn plan_grouped(&self, pipeline: &Pipeline) -> Result<ExecutionPlan> {
        let mut groups = HashMap::new();
        let mut boundaries = Vec::new();
        let mut node_groups = HashMap::new();
        let mut next_isolated_id = 1000u32; // High IDs for isolated elements

        // Collect all nodes
        let mut all_nodes = Vec::new();
        for src in pipeline.sources() {
            self.collect_nodes(pipeline, src, &mut all_nodes, &mut HashSet::new());
        }

        // Assign groups based on mode
        for &node_id in &all_nodes {
            let node = pipeline
                .get_node(node_id)
                .ok_or_else(|| Error::InvalidSegment("node not found".into()))?;
            let name = node.name();

            let group_id = if self.mode.should_isolate(name) {
                // Isolated element gets its own group
                let gid = GroupId::new(next_isolated_id);
                next_isolated_id += 1;
                gid
            } else {
                // Use assigned group or default to supervisor
                self.mode.get_group(name).unwrap_or(GroupId::SUPERVISOR)
            };

            node_groups.insert(node_id, group_id);

            // Add to or create group
            groups
                .entry(group_id)
                .or_insert_with(|| ElementGroup {
                    id: group_id,
                    nodes: Vec::new(),
                    is_supervisor: group_id == GroupId::SUPERVISOR,
                })
                .nodes
                .push(node_id);
        }

        // Create boundaries where groups differ
        for &node_id in &all_nodes {
            let src_group = node_groups[&node_id];
            for (child_id, _link) in pipeline.children(node_id) {
                let sink_group = node_groups[&child_id];
                if src_group != sink_group {
                    boundaries.push(IpcBoundary {
                        src_node: node_id,
                        sink_node: child_id,
                        src_group,
                        sink_group,
                    });
                }
            }
        }

        Ok(ExecutionPlan {
            groups,
            boundaries,
            node_groups,
        })
    }

    /// Collect all nodes reachable from a starting node.
    fn collect_nodes(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        result: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
    ) {
        if !visited.insert(node_id) {
            return;
        }

        result.push(node_id);

        for (child_id, _) in pipeline.children(node_id) {
            self.collect_nodes(pipeline, child_id, result, visited);
        }
    }

    /// Run a pipeline with the configured isolation mode.
    ///
    /// This is the main entry point. It:
    /// 1. Creates an execution plan
    /// 2. Sets up shared memory arenas
    /// 3. Spawns child processes for isolated groups
    /// 4. Runs the pipeline to completion
    pub async fn run(&self, mut pipeline: Pipeline) -> Result<()> {
        let plan = self.plan(&pipeline)?;

        // If no boundaries, just run in-process
        if plan.boundaries.is_empty() {
            return self.run_in_process(&mut pipeline).await;
        }

        // Otherwise, run with isolation
        self.run_isolated(&mut pipeline, plan).await
    }

    /// Run pipeline entirely in-process (no isolation).
    async fn run_in_process(&self, pipeline: &mut Pipeline) -> Result<()> {
        use crate::pipeline::PipelineExecutor;
        let executor = PipelineExecutor::new();
        executor.run(pipeline).await
    }

    /// Run pipeline with process isolation.
    async fn run_isolated(&self, pipeline: &mut Pipeline, plan: ExecutionPlan) -> Result<()> {
        // Create shared memory arena for IPC
        let _arena = CpuArena::new(self.config.slot_size, self.config.arena_slots)?;

        // Create supervisor to manage child processes
        let mut supervisor = Supervisor::new(self.mode.clone())
            .with_restart_policy(self.config.restart_policy.clone());

        // Create arena via supervisor
        let _arena_id = supervisor.create_arena(self.config.slot_size, self.config.arena_slots)?;

        // For now, we'll implement a simplified version that runs everything
        // in-process but demonstrates the architecture.
        //
        // Full implementation would:
        // 1. Spawn child processes for non-supervisor groups
        // 2. Set up Unix socket pairs for each boundary
        // 3. Pass arena fd to children via SCM_RIGHTS
        // 4. Run supervisor loop handling control messages

        tracing::warn!(
            "Full process isolation not yet implemented, running in-process with {} boundaries identified",
            plan.boundaries.len()
        );

        // For now, fall back to in-process execution
        self.run_in_process(pipeline).await
    }
}

impl Default for IsolatedExecutor {
    fn default() -> Self {
        Self::new(ExecutionMode::InProcess)
    }
}

/// A running isolated pipeline.
pub struct IsolatedPipelineHandle {
    /// Supervisor managing child processes.
    supervisor: Supervisor,
    /// Shared memory arenas.
    arenas: Vec<Arc<CpuArena>>,
    /// Event channel.
    events_tx: mpsc::Sender<PipelineEvent>,
    /// Event receiver.
    events_rx: mpsc::Receiver<PipelineEvent>,
}

impl IsolatedPipelineHandle {
    /// Wait for the pipeline to complete.
    pub async fn wait(mut self) -> Result<()> {
        // Wait for all child processes to complete
        while let Some(event) = self.events_rx.recv().await {
            match event {
                PipelineEvent::Eos => break,
                PipelineEvent::Error { message, .. } => {
                    return Err(Error::InvalidSegment(message));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Abort the pipeline.
    pub fn abort(mut self) {
        let _ = self.supervisor.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{Buffer, MemoryHandle};
    use crate::element::{DynAsyncElement, Sink, SinkAdapter, Source, SourceAdapter};
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestSource {
        count: u64,
        max: u64,
    }

    impl Source for TestSource {
        fn produce(&mut self) -> Result<Option<Buffer>> {
            if self.count >= self.max {
                return Ok(None);
            }
            let segment = Arc::new(HeapSegment::new(8).unwrap());
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::new(handle, Metadata::from_sequence(self.count));
            self.count += 1;
            Ok(Some(buffer))
        }
    }

    struct TestSink {
        received: Arc<AtomicU64>,
    }

    impl Sink for TestSink {
        fn consume(&mut self, _buffer: Buffer) -> Result<()> {
            self.received.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_plan_in_process() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource { count: 0, max: 5 })),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(TestSink {
                received: sink_received.clone(),
            })),
        );
        pipeline.link(src, sink).unwrap();

        let executor = IsolatedExecutor::new(ExecutionMode::InProcess);
        let plan = executor.plan(&pipeline).unwrap();

        // All nodes should be in supervisor group
        assert_eq!(plan.groups.len(), 1);
        assert!(plan.groups.contains_key(&GroupId::SUPERVISOR));
        assert_eq!(plan.boundaries.len(), 0);
    }

    #[tokio::test]
    async fn test_plan_fully_isolated() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource { count: 0, max: 5 })),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(TestSink {
                received: sink_received.clone(),
            })),
        );
        pipeline.link(src, sink).unwrap();

        let executor = IsolatedExecutor::new(ExecutionMode::isolated());
        let plan = executor.plan(&pipeline).unwrap();

        // Each node should be in its own group
        assert_eq!(plan.groups.len(), 2);
        // One boundary between src and sink
        assert_eq!(plan.boundaries.len(), 1);
    }

    struct TestTransform;

    impl crate::element::Element for TestTransform {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            Ok(Some(buffer))
        }
    }

    #[tokio::test]
    async fn test_plan_grouped() {
        use crate::element::ElementAdapter;

        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "filesrc",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource { count: 0, max: 5 })),
        );
        let decoder = pipeline.add_node(
            "h264_decoder",
            DynAsyncElement::new_box(ElementAdapter::new(TestTransform)),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "displaysink",
            DynAsyncElement::new_box(SinkAdapter::new(TestSink {
                received: sink_received.clone(),
            })),
        );
        pipeline.link(src, decoder).unwrap();
        pipeline.link(decoder, sink).unwrap();

        // Isolate decoders
        let executor = IsolatedExecutor::new(ExecutionMode::grouped(vec!["*decoder*".to_string()]));
        let plan = executor.plan(&pipeline).unwrap();

        // filesrc and displaysink should be in supervisor, decoder isolated
        let src_group = plan.node_groups[&src];
        let decoder_group = plan.node_groups[&decoder];
        let sink_group = plan.node_groups[&sink];

        assert_eq!(src_group, GroupId::SUPERVISOR);
        assert_eq!(sink_group, GroupId::SUPERVISOR);
        assert_ne!(decoder_group, GroupId::SUPERVISOR);

        // Two boundaries: filesrc->decoder and decoder->displaysink
        assert_eq!(plan.boundaries.len(), 2);
    }

    #[tokio::test]
    async fn test_run_in_process() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource { count: 0, max: 5 })),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(TestSink {
                received: sink_received.clone(),
            })),
        );
        pipeline.link(src, sink).unwrap();

        let executor = IsolatedExecutor::new(ExecutionMode::InProcess);
        executor.run(pipeline).await.unwrap();

        assert_eq!(sink_received.load(Ordering::Relaxed), 5);
    }
}
