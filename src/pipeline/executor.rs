//! Pipeline executor using Tokio and Kanal channels.
//!
//! The executor spawns a Tokio task for each node in the pipeline and
//! connects them using Kanal channels for zero-copy buffer passing.

use crate::buffer::Buffer;
use crate::element::{ElementDyn, ElementType};
use crate::error::{Error, Result};
use crate::pipeline::{EventReceiver, EventSender, NodeId, Pipeline, PipelineEvent, PipelineState};
use kanal::{AsyncReceiver, AsyncSender, bounded_async};
use std::collections::HashMap;
use tokio::task::JoinHandle;

/// Configuration for the pipeline executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Channel buffer size between elements.
    pub channel_capacity: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 16,
        }
    }
}

impl ExecutorConfig {
    /// Create a new executor config with custom channel capacity.
    pub fn with_capacity(channel_capacity: usize) -> Self {
        Self { channel_capacity }
    }
}

/// Message passed between pipeline elements.
#[derive(Debug)]
enum Message {
    /// A data buffer.
    Buffer(Buffer),
    /// End of stream signal.
    Eos,
}

/// Handle to a running pipeline.
///
/// Allows waiting for completion or stopping the pipeline.
pub struct PipelineHandle {
    /// Join handles for all spawned tasks.
    tasks: Vec<JoinHandle<Result<()>>>,
    /// Event sender for the pipeline.
    events: EventSender,
}

impl PipelineHandle {
    /// Wait for the pipeline to complete.
    ///
    /// Returns `Ok(())` if all elements finished successfully, or the first error encountered.
    pub async fn wait(self) -> Result<()> {
        let mut first_error = None;
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
                    let err = Error::InvalidSegment(format!("task panicked: {e}"));
                    self.events.send_error(err.to_string(), None);
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }

        if first_error.is_none() {
            self.events.send_eos();
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Abort all pipeline tasks.
    pub fn abort(self) {
        for task in self.tasks {
            task.abort();
        }
        self.events.send(PipelineEvent::Stopped);
    }

    /// Subscribe to pipeline events.
    pub fn subscribe(&self) -> EventReceiver {
        self.events.subscribe()
    }

    /// Get the event sender for this pipeline.
    pub fn event_sender(&self) -> &EventSender {
        &self.events
    }
}

/// Executor that runs a pipeline.
pub struct PipelineExecutor {
    config: ExecutorConfig,
}

impl PipelineExecutor {
    /// Create a new executor with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
        }
    }

    /// Create a new executor with custom configuration.
    pub fn with_config(config: ExecutorConfig) -> Self {
        Self { config }
    }

    /// Run the pipeline to completion.
    ///
    /// This is a convenience method that starts the pipeline and waits for it to finish.
    pub async fn run(&self, pipeline: &mut Pipeline) -> Result<()> {
        let handle = self.start(pipeline)?;
        handle.wait().await
    }

    /// Start the pipeline and return a handle.
    ///
    /// The pipeline runs in the background. Use the handle to wait for completion
    /// or abort the pipeline.
    pub fn start(&self, pipeline: &mut Pipeline) -> Result<PipelineHandle> {
        // Validate pipeline structure
        pipeline.validate()?;

        // Create event sender
        let events = EventSender::new(256);

        // Emit state change event
        let old_state = pipeline.state();
        pipeline.set_state(PipelineState::Running);
        events.send_state_changed(old_state, PipelineState::Running);
        events.send(PipelineEvent::Started);

        // Build the channel network
        let channels = self.build_channels(pipeline);

        // Spawn tasks for each node
        let tasks = self.spawn_tasks(pipeline, channels, &events)?;

        Ok(PipelineHandle { tasks, events })
    }

    /// Build channels between connected nodes.
    fn build_channels(&self, pipeline: &Pipeline) -> ChannelNetwork {
        let mut network = ChannelNetwork::new();

        // For each node, create output channels to its children
        for src_id in pipeline.sources() {
            self.build_channels_from(pipeline, src_id, &mut network);
        }

        network
    }

    /// Recursively build channels starting from a node.
    fn build_channels_from(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        network: &mut ChannelNetwork,
    ) {
        let children = pipeline.children(node_id);

        for (child_id, _link) in children {
            // Create channel if not already exists
            if !network.has_channel(node_id, child_id) {
                let (tx, rx) = bounded_async::<Message>(self.config.channel_capacity);
                network.add_channel(node_id, child_id, tx, rx);
            }

            // Recurse to children
            self.build_channels_from(pipeline, child_id, network);
        }
    }

    /// Spawn Tokio tasks for each node.
    fn spawn_tasks(
        &self,
        pipeline: &mut Pipeline,
        mut channels: ChannelNetwork,
        events: &EventSender,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut tasks = Vec::new();

        // Collect node metadata
        let node_ids: Vec<NodeId> = pipeline
            .sources()
            .into_iter()
            .chain(self.collect_reachable_nodes(pipeline))
            .collect();

        // Deduplicate while preserving order
        let mut seen = std::collections::HashSet::new();
        let node_ids: Vec<NodeId> = node_ids.into_iter().filter(|id| seen.insert(*id)).collect();

        for node_id in node_ids {
            let node = pipeline
                .get_node_mut(node_id)
                .ok_or_else(|| Error::InvalidSegment("node not found".into()))?;

            let element_type = node.element_type();
            let node_name = node.name().to_string();

            // Take the element out of the node for execution
            let element = node.take_element().ok_or_else(|| {
                Error::InvalidSegment(format!("element '{}' already taken", node_name))
            })?;

            // Get channels for this node
            let inputs = channels.take_inputs(node_id);
            let outputs = channels.take_outputs(node_id);

            let events_clone = events.clone();
            match element_type {
                ElementType::Source => {
                    let task = spawn_source_task(node_name, element, outputs, events_clone);
                    tasks.push(task);
                }
                ElementType::Sink => {
                    let task = spawn_sink_task(node_name, element, inputs, events_clone);
                    tasks.push(task);
                }
                ElementType::Transform => {
                    let task =
                        spawn_transform_task(node_name, element, inputs, outputs, events_clone);
                    tasks.push(task);
                }
            }
        }

        Ok(tasks)
    }

    /// Collect all nodes reachable from sources.
    fn collect_reachable_nodes(&self, pipeline: &Pipeline) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for src in pipeline.sources() {
            self.collect_reachable_from(pipeline, src, &mut result, &mut visited);
        }

        result
    }

    fn collect_reachable_from(
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
            self.collect_reachable_from(pipeline, child_id, result, visited);
        }
    }
}

impl Default for PipelineExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Network of channels connecting pipeline nodes.
struct ChannelNetwork {
    /// Channels indexed by (src_node, sink_node).
    channels: HashMap<(NodeId, NodeId), (AsyncSender<Message>, AsyncReceiver<Message>)>,
    /// Output channels per node (node -> list of senders).
    outputs: HashMap<NodeId, Vec<AsyncSender<Message>>>,
    /// Input channels per node (node -> list of receivers).
    inputs: HashMap<NodeId, Vec<AsyncReceiver<Message>>>,
}

impl ChannelNetwork {
    fn new() -> Self {
        Self {
            channels: HashMap::new(),
            outputs: HashMap::new(),
            inputs: HashMap::new(),
        }
    }

    fn has_channel(&self, src: NodeId, sink: NodeId) -> bool {
        self.channels.contains_key(&(src, sink))
    }

    fn add_channel(
        &mut self,
        src: NodeId,
        sink: NodeId,
        tx: AsyncSender<Message>,
        rx: AsyncReceiver<Message>,
    ) {
        self.channels.insert((src, sink), (tx.clone(), rx.clone()));
        self.outputs.entry(src).or_default().push(tx);
        self.inputs.entry(sink).or_default().push(rx);
    }

    fn take_outputs(&mut self, node: NodeId) -> Vec<AsyncSender<Message>> {
        self.outputs.remove(&node).unwrap_or_default()
    }

    fn take_inputs(&mut self, node: NodeId) -> Vec<AsyncReceiver<Message>> {
        self.inputs.remove(&node).unwrap_or_default()
    }
}

/// Spawn a task for a source element.
fn spawn_source_task(
    name: String,
    mut element: Box<dyn ElementDyn>,
    outputs: Vec<AsyncSender<Message>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("source task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        // Produce buffers until EOS
        loop {
            match element.process(None) {
                Ok(Some(buffer)) => {
                    buffers_processed += 1;
                    // Send buffer to all outputs
                    for tx in &outputs {
                        if tx.send(Message::Buffer(buffer.clone())).await.is_err() {
                            // Receiver dropped
                            tracing::warn!("source '{}': downstream receiver dropped", name);
                        }
                    }
                }
                Ok(None) => {
                    // EOS - send EOS to all outputs
                    tracing::debug!("source '{}' reached EOS", name);
                    for tx in &outputs {
                        let _ = tx.send(Message::Eos).await;
                    }
                    break;
                }
                Err(e) => {
                    tracing::error!("source '{}' error: {}", name, e);
                    events.send_error(e.to_string(), Some(name.clone()));
                    return Err(e);
                }
            }
        }

        tracing::debug!("source task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
        Ok(())
    })
}

/// Spawn a task for a sink element.
fn spawn_sink_task(
    name: String,
    mut element: Box<dyn ElementDyn>,
    inputs: Vec<AsyncReceiver<Message>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("sink task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        // For multiple inputs, we'd need to select/merge them
        // For now, assume single input
        if let Some(rx) = inputs.into_iter().next() {
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        buffers_processed += 1;
                        // Process the buffer through the sink element
                        if let Err(e) = element.process(Some(buffer)) {
                            tracing::error!("sink '{}' error: {}", name, e);
                            events.send_error(e.to_string(), Some(name.clone()));
                            return Err(e);
                        }
                    }
                    Ok(Message::Eos) => {
                        tracing::debug!("sink '{}' received EOS", name);
                        break;
                    }
                    Err(_) => {
                        // Channel closed
                        tracing::debug!("sink '{}': channel closed", name);
                        break;
                    }
                }
            }
        }

        tracing::debug!("sink task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
        Ok(())
    })
}

/// Spawn a task for a transform element.
fn spawn_transform_task(
    name: String,
    mut element: Box<dyn ElementDyn>,
    inputs: Vec<AsyncReceiver<Message>>,
    outputs: Vec<AsyncSender<Message>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("transform task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        // For now, assume single input
        if let Some(rx) = inputs.into_iter().next() {
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        buffers_processed += 1;
                        // Process the buffer through the transform element
                        match element.process(Some(buffer)) {
                            Ok(Some(out_buffer)) => {
                                // Send transformed buffer to all outputs
                                for tx in &outputs {
                                    if tx.send(Message::Buffer(out_buffer.clone())).await.is_err() {
                                        tracing::warn!(
                                            "transform '{}': downstream receiver dropped",
                                            name
                                        );
                                    }
                                }
                            }
                            Ok(None) => {
                                // Element filtered out the buffer, don't forward
                                tracing::trace!("transform '{}' filtered out buffer", name);
                            }
                            Err(e) => {
                                tracing::error!("transform '{}' error: {}", name, e);
                                events.send_error(e.to_string(), Some(name.clone()));
                                return Err(e);
                            }
                        }
                    }
                    Ok(Message::Eos) => {
                        // Forward EOS to all outputs
                        tracing::debug!("transform '{}' received EOS", name);
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                    Err(_) => {
                        // Channel closed, forward EOS
                        tracing::debug!("transform '{}': channel closed", name);
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                }
            }
        }

        tracing::debug!("transform task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::element::{Element, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter};
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct CountingSource {
        count: u64,
        max: u64,
    }

    impl Source for CountingSource {
        fn produce(&mut self) -> Result<Option<Buffer>> {
            if self.count >= self.max {
                return Ok(None);
            }
            let segment = Arc::new(HeapSegment::new(8).unwrap());
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::new(handle, Metadata::with_sequence(self.count));
            self.count += 1;
            Ok(Some(buffer))
        }
    }

    struct CountingSink {
        received: Arc<AtomicU64>,
    }

    impl Sink for CountingSink {
        fn consume(&mut self, _buffer: Buffer) -> Result<()> {
            self.received.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct PassThrough;

    impl Element for PassThrough {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            Ok(Some(buffer))
        }
    }

    struct FilterEven;

    impl Element for FilterEven {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            // Only pass buffers with even sequence numbers
            if buffer.metadata().sequence % 2 == 0 {
                Ok(Some(buffer))
            } else {
                Ok(None)
            }
        }
    }

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = PipelineExecutor::new();
        assert_eq!(executor.config.channel_capacity, 16);

        let executor = PipelineExecutor::with_config(ExecutorConfig::with_capacity(32));
        assert_eq!(executor.config.channel_capacity, 32);
    }

    #[tokio::test]
    async fn test_simple_pipeline_execution() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            Box::new(SourceAdapter::new(CountingSource { count: 0, max: 5 })),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            Box::new(SinkAdapter::new(CountingSink {
                received: sink_received.clone(),
            })),
        );

        pipeline.link(src, sink).unwrap();

        let executor = PipelineExecutor::new();
        executor.run(&mut pipeline).await.unwrap();

        // Verify that the sink received all 5 buffers
        assert_eq!(sink_received.load(Ordering::Relaxed), 5);
        assert_eq!(pipeline.state(), PipelineState::Running);
    }

    #[tokio::test]
    async fn test_pipeline_with_transform() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            Box::new(SourceAdapter::new(CountingSource { count: 0, max: 10 })),
        );
        let transform = pipeline.add_node("transform", Box::new(ElementAdapter::new(PassThrough)));
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            Box::new(SinkAdapter::new(CountingSink {
                received: sink_received.clone(),
            })),
        );

        pipeline.link(src, transform).unwrap();
        pipeline.link(transform, sink).unwrap();

        let executor = PipelineExecutor::new();
        executor.run(&mut pipeline).await.unwrap();

        // PassThrough should pass all 10 buffers
        assert_eq!(sink_received.load(Ordering::Relaxed), 10);
    }

    #[tokio::test]
    async fn test_pipeline_with_filter() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            Box::new(SourceAdapter::new(CountingSource { count: 0, max: 10 })),
        );
        let filter = pipeline.add_node("filter", Box::new(ElementAdapter::new(FilterEven)));
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            Box::new(SinkAdapter::new(CountingSink {
                received: sink_received.clone(),
            })),
        );

        pipeline.link(src, filter).unwrap();
        pipeline.link(filter, sink).unwrap();

        let executor = PipelineExecutor::new();
        executor.run(&mut pipeline).await.unwrap();

        // FilterEven should only pass 5 buffers (0, 2, 4, 6, 8)
        assert_eq!(sink_received.load(Ordering::Relaxed), 5);
    }

    #[tokio::test]
    async fn test_invalid_pipeline_fails() {
        let mut pipeline = Pipeline::new();

        // Empty pipeline should fail validation
        let executor = PipelineExecutor::new();
        let result = executor.run(&mut pipeline).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_channel_network() {
        let mut network = ChannelNetwork::new();

        let node_a = NodeId(daggy::NodeIndex::new(0));
        let node_b = NodeId(daggy::NodeIndex::new(1));

        assert!(!network.has_channel(node_a, node_b));

        let (tx, rx) = bounded_async::<Message>(16);
        network.add_channel(node_a, node_b, tx, rx);

        assert!(network.has_channel(node_a, node_b));

        let outputs = network.take_outputs(node_a);
        assert_eq!(outputs.len(), 1);

        let inputs = network.take_inputs(node_b);
        assert_eq!(inputs.len(), 1);
    }

    #[tokio::test]
    async fn test_pipeline_handle_abort() {
        let mut pipeline = Pipeline::new();

        // Create a source that never ends
        struct InfiniteSource;
        impl Source for InfiniteSource {
            fn produce(&mut self) -> Result<Option<Buffer>> {
                let segment = Arc::new(HeapSegment::new(8).unwrap());
                let handle = MemoryHandle::from_segment(segment);
                Ok(Some(Buffer::new(handle, Metadata::with_sequence(0))))
            }
        }

        let src = pipeline.add_node("src", Box::new(SourceAdapter::new(InfiniteSource)));
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            Box::new(SinkAdapter::new(CountingSink {
                received: sink_received.clone(),
            })),
        );

        pipeline.link(src, sink).unwrap();

        let executor = PipelineExecutor::new();
        let handle = executor.start(&mut pipeline).unwrap();

        // Let it run briefly
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Should have received some buffers
        assert!(sink_received.load(Ordering::Relaxed) > 0);

        // Abort the pipeline
        handle.abort();
    }
}
