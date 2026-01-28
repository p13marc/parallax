//! Pipeline executor using Tokio and Kanal channels.
//!
//! The executor spawns a Tokio task for each node in the pipeline and
//! connects them using Kanal channels for zero-copy buffer passing.

use crate::buffer::Buffer;
use crate::element::{AsyncElementDyn, DynAsyncElement, ElementType};
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
    ///
    /// This handles state transitions automatically:
    /// - Suspended → Idle (prepare: validate, negotiate)
    /// - Idle → Running (activate)
    pub fn start(&self, pipeline: &mut Pipeline) -> Result<PipelineHandle> {
        // Create event sender
        let events = EventSender::new(256);

        // Transition through states properly
        let old_state = pipeline.state();

        // Prepare if needed (Suspended → Idle)
        if old_state == PipelineState::Suspended {
            pipeline.prepare()?;
            events.send_state_changed(old_state, PipelineState::Idle);
        }

        // Activate (Idle → Running)
        let idle_state = pipeline.state();
        pipeline.activate()?;
        events.send_state_changed(idle_state, PipelineState::Running);
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

        for (child_id, link) in children {
            // Create channel if not already exists
            if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                let (tx, rx) = bounded_async::<Message>(self.config.channel_capacity);
                network.add_channel(
                    node_id,
                    link.src_pad.clone(),
                    child_id,
                    link.sink_pad.clone(),
                    tx,
                    rx,
                );
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
                ElementType::Demuxer => {
                    // Demuxers use per-pad output channels
                    let outputs_by_pad = channels.take_outputs_by_pad(node_id);
                    let task = spawn_demuxer_task(
                        node_name,
                        element,
                        inputs,
                        outputs_by_pad,
                        events_clone,
                    );
                    tasks.push(task);
                }
                ElementType::Muxer => {
                    // Muxers use per-pad input channels
                    let inputs_by_pad = channels.take_inputs_by_pad(node_id);
                    let task =
                        spawn_muxer_task(node_name, element, inputs_by_pad, outputs, events_clone);
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

/// Key for a channel: (source_node, source_pad, sink_node, sink_pad).
type ChannelKey = (NodeId, String, NodeId, String);

/// Network of channels connecting pipeline nodes.
struct ChannelNetwork {
    /// Channels indexed by (src_node, src_pad, sink_node, sink_pad).
    channels: HashMap<ChannelKey, (AsyncSender<Message>, AsyncReceiver<Message>)>,
    /// Output channels per node+pad: (node, pad_name) -> list of senders.
    outputs: HashMap<(NodeId, String), Vec<AsyncSender<Message>>>,
    /// Input channels per node+pad: (node, pad_name) -> list of receivers.
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

    fn add_channel(
        &mut self,
        src: NodeId,
        src_pad: String,
        sink: NodeId,
        sink_pad: String,
        tx: AsyncSender<Message>,
        rx: AsyncReceiver<Message>,
    ) {
        self.channels.insert(
            (src, src_pad.clone(), sink, sink_pad.clone()),
            (tx.clone(), rx.clone()),
        );
        self.outputs.entry((src, src_pad)).or_default().push(tx);
        self.inputs.entry((sink, sink_pad)).or_default().push(rx);
    }

    /// Take all outputs for a node, grouped by pad name.
    fn take_outputs_by_pad(&mut self, node: NodeId) -> HashMap<String, Vec<AsyncSender<Message>>> {
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

    /// Take all inputs for a node, grouped by pad name.
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

    /// Take all outputs for a node (flattened, for backwards compatibility).
    fn take_outputs(&mut self, node: NodeId) -> Vec<AsyncSender<Message>> {
        self.take_outputs_by_pad(node)
            .into_values()
            .flatten()
            .collect()
    }

    /// Take all inputs for a node (flattened, for backwards compatibility).
    fn take_inputs(&mut self, node: NodeId) -> Vec<AsyncReceiver<Message>> {
        self.take_inputs_by_pad(node)
            .into_values()
            .flatten()
            .collect()
    }
}

/// Spawn a task for a source element.
fn spawn_source_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    outputs: Vec<AsyncSender<Message>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("source task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        // Produce buffers until EOS
        loop {
            match element.process(None).await {
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
    mut element: Box<DynAsyncElement<'static>>,
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
                        if let Err(e) = element.process(Some(buffer)).await {
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
    mut element: Box<DynAsyncElement<'static>>,
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
                        match element.process(Some(buffer)).await {
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

/// Spawn a task for a demuxer element.
///
/// Demuxers have a single input and multiple outputs routed by pad name.
/// The demuxer element is responsible for routing buffers to the correct output pads
/// via `process_all()` which returns routed output.
fn spawn_demuxer_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    outputs_by_pad: HashMap<String, Vec<AsyncSender<Message>>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("demuxer task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        // Demuxers have a single input
        if let Some(rx) = inputs.into_iter().next() {
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        buffers_processed += 1;

                        // Process the buffer through the demuxer
                        // For now, use process() which returns buffers one at a time
                        // The DemuxerAdapter handles routing internally
                        match element.process(Some(buffer)).await {
                            Ok(Some(out_buffer)) => {
                                // Send to all connected pads (default "src" pad)
                                // In future, DemuxerAdapter could carry pad info in buffer metadata
                                for senders in outputs_by_pad.values() {
                                    for tx in senders {
                                        if tx
                                            .send(Message::Buffer(out_buffer.clone()))
                                            .await
                                            .is_err()
                                        {
                                            tracing::warn!(
                                                "demuxer '{}': downstream receiver dropped",
                                                name
                                            );
                                        }
                                    }
                                }
                            }
                            Ok(None) => {
                                // Buffer was filtered/dropped
                                tracing::trace!("demuxer '{}' filtered out buffer", name);
                            }
                            Err(e) => {
                                tracing::error!("demuxer '{}' error: {}", name, e);
                                events.send_error(e.to_string(), Some(name.clone()));
                                return Err(e);
                            }
                        }
                    }
                    Ok(Message::Eos) => {
                        // Forward EOS to all output pads
                        tracing::debug!("demuxer '{}' received EOS", name);
                        for senders in outputs_by_pad.values() {
                            for tx in senders {
                                let _ = tx.send(Message::Eos).await;
                            }
                        }
                        break;
                    }
                    Err(_) => {
                        // Channel closed, forward EOS
                        tracing::debug!("demuxer '{}': channel closed", name);
                        for senders in outputs_by_pad.values() {
                            for tx in senders {
                                let _ = tx.send(Message::Eos).await;
                            }
                        }
                        break;
                    }
                }
            }
        }

        tracing::debug!("demuxer task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
        Ok(())
    })
}

/// Spawn a task for a muxer element.
///
/// Muxers have multiple inputs (by pad name) and a single output.
/// Buffers are received from all input pads and combined by the muxer element.
fn spawn_muxer_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    inputs_by_pad: HashMap<String, Vec<AsyncReceiver<Message>>>,
    outputs: Vec<AsyncSender<Message>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    use futures::stream::{FuturesUnordered, StreamExt};

    tokio::spawn(async move {
        tracing::debug!("muxer task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        // Flatten all receivers from all pads into a stream
        let mut receivers: FuturesUnordered<_> = inputs_by_pad
            .into_iter()
            .flat_map(|(pad_name, rxs)| {
                rxs.into_iter().map(move |rx| {
                    let pad = pad_name.clone();
                    async move { (pad, rx.recv().await) }
                })
            })
            .collect();

        let total_inputs = receivers.len();
        let mut eos_count = 0;

        while let Some((pad_name, msg)) = receivers.next().await {
            match msg {
                Ok(Message::Buffer(buffer)) => {
                    buffers_processed += 1;
                    tracing::trace!("muxer '{}' received buffer on pad '{}'", name, pad_name);

                    // Process the buffer through the muxer
                    match element.process(Some(buffer)).await {
                        Ok(Some(out_buffer)) => {
                            // Send output to all downstream elements
                            for tx in &outputs {
                                if tx.send(Message::Buffer(out_buffer.clone())).await.is_err() {
                                    tracing::warn!("muxer '{}': downstream receiver dropped", name);
                                }
                            }
                        }
                        Ok(None) => {
                            // Muxer is buffering, no output yet
                            tracing::trace!("muxer '{}' buffering (no output)", name);
                        }
                        Err(e) => {
                            tracing::error!("muxer '{}' error: {}", name, e);
                            events.send_error(e.to_string(), Some(name.clone()));
                            return Err(e);
                        }
                    }
                }
                Ok(Message::Eos) => {
                    eos_count += 1;
                    tracing::debug!(
                        "muxer '{}' received EOS on pad '{}' ({}/{})",
                        name,
                        pad_name,
                        eos_count,
                        total_inputs
                    );

                    // When all inputs have sent EOS, flush and forward EOS
                    if eos_count >= total_inputs {
                        // Flush the muxer (send None to signal EOS)
                        if let Ok(Some(out_buffer)) = element.process(None).await {
                            for tx in &outputs {
                                let _ = tx.send(Message::Buffer(out_buffer.clone())).await;
                            }
                        }

                        // Send EOS downstream
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                }
                Err(_) => {
                    // Channel closed for this pad
                    eos_count += 1;
                    tracing::debug!(
                        "muxer '{}': channel closed on pad '{}' ({}/{})",
                        name,
                        pad_name,
                        eos_count,
                        total_inputs
                    );

                    if eos_count >= total_inputs {
                        // Flush and forward EOS
                        if let Ok(Some(out_buffer)) = element.process(None).await {
                            for tx in &outputs {
                                let _ = tx.send(Message::Buffer(out_buffer.clone())).await;
                            }
                        }
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                }
            }
        }

        tracing::debug!("muxer task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::element::{
        ConsumeContext, DynAsyncElement, Element, ElementAdapter, ProduceContext, ProduceResult,
        Sink, SinkAdapter, Source, SourceAdapter,
    };
    use crate::memory::CpuSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct CountingSource {
        count: u64,
        max: u64,
    }

    impl Source for CountingSource {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
            if self.count >= self.max {
                return Ok(ProduceResult::Eos);
            }
            let segment = Arc::new(CpuSegment::new(8).unwrap());
            let handle = MemoryHandle::from_segment(segment);
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
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 5 })),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
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
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 10 })),
        );
        let transform = pipeline.add_node(
            "transform",
            DynAsyncElement::new_box(ElementAdapter::new(PassThrough)),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
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
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 10 })),
        );
        let filter = pipeline.add_node(
            "filter",
            DynAsyncElement::new_box(ElementAdapter::new(FilterEven)),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
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

        assert!(!network.has_channel(node_a, "src", node_b, "sink"));

        let (tx, rx) = bounded_async::<Message>(16);
        network.add_channel(
            node_a,
            "src".to_string(),
            node_b,
            "sink".to_string(),
            tx,
            rx,
        );

        assert!(network.has_channel(node_a, "src", node_b, "sink"));

        let outputs = network.take_outputs(node_a);
        assert_eq!(outputs.len(), 1);

        let inputs = network.take_inputs(node_b);
        assert_eq!(inputs.len(), 1);
    }

    #[tokio::test]
    async fn test_channel_network_multi_pad() {
        let mut network = ChannelNetwork::new();

        let node_a = NodeId(daggy::NodeIndex::new(0));
        let node_b = NodeId(daggy::NodeIndex::new(1));
        let node_c = NodeId(daggy::NodeIndex::new(2));

        // Connect node_a.src_0 -> node_b.sink
        let (tx1, rx1) = bounded_async::<Message>(16);
        network.add_channel(
            node_a,
            "src_0".to_string(),
            node_b,
            "sink".to_string(),
            tx1,
            rx1,
        );

        // Connect node_a.src_1 -> node_c.sink
        let (tx2, rx2) = bounded_async::<Message>(16);
        network.add_channel(
            node_a,
            "src_1".to_string(),
            node_c,
            "sink".to_string(),
            tx2,
            rx2,
        );

        // Verify channels exist
        assert!(network.has_channel(node_a, "src_0", node_b, "sink"));
        assert!(network.has_channel(node_a, "src_1", node_c, "sink"));
        assert!(!network.has_channel(node_a, "src_0", node_c, "sink"));

        // Take outputs by pad
        let outputs_by_pad = network.take_outputs_by_pad(node_a);
        assert_eq!(outputs_by_pad.len(), 2);
        assert!(outputs_by_pad.contains_key("src_0"));
        assert!(outputs_by_pad.contains_key("src_1"));

        // Inputs should have one each
        let inputs_b = network.take_inputs(node_b);
        assert_eq!(inputs_b.len(), 1);

        let inputs_c = network.take_inputs(node_c);
        assert_eq!(inputs_c.len(), 1);
    }

    #[tokio::test]
    async fn test_pipeline_handle_abort() {
        let mut pipeline = Pipeline::new();

        // Create a source that never ends
        struct InfiniteSource;
        impl Source for InfiniteSource {
            fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
                let segment = Arc::new(CpuSegment::new(8).unwrap());
                let handle = MemoryHandle::from_segment(segment);
                Ok(ProduceResult::OwnBuffer(Buffer::new(
                    handle,
                    Metadata::from_sequence(0),
                )))
            }
        }

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(InfiniteSource)),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
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
