//! Pipeline graph structure using daggy.

use crate::element::{AsyncElementDyn, DynAsyncElement, ElementType, Pad};
use crate::error::{Error, Result};
use crate::format::{Caps, MediaFormat};
use crate::memory::MemoryType;
use crate::negotiation::{
    ConverterInsertion, ConverterRegistry, ElementCaps, LinkInfo as NegLinkInfo, NegotiationResult,
    NegotiationSolver,
};
use daggy::petgraph::visit::EdgeRef;
use daggy::{Dag, EdgeIndex, NodeIndex, Walker};
use std::collections::HashMap;
use std::fmt::Write;

/// Unique identifier for a node in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) NodeIndex);

impl NodeId {
    /// Get the underlying index.
    pub fn index(&self) -> usize {
        self.0.index()
    }
}

/// Unique identifier for a link (edge) in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinkId(pub(crate) EdgeIndex);

impl LinkId {
    /// Get the underlying index.
    pub fn index(&self) -> usize {
        self.0.index()
    }
}

/// Information about a link in the pipeline, including negotiation results.
#[derive(Debug, Clone)]
pub struct LinkInfo {
    /// The link ID.
    pub id: LinkId,
    /// Source node ID.
    pub source_id: NodeId,
    /// Source node name.
    pub source_name: String,
    /// Source pad name.
    pub source_pad: String,
    /// Sink node ID.
    pub sink_id: NodeId,
    /// Sink node name.
    pub sink_name: String,
    /// Sink pad name.
    pub sink_pad: String,
    /// Negotiated format (if negotiation has been run).
    pub negotiated_format: Option<MediaFormat>,
    /// Negotiated memory type (if negotiation has been run).
    pub negotiated_memory: Option<MemoryType>,
}

/// State of the pipeline (PipeWire-inspired 3-state model).
///
/// This follows PipeWire's state machine:
/// - **Suspended**: Resources deallocated, minimal memory usage
/// - **Idle**: Resources allocated, ready to process, but not actively running
/// - **Running**: Actively processing data
///
/// State transitions:
/// ```text
/// Suspended <-> Idle <-> Running
/// ```
///
/// The key insight from PipeWire is that "paused" and "stopped" are the same
/// state (Idle) - the difference is just whether we intend to resume soon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PipelineState {
    /// Resources are deallocated. Minimal memory footprint.
    /// Transition to Idle to allocate resources.
    #[default]
    Suspended,

    /// Resources are allocated and ready, but not processing.
    /// This is the "paused" state - ready to run immediately.
    /// Transition to Running to start processing.
    Idle,

    /// Actively processing data through the pipeline.
    /// Transition to Idle to pause, or to Suspended to release resources.
    Running,

    /// Pipeline encountered an unrecoverable error.
    /// Must transition to Suspended before recovering.
    Error,
}

/// A node in the pipeline graph.
pub struct Node {
    /// Unique name of this node.
    name: String,
    /// The element wrapped by this node.
    /// This is an Option so that elements can be taken out for execution.
    element: Option<Box<DynAsyncElement<'static>>>,
    /// Cached element type (so we don't need the element to query it).
    element_type: ElementType,
    /// Cached input caps (so we don't need the element to query them).
    input_caps: Caps,
    /// Cached output caps (so we don't need the element to query them).
    output_caps: Caps,
    /// Input pads.
    input_pads: Vec<Pad>,
    /// Output pads.
    output_pads: Vec<Pad>,
}

impl Node {
    /// Create a new node.
    pub fn new(name: impl Into<String>, element: Box<DynAsyncElement<'static>>) -> Self {
        let name = name.into();
        let element_type = element.element_type();
        let input_caps = element.input_caps();
        let output_caps = element.output_caps();

        // Create default pads based on element type
        let (input_pads, output_pads) = match element_type {
            ElementType::Source => (vec![], vec![Pad::src()]),
            ElementType::Sink => (vec![Pad::sink()], vec![]),
            ElementType::Transform => (vec![Pad::sink()], vec![Pad::src()]),
            // Demuxers have one input and multiple outputs (pads created dynamically)
            ElementType::Demuxer => (vec![Pad::sink()], vec![Pad::src()]),
            // Muxers have multiple inputs and one output (pads created dynamically)
            ElementType::Muxer => (vec![Pad::sink()], vec![Pad::src()]),
        };

        Self {
            name,
            element: Some(element),
            element_type,
            input_caps,
            output_caps,
            input_pads,
            output_pads,
        }
    }

    /// Get the node's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get a reference to the element.
    ///
    /// Returns `None` if the element has been taken for execution.
    pub fn element(&self) -> Option<&DynAsyncElement<'static>> {
        self.element.as_ref().map(|e| e.as_ref())
    }

    /// Get a mutable reference to the element.
    ///
    /// Returns `None` if the element has been taken for execution.
    pub fn element_mut(&mut self) -> Option<&mut Box<DynAsyncElement<'static>>> {
        self.element.as_mut()
    }

    /// Take the element out of this node for execution.
    ///
    /// Returns `None` if the element has already been taken.
    pub fn take_element(&mut self) -> Option<Box<DynAsyncElement<'static>>> {
        self.element.take()
    }

    /// Get the element type.
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Get the input caps (formats this element accepts).
    pub fn input_caps(&self) -> &Caps {
        &self.input_caps
    }

    /// Get the output caps (formats this element produces).
    pub fn output_caps(&self) -> &Caps {
        &self.output_caps
    }

    /// Get input pads.
    pub fn input_pads(&self) -> &[Pad] {
        &self.input_pads
    }

    /// Get output pads.
    pub fn output_pads(&self) -> &[Pad] {
        &self.output_pads
    }

    /// Add an output pad.
    pub fn add_output_pad(&mut self, pad: Pad) {
        debug_assert!(pad.is_output());
        self.output_pads.push(pad);
    }

    /// Add an input pad.
    pub fn add_input_pad(&mut self, pad: Pad) {
        debug_assert!(pad.is_input());
        self.input_pads.push(pad);
    }

    /// Remove an output pad by name.
    ///
    /// Returns the removed pad if found.
    pub fn remove_output_pad(&mut self, name: &str) -> Option<Pad> {
        if let Some(pos) = self.output_pads.iter().position(|p| p.name() == name) {
            Some(self.output_pads.remove(pos))
        } else {
            None
        }
    }

    /// Remove an input pad by name.
    ///
    /// Returns the removed pad if found.
    pub fn remove_input_pad(&mut self, name: &str) -> Option<Pad> {
        if let Some(pos) = self.input_pads.iter().position(|p| p.name() == name) {
            Some(self.input_pads.remove(pos))
        } else {
            None
        }
    }

    /// Get an output pad by name.
    pub fn get_output_pad(&self, name: &str) -> Option<&Pad> {
        self.output_pads.iter().find(|p| p.name() == name)
    }

    /// Get an input pad by name.
    pub fn get_input_pad(&self, name: &str) -> Option<&Pad> {
        self.input_pads.iter().find(|p| p.name() == name)
    }

    /// Get the scheduling affinity for this node's element.
    ///
    /// Returns `Affinity::Auto` if the element has been taken.
    pub fn affinity(&self) -> crate::element::Affinity {
        self.element
            .as_ref()
            .map(|e| e.affinity())
            .unwrap_or(crate::element::Affinity::Auto)
    }

    /// Check if this node's element is safe to run in a real-time context.
    ///
    /// Returns `false` if the element has been taken.
    pub fn is_rt_safe(&self) -> bool {
        self.element
            .as_ref()
            .map(|e| e.is_rt_safe())
            .unwrap_or(false)
    }

    /// Get execution hints for this node's element.
    ///
    /// Returns default hints if the element has been taken.
    pub fn execution_hints(&self) -> crate::element::ExecutionHints {
        self.element
            .as_ref()
            .map(|e| e.execution_hints())
            .unwrap_or_default()
    }
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("name", &self.name)
            .field("element_type", &self.element_type())
            .field("input_pads", &self.input_pads.len())
            .field("output_pads", &self.output_pads.len())
            .finish()
    }
}

/// A link between two nodes in the pipeline.
#[derive(Debug, Clone)]
pub struct Link {
    /// Name of the source pad.
    pub src_pad: String,
    /// Name of the sink pad.
    pub sink_pad: String,
}

impl Default for Link {
    fn default() -> Self {
        Self {
            src_pad: "src".to_string(),
            sink_pad: "sink".to_string(),
        }
    }
}

impl Link {
    /// Create a new link with default pad names.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a link with specific pad names.
    pub fn with_pads(src_pad: impl Into<String>, sink_pad: impl Into<String>) -> Self {
        Self {
            src_pad: src_pad.into(),
            sink_pad: sink_pad.into(),
        }
    }
}

/// A streaming pipeline represented as a directed acyclic graph.
pub struct Pipeline {
    /// The DAG structure.
    graph: Dag<Node, Link>,
    /// Name-to-NodeId mapping for quick lookup.
    nodes_by_name: HashMap<String, NodeId>,
    /// Current state of the pipeline.
    state: PipelineState,
    /// Auto-incrementing counter for anonymous node names.
    name_counter: u64,
    /// Negotiation results (populated after negotiate() is called).
    negotiation: Option<NegotiationResult>,
}

impl Pipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            graph: Dag::new(),
            nodes_by_name: HashMap::new(),
            state: PipelineState::Suspended,
            name_counter: 0,
            negotiation: None,
        }
    }

    /// Get the current pipeline state.
    pub fn state(&self) -> PipelineState {
        self.state
    }

    /// Set the pipeline state directly (internal use).
    pub fn set_state(&mut self, state: PipelineState) {
        self.state = state;
    }

    /// Transition from Suspended to Idle (allocate resources).
    ///
    /// This prepares the pipeline for execution by:
    /// - Running caps negotiation if needed
    /// - Validating the pipeline structure
    ///
    /// Returns an error if the transition is invalid.
    pub fn prepare(&mut self) -> Result<()> {
        match self.state {
            PipelineState::Suspended => {
                // Validate pipeline structure
                self.validate()?;

                // Run caps negotiation if needed
                if !self.is_negotiated() {
                    self.negotiate()?;
                }

                self.state = PipelineState::Idle;
                Ok(())
            }
            PipelineState::Idle => {
                // Already prepared, no-op
                Ok(())
            }
            PipelineState::Running => Err(Error::InvalidSegment(
                "cannot prepare while running, pause first".into(),
            )),
            PipelineState::Error => Err(Error::InvalidSegment(
                "cannot prepare from error state, reset first".into(),
            )),
        }
    }

    /// Transition from Idle to Running (start processing).
    ///
    /// The pipeline must be in Idle state (call `prepare()` first).
    pub fn activate(&mut self) -> Result<()> {
        match self.state {
            PipelineState::Idle => {
                self.state = PipelineState::Running;
                Ok(())
            }
            PipelineState::Running => {
                // Already running, no-op
                Ok(())
            }
            PipelineState::Suspended => Err(Error::InvalidSegment(
                "cannot activate from suspended, call prepare() first".into(),
            )),
            PipelineState::Error => Err(Error::InvalidSegment(
                "cannot activate from error state, reset first".into(),
            )),
        }
    }

    /// Transition from Running to Idle (pause processing).
    ///
    /// Data threads stop processing but resources remain allocated.
    pub fn pause(&mut self) -> Result<()> {
        match self.state {
            PipelineState::Running => {
                self.state = PipelineState::Idle;
                Ok(())
            }
            PipelineState::Idle => {
                // Already paused, no-op
                Ok(())
            }
            PipelineState::Suspended => Err(Error::InvalidSegment(
                "cannot pause from suspended state".into(),
            )),
            PipelineState::Error => Err(Error::InvalidSegment(
                "cannot pause from error state".into(),
            )),
        }
    }

    /// Transition from Idle to Suspended (release resources).
    ///
    /// This deallocates buffers and releases resources.
    pub fn suspend(&mut self) -> Result<()> {
        match self.state {
            PipelineState::Idle => {
                // Clear negotiation results (resources will be deallocated)
                self.negotiation = None;
                self.state = PipelineState::Suspended;
                Ok(())
            }
            PipelineState::Suspended => {
                // Already suspended, no-op
                Ok(())
            }
            PipelineState::Running => Err(Error::InvalidSegment(
                "cannot suspend while running, pause first".into(),
            )),
            PipelineState::Error => {
                // Allow recovery from error by suspending
                self.negotiation = None;
                self.state = PipelineState::Suspended;
                Ok(())
            }
        }
    }

    /// Mark the pipeline as having encountered an error.
    pub fn set_error(&mut self) {
        self.state = PipelineState::Error;
    }

    /// Check if the pipeline is in a runnable state.
    pub fn is_runnable(&self) -> bool {
        matches!(self.state, PipelineState::Idle | PipelineState::Running)
    }

    /// Check if the pipeline is currently running.
    pub fn is_running(&self) -> bool {
        self.state == PipelineState::Running
    }

    /// Check if the pipeline has encountered an error.
    pub fn has_error(&self) -> bool {
        self.state == PipelineState::Error
    }

    /// Add a node to the pipeline.
    ///
    /// Returns the node's ID for linking.
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        element: Box<DynAsyncElement<'static>>,
    ) -> NodeId {
        let name = name.into();
        let node = Node::new(name.clone(), element);
        let idx = self.graph.add_node(node);
        let id = NodeId(idx);
        self.nodes_by_name.insert(name, id);
        id
    }

    /// Add a node with an auto-generated name.
    pub fn add_node_auto(&mut self, element: Box<DynAsyncElement<'static>>) -> NodeId {
        let name = format!("node_{}", self.name_counter);
        self.name_counter += 1;
        self.add_node(name, element)
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.graph.node_weight(id.0)
    }

    /// Get a mutable reference to a node by ID.
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.graph.node_weight_mut(id.0)
    }

    /// Get a node ID by name.
    pub fn get_node_id(&self, name: &str) -> Option<NodeId> {
        self.nodes_by_name.get(name).copied()
    }

    /// Link two nodes with default pad names.
    ///
    /// Creates an edge from `src` to `sink` using the default "src" and "sink" pads.
    pub fn link(&mut self, src: NodeId, sink: NodeId) -> Result<()> {
        self.link_pads(src, "src", sink, "sink")
    }

    /// Link two nodes with specific pad names.
    pub fn link_pads(
        &mut self,
        src: NodeId,
        src_pad: &str,
        sink: NodeId,
        sink_pad: &str,
    ) -> Result<()> {
        // Validate source node and pad
        let src_node = self
            .graph
            .node_weight(src.0)
            .ok_or_else(|| Error::InvalidSegment("source node not found".into()))?;

        if !src_node.output_pads.iter().any(|p| p.name() == src_pad) {
            return Err(Error::InvalidSegment(format!(
                "source node '{}' has no output pad '{}'",
                src_node.name, src_pad
            )));
        }

        // Validate sink node and pad
        let sink_node = self
            .graph
            .node_weight(sink.0)
            .ok_or_else(|| Error::InvalidSegment("sink node not found".into()))?;

        if !sink_node.input_pads.iter().any(|p| p.name() == sink_pad) {
            return Err(Error::InvalidSegment(format!(
                "sink node '{}' has no input pad '{}'",
                sink_node.name, sink_pad
            )));
        }

        // Create the link
        let link = Link::with_pads(src_pad, sink_pad);

        // Add edge (daggy ensures no cycles)
        self.graph
            .add_edge(src.0, sink.0, link)
            .map_err(|_| Error::InvalidSegment("linking would create a cycle".into()))?;

        // Invalidate negotiation since the graph has changed
        self.negotiation = None;

        Ok(())
    }

    /// Get all source nodes (nodes with no incoming edges).
    pub fn sources(&self) -> Vec<NodeId> {
        self.graph
            .graph()
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .graph()
                    .neighbors_directed(idx, daggy::petgraph::Direction::Incoming)
                    .count()
                    == 0
            })
            .map(NodeId)
            .collect()
    }

    /// Get all sink nodes (nodes with no outgoing edges).
    pub fn sinks(&self) -> Vec<NodeId> {
        self.graph
            .graph()
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .graph()
                    .neighbors_directed(idx, daggy::petgraph::Direction::Outgoing)
                    .count()
                    == 0
            })
            .map(NodeId)
            .collect()
    }

    /// Get the children (downstream nodes) of a node.
    pub fn children(&self, id: NodeId) -> Vec<(NodeId, &Link)> {
        self.graph
            .children(id.0)
            .iter(&self.graph)
            .map(|(edge_idx, node_idx)| {
                let link = self.graph.edge_weight(edge_idx).unwrap();
                (NodeId(node_idx), link)
            })
            .collect()
    }

    /// Get all node IDs in the pipeline.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.graph.graph().node_indices().map(NodeId).collect()
    }

    /// Get the parents (upstream nodes) of a node.
    pub fn parents(&self, id: NodeId) -> Vec<(NodeId, &Link)> {
        self.graph
            .parents(id.0)
            .iter(&self.graph)
            .map(|(edge_idx, node_idx)| {
                let link = self.graph.edge_weight(edge_idx).unwrap();
                (NodeId(node_idx), link)
            })
            .collect()
    }

    /// Get the number of nodes in the pipeline.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges (links) in the pipeline.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Validate the pipeline structure.
    ///
    /// Checks that:
    /// - All nodes are connected
    /// - There is at least one source and one sink
    /// - No dangling pads
    pub fn validate(&self) -> Result<()> {
        if self.is_empty() {
            return Err(Error::InvalidSegment("pipeline is empty".into()));
        }

        let sources = self.sources();
        let sinks = self.sinks();

        if sources.is_empty() {
            return Err(Error::InvalidSegment("pipeline has no source nodes".into()));
        }

        if sinks.is_empty() {
            return Err(Error::InvalidSegment("pipeline has no sink nodes".into()));
        }

        // Verify all source nodes are actually Source elements
        for src_id in &sources {
            let node = self.get_node(*src_id).unwrap();
            if node.element_type() != ElementType::Source {
                return Err(Error::InvalidSegment(format!(
                    "node '{}' has no inputs but is not a Source element",
                    node.name()
                )));
            }
        }

        // Verify all sink nodes are actually Sink elements
        for sink_id in &sinks {
            let node = self.get_node(*sink_id).unwrap();
            if node.element_type() != ElementType::Sink {
                return Err(Error::InvalidSegment(format!(
                    "node '{}' has no outputs but is not a Sink element",
                    node.name()
                )));
            }
        }

        Ok(())
    }

    // ========================================================================
    // Introspection API
    // ========================================================================

    /// Iterate over all nodes in the pipeline.
    ///
    /// Returns an iterator of (NodeId, &Node) pairs.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::pipeline::Pipeline;
    ///
    /// let pipeline = Pipeline::parse("nullsource ! passthrough ! nullsink").unwrap();
    /// for (id, node) in pipeline.nodes() {
    ///     println!("{}: {:?}", node.name(), node.element_type());
    /// }
    /// ```
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.graph
            .graph()
            .node_indices()
            .map(|idx| (NodeId(idx), self.graph.node_weight(idx).unwrap()))
    }

    /// Iterate over all links in the pipeline with full information.
    ///
    /// Returns an iterator of `LinkInfo` structs containing source/sink
    /// information and negotiated formats (if negotiation has been run).
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::pipeline::Pipeline;
    ///
    /// let pipeline = Pipeline::parse("nullsource ! passthrough ! nullsink").unwrap();
    /// for link in pipeline.links() {
    ///     println!("{} -> {}", link.source_name, link.sink_name);
    ///     if let Some(format) = &link.negotiated_format {
    ///         println!("  format: {:?}", format);
    ///     }
    /// }
    /// ```
    pub fn links(&self) -> impl Iterator<Item = LinkInfo> + '_ {
        self.graph.graph().edge_references().map(|edge| {
            let src_idx = edge.source();
            let sink_idx = edge.target();
            let link = edge.weight();
            let link_id = LinkId(edge.id());

            let src_node = self.graph.node_weight(src_idx).unwrap();
            let sink_node = self.graph.node_weight(sink_idx).unwrap();

            // Look up negotiation results if available
            let (negotiated_format, negotiated_memory) = self
                .negotiation
                .as_ref()
                .and_then(|neg| neg.link_caps.get(&link_id.index()))
                .map(|cap| (Some(cap.format.clone()), Some(cap.memory_type)))
                .unwrap_or((None, None));

            LinkInfo {
                id: link_id,
                source_id: NodeId(src_idx),
                source_name: src_node.name().to_string(),
                source_pad: link.src_pad.clone(),
                sink_id: NodeId(sink_idx),
                sink_name: sink_node.name().to_string(),
                sink_pad: link.sink_pad.clone(),
                negotiated_format,
                negotiated_memory,
            }
        })
    }

    /// Get information about a specific link by ID.
    pub fn get_link(&self, id: LinkId) -> Option<LinkInfo> {
        let (src_idx, sink_idx) = self.graph.graph().edge_endpoints(id.0)?;
        let link = self.graph.edge_weight(id.0)?;
        let src_node = self.graph.node_weight(src_idx)?;
        let sink_node = self.graph.node_weight(sink_idx)?;

        let (negotiated_format, negotiated_memory) = self
            .negotiation
            .as_ref()
            .and_then(|neg| neg.link_caps.get(&id.index()))
            .map(|cap| (Some(cap.format.clone()), Some(cap.memory_type)))
            .unwrap_or((None, None));

        Some(LinkInfo {
            id,
            source_id: NodeId(src_idx),
            source_name: src_node.name().to_string(),
            source_pad: link.src_pad.clone(),
            sink_id: NodeId(sink_idx),
            sink_name: sink_node.name().to_string(),
            sink_pad: link.sink_pad.clone(),
            negotiated_format,
            negotiated_memory,
        })
    }

    /// Get the negotiated format for a specific link.
    ///
    /// Returns `None` if negotiation hasn't been run or the link doesn't exist.
    pub fn link_format(&self, id: LinkId) -> Option<&MediaFormat> {
        self.negotiation
            .as_ref()
            .and_then(|neg| neg.link_caps.get(&id.index()))
            .map(|cap| &cap.format)
    }

    /// Get the negotiated memory type for a specific link.
    ///
    /// Returns `None` if negotiation hasn't been run or the link doesn't exist.
    pub fn link_memory_type(&self, id: LinkId) -> Option<MemoryType> {
        self.negotiation
            .as_ref()
            .and_then(|neg| neg.link_caps.get(&id.index()))
            .map(|cap| cap.memory_type)
    }

    /// Check if negotiation has been performed.
    pub fn is_negotiated(&self) -> bool {
        self.negotiation.is_some()
    }

    /// Get the full negotiation result (if negotiation has been run).
    pub fn negotiation_result(&self) -> Option<&NegotiationResult> {
        self.negotiation.as_ref()
    }

    /// Get the list of converters that need to be inserted (if any).
    ///
    /// After negotiation, this returns the converters that should be
    /// inserted to bridge incompatible formats between elements.
    pub fn pending_converters(&self) -> &[ConverterInsertion] {
        self.negotiation
            .as_ref()
            .map(|n| n.converters.as_slice())
            .unwrap_or(&[])
    }

    // ========================================================================
    // Caps Negotiation
    // ========================================================================

    /// Run caps negotiation on the pipeline.
    ///
    /// This analyzes all elements' input/output caps and finds compatible
    /// formats for each link. After calling this, you can query negotiated
    /// formats via `links()`, `link_format()`, etc.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use parallax::pipeline::Pipeline;
    ///
    /// let mut pipeline = Pipeline::parse("videotestsrc ! videoconvert ! displaysink")?;
    /// pipeline.negotiate()?;
    ///
    /// for link in pipeline.links() {
    ///     println!("{} -> {}: {:?}",
    ///         link.source_name,
    ///         link.sink_name,
    ///         link.negotiated_format);
    /// }
    /// ```
    pub fn negotiate(&mut self) -> Result<()> {
        self.negotiate_with_registry(None)
    }

    /// Run caps negotiation with a custom converter registry.
    ///
    /// The converter registry allows automatic insertion of format converters
    /// when direct negotiation fails.
    pub fn negotiate_with_registry(&mut self, registry: Option<ConverterRegistry>) -> Result<()> {
        let mut solver = NegotiationSolver::new();

        if let Some(reg) = registry {
            solver = solver.with_converters(reg);
        }

        // Collect element caps
        for (node_id, node) in self.nodes() {
            let element_caps = self.collect_element_caps(node_id, node)?;
            solver.add_element(element_caps);
        }

        // Collect links
        for (link_idx, edge) in self.graph.graph().edge_references().enumerate() {
            let src_node = self.graph.node_weight(edge.source()).unwrap();
            let sink_node = self.graph.node_weight(edge.target()).unwrap();
            let link = edge.weight();

            solver.add_link(NegLinkInfo {
                id: link_idx,
                source_element: src_node.name().to_string(),
                source_pad: link.src_pad.clone(),
                sink_element: sink_node.name().to_string(),
                sink_pad: link.sink_pad.clone(),
            });
        }

        // Run negotiation
        let result = solver
            .solve()
            .map_err(|e| Error::InvalidSegment(format!("Negotiation failed: {}", e)))?;

        self.negotiation = Some(result);
        Ok(())
    }

    /// Collect caps from an element for negotiation.
    fn collect_element_caps(&self, _node_id: NodeId, node: &Node) -> Result<ElementCaps> {
        use crate::format::MediaCaps;

        // Convert Caps to MediaCaps
        let to_media_caps = |caps: &Caps| -> MediaCaps {
            if caps.is_any() {
                MediaCaps::any()
            } else if let Some(first) = caps.formats().first() {
                MediaCaps::from(first.clone())
            } else {
                MediaCaps::any()
            }
        };

        let mut element_caps = ElementCaps::new(node.name());

        // For now, use default pad names "sink" and "src"
        // TODO: In the future, iterate over actual pads from the node
        // and collect caps per-pad using element.output_caps_for_pad(pad_name)

        // Add sink pad caps (input)
        let input_caps = node.input_caps();
        if !input_caps.is_any() || node.element_type() != ElementType::Source {
            element_caps.add_sink_pad("sink", to_media_caps(input_caps));
        }

        // Add source pad caps (output)
        let output_caps = node.output_caps();
        if !output_caps.is_any() || node.element_type() != ElementType::Sink {
            element_caps.add_source_pad("src", to_media_caps(output_caps));
        }

        Ok(element_caps)
    }

    // ========================================================================
    // Dynamic Pad Management
    // ========================================================================

    /// Add an output pad to a node at runtime.
    ///
    /// This is used by demuxers and other elements that create pads dynamically.
    /// After adding a pad, you can link it to downstream elements.
    ///
    /// **Note**: Adding a pad invalidates any previous negotiation. Call
    /// `negotiate()` again after linking the new pad.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use parallax::pipeline::Pipeline;
    /// use parallax::element::Pad;
    ///
    /// let mut pipeline = Pipeline::new();
    /// let demuxer = pipeline.add_node("demux", my_demuxer);
    ///
    /// // Later, when the demuxer discovers a new stream:
    /// pipeline.add_output_pad(demuxer, Pad::src_named("video_0"))?;
    /// ```
    pub fn add_output_pad(&mut self, node_id: NodeId, pad: Pad) -> Result<()> {
        let node = self
            .get_node_mut(node_id)
            .ok_or_else(|| Error::InvalidSegment(format!("Node {:?} not found", node_id)))?;

        if !pad.is_output() {
            return Err(Error::InvalidSegment(
                "Cannot add input pad as output pad".into(),
            ));
        }

        // Check for duplicate pad name
        if node.get_output_pad(pad.name()).is_some() {
            return Err(Error::InvalidSegment(format!(
                "Output pad '{}' already exists on node '{}'",
                pad.name(),
                node.name()
            )));
        }

        node.add_output_pad(pad);

        // Invalidate negotiation since the graph has changed
        self.negotiation = None;

        Ok(())
    }

    /// Add an input pad to a node at runtime.
    ///
    /// This is used by muxers and other elements that accept dynamic inputs.
    /// After adding a pad, upstream elements can link to it.
    ///
    /// **Note**: Adding a pad invalidates any previous negotiation. Call
    /// `negotiate()` again after linking the new pad.
    pub fn add_input_pad(&mut self, node_id: NodeId, pad: Pad) -> Result<()> {
        let node = self
            .get_node_mut(node_id)
            .ok_or_else(|| Error::InvalidSegment(format!("Node {:?} not found", node_id)))?;

        if !pad.is_input() {
            return Err(Error::InvalidSegment(
                "Cannot add output pad as input pad".into(),
            ));
        }

        // Check for duplicate pad name
        if node.get_input_pad(pad.name()).is_some() {
            return Err(Error::InvalidSegment(format!(
                "Input pad '{}' already exists on node '{}'",
                pad.name(),
                node.name()
            )));
        }

        node.add_input_pad(pad);

        // Invalidate negotiation since the graph has changed
        self.negotiation = None;

        Ok(())
    }

    /// Remove an output pad from a node.
    ///
    /// This also removes any links connected to the pad.
    /// Returns the removed pad if found.
    pub fn remove_output_pad(&mut self, node_id: NodeId, pad_name: &str) -> Result<Option<Pad>> {
        // First, remove any links connected to this pad
        self.remove_links_for_pad(node_id, pad_name, true)?;

        let node = self
            .get_node_mut(node_id)
            .ok_or_else(|| Error::InvalidSegment(format!("Node {:?} not found", node_id)))?;

        let pad = node.remove_output_pad(pad_name);

        if pad.is_some() {
            // Invalidate negotiation since the graph has changed
            self.negotiation = None;
        }

        Ok(pad)
    }

    /// Remove an input pad from a node.
    ///
    /// This also removes any links connected to the pad.
    /// Returns the removed pad if found.
    pub fn remove_input_pad(&mut self, node_id: NodeId, pad_name: &str) -> Result<Option<Pad>> {
        // First, remove any links connected to this pad
        self.remove_links_for_pad(node_id, pad_name, false)?;

        let node = self
            .get_node_mut(node_id)
            .ok_or_else(|| Error::InvalidSegment(format!("Node {:?} not found", node_id)))?;

        let pad = node.remove_input_pad(pad_name);

        if pad.is_some() {
            // Invalidate negotiation since the graph has changed
            self.negotiation = None;
        }

        Ok(pad)
    }

    /// Remove all links connected to a specific pad.
    fn remove_links_for_pad(
        &mut self,
        node_id: NodeId,
        pad_name: &str,
        is_output: bool,
    ) -> Result<()> {
        // Collect edges to remove
        let edges_to_remove: Vec<_> = self
            .graph
            .graph()
            .edge_references()
            .filter(|edge| {
                let link = edge.weight();
                if is_output {
                    edge.source() == node_id.0 && link.src_pad == pad_name
                } else {
                    edge.target() == node_id.0 && link.sink_pad == pad_name
                }
            })
            .map(|edge| edge.id())
            .collect();

        // Remove the edges
        for edge_id in edges_to_remove {
            self.graph.remove_edge(edge_id);
        }

        Ok(())
    }

    /// Check if the pipeline needs re-negotiation.
    ///
    /// Returns `true` if pads have been added/removed since the last negotiation.
    pub fn needs_renegotiation(&self) -> bool {
        self.negotiation.is_none()
    }

    /// Invalidate the current negotiation, forcing re-negotiation before execution.
    pub fn invalidate_negotiation(&mut self) {
        self.negotiation = None;
    }

    /// Generate a human-readable description of the pipeline.
    ///
    /// Shows all elements, their connections, and negotiated formats.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::pipeline::Pipeline;
    ///
    /// let pipeline = Pipeline::parse("nullsource ! passthrough ! nullsink").unwrap();
    /// println!("{}", pipeline.describe());
    /// ```
    pub fn describe(&self) -> String {
        let mut out = String::new();

        writeln!(out, "Pipeline ({:?})", self.state).unwrap();
        writeln!(out, "  Nodes: {}", self.node_count()).unwrap();
        writeln!(out, "  Links: {}", self.edge_count()).unwrap();
        writeln!(
            out,
            "  Negotiated: {}",
            if self.is_negotiated() { "yes" } else { "no" }
        )
        .unwrap();
        writeln!(out).unwrap();

        // List nodes
        writeln!(out, "Elements:").unwrap();
        for (id, node) in self.nodes() {
            let input = node.input_caps();
            let output = node.output_caps();
            let caps_info = format!(
                "in={}, out={}",
                if input.is_any() {
                    "any".to_string()
                } else {
                    format!("{:?}", input.formats())
                },
                if output.is_any() {
                    "any".to_string()
                } else {
                    format!("{:?}", output.formats())
                }
            );

            writeln!(
                out,
                "  [{}] {} ({:?}) - {}",
                id.index(),
                node.name(),
                node.element_type(),
                caps_info
            )
            .unwrap();
        }
        writeln!(out).unwrap();

        // List links with negotiation info
        writeln!(out, "Links:").unwrap();
        for link in self.links() {
            write!(
                out,
                "  {} ({}) -> {} ({})",
                link.source_name, link.source_pad, link.sink_name, link.sink_pad
            )
            .unwrap();

            if let Some(format) = &link.negotiated_format {
                write!(out, " [format: {:?}", format).unwrap();
                if let Some(mem) = link.negotiated_memory {
                    write!(out, ", memory: {:?}", mem).unwrap();
                }
                write!(out, "]").unwrap();
            }
            writeln!(out).unwrap();
        }

        // List pending converters
        let converters = self.pending_converters();
        if !converters.is_empty() {
            writeln!(out).unwrap();
            writeln!(out, "Converters to insert:").unwrap();
            for conv in converters {
                writeln!(
                    out,
                    "  Link {}: {} (cost: {})",
                    conv.link_id, conv.reason, conv.cost
                )
                .unwrap();
            }
        }

        out
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    /// Parse a pipeline description string and build a pipeline.
    ///
    /// Uses the default element factory with built-in elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::pipeline::Pipeline;
    ///
    /// let pipeline = Pipeline::parse("nullsource count=10 ! passthrough ! nullsink").unwrap();
    /// assert_eq!(pipeline.node_count(), 3);
    /// ```
    pub fn parse(description: &str) -> Result<Self> {
        Self::parse_with_factory(
            description,
            &crate::pipeline::factory::ElementFactory::new(),
        )
    }

    /// Parse a pipeline description string using a custom factory.
    pub fn parse_with_factory(
        description: &str,
        factory: &crate::pipeline::factory::ElementFactory,
    ) -> Result<Self> {
        let parsed = crate::pipeline::parser::parse_pipeline(description)?;

        let mut pipeline = Pipeline::new();
        let mut prev_node: Option<NodeId> = None;

        for (i, elem) in parsed.elements.iter().enumerate() {
            let element = factory.create(elem)?;
            let name = format!("{}_{}", elem.name, i);
            let node_id = pipeline.add_node(name, element);

            // Link to previous element
            if let Some(prev) = prev_node {
                pipeline.link(prev, node_id)?;
            }

            prev_node = Some(node_id);
        }

        Ok(pipeline)
    }
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("nodes", &self.node_count())
            .field("edges", &self.edge_count())
            .field("state", &self.state)
            .finish()
    }
}

// ============================================================================
// Graph Export (DOT, JSON)
// ============================================================================

impl Pipeline {
    /// Export the pipeline graph to DOT format (Graphviz).
    ///
    /// This can be rendered using `dot -Tpng pipeline.dot -o pipeline.png`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::pipeline::Pipeline;
    ///
    /// let mut pipeline = Pipeline::parse("nullsource ! passthrough ! nullsink").unwrap();
    /// let dot = pipeline.to_dot();
    /// println!("{}", dot);
    /// ```
    pub fn to_dot(&self) -> String {
        self.to_dot_with_options(&DotOptions::default())
    }

    /// Export the pipeline graph to DOT format with custom options.
    pub fn to_dot_with_options(&self, options: &DotOptions) -> String {
        let mut dot = String::new();

        // Graph header
        dot.push_str("digraph pipeline {\n");
        dot.push_str("    rankdir=LR;\n");
        dot.push_str("    node [shape=box, style=rounded];\n");
        dot.push('\n');

        // Add nodes
        for idx in self.graph.graph().node_indices() {
            let node = self.graph.node_weight(idx).unwrap();
            let (shape, color) = match node.element_type() {
                ElementType::Source => ("ellipse", "lightgreen"),
                ElementType::Sink => ("ellipse", "lightcoral"),
                ElementType::Transform => ("box", "lightblue"),
                ElementType::Demuxer => ("trapezium", "lightyellow"),
                ElementType::Muxer => ("invtrapezium", "lightgoldenrod"),
            };

            let label = if options.show_element_type {
                format!("{}\\n({:?})", node.name(), node.element_type())
            } else {
                node.name().to_string()
            };

            dot.push_str(&format!(
                "    \"{}\" [label=\"{}\", shape={}, fillcolor={}, style=filled];\n",
                node.name(),
                label,
                shape,
                color
            ));
        }

        dot.push('\n');

        // Add edges
        for edge_idx in self.graph.graph().edge_indices() {
            let (src_idx, sink_idx) = self.graph.graph().edge_endpoints(edge_idx).unwrap();
            let src_node = self.graph.node_weight(src_idx).unwrap();
            let sink_node = self.graph.node_weight(sink_idx).unwrap();
            let link = self.graph.edge_weight(edge_idx).unwrap();

            let edge_label = if options.show_pad_names {
                format!(" [label=\"{} -> {}\"]", link.src_pad, link.sink_pad)
            } else {
                String::new()
            };

            dot.push_str(&format!(
                "    \"{}\" -> \"{}\"{};\n",
                src_node.name(),
                sink_node.name(),
                edge_label
            ));
        }

        // Legend (optional)
        if options.show_legend {
            dot.push('\n');
            dot.push_str("    subgraph cluster_legend {\n");
            dot.push_str("        label=\"Legend\";\n");
            dot.push_str("        style=dashed;\n");
            dot.push_str("        legend_source [label=\"Source\", shape=ellipse, fillcolor=lightgreen, style=filled];\n");
            dot.push_str("        legend_transform [label=\"Transform\", shape=box, fillcolor=lightblue, style=filled];\n");
            dot.push_str("        legend_sink [label=\"Sink\", shape=ellipse, fillcolor=lightcoral, style=filled];\n");
            dot.push_str("    }\n");
        }

        dot.push_str("}\n");
        dot
    }

    /// Export the pipeline graph to JSON format.
    ///
    /// The JSON structure includes nodes and edges arrays.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::pipeline::Pipeline;
    ///
    /// let mut pipeline = Pipeline::parse("nullsource ! passthrough ! nullsink").unwrap();
    /// let json = pipeline.to_json();
    /// println!("{}", json);
    /// ```
    pub fn to_json(&self) -> String {
        let mut json = String::new();

        json.push_str("{\n");

        // Pipeline metadata
        json.push_str(&format!("  \"state\": \"{:?}\",\n", self.state));
        json.push_str(&format!("  \"node_count\": {},\n", self.node_count()));
        json.push_str(&format!("  \"edge_count\": {},\n", self.edge_count()));

        // Nodes array
        json.push_str("  \"nodes\": [\n");
        let node_indices: Vec<_> = self.graph.graph().node_indices().collect();
        for (i, idx) in node_indices.iter().enumerate() {
            let node = self.graph.node_weight(*idx).unwrap();
            json.push_str("    {\n");
            json.push_str(&format!("      \"id\": {},\n", idx.index()));
            json.push_str(&format!("      \"name\": \"{}\",\n", node.name()));
            json.push_str(&format!("      \"type\": \"{:?}\",\n", node.element_type()));
            json.push_str(&format!(
                "      \"input_pads\": {},\n",
                node.input_pads().len()
            ));
            json.push_str(&format!(
                "      \"output_pads\": {}\n",
                node.output_pads().len()
            ));
            if i < node_indices.len() - 1 {
                json.push_str("    },\n");
            } else {
                json.push_str("    }\n");
            }
        }
        json.push_str("  ],\n");

        // Edges array
        json.push_str("  \"edges\": [\n");
        let edge_indices: Vec<_> = self.graph.graph().edge_indices().collect();
        for (i, edge_idx) in edge_indices.iter().enumerate() {
            let (src_idx, sink_idx) = self.graph.graph().edge_endpoints(*edge_idx).unwrap();
            let link = self.graph.edge_weight(*edge_idx).unwrap();
            json.push_str("    {\n");
            json.push_str(&format!("      \"from\": {},\n", src_idx.index()));
            json.push_str(&format!("      \"to\": {},\n", sink_idx.index()));
            json.push_str(&format!("      \"src_pad\": \"{}\",\n", link.src_pad));
            json.push_str(&format!("      \"sink_pad\": \"{}\"\n", link.sink_pad));
            if i < edge_indices.len() - 1 {
                json.push_str("    },\n");
            } else {
                json.push_str("    }\n");
            }
        }
        json.push_str("  ]\n");

        json.push_str("}\n");
        json
    }

    /// Write the DOT representation to a file.
    pub fn write_dot(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_dot())?;
        Ok(())
    }

    /// Write the JSON representation to a file.
    pub fn write_json(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_json())?;
        Ok(())
    }
}

// ============================================================================
// Execution with Isolation
// ============================================================================

impl Pipeline {
    /// Run the pipeline with the default in-process executor.
    ///
    /// This is the simplest way to run a pipeline - all elements execute
    /// as Tokio tasks in the current process.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use parallax::pipeline::Pipeline;
    ///
    /// let mut pipeline = Pipeline::parse("nullsource count=10 ! nullsink")?;
    /// pipeline.run().await?;
    /// ```
    pub async fn run(&mut self) -> Result<()> {
        let executor = crate::pipeline::Executor::new();
        executor.run(self).await
    }

    /// Run the pipeline with custom executor configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use parallax::pipeline::{Pipeline, UnifiedExecutorConfig, SchedulingMode};
    ///
    /// let mut pipeline = Pipeline::parse("audiosrc ! sink")?;
    ///
    /// // Use low-latency audio configuration
    /// pipeline.run_with_config(UnifiedExecutorConfig::low_latency_audio()).await?;
    /// ```
    pub async fn run_with_config(
        &mut self,
        config: crate::pipeline::UnifiedExecutorConfig,
    ) -> Result<()> {
        let executor = crate::pipeline::Executor::with_config(config);
        executor.run(self).await
    }

    /// Start the pipeline and return a handle for control.
    ///
    /// Unlike `run()`, this returns immediately with a handle that can be
    /// used to wait for completion, abort, or subscribe to events.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use parallax::pipeline::Pipeline;
    ///
    /// let mut pipeline = Pipeline::parse("source ! sink")?;
    /// let handle = pipeline.start()?;
    ///
    /// // Subscribe to events
    /// let mut events = handle.subscribe();
    /// tokio::spawn(async move {
    ///     while let Ok(event) = events.recv().await {
    ///         println!("Event: {:?}", event);
    ///     }
    /// });
    ///
    /// // Wait for completion
    /// handle.wait().await?;
    /// ```
    pub fn start(&mut self) -> Result<crate::pipeline::UnifiedPipelineHandle> {
        let executor = crate::pipeline::Executor::new();
        executor.start(self)
    }

    /// Start the pipeline with custom executor configuration.
    pub fn start_with_config(
        &mut self,
        config: crate::pipeline::UnifiedExecutorConfig,
    ) -> Result<crate::pipeline::UnifiedPipelineHandle> {
        let executor = crate::pipeline::Executor::with_config(config);
        executor.start(self)
    }

    /// Run the pipeline with a specific execution mode.
    ///
    /// This allows transparent process isolation without manually adding
    /// IpcSrc/IpcSink elements. The executor automatically injects IPC
    /// boundaries where needed.
    ///
    /// # Execution Modes
    ///
    /// - `InProcess`: All elements run in the current process (default, fastest)
    /// - `Isolated`: Each element runs in its own sandboxed process (maximum isolation)
    /// - `Grouped`: Selective isolation based on patterns (balance of safety and performance)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use parallax::pipeline::Pipeline;
    /// use parallax::execution::ExecutionMode;
    ///
    /// let mut pipeline = Pipeline::parse("filesrc ! h264dec ! displaysink")?;
    ///
    /// // Isolate decoders (they process untrusted input)
    /// pipeline.run_with_mode(ExecutionMode::grouped(vec!["*dec*".to_string()])).await?;
    /// ```
    pub async fn run_with_mode(self, mode: crate::execution::ExecutionMode) -> Result<()> {
        let executor = crate::execution::IsolatedExecutor::new(mode);
        executor.run(self).await
    }

    /// Run the pipeline with full isolation for all elements.
    ///
    /// Each element runs in its own sandboxed process. This provides
    /// maximum security at the cost of IPC overhead.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use parallax::pipeline::Pipeline;
    ///
    /// let mut pipeline = Pipeline::parse("filesrc ! decoder ! sink")?;
    /// pipeline.run_isolated().await?;
    /// ```
    pub async fn run_isolated(self) -> Result<()> {
        self.run_with_mode(crate::execution::ExecutionMode::isolated())
            .await
    }

    /// Run the pipeline with selective isolation.
    ///
    /// Elements matching any of the patterns run in isolated processes.
    /// Other elements run together in the main process.
    ///
    /// # Pattern Syntax
    ///
    /// - `*` matches any sequence of characters
    /// - `?` matches any single character
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use parallax::pipeline::Pipeline;
    ///
    /// let mut pipeline = Pipeline::parse("filesrc ! h264dec ! x264enc ! filesink")?;
    ///
    /// // Isolate all codecs (decoders and encoders)
    /// pipeline.run_isolating(vec!["*dec*", "*enc*"]).await?;
    /// ```
    pub async fn run_isolating(self, patterns: Vec<&str>) -> Result<()> {
        let patterns: Vec<String> = patterns.into_iter().map(|s| s.to_string()).collect();
        self.run_with_mode(crate::execution::ExecutionMode::grouped(patterns))
            .await
    }
}

/// Options for DOT export.
#[derive(Debug, Clone)]
pub struct DotOptions {
    /// Show element type in node labels.
    pub show_element_type: bool,
    /// Show pad names on edges.
    pub show_pad_names: bool,
    /// Include a legend subgraph.
    pub show_legend: bool,
}

impl Default for DotOptions {
    fn default() -> Self {
        Self {
            show_element_type: true,
            show_pad_names: false,
            show_legend: false,
        }
    }
}

impl DotOptions {
    /// Create options with all details shown.
    pub fn verbose() -> Self {
        Self {
            show_element_type: true,
            show_pad_names: true,
            show_legend: true,
        }
    }

    /// Create minimal options.
    pub fn minimal() -> Self {
        Self {
            show_element_type: false,
            show_pad_names: false,
            show_legend: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::element::{
        ConsumeContext, DynAsyncElement, Element, ElementAdapter, PadDirection, ProduceContext,
        ProduceResult, Sink, SinkAdapter, Source, SourceAdapter,
    };

    struct TestSource;
    impl Source for TestSource {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
            Ok(ProduceResult::Eos)
        }
    }

    struct TestSink;
    impl Sink for TestSink {
        fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
            Ok(())
        }
    }

    struct TestElement;
    impl Element for TestElement {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            Ok(Some(buffer))
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = Pipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.state(), PipelineState::Suspended);
    }

    #[test]
    fn test_add_nodes() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let filter = pipeline.add_node(
            "filter",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        assert_eq!(pipeline.node_count(), 3);
        assert_eq!(pipeline.get_node_id("src"), Some(src));
        assert_eq!(pipeline.get_node_id("filter"), Some(filter));
        assert_eq!(pipeline.get_node_id("sink"), Some(sink));
    }

    #[test]
    fn test_link_nodes() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let filter = pipeline.add_node(
            "filter",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, filter).unwrap();
        pipeline.link(filter, sink).unwrap();

        assert_eq!(pipeline.edge_count(), 2);

        // Check children/parents
        let src_children = pipeline.children(src);
        assert_eq!(src_children.len(), 1);
        assert_eq!(src_children[0].0, filter);

        let filter_parents = pipeline.parents(filter);
        assert_eq!(filter_parents.len(), 1);
        assert_eq!(filter_parents[0].0, src);
    }

    #[test]
    fn test_sources_and_sinks() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let filter = pipeline.add_node(
            "filter",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, filter).unwrap();
        pipeline.link(filter, sink).unwrap();

        let sources = pipeline.sources();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0], src);

        let sinks = pipeline.sinks();
        assert_eq!(sinks.len(), 1);
        assert_eq!(sinks[0], sink);
    }

    #[test]
    fn test_cycle_detection() {
        let mut pipeline = Pipeline::new();

        let a = pipeline.add_node(
            "a",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );
        let b = pipeline.add_node(
            "b",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );

        pipeline.link(a, b).unwrap();

        // This should fail because it would create a cycle
        let result = pipeline.link(b, a);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_empty_pipeline() {
        let pipeline = Pipeline::new();
        assert!(pipeline.validate().is_err());
    }

    #[test]
    fn test_validate_valid_pipeline() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, sink).unwrap();

        assert!(pipeline.validate().is_ok());
    }

    #[test]
    fn test_auto_naming() {
        let mut pipeline = Pipeline::new();

        let n1 = pipeline.add_node_auto(DynAsyncElement::new_box(SourceAdapter::new(TestSource)));
        let n2 = pipeline.add_node_auto(DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        assert_eq!(pipeline.get_node(n1).unwrap().name(), "node_0");
        assert_eq!(pipeline.get_node(n2).unwrap().name(), "node_1");
    }

    // ========================================================================
    // Introspection API Tests
    // ========================================================================

    #[test]
    fn test_nodes_iterator() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let filter = pipeline.add_node(
            "filter",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, filter).unwrap();
        pipeline.link(filter, sink).unwrap();

        // Test nodes() iterator
        let nodes: Vec<_> = pipeline.nodes().collect();
        assert_eq!(nodes.len(), 3);

        // Check all nodes are present
        let names: Vec<_> = nodes.iter().map(|(_, n)| n.name()).collect();
        assert!(names.contains(&"src"));
        assert!(names.contains(&"filter"));
        assert!(names.contains(&"sink"));
    }

    #[test]
    fn test_links_iterator() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let filter = pipeline.add_node(
            "filter",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, filter).unwrap();
        pipeline.link(filter, sink).unwrap();

        // Test links() iterator
        let links: Vec<_> = pipeline.links().collect();
        assert_eq!(links.len(), 2);

        // Check link info
        let link1 = &links[0];
        assert_eq!(link1.source_name, "src");
        assert_eq!(link1.sink_name, "filter");
        assert_eq!(link1.source_pad, "src");
        assert_eq!(link1.sink_pad, "sink");
        // Before negotiation, format should be None
        assert!(link1.negotiated_format.is_none());
    }

    #[test]
    fn test_node_caps() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );

        // Test cached caps
        let node = pipeline.get_node(src).unwrap();
        // TestSource doesn't override caps, so should be Any
        assert!(node.input_caps().is_any());
        assert!(node.output_caps().is_any());
    }

    #[test]
    fn test_negotiate_pipeline() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, sink).unwrap();

        // Before negotiation
        assert!(!pipeline.is_negotiated());

        // Run negotiation
        pipeline.negotiate().unwrap();

        // After negotiation
        assert!(pipeline.is_negotiated());
        assert!(pipeline.negotiation_result().is_some());

        // Check link has negotiated format
        let links: Vec<_> = pipeline.links().collect();
        assert_eq!(links.len(), 1);
        // With Any/Any caps, negotiation should still succeed with defaults
        assert!(links[0].negotiated_format.is_some());
    }

    #[test]
    fn test_describe_pipeline() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, sink).unwrap();

        // Get description
        let desc = pipeline.describe();

        // Should contain key info
        assert!(desc.contains("Pipeline"));
        assert!(desc.contains("Nodes: 2"));
        assert!(desc.contains("Links: 1"));
        assert!(desc.contains("src"));
        assert!(desc.contains("sink"));
        assert!(desc.contains("Source"));
        assert!(desc.contains("Sink"));
    }

    #[test]
    fn test_get_link() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, sink).unwrap();

        // Get link via iterator first to get the ID
        let links: Vec<_> = pipeline.links().collect();
        let link_id = links[0].id;

        // Now test get_link
        let link = pipeline.get_link(link_id).unwrap();
        assert_eq!(link.source_name, "src");
        assert_eq!(link.sink_name, "sink");
    }

    #[test]
    fn test_link_format_query() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, sink).unwrap();

        // Get link ID
        let links: Vec<_> = pipeline.links().collect();
        let link_id = links[0].id;

        // Before negotiation
        assert!(pipeline.link_format(link_id).is_none());
        assert!(pipeline.link_memory_type(link_id).is_none());

        // After negotiation
        pipeline.negotiate().unwrap();
        assert!(pipeline.link_format(link_id).is_some());
        assert!(pipeline.link_memory_type(link_id).is_some());
    }

    // ========================================================================
    // Dynamic Pad Management Tests
    // ========================================================================

    #[test]
    fn test_add_output_pad() {
        let mut pipeline = Pipeline::new();

        let node_id = pipeline.add_node(
            "demux",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );

        // Initially has default pads
        let node = pipeline.get_node(node_id).unwrap();
        assert_eq!(node.output_pads().len(), 1); // default "src" pad

        // Add a new output pad
        pipeline
            .add_output_pad(node_id, Pad::new("video_0", PadDirection::Output))
            .unwrap();

        let node = pipeline.get_node(node_id).unwrap();
        assert_eq!(node.output_pads().len(), 2);
        assert!(node.get_output_pad("video_0").is_some());
    }

    #[test]
    fn test_add_input_pad() {
        let mut pipeline = Pipeline::new();

        let node_id = pipeline.add_node(
            "mux",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );

        // Initially has default pads
        let node = pipeline.get_node(node_id).unwrap();
        assert_eq!(node.input_pads().len(), 1); // default "sink" pad

        // Add a new input pad
        pipeline
            .add_input_pad(node_id, Pad::new("audio_0", PadDirection::Input))
            .unwrap();

        let node = pipeline.get_node(node_id).unwrap();
        assert_eq!(node.input_pads().len(), 2);
        assert!(node.get_input_pad("audio_0").is_some());
    }

    #[test]
    fn test_add_duplicate_pad_fails() {
        let mut pipeline = Pipeline::new();

        let node_id = pipeline.add_node(
            "node",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );

        // Try to add a pad with the same name as default
        let result = pipeline.add_output_pad(node_id, Pad::new("src", PadDirection::Output));
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_output_pad() {
        let mut pipeline = Pipeline::new();

        let node_id = pipeline.add_node(
            "demux",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );

        // Add a pad then remove it
        pipeline
            .add_output_pad(node_id, Pad::new("video_0", PadDirection::Output))
            .unwrap();

        let removed = pipeline.remove_output_pad(node_id, "video_0").unwrap();
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name(), "video_0");

        // Verify it's gone
        let node = pipeline.get_node(node_id).unwrap();
        assert!(node.get_output_pad("video_0").is_none());
    }

    #[test]
    fn test_remove_pad_removes_links() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        // Link the nodes
        pipeline.link(src, sink).unwrap();
        assert_eq!(pipeline.edge_count(), 1);

        // Remove the output pad - should also remove the link
        pipeline.remove_output_pad(src, "src").unwrap();
        assert_eq!(pipeline.edge_count(), 0);
    }

    #[test]
    fn test_dynamic_pad_invalidates_negotiation() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        pipeline.link(src, sink).unwrap();

        // Negotiate
        pipeline.negotiate().unwrap();
        assert!(pipeline.is_negotiated());

        // Add a pad - should invalidate negotiation
        pipeline
            .add_output_pad(src, Pad::new("extra", PadDirection::Output))
            .unwrap();
        assert!(!pipeline.is_negotiated());
        assert!(pipeline.needs_renegotiation());
    }

    #[test]
    fn test_link_pads_with_custom_names() {
        let mut pipeline = Pipeline::new();

        let demux = pipeline.add_node(
            "demux",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );
        let decoder = pipeline.add_node(
            "decoder",
            DynAsyncElement::new_box(ElementAdapter::new(TestElement)),
        );

        // Add custom pads
        pipeline
            .add_output_pad(demux, Pad::new("video_0", PadDirection::Output))
            .unwrap();
        pipeline
            .add_input_pad(decoder, Pad::new("video_in", PadDirection::Input))
            .unwrap();

        // Link using specific pad names
        pipeline
            .link_pads(demux, "video_0", decoder, "video_in")
            .unwrap();

        // Verify the link
        let links: Vec<_> = pipeline.links().collect();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].source_pad, "video_0");
        assert_eq!(links[0].sink_pad, "video_in");
    }

    #[test]
    fn test_link_pads_nonexistent_pad_fails() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));

        // Try to link with a non-existent source pad
        let result = pipeline.link_pads(src, "nonexistent", sink, "sink");
        assert!(result.is_err());

        // Try to link with a non-existent sink pad
        let result = pipeline.link_pads(src, "src", sink, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_state_transitions() {
        let mut pipeline = Pipeline::new();

        // Initial state is Suspended
        assert_eq!(pipeline.state(), PipelineState::Suspended);
        assert!(!pipeline.is_running());
        assert!(!pipeline.is_runnable());

        // Add elements to make it valid
        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));
        pipeline.link(src, sink).unwrap();

        // Suspended -> Idle (prepare)
        pipeline.prepare().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Idle);
        assert!(pipeline.is_runnable());
        assert!(!pipeline.is_running());

        // Idle -> Running (activate)
        pipeline.activate().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Running);
        assert!(pipeline.is_running());
        assert!(pipeline.is_runnable());

        // Running -> Idle (pause)
        pipeline.pause().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Idle);
        assert!(!pipeline.is_running());

        // Idle -> Suspended (suspend)
        pipeline.suspend().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Suspended);
        assert!(!pipeline.is_runnable());
    }

    #[test]
    fn test_invalid_state_transitions() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));
        pipeline.link(src, sink).unwrap();

        // Cannot activate from Suspended
        assert!(pipeline.activate().is_err());

        // Cannot pause from Suspended
        assert!(pipeline.pause().is_err());

        // Prepare to Idle
        pipeline.prepare().unwrap();

        // Cannot suspend while Running (need to pause first)
        pipeline.activate().unwrap();
        assert!(pipeline.suspend().is_err());

        // Pause first, then suspend works
        pipeline.pause().unwrap();
        pipeline.suspend().unwrap();
    }

    #[test]
    fn test_error_state_recovery() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));
        pipeline.link(src, sink).unwrap();

        // Simulate error
        pipeline.set_error();
        assert!(pipeline.has_error());
        assert_eq!(pipeline.state(), PipelineState::Error);

        // Cannot prepare/activate from error
        assert!(pipeline.prepare().is_err());
        assert!(pipeline.activate().is_err());

        // Can recover by suspending
        pipeline.suspend().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Suspended);
        assert!(!pipeline.has_error());

        // Now can prepare again
        pipeline.prepare().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Idle);
    }

    #[test]
    fn test_idempotent_transitions() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(TestSource)),
        );
        let sink = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(TestSink)));
        pipeline.link(src, sink).unwrap();

        // Double prepare is idempotent
        pipeline.prepare().unwrap();
        pipeline.prepare().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Idle);

        // Double activate is idempotent
        pipeline.activate().unwrap();
        pipeline.activate().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Running);

        // Double pause is idempotent
        pipeline.pause().unwrap();
        pipeline.pause().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Idle);

        // Double suspend is idempotent
        pipeline.suspend().unwrap();
        pipeline.suspend().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Suspended);
    }
}
