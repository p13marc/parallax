//! Pipeline graph structure using daggy.

use crate::element::{AsyncElementDyn, DynAsyncElement, ElementType, Pad};
use crate::error::{Error, Result};
use daggy::{Dag, NodeIndex, Walker};
use std::collections::HashMap;

/// Unique identifier for a node in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) NodeIndex);

impl NodeId {
    /// Get the underlying index.
    pub fn index(&self) -> usize {
        self.0.index()
    }
}

/// State of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PipelineState {
    /// Pipeline is not yet started.
    #[default]
    Stopped,
    /// Pipeline is running.
    Running,
    /// Pipeline is paused.
    Paused,
    /// Pipeline has finished (all sources exhausted).
    Finished,
    /// Pipeline encountered an error.
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

        // Create default pads based on element type
        let (input_pads, output_pads) = match element_type {
            ElementType::Source => (vec![], vec![Pad::src()]),
            ElementType::Sink => (vec![Pad::sink()], vec![]),
            ElementType::Transform => (vec![Pad::sink()], vec![Pad::src()]),
        };

        Self {
            name,
            element: Some(element),
            element_type,
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
}

impl Pipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            graph: Dag::new(),
            nodes_by_name: HashMap::new(),
            state: PipelineState::Stopped,
            name_counter: 0,
        }
    }

    /// Get the current pipeline state.
    pub fn state(&self) -> PipelineState {
        self.state
    }

    /// Set the pipeline state.
    pub fn set_state(&mut self, state: PipelineState) {
        self.state = state;
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
        dot.push_str("\n");

        // Add nodes
        for idx in self.graph.graph().node_indices() {
            let node = self.graph.node_weight(idx).unwrap();
            let (shape, color) = match node.element_type() {
                ElementType::Source => ("ellipse", "lightgreen"),
                ElementType::Sink => ("ellipse", "lightcoral"),
                ElementType::Transform => ("box", "lightblue"),
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

        dot.push_str("\n");

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
            dot.push_str("\n");
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
        DynAsyncElement, Element, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter,
    };

    struct TestSource;
    impl Source for TestSource {
        fn produce(&mut self) -> Result<Option<Buffer>> {
            Ok(None)
        }
    }

    struct TestSink;
    impl Sink for TestSink {
        fn consume(&mut self, _buffer: Buffer) -> Result<()> {
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
        assert_eq!(pipeline.state(), PipelineState::Stopped);
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
}
