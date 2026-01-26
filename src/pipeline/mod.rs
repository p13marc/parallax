//! Pipeline construction and execution.
//!
//! This module provides the core pipeline infrastructure:
//!
//! - [`Pipeline`]: The main pipeline container and DAG
//! - [`Node`]: A node in the pipeline graph (wraps an element)
//! - [`Link`]: A connection between nodes
//! - [`PipelineEvent`]: Async events emitted during execution
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::Pipeline;
//! use parallax::elements::{FileSrc, PassThrough, ConsoleSink};
//!
//! let mut pipeline = Pipeline::new();
//!
//! // Add elements
//! let src = pipeline.add_source("src", FileSrc::new("input.bin"));
//! let filter = pipeline.add_element("filter", PassThrough::new());
//! let sink = pipeline.add_sink("sink", ConsoleSink::new());
//!
//! // Link them
//! pipeline.link(src, filter)?;
//! pipeline.link(filter, sink)?;
//!
//! // Run the pipeline
//! pipeline.run().await?;
//! ```

mod events;
mod executor;
pub mod factory;
mod graph;
pub mod parser;
pub mod rt_bridge;

pub use events::{EventReceiver, EventSender, EventStream, PipelineEvent};
pub use executor::{ExecutorConfig, PipelineExecutor, PipelineHandle};
pub use factory::ElementFactory;
pub use graph::{DotOptions, Link, LinkId, LinkInfo, Node, NodeId, Pipeline, PipelineState};
pub use parser::{ParsedElement, ParsedPipeline, PropertyValue, parse_pipeline};
pub use rt_bridge::{AsyncRtBridge, BridgeConfig, EventFd, SharedBridge, shared_bridge};
