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

mod driver;
mod events;
mod executor;
pub mod factory;
mod graph;
mod hybrid_executor;
pub mod parser;
pub mod rt_bridge;
pub mod rt_scheduler;
mod unified_executor;

pub use driver::{
    DriverConfig, DriverStats, ManualDriver, RtTimerDriverHandle, TimerDriver, TimerDriverHandle,
};
pub use events::{EventReceiver, EventSender, EventStream, PipelineEvent};
pub use factory::ElementFactory;
pub use graph::{DotOptions, Link, LinkId, LinkInfo, Node, NodeId, Pipeline, PipelineState};
pub use parser::{ParsedElement, ParsedPipeline, PropertyValue, parse_pipeline};
pub use rt_bridge::{AsyncRtBridge, BridgeConfig, EventFd, SharedBridge, shared_bridge};
pub use rt_scheduler::{
    ActivationRecord, BoundaryDirection, BoundaryEdge, DataThreadHandle, GraphPartition,
    NodeStatus, RtConfig, RtScheduler, SchedulingMode,
};
// Unified executor (primary API)
pub use unified_executor::{
    Executor, ExecutorConfig as UnifiedExecutorConfig, PipelineHandle as UnifiedPipelineHandle,
};

// Legacy exports (deprecated - use Executor instead)
#[deprecated(since = "0.2.0", note = "Use UnifiedExecutorConfig instead")]
pub use executor::ExecutorConfig;
#[deprecated(since = "0.2.0", note = "Use Executor instead")]
pub use executor::PipelineExecutor;
#[deprecated(since = "0.2.0", note = "Use UnifiedPipelineHandle instead")]
pub use executor::PipelineHandle;
#[deprecated(since = "0.2.0", note = "Use Executor instead")]
pub use hybrid_executor::HybridExecutor;
#[deprecated(since = "0.2.0", note = "Use UnifiedPipelineHandle instead")]
pub use hybrid_executor::HybridPipelineHandle;
