//! Pipeline construction and execution.
//!
//! This module provides the core pipeline infrastructure:
//!
//! - [`Pipeline`]: The main pipeline container and DAG
//! - [`PipelineBuilder`]: Fluent builder for constructing pipelines
//! - [`Node`]: A node in the pipeline graph (wraps an element)
//! - [`Link`]: A connection between nodes
//! - [`PipelineEvent`]: Async events emitted during execution
//!
//! # Builder Example
//!
//! ```rust,ignore
//! use parallax::pipeline::PipelineBuilder;
//!
//! // Fluent builder API
//! let pipeline = PipelineBuilder::new()
//!     .source(VideoTestSrc::new())
//!     .then(VideoScale::new(1920, 1080, 1280, 720))
//!     .sink(FileSink::new("output.yuv"))
//!     .build()?;
//!
//! pipeline.run().await?;
//! ```
//!
//! # Manual Example
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

mod builder;
mod driver;
mod events;
pub mod factory;
mod graph;
pub mod parser;
pub mod rt_bridge;
pub mod rt_scheduler;
mod unified_executor;

pub use driver::{
    DriverConfig, DriverStats, ManualDriver, RtTimerDriverHandle, TimerDriver, TimerDriverHandle,
};
pub use events::{EventReceiver, EventSender, EventStream, PipelineEvent};
pub use factory::ElementFactory;
pub use graph::{
    ConverterPolicy, DotOptions, Link, LinkId, LinkInfo, Node, NodeId, Pipeline, PipelineState,
};
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

// Builder API
pub use builder::{
    BranchBuilder, BuiltPipeline, ChainedTransform, ChainedTransform2, FromSource, PipelineBuilder,
    PipelineFragment, TeeBuilder, ToSink, from, to,
};
