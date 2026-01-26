//! Process execution and isolation for pipeline elements.
//!
//! This module provides different execution modes for running pipeline elements:
//!
//! - **InProcess**: All elements run as Tokio tasks in a single process (default)
//! - **Isolated**: Each element runs in a separate sandboxed process
//! - **Grouped**: Elements are grouped to minimize process count while isolating untrusted code
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Supervisor                               │
//! │  - Owns shared memory arenas                                    │
//! │  - Spawns/monitors element processes                            │
//! │  - Handles crash recovery                                       │
//! └───────────────┬─────────────────────────────────────────────────┘
//!                 │ Control (Unix socket + rkyv)
//!     ┌───────────┼───────────┬───────────────────┐
//!     ▼           ▼           ▼                   ▼
//! ┌───────┐  ┌───────┐  ┌───────────────────┐ ┌───────┐
//! │Element│  │Element│  │    Element        │ │Element│
//! │Process│  │Process│  │    Process        │ │Process│
//! │(codec)│  │(filter)│ │(untrusted plugin) │ │(sink) │
//! └───────┘  └───────┘  └───────────────────┘ └───────┘
//!     │           │              │                 │
//!     └───────────┴──────────────┴─────────────────┘
//!                 Shared Memory (memfd arenas)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::execution::{ExecutionMode, ElementSandbox};
//! use parallax::pipeline::Pipeline;
//!
//! let mut pipeline = Pipeline::new();
//! pipeline.parse("filesrc ! decoder ! videosink")?;
//!
//! // Run with element isolation
//! pipeline.run_with_mode(ExecutionMode::Isolated {
//!     sandbox: ElementSandbox::default(),
//! }).await?;
//!
//! // Or group elements to minimize overhead
//! pipeline.run_with_mode(ExecutionMode::Grouped {
//!     isolated_patterns: vec!["decoder".into()],  // Isolate decoders
//!     sandbox: ElementSandbox::default(),
//!     groups: None,  // Auto-group the rest
//! }).await?;
//! ```

mod mode;
mod protocol;
mod sandbox;
mod supervisor;

pub use mode::{ExecutionMode, GroupId};
pub use protocol::{
    ControlMessage, ElementState, SerializableCaps, SerializableMetadata, frame_message,
    unframe_message,
};
pub use sandbox::{
    AllowedPath, ArgConstraint, ArgOp, CgroupLimits, ElementSandbox, SeccompPolicy, SeccompRule,
};
pub use supervisor::{
    BackoffStrategy, ElementId, ElementProcess, ElementStats, RestartPolicy, Supervisor,
};
