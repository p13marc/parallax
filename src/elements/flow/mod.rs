//! Flow control and routing elements.
//!
//! ## Buffering
//! - [`Queue`]: Async buffer queue with backpressure
//!
//! ## Routing
//! - [`Tee`]: 1-to-N fanout (duplicates buffers)
//! - [`Funnel`]: N-to-1 merge
//! - [`InputSelector`]: N-to-1 switching (selects one input)
//! - [`OutputSelector`]: 1-to-N routing (routes to one output)
//! - [`Concat`]: Sequential stream concatenation
//!
//! ## Control
//! - [`Valve`]: On/off flow control

mod concat;
mod funnel;
mod queue;
mod selector;
mod tee;
mod valve;

pub use concat::{Concat, ConcatStats, ConcatStream};
pub use funnel::{Funnel, FunnelInput, FunnelStats};
pub use queue::{LeakyMode, Queue, QueueStats};
pub use selector::{
    InputSelector, InputSelectorStats, OutputSelector, OutputSelectorStats, SelectorInput,
    SelectorOutput,
};
pub use tee::Tee;
pub use valve::{Valve, ValveControl, ValveStats};
