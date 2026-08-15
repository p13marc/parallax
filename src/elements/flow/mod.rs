//! Flow control and routing elements.
//!
//! ## Buffering
//! - [`Queue2`]: Network buffering (stream/download/timeshift). There is no
//!   plain `Queue` element: every element runs in its own task behind a
//!   bounded channel, so the link itself is the queue — capacity via
//!   `link_pads_full`, loss policy via `LinkPolicy`, occupancy-driven flow
//!   signals via `Pipeline::monitor_link`.
//!
//! ## Inspection
//! - [`Inspect`]: 1-in/1-out passthrough counter (formerly, and misleadingly,
//!   called `Tee` — fan-out needs no element: link one src-pad to several sinks)
//!
//! ## Routing
//! - [`Funnel`]: N-to-1 merge
//! - [`InputSelector`]: N-to-1 switching (selects one input)
//! - [`OutputSelector`]: 1-to-N routing (routes to one output)
//! - [`Concat`]: Sequential stream concatenation
//!
//! ## Control
//! - [`Valve`]: On/off flow control

mod concat;
mod funnel;
mod inspect;
mod queue2;
mod selector;
mod valve;

pub use concat::{Concat, ConcatStats, ConcatStream};
pub use funnel::{Funnel, FunnelInput, FunnelStats};
pub use inspect::Inspect;
pub use queue2::{BufferingConfig, DownloadedRanges, Queue2, Queue2RangesHandle, Queue2Stats};
pub use selector::{
    InputSelector, InputSelectorStats, OutputSelector, OutputSelectorStats, SelectorInput,
    SelectorOutput,
};
pub use valve::{Valve, ValveControl, ValveStats};
