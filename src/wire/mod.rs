//! Wire types shared across transports.
//!
//! Always compiled, deliberately: this module sits at the intersection of a
//! feature-gated transport (`zenoh`) and a default-compiled one (the IPC
//! descriptor in [`memory::ipc_ring`](crate::memory)), and a downstream crate
//! must be able to name a wire type without enabling either.
//!
//! # Two planes, two schemas — on purpose
//!
//! - The **parallax plane** ([`elements::network::zenoh_wire`]) carries an
//!   rkyv `WireMetadata`: everything a `Metadata` holds, for parallax↔parallax
//!   links.
//! - The **`@media` plane** carries a CBOR [`FrameMeta`]: seven fields, for
//!   consumers that are not parallax — the zensight GUI, a browser.
//!
//! They are not unified and must not be. `FrameMeta` deliberately carries
//! strictly less (no `RtpMeta`, no `PlaneLayout`, no custom entries) because
//! it is a consumer-facing contract owned by the `@media` specification, not
//! by parallax. Widening it to carry parallax's internals would break that
//! contract; narrowing `WireMetadata` to match would lose metadata on
//! parallax's own links.
//!
//! [`elements::network::zenoh_wire`]: crate::elements::network::zenoh_wire

mod frame_meta;

pub use frame_meta::FrameMeta;
