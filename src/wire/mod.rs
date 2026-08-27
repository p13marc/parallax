//! Wire types shared across transports.
//!
//! Always compiled, deliberately: this module sits at the intersection of a
//! feature-gated transport (`zenoh`) and a default-compiled one (the IPC
//! descriptor in [`memory::ipc_ring`](crate::memory)), and a downstream crate
//! must be able to name a wire type without enabling either.
//!
//! # Two planes, two schemas — on purpose
//!
//! - The **parallax plane** (`elements::network::zenoh_wire`, feature-gated on
//!   `zenoh`, hence not linked) carries an rkyv `WireMetadata`: everything a
//!   `Metadata` holds, for parallax↔parallax links.
//! - The **`@media` plane** carries a CBOR [`FrameMeta`]: seven fields, for
//!   consumers that are not parallax — the zensight GUI, a browser.
//!
//! They are not unified and must not be. `FrameMeta` deliberately carries
//! strictly less (no `RtpMeta`, no `PlaneLayout`, no custom entries) because
//! it is a consumer-facing contract owned by the `@media` specification, not
//! by parallax. Widening it to carry parallax's internals would break that
//! contract; narrowing `WireMetadata` to match would lose metadata on
//! parallax's own links.

pub mod codes;
pub mod descriptor;
mod frame_meta;
mod metadata;

pub use frame_meta::FrameMeta;
pub use metadata::{CUSTOM_META, WireFormat, WireMetadata, WirePlanes, WireRtp, WireTerminal};

/// Attachment magic bytes (`"PX"`).
pub const WIRE_MAGIC: [u8; 2] = [0x50, 0x58];

/// The internal schema's version.
///
/// 5: `color` (#265) — colorimetry, so the same file stops rendering three
/// different ways. 4: `rtp`, `planes` and `WireFormat::Rtp` (#219) — the
/// fields the two wires had already diverged on. 3: `terminal` (#224).
/// 2: `coded` (#236).
///
/// **The two versions in this tree change independently and must not be
/// merged.** This one is *semantics*: what a field means, which fields exist.
/// [`IPC_RING_VERSION`](crate::memory::IPC_RING_VERSION) is *mechanics*: the
/// ring's layout and handshake. A doc saying "remember to bump both" is a doc
/// that will be violated.
pub const WIRE_VERSION: u8 = 5;
