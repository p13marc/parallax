//! Metadata encoding/decoding elements.
//!
//! - [`KlvEncoder`]: KLV encoder for STANAG 4609 / MISB metadata
//! - [`StanagMetadataBuilder`]: Convenient builder for STANAG metadata

mod klv;

pub use klv::{KlvEncoder, KlvTag, StanagMetadataBuilder, Uls, decode_ber_length};
