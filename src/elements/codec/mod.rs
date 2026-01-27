//! Video codec elements.
//!
//! This module provides video encoding and decoding elements:
//!
//! - [`Dav1dDecoder`] - AV1 software decoder using dav1d (requires `av1-decode` feature)
//! - [`Rav1eEncoder`] - AV1 software encoder using rav1e (requires `av1-encode` feature)
//!
//! # Feature Flags
//!
//! These elements can be enabled individually or together:
//!
//! ```toml
//! # Encoder only
//! parallax = { version = "0.1", features = ["av1-encode"] }
//!
//! # Decoder only
//! parallax = { version = "0.1", features = ["av1-decode"] }
//!
//! # Both encoder and decoder
//! parallax = { version = "0.1", features = ["software-codecs"] }
//! ```
//!
//! ## Build Dependencies
//!
//! ### av1-encode (rav1e)
//!
//! Requires **nasm** assembler for x86_64 optimizations:
//!
//! - **Fedora/RHEL**: `sudo dnf install nasm`
//! - **Debian/Ubuntu**: `sudo apt install nasm`
//! - **Arch**: `sudo pacman -S nasm`
//! - **macOS**: `brew install nasm`
//!
//! ### av1-decode (dav1d)
//!
//! Requires the **libdav1d** library:
//!
//! - **Fedora/RHEL**: `sudo dnf install libdav1d-devel`
//! - **Debian/Ubuntu**: `sudo apt install libdav1d-dev`
//! - **Arch**: `sudo pacman -S dav1d`
//! - **macOS**: `brew install dav1d`

mod common;

#[cfg(feature = "av1-decode")]
mod decoder;

#[cfg(feature = "av1-encode")]
mod encoder;

// Re-export common types
pub use common::{PixelFormat, VideoFrame};

// Re-export decoder
#[cfg(feature = "av1-decode")]
pub use decoder::Dav1dDecoder;

// Re-export encoder
#[cfg(feature = "av1-encode")]
pub use encoder::{Rav1eConfig, Rav1eEncoder};
