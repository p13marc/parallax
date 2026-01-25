//! Plugin system for dynamically loading elements.
//!
//! This module provides a plugin architecture that allows elements to be
//! loaded at runtime from shared libraries. The plugin ABI is kept minimal
//! and C-compatible for maximum interoperability.
//!
//! # Plugin Structure
//!
//! A plugin is a shared library (.so on Linux) that exports a single symbol:
//!
//! ```c
//! const PluginDescriptor* parallax_plugin_descriptor();
//! ```
//!
//! The descriptor contains metadata about the plugin and a function to
//! create element instances.
//!
//! # Example Plugin (Rust)
//!
//! ```ignore
//! use parallax::plugin::{PluginDescriptor, ElementDescriptor};
//! use parallax::element::Element;
//!
//! struct MyFilter;
//!
//! impl Element for MyFilter {
//!     fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
//!         Ok(Some(buffer))
//!     }
//! }
//!
//! #[no_mangle]
//! pub extern "C" fn parallax_plugin_descriptor() -> *const PluginDescriptor {
//!     // Return static descriptor
//! }
//! ```

mod descriptor;
mod loader;
mod registry;

pub use descriptor::{
    ElementDescriptor, ElementInfo, PARALLAX_ABI_VERSION, PluginDescriptor, PluginInfo,
    element_from_raw, element_to_raw,
};
pub use loader::{Plugin, PluginError, PluginLoader};
pub use registry::PluginRegistry;
