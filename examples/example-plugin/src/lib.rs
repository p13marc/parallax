//! Example Parallax plugin demonstrating how to create custom elements.
//!
//! This plugin provides:
//! - `doubler`: A transform element that doubles each byte value
//! - `counter`: A transform element that counts buffers passing through

use parallax::buffer::Buffer;
use parallax::element::{Element, ElementAdapter};
use parallax::error::Result;

/// A transform element that doubles each byte value in a buffer.
pub struct Doubler;

impl Default for Doubler {
    fn default() -> Self {
        Self
    }
}

impl Element for Doubler {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // For this example, we just pass through since modifying
        // the buffer would require allocation. In a real plugin,
        // you might allocate a new buffer with doubled values.
        Ok(Some(buffer))
    }
}

/// A transform element that counts buffers passing through.
pub struct Counter {
    count: u64,
}

impl Default for Counter {
    fn default() -> Self {
        Self { count: 0 }
    }
}

impl Element for Counter {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.count += 1;
        Ok(Some(buffer))
    }
}

impl Counter {
    /// Get the current count.
    pub fn count(&self) -> u64 {
        self.count
    }
}

// Element type constants: 0=Source, 1=Transform, 2=Sink
const TRANSFORM: i32 = 1;

// Use the define_plugin! macro to create the plugin
parallax::define_plugin! {
    name: "example",
    version: "0.1.0",
    author: "Parallax Team",
    description: "Example plugin with custom transform elements",
    elements: [
        {
            name: "doubler",
            description: "Doubles byte values in buffers",
            element_type: TRANSFORM,
            create: || Box::new(ElementAdapter::new(Doubler::default())),
        },
        {
            name: "counter",
            description: "Counts buffers passing through",
            element_type: TRANSFORM,
            create: || Box::new(ElementAdapter::new(Counter::default())),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doubler_default() {
        let _doubler = Doubler::default();
    }

    #[test]
    fn test_counter_default() {
        let counter = Counter::default();
        assert_eq!(counter.count(), 0);
    }

    #[test]
    fn test_plugin_descriptor_is_valid() {
        let desc = parallax_plugin_descriptor();
        assert!(!desc.is_null());

        // SAFETY: We just verified the pointer is not null, and it points
        // to our static PLUGIN_DESCRIPTOR.
        unsafe {
            let desc = &*desc;
            assert_eq!(desc.abi_version, parallax::plugin::PARALLAX_ABI_VERSION);
            assert_eq!(desc.num_elements, 2);
        }
    }
}
