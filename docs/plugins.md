# Plugin Development

This guide explains how to create plugins for Parallax.

## Overview

Parallax plugins are dynamic libraries (`.so` files on Linux) that provide additional elements. Plugins use a C-compatible ABI for stability across Rust versions.

## Creating a Plugin

### 1. Create a New Crate

```bash
cargo new --lib my-plugin
cd my-plugin
```

### 2. Configure Cargo.toml

```toml
[package]
name = "my-plugin"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]  # Important: creates a .so file

[dependencies]
parallax = "0.1"
```

### 3. Implement Elements

```rust
// src/lib.rs
use parallax::buffer::Buffer;
use parallax::element::Element;
use parallax::error::Result;

/// A simple element that doubles values
pub struct Doubler;

impl Default for Doubler {
    fn default() -> Self {
        Self
    }
}

impl Element for Doubler {
    fn process(&mut self, buffer: Buffer<()>) -> Result<Option<Buffer<()>>> {
        // Process the buffer...
        Ok(Some(buffer))
    }
}

/// A counter element
pub struct Counter {
    count: u64,
}

impl Default for Counter {
    fn default() -> Self {
        Self { count: 0 }
    }
}

impl Element for Counter {
    fn process(&mut self, buffer: Buffer<()>) -> Result<Option<Buffer<()>>> {
        self.count += 1;
        Ok(Some(buffer))
    }
}
```

### 4. Define the Plugin Descriptor

```rust
use parallax::element::ElementAdapter;

// Element type constants
const TRANSFORM: i32 = 1;

// Use the define_plugin! macro
parallax::define_plugin! {
    name: "my-plugin",
    version: "0.1.0",
    author: "Your Name",
    description: "My custom Parallax plugin",
    elements: [
        {
            name: "doubler",
            description: "Doubles values in buffers",
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
```

### 5. Build the Plugin

```bash
cargo build --release
# Output: target/release/libmy_plugin.so
```

## Plugin Structure

### Plugin Descriptor

```rust
#[repr(C)]
pub struct PluginDescriptor {
    /// ABI version (must match PARALLAX_ABI_VERSION)
    pub abi_version: u32,
    
    /// Plugin name (null-terminated C string)
    pub name: *const c_char,
    
    /// Plugin version (null-terminated C string)
    pub version: *const c_char,
    
    /// Plugin author (null-terminated C string)
    pub author: *const c_char,
    
    /// Plugin description (null-terminated C string)
    pub description: *const c_char,
    
    /// Array of element descriptors
    pub elements: *const ElementDescriptor,
    
    /// Number of elements
    pub num_elements: usize,
}
```

### Element Descriptor

```rust
#[repr(C)]
pub struct ElementDescriptor {
    /// Element name (null-terminated C string)
    pub name: *const c_char,
    
    /// Element description (null-terminated C string)
    pub description: *const c_char,
    
    /// Element type (0=Source, 1=Transform, 2=Sink)
    pub element_type: i32,
    
    /// Factory function to create instances
    pub create: Option<extern "C" fn() -> *mut c_void>,
    
    /// Destructor function
    pub destroy: Option<extern "C" fn(*mut c_void)>,
}
```

### Entry Point

Every plugin must export a function named `parallax_plugin_descriptor`:

```rust
#[no_mangle]
pub extern "C" fn parallax_plugin_descriptor() -> *const PluginDescriptor {
    &PLUGIN_DESCRIPTOR
}
```

The `define_plugin!` macro generates this automatically.

## Loading Plugins

### Using PluginLoader

```rust
use parallax::plugin::{PluginLoader, PluginRegistry};

// Create a registry
let mut registry = PluginRegistry::new();

// Create a loader
let loader = PluginLoader::new();

// Load a plugin
loader.load(&mut registry, "path/to/libmy_plugin.so")?;

// Use the registered elements
if let Some(factory) = registry.get("doubler") {
    let element = factory.create();
    // Use element...
}
```

### Using PluginRegistry

```rust
use parallax::plugin::PluginRegistry;

// Create and configure registry
let mut registry = PluginRegistry::new();
registry.add_search_path("/usr/lib/parallax/plugins");
registry.add_search_path("./plugins");

// Scan and load all plugins
registry.scan_and_load()?;

// List available elements
for name in registry.list_elements() {
    println!("Available: {}", name);
}
```

## Element Types

### Source Elements

```rust
use parallax::element::{Source, SourceAdapter};

pub struct MySource {
    // ...
}

impl Source for MySource {
    fn produce(&mut self) -> Result<Option<Buffer<()>>> {
        // Produce buffers...
    }
}

// In plugin descriptor:
create: || Box::new(SourceAdapter::new(MySource::new())),
```

### Sink Elements

```rust
use parallax::element::{Sink, SinkAdapter};

pub struct MySink {
    // ...
}

impl Sink for MySink {
    fn consume(&mut self, buffer: Buffer<()>) -> Result<()> {
        // Consume buffer...
    }
}

// In plugin descriptor:
create: || Box::new(SinkAdapter::new(MySink::new())),
```

### Transform Elements

```rust
use parallax::element::{Element, ElementAdapter};

pub struct MyTransform {
    // ...
}

impl Element for MyTransform {
    fn process(&mut self, buffer: Buffer<()>) -> Result<Option<Buffer<()>>> {
        // Transform buffer...
    }
}

// In plugin descriptor:
create: || Box::new(ElementAdapter::new(MyTransform::new())),
```

## Best Practices

### ABI Stability

- Always use `#[repr(C)]` for types crossing the ABI boundary
- Check `abi_version` before using plugin data
- Don't pass Rust-specific types (Vec, String) across the boundary

### Memory Safety

```rust
// Good: Use Box for heap allocation
create: || Box::new(ElementAdapter::new(MyElement::default())),

// The plugin system handles memory correctly:
// - create() returns a Box that's converted to raw pointer
// - destroy() converts back to Box and drops it
```

### Error Handling

```rust
impl Element for MyElement {
    fn process(&mut self, buffer: Buffer<()>) -> Result<Option<Buffer<()>>> {
        // Use ? for error propagation
        let data = self.parse_data(&buffer)?;
        
        // Return errors for unrecoverable situations
        if data.is_invalid() {
            return Err(Error::Element("invalid data".into()));
        }
        
        Ok(Some(buffer))
    }
}
```

### Testing Plugins

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_element_creation() {
        let _element = Doubler::default();
    }
    
    #[test]
    fn test_plugin_descriptor() {
        let desc = parallax_plugin_descriptor();
        assert!(!desc.is_null());
        
        unsafe {
            let desc = &*desc;
            assert_eq!(desc.abi_version, parallax::plugin::PARALLAX_ABI_VERSION);
        }
    }
}
```

## Example: Complete Plugin

See `examples/example-plugin/` in the Parallax repository for a complete working example.

```bash
# Build the example plugin
cd examples/example-plugin
cargo build --release

# The plugin is at:
# target/release/libexample_plugin.so
```

## Troubleshooting

### Plugin Won't Load

1. Check ABI version matches
2. Ensure crate-type is "cdylib"
3. Check for missing dependencies

### Symbol Not Found

```bash
# Check exported symbols
nm -D target/release/libmy_plugin.so | grep parallax
# Should show: parallax_plugin_descriptor
```

### Segmentation Fault

- Ensure all pointers are valid
- Check that strings are null-terminated
- Verify Box/raw pointer conversions are correct
