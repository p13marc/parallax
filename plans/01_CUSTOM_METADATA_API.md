# Plan 01: Custom Metadata API

**Priority:** High (Immediate)  
**Effort:** Small (1-2 days)  
**Dependencies:** None  
**Addresses:** Pain Point 1.2 (Metadata Attachment to Buffers)

---

## Problem Statement

Currently, the `Metadata` struct has fixed fields (`sequence`, `timestamp`, `duration`, `flags`). There's no way to attach arbitrary domain-specific data like:
- KLV packets (STANAG 4609)
- SEI NALUs (H.264/HEVC)
- Closed captions (CEA-608/708)
- Custom application data

Developers must pass this data out-of-band, losing the association with specific buffers.

---

## Proposed Solution

Add an extensible metadata system using type-erased storage with type-safe accessors.

### Design Approach

**Option A: HashMap with TypeId (Simple)**
```rust
pub struct Metadata {
    // Existing fields
    pub sequence: u64,
    pub timestamp: ClockTime,
    pub duration: Option<ClockTime>,
    pub flags: BufferFlags,
    
    // New: extensible custom data
    custom: HashMap<&'static str, Box<dyn Any + Send + Sync>>,
}
```

**Option B: SmallVec with Tagged Union (Performance)**
```rust
pub struct Metadata {
    // ... existing fields ...
    
    // Inline storage for common cases, heap for overflow
    custom: SmallVec<[MetaEntry; 2]>,
}

struct MetaEntry {
    key: MetaKey,
    value: MetaValue,
}

enum MetaKey {
    Klv,
    Sei,
    ClosedCaption,
    Custom(&'static str),
}
```

**Recommendation:** Start with Option A for simplicity, optimize to Option B if profiling shows overhead.

---

## API Design

### Setting Custom Metadata

```rust
impl Metadata {
    /// Attach custom data with a string key.
    /// The key should use namespacing: "domain/type" (e.g., "stanag/klv", "h264/sei")
    pub fn set<T: Any + Send + Sync + 'static>(&mut self, key: &'static str, value: T) {
        self.custom.insert(key, Box::new(value));
    }
    
    /// Attach bytes directly (common case, avoids Box overhead for small data)
    pub fn set_bytes(&mut self, key: &'static str, data: Vec<u8>) {
        self.set(key, data);
    }
}
```

### Getting Custom Metadata

```rust
impl Metadata {
    /// Get custom data by key, returns None if not present or wrong type.
    pub fn get<T: Any + Send + Sync + 'static>(&self, key: &'static str) -> Option<&T> {
        self.custom.get(key)?.downcast_ref()
    }
    
    /// Get bytes directly (common case)
    pub fn get_bytes(&self, key: &'static str) -> Option<&[u8]> {
        self.get::<Vec<u8>>(key).map(|v| v.as_slice())
    }
    
    /// Check if a key exists
    pub fn has(&self, key: &'static str) -> bool {
        self.custom.contains_key(key)
    }
    
    /// Remove and return custom data
    pub fn remove<T: Any + Send + Sync + 'static>(&mut self, key: &'static str) -> Option<T> {
        self.custom.remove(key)?.downcast().ok().map(|b| *b)
    }
}
```

### Convenience Methods for Common Types

```rust
impl Metadata {
    /// Attach KLV data (STANAG 4609)
    pub fn set_klv(&mut self, data: Vec<u8>) {
        self.set_bytes("stanag/klv", data);
    }
    
    pub fn klv(&self) -> Option<&[u8]> {
        self.get_bytes("stanag/klv")
    }
    
    /// Attach SEI NALUs
    pub fn set_sei(&mut self, nalus: Vec<Vec<u8>>) {
        self.set("h264/sei", nalus);
    }
    
    pub fn sei(&self) -> Option<&Vec<Vec<u8>>> {
        self.get("h264/sei")
    }
}
```

---

## Implementation Steps

### Step 1: Modify Metadata Struct

**File:** `src/metadata.rs`

```rust
use std::any::Any;
use std::collections::HashMap;

pub struct Metadata {
    pub sequence: u64,
    pub timestamp: ClockTime,
    pub duration: Option<ClockTime>,
    pub flags: BufferFlags,
    
    // Extensible custom metadata
    custom: HashMap<&'static str, Box<dyn Any + Send + Sync>>,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            sequence: 0,
            timestamp: ClockTime::ZERO,
            duration: None,
            flags: BufferFlags::empty(),
            custom: HashMap::new(),
        }
    }
}
```

### Step 2: Add Accessor Methods

Add the `set`, `get`, `has`, `remove` methods as shown above.

### Step 3: Add Clone Support

Custom metadata needs careful Clone handling:

```rust
impl Clone for Metadata {
    fn clone(&self) -> Self {
        // Note: custom data is NOT cloned by default (expensive)
        // Use clone_with_custom() if needed
        Self {
            sequence: self.sequence,
            timestamp: self.timestamp,
            duration: self.duration,
            flags: self.flags,
            custom: HashMap::new(),  // Empty by default
        }
    }
}

impl Metadata {
    /// Clone including custom metadata (requires T: Clone)
    pub fn clone_deep(&self) -> Self {
        // Would need runtime clone support, consider later
        self.clone()
    }
}
```

### Step 4: Update Buffer API

**File:** `src/buffer.rs`

```rust
impl Buffer {
    /// Get mutable access to metadata for setting custom data
    pub fn metadata_mut(&mut self) -> &mut Metadata {
        &mut self.metadata
    }
}
```

### Step 5: Add Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_custom_metadata_bytes() {
        let mut meta = Metadata::default();
        meta.set_bytes("test/data", vec![1, 2, 3, 4]);
        
        assert!(meta.has("test/data"));
        assert_eq!(meta.get_bytes("test/data"), Some(&[1, 2, 3, 4][..]));
        assert_eq!(meta.get_bytes("nonexistent"), None);
    }
    
    #[test]
    fn test_custom_metadata_typed() {
        let mut meta = Metadata::default();
        
        #[derive(Debug, PartialEq)]
        struct GpsPosition { lat: f64, lon: f64 }
        
        meta.set("gps/position", GpsPosition { lat: 37.0, lon: -122.0 });
        
        let pos = meta.get::<GpsPosition>("gps/position").unwrap();
        assert_eq!(pos.lat, 37.0);
    }
    
    #[test]
    fn test_klv_convenience() {
        let mut meta = Metadata::default();
        meta.set_klv(vec![0x06, 0x0E, 0x2B, 0x34]);
        
        assert_eq!(meta.klv(), Some(&[0x06, 0x0E, 0x2B, 0x34][..]));
    }
}
```

### Step 6: Update KLV Example

**File:** `examples/31_av1_pipeline_stanag.rs`

```rust
// Before: KLV logged separately
if let Some(ref sensor_meta) = latest_metadata {
    let klv_data = sensor_meta.to_klv();
    println!("klv={} bytes", klv_data.len());
}

// After: KLV attached to buffer
if let Some(ref sensor_meta) = latest_metadata {
    metadata.set_klv(sensor_meta.to_klv());
}
let buffer = Buffer::new(handle, metadata);
```

---

## Key Conventions

### Namespace Keys

Use hierarchical namespacing to avoid collisions:

| Domain | Key Pattern | Examples |
|--------|-------------|----------|
| STANAG/MISB | `stanag/*` | `stanag/klv`, `stanag/vmti` |
| H.264 | `h264/*` | `h264/sei`, `h264/sps`, `h264/pps` |
| H.265 | `h265/*` | `h265/sei`, `h265/vps` |
| AV1 | `av1/*` | `av1/metadata_obu` |
| Captions | `caption/*` | `caption/cea608`, `caption/cea708` |
| Audio | `audio/*` | `audio/loudness`, `audio/language` |
| App | `app/*` | `app/custom`, `app/mydata` |

### Performance Considerations

1. **Avoid allocation in hot paths:** Reuse metadata objects when possible
2. **Small data inline:** For data < 64 bytes, consider `SmallVec` storage
3. **Lazy initialization:** `custom` HashMap created only when first used

---

## Validation Criteria

- [ ] `Metadata::set` and `Metadata::get` work with arbitrary types
- [ ] `Metadata::set_bytes` and `Metadata::get_bytes` work for `Vec<u8>`
- [ ] Convenience methods `set_klv`/`klv` work correctly
- [ ] Example 31 updated to attach KLV to buffers
- [ ] All existing tests pass
- [ ] New tests for custom metadata pass

---

## Future Enhancements

1. **Typed metadata registry:** Pre-register known metadata types for better performance
2. **Metadata forwarding rules:** Which metadata should pass through transforms?
3. **Serialization support:** rkyv/serde for IPC metadata transfer
4. **Metadata inspection:** Debug/introspection tools for pipeline debugging

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/metadata.rs` | Add `custom` field, accessor methods |
| `src/buffer.rs` | Add `metadata_mut()` method |
| `examples/31_av1_pipeline_stanag.rs` | Use new API |
| `tests/metadata_test.rs` | New test file |
