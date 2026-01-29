# Plan: VideoConvert SIMD Upgrade with Memory Alignment Negotiation

## Goal

Replace the current pure-Rust scalar implementation of `VideoConvert` with a high-performance SIMD-accelerated library, and extend the caps negotiation system to include memory alignment requirements for optimal SIMD performance.

## Research Summary

### Rust Color Conversion Libraries

| Library | SIMD Support | Threading | Formats | Notes |
|---------|--------------|-----------|---------|-------|
| [yuvutils-rs](https://github.com/awxkee/yuvutils-rs) | AVX-512, AVX2, SSE4.1, NEON, WASM | Rayon (optional) | I420, NV12, YUYV, UYVY, P010, P012, I010, RGB/RGBA/BGR/BGRA | **Best choice** - matches/exceeds libyuv performance |
| [ezk-image](https://lib.rs/crates/ezk-image) | Yes (internal) | Yes (`convert_multi_thread`) | I420, NV12, YUYV, RGB variants, HDR | Good API, but less format coverage |
| [colorutils-rs](https://github.com/awxkee/colorutils-rs) | Yes | Unknown | RGB color space conversions | Focused on color math, not video |

**Recommendation: `yuvutils-rs`** (crate name: `yuv`)

- Performance equal to or better than libyuv
- Runtime CPU feature detection (no compile flags needed for SSE/AVX2)
- Extensive format support including HDR (10-bit, 12-bit)
- Pure Rust with optional SIMD assembly
- Actively maintained (same author as colorutils-rs)

### Memory Alignment Requirements

From research on [SSE/AVX memory alignment](https://blog.ngzhian.com/sse-avx-memory-alignment.html):

| SIMD Level | Register Size | Optimal Alignment |
|------------|---------------|-------------------|
| SSE | 128-bit | 16 bytes |
| AVX/AVX2 | 256-bit | 32 bytes |
| AVX-512 | 512-bit | 64 bytes |

**Key insight**: Modern AVX can handle unaligned data, but with performance penalties when crossing cache line boundaries (64 bytes). Aligned data is always faster, especially for stores.

### GStreamer's Approach

GStreamer handles this via:

1. **[GstVideoAlignment](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html)**: Structure specifying padding and stride alignment for video buffers
2. **ALLOCATION query**: Elements negotiate alignment requirements during caps negotiation
3. **GstBufferPool**: Pre-allocates buffers with correct alignment
4. **[ORC (Optimized Inner Loop Runtime Compiler)](https://gstreamer.freedesktop.org/documentation/additional/design/orc-integration.html)**: JIT compiler for SIMD code generation

For Parallax, we don't need the complexity of ORC since `yuvutils-rs` already provides optimized SIMD implementations. We just need to negotiate alignment.

---

## Implementation Plan

### Phase 1: Add Memory Layout to Caps System

**Effort: 1-2 days**

#### 1.1 Define MemoryLayout in `src/format.rs`

```rust
/// Memory layout requirements for video buffers.
///
/// SIMD operations require specific memory alignment for optimal performance:
/// - SSE: 16-byte alignment
/// - AVX/AVX2: 32-byte alignment  
/// - AVX-512: 64-byte alignment
///
/// Stride padding ensures each row starts at an aligned address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MemoryLayout {
    /// Required alignment for buffer start address (in bytes).
    /// Common values: 16 (SSE), 32 (AVX), 64 (AVX-512/cache line).
    pub alignment: u32,
    
    /// Stride alignment for each row (in bytes).
    /// If non-zero, stride = align_up(width * bytes_per_pixel, stride_alignment).
    /// Common values: 16, 32, 64.
    pub stride_alignment: u32,
    
    /// Extra padding at end of each plane (in bytes).
    /// Useful for SIMD that reads past the end of a row.
    pub plane_padding: u32,
}

impl MemoryLayout {
    /// No alignment requirements (any alignment acceptable).
    pub const NONE: Self = Self {
        alignment: 1,
        stride_alignment: 1,
        plane_padding: 0,
    };
    
    /// SSE-optimized: 16-byte alignment.
    pub const SSE: Self = Self {
        alignment: 16,
        stride_alignment: 16,
        plane_padding: 0,
    };
    
    /// AVX-optimized: 32-byte alignment.
    pub const AVX: Self = Self {
        alignment: 32,
        stride_alignment: 32,
        plane_padding: 0,
    };
    
    /// AVX-512 / cache-line optimized: 64-byte alignment.
    pub const AVX512: Self = Self {
        alignment: 64,
        stride_alignment: 64,
        plane_padding: 0,
    };
    
    /// Calculate padded stride for given width and bytes per pixel.
    pub fn padded_stride(&self, width: u32, bytes_per_pixel: u32) -> u32 {
        let raw_stride = width * bytes_per_pixel;
        if self.stride_alignment <= 1 {
            raw_stride
        } else {
            (raw_stride + self.stride_alignment - 1) / self.stride_alignment * self.stride_alignment
        }
    }
    
    /// Calculate total buffer size with alignment padding.
    pub fn buffer_size(&self, width: u32, height: u32, bytes_per_pixel: u32) -> usize {
        let stride = self.padded_stride(width, bytes_per_pixel);
        (stride * height) as usize + self.plane_padding as usize
    }
    
    /// Merge with another layout, taking the stricter requirements.
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            alignment: self.alignment.max(other.alignment),
            stride_alignment: self.stride_alignment.max(other.stride_alignment),
            plane_padding: self.plane_padding.max(other.plane_padding),
        }
    }
    
    /// Check if a pointer is properly aligned.
    pub fn is_aligned(&self, ptr: *const u8) -> bool {
        (ptr as usize) % (self.alignment as usize) == 0
    }
}
```

#### 1.2 Extend VideoFormatCaps with Layout

```rust
/// Video format with constraints for negotiation.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoFormatCaps {
    /// Width constraint.
    pub width: CapsValue<u32>,
    /// Height constraint.
    pub height: CapsValue<u32>,
    /// Pixel format constraint.
    pub pixel_format: CapsValue<PixelFormat>,
    /// Framerate constraint.
    pub framerate: CapsValue<Framerate>,
    /// Memory layout requirements (NEW).
    pub layout: MemoryLayout,
}
```

#### 1.3 Update FormatMemoryCap

```rust
/// A single format+memory capability.
#[derive(Clone, Debug, PartialEq)]
pub struct FormatMemoryCap {
    /// Format constraints.
    pub format: FormatCaps,
    /// Memory type constraints.
    pub memory: MemoryCaps,
    /// Memory layout requirements (NEW).
    pub layout: MemoryLayout,
}

impl FormatMemoryCap {
    /// Try to intersect this capability with another.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        Some(Self {
            format: self.format.intersect(&other.format)?,
            memory: self.memory.intersect(&other.memory)?,
            // Merge layouts (take stricter requirements)
            layout: self.layout.merge(&other.layout),
        })
    }
}
```

### Phase 2: Integrate yuvutils-rs

**Effort: 1 day**

#### 2.1 Add Dependency

```toml
# Cargo.toml
[dependencies]
yuv = { version = "0.8", optional = true }

[features]
simd-convert = ["yuv"]
```

#### 2.2 Create SIMD Converter Backend

Create `src/converters/colorspace_simd.rs`:

```rust
//! SIMD-accelerated color conversion using yuvutils-rs.

use yuv::{
    YuvPlanarImageMut, YuvPackedImageMut, YuvRange, YuvStandardMatrix,
    YuvChromaSubsampling, rgb_to_yuv420, yuv420_to_rgba, yuyv_to_rgba,
    // ... etc
};

use crate::error::Result;

/// SIMD-accelerated video converter.
pub struct SimdVideoConvert {
    input_format: PixelFormat,
    output_format: PixelFormat,
    width: u32,
    height: u32,
    color_matrix: YuvStandardMatrix,
    yuv_range: YuvRange,
}

impl SimdVideoConvert {
    pub fn new(
        input_format: PixelFormat,
        output_format: PixelFormat,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        // Validate dimensions for YUV formats
        // ...
        
        Ok(Self {
            input_format,
            output_format,
            width,
            height,
            color_matrix: YuvStandardMatrix::Bt601, // or detect from caps
            yuv_range: YuvRange::Limited,
        })
    }
    
    /// Convert with SIMD acceleration.
    /// 
    /// For optimal performance, buffers should be aligned to 32 bytes (AVX2)
    /// or 64 bytes (AVX-512).
    pub fn convert(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        match (self.input_format, self.output_format) {
            (PixelFormat::I420, PixelFormat::Rgba) => {
                self.i420_to_rgba_simd(input, output)
            }
            (PixelFormat::Nv12, PixelFormat::Rgba) => {
                self.nv12_to_rgba_simd(input, output)
            }
            (PixelFormat::Yuyv, PixelFormat::Rgba) => {
                self.yuyv_to_rgba_simd(input, output)
            }
            (PixelFormat::Rgba, PixelFormat::I420) => {
                self.rgba_to_i420_simd(input, output)
            }
            // ... more conversions
            _ => {
                // Fallback to scalar implementation
                self.convert_scalar(input, output)
            }
        }
    }
    
    fn i420_to_rgba_simd(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let w = self.width as usize;
        let h = self.height as usize;
        
        let y_plane = &input[0..w * h];
        let u_plane = &input[w * h..w * h + (w / 2) * (h / 2)];
        let v_plane = &input[w * h + (w / 2) * (h / 2)..];
        
        yuv::yuv420_to_rgba(
            y_plane,
            w,  // y_stride
            u_plane,
            w / 2,  // u_stride
            v_plane,
            w / 2,  // v_stride
            output,
            w * 4,  // rgba_stride
            self.width,
            self.height,
            self.yuv_range,
            self.color_matrix,
        ).map_err(|e| Error::Conversion(e.to_string()))
    }
    
    // ... other conversion methods
}
```

#### 2.3 Update VideoConvert to Use SIMD Backend

```rust
// src/converters/colorspace.rs

#[cfg(feature = "simd-convert")]
use super::colorspace_simd::SimdVideoConvert;

pub struct VideoConvert {
    #[cfg(feature = "simd-convert")]
    simd: Option<SimdVideoConvert>,
    
    // Keep scalar fallback
    input_format: PixelFormat,
    output_format: PixelFormat,
    width: u32,
    height: u32,
    color_matrix: ColorMatrix,
}

impl VideoConvert {
    pub fn convert(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        #[cfg(feature = "simd-convert")]
        if let Some(ref simd) = self.simd {
            return simd.convert(input, output);
        }
        
        // Scalar fallback
        self.convert_scalar(input, output)
    }
}
```

### Phase 3: Aligned Buffer Allocation

**Effort: 0.5 days**

#### 3.1 Update SharedArena for Aligned Allocation

The current `SharedArena` uses `memfd_create` + `mmap`. We need to ensure alignment:

```rust
// src/memory/shared_refcount.rs

impl SharedArena {
    /// Create arena with specific alignment.
    pub fn with_alignment(
        slot_size: usize,
        slot_count: usize,
        alignment: usize,
    ) -> Result<Self> {
        // Round up slot_size to alignment
        let aligned_slot_size = (slot_size + alignment - 1) / alignment * alignment;
        
        // Ensure mmap returns aligned memory (it typically does for page-aligned sizes)
        // For sub-page alignment, we may need to over-allocate and adjust
        
        Self::new(aligned_slot_size, slot_count)
    }
}
```

#### 3.2 Add AlignedBuffer Type

```rust
// src/memory/aligned.rs

/// A buffer with guaranteed memory alignment.
pub struct AlignedBuffer {
    data: Vec<u8>,
    alignment: usize,
}

impl AlignedBuffer {
    /// Allocate an aligned buffer.
    pub fn new(size: usize, alignment: usize) -> Self {
        // Use Vec with custom layout or posix_memalign
        let layout = std::alloc::Layout::from_size_align(size, alignment)
            .expect("invalid alignment");
        
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let data = unsafe { Vec::from_raw_parts(ptr, size, size) };
        
        Self { data, alignment }
    }
    
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
    
    pub fn is_aligned(&self, required: usize) -> bool {
        (self.data.as_ptr() as usize) % required == 0
    }
}
```

### Phase 4: VideoConvertElement Updates

**Effort: 0.5 days**

#### 4.1 Declare Layout Requirements in Caps

```rust
// src/elements/transform/videoconvert.rs

impl Element for VideoConvertElement {
    fn input_media_caps(&self) -> ElementMediaCaps {
        // Accept any raw video format with AVX-friendly alignment
        let format = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Any,
            framerate: CapsValue::Any,
            layout: MemoryLayout::AVX,  // Request 32-byte alignment
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(format),
            MemoryCaps::cpu_only(),
        )])
    }
    
    fn output_media_caps(&self) -> ElementMediaCaps {
        let format = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Fixed(convert_pixel_format(self.output_format)),
            framerate: CapsValue::Any,
            layout: MemoryLayout::AVX,  // Produce aligned output
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(format),
            MemoryCaps::cpu_only(),
        )])
    }
}
```

#### 4.2 Use Aligned Arena for Output

```rust
impl Element for VideoConvertElement {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Ensure output arena has correct alignment
        let layout = MemoryLayout::AVX;
        let output_size = layout.buffer_size(
            self.width,
            self.height,
            self.output_format.bytes_per_pixel().unwrap_or(4) as u32,
        );

        if self.arena.is_none() || self.arena.as_ref().unwrap().slot_size() < output_size {
            self.arena = Some(SharedArena::with_alignment(
                output_size,
                32,
                layout.alignment as usize,
            )?);
        }
        
        // ... rest of conversion
    }
}
```

### Phase 5: Multi-threading Support (Optional)

**Effort: 0.5-1 day**

For very large frames (4K+), multi-threaded conversion can help.

```rust
impl SimdVideoConvert {
    /// Convert using multiple threads for large frames.
    pub fn convert_parallel(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        // Only use parallelism for large frames
        let pixels = (self.width * self.height) as usize;
        if pixels < 1920 * 1080 {
            return self.convert(input, output);
        }
        
        // Split frame into horizontal strips
        // Process each strip in parallel using rayon
        // ...
    }
}
```

Note: `yuvutils-rs` mentions "Some paths have multi-threading support" but recommends single-threaded for YUV work as SIMD already saturates memory bandwidth.

---

## Summary

| Phase | Description | Effort | Status |
|-------|-------------|--------|--------|
| 1 | Add MemoryLayout to caps system | 1-2 days | ✅ Complete |
| 2 | Integrate yuvutils-rs (yuv crate) | 1 day | ✅ Complete |
| 3 | Aligned buffer allocation | 0.5 days | ✅ Complete |
| 4 | Update VideoConvertElement | 0.5 days | ✅ Complete |
| 5 | Multi-threading (optional) | 0.5-1 day | Deferred |

**Total: 3-5 days**

---

## Implementation Notes

### What Was Implemented

1. **MemoryLayout struct** in `src/format.rs`:
   - Constants: `NONE`, `SSE` (16-byte), `AVX` (32-byte), `AVX512` (64-byte)
   - Methods: `padded_stride()`, `plane_size()`, `merge()`, `is_aligned()`
   - Integrated into `VideoFormatCaps` with layout negotiation via `merge()`

2. **SIMD conversions** in `src/converters/colorspace.rs`:
   - Feature flag: `simd-colorspace`
   - Uses `yuv` crate (v0.8) for accelerated conversions
   - SIMD paths for: I420→RGB/RGBA/BGR/BGRA, YUYV→RGB/RGBA/BGR/BGRA, UYVY→RGB/RGBA
   - Automatic fallback to pure Rust when feature disabled
   - Runtime CPU detection (AVX2, AVX-512, SSE4.1, NEON)

3. **Aligned arena allocation** in `src/memory/shared_refcount.rs`:
   - New field `slot_stride` tracks aligned spacing between slots
   - `with_alignment(name, slot_size, slot_count, alignment)` for custom alignment
   - `new_avx(slot_size, slot_count)` for 32-byte aligned arenas
   - `new_avx512(slot_size, slot_count)` for 64-byte aligned arenas

4. **VideoConvertElement updates**:
   - Declares `MemoryLayout::AVX` in input/output caps
   - Uses `SharedArena::new_avx()` for aligned output buffers

---

## Verification

### Benchmarks

```bash
# Run conversion benchmarks
cargo bench --features simd-colorspace -- videoconvert

# Compare with scalar baseline
cargo bench -- videoconvert
```

### Tests

```bash
# Run all tests without SIMD
cargo nextest run

# Run all tests with SIMD enabled
cargo nextest run --features simd-colorspace

# Test aligned layout calculation
cargo nextest run test_layout_calculation
```

### Performance Targets

Based on yuv crate benchmarks for 1997×1331 images:

| Conversion | Target (AVX2) | Pure Rust Scalar |
|------------|---------------|------------------|
| I420 → RGBA | ~1ms | ~5-10ms |
| YUYV → RGBA | ~0.8ms | ~4-8ms |
| RGBA → I420 | ~1.2ms | ~6-12ms |

---

## Future Enhancements

### Short Term (Next Release)

1. **RGB to YUV SIMD paths**: Add SIMD-accelerated `rgb_to_yuv420`, `rgba_to_yuv420`, `bgra_to_yuv420`
2. **NV12 SIMD support**: Use `YuvBiPlanarImage` for NV12↔RGB SIMD conversions
3. **Store alignment in ArenaHeader**: Add `alignment` field to header for proper cross-process stride calculation
4. **Benchmarks**: Add criterion benchmarks comparing SIMD vs scalar conversion performance
5. **Documentation updates**:
   - Update `CLAUDE.md` with simd-colorspace feature documentation
   - Update `README.md` with SIMD feature flag and performance notes
   - Add `docs/simd-conversion.md` with detailed usage guide

### Medium Term

4. **10-bit HDR support**: 
   - Add `PixelFormat::P010`, `PixelFormat::I010` for HDR video
   - Integrate `yuv` crate's 10-bit/12-bit conversion functions
   - Extend `MemoryLayout` for wider pixel types

5. **Dynamic layout detection**:
   - Query CPU features at runtime (`std::is_x86_feature_detected!`)
   - Auto-select `MemoryLayout::AVX512` on supported CPUs
   - Fall back gracefully on older hardware

6. **Pipeline-level alignment propagation**:
   - Solver considers layout requirements during negotiation
   - Auto-insert conversion elements when layouts incompatible
   - Upstream sources receive negotiated layout hints

### Long Term

7. **GPU conversion path**:
   - Vulkan compute shader for GPU-resident buffers
   - DMA-BUF import/export for zero-copy GPU↔CPU
   - Automatic path selection based on buffer location

8. **Zero-copy DMA-BUF chain**:
   - When both sides support DMA-BUF, negotiate GPU-friendly tiled layouts
   - V4L2 → GPU encoder without CPU copy
   - Hardware-specific modifiers for optimal memory layout

9. **Multi-threaded conversion for 4K+**:
   - Use rayon for horizontal strip parallelism on very large frames
   - Benchmark to determine crossover point (likely 4K or higher)
   - Note: SIMD already saturates memory bandwidth for most cases

---

## References

- [yuvutils-rs GitHub](https://github.com/awxkee/yuvutils-rs)
- [GStreamer Memory Allocation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html)
- [GStreamer ORC Integration](https://gstreamer.freedesktop.org/documentation/additional/design/orc-integration.html)
- [AVX Memory Alignment](https://blog.ngzhian.com/sse-avx-memory-alignment.html)
- [Intel SSE/AVX Alignment](https://community.intel.com/t5/Software-Tuning-Performance/Why-should-data-be-aligned-to-16-bytes-for-SSE-instructions/td-p/1164004)
