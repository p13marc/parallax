//! Video scaling/resizing element.
//!
//! This module provides video scaling for YUV420 planar frames.
//! Supports bilinear and nearest-neighbor interpolation.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::transform::{VideoScale, ScaleMode};
//!
//! // Create scaler: 1920x1080 -> 1280x720
//! let mut scaler = VideoScale::new(1920, 1080, 1280, 720);
//!
//! // Or with explicit mode
//! let mut scaler = VideoScale::new(1920, 1080, 640, 480)
//!     .with_mode(ScaleMode::NearestNeighbor);
//!
//! // Scale a YUV420 frame
//! let scaled_yuv = scaler.scale_yuv420(&input_yuv)?;
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Element;
use crate::error::{Error, Result};
use crate::format::PixelFormat;
use crate::memory::SharedArena;
use crate::metadata::Metadata;

// ============================================================================
// Scale Mode
// ============================================================================

/// Scaling interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleMode {
    /// Bilinear interpolation (smoother, slower).
    #[default]
    Bilinear,
    /// Nearest neighbor (faster, pixelated).
    NearestNeighbor,
}

// ============================================================================
// Runtime Control
// ============================================================================

/// Cloneable handle to retarget a **running** [`VideoScale`].
///
/// Resolution is the second-biggest bandwidth lever after bitrate — cutting
/// both dimensions in half quarters the pixel count — and the only one that
/// also cuts encoder CPU. Like the codec control handles, this must be cloned
/// from the element **before** `executor.start()`: elements are moved into
/// their executor tasks at start and cannot be reached through the graph
/// afterwards.
///
/// The target is resolved against the *current* source dimensions on every
/// frame, so [`set_max_height`](Self::set_max_height) keeps working when the
/// source itself changes resolution mid-stream.
///
/// # Example
///
/// ```rust,ignore
/// let scaler = VideoScale::new(1920, 1080, 1920, 1080);
/// let scale = scaler.control();          // BEFORE start
/// pipeline.add_filter("scale", scaler);
/// let handle = executor.start(&mut pipeline)?;
///
/// scale.set_max_height(360);  // downscale to 360p, aspect preserved
/// scale.passthrough();        // back to source resolution
/// ```
#[derive(Clone, Debug)]
pub struct ScaleControl(Arc<ScaleTarget>);

/// The requested output size.
///
/// A `width` of 0 means "derive from the source aspect ratio"; a `height` of 0
/// with a 0 width means "no scaling".
#[derive(Debug, Default)]
struct ScaleTarget {
    width: AtomicU32,
    height: AtomicU32,
}

impl ScaleControl {
    /// Create a handle with no scaling requested (passthrough).
    pub fn new() -> Self {
        Self(Arc::new(ScaleTarget::default()))
    }

    /// Scale to exactly `width` x `height`.
    ///
    /// Odd dimensions are rounded down to even — YUV420 chroma is subsampled by
    /// two and cannot represent an odd width or height.
    pub fn set_target(&self, width: u32, height: u32) {
        self.0.width.store(even(width), Ordering::Release);
        self.0.height.store(even(height), Ordering::Release);
    }

    /// Fit the frame within `height`, preserving the source aspect ratio.
    ///
    /// The width is derived per frame from the *current* source, so this
    /// survives a source resolution change. A source already at or below
    /// `height` is passed through untouched — this never upscales.
    pub fn set_max_height(&self, height: u32) {
        self.0.width.store(0, Ordering::Release);
        self.0.height.store(even(height), Ordering::Release);
    }

    /// Stop scaling: emit frames at the source resolution.
    pub fn passthrough(&self) {
        self.0.width.store(0, Ordering::Release);
        self.0.height.store(0, Ordering::Release);
    }

    /// The requested target, as `(width, height)` where 0 means "unconstrained".
    pub fn target(&self) -> (u32, u32) {
        (
            self.0.width.load(Ordering::Acquire),
            self.0.height.load(Ordering::Acquire),
        )
    }

    /// Resolve the request against a concrete source size.
    ///
    /// An explicit [`set_target`](Self::set_target) is honoured as given, up or
    /// down. An aspect-preserving *bound* ([`set_max_height`](Self::set_max_height))
    /// only ever shrinks: a bound above the source means "fits already", not
    /// "invent detail".
    fn resolve(&self, src_width: u32, src_height: u32) -> (u32, u32) {
        let (want_width, want_height) = self.target();

        match (want_width, want_height) {
            (0, 0) => (src_width, src_height),

            // Bounding height: derive the width from the source aspect ratio,
            // and pass through a source that already fits.
            (0, h) if src_height > 0 => {
                if even(h) >= src_height || even(h) == 0 {
                    return (src_width, src_height);
                }
                let w = (src_width as u64 * h as u64).div_ceil(src_height as u64) as u32;
                (even(w).max(2), even(h))
            }
            // Bounding width: the mirror case.
            (w, 0) if src_width > 0 => {
                if even(w) >= src_width || even(w) == 0 {
                    return (src_width, src_height);
                }
                let h = (src_height as u64 * w as u64).div_ceil(src_width as u64) as u32;
                (even(w), even(h).max(2))
            }

            // Explicit target: exactly what was asked for.
            (w, h) => (even(w).max(2), even(h).max(2)),
        }
    }
}

impl Default for ScaleControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Round down to an even number (YUV420 cannot represent odd dimensions).
fn even(value: u32) -> u32 {
    value & !1
}

// ============================================================================
// Video Scaler
// ============================================================================

/// Video scaling element for YUV420 planar frames.
///
/// Scales video frames from source dimensions to target dimensions.
///
/// Both ends are dynamic: the source size is taken from each buffer's metadata
/// (falling back to the constructor's value), and the target can be changed on
/// a running pipeline through [`control`](Self::control). Output buffers are
/// stamped with the dimensions actually produced, so a downstream encoder
/// follows the change instead of encoding at a stale geometry.
pub struct VideoScale {
    /// Source width (seeded by the constructor, updated from buffer metadata).
    src_width: u32,
    /// Source height (seeded by the constructor, updated from buffer metadata).
    src_height: u32,
    /// Target width, resolved from [`Self::control`] for the current source.
    dst_width: u32,
    /// Target height, resolved from [`Self::control`] for the current source.
    dst_height: u32,
    /// The requested target, shared with [`Self::control`] handles.
    control: ScaleControl,
    /// Interpolation mode.
    mode: ScaleMode,
    /// Statistics.
    frames_processed: u64,
    /// Arena for output buffers.
    arena: Option<SharedArena>,
}

impl std::fmt::Debug for VideoScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoScale")
            .field("src_width", &self.src_width)
            .field("src_height", &self.src_height)
            .field("dst_width", &self.dst_width)
            .field("dst_height", &self.dst_height)
            .field("mode", &self.mode)
            .field("frames_processed", &self.frames_processed)
            .field("arena", &self.arena.as_ref().map(|_| "SharedArena(...)"))
            .finish()
    }
}

impl VideoScale {
    /// Create a new video scaler.
    ///
    /// # Arguments
    ///
    /// * `src_width` - Source frame width.
    /// * `src_height` - Source frame height.
    /// * `dst_width` - Target frame width.
    /// * `dst_height` - Target frame height.
    pub fn new(src_width: u32, src_height: u32, dst_width: u32, dst_height: u32) -> Self {
        let control = ScaleControl::new();
        control.set_target(dst_width, dst_height);
        let (dst_width, dst_height) = control.resolve(src_width, src_height);

        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            control,
            mode: ScaleMode::default(),
            frames_processed: 0,
            arena: None,
        }
    }

    /// Get a cloneable handle for retargeting this scaler at runtime.
    ///
    /// Clone it *before* the pipeline starts — see [`ScaleControl`].
    pub fn control(&self) -> ScaleControl {
        self.control.clone()
    }

    /// Set the interpolation mode.
    pub fn with_mode(mut self, mode: ScaleMode) -> Self {
        self.mode = mode;
        self
    }

    /// Adopt `width` x `height` as the source size and re-resolve the target.
    ///
    /// Called for every buffer: the source can change (a camera renegotiating,
    /// an upstream scaler being retargeted) and an aspect-preserving target
    /// depends on it.
    fn set_source(&mut self, width: u32, height: u32) {
        self.src_width = width;
        self.src_height = height;
        let (dst_width, dst_height) = self.control.resolve(width, height);
        self.dst_width = dst_width;
        self.dst_height = dst_height;
    }

    /// Get source dimensions.
    pub fn src_dimensions(&self) -> (u32, u32) {
        (self.src_width, self.src_height)
    }

    /// Get target dimensions.
    pub fn dst_dimensions(&self) -> (u32, u32) {
        (self.dst_width, self.dst_height)
    }

    /// Get frames processed count.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Check if scaling is a no-op (same dimensions).
    pub fn is_noop(&self) -> bool {
        self.src_width == self.dst_width && self.src_height == self.dst_height
    }

    /// Calculate YUV420 buffer size for given dimensions.
    pub fn yuv420_size(width: u32, height: u32) -> usize {
        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2)) as usize;
        y_size + 2 * uv_size
    }

    /// Scale a YUV420 planar frame.
    ///
    /// Input format: Y plane followed by U plane followed by V plane.
    /// Each plane is width*height for Y, (width/2)*(height/2) for U and V.
    pub fn scale_yuv420(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        let expected_size = Self::yuv420_size(self.src_width, self.src_height);
        if input.len() < expected_size {
            return Err(Error::Element(format!(
                "Input buffer too small: {} < {} (expected for {}x{})",
                input.len(),
                expected_size,
                self.src_width,
                self.src_height
            )));
        }

        // If no scaling needed, just copy
        if self.is_noop() {
            self.frames_processed += 1;
            return Ok(input[..expected_size].to_vec());
        }

        // Calculate plane sizes
        let src_y_size = (self.src_width * self.src_height) as usize;
        let src_uv_width = self.src_width / 2;
        let src_uv_height = self.src_height / 2;
        let src_uv_size = (src_uv_width * src_uv_height) as usize;

        let dst_y_size = (self.dst_width * self.dst_height) as usize;
        let dst_uv_width = self.dst_width / 2;
        let dst_uv_height = self.dst_height / 2;
        let dst_uv_size = (dst_uv_width * dst_uv_height) as usize;

        // Split input into planes
        let y_plane = &input[0..src_y_size];
        let u_plane = &input[src_y_size..src_y_size + src_uv_size];
        let v_plane = &input[src_y_size + src_uv_size..src_y_size + 2 * src_uv_size];

        // Allocate output
        let mut output = vec![0u8; dst_y_size + 2 * dst_uv_size];

        // Scale each plane
        match self.mode {
            ScaleMode::Bilinear => {
                scale_plane_bilinear(
                    y_plane,
                    self.src_width,
                    self.src_height,
                    &mut output[0..dst_y_size],
                    self.dst_width,
                    self.dst_height,
                );
                scale_plane_bilinear(
                    u_plane,
                    src_uv_width,
                    src_uv_height,
                    &mut output[dst_y_size..dst_y_size + dst_uv_size],
                    dst_uv_width,
                    dst_uv_height,
                );
                scale_plane_bilinear(
                    v_plane,
                    src_uv_width,
                    src_uv_height,
                    &mut output[dst_y_size + dst_uv_size..],
                    dst_uv_width,
                    dst_uv_height,
                );
            }
            ScaleMode::NearestNeighbor => {
                scale_plane_nearest(
                    y_plane,
                    self.src_width,
                    self.src_height,
                    &mut output[0..dst_y_size],
                    self.dst_width,
                    self.dst_height,
                );
                scale_plane_nearest(
                    u_plane,
                    src_uv_width,
                    src_uv_height,
                    &mut output[dst_y_size..dst_y_size + dst_uv_size],
                    dst_uv_width,
                    dst_uv_height,
                );
                scale_plane_nearest(
                    v_plane,
                    src_uv_width,
                    src_uv_height,
                    &mut output[dst_y_size + dst_uv_size..],
                    dst_uv_width,
                    dst_uv_height,
                );
            }
        }

        self.frames_processed += 1;
        Ok(output)
    }

    /// Scale a YUV420 frame with separate plane strides.
    ///
    /// Useful when planes have padding (stride > width).
    pub fn scale_yuv420_strided(
        &mut self,
        y_plane: &[u8],
        y_stride: u32,
        u_plane: &[u8],
        u_stride: u32,
        v_plane: &[u8],
        v_stride: u32,
    ) -> Result<Vec<u8>> {
        // First, copy to contiguous buffer removing stride padding
        let src_y_size = (self.src_width * self.src_height) as usize;
        let src_uv_width = self.src_width / 2;
        let src_uv_height = self.src_height / 2;
        let src_uv_size = (src_uv_width * src_uv_height) as usize;

        let mut contiguous = vec![0u8; src_y_size + 2 * src_uv_size];

        // Copy Y plane
        for row in 0..self.src_height as usize {
            let src_start = row * y_stride as usize;
            let dst_start = row * self.src_width as usize;
            contiguous[dst_start..dst_start + self.src_width as usize]
                .copy_from_slice(&y_plane[src_start..src_start + self.src_width as usize]);
        }

        // Copy U plane
        for row in 0..src_uv_height as usize {
            let src_start = row * u_stride as usize;
            let dst_start = src_y_size + row * src_uv_width as usize;
            contiguous[dst_start..dst_start + src_uv_width as usize]
                .copy_from_slice(&u_plane[src_start..src_start + src_uv_width as usize]);
        }

        // Copy V plane
        for row in 0..src_uv_height as usize {
            let src_start = row * v_stride as usize;
            let dst_start = src_y_size + src_uv_size + row * src_uv_width as usize;
            contiguous[dst_start..dst_start + src_uv_width as usize]
                .copy_from_slice(&v_plane[src_start..src_start + src_uv_width as usize]);
        }

        self.scale_yuv420(&contiguous)
    }

    /// Create a Buffer from scaled YUV data.
    pub fn scale_to_buffer(&mut self, input: &[u8], metadata: Metadata) -> Result<Buffer> {
        let scaled = self.scale_yuv420(input)?;
        let output_size = scaled.len();

        if self.arena.is_none() || self.arena.as_ref().unwrap().slot_size() < output_size {
            self.arena = Some(SharedArena::new(output_size, 32)?);
        }

        let arena = self.arena.as_mut().unwrap();
        arena.reclaim();
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("arena exhausted".into()))?;

        slot.data_mut()[..output_size].copy_from_slice(&scaled);

        let handle = MemoryHandle::with_len(slot, output_size);
        Ok(Buffer::new(handle, metadata))
    }
}

// ============================================================================
// Element Implementation
// ============================================================================

impl Element for VideoScale {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Resolve both ends per frame: the source may have changed, and the
        // target may have been retargeted through a ScaleControl handle while
        // the pipeline runs.
        let (src_width, src_height) = buffer
            .metadata()
            .video_dims()
            .unwrap_or((self.src_width, self.src_height));

        let previous = (self.dst_width, self.dst_height);
        self.set_source(src_width, src_height);
        if previous != (self.dst_width, self.dst_height) {
            tracing::info!(
                "VideoScale: {}x{} -> {}x{}",
                self.src_width,
                self.src_height,
                self.dst_width,
                self.dst_height
            );
        }

        let mut metadata = buffer.metadata().clone();
        // Tell downstream what we actually produced — without this the encoder
        // keeps encoding at the old geometry and the resize never takes effect.
        metadata.set_video_dims(self.dst_width, self.dst_height, PixelFormat::I420);

        let scaled_buffer = self.scale_to_buffer(buffer.as_bytes(), metadata)?;
        Ok(Some(scaled_buffer))
    }
}

// ============================================================================
// Scaling Algorithms
// ============================================================================

/// Bilinear interpolation for a single plane.
fn scale_plane_bilinear(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst: &mut [u8],
    dst_width: u32,
    dst_height: u32,
) {
    let x_ratio = src_width as f32 / dst_width as f32;
    let y_ratio = src_height as f32 / dst_height as f32;

    for dst_y in 0..dst_height {
        let src_y_f = dst_y as f32 * y_ratio;
        let src_y0 = src_y_f.floor() as u32;
        let src_y1 = (src_y0 + 1).min(src_height - 1);
        let y_frac = src_y_f - src_y0 as f32;

        for dst_x in 0..dst_width {
            let src_x_f = dst_x as f32 * x_ratio;
            let src_x0 = src_x_f.floor() as u32;
            let src_x1 = (src_x0 + 1).min(src_width - 1);
            let x_frac = src_x_f - src_x0 as f32;

            // Get four neighboring pixels
            let p00 = src[(src_y0 * src_width + src_x0) as usize] as f32;
            let p10 = src[(src_y0 * src_width + src_x1) as usize] as f32;
            let p01 = src[(src_y1 * src_width + src_x0) as usize] as f32;
            let p11 = src[(src_y1 * src_width + src_x1) as usize] as f32;

            // Bilinear interpolation
            let top = p00 * (1.0 - x_frac) + p10 * x_frac;
            let bottom = p01 * (1.0 - x_frac) + p11 * x_frac;
            let value = top * (1.0 - y_frac) + bottom * y_frac;

            dst[(dst_y * dst_width + dst_x) as usize] = value.round() as u8;
        }
    }
}

/// Nearest neighbor scaling for a single plane.
fn scale_plane_nearest(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst: &mut [u8],
    dst_width: u32,
    dst_height: u32,
) {
    let x_ratio = src_width as f32 / dst_width as f32;
    let y_ratio = src_height as f32 / dst_height as f32;

    for dst_y in 0..dst_height {
        let src_y = ((dst_y as f32 + 0.5) * y_ratio) as u32;
        let src_y = src_y.min(src_height - 1);

        for dst_x in 0..dst_width {
            let src_x = ((dst_x as f32 + 0.5) * x_ratio) as u32;
            let src_x = src_x.min(src_width - 1);

            dst[(dst_y * dst_width + dst_x) as usize] = src[(src_y * src_width + src_x) as usize];
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaler_creation() {
        let scaler = VideoScale::new(1920, 1080, 1280, 720);
        assert_eq!(scaler.src_dimensions(), (1920, 1080));
        assert_eq!(scaler.dst_dimensions(), (1280, 720));
        assert!(!scaler.is_noop());
    }

    #[test]
    fn test_scaler_noop() {
        let scaler = VideoScale::new(640, 480, 640, 480);
        assert!(scaler.is_noop());
    }

    #[test]
    fn test_yuv420_size() {
        // 640x480: Y=307200, U=76800, V=76800 = 460800
        assert_eq!(VideoScale::yuv420_size(640, 480), 460800);
        // 1920x1080: Y=2073600, U=518400, V=518400 = 3110400
        assert_eq!(VideoScale::yuv420_size(1920, 1080), 3110400);
    }

    #[test]
    fn test_scale_downscale() {
        let mut scaler = VideoScale::new(4, 4, 2, 2);

        // Create simple 4x4 Y plane (16 bytes) + U (1 byte) + V (1 byte) = 24 bytes
        // But YUV420: Y=16, U=4, V=4 = 24 bytes for 4x4
        let input = vec![
            // Y plane (4x4 = 16)
            100, 100, 200, 200, 100, 100, 200, 200, 150, 150, 250, 250, 150, 150, 250, 250,
            // U plane (2x2 = 4)
            128, 128, 128, 128, // V plane (2x2 = 4)
            128, 128, 128, 128,
        ];

        let output = scaler.scale_yuv420(&input).unwrap();

        // Output should be 2x2: Y=4, U=1, V=1 = 6 bytes
        assert_eq!(output.len(), 6);
        assert_eq!(scaler.frames_processed(), 1);
    }

    #[test]
    fn test_scale_upscale() {
        let mut scaler = VideoScale::new(2, 2, 4, 4);

        // 2x2 YUV420: Y=4, U=1, V=1 = 6 bytes
        let input = vec![
            // Y plane (2x2)
            0, 255, 128, 64,  // U plane (1x1)
            128, // V plane (1x1)
            128,
        ];

        let output = scaler.scale_yuv420(&input).unwrap();

        // Output should be 4x4: Y=16, U=4, V=4 = 24 bytes
        assert_eq!(output.len(), 24);
    }

    #[test]
    fn test_scale_nearest_neighbor() {
        let mut scaler = VideoScale::new(4, 4, 2, 2).with_mode(ScaleMode::NearestNeighbor);

        let input = vec![
            // Y plane (4x4)
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
            // U plane (2x2)
            128, 128, 128, 128, // V plane (2x2)
            128, 128, 128, 128,
        ];

        let output = scaler.scale_yuv420(&input).unwrap();
        assert_eq!(output.len(), 6);
    }

    #[test]
    fn test_scale_preserves_noop() {
        let mut scaler = VideoScale::new(4, 4, 4, 4);

        let input: Vec<u8> = (0..24).collect();
        let output = scaler.scale_yuv420(&input).unwrap();

        assert_eq!(output, input);
    }

    #[test]
    fn test_scale_mode_default() {
        let scaler = VideoScale::new(100, 100, 50, 50);
        assert_eq!(scaler.mode, ScaleMode::Bilinear);
    }

    #[test]
    fn test_scale_too_small_input() {
        let mut scaler = VideoScale::new(640, 480, 320, 240);
        let small_input = vec![0u8; 100];
        let result = scaler.scale_yuv420(&small_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_element_trait() {
        let mut scaler = VideoScale::new(4, 4, 2, 2);

        // Create input buffer
        let input_data: Vec<u8> = vec![128; 24]; // 4x4 YUV420
        let arena = SharedArena::new(input_data.len(), 1).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..input_data.len()].copy_from_slice(&input_data);
        let handle = MemoryHandle::with_len(slot, input_data.len());
        let buffer = Buffer::new(handle, Metadata::new());

        // Process through element
        let result = scaler.process(buffer).unwrap();
        assert!(result.is_some());

        let output = result.unwrap();
        assert_eq!(output.len(), 6); // 2x2 YUV420
    }

    // ========================================================================
    // Runtime retargeting (#28)
    // ========================================================================

    /// An I420 frame carrying its geometry in metadata.
    fn frame(width: u32, height: u32) -> Buffer {
        let size = VideoScale::yuv420_size(width, height);
        let arena = SharedArena::new(size, 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..size].fill(0x80);

        let mut metadata = Metadata::new();
        metadata.set_video_dims(width, height, PixelFormat::I420);
        Buffer::new(MemoryHandle::with_len(slot, size), metadata)
    }

    #[test]
    fn retarget_takes_effect_on_the_next_frame() {
        let mut scaler = VideoScale::new(640, 480, 640, 480);
        let control = scaler.control();

        let full = scaler.process(frame(640, 480)).unwrap().unwrap();
        assert_eq!(full.metadata().video_dims(), Some((640, 480)));

        control.set_target(320, 240);
        let small = scaler.process(frame(640, 480)).unwrap().unwrap();
        assert_eq!(small.metadata().video_dims(), Some((320, 240)));
        assert_eq!(
            small.as_bytes().len(),
            VideoScale::yuv420_size(320, 240),
            "output must be sized for the new target, not the arena slot"
        );
    }

    #[test]
    fn max_height_preserves_aspect_ratio() {
        let mut scaler = VideoScale::new(1920, 1080, 1920, 1080);
        scaler.control().set_max_height(360);

        let out = scaler.process(frame(1920, 1080)).unwrap().unwrap();
        // 16:9 -> 640x360
        assert_eq!(out.metadata().video_dims(), Some((640, 360)));
    }

    #[test]
    fn max_height_follows_a_source_resolution_change() {
        // The reason the target is resolved per frame rather than at set time.
        let mut scaler = VideoScale::new(1920, 1080, 1920, 1080);
        scaler.control().set_max_height(240);

        let wide = scaler.process(frame(1920, 1080)).unwrap().unwrap();
        assert_eq!(wide.metadata().video_dims(), Some((426, 240)));

        let squarish = scaler.process(frame(640, 480)).unwrap().unwrap();
        assert_eq!(
            squarish.metadata().video_dims(),
            Some((320, 240)),
            "4:3 source must yield a 4:3 target"
        );
    }

    #[test]
    fn never_upscales() {
        let mut scaler = VideoScale::new(320, 240, 320, 240);
        scaler.control().set_max_height(1080);

        let out = scaler.process(frame(320, 240)).unwrap().unwrap();
        assert_eq!(
            out.metadata().video_dims(),
            Some((320, 240)),
            "a target above the source resolution must pass through, not invent detail"
        );
    }

    #[test]
    fn passthrough_restores_the_source_resolution() {
        let mut scaler = VideoScale::new(640, 480, 640, 480);
        let control = scaler.control();

        control.set_max_height(240);
        let small = scaler.process(frame(640, 480)).unwrap().unwrap();
        assert_eq!(small.metadata().video_dims(), Some((320, 240)));

        control.passthrough();
        let full = scaler.process(frame(640, 480)).unwrap().unwrap();
        assert_eq!(full.metadata().video_dims(), Some((640, 480)));
    }

    #[test]
    fn odd_targets_round_down_to_even() {
        // YUV420 chroma is subsampled by two; odd dimensions cannot be
        // represented and would corrupt the planes.
        let control = ScaleControl::new();
        control.set_target(641, 361);
        assert_eq!(control.target(), (640, 360));

        // ...including aspect-derived widths.
        let mut scaler = VideoScale::new(1000, 500, 1000, 500);
        scaler.control().set_max_height(101);
        let out = scaler.process(frame(1000, 500)).unwrap().unwrap();
        let (w, h) = out.metadata().video_dims().unwrap();
        assert_eq!(w % 2, 0, "width {w} must be even");
        assert_eq!(h % 2, 0, "height {h} must be even");
    }

    #[test]
    fn source_dimensions_come_from_buffer_metadata() {
        // Constructed for 1920x1080, but fed 640x480: the buffer wins.
        let mut scaler = VideoScale::new(1920, 1080, 1920, 1080);
        scaler.control().set_max_height(240);

        let out = scaler.process(frame(640, 480)).unwrap().unwrap();
        assert_eq!(scaler.src_dimensions(), (640, 480));
        assert_eq!(out.metadata().video_dims(), Some((320, 240)));
    }

    #[test]
    fn scaled_output_is_stamped_for_both_metadata_conventions() {
        let mut scaler = VideoScale::new(640, 480, 320, 240);
        let out = scaler.process(frame(640, 480)).unwrap().unwrap();

        assert_eq!(out.metadata().video_pixel_format(), Some(PixelFormat::I420));
        assert_eq!(out.metadata().get::<u64>("width"), Some(&320));
        assert_eq!(out.metadata().get::<u64>("height"), Some(&240));
    }
}
