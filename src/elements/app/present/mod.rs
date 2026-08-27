//! Render backends for [`AutoVideoSink`](super::AutoVideoSink) (#190).
//!
//! The element side (pacing, QoS, caps, the consume task) never knows which
//! backend won; the display thread picks one when the window exists:
//!
//! - [`WgpuBackend`](wgpu_backend::WgpuBackend) (feature `display-gpu`):
//!   uploads I420/NV12/RGB planes to the GPU, converts colorspace and
//!   scales (bilinear) in a fragment shader, presents via swapchain.
//! - [`CpuBackend`]: softbuffer + the row-wise letterbox blit — the
//!   original path, still the fallback when the GPU probe or init fails,
//!   with an in-thread YUV→BGRA conversion for frames negotiated by the
//!   GPU caps that end up here anyway.

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::format::PixelFormat;
use std::num::NonZeroU32;
use std::sync::Arc;
use winit::window::Window;

#[cfg(feature = "display-gpu")]
pub(crate) mod dmabuf_import;
#[cfg(feature = "display-gpu")]
pub(crate) mod wgpu_backend;

/// Frame data sent to the display thread.
///
/// Carries the pipeline `Buffer` itself — a clone is three atomic
/// increments and (since #138) an allocation-free metadata copy, so the
/// display thread shares the arena slot instead of receiving a full-frame
/// `Vec` copy (#141). The slot stays pinned until the frame is replaced,
/// which is why the display channel is kept shallow. Carrying the whole
/// `Buffer` is also what keeps a future dmabuf frame (#62) a backend
/// branch instead of a trait change.
pub(crate) struct DisplayFrame {
    /// Pixel data, arena- or externally-backed (#194).
    pub(crate) data: Buffer,
    /// Frame width in pixels.
    pub(crate) width: u32,
    /// Frame height in pixels.
    pub(crate) height: u32,
    /// Payload layout. Only `I420`, `Nv12`, `Rgba` and `Bgra` reach the
    /// display thread — the caps admit nothing else.
    pub(crate) format: PixelFormat,
    /// Non-packed plane layout, when the producer declared one (#194):
    /// External frames carry the decoder's real strides. `None` = packed.
    pub(crate) layout: Option<crate::format::PlaneLayout>,
    /// What the stream said about its colour (#265). `None` = nothing said.
    pub(crate) color: Option<crate::format::ColorInfo>,
    /// Which flush generation this frame belongs to (#265). A flushing seek
    /// bumps the sink's counter, and the display loop skips anything stamped
    /// below it — otherwise the queued pre-seek frames are blitted *after*
    /// the seek, flashing the old picture back over the new one.
    pub(crate) generation: u64,
}

impl DisplayFrame {
    /// The colour to render with: what the stream said, or the height
    /// heuristic when it said nothing (#265).
    pub(crate) fn resolved_color(&self) -> crate::format::ColorInfo {
        let mut color = self.color.unwrap_or_default();
        color.matrix = color.matrix_or(crate::video::color::matrix_for_height(self.height));
        color
    }
}

/// A display-thread-local presentation backend (deliberately not `Send`).
pub(crate) trait RenderBackend {
    /// Upload + present one frame, letterboxed into the current window size.
    fn render(&mut self, frame: &DisplayFrame, window_size: (u32, u32)) -> Result<()>;

    /// The window was resized (GPU: reconfigure the swapchain on the next
    /// render; CPU: nothing — the surface resizes per render).
    fn resized(&mut self, width: u32, height: u32);

    /// Backend name for logs: `"wgpu"` or `"softbuffer"`.
    fn name(&self) -> &'static str;
}

/// Pick a backend for `window`.
///
/// Tries the GPU (feature `display-gpu`, `prefer_gpu`, probe) and falls
/// back to softbuffer on any init error — a compositor quirk must degrade
/// to a slower picture, never kill the sink. `Err` only when *both* fail.
pub(crate) fn create_backend(
    window: Arc<Window>,
    prefer_gpu: bool,
) -> Result<Box<dyn RenderBackend>> {
    #[cfg(feature = "display-gpu")]
    if prefer_gpu && wgpu_backend::gpu_available() {
        match wgpu_backend::WgpuBackend::new(window.clone()) {
            Ok(backend) => {
                tracing::info!("autovideosink: presenting via wgpu");
                return Ok(Box::new(backend));
            }
            Err(e) => {
                tracing::warn!(
                    "autovideosink: GPU probe passed but init failed ({e}); \
                     falling back to softbuffer (CPU convert + blit)"
                );
            }
        }
    }
    #[cfg(not(feature = "display-gpu"))]
    let _ = prefer_gpu;

    Ok(Box::new(CpuBackend::new(window)?))
}

/// Whether the wgpu presentation path is available on this machine.
///
/// `false` without the `display-gpu` feature, when `PARALLAX_NO_GPU` is
/// set, or when no non-CPU adapter exists (llvmpipe keeps softbuffer —
/// `PARALLAX_FORCE_GPU=1` overrides for testing). The probe runs once per
/// process, before any window exists, and is what flips the sink's caps to
/// accept YUV directly.
pub fn gpu_present_available() -> bool {
    #[cfg(feature = "display-gpu")]
    {
        wgpu_backend::gpu_available()
    }
    #[cfg(not(feature = "display-gpu"))]
    {
        false
    }
}

/// Whether the GPU path can *import* a dma-buf rather than uploading it.
///
/// Strictly narrower than [`gpu_present_available`]: presentation works on
/// any adapter, importing needs Vulkan with the external-memory extensions.
/// The sink must not advertise `MemoryType::DmaBuf` on the strength of the
/// former, or a producer that owns GPU memory hands over frames the sink
/// cannot read.
pub fn gpu_dmabuf_import_available() -> bool {
    #[cfg(feature = "display-gpu")]
    {
        wgpu_backend::dmabuf_import_available()
    }
    #[cfg(not(feature = "display-gpu"))]
    {
        false
    }
}

// The presentation maths lives in `crate::video`, which is unfeatured and
// wgpu-free so a host toolkit on a different graphics stack can share it —
// including the colour matrices, which now reach the shader through
// `video::shader_mode_for` rather than being re-derived here.
pub(crate) use crate::video::fit::letterbox_rect;

// ============================================================================
// CPU backend (softbuffer)
// ============================================================================

/// Horizontal nearest-neighbour map, cached across frames — rebuilding it
/// costs one small allocation only when the source or window geometry
/// changes.
#[derive(Default)]
pub(crate) struct BlitCache {
    key: (usize, usize),
    x_map: Vec<u32>,
}

/// The softbuffer path: CPU letterbox blit into the window's shm buffer.
pub(crate) struct CpuBackend {
    // Field order: surface before context before window — softbuffer's
    // surface borrows the display connection the context owns.
    surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
    _context: softbuffer::Context<Arc<Window>>,
    blit_cache: BlitCache,
    /// YUV frames arriving on a GPU-negotiated pipeline after a runtime
    /// fallback are converted here, then blitted as BGRA. Empty until the
    /// first YUV frame.
    scratch: Vec<u8>,
    /// A frame whose buffer ends tight against its last row cannot be read
    /// row-wise by the engine (#196), so it is repacked here first. Real
    /// strided producers allocate whole rows, so this stays empty.
    repack_scratch: Vec<u8>,
    converter: Option<(
        (PixelFormat, u32, u32, crate::format::ColorInfo),
        crate::converters::VideoConvert,
    )>,
    warned_yuv: bool,
}

impl CpuBackend {
    pub(crate) fn new(window: Arc<Window>) -> Result<Self> {
        let context = softbuffer::Context::new(window.clone())
            .map_err(|e| Error::Element(format!("softbuffer context: {e}")))?;
        let surface = softbuffer::Surface::new(&context, window)
            .map_err(|e| Error::Element(format!("softbuffer surface: {e}")))?;
        Ok(Self {
            surface,
            _context: context,
            blit_cache: BlitCache::default(),
            scratch: Vec::new(),
            repack_scratch: Vec::new(),
            converter: None,
            warned_yuv: false,
        })
    }

    /// Convert a YUV frame into `self.scratch` as BGRA.
    ///
    /// A strided frame is read in place through its declared plane layout
    /// (#196) — I420 and NV12 are exactly the two formats this path takes,
    /// and both engine families address planes by stride. The repack below
    /// only catches a frame whose buffer ends tight against its last row.
    fn convert_yuv(&mut self, frame: &DisplayFrame) -> Result<()> {
        use crate::converters::{PixelFormat as ConvFormat, VideoConvert};
        if !self.warned_yuv {
            self.warned_yuv = true;
            tracing::warn!(
                "autovideosink: converting {:?} on the CPU — the GPU backend was \
                 negotiated but is not rendering (degraded fallback)",
                frame.format
            );
        }
        let conv_format = match frame.format {
            PixelFormat::I420 => ConvFormat::I420,
            PixelFormat::Nv12 => ConvFormat::Nv12,
            other => {
                return Err(Error::Element(format!(
                    "autovideosink: unsupported display format {other:?}"
                )));
            }
        };
        // Colour is part of the key (#265): a stream that changes matrix or
        // range mid-flight must not keep the previous one's converter.
        let color = frame.resolved_color();
        let key = (frame.format, frame.width, frame.height, color);
        if self.converter.as_ref().is_none_or(|(k, _)| *k != key) {
            self.converter = Some((
                key,
                VideoConvert::new(conv_format, ConvFormat::Bgra, frame.width, frame.height)?
                    .with_color_info(color),
            ));
        }
        // Field borrows below are disjoint (repack_scratch read, scratch
        // written, converter read).
        let (w, h, fmt) = (frame.width, frame.height, frame.format);
        let packed_layout = crate::format::PlaneLayout::packed(fmt, w, h);
        let layout = frame.layout.unwrap_or(packed_layout);
        let bytes = frame.data.as_bytes();
        let (data, layout) = if bytes.len() >= layout.full_span_len(fmt, w, h) {
            (bytes, layout)
        } else {
            let packed_len = packed_layout.required_len(fmt, w, h);
            self.repack_scratch.resize(packed_len, 0);
            layout
                .repack_into(bytes, fmt, w, h, &mut self.repack_scratch)
                .map_err(Error::Element)?;
            (&self.repack_scratch[..], packed_layout)
        };
        let out_len = w as usize * h as usize * 4;
        self.scratch.resize(out_len, 0);
        let (_, converter) = self.converter.as_ref().expect("installed above");
        converter.convert(data, layout, &mut self.scratch)
    }
}

impl RenderBackend for CpuBackend {
    fn render(&mut self, frame: &DisplayFrame, window_size: (u32, u32)) -> Result<()> {
        let (width, height) = window_size;
        if width == 0 || height == 0 {
            return Ok(());
        }
        if let (Some(w), Some(h)) = (NonZeroU32::new(width), NonZeroU32::new(height))
            && self.surface.resize(w, h).is_err()
        {
            return Ok(());
        }

        let bgra = match frame.format {
            PixelFormat::Bgra => true,
            PixelFormat::Rgba => false,
            _ => {
                self.convert_yuv(frame)?;
                true
            }
        };

        let Ok(mut buffer) = self.surface.buffer_mut() else {
            return Ok(());
        };
        let src: &[u8] = match frame.format {
            PixelFormat::Bgra | PixelFormat::Rgba => {
                debug_assert!(frame.layout.is_none(), "no strided RGB producers exist");
                frame.data.as_bytes()
            }
            _ => &self.scratch,
        };
        blit_frame(
            src,
            frame.width as usize,
            frame.height as usize,
            bgra,
            &mut buffer,
            width as usize,
            height as usize,
            &mut self.blit_cache,
        );
        let _ = buffer.present();
        Ok(())
    }

    fn resized(&mut self, _width: u32, _height: u32) {
        // The surface resizes per render.
    }

    fn name(&self) -> &'static str {
        "softbuffer"
    }
}

/// Blit a packed 4-byte-per-pixel frame into the presentation buffer with
/// nearest-neighbour letterbox scaling.
#[allow(clippy::too_many_arguments)]
fn blit_frame(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    bgra: bool,
    buffer: &mut [u32],
    dst_width: usize,
    dst_height: usize,
    cache: &mut BlitCache,
) {
    if src_width == 0
        || src_height == 0
        || dst_width == 0
        || dst_height == 0
        || src.len() < src_width * src_height * 4
        || buffer.len() < dst_width * dst_height
    {
        return;
    }

    // Aspect-preserving letterbox: scale to fit, center, black bars around.
    // Stretching to the window distorted anything whose aspect ratio did not
    // match — obvious the moment a 16:9 stream met a fullscreen 16:10 panel.
    let (x0, y0, out_width, out_height) =
        letterbox_rect(src_width, src_height, dst_width, dst_height);

    if cache.key != (src_width, out_width) {
        cache.key = (src_width, out_width);
        cache.x_map.clear();
        cache
            .x_map
            .extend((0..out_width).map(|x| ((x * src_width) / out_width) as u32));
    }

    // Row-wise: bars fill by slice, the image row converts RGBA → 0RGB with
    // all bounds established once per row (#141) — the previous per-pixel
    // loop paid a bounds check, two `contains`, and a mul+div per pixel.
    for dst_y in 0..dst_height {
        let row = &mut buffer[dst_y * dst_width..(dst_y + 1) * dst_width];
        if dst_y < y0 || dst_y >= y0 + out_height {
            row.fill(0); // letterbox bar
            continue;
        }
        row[..x0].fill(0);
        row[x0 + out_width..].fill(0);

        let src_y = ((dst_y - y0) * src_height) / out_height;
        let src_row = &src[src_y * src_width * 4..(src_y + 1) * src_width * 4];
        let out_row = &mut row[x0..x0 + out_width];
        match (bgra, out_width == src_width) {
            // BGRA is softbuffer's 0RGB layout already (the alpha byte lands
            // in the ignored top bits), so no per-pixel channel shuffle —
            // which used to be ~37% of the player's entire CPU (#161). The
            // `try_into` on a 4-byte subslice compiles to a single 32-bit
            // load; spelling out the four bytes kept four byte-loads plus
            // shifts (measured).
            (true, true) => {
                // 1:1: a straight row copy at memcpy speed.
                for (dst, px) in out_row.iter_mut().zip(src_row.as_chunks::<4>().0) {
                    *dst = u32::from_le_bytes(*px);
                }
            }
            (true, false) => {
                for (dst, &sx) in out_row.iter_mut().zip(&cache.x_map[..out_width]) {
                    let o = sx as usize * 4;
                    *dst = u32::from_le_bytes(src_row[o..o + 4].try_into().unwrap());
                }
            }
            // RGBA compat: shuffle each pixel into 0xRRGGBB.
            (false, true) => {
                // 1:1 horizontal: straight zip, no index math.
                for (dst, px) in out_row.iter_mut().zip(src_row.as_chunks::<4>().0) {
                    *dst = ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | (px[2] as u32);
                }
            }
            (false, false) => {
                for (dst, &sx) in out_row.iter_mut().zip(&cache.x_map[..out_width]) {
                    let o = sx as usize * 4;
                    *dst = ((src_row[o] as u32) << 16)
                        | ((src_row[o + 1] as u32) << 8)
                        | (src_row[o + 2] as u32);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An arena-backed frame of `w`x`h` with every byte `fill`, for blits.
    fn filled_frame(w: u32, h: u32, fill: u8, format: PixelFormat) -> DisplayFrame {
        let len = (w * h * 4) as usize;
        let arena = crate::memory::SharedArena::new(len, 2).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..len].fill(fill);
        DisplayFrame {
            generation: 0,
            data: Buffer::new(
                crate::buffer::MemoryHandle::with_len(slot, len),
                crate::metadata::Metadata::new(),
            ),
            width: w,
            height: h,
            format,
            layout: None,
            color: None,
        }
    }

    /// An arena-backed white RGBA frame of `w`x`h` for blit tests.
    fn white_frame(w: u32, h: u32) -> DisplayFrame {
        filled_frame(w, h, 0xFF, PixelFormat::Rgba)
    }

    /// A single-pixel frame with explicit channel bytes, for format tests.
    fn pixel_frame(bytes: [u8; 4], format: PixelFormat) -> DisplayFrame {
        let arena = crate::memory::SharedArena::new(64, 2).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..4].copy_from_slice(&bytes);
        DisplayFrame {
            generation: 0,
            data: Buffer::new(
                crate::buffer::MemoryHandle::with_len(slot, 4),
                crate::metadata::Metadata::new(),
            ),
            width: 1,
            height: 1,
            format,
            layout: None,
            color: None,
        }
    }

    fn blit(frame: &DisplayFrame, buffer: &mut [u32], w: usize, h: usize, cache: &mut BlitCache) {
        blit_frame(
            frame.data.as_bytes(),
            frame.width as usize,
            frame.height as usize,
            frame.format == PixelFormat::Bgra,
            buffer,
            w,
            h,
            cache,
        );
    }

    /// Both formats land the same color in the presentation buffer: an
    /// RGBA pixel goes through the shuffle, a BGRA pixel is copied verbatim
    /// (its little-endian u32 IS softbuffer's 0RGB layout).
    #[test]
    fn bgra_and_rgba_blit_the_same_color() {
        // R=0x11 G=0x22 B=0x33 → presentation 0x112233 (+ alpha in the
        // ignored top byte on the BGRA path).
        let rgba = pixel_frame([0x11, 0x22, 0x33, 0xFF], PixelFormat::Rgba);
        let bgra = pixel_frame([0x33, 0x22, 0x11, 0xFF], PixelFormat::Bgra);

        for (frame, mask) in [(&rgba, 0xFF_FFFFu32), (&bgra, 0x00FF_FFFF)] {
            // 1:1 path.
            let mut one = vec![0u32; 1];
            blit(frame, &mut one, 1, 1, &mut BlitCache::default());
            assert_eq!(one[0] & mask, 0x112233);
            // Scaled path (1x1 → 2x2).
            let mut four = vec![0u32; 4];
            blit(frame, &mut four, 2, 2, &mut BlitCache::default());
            for px in four {
                assert_eq!(px & mask, 0x112233);
            }
        }
    }

    #[test]
    fn letterbox_centers_and_paints_bars_black() {
        // 2x2 white frame into a 4x2 window: 1:1 content in the middle two
        // columns, black pillars on columns 0 and 3.
        let frame = white_frame(2, 2);
        let mut buffer = vec![0x123456u32; 4 * 2];
        blit(&frame, &mut buffer, 4, 2, &mut BlitCache::default());

        for y in 0..2 {
            assert_eq!(buffer[y * 4], 0, "left pillar is black");
            assert_eq!(buffer[y * 4 + 3], 0, "right pillar is black");
            assert_eq!(buffer[y * 4 + 1], 0xFFFFFF, "content");
            assert_eq!(buffer[y * 4 + 2], 0xFFFFFF, "content");
        }
    }

    #[test]
    fn matching_aspect_fills_the_window() {
        let frame = white_frame(2, 2);
        let mut buffer = vec![0u32; 8 * 8];
        blit(&frame, &mut buffer, 8, 8, &mut BlitCache::default());
        assert!(buffer.iter().all(|p| *p == 0xFFFFFF), "no bars");
    }

    /// The scaled path via the cached x-map produces the same result when
    /// geometry changes between frames (cache rebuild).
    #[test]
    fn blit_cache_survives_geometry_changes() {
        let mut cache = BlitCache::default();
        let frame = white_frame(2, 2);
        let mut a = vec![0u32; 8 * 8];
        blit(&frame, &mut a, 8, 8, &mut cache);
        let mut b = vec![0u32; 6 * 6];
        blit(&frame, &mut b, 6, 6, &mut cache);
        assert!(a.iter().all(|p| *p == 0xFFFFFF));
        assert!(b.iter().all(|p| *p == 0xFFFFFF));
    }
}
