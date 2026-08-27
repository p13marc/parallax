//! Importing a dma-buf frame as GPU textures, instead of uploading it.
//!
//! The upload path in [`super::wgpu_backend`] reads a frame's bytes and
//! copies them into textures. When the producer already owns GPU-visible
//! memory — a VA-API decode target, a V4L2 capture buffer — that copy is
//! pure waste, and worse, it may not even be possible: an Intel VA decode
//! target is **Y-tiled**, so its bytes are not rows and reading them costs a
//! de-tiling pass before the upload can begin.
//!
//! Importing skips both. The dma-buf's fd, its DRM format modifier and the
//! per-plane offsets and pitches are handed to Vulkan, which creates images
//! that alias the producer's memory directly. Nothing is copied and the CPU
//! never touches a pixel.
//!
//! # What this module is and is not
//!
//! The actual `VkImage` creation lives in `wgpu-hal`
//! (`Device::texture_from_dmabuf_fd`), which does the whole
//! `VkExternalMemoryImageCreateInfo` +
//! `VkImageDrmFormatModifierExplicitCreateInfoEXT` +
//! `vkGetMemoryFdPropertiesKHR` + dedicated-allocation sequence and gets the
//! fd ownership right on every failure path. Rolling that by hand would mean
//! re-auditing it at every wgpu bump for no capability gain.
//!
//! What is ours is everything around it: deciding the per-plane geometry
//! (which is the part that can be wrong in ways that look like a picture),
//! duplicating the fd so the producer keeps its own, and choosing the
//! initial image layout.
//!
//! # One image per plane
//!
//! `texture_from_dmabuf_fd` takes a single `(offset, stride)` pair, so a
//! disjoint multi-planar `VkImage` is not reachable through wgpu's public
//! API — and it is not wanted anyway. `present.wgsl` binds each plane as its
//! own `texture_2d<f32>`, so N single-plane images aliasing one imported
//! allocation drop straight into the existing bind group with no shader
//! change. Two imports of one dma-buf produce two `VkDeviceMemory` objects
//! over the same kernel object, which is legal: both images are sampled
//! only, and they address disjoint byte ranges.

use crate::error::{Error, Result};
use crate::video::layout::PlaneImport;

/// Whether this adapter can import dma-bufs at all.
///
/// Backend *and* feature: `VULKAN_EXTERNAL_MEMORY_DMA_BUF` is Vulkan-only,
/// and wgpu reports it exactly when `VK_KHR_external_memory_fd`,
/// `VK_EXT_external_memory_dma_buf` and `VK_EXT_image_drm_format_modifier`
/// are all present — the three this needs — and enables them itself. So
/// there is no hand-rolled device creation here: the feature bit is the
/// whole story.
pub(crate) fn import_supported(adapter: &wgpu::Adapter) -> bool {
    adapter.get_info().backend == wgpu::Backend::Vulkan
        && adapter
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
}

/// The hal-side usage bits matching a `TextureUsages` set.
///
/// Only the two an imported texture can have: it is read by the shader, and
/// a test may copy it out to check the import got the layout right. It is
/// never written — the producer owns these bytes.
fn hal_uses(usage: wgpu::TextureUsages) -> wgpu::wgt::TextureUses {
    let mut uses = wgpu::wgt::TextureUses::empty();
    if usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        uses |= wgpu::wgt::TextureUses::RESOURCE;
    }
    if usage.contains(wgpu::TextureUsages::COPY_SRC) {
        uses |= wgpu::wgt::TextureUses::COPY_SRC;
    }
    uses
}

/// Import one plane as a texture aliasing the dma-buf.
///
/// # Safety
///
/// `fd` must be a dma-buf whose contents match `plane` and `modifier`. The
/// import is not validated by the driver beyond the layout being plausible,
/// so a wrong stride or modifier yields a wrong picture rather than an error.
pub(crate) unsafe fn import_plane(
    device: &wgpu::Device,
    fd: std::os::fd::BorrowedFd<'_>,
    plane: &PlaneImport,
    modifier: u64,
    usage: wgpu::TextureUsages,
    label: &str,
) -> Result<wgpu::Texture> {
    use std::os::fd::AsFd;

    // A fresh dup per import. `texture_from_dmabuf_fd` takes ownership and
    // closes the fd on every failure path, while the producer's slot must
    // keep its own — it is what the frame pool recycles.
    let owned = fd
        .as_fd()
        .try_clone_to_owned()
        .map_err(|e| Error::Element(format!("autovideosink/wgpu: duplicating a dmabuf fd: {e}")))?;

    let size = wgpu::Extent3d {
        width: plane.width,
        height: plane.height,
        depth_or_array_layers: 1,
    };
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: super::wgpu_backend::texel(plane.format),
        usage: hal_uses(usage),
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    // SAFETY: the caller's contract, plus wgpu-hal's own fd handling.
    let hal_texture = unsafe {
        let hal_device = device
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or_else(|| Error::Element("autovideosink/wgpu: not a Vulkan device".into()))?;
        hal_device
            .texture_from_dmabuf_fd(owned, &hal_desc, modifier, plane.stride, plane.offset)
            .map_err(|e| {
                Error::Element(format!("autovideosink/wgpu: dmabuf import failed: {e:?}"))
            })?
    };

    // SAFETY: the hal texture was just created by this device.
    Ok(unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: super::wgpu_backend::texel(plane.format),
                usage,
                view_formats: &[],
            },
            // The only honest value: the image was created with
            // `VK_IMAGE_LAYOUT_UNDEFINED`, so claiming it is already
            // shader-readable would skip the transition that makes it so.
            wgpu::wgt::TextureUses::UNINITIALIZED,
        )
    })
}

#[cfg(all(test, feature = "vaapi"))]
mod hardware_tests {
    use super::*;
    use crate::format::{PixelFormat, PlaneDesc, PlaneLayout};
    use crate::gpu::vaapi::VaFrame;
    use crate::memory::I915_FORMAT_MOD_Y_TILED;
    use crate::video::layout::plane_imports;

    const TILE_W: usize = 128;
    const TILE_H: usize = 32;
    const TILE_COL: usize = 16;

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .ok()?;
        if !import_supported(&adapter) {
            eprintln!("skipping: this adapter cannot import dma-bufs");
            return None;
        }
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF,
            ..Default::default()
        }))
        .ok()
    }

    /// Write `value(x, y)` into a plane in Y-tile order.
    fn write_tiled(
        bytes: &mut [u8],
        offset: usize,
        pitch: usize,
        w: usize,
        h: usize,
        value: impl Fn(usize, usize) -> u8,
    ) {
        for y in 0..h {
            for x in 0..w {
                let tile = (y / TILE_H) * (pitch / TILE_W) + x / TILE_W;
                let off = offset
                    + tile * TILE_W * TILE_H
                    + ((x % TILE_W) / TILE_COL) * TILE_COL * TILE_H
                    + (y % TILE_H) * TILE_COL
                    + x % TILE_COL;
                bytes[off] = value(x, y);
            }
        }
    }

    /// Copy a texture back into packed bytes.
    fn readback(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        bytes_wide: u32,
        rows: u32,
    ) -> Vec<u8> {
        let bpr = bytes_wide.next_multiple_of(256);
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bpr as u64) * rows as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &out,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(rows),
                },
            },
            wgpu::Extent3d {
                width: texture.width(),
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        let slice = out.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mapped = slice.get_mapped_range().expect("mapped");
        (0..rows as usize)
            .flat_map(|r| mapped[r * bpr as usize..r * bpr as usize + bytes_wide as usize].to_vec())
            .collect()
    }

    /// The GPU reads a Y-tiled dma-buf exactly as the producer wrote it.
    ///
    /// Both planes: the chroma plane is where the two easy mistakes live —
    /// its offset (a whole tile-aligned luma plane further in, not
    /// `width * height`) and its texel width (half the luma's, at the same
    /// byte pitch, because NV12 chroma is two bytes per texel).
    #[test]
    fn an_imported_frame_reads_back_exactly() {
        const W: usize = 256;
        const H: usize = 64;

        let Some((device, queue)) = gpu() else { return };
        let res = |w, h| cros_codecs::Resolution {
            width: w,
            height: h,
        };
        let Ok(mut frame) = VaFrame::new(res(W as u32, H as u32), res(W as u32, H as u32)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };

        let luma = |x: usize, y: usize| ((x * 7 + y * 31) % 251) as u8;
        let chroma = |x: usize, y: usize| ((x * 13 + y * 3 + 5) % 251) as u8;
        let pitch = frame.pitches()[0];
        let (luma_off, chroma_off) = (frame.offsets()[0], frame.offsets()[1]);
        {
            let bytes = frame.as_bytes_mut();
            write_tiled(bytes, luma_off, pitch, W, H, luma);
            // The chroma plane is byte-addressed like any other: W bytes
            // wide (W/2 texels of two bytes), H/2 rows.
            write_tiled(bytes, chroma_off, pitch, W, H / 2, chroma);
        }

        let layout = PlaneLayout::from_planes(&[
            PlaneDesc {
                offset: luma_off,
                stride: pitch,
            },
            PlaneDesc {
                offset: chroma_off,
                stride: frame.pitches()[1],
            },
        ]);
        let planes =
            plane_imports(PixelFormat::Nv12, W as u32, H as u32, &layout, 0).expect("geometry");

        // SAFETY: layout and modifier come from the allocator above.
        let y_tex = unsafe {
            import_plane(
                &device,
                frame.dmabuf(),
                &planes[0],
                I915_FORMAT_MOD_Y_TILED,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                "test-y",
            )
        }
        .expect("import luma");
        let uv_tex = unsafe {
            import_plane(
                &device,
                frame.dmabuf(),
                &planes[1],
                I915_FORMAT_MOD_Y_TILED,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                "test-uv",
            )
        }
        .expect("import chroma");

        let got = readback(&device, &queue, &y_tex, W as u32, H as u32);
        for y in 0..H {
            for x in 0..W {
                assert_eq!(got[y * W + x], luma(x, y), "luma at ({x},{y})");
            }
        }

        let got = readback(&device, &queue, &uv_tex, W as u32, (H / 2) as u32);
        for y in 0..H / 2 {
            for x in 0..W {
                assert_eq!(got[y * W + x], chroma(x, y), "chroma at ({x},{y})");
            }
        }
    }
}
