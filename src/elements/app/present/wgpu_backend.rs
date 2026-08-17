//! wgpu presentation backend for AutoVideoSink (#190).
//!
//! Uploads the decoder's planes as textures, converts colorspace and
//! letterbox-scales (bilinear) in a fragment shader, presents through the
//! surface's swapchain. Everything the CPU path did per frame — the
//! I420→BGRA convert (8.3 MB written at 1080p) and the letterbox blit
//! (8.3 MB read + 8.3 MB written) — collapses into one ~3.1 MB texture
//! upload the driver performs.

use super::{DisplayFrame, RenderBackend, color, letterbox_rect};
use crate::error::{Error, Result};
use crate::format::PixelFormat;
use std::sync::Arc;
use std::sync::OnceLock;
use winit::window::Window;

/// Process-wide GPU probe: is a non-CPU adapter available?
///
/// Runs once, before any window exists (no surface is created), so the
/// element can flip its caps at negotiation time. llvmpipe is rejected —
/// a software Vulkan raster is slower than the softbuffer blit — unless
/// `PARALLAX_FORCE_GPU=1` (testing). `PARALLAX_NO_GPU=1` vetoes outright.
pub(crate) fn gpu_available() -> bool {
    static PROBE: OnceLock<bool> = OnceLock::new();
    *PROBE.get_or_init(|| {
        if std::env::var_os("PARALLAX_NO_GPU").is_some_and(|v| v != "0") {
            tracing::info!("autovideosink: GPU presentation disabled (PARALLAX_NO_GPU)");
            return false;
        }
        let force = std::env::var_os("PARALLAX_FORCE_GPU").is_some_and(|v| v != "0");
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }));
        match adapter {
            Ok(adapter) => {
                let info = adapter.get_info();
                let usable = force || info.device_type != wgpu::DeviceType::Cpu;
                tracing::info!(
                    "autovideosink: GPU probe found {:?} ({:?}, {:?}) — {}",
                    info.name,
                    info.device_type,
                    info.backend,
                    if usable {
                        "using it"
                    } else {
                        "software rasterizer, keeping softbuffer"
                    }
                );
                usable
            }
            Err(e) => {
                tracing::info!("autovideosink: no GPU adapter ({e}); keeping softbuffer");
                false
            }
        }
    })
}

/// The per-format texture set bound to the shader.
struct PlaneTextures {
    /// `(format, width, height)` these textures were built for.
    key: (PixelFormat, u32, u32),
    y: wgpu::Texture,
    u: wgpu::Texture,
    v: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    /// Shader mode (see present.wgsl) for this format + geometry.
    mode: u32,
}

pub(crate) struct WgpuBackend {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    textures: Option<PlaneTextures>,
    /// Set by `resized`; applied (reconfigure) on the next render.
    pending_size: Option<(u32, u32)>,
}

impl WgpuBackend {
    pub(crate) fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(window)
            .map_err(|e| Error::Element(format!("wgpu surface: {e}")))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .map_err(|e| Error::Element(format!("wgpu adapter: {e}")))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| Error::Element(format!("wgpu device: {e}")))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Bgra8Unorm)
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| Error::Element("wgpu: surface reports no formats".into()))?;
        // The element paces to PTS against the pipeline clock; presentation
        // must not add a second governor. Mailbox never blocks and shows
        // the latest frame; Fifo's worst case blocks one vsync, which the
        // shallow display channel absorbs.
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Opaque),
            view_formats: Vec::new(),
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("autovideosink-present"),
            source: wgpu::ShaderSource::Wgsl(include_str!("present.wgsl").into()),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("autovideosink-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("autovideosink-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("autovideosink-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Bilinear: a visible quality win over the CPU path's
        // nearest-neighbour when the window size differs from the video.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("autovideosink-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("autovideosink-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_layout,
            sampler,
            params,
            textures: None,
            pending_size: None,
        })
    }

    /// (Re)build the texture set for this frame's format + geometry.
    fn ensure_textures(&mut self, frame: &DisplayFrame) -> Result<()> {
        let key = (frame.format, frame.width, frame.height);
        if self.textures.as_ref().is_some_and(|t| t.key == key) {
            return Ok(());
        }

        let (w, h) = (frame.width, frame.height);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let make = |label: &str, w: u32, h: u32, format: wgpu::TextureFormat| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };

        let matrix = color::matrix_for_height(h);
        let (y, u, v, mode) = match frame.format {
            PixelFormat::I420 => (
                make("y", w, h, wgpu::TextureFormat::R8Unorm),
                make("u", cw, ch, wgpu::TextureFormat::R8Unorm),
                make("v", cw, ch, wgpu::TextureFormat::R8Unorm),
                match matrix {
                    color::ColorMatrix::Bt709 => 0,
                    color::ColorMatrix::Bt601 => 1,
                },
            ),
            PixelFormat::Nv12 => (
                make("y", w, h, wgpu::TextureFormat::R8Unorm),
                make("uv", cw, ch, wgpu::TextureFormat::Rg8Unorm),
                make("dummy-v", 1, 1, wgpu::TextureFormat::R8Unorm),
                match matrix {
                    color::ColorMatrix::Bt709 => 2,
                    color::ColorMatrix::Bt601 => 3,
                },
            ),
            PixelFormat::Rgba => (
                make("rgb", w, h, wgpu::TextureFormat::Rgba8Unorm),
                make("dummy-u", 1, 1, wgpu::TextureFormat::R8Unorm),
                make("dummy-v", 1, 1, wgpu::TextureFormat::R8Unorm),
                4,
            ),
            PixelFormat::Bgra => (
                make("rgb", w, h, wgpu::TextureFormat::Bgra8Unorm),
                make("dummy-u", 1, 1, wgpu::TextureFormat::R8Unorm),
                make("dummy-v", 1, 1, wgpu::TextureFormat::R8Unorm),
                4,
            ),
            other => {
                return Err(Error::Element(format!(
                    "autovideosink/wgpu: unsupported display format {other:?}"
                )));
            }
        };

        self.queue.write_buffer(
            &self.params,
            0,
            &[mode as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );

        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("autovideosink-bind"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view(&y)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view(&u)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view(&v)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params.as_entire_binding(),
                },
            ],
        });

        self.textures = Some(PlaneTextures {
            key,
            y,
            u,
            v,
            bind_group,
            mode,
        });
        Ok(())
    }

    /// Upload this frame's planes into the bound textures.
    ///
    /// `write_texture` uploads STRIDED sources natively: `bytes_per_row`
    /// is the frame's real stride (`Metadata` plane layout, #194 — packed
    /// derived when none was declared), so External frames upload without
    /// any repack. `Queue::write_texture` has no row-alignment
    /// requirement.
    fn upload(&self, frame: &DisplayFrame) -> Result<()> {
        let textures = self.textures.as_ref().expect("ensure_textures ran");
        let data = frame.data.as_bytes();

        let uploads = plane_uploads(
            frame.format,
            frame.width,
            frame.height,
            frame.layout.as_ref(),
            data.len(),
        )?;
        let targets = [&textures.y, &textures.u, &textures.v];

        for (plane, tex) in uploads.iter().zip(targets) {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data[plane.offset..plane.end()],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(plane.bytes_per_row as u32),
                    rows_per_image: Some(plane.rows as u32),
                },
                wgpu::Extent3d {
                    width: plane.width_texels,
                    height: plane.rows as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
        // The mode rides the texture set; nothing else changes per frame.
        let _ = textures.mode;
        Ok(())
    }
}

/// One plane's slice geometry for `write_texture` — pure math, testable
/// without a GPU.
#[derive(Debug, PartialEq, Eq)]
struct PlaneUpload {
    /// Byte offset of the plane's first row in the frame data.
    offset: usize,
    /// Real row stride (becomes `bytes_per_row`).
    bytes_per_row: usize,
    /// Rows to copy.
    rows: usize,
    /// Used bytes in each row (the final row needs only this much data).
    row_bytes: usize,
    /// Texture extent width in texels (row_bytes / bytes-per-texel).
    width_texels: u32,
}

impl PlaneUpload {
    /// End of the byte range `write_texture` reads: full strides for all
    /// rows but the last, which needs only its used bytes.
    fn end(&self) -> usize {
        self.offset + self.bytes_per_row * (self.rows - 1) + self.row_bytes
    }
}

/// Resolve a display frame's planes against its (possibly strided) layout
/// into `write_texture` geometry, validating the data length. Plane order
/// matches the texture binding order: Y/U/V for I420, Y/UV for NV12, one
/// RGBA plane for RGB.
fn plane_uploads(
    format: PixelFormat,
    width: u32,
    height: u32,
    layout: Option<&crate::format::PlaneLayout>,
    data_len: usize,
) -> Result<Vec<PlaneUpload>> {
    // Bytes per texel per plane index, mirroring ensure_textures' formats.
    let texel_bytes: &[usize] = match format {
        PixelFormat::I420 => &[1, 1, 1],
        PixelFormat::Nv12 => &[1, 2],
        PixelFormat::Rgba | PixelFormat::Bgra => &[4],
        other => {
            return Err(Error::Element(format!(
                "autovideosink/wgpu: unsupported display format {other:?}"
            )));
        }
    };

    let packed;
    let layout = match layout {
        Some(l) => l,
        None => {
            packed = crate::format::PlaneLayout::packed(format, width, height);
            &packed
        }
    };

    let mut uploads = Vec::with_capacity(texel_bytes.len());
    for (plane, &texel) in layout.resolved(format, width, height).zip(texel_bytes) {
        if plane.stride < plane.row_bytes || plane.rows == 0 {
            return Err(Error::Element(format!(
                "autovideosink/wgpu: invalid plane layout (stride {} < row {})",
                plane.stride, plane.row_bytes
            )));
        }
        let upload = PlaneUpload {
            offset: plane.offset,
            bytes_per_row: plane.stride,
            rows: plane.rows,
            row_bytes: plane.row_bytes,
            width_texels: (plane.row_bytes / texel) as u32,
        };
        if upload.end() > data_len {
            return Err(Error::Element(format!(
                "autovideosink/wgpu: {format:?} frame is {data_len} bytes, plane needs {}",
                upload.end()
            )));
        }
        uploads.push(upload);
    }
    if uploads.len() != texel_bytes.len() {
        return Err(Error::Element(format!(
            "autovideosink/wgpu: {format:?} frame declares {} planes, expected {}",
            uploads.len(),
            texel_bytes.len()
        )));
    }
    Ok(uploads)
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl RenderBackend for WgpuBackend {
    fn render(&mut self, frame: &DisplayFrame, window_size: (u32, u32)) -> Result<()> {
        let (width, height) = window_size;
        if width == 0 || height == 0 {
            return Ok(());
        }
        if let Some((w, h)) = self.pending_size.take()
            && (w, h) != (self.config.width, self.config.height)
            && w > 0
            && h > 0
        {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }

        self.ensure_textures(frame)?;
        self.upload(frame)?;

        use wgpu::CurrentSurfaceTexture;
        let target = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(t) | CurrentSurfaceTexture::Suboptimal(t) => t,
            // A resize raced us (Wayland reports through both routes):
            // reconfigure at the current size and try once more. Never
            // fatal — the next frame gets another chance.
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                self.config.width = width.max(1);
                self.config.height = height.max(1);
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    CurrentSurfaceTexture::Success(t) | CurrentSurfaceTexture::Suboptimal(t) => t,
                    other => {
                        tracing::debug!("autovideosink/wgpu: acquire after reconfigure: {other:?}");
                        return Ok(());
                    }
                }
            }
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return Ok(()),
            CurrentSurfaceTexture::Validation => {
                return Err(Error::Element(
                    "wgpu surface acquire: validation error".into(),
                ));
            }
        };

        let view = target
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("autovideosink-present"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("autovideosink-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Black clear = the letterbox bars, for free.
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let (x0, y0, out_w, out_h) = letterbox_rect(
                frame.width as usize,
                frame.height as usize,
                self.config.width as usize,
                self.config.height as usize,
            );
            pass.set_pipeline(&self.pipeline);
            pass.set_viewport(x0 as f32, y0 as f32, out_w as f32, out_h as f32, 0.0, 1.0);
            let textures = self.textures.as_ref().expect("ensured above");
            pass.set_bind_group(0, &textures.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.queue.present(target);
        Ok(())
    }

    fn resized(&mut self, width: u32, height: u32) {
        self.pending_size = Some((width, height));
    }

    fn name(&self) -> &'static str {
        "wgpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{PlaneDesc, PlaneLayout};

    #[test]
    fn packed_i420_uploads_match_the_split_at_math() {
        // 64x48 packed I420: exactly the geometry the old split_at code
        // produced.
        let ups = plane_uploads(PixelFormat::I420, 64, 48, None, 64 * 48 * 3 / 2).unwrap();
        assert_eq!(ups.len(), 3);
        assert_eq!(
            (
                ups[0].offset,
                ups[0].bytes_per_row,
                ups[0].rows,
                ups[0].width_texels
            ),
            (0, 64, 48, 64)
        );
        assert_eq!(
            (
                ups[1].offset,
                ups[1].bytes_per_row,
                ups[1].rows,
                ups[1].width_texels
            ),
            (64 * 48, 32, 24, 32)
        );
        assert_eq!(ups[2].offset, 64 * 48 + 32 * 24);
    }

    #[test]
    fn strided_i420_uses_real_strides_and_offsets() {
        // dav1d-shaped layout: Y stride 128, chroma stride 64, planes
        // spaced with allocator padding.
        let layout = PlaneLayout::from_planes(&[
            PlaneDesc {
                offset: 0,
                stride: 128,
            },
            PlaneDesc {
                offset: 128 * 64,
                stride: 64,
            },
            PlaneDesc {
                offset: 128 * 64 + 64 * 32,
                stride: 64,
            },
        ]);
        let data_len = 128 * 64 + 2 * (64 * 32);
        let ups = plane_uploads(PixelFormat::I420, 64, 48, Some(&layout), data_len).unwrap();
        assert_eq!(ups[0].bytes_per_row, 128, "real stride, not width");
        assert_eq!(ups[0].width_texels, 64, "extent from real plane width");
        assert_eq!(ups[0].rows, 48);
        assert_eq!(ups[1].offset, 128 * 64);
        assert_eq!(ups[1].bytes_per_row, 64);
        assert_eq!(ups[1].width_texels, 32);
        // Last row needs only row_bytes, not a full stride.
        assert_eq!(ups[0].end(), 128 * 47 + 64);
    }

    #[test]
    fn nv12_interleaved_chroma_is_two_byte_texels() {
        let ups = plane_uploads(PixelFormat::Nv12, 64, 48, None, 64 * 48 * 3 / 2).unwrap();
        assert_eq!(ups.len(), 2);
        assert_eq!(ups[1].bytes_per_row, 64, "cw * 2 bytes");
        assert_eq!(ups[1].width_texels, 32, "Rg8 texels");
        assert_eq!(ups[1].rows, 24);
    }

    #[test]
    fn short_buffers_and_bad_strides_are_rejected() {
        assert!(plane_uploads(PixelFormat::I420, 64, 48, None, 100).is_err());
        let bad = PlaneLayout::from_planes(&[
            PlaneDesc {
                offset: 0,
                stride: 8,
            }, // < row_bytes 64
            PlaneDesc {
                offset: 512,
                stride: 32,
            },
            PlaneDesc {
                offset: 1024,
                stride: 32,
            },
        ]);
        assert!(plane_uploads(PixelFormat::I420, 64, 48, Some(&bad), 1 << 20).is_err());
    }
}
