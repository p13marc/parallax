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
    /// Decoders emit tightly-packed planes (dav1d/vpx de-stride on
    /// copy-out), so `bytes_per_row` is exactly the plane width —
    /// `Queue::write_texture` has no row-alignment requirement.
    fn upload(&self, frame: &DisplayFrame) -> Result<()> {
        let textures = self.textures.as_ref().expect("ensure_textures ran");
        let data = frame.data.as_bytes();
        let (w, h) = (frame.width as usize, frame.height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));

        let write = |tex: &wgpu::Texture, bytes: &[u8], row: usize, rows: usize| {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row as u32),
                    rows_per_image: Some(rows as u32),
                },
                wgpu::Extent3d {
                    width: (row
                        / match tex.format() {
                            wgpu::TextureFormat::Rg8Unorm => 2,
                            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm => 4,
                            _ => 1,
                        }) as u32,
                    height: rows as u32,
                    depth_or_array_layers: 1,
                },
            );
        };

        match frame.format {
            PixelFormat::I420 => {
                let need = w * h + 2 * (cw * ch);
                if data.len() < need {
                    return Err(Error::Element(format!(
                        "autovideosink/wgpu: I420 frame is {} bytes, needs {need}",
                        data.len()
                    )));
                }
                let (y, rest) = data.split_at(w * h);
                let (u, rest) = rest.split_at(cw * ch);
                let v = &rest[..cw * ch];
                write(&textures.y, y, w, h);
                write(&textures.u, u, cw, ch);
                write(&textures.v, v, cw, ch);
            }
            PixelFormat::Nv12 => {
                let need = w * h + 2 * (cw * ch);
                if data.len() < need {
                    return Err(Error::Element(format!(
                        "autovideosink/wgpu: NV12 frame is {} bytes, needs {need}",
                        data.len()
                    )));
                }
                let (y, rest) = data.split_at(w * h);
                let uv = &rest[..2 * cw * ch];
                write(&textures.y, y, w, h);
                write(&textures.u, uv, cw * 2, ch);
            }
            PixelFormat::Rgba | PixelFormat::Bgra => {
                let need = w * h * 4;
                if data.len() < need {
                    return Err(Error::Element(format!(
                        "autovideosink/wgpu: RGB frame is {} bytes, needs {need}",
                        data.len()
                    )));
                }
                write(&textures.y, &data[..need], w * 4, h);
            }
            _ => unreachable!("ensure_textures rejected it"),
        }
        // The mode rides the texture set; nothing else changes per frame.
        let _ = textures.mode;
        Ok(())
    }
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
