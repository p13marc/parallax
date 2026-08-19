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
/// What the one-shot adapter probe learned.
#[derive(Clone, Copy)]
struct GpuProbe {
    /// A real (non-software) adapter exists.
    usable: bool,
    /// ...and it can import dma-bufs, so the sink may advertise `DmaBuf`.
    dmabuf_import: bool,
}

pub(crate) fn gpu_available() -> bool {
    probe().usable
}

/// Whether a dma-buf frame can be imported rather than uploaded.
///
/// Strictly narrower than [`gpu_available`]: a GL or software-Vulkan
/// adapter presents fine and cannot import anything, and advertising
/// `DmaBuf` on that basis would strand a producer with frames the sink
/// cannot read. `PARALLAX_NO_DMABUF_IMPORT=1` forces the upload path so the
/// two can be compared on one machine without rebuilding.
pub(crate) fn dmabuf_import_available() -> bool {
    probe().dmabuf_import
}

fn probe() -> GpuProbe {
    static PROBE: OnceLock<GpuProbe> = OnceLock::new();
    *PROBE.get_or_init(|| {
        if std::env::var_os("PARALLAX_NO_GPU").is_some_and(|v| v != "0") {
            tracing::info!("autovideosink: GPU presentation disabled (PARALLAX_NO_GPU)");
            return GpuProbe {
                usable: false,
                dmabuf_import: false,
            };
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
                let no_import =
                    std::env::var_os("PARALLAX_NO_DMABUF_IMPORT").is_some_and(|v| v != "0");
                let dmabuf_import =
                    usable && !no_import && super::dmabuf_import::import_supported(&adapter);
                tracing::info!(
                    "autovideosink: GPU probe found {:?} ({:?}, {:?}) — {}{}",
                    info.name,
                    info.device_type,
                    info.backend,
                    if usable {
                        "using it"
                    } else {
                        "software rasterizer, keeping softbuffer"
                    },
                    if dmabuf_import {
                        ", dma-buf import available"
                    } else {
                        ""
                    }
                );
                GpuProbe {
                    usable,
                    dmabuf_import,
                }
            }
            Err(e) => {
                tracing::info!("autovideosink: no GPU adapter ({e}); keeping softbuffer");
                GpuProbe {
                    usable: false,
                    dmabuf_import: false,
                }
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
    /// Whether this device can import dma-bufs (feature requested and got).
    dmabuf_import: bool,
    /// Imported frames, keyed by the producer allocation they alias.
    ///
    /// A pooled producer hands out a fresh slot per frame over a *stable*
    /// segment, so importing per frame would mean a `dup`, a
    /// `vkAllocateMemory` and a `vkCreateImage` every frame — most of the
    /// syscall cost zero-copy exists to remove. The segment is the identity
    /// of the underlying allocation, and holding an `Arc` of it is what
    /// stops that identity being reused under the key.
    imported: Vec<ImportedFrame>,
    /// Frames whose submission may still be reading their memory.
    ///
    /// With an upload the bytes are copied at `write_texture` time and the
    /// producer's buffer can go immediately. With an import the GPU samples
    /// the producer's memory during the pass, so releasing the buffer — and
    /// with it the pool slot the decoder would refill — before the
    /// submission retires would let the next frame be decoded over the one
    /// being drawn.
    in_flight: std::collections::VecDeque<(wgpu::SubmissionIndex, crate::buffer::Buffer)>,
    /// Monotonic tick for the import cache's LRU order.
    clock: u64,
}

/// How many submissions may be outstanding before the oldest frame's memory
/// is released. Two is enough to never stall at vsync pacing while keeping
/// the wait exact rather than hopeful.
const GPU_IN_FLIGHT: usize = 2;

/// Textures aliasing one producer allocation, plus its ready bind group.
struct ImportedFrame {
    /// Identity of the aliased allocation.
    key: usize,
    /// Geometry, since a resize reuses the allocation for a different shape.
    geometry: (PixelFormat, u32, u32),
    /// Keeps the allocation's identity alive so `key` cannot be recycled.
    _segment: std::sync::Arc<crate::memory::DmaBufSegment>,
    /// Kept alive for as long as the bind group refers to them.
    _planes: Vec<wgpu::Texture>,
    bind_group: wgpu::BindGroup,
    mode: u32,
    /// Bumped on use; the least recently used entry is evicted first.
    used: u64,
}

/// Imported allocations kept at once. A producer pool is 4-8 frames.
const IMPORT_CACHE: usize = 8;

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
        // Re-ask the adapter rather than trusting the probe: the probe's
        // adapter is surfaceless and on a multi-GPU box need not be this one.
        let want_import =
            dmabuf_import_available() && super::dmabuf_import::import_supported(&adapter);
        let descriptor = wgpu::DeviceDescriptor {
            required_features: if want_import {
                wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF
            } else {
                wgpu::Features::empty()
            },
            ..Default::default()
        };
        let (device, queue, dmabuf_import) =
            match pollster::block_on(adapter.request_device(&descriptor)) {
                Ok((d, q)) => (d, q, want_import),
                // A feature bit must never cost us the whole GPU backend.
                Err(e) if want_import => {
                    tracing::warn!(
                        "autovideosink: device without dma-buf import ({e}); uploading instead"
                    );
                    let (d, q) = pollster::block_on(
                        adapter.request_device(&wgpu::DeviceDescriptor::default()),
                    )
                    .map_err(|e| Error::Element(format!("wgpu device: {e}")))?;
                    (d, q, false)
                }
                Err(e) => return Err(Error::Element(format!("wgpu device: {e}"))),
            };

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
            dmabuf_import,
            imported: Vec::new(),
            in_flight: std::collections::VecDeque::new(),
            clock: 0,
        })
    }

    /// The shader mode for a frame's format and height.
    fn shader_mode(format: PixelFormat, height: u32) -> Result<u32> {
        let matrix = color::matrix_for_height(height);
        Ok(match (format, matrix) {
            (PixelFormat::I420, color::ColorMatrix::Bt709) => 0,
            (PixelFormat::I420, color::ColorMatrix::Bt601) => 1,
            (PixelFormat::Nv12, color::ColorMatrix::Bt709) => 2,
            (PixelFormat::Nv12, color::ColorMatrix::Bt601) => 3,
            (PixelFormat::Rgba | PixelFormat::Bgra, _) => 4,
            (other, _) => {
                return Err(Error::Element(format!(
                    "autovideosink/wgpu: unsupported display format {other:?}"
                )));
            }
        })
    }

    /// Import this frame's dma-buf as textures, or reuse an earlier import
    /// of the same allocation.
    ///
    /// Returns `Ok(false)` when the frame is not importable at all — no
    /// dma-buf, no declared plane layout, import unavailable — and the
    /// caller uploads instead. An import that *fails* is different: it is
    /// reported, import is switched off for this backend, and the upload
    /// path takes over permanently rather than failing every frame.
    fn prepare_imported(&mut self, frame: &DisplayFrame) -> Result<bool> {
        if !self.dmabuf_import {
            return Ok(false);
        }
        let Some(slot) = frame.data.memory().dmabuf_slot() else {
            return Ok(false);
        };
        // Without a layout there is nothing to import *by*: the offsets and
        // pitches are the whole content of the import.
        let Some(layout) = frame.layout.as_ref() else {
            return Ok(false);
        };

        let key = std::sync::Arc::as_ptr(slot.shared_segment()) as usize;
        let geometry = (frame.format, frame.width, frame.height);
        self.clock += 1;
        if let Some(entry) = self
            .imported
            .iter_mut()
            .find(|e| e.key == key && e.geometry == geometry)
        {
            entry.used = self.clock;
            let mode = entry.mode;
            self.queue.write_buffer(
                &self.params,
                0,
                &[mode as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            );
            return Ok(true);
        }

        let planes = match super::dmabuf_import::plane_imports(
            frame.format,
            frame.width,
            frame.height,
            layout,
            frame.data.memory().offset(),
        ) {
            Ok(p) => p,
            Err(e) => return self.disable_import(e),
        };

        let mut textures = Vec::with_capacity(planes.len());
        for plane in &planes {
            // SAFETY: the layout and modifier come from the producer that
            // allocated this dma-buf, which is the only thing that knows them.
            let imported = unsafe {
                super::dmabuf_import::import_plane(
                    &self.device,
                    slot.fd(),
                    plane,
                    slot.modifier(),
                    // Sampled only: the producer owns these bytes.
                    wgpu::TextureUsages::TEXTURE_BINDING,
                    "autovideosink-imported-plane",
                )
            };
            match imported {
                Ok(t) => textures.push(t),
                Err(e) => return self.disable_import(e),
            }
        }

        // The shader always binds three textures; pad with 1x1 dummies.
        let dummy = |label: &str| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        while textures.len() < 3 {
            textures.push(dummy("import-dummy"));
        }

        let mode = Self::shader_mode(frame.format, frame.height)?;
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("autovideosink-imported-bind"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view(&textures[0])),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view(&textures[1])),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view(&textures[2])),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params.as_entire_binding(),
                },
            ],
        });
        self.queue.write_buffer(
            &self.params,
            0,
            &[mode as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );

        if self.imported.len() >= IMPORT_CACHE
            && let Some(oldest) = self
                .imported
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i)
        {
            self.imported.swap_remove(oldest);
        }
        if self.imported.is_empty() {
            tracing::info!(
                "autovideosink: importing {:?} frames from the producer's dma-buf \
                 (modifier {:#x}) — no upload",
                frame.format,
                slot.modifier()
            );
        }
        self.imported.push(ImportedFrame {
            key,
            geometry,
            _segment: std::sync::Arc::clone(slot.shared_segment()),
            _planes: textures,
            bind_group,
            mode,
            used: self.clock,
        });
        // The upload path's textures are now stale for this frame; drop them
        // so a later CPU frame rebuilds rather than showing old pixels.
        self.textures = None;
        Ok(true)
    }

    /// Give up on importing and say why, once.
    fn disable_import(&mut self, e: Error) -> Result<bool> {
        tracing::warn!("autovideosink: dma-buf import failed ({e}); uploading instead");
        self.dmabuf_import = false;
        self.imported.clear();
        Ok(false)
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

        // Zero-copy first: an imported frame needs no textures of ours and
        // no upload at all.
        let imported = self.prepare_imported(frame)?;
        if !imported {
            self.ensure_textures(frame)?;
            self.upload(frame)?;
        }

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
            let bind_group = if imported {
                &self
                    .imported
                    .iter()
                    .max_by_key(|e| e.used)
                    .expect("just imported")
                    .bind_group
            } else {
                &self.textures.as_ref().expect("ensured above").bind_group
            };
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        let submission = self.queue.submit(Some(encoder.finish()));
        self.queue.present(target);

        if imported {
            // The pass samples the producer's memory, so its buffer — and
            // through it the pool slot the producer would refill — must
            // outlive the submission. Waiting on the *specific* submission
            // is exact; at this depth and vsync pacing it has already
            // retired by the time we ask.
            self.in_flight.push_back((submission, frame.data.clone()));
            while self.in_flight.len() > GPU_IN_FLIGHT {
                let (index, buffer) = self.in_flight.pop_front().expect("checked len");
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: Some(std::time::Duration::from_millis(100)),
                });
                drop(buffer);
            }
        }
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
