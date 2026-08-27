//! The wgpu half: textures, the pipeline, and one draw per frame.
//!
//! Everything here compiles against **Iced's** wgpu, reached only through the
//! `iced::wgpu` re-export. Parallax's own `display-gpu` backend is on a
//! different major and the two cannot exchange types — what they share is
//! [`parallax::video`], which is arithmetic, a `&'static str` of WGSL and the
//! golden tests that pin them together.

use std::collections::HashMap;

use iced::wgpu;
use iced::widget::shader::{self, Viewport};
use iced::{Rectangle, Size};

use parallax::format::PixelFormat;
use parallax::video::{self, TexelFormat, VideoFrame, layout};

/// Frames the renderer keeps alive beyond the one it is drawing.
///
/// Zero on the upload path: `write_texture` copies into wgpu's staging belt
/// at call time, so the buffer may drop the moment `prepare` returns. The
/// import path (#269) will need a real hold, because an import aliases the
/// producer's memory rather than copying it.
pub const RENDERER_RETENTION: usize = 0;

/// Frames a texture set may go untouched before it is reclaimed.
const EVICT_AFTER: u64 = 120;

/// Iced's own texture-format map for our four texel formats.
fn texel(format: TexelFormat) -> wgpu::TextureFormat {
    match format {
        TexelFormat::R8 => wgpu::TextureFormat::R8Unorm,
        TexelFormat::Rg8 => wgpu::TextureFormat::Rg8Unorm,
        TexelFormat::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
        TexelFormat::Bgra8 => wgpu::TextureFormat::Bgra8Unorm,
    }
}

/// One video's texture set, rebuilt only when its geometry or format changes.
struct VideoTextures {
    key: (PixelFormat, u32, u32),
    /// Always three, because the shader always binds three. Unused planes are
    /// 1x1 padding, which is what keeps the bind-group layout fixed.
    planes: [wgpu::Texture; 3],
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    touched: u64,
}

/// The wgpu state every [`VideoPrimitive`] shares.
///
/// Iced caches exactly one of these per primitive type for the whole process,
/// so several video widgets share it and each keeps its own texture set under
/// its own id.
pub struct VideoPipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    videos: HashMap<u64, VideoTextures>,
    frame: u64,
}

impl std::fmt::Debug for VideoPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoPipeline")
            .field("videos", &self.videos.len())
            .finish()
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

impl shader::Pipeline for VideoPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("parallax-video-shader"),
            source: wgpu::ShaderSource::Wgsl(video::PRESENT_WGSL.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("parallax-video-bind-layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("parallax-video-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("parallax-video-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // A fullscreen triangle generated from the vertex index; the
                // widget's bounds are already the render pass's viewport.
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
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("parallax-video-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            layout,
            sampler,
            videos: HashMap::new(),
            frame: 0,
        }
    }

    /// Reclaim texture sets nobody has drawn for a while.
    ///
    /// This is what releases a torn-down pipeline's textures: the widget goes
    /// away, its id stops being touched, and a couple of seconds later its
    /// planes are freed.
    fn trim(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        let frame = self.frame;
        self.videos
            .retain(|_, v| frame.saturating_sub(v.touched) < EVICT_AFTER);
    }
}

impl VideoPipeline {
    /// Build (or reuse) the texture set for one video's current geometry.
    fn ensure(&mut self, device: &wgpu::Device, id: u64, frame: &VideoFrame) -> Result<(), String> {
        let key = (frame.format, frame.width, frame.height);
        if let Some(existing) = self.videos.get_mut(&id)
            && existing.key == key
        {
            existing.touched = self.frame;
            return Ok(());
        }

        let planes = layout::plane_formats(frame.format).map_err(|e| e.to_string())?;
        let (w, h) = (frame.width.max(1), frame.height.max(1));
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));

        let make = |label: &str, w: u32, h: u32, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
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
        // The shader always binds three textures; unused ones are 1x1 padding.
        let dummy = || make("parallax-video-dummy", 1, 1, wgpu::TextureFormat::R8Unorm);
        let planes: [wgpu::Texture; 3] = match frame.format {
            PixelFormat::I420 => [
                make("parallax-video-y", w, h, texel(planes[0])),
                make("parallax-video-u", cw, ch, texel(planes[1])),
                make("parallax-video-v", cw, ch, texel(planes[2])),
            ],
            PixelFormat::Nv12 => [
                make("parallax-video-y", w, h, texel(planes[0])),
                make("parallax-video-uv", cw, ch, texel(planes[1])),
                dummy(),
            ],
            _ => [
                make("parallax-video-rgb", w, h, texel(planes[0])),
                dummy(),
                dummy(),
            ],
        };

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("parallax-video-params"),
            size: video::PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("parallax-video-bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view(&planes[0])),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view(&planes[1])),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view(&planes[2])),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        self.videos.insert(
            id,
            VideoTextures {
                key,
                planes,
                params,
                bind_group,
                touched: self.frame,
            },
        );
        Ok(())
    }

    /// Upload one frame's planes into its textures.
    fn upload(&self, queue: &wgpu::Queue, id: u64, frame: &VideoFrame) -> Result<(), String> {
        let set = self.videos.get(&id).ok_or("no texture set")?;
        let data = frame.data.as_bytes();
        let uploads = layout::plane_uploads(
            frame.format,
            frame.width,
            frame.height,
            frame.layout.as_ref(),
            data.len(),
        )
        .map_err(|e| e.to_string())?;

        for (plane, texture) in uploads.iter().zip(set.planes.iter()) {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data[plane.offset..plane.end()],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // The producer's real stride, so a decoder's own strided
                    // picture (#194) uploads with no repack.
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
        Ok(())
    }
}

/// One frame, ready to draw.
#[derive(Debug)]
pub struct VideoPrimitive {
    /// Which texture set this frame belongs to.
    pub id: u64,
    /// The pixels. A clone is two atomic increments, so holding it here costs
    /// nothing and keeps the arena slot alive until the draw is submitted.
    ///
    /// `None` before the first frame arrives. The widget draws nothing then —
    /// Iced's pass loads rather than clears, so whatever the application
    /// painted underneath shows through, which is the right default for a
    /// player that has not opened a file yet.
    pub frame: Option<VideoFrame>,
    /// Whether the target expects linear values (an sRGB surface encodes for
    /// us, so the shader must undo the gamma it would otherwise double).
    pub srgb_target: bool,
}

impl shader::Primitive for VideoPrimitive {
    type Pipeline = VideoPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        let Some(frame) = &self.frame else {
            return;
        };
        if let Err(e) = pipeline.ensure(device, self.id, frame) {
            tracing::warn!("parallax-iced: {e}");
            return;
        }
        if let Err(e) = pipeline.upload(queue, self.id, frame) {
            tracing::warn!("parallax-iced: {e}");
            return;
        }

        // Letterbox in the shader rather than with a viewport: Iced's render
        // pass loads instead of clearing, so narrowing the viewport would
        // leave whatever was underneath showing in the bars.
        let scale = viewport.scale_factor();
        let (bw, bh) = (
            (bounds.width * scale).max(1.0),
            (bounds.height * scale).max(1.0),
        );
        let fit = fit_rect(
            Size::new(frame.width as f32, frame.height as f32),
            Size::new(bw, bh),
        );

        // `LAYOUT_RGB` is the harmless fallback: an unsupported format never
        // reaches here — `prepare` would have failed to build textures for it.
        let layout = video::shader_layout(frame.format).unwrap_or(video::LAYOUT_RGB);
        if let Some(set) = pipeline.videos.get(&self.id) {
            queue.write_buffer(
                &set.params,
                0,
                &video::shader_params(layout, self.srgb_target, fit, frame.resolved_color()),
            );
        }
    }

    fn draw(&self, pipeline: &Self::Pipeline, render_pass: &mut wgpu::RenderPass<'_>) -> bool {
        let Some(set) = self.frame.as_ref().and(pipeline.videos.get(&self.id)) else {
            return true;
        };
        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, &set.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        true
    }
}

/// Aspect-preserving fit, expressed the way the shader wants it: the
/// picture's rect inside the viewport in 0..1 space.
///
/// The integer arithmetic is [`parallax::video::fit::letterbox_rect`], the
/// same function `AutoVideoSink` uses, so both put the picture in the same
/// place.
fn fit_rect(src: Size, dst: Size) -> [f32; 4] {
    if src.width <= 0.0 || src.height <= 0.0 || dst.width <= 0.0 || dst.height <= 0.0 {
        return [0.0, 0.0, 1.0, 1.0];
    }
    let (x0, y0, w, h) = video::fit::letterbox_rect(
        src.width as usize,
        src.height as usize,
        dst.width as usize,
        dst.height as usize,
    );
    [
        x0 as f32 / dst.width,
        y0 as f32 / dst.height,
        w as f32 / dst.width,
        h as f32 / dst.height,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: [f32; 4], b: [f32; 4]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-3, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn a_matching_aspect_fills_the_widget() {
        approx(
            fit_rect(Size::new(1920.0, 1080.0), Size::new(1280.0, 720.0)),
            [0.0, 0.0, 1.0, 1.0],
        );
    }

    #[test]
    fn a_taller_widget_gets_bars_top_and_bottom() {
        // 16:9 into 16:10 — the same case `letterbox_rect`'s golden test uses.
        let fit = fit_rect(Size::new(1920.0, 1080.0), Size::new(1920.0, 1200.0));
        approx(fit, [0.0, 0.05, 1.0, 0.9]);
    }

    #[test]
    fn a_wider_widget_gets_bars_left_and_right() {
        let fit = fit_rect(Size::new(1000.0, 1000.0), Size::new(2000.0, 1000.0));
        approx(fit, [0.25, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn a_degenerate_size_falls_back_to_the_whole_widget() {
        // Before layout has run, bounds can legitimately be zero — dividing
        // by that inside `letterbox_rect` would panic.
        approx(
            fit_rect(Size::new(0.0, 0.0), Size::new(100.0, 100.0)),
            [0.0, 0.0, 1.0, 1.0],
        );
        approx(
            fit_rect(Size::new(100.0, 100.0), Size::new(0.0, 0.0)),
            [0.0, 0.0, 1.0, 1.0],
        );
    }
}
