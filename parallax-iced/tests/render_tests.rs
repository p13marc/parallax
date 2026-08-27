//! Render the widget offscreen and look at the pixels.
//!
//! No window, no Iced runtime: `VideoPipeline` and `VideoPrimitive` are the
//! real API, so this drives them directly against whatever adapter exists —
//! including a software one, which is what makes it runnable in CI. Skips
//! (passes with a message) where wgpu finds no adapter at all.
//!
//! What it pins is the part that fails silently: a wrong colour or a picture
//! in the wrong place looks like a picture.

use std::sync::Arc;

use iced::wgpu;
use iced::widget::shader::{Pipeline, Primitive, Viewport};
use iced::{Rectangle, Size};

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::format::PixelFormat;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::video::{VideoFrame, color};
use parallax_iced::{VideoPipeline, VideoPrimitive};

const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A device, or `None` → skip. Software adapters are welcome: this is a
/// correctness test, not a performance one.
fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }));
    let adapter = match adapter {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skipping: no wgpu adapter ({e})");
            return None;
        }
    };
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("parallax-iced-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    })) {
        Ok(pair) => Some(pair),
        Err(e) => {
            eprintln!("skipping: no wgpu device ({e})");
            None
        }
    }
}

/// A packed I420 frame of one flat colour.
fn i420(width: u32, height: u32, y: u8, cb: u8, cr: u8) -> VideoFrame {
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let luma = (width * height) as usize;
    let chroma = (cw * ch) as usize;
    let len = luma + 2 * chroma;

    let arena = SharedArena::new(len.next_power_of_two().max(4096), 2).expect("arena");
    let mut slot = arena.acquire().expect("slot");
    {
        let data = &mut slot.data_mut()[..len];
        data[..luma].fill(y);
        data[luma..luma + chroma].fill(cb);
        data[luma + chroma..len].fill(cr);
    }
    let mut meta = Metadata::default();
    meta.set_video_dims(width, height, PixelFormat::I420);
    VideoFrame::from_buffer(&Buffer::new(MemoryHandle::with_len(slot, len), meta))
        .expect("a well-formed I420 frame")
}

/// Draw one primitive into a `size`-square RGBA target and read it back.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    primitive: &VideoPrimitive,
    size: u32,
) -> Vec<[u8; 4]> {
    let mut pipeline = VideoPipeline::new(device, queue, TARGET);
    let bounds = Rectangle::new(iced::Point::ORIGIN, Size::new(size as f32, size as f32));
    let viewport = Viewport::with_physical_size(Size::new(size, size), 1.0);
    primitive.prepare(&mut pipeline, device, queue, &bounds, &viewport);

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // `bytes_per_row` must be a multiple of 256 for a texture-to-buffer copy.
    let row = (size as usize * 4).next_multiple_of(256);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * size as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Iced loads rather than clears, which is exactly why the
                    // widget paints its own letterbox bars. Load here too, so
                    // a bar that is *not* painted shows up as this magenta.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 0.0,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_viewport(0.0, 0.0, size as f32, size as f32, 0.0, 1.0);
        assert!(primitive.draw(&pipeline, &mut pass), "the primitive drew");
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row as u32),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("gpu work retires");
    let mapped = slice.get_mapped_range();
    let mut out = Vec::with_capacity((size * size) as usize);
    for y in 0..size as usize {
        for x in 0..size as usize {
            let at = y * row + x * 4;
            out.push([mapped[at], mapped[at + 1], mapped[at + 2], mapped[at + 3]]);
        }
    }
    out
}

fn at(pixels: &[[u8; 4]], size: u32, x: u32, y: u32) -> [u8; 4] {
    pixels[(y * size + x) as usize]
}

/// Limited-range black and white must land on the rails, and mid-grey must
/// land where the transfer function says — which is the whole sRGB question.
///
/// Iced renders into an sRGB surface, so the shader hands it *linear* values
/// and lets the hardware encode. Skipping that step leaves the encoded value
/// where the linear one belongs and the picture washes out; the two flags
/// below differ by exactly that.
#[test]
fn limited_range_lands_on_the_rails_and_grey_is_linearised() {
    let Some((device, queue)) = device() else {
        return;
    };
    const SIZE: u32 = 16;

    let black = VideoPrimitive {
        id: 1,
        frame: Some(i420(SIZE, SIZE, 16, 128, 128)),
        srgb_target: false,
    };
    let px = render(&device, &queue, &black, SIZE);
    assert_eq!(at(&px, SIZE, 8, 8), [0, 0, 0, 255], "limited-range black");

    let white = VideoPrimitive {
        id: 2,
        frame: Some(i420(SIZE, SIZE, 235, 128, 128)),
        srgb_target: false,
    };
    let px = render(&device, &queue, &white, SIZE);
    assert_eq!(
        at(&px, SIZE, 8, 8),
        [255, 255, 255, 255],
        "limited-range white"
    );

    // Mid-grey, the value that makes a double encode obvious. Y=126 sits
    // near the middle of 16..235.
    let encoded = VideoPrimitive {
        id: 3,
        frame: Some(i420(SIZE, SIZE, 126, 128, 128)),
        srgb_target: false,
    };
    let raw = at(&render(&device, &queue, &encoded, SIZE), SIZE, 8, 8)[0];

    let linear = VideoPrimitive {
        id: 4,
        frame: Some(i420(SIZE, SIZE, 126, 128, 128)),
        srgb_target: true,
    };
    let linearised = at(&render(&device, &queue, &linear, SIZE), SIZE, 8, 8)[0];

    assert!(
        linearised < raw,
        "linearising must darken the stored value ({linearised} vs {raw})"
    );
    // And it must darken it by exactly the amount `video::color` says, which
    // is what keeps the shader's copy of the curve honest.
    let expected = (color::srgb_to_linear(raw as f32 / 255.0) * 255.0).round() as i32;
    assert!(
        (linearised as i32 - expected).abs() <= 2,
        "shader gave {linearised}, video::color says {expected}"
    );
}

/// A frame whose aspect does not match the widget gets black bars, painted by
/// the shader rather than left as whatever was underneath.
#[test]
fn the_letterbox_bars_are_painted_not_left_behind() {
    let Some((device, queue)) = device() else {
        return;
    };
    const SIZE: u32 = 32;

    // 2:1 content in a square widget: bars top and bottom, a quarter each.
    let wide = VideoPrimitive {
        id: 5,
        frame: Some(i420(64, 32, 235, 128, 128)),
        srgb_target: false,
    };
    let px = render(&device, &queue, &wide, SIZE);

    assert_eq!(at(&px, SIZE, 16, 1), [0, 0, 0, 255], "top bar is black");
    assert_eq!(
        at(&px, SIZE, 16, SIZE - 2),
        [0, 0, 0, 255],
        "bottom bar is black"
    );
    assert_eq!(
        at(&px, SIZE, 16, SIZE / 2),
        [255, 255, 255, 255],
        "the picture is in the middle"
    );
}

/// Before a frame arrives there is nothing to draw, and the widget must leave
/// the target alone rather than paint over the application's own background.
#[test]
fn an_empty_widget_draws_nothing() {
    let Some((device, queue)) = device() else {
        return;
    };
    const SIZE: u32 = 8;

    let empty = VideoPrimitive {
        id: 6,
        frame: None,
        srgb_target: true,
    };
    let px = render(&device, &queue, &empty, SIZE);
    assert_eq!(
        at(&px, SIZE, 4, 4),
        [255, 0, 255, 255],
        "the clear colour survived"
    );
}

/// The mailbox's id is what keys a texture set, so two videos in one process
/// cannot end up sharing one.
#[test]
fn two_videos_keep_their_own_textures() {
    let Some((device, queue)) = device() else {
        return;
    };
    const SIZE: u32 = 8;
    let mut pipeline = VideoPipeline::new(&device, &queue, TARGET);
    let bounds = Rectangle::new(iced::Point::ORIGIN, Size::new(SIZE as f32, SIZE as f32));
    let viewport = Viewport::with_physical_size(Size::new(SIZE, SIZE), 1.0);

    for (id, y) in [(10u64, 16u8), (11, 235)] {
        let p = VideoPrimitive {
            id,
            frame: Some(i420(SIZE, SIZE, y, 128, 128)),
            srgb_target: false,
        };
        p.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);
    }
    // Both survived: re-preparing the first must not have to rebuild.
    let first = VideoPrimitive {
        id: 10,
        frame: Some(i420(SIZE, SIZE, 16, 128, 128)),
        srgb_target: false,
    };
    first.prepare(&mut pipeline, &device, &queue, &bounds, &viewport);
    let debug = format!("{pipeline:?}");
    assert!(debug.contains("videos: 2"), "{debug}");
}

/// Keep the `Arc` import honest: a real host clones the mailbox around.
#[allow(dead_code)]
fn _mailbox_is_shareable(m: Arc<parallax::video::FrameMailbox>) -> u64 {
    m.id()
}
