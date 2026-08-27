//! Presentation maths, with no graphics dependency.
//!
//! Everything a renderer needs to turn a decoded [`Buffer`](crate::buffer::Buffer)
//! into a picture — plane strides, the letterbox rect, the YUV matrices, the
//! shader itself — and nothing that names a graphics type.
//!
//! That constraint is the whole point. Parallax's own `AutoVideoSink` renders
//! through one `wgpu` major; a host application embedding parallax in a GUI
//! toolkit renders through whichever major that toolkit pins, and the two
//! cannot exchange types. What they *can* share is arithmetic, a `&'static str`
//! of WGSL, and the golden tests that pin them together — so that lives here,
//! unfeatured, and both sides map [`TexelFormat`] to their own enum.
//!
//! ```no_run
//! use parallax::video::{VideoFrame, fit, layout};
//! # fn demo(buffer: &parallax::buffer::Buffer) -> parallax::error::Result<()> {
//! let frame = VideoFrame::from_buffer(buffer)?;
//! let planes = layout::plane_uploads(
//!     frame.format,
//!     frame.width,
//!     frame.height,
//!     frame.layout.as_ref(),
//!     frame.data.len(),
//! )?;
//! let (x0, y0, w, h) =
//!     fit::letterbox_rect(frame.width as usize, frame.height as usize, 1280, 720);
//! # let _ = (planes, x0, y0, w, h);
//! # Ok(())
//! # }
//! ```

pub mod color;
pub mod fit;
pub mod layout;

mod frame;
mod mailbox;

pub use frame::{FrameView, VideoFrame};
pub use layout::TexelFormat;
pub use mailbox::{
    DEFAULT_QUEUE_DEPTH, FrameAcceptance, FrameMailbox, FrameStats, PacedFrame, PresentationClock,
};

use crate::error::{Error, Result};
use crate::format::PixelFormat;

/// The presentation shader, shared verbatim by every backend.
///
/// Plain WGSL: no extensions, no `f16`, one bind group of five entries. That
/// is what lets one string compile under two different `naga` majors, and it
/// is worth keeping true — see the module docs.
///
/// Bindings are `0` sampler, `1` luma (or the packed RGB plane), `2` Cb (or
/// NV12's interleaved chroma), `3` Cr, `4` the [`shader_params`] uniform.
pub const PRESENT_WGSL: &str = include_str!("present.wgsl");

/// Which plane arrangement [`PRESENT_WGSL`] should sample.
///
/// Planar (three single-channel planes), semi-planar (luma plus interleaved
/// chroma) or packed RGB — and *nothing else*. Colour is not encoded here:
/// it rides [`shader_params`] as numbers. That separation is the point. The
/// old `mode` enum multiplied layout by matrix into one five-value table, and
/// the shader discriminated planes with `mode >= 2u`, so adding a range axis
/// would have silently changed which texture the chroma came from.
pub const LAYOUT_PLANAR: u32 = 0;
/// See [`LAYOUT_PLANAR`]: luma plane plus an interleaved chroma plane (NV12).
pub const LAYOUT_SEMIPLANAR: u32 = 1;
/// See [`LAYOUT_PLANAR`]: one packed RGBA/BGRA texture, no conversion.
pub const LAYOUT_RGB: u32 = 2;

/// The `params.layout` value [`PRESENT_WGSL`] expects for a pixel format.
///
/// One function owns this mapping: it used to be written out three times —
/// in the wgpu backend's mode lookup, again inline where it built its
/// textures, and a third time in the dma-buf import's plane table — which is
/// three chances for a picture to come out green.
pub fn shader_layout(format: PixelFormat) -> Result<u32> {
    Ok(match format {
        PixelFormat::I420 => LAYOUT_PLANAR,
        PixelFormat::Nv12 => LAYOUT_SEMIPLANAR,
        PixelFormat::Rgba | PixelFormat::Bgra => LAYOUT_RGB,
        other => {
            return Err(Error::Element(format!(
                "video: unsupported display format {other:?}"
            )));
        }
    })
}

/// Size of the uniform block [`PRESENT_WGSL`] declares, in bytes.
pub const PARAMS_SIZE: u64 = 64;

/// Build the uniform block for one frame.
///
/// One function owns the layout because two backends write it, on two
/// different graphics stacks, and a struct they cannot share is a struct that
/// will drift. `fit` is `(x0, y0, width, height)` of the picture inside the
/// viewport in 0..1 space — `(0, 0, 1, 1)` for a backend that letterboxes
/// with its own viewport and a black clear.
///
/// `color` should already be resolved (#265): an `Unspecified` matrix falls
/// back to BT.601 here, which is the historical assumption but a worse answer
/// than the caller's own — [`VideoFrame::resolved_color`] is what the
/// backends in this tree pass. `rotation` (#266) is the turn `fit` was
/// already computed for: the shader inverts it when sampling, so the two must
/// come from the same [`DisplayGeometry`](crate::format::DisplayGeometry).
pub fn shader_params(
    layout: u32,
    srgb_target: bool,
    fit: [f32; 4],
    color: crate::format::ColorInfo,
    rotation: crate::format::Rotation,
) -> [u8; PARAMS_SIZE as usize] {
    let mut out = [0u8; PARAMS_SIZE as usize];
    out[0..4].copy_from_slice(&layout.to_le_bytes());
    out[4..8].copy_from_slice(&u32::from(srgb_target).to_le_bytes());
    // Quarter-turns (#266). This lives in one of the two words WGSL needed
    // as padding to put `fit` on a 16-byte boundary, so it cost nothing.
    out[8..12].copy_from_slice(&u32::from(rotation as u8).to_le_bytes());
    let mut put = |at: usize, v: [f32; 4]| {
        for (i, x) in v.iter().enumerate() {
            out[at + i * 4..at + i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
    };
    put(16, fit);
    put(32, color::coefficients(color.matrix));
    // The shader works in 0..1, so the luma offset is normalised here rather
    // than there — one fewer constant for the WGSL to get wrong.
    let (y_offset, y_scale, c_scale) = color::expansion(color.range);
    put(48, [y_offset / 255.0, y_scale, c_scale, 0.0]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_is_the_one_the_backends_compile() {
        // Cheap canary: the bindings, the uniform the Rust side writes and
        // the module that owns its constants are actually in the file.
        assert!(PRESENT_WGSL.contains("@group(0) @binding(4) var<uniform> params: Params;"));
        assert!(PRESENT_WGSL.contains("fn vs_main"));
        assert!(PRESENT_WGSL.contains("fn fs_main"));
        // `layout` is a WGSL reserved keyword and `planes` is what the field
        // had to be called; a rename back would compile in Rust and fail only
        // on a machine with a GPU.
        assert!(PRESENT_WGSL.contains("    planes: u32,"));
        assert!(PRESENT_WGSL.contains("    coeffs: vec4<f32>,"));
        assert!(PRESENT_WGSL.contains("    expansion: vec4<f32>,"));
        assert!(PRESENT_WGSL.contains("    rotation: u32,"));
        assert!(
            PRESENT_WGSL.contains("src/video/color.rs"),
            "the shader must point at the module that owns its constants"
        );
    }

    #[test]
    fn layouts_match_the_shaders_table() {
        assert_eq!(shader_layout(PixelFormat::I420).unwrap(), LAYOUT_PLANAR);
        assert_eq!(shader_layout(PixelFormat::Nv12).unwrap(), LAYOUT_SEMIPLANAR);
        assert_eq!(shader_layout(PixelFormat::Rgba).unwrap(), LAYOUT_RGB);
        assert_eq!(shader_layout(PixelFormat::Bgra).unwrap(), LAYOUT_RGB);
        assert!(shader_layout(PixelFormat::Yuyv).is_err());
    }

    #[test]
    fn the_uniform_block_carries_the_colour_the_caller_asked_for() {
        use crate::format::{ColorInfo, ColorMatrix, ColorRange};

        let read =
            |bytes: &[u8], at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());

        let hd = ColorInfo {
            matrix: ColorMatrix::Bt709,
            range: ColorRange::Limited,
            ..ColorInfo::default()
        };
        let p = shader_params(
            LAYOUT_PLANAR,
            false,
            [0.0, 0.0, 1.0, 1.0],
            hd,
            crate::format::Rotation::None,
        );
        assert_eq!(p.len(), PARAMS_SIZE as usize);
        for (i, c) in color::coefficients(ColorMatrix::Bt709).iter().enumerate() {
            assert_eq!(read(&p, 32 + i * 4), *c);
        }
        assert_eq!(read(&p, 48), 16.0 / 255.0);
        assert_eq!(read(&p, 52), color::Y_SCALE);
        assert_eq!(read(&p, 56), color::C_SCALE);

        // Full range: no offset, no expansion. The bytes are the whole
        // difference between the two pictures, which is what makes this
        // testable at all without a GPU.
        let full = ColorInfo {
            range: ColorRange::Full,
            ..hd
        };
        let p = shader_params(
            LAYOUT_PLANAR,
            true,
            [0.0, 0.0, 1.0, 1.0],
            full,
            crate::format::Rotation::Cw270,
        );
        assert_eq!(u32::from_le_bytes(p[4..8].try_into().unwrap()), 1);
        // Quarter-turns, in the word that used to be padding.
        assert_eq!(u32::from_le_bytes(p[8..12].try_into().unwrap()), 3);
        assert_eq!(read(&p, 48), 0.0);
        assert_eq!(read(&p, 52), 1.0);
        assert_eq!(read(&p, 56), 1.0);
    }
}
