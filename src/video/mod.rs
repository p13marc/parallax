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
/// NV12's interleaved chroma), `3` Cr, `4` the [`shader_mode`] uniform.
pub const PRESENT_WGSL: &str = include_str!("present.wgsl");

/// The `params.mode` value [`PRESENT_WGSL`] expects for a format and matrix.
///
/// One function owns this mapping: it used to be written out three times —
/// in the wgpu backend's mode lookup, again inline where it built its
/// textures, and a third time in the dma-buf import's plane table — which is
/// three chances for a picture to come out green.
pub fn shader_mode(format: PixelFormat, matrix: color::ColorMatrix) -> Result<u32> {
    use color::ColorMatrix::{Bt601, Bt709};
    Ok(match (format, matrix) {
        (PixelFormat::I420, Bt709) => 0,
        (PixelFormat::I420, Bt601) => 1,
        (PixelFormat::Nv12, Bt709) => 2,
        (PixelFormat::Nv12, Bt601) => 3,
        (PixelFormat::Rgba | PixelFormat::Bgra, _) => 4,
        (other, _) => {
            return Err(Error::Element(format!(
                "video: unsupported display format {other:?}"
            )));
        }
    })
}

/// [`shader_mode`] with the matrix chosen by [`color::matrix_for_height`].
pub fn shader_mode_for(format: PixelFormat, height: u32) -> Result<u32> {
    shader_mode(format, color::matrix_for_height(height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_is_the_one_the_backends_compile() {
        // Cheap canary: the bindings and the mode table the Rust side
        // assumes are actually in the file.
        assert!(PRESENT_WGSL.contains("@group(0) @binding(4) var<uniform> params: Params;"));
        assert!(PRESENT_WGSL.contains("fn vs_main"));
        assert!(PRESENT_WGSL.contains("fn fs_main"));
        assert!(
            PRESENT_WGSL.contains("src/video/color.rs"),
            "the shader must point at the module that owns its constants"
        );
    }

    #[test]
    fn modes_match_the_shaders_table() {
        use color::ColorMatrix::{Bt601, Bt709};
        assert_eq!(shader_mode(PixelFormat::I420, Bt709).unwrap(), 0);
        assert_eq!(shader_mode(PixelFormat::I420, Bt601).unwrap(), 1);
        assert_eq!(shader_mode(PixelFormat::Nv12, Bt709).unwrap(), 2);
        assert_eq!(shader_mode(PixelFormat::Nv12, Bt601).unwrap(), 3);
        assert_eq!(shader_mode(PixelFormat::Rgba, Bt601).unwrap(), 4);
        assert_eq!(shader_mode(PixelFormat::Bgra, Bt709).unwrap(), 4);
        assert!(shader_mode(PixelFormat::Yuyv, Bt709).is_err());
        // The height heuristic picks the matrix when nothing signalled one.
        assert_eq!(shader_mode_for(PixelFormat::I420, 1080).unwrap(), 0);
        assert_eq!(shader_mode_for(PixelFormat::I420, 480).unwrap(), 1);
    }
}
