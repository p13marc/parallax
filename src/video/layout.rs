//! Per-plane geometry for uploading or importing a frame into textures.
//!
//! Pure arithmetic, so it is unit-tested without a GPU — which matters,
//! because a mistake here does not fail loudly. It produces a picture that
//! is sheared, shifted or checkerboarded, and only a comparison catches it.
//!
//! [`TexelFormat`] exists instead of `wgpu::TextureFormat` so this module
//! has no graphics dependency at all: parallax's own backend is on one wgpu
//! major and a host toolkit's renderer may be on another, and neither can
//! name the other's types. Each maps these four arms to its own enum.
//!
//! [`plane_uploads`] and [`plane_imports`] look like the same function and
//! are not. Upload geometry is resolved against [`PlaneLayout::resolved`],
//! which derives row counts and rejects a stride narrower than a row or a
//! buffer too short to hold the planes. Import geometry reads the declared
//! plane descriptors directly — a dma-buf's rows are the producer's business
//! and there are no bytes here to bounds-check. Keep them separate.

use crate::error::{Error, Result};
use crate::format::{PixelFormat, PlaneLayout};

/// The texel formats a video plane is ever bound as.
///
/// I420 binds three single-byte planes; NV12 binds luma as [`R8`](Self::R8)
/// and its interleaved chroma as [`Rg8`](Self::Rg8); packed RGB binds one
/// four-byte plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexelFormat {
    /// One unsigned byte per texel.
    R8,
    /// Two unsigned bytes per texel — NV12's interleaved Cb/Cr.
    Rg8,
    /// Four unsigned bytes per texel, red first.
    Rgba8,
    /// Four unsigned bytes per texel, blue first.
    Bgra8,
}

impl TexelFormat {
    /// Bytes one texel occupies.
    pub const fn bytes(self) -> usize {
        match self {
            TexelFormat::R8 => 1,
            TexelFormat::Rg8 => 2,
            TexelFormat::Rgba8 | TexelFormat::Bgra8 => 4,
        }
    }
}

/// One plane's slice geometry for a strided texture write.
#[derive(Debug, PartialEq, Eq)]
pub struct PlaneUpload {
    /// Byte offset of the plane's first row in the frame data.
    pub offset: usize,
    /// Real row stride (becomes `bytes_per_row`).
    pub bytes_per_row: usize,
    /// Rows to copy.
    pub rows: usize,
    /// Used bytes in each row (the final row needs only this much data).
    pub row_bytes: usize,
    /// Texture extent width in texels (row_bytes / bytes-per-texel).
    pub width_texels: u32,
}

impl PlaneUpload {
    /// End of the byte range a texture write reads: full strides for all
    /// rows but the last, which needs only its used bytes.
    pub fn end(&self) -> usize {
        self.offset + self.bytes_per_row * (self.rows - 1) + self.row_bytes
    }
}

/// The texel format of each plane, in shader binding order.
///
/// `Err` for a format that has no presentation path.
pub fn plane_formats(format: PixelFormat) -> Result<&'static [TexelFormat]> {
    Ok(match format {
        PixelFormat::I420 => &[TexelFormat::R8, TexelFormat::R8, TexelFormat::R8],
        PixelFormat::Nv12 => &[TexelFormat::R8, TexelFormat::Rg8],
        PixelFormat::Rgba => &[TexelFormat::Rgba8],
        PixelFormat::Bgra => &[TexelFormat::Bgra8],
        other => {
            return Err(Error::Element(format!(
                "video: unsupported display format {other:?}"
            )));
        }
    })
}

/// Resolve a frame's planes against its (possibly strided) layout into
/// texture-write geometry, validating the data length. Plane order matches
/// the texture binding order: Y/U/V for I420, Y/UV for NV12, one packed
/// plane for RGB.
pub fn plane_uploads(
    format: PixelFormat,
    width: u32,
    height: u32,
    layout: Option<&PlaneLayout>,
    data_len: usize,
) -> Result<Vec<PlaneUpload>> {
    let texels = plane_formats(format)?;

    let packed;
    let layout = match layout {
        Some(l) => l,
        None => {
            packed = PlaneLayout::packed(format, width, height);
            &packed
        }
    };

    let mut uploads = Vec::with_capacity(texels.len());
    for (plane, &texel) in layout.resolved(format, width, height).zip(texels) {
        if plane.stride < plane.row_bytes || plane.rows == 0 {
            return Err(Error::Element(format!(
                "video: invalid plane layout (stride {} < row {})",
                plane.stride, plane.row_bytes
            )));
        }
        let upload = PlaneUpload {
            offset: plane.offset,
            bytes_per_row: plane.stride,
            rows: plane.rows,
            row_bytes: plane.row_bytes,
            width_texels: (plane.row_bytes / texel.bytes()) as u32,
        };
        if upload.end() > data_len {
            return Err(Error::Element(format!(
                "video: {format:?} frame is {data_len} bytes, plane needs {}",
                upload.end()
            )));
        }
        uploads.push(upload);
    }
    if uploads.len() != texels.len() {
        return Err(Error::Element(format!(
            "video: {format:?} frame declares {} planes, expected {}",
            uploads.len(),
            texels.len()
        )));
    }
    Ok(uploads)
}

/// One plane's worth of import geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneImport {
    /// Texel format the shader expects for this plane.
    pub format: TexelFormat,
    /// Size in *texels*, which for [`TexelFormat::Rg8`] chroma is half the
    /// byte width.
    pub width: u32,
    /// Height of the plane in rows.
    pub height: u32,
    /// Byte offset of the plane within the dma-buf.
    pub offset: u64,
    /// Byte pitch of one row.
    pub stride: u64,
}

/// Resolve a frame's plane layout into per-plane import geometry.
///
/// `base_offset` is the buffer's own offset into its allocation: a `slice`d
/// dma-buf buffer is legal and its offset is not part of the [`PlaneLayout`].
pub fn plane_imports(
    format: PixelFormat,
    width: u32,
    height: u32,
    layout: &PlaneLayout,
    base_offset: usize,
) -> Result<Vec<PlaneImport>> {
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    // (texel format, texels wide, rows) per plane, in shader binding order.
    let planes: &[(TexelFormat, u32, u32)] = match format {
        PixelFormat::I420 => &[
            (TexelFormat::R8, width, height),
            (TexelFormat::R8, cw, ch),
            (TexelFormat::R8, cw, ch),
        ],
        // NV12's chroma is two bytes per texel, so `cw` texels is `cw * 2`
        // bytes — the pitch is in bytes either way.
        PixelFormat::Nv12 => &[(TexelFormat::R8, width, height), (TexelFormat::Rg8, cw, ch)],
        PixelFormat::Rgba => &[(TexelFormat::Rgba8, width, height)],
        PixelFormat::Bgra => &[(TexelFormat::Bgra8, width, height)],
        other => {
            return Err(Error::Element(format!(
                "video: cannot import {other:?} frames"
            )));
        }
    };

    let declared = layout.planes();
    if declared.len() < planes.len() {
        return Err(Error::Element(format!(
            "video: {format:?} needs {} planes, the frame declares {}",
            planes.len(),
            declared.len()
        )));
    }

    Ok(planes
        .iter()
        .zip(declared)
        .map(|(&(format, width, height), desc)| PlaneImport {
            format,
            width,
            height,
            offset: (base_offset + desc.offset) as u64,
            stride: desc.stride as u64,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::PlaneDesc;

    fn layout(planes: &[(usize, usize)]) -> PlaneLayout {
        PlaneLayout::from_planes(
            &planes
                .iter()
                .map(|&(offset, stride)| PlaneDesc { offset, stride })
                .collect::<Vec<_>>(),
        )
    }

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
        let layout = layout(&[(0, 128), (128 * 64, 64), (128 * 64 + 64 * 32, 64)]);
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
        // Y stride 8 is narrower than its 64-byte row.
        let bad = layout(&[(0, 8), (512, 32), (1024, 32)]);
        assert!(plane_uploads(PixelFormat::I420, 64, 48, Some(&bad), 1 << 20).is_err());
    }

    /// NV12's chroma plane is half the texels wide but the same bytes wide,
    /// which is exactly the confusion that produces a half-width picture.
    #[test]
    fn nv12_chroma_is_half_the_texels_at_the_full_pitch() {
        let l = layout(&[(0, 1920), (1920 * 1088, 1920)]);
        let p = plane_imports(PixelFormat::Nv12, 1920, 1080, &l, 0).unwrap();

        assert_eq!(p.len(), 2, "NV12 is two planes, not three");
        assert_eq!(p[0].format, TexelFormat::R8);
        assert_eq!((p[0].width, p[0].height), (1920, 1080));
        assert_eq!((p[0].offset, p[0].stride), (0, 1920));

        assert_eq!(p[1].format, TexelFormat::Rg8);
        assert_eq!((p[1].width, p[1].height), (960, 540), "texels, not bytes");
        assert_eq!(p[1].stride, 1920, "bytes per row is unchanged");
        assert_eq!(p[1].offset, 1920 * 1088);
    }

    /// Odd dimensions round chroma up: a 1x1 chroma plane still exists for a
    /// 1x1 frame, and dropping it would index past the end of the layout.
    #[test]
    fn odd_dimensions_round_the_chroma_plane_up() {
        let l = layout(&[(0, 128), (128 * 32, 128), (128 * 48, 128)]);
        let p = plane_imports(PixelFormat::I420, 17, 9, &l, 0).unwrap();
        assert_eq!((p[1].width, p[1].height), (9, 5));
        assert_eq!((p[2].width, p[2].height), (9, 5));
    }

    /// A sliced buffer's own offset has to reach the import, or every plane
    /// is read from the wrong place.
    #[test]
    fn the_buffers_base_offset_is_added_to_every_plane() {
        let l = layout(&[(0, 64), (64 * 32, 64)]);
        let p = plane_imports(PixelFormat::Nv12, 64, 64, &l, 4096).unwrap();
        assert_eq!(p[0].offset, 4096);
        assert_eq!(p[1].offset, 4096 + 64 * 32);
    }

    #[test]
    fn a_frame_with_too_few_planes_is_refused() {
        let l = layout(&[(0, 64)]);
        assert!(plane_imports(PixelFormat::I420, 64, 64, &l, 0).is_err());
    }

    #[test]
    fn packed_formats_import_as_one_plane() {
        let l = layout(&[(0, 256)]);
        let p = plane_imports(PixelFormat::Bgra, 64, 64, &l, 0).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].format, TexelFormat::Bgra8);
    }

    #[test]
    fn texel_sizes_match_the_binding_table() {
        assert_eq!(TexelFormat::R8.bytes(), 1);
        assert_eq!(TexelFormat::Rg8.bytes(), 2);
        assert_eq!(TexelFormat::Rgba8.bytes(), 4);
        assert_eq!(TexelFormat::Bgra8.bytes(), 4);
        // The two resolvers must agree on how many planes a format has.
        for f in [
            PixelFormat::I420,
            PixelFormat::Nv12,
            PixelFormat::Rgba,
            PixelFormat::Bgra,
        ] {
            let uploads = plane_uploads(f, 64, 64, None, 64 * 64 * 4).unwrap();
            assert_eq!(uploads.len(), plane_formats(f).unwrap().len(), "{f:?}");
        }
    }
}
