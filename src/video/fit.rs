//! Where the picture goes in the window, and how big a frame should be.
//!
//! Both are pure integer arithmetic with no rendering dependency, which is
//! the point: the CPU blit, the wgpu viewport and a host toolkit's shader
//! all call the same function, so they put the picture in exactly the same
//! place.

/// Expected byte length of a 4:2:0 planar/semi-planar frame.
///
/// I420 and NV12 pack the same total: `w*h` luma plus two half-resolution
/// chroma planes (interleaved into one for NV12).
pub fn yuv420_len(width: u32, height: u32) -> usize {
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    w * h + 2 * (cw * ch)
}

/// Aspect-preserving letterbox: `(x0, y0, out_width, out_height)` of the
/// image rect inside a `dst_width`×`dst_height` window; black bars fill
/// the rest.
///
/// Callers must reject zero dimensions before calling — this divides by
/// `src_width`/`src_height`.
pub fn letterbox_rect(
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) -> (usize, usize, usize, usize) {
    let (out_width, out_height) = if src_width * dst_height >= src_height * dst_width {
        // Width-bound: pillar-free, bars top/bottom.
        (dst_width, (src_height * dst_width / src_width).max(1))
    } else {
        // Height-bound: bars left/right.
        ((src_width * dst_height / src_height).max(1), dst_height)
    };
    (
        (dst_width - out_width) / 2,
        (dst_height - out_height) / 2,
        out_width,
        out_height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuv420_len_matches_the_plane_math() {
        assert_eq!(yuv420_len(1920, 1080), 1920 * 1080 * 3 / 2);
        // Odd dimensions round chroma up.
        assert_eq!(yuv420_len(3, 3), 9 + 2 * 4);
    }

    #[test]
    fn letterbox_rect_is_shared_truth() {
        // 16:9 into 16:10: width-bound, bars top and bottom.
        let (x0, y0, w, h) = letterbox_rect(1920, 1080, 1920, 1200);
        assert_eq!((x0, w), (0, 1920));
        assert_eq!(h, 1080);
        assert_eq!(y0, 60);
        // Matching aspect fills.
        assert_eq!(letterbox_rect(2, 2, 8, 8), (0, 0, 8, 8));
    }
}
