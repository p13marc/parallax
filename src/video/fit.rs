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

/// Where a frame's *picture* goes in the window (#266).
///
/// [`letterbox_rect`] with the display geometry applied first: non-square
/// pixels are stretched and a rotation swaps the axes, so an anamorphic DVD
/// fills the window at 16:9 rather than 3:2 and a portrait phone video gets
/// pillarboxed rather than letterboxed.
///
/// This is the function presenters call. `letterbox_rect` is the square-pixel
/// core underneath it, kept public because the arithmetic is worth testing on
/// its own.
pub fn display_rect(
    src_width: usize,
    src_height: usize,
    display: crate::format::DisplayGeometry,
    dst_width: usize,
    dst_height: usize,
) -> (usize, usize, usize, usize) {
    let (w, h) = display.display_size(src_width as u32, src_height as u32);
    letterbox_rect(w as usize, h as usize, dst_width, dst_height)
}

/// Aspect-preserving letterbox: `(x0, y0, out_width, out_height)` of the
/// image rect inside a `dst_width`×`dst_height` window; black bars fill
/// the rest.
///
/// Square pixels, upright — see [`display_rect`] for the one that honours
/// what the container said.
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

    #[test]
    fn display_geometry_reshapes_the_picture_before_it_is_fitted() {
        use crate::format::{DisplayGeometry, Rotation};

        // Anamorphic NTSC DVD: 720x480 stored, 32:27 pixels, shown at 16:9.
        // In a 16:9 window it fills to the pixel — 853 is what 720x32/27
        // rounds to, so 1599 rather than 1600 is the honest answer, and the
        // bar is half a pixel wide.
        let par = DisplayGeometry {
            par: (32, 27),
            rotation: Rotation::None,
        };
        let (_, y0, w, h) = display_rect(720, 480, par, 1600, 900);
        assert_eq!((y0, h), (0, 900));
        assert!(w >= 1599, "essentially full width, got {w}");
        // Ignoring the PAR, the same frame is 3:2 and loses 250 px a side.
        assert_eq!(letterbox_rect(720, 480, 1600, 900), (125, 0, 1350, 900));

        // Portrait phone video: 1920x1080 stored, rotated a quarter turn. In
        // a landscape window it is pillarboxed, not letterboxed.
        let turned = DisplayGeometry {
            par: (1, 1),
            rotation: Rotation::Cw90,
        };
        let (x0, _, w, h) = display_rect(1920, 1080, turned, 1600, 900);
        assert_eq!((w, h), (506, 900), "tall picture, height-bound");
        assert_eq!(x0, (1600 - 506) / 2);

        // The identity changes nothing, which is what lets a consumer apply it
        // unconditionally.
        let same = DisplayGeometry::default();
        assert_eq!(
            display_rect(1920, 1080, same, 1280, 720),
            letterbox_rect(1920, 1080, 1280, 720)
        );
    }

    #[test]
    fn a_rotated_anamorphic_frame_stretches_along_the_right_axis() {
        use crate::format::{DisplayGeometry, Rotation};

        // The order matters: stretch, then swap. 720x480 at 32:27 displays as
        // 853x480 upright, so a quarter turn makes it 480 wide and 853 tall —
        // not 853 wide and 480 tall with the stretch on the other axis.
        let g = DisplayGeometry {
            par: (32, 27),
            rotation: Rotation::Cw90,
        };
        assert_eq!(g.display_size(720, 480), (480, 853));
    }
}
