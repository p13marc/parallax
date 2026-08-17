//! YUV→RGB conversion constants shared by the render backends (#190).
//!
//! One module owns the numbers: the WGSL shader in `present.wgsl` mirrors
//! these constants (a comment there points back here), and the golden tests
//! below pin them, so shader drift is reviewable as a diff against this
//! file. Limited (studio) range only — that is what every decoder in the
//! tree emits.

/// Which YUV→RGB matrix a stream wants.
///
/// Chosen by the height heuristic in [`matrix_for_height`] until
/// colorimetry rides in metadata: SD content predates BT.709.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColorMatrix {
    /// ITU-R BT.601 (SD).
    Bt601,
    /// ITU-R BT.709 (HD).
    Bt709,
}

/// The height heuristic: ≥720 lines is HD and HD means BT.709.
#[cfg_attr(not(any(test, feature = "display-gpu")), allow(dead_code))]
pub(crate) fn matrix_for_height(height: u32) -> ColorMatrix {
    if height >= 720 {
        ColorMatrix::Bt709
    } else {
        ColorMatrix::Bt601
    }
}

/// `[cr_r, cb_g, cr_g, cb_b]`: the four non-trivial matrix coefficients,
/// applied to limited-range-expanded Y'CbCr as
///
/// ```text
/// r = y + cr_r·cr
/// g = y − cb_g·cb − cr_g·cr
/// b = y + cb_b·cb
/// ```
// The reference items below are exercised by the golden tests and mirrored
// by present.wgsl — they have no runtime caller by design.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const fn coefficients(matrix: ColorMatrix) -> [f32; 4] {
    match matrix {
        ColorMatrix::Bt601 => [1.402, 0.344136, 0.714136, 1.772],
        ColorMatrix::Bt709 => [1.5748, 0.187324, 0.468124, 1.8556],
    }
}

/// Limited-range expansion factors: Y spans 16..=235, chroma 16..=240.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const Y_SCALE: f32 = 255.0 / 219.0;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const C_SCALE: f32 = 255.0 / 224.0;

/// Reference conversion of one limited-range Y'CbCr sample to 8-bit RGB.
///
/// This scalar path is the truth the shader constants are pinned to; it is
/// not a hot path anywhere.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn yuv_to_rgb(y: u8, cb: u8, cr: u8, matrix: ColorMatrix) -> [u8; 3] {
    let [cr_r, cb_g, cr_g, cb_b] = coefficients(matrix);
    let y = (y as f32 - 16.0) * Y_SCALE;
    let cb = (cb as f32 - 128.0) * C_SCALE;
    let cr = (cr as f32 - 128.0) * C_SCALE;

    let r = y + cr_r * cr;
    let g = y - cb_g * cb - cr_g * cr;
    let b = y + cb_b * cb;
    [
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_range_black_and_white_hit_the_rails() {
        for m in [ColorMatrix::Bt601, ColorMatrix::Bt709] {
            assert_eq!(yuv_to_rgb(16, 128, 128, m), [0, 0, 0], "{m:?} black");
            assert_eq!(yuv_to_rgb(235, 128, 128, m), [255, 255, 255], "{m:?} white");
        }
    }

    #[test]
    fn below_black_and_above_white_clamp() {
        assert_eq!(yuv_to_rgb(0, 128, 128, ColorMatrix::Bt709), [0, 0, 0]);
        assert_eq!(
            yuv_to_rgb(255, 128, 128, ColorMatrix::Bt709),
            [255, 255, 255]
        );
    }

    #[test]
    fn primaries_land_where_the_standards_put_them() {
        // Limited-range Y'CbCr encodings of pure red/green/blue, computed
        // from the forward BT.601/709 matrices. Tolerance ±2 for the 8-bit
        // round trip.
        fn close(a: [u8; 3], b: [u8; 3]) -> bool {
            a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= 2)
        }
        // BT.601: red (81, 90, 240), green (145, 54, 34), blue (41, 240, 110)
        assert!(close(
            yuv_to_rgb(81, 90, 240, ColorMatrix::Bt601),
            [255, 0, 0]
        ));
        assert!(close(
            yuv_to_rgb(145, 54, 34, ColorMatrix::Bt601),
            [0, 255, 0]
        ));
        assert!(close(
            yuv_to_rgb(41, 240, 110, ColorMatrix::Bt601),
            [0, 0, 255]
        ));
        // BT.709: red (63, 102, 240), green (173, 42, 26), blue (32, 240, 118)
        assert!(close(
            yuv_to_rgb(63, 102, 240, ColorMatrix::Bt709),
            [255, 0, 0]
        ));
        assert!(close(
            yuv_to_rgb(173, 42, 26, ColorMatrix::Bt709),
            [0, 255, 0]
        ));
        assert!(close(
            yuv_to_rgb(32, 240, 118, ColorMatrix::Bt709),
            [0, 0, 255]
        ));
    }

    #[test]
    fn the_height_heuristic_splits_at_720() {
        assert_eq!(matrix_for_height(480), ColorMatrix::Bt601);
        assert_eq!(matrix_for_height(576), ColorMatrix::Bt601);
        assert_eq!(matrix_for_height(719), ColorMatrix::Bt601);
        assert_eq!(matrix_for_height(720), ColorMatrix::Bt709);
        assert_eq!(matrix_for_height(1080), ColorMatrix::Bt709);
    }
}
