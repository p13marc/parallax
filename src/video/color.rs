//! YUV→RGB conversion constants shared by every presentation backend (#190).
//!
//! One module owns the numbers: the WGSL in [`PRESENT_WGSL`](super::PRESENT_WGSL)
//! mirrors these constants (a comment there points back here), and the golden
//! tests below pin them, so shader drift is reviewable as a diff against this
//! file.
//!
//! Since #265 both axes are carried rather than assumed. Limited range is
//! still the overwhelmingly common case — and what an unsignalled stream is
//! read as — but a file that says it is full range is now believed, and
//! [`expansion`] is the one place either answer turns into numbers.
//!
//! The shader expresses the same algebra normalised to 0..1 where this module
//! works in 0..255. That difference is deliberate, not drift.

pub use crate::format::{ColorMatrix, ColorRange};

/// The height heuristic: ≥720 lines is HD and HD means BT.709.
///
/// A *fallback*, for a stream that signalled nothing. Since #265 a file that
/// actually declared its matrix outranks this — resolve with
/// [`ColorInfo::matrix_or`](crate::format::ColorInfo::matrix_or), which is
/// what every consumer in the tree does.
pub fn matrix_for_height(height: u32) -> ColorMatrix {
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
/// `Unspecified` answers with BT.601's, which is the historical assumption —
/// but a caller should have resolved it first, because "nothing was said" and
/// "SD" are different claims.
pub const fn coefficients(matrix: ColorMatrix) -> [f32; 4] {
    match matrix {
        ColorMatrix::Bt601 | ColorMatrix::Unspecified => [1.402, 0.344136, 0.714136, 1.772],
        ColorMatrix::Bt709 => [1.5748, 0.187324, 0.468124, 1.8556],
        ColorMatrix::Bt2020Ncl => [1.4746, 0.164553, 0.571353, 1.8814],
    }
}

/// Limited-range luma expansion factor: Y spans 16..=235.
pub const Y_SCALE: f32 = 255.0 / 219.0;
/// Limited-range chroma expansion factor: Cb/Cr span 16..=240.
pub const C_SCALE: f32 = 255.0 / 224.0;

/// How a range maps a stored sample onto 0..255: `(luma offset, luma scale,
/// chroma scale)`.
///
/// One function so the scalar converter, the SIMD converter and the shader
/// cannot each decide for themselves — which is exactly what they were doing
/// before #265.
pub const fn expansion(range: ColorRange) -> (f32, f32, f32) {
    if range.is_limited() {
        (16.0, Y_SCALE, C_SCALE)
    } else {
        (0.0, 1.0, 1.0)
    }
}

/// Reference conversion of one Y'CbCr sample to 8-bit RGB.
///
/// This scalar path is the truth the shader constants and the converter arms
/// are pinned to; it is not a hot path anywhere.
pub fn yuv_to_rgb(y: u8, cb: u8, cr: u8, matrix: ColorMatrix, range: ColorRange) -> [u8; 3] {
    let [cr_r, cb_g, cr_g, cb_b] = coefficients(matrix);
    let (y_offset, y_scale, c_scale) = expansion(range);
    let y = (y as f32 - y_offset) * y_scale;
    let cb = (cb as f32 - 128.0) * c_scale;
    let cr = (cr as f32 - 128.0) * c_scale;

    let r = y + cr_r * cr;
    let g = y - cb_g * cb - cr_g * cr;
    let b = y + cb_b * cb;
    [
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
    ]
}

/// `(kr, kg, kb)`: the luma weights, derived from the same four coefficients
/// [`coefficients`] returns.
///
/// Derived rather than tabulated so the forward and inverse directions cannot
/// drift: `cr_r = 2(1 − kr)` and `cb_b = 2(1 − kb)` by construction, so one
/// table defines both.
pub fn luma_weights(matrix: ColorMatrix) -> (f32, f32, f32) {
    let [cr_r, _, _, cb_b] = coefficients(matrix);
    let kr = 1.0 - cr_r / 2.0;
    let kb = 1.0 - cb_b / 2.0;
    (kr, 1.0 - kr - kb, kb)
}

/// Reference conversion of one 8-bit RGB sample to Y'CbCr — the inverse of
/// [`yuv_to_rgb`], and the truth the converter's scalar arm is pinned to.
pub fn rgb_to_yuv(r: u8, g: u8, b: u8, matrix: ColorMatrix, range: ColorRange) -> [u8; 3] {
    let (kr, kg, kb) = luma_weights(matrix);
    let [cr_r, _, _, cb_b] = coefficients(matrix);
    let (r, g, b) = (r as f32, g as f32, b as f32);

    let y = kr * r + kg * g + kb * b;
    let cb = (b - y) / cb_b;
    let cr = (r - y) / cr_r;

    // Compress into the stored range: limited puts Y in 16..=235 and chroma in
    // 16..=240, full leaves both alone.
    let (y_offset, y_scale, c_scale) = expansion(range);
    let y = y_offset + y / y_scale;
    let cb = 128.0 + cb / c_scale;
    let cr = 128.0 + cr / c_scale;

    [
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    ]
}

/// The sRGB electro-optical transfer function: encoded value → linear light.
///
/// Needed when the render target is sRGB and will encode for us. Everything
/// upstream of presentation — `yuv_to_rgb`, the decoders, the converters —
/// works in gamma-encoded R'G'B', so handing those values straight to an sRGB
/// surface applies the transfer function twice and washes the picture out.
///
/// This scalar path is the truth `PRESENT_WGSL`'s copy is pinned to.
pub fn srgb_to_linear(encoded: f32) -> f32 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// The inverse of [`srgb_to_linear`]: linear light → encoded value.
pub fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_transfer_function_pins_its_rails_and_its_knee() {
        // The two ends must be exact: a target that encodes what we
        // linearised has to give black back as black and white as white, or
        // every frame is subtly wrong.
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);

        // Mid-grey: the number that makes a double-encode obvious. 0.5
        // encoded is 0.2140 linear; re-encoding *that* would give 0.5 again,
        // and skipping the step leaves 0.5 where 0.2140 belongs.
        assert!((srgb_to_linear(0.5) - 0.214_041).abs() < 1e-5);

        // Both sides of the linear-segment knee, and a round trip.
        for v in [0.0, 0.01, 0.04045, 0.04046, 0.2, 0.5, 0.9, 1.0] {
            let round = linear_to_srgb(srgb_to_linear(v));
            assert!((round - v).abs() < 1e-5, "{v} round-tripped to {round}");
        }
    }

    const LIM: ColorRange = ColorRange::Limited;

    #[test]
    fn limited_range_black_and_white_hit_the_rails() {
        for m in [ColorMatrix::Bt601, ColorMatrix::Bt709] {
            assert_eq!(yuv_to_rgb(16, 128, 128, m, LIM), [0, 0, 0], "{m:?} black");
            assert_eq!(
                yuv_to_rgb(235, 128, 128, m, LIM),
                [255, 255, 255],
                "{m:?} white"
            );
        }
    }

    #[test]
    fn full_range_black_and_white_are_the_other_two_numbers() {
        // The whole point of carrying the range: the same stored bytes mean
        // different light. 16 is near-black in limited range and a real dark
        // grey in full range, and a file that says which one it is must be
        // believed rather than guessed at.
        for m in [ColorMatrix::Bt601, ColorMatrix::Bt709] {
            let full = ColorRange::Full;
            assert_eq!(yuv_to_rgb(0, 128, 128, m, full), [0, 0, 0], "{m:?} black");
            assert_eq!(
                yuv_to_rgb(255, 128, 128, m, full),
                [255, 255, 255],
                "{m:?} white"
            );
            assert_eq!(yuv_to_rgb(16, 128, 128, m, full), [16, 16, 16], "{m:?} 16");
            // ...and the same sample read the other way is black.
            assert_eq!(yuv_to_rgb(16, 128, 128, m, LIM), [0, 0, 0], "{m:?} 16 lim");
        }
    }

    #[test]
    fn an_unsignalled_range_reads_as_limited() {
        // Because that is what essentially every decoder in this tree emits,
        // and because guessing full range on limited content crushes blacks
        // into grey — the more visible of the two mistakes.
        assert_eq!(
            yuv_to_rgb(16, 128, 128, ColorMatrix::Bt709, ColorRange::Unspecified),
            yuv_to_rgb(16, 128, 128, ColorMatrix::Bt709, LIM)
        );
    }

    #[test]
    fn below_black_and_above_white_clamp() {
        assert_eq!(yuv_to_rgb(0, 128, 128, ColorMatrix::Bt709, LIM), [0, 0, 0]);
        assert_eq!(
            yuv_to_rgb(255, 128, 128, ColorMatrix::Bt709, LIM),
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
            yuv_to_rgb(81, 90, 240, ColorMatrix::Bt601, LIM),
            [255, 0, 0]
        ));
        assert!(close(
            yuv_to_rgb(145, 54, 34, ColorMatrix::Bt601, LIM),
            [0, 255, 0]
        ));
        assert!(close(
            yuv_to_rgb(41, 240, 110, ColorMatrix::Bt601, LIM),
            [0, 0, 255]
        ));
        // BT.709: red (63, 102, 240), green (173, 42, 26), blue (32, 240, 118)
        assert!(close(
            yuv_to_rgb(63, 102, 240, ColorMatrix::Bt709, LIM),
            [255, 0, 0]
        ));
        assert!(close(
            yuv_to_rgb(173, 42, 26, ColorMatrix::Bt709, LIM),
            [0, 255, 0]
        ));
        assert!(close(
            yuv_to_rgb(32, 240, 118, ColorMatrix::Bt709, LIM),
            [0, 0, 255]
        ));
    }

    #[test]
    fn the_two_directions_are_inverses() {
        // One table defines both, so this is really asking whether the
        // derivation is right — and it is the check the converter's scalar
        // arm never had, which is how it ended up doing full-range maths on
        // limited-range input for as long as it did.
        for matrix in [
            ColorMatrix::Bt601,
            ColorMatrix::Bt709,
            ColorMatrix::Bt2020Ncl,
        ] {
            for range in [ColorRange::Limited, ColorRange::Full] {
                for rgb in [
                    [0u8, 0, 0],
                    [255, 255, 255],
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [37, 140, 200],
                ] {
                    let [y, cb, cr] = rgb_to_yuv(rgb[0], rgb[1], rgb[2], matrix, range);
                    let back = yuv_to_rgb(y, cb, cr, matrix, range);
                    for (a, b) in back.iter().zip(rgb) {
                        assert!(
                            a.abs_diff(b) <= 3,
                            "{matrix:?}/{range:?}: {rgb:?} -> {:?} -> {back:?}",
                            [y, cb, cr]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_luma_weights_are_the_ones_the_standards_name() {
        let (kr, kg, kb) = luma_weights(ColorMatrix::Bt601);
        assert!((kr - 0.299).abs() < 1e-3, "{kr}");
        assert!((kg - 0.587).abs() < 1e-3, "{kg}");
        assert!((kb - 0.114).abs() < 1e-3, "{kb}");
        let (kr, kg, kb) = luma_weights(ColorMatrix::Bt709);
        assert!((kr - 0.2126).abs() < 1e-3, "{kr}");
        assert!((kg - 0.7152).abs() < 1e-3, "{kg}");
        assert!((kb - 0.0722).abs() < 1e-3, "{kb}");
    }

    #[test]
    fn white_encodes_to_the_range_it_is_told() {
        // The assertion that would have caught the whole bug: limited-range
        // white is stored as 235, full-range white as 255.
        assert_eq!(rgb_to_yuv(255, 255, 255, ColorMatrix::Bt709, LIM)[0], 235);
        assert_eq!(
            rgb_to_yuv(255, 255, 255, ColorMatrix::Bt709, ColorRange::Full)[0],
            255
        );
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
