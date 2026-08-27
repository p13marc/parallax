// Parallax video presentation shader (#190).
//
// One pipeline for every input layout. `params.planes` says how the planes
// are arranged and nothing else:
//   0 = planar YUV (I420)   1 = semi-planar YUV (NV12)   2 = packed RGB
//
// Colour is *data*, not a branch: the matrix coefficients and the range
// expansion arrive in the uniform block, computed by src/video/color.rs, and
// so does the display rotation (#266).
// Before #265 they were `if`s over a mode enum here, which meant the shader
// and the CPU converters each carried their own copy of the numbers — and
// disagreed. Now there is one copy, and this file has none.

struct Params {
    planes: u32,
    // 1 when the render target is sRGB and will encode for us, so the
    // gamma-encoded R'G'B' below has to be linearised first or the transfer
    // function is applied twice and the picture washes out. Iced picks an
    // sRGB surface; a backend that configures its own may not.
    srgb_target: u32,
    // Quarter-turns clockwise to apply when sampling: 0, 1, 2, 3. The frame
    // is stored one way up and displayed another (#266); `fit` is already in
    // display space, so this only has to invert the turn when reading.
    rotation: u32,
    _pad1: u32,
    // Where the picture sits inside the viewport, in 0..1 viewport space:
    // (x0, y0, width, height). Everything outside is painted black.
    //
    // A backend that owns its surface can letterbox with a scissor and a
    // black clear instead and pass (0, 0, 1, 1) here. One embedded in a GUI
    // cannot: its render pass loads rather than clears, so a narrowed
    // viewport would leave whatever was underneath showing in the bars.
    fit: vec4<f32>,
    // [cr_r, cb_g, cr_g, cb_b] — video::color::coefficients.
    coeffs: vec4<f32>,
    // [luma offset, luma scale, chroma scale, unused] — video::color::expansion,
    // with the offset already normalised to 0..1.
    expansion: vec4<f32>,
}

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex_y: texture_2d<f32>;
@group(0) @binding(2) var tex_u: texture_2d<f32>; // NV12: the RG chroma plane
@group(0) @binding(3) var tex_v: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: Params;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle: three vertices cover the whole viewport, which the
// backend sets to the letterbox rect — no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    var out: VsOut;
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index >> 1u) * 4 - 1);
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    // v flips: NDC y points up, texture v points down.
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// The sRGB electro-optical transfer function. MIRRORS
// video::color::srgb_to_linear — change both together; the golden tests
// there are the truth.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// Hand the target what it expects: linear when it will encode sRGB itself,
// the encoded values as-is otherwise.
fn present(rgb: vec3<f32>) -> vec4<f32> {
    let clamped = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    if params.srgb_target == 1u {
        return vec4<f32>(srgb_to_linear(clamped), 1.0);
    }
    return vec4<f32>(clamped, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let fit = (in.uv - params.fit.xy) / params.fit.zw;
    if fit.x < 0.0 || fit.x > 1.0 || fit.y < 0.0 || fit.y > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // Undo the rotation to get back to stored-frame coordinates. Cheaper
    // and exacter than a matrix: these are the only four angles a container
    // can express without resampling.
    var uv = fit;
    if params.rotation == 1u {
        uv = vec2<f32>(fit.y, 1.0 - fit.x);
    } else if params.rotation == 2u {
        uv = vec2<f32>(1.0 - fit.x, 1.0 - fit.y);
    } else if params.rotation == 3u {
        uv = vec2<f32>(1.0 - fit.y, fit.x);
    }

    if params.planes == 2u {
        return present(textureSample(tex_y, samp, uv).rgb);
    }

    var cbcr: vec2<f32>;
    if params.planes == 1u {
        cbcr = textureSample(tex_u, samp, uv).rg;
    } else {
        cbcr = vec2<f32>(
            textureSample(tex_u, samp, uv).r,
            textureSample(tex_v, samp, uv).r,
        );
    }

    // Range expansion: limited (studio) range stores Y in 16..=235 and
    // chroma in 16..=240, full range stores both whole. The uniform says
    // which, so this is the same arithmetic either way.
    let y = (textureSample(tex_y, samp, uv).r - params.expansion.x) * params.expansion.y;
    cbcr = (cbcr - 128.0 / 255.0) * params.expansion.z;

    let c = params.coeffs;
    let rgb = vec3<f32>(
        y + c.x * cbcr.y,
        y - c.y * cbcr.x - c.z * cbcr.y,
        y + c.w * cbcr.x,
    );
    return present(rgb);
}
