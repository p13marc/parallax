// AutoVideoSink presentation shader (#190).
//
// One pipeline for every input layout; `params.mode` selects:
//   0 = I420 · BT.709     1 = I420 · BT.601
//   2 = NV12 · BT.709     3 = NV12 · BT.601
//   4 = RGB passthrough (tex_y holds the RGBA/BGRA texture)
//
// The YUV matrix constants MIRROR src/elements/app/present/color.rs —
// change both together; the golden tests there are the truth.

struct Params {
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if params.mode == 4u {
        return vec4<f32>(textureSample(tex_y, samp, in.uv).rgb, 1.0);
    }

    // Limited (studio) range expansion: Y 16..=235, chroma 16..=240.
    let y = (textureSample(tex_y, samp, in.uv).r - 16.0 / 255.0) * (255.0 / 219.0);
    var cbcr: vec2<f32>;
    if params.mode >= 2u {
        cbcr = textureSample(tex_u, samp, in.uv).rg;
    } else {
        cbcr = vec2<f32>(
            textureSample(tex_u, samp, in.uv).r,
            textureSample(tex_v, samp, in.uv).r,
        );
    }
    cbcr = (cbcr - 128.0 / 255.0) * (255.0 / 224.0);

    // [cr_r, cb_g, cr_g, cb_b] — see color.rs::coefficients.
    var c: vec4<f32>;
    if params.mode == 0u || params.mode == 2u {
        c = vec4<f32>(1.5748, 0.187324, 0.468124, 1.8556); // BT.709
    } else {
        c = vec4<f32>(1.402, 0.344136, 0.714136, 1.772); // BT.601
    }
    let rgb = vec3<f32>(
        y + c.x * cbcr.y,
        y - c.y * cbcr.x - c.z * cbcr.y,
        y + c.w * cbcr.x,
    );
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
