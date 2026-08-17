//! Factory registrations: codec elements (each under its feature).
//!
//! The module itself is gated on `any(...)` of the features it registers —
//! a strict subset of the `elements::codec` gate in `elements/mod.rs`, so
//! whenever this compiles the codec module exists (CLAUDE.md gotcha 14).

use super::{ElementFactory, Props};
use crate::element::DynAsyncElement;
#[cfg(any(
    feature = "h264",
    feature = "av1-encode",
    feature = "av1-decode",
    feature = "vpx",
    feature = "image-jpeg",
    feature = "image-png"
))]
use crate::element::ElementAdapter;
#[cfg(feature = "opus")]
use crate::element::TransformAdapter;
use crate::error::Result;

pub(super) fn register(f: &mut ElementFactory) {
    #[cfg(feature = "h264")]
    {
        f.register("h264enc", create_h264enc);
        f.register("h264dec", create_h264dec);
    }
    #[cfg(feature = "av1-encode")]
    f.register("av1enc", create_av1enc);
    #[cfg(feature = "av1-decode")]
    f.register("av1dec", create_av1dec);
    #[cfg(feature = "vpx")]
    {
        f.register("vp8dec", create_vp8dec);
        f.register("vp9dec", create_vp9dec);
    }
    #[cfg(feature = "opus")]
    {
        f.register("opusenc", create_opusenc);
        f.register("opusdec", create_opusdec);
    }
    #[cfg(feature = "image-jpeg")]
    {
        f.register("jpegenc", create_jpegenc);
        f.register("jpegdec", create_jpegdec);
    }
    #[cfg(feature = "image-png")]
    {
        f.register("pngenc", create_pngenc);
        f.register("pngdec", create_pngdec);
    }
    // Silence "unused" when only a subset of codec features is on.
    let _ = f;
}

#[cfg(feature = "h264")]
fn create_h264enc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::{Complexity, H264Encoder, H264EncoderConfig, Profile, UsageType};

    let mut config = H264EncoderConfig::new();
    if let Some(b) = props.get_u32("bitrate")? {
        config.bitrate_bps = b;
    }
    if let Some(fps) = props.get_f32("fps")? {
        config.max_frame_rate = fps;
    }
    if let Some(qp) = props.get_u8("qp")? {
        config.qp = qp;
    }
    if let Some(n) = props.get_u32("keyframe-interval")? {
        config.keyframe_interval = n;
    }
    if let Some(n) = props.get_u32("threads")? {
        config.num_threads = n;
    }
    if let Some(b) = props.get_bool("scene-change")? {
        config.scene_change_detect = b;
    }
    if let Some(len) = props.get_u32("max-slice-len")? {
        config.max_slice_len = Some(len);
    }
    if let Some(b) = props.get_bool("skip-frames")? {
        config.skip_frames = b;
    }
    if let Some(p) = props.get_enum(
        "profile",
        &[
            ("baseline", Profile::Baseline),
            ("main", Profile::Main),
            ("high", Profile::High),
        ],
    )? {
        config.profile = Some(p);
    }
    if let Some(c) = props.get_enum(
        "complexity",
        &[
            ("low", Complexity::Low),
            ("medium", Complexity::Medium),
            ("high", Complexity::High),
        ],
    )? {
        config.complexity = c;
    }
    if let Some(u) = props.get_enum(
        "usage",
        &[
            ("camera", UsageType::CameraRealtime),
            ("screen", UsageType::ScreenRealtime),
            ("camera-offline", UsageType::CameraNonRealtime),
            ("screen-offline", UsageType::ScreenNonRealtime),
        ],
    )? {
        config.usage_type = u;
    }

    let enc = H264Encoder::new(config)?;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(enc)))
}

#[cfg(feature = "h264")]
fn create_h264dec(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::H264Decoder;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        H264Decoder::new()?,
    )))
}

#[cfg(feature = "av1-encode")]
fn create_av1enc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::{Rav1eConfig, Rav1eEncoder};

    let mut config = Rav1eConfig::default();
    if let Some(s) = props.get_usize("speed")? {
        config.speed = s;
    }
    if let Some(q) = props.get_usize("quantizer")? {
        config.quantizer = q;
    }
    if let Some(b) = props.get_usize("bitrate")? {
        config.bitrate = b;
    }
    let enc = Rav1eEncoder::new(config)?;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(enc)))
}

#[cfg(feature = "av1-decode")]
fn create_av1dec(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::Dav1dDecoder;
    let mut dec = Dav1dDecoder::new()?;
    if let Some(n) = props.get_u32("threads")? {
        dec = dec.with_threads(n)?;
    }
    if let Some(d) = props.get_u32("max-frame-delay")? {
        dec = dec.with_max_frame_delay(d)?;
    }
    if let Some(g) = props.get_bool("apply-grain")? {
        dec = dec.with_apply_grain(g)?;
    }
    Ok(DynAsyncElement::new_box(ElementAdapter::new(dec)))
}

#[cfg(feature = "vpx")]
fn create_vp8dec(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::VpxDecoder;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        VpxDecoder::vp8()?,
    )))
}

#[cfg(feature = "vpx")]
fn create_vp9dec(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::VpxDecoder;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        VpxDecoder::vp9()?,
    )))
}

#[cfg(feature = "opus")]
fn create_opusenc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::{AudioEncoderElement, OpusApplication, OpusEncoder};

    let rate = props.get_u32("rate")?.unwrap_or(48000);
    let channels = props.get_u32("channels")?.unwrap_or(2);
    let bitrate = props.get_u32("bitrate")?.unwrap_or(128_000);
    let application = props
        .get_enum(
            "application",
            &[
                ("audio", OpusApplication::Audio),
                ("voip", OpusApplication::Voip),
                ("lowdelay", OpusApplication::LowDelay),
            ],
        )?
        .unwrap_or(OpusApplication::Audio);

    let enc = OpusEncoder::new(rate, channels, bitrate, application)?;
    let element = AudioEncoderElement::new_s16(enc, rate, channels)?;
    Ok(DynAsyncElement::new_box(TransformAdapter::new(element)))
}

#[cfg(feature = "opus")]
fn create_opusdec(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::{AudioDecoderElement, OpusDecoder};

    let rate = props.get_u32("rate")?.unwrap_or(48000);
    let channels = props.get_u32("channels")?.unwrap_or(2);
    let dec = OpusDecoder::new(rate, channels)?;
    Ok(DynAsyncElement::new_box(TransformAdapter::new(
        AudioDecoderElement::new(dec),
    )))
}

#[cfg(feature = "image-jpeg")]
fn create_jpegenc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::JpegEncoder;
    let mut enc = JpegEncoder::new();
    if let Some(q) = props.get_u8("quality")? {
        enc = enc.with_quality(q);
    }
    Ok(DynAsyncElement::new_box(ElementAdapter::new(enc)))
}

#[cfg(feature = "image-jpeg")]
fn create_jpegdec(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::JpegDecoder;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        JpegDecoder::new(),
    )))
}

#[cfg(feature = "image-png")]
fn create_pngenc(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::PngEncoder;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        PngEncoder::new(),
    )))
}

#[cfg(feature = "image-png")]
fn create_pngdec(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::codec::PngDecoder;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        PngDecoder::new(),
    )))
}
