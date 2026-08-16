//! Factory registrations: transform and timing elements.

use super::{ElementFactory, Props};
use crate::element::{DynAsyncElement, ElementAdapter, SourceAdapter, TransformAdapter};
use crate::elements::transform::{AudioConvertElement, AudioDownmix, AudioResampleElement};
use crate::elements::{
    Batch, BufferPad, BufferSlice, BufferTrim, Debounce, Delay, Gain, RateLimiter, SequenceNumber,
    Throttle, Timeout, TimestampMode, Timestamper, Unbatch, VideoScale,
};
use crate::error::{Error, Result};

use crate::converters::{SampleFormat, ScaleMode};

pub(super) fn register(f: &mut ElementFactory) {
    f.register("videotestsrc", create_videotestsrc);
    f.register("videoconvert", create_videoconvert);
    f.register("videoscale", create_videoscale);
    f.register("audioconvert", create_audioconvert);
    f.register("audioresample", create_audioresample);
    f.register("audiodownmix", create_audiodownmix);
    f.register("gain", create_gain);
    f.register("batch", create_batch);
    f.register("unbatch", create_unbatch);
    f.register("buffertrim", create_buffertrim);
    f.register("bufferslice", create_bufferslice);
    f.register("bufferpad", create_bufferpad);
    f.register("timestamper", create_timestamper);
    f.register("sequencenumber", create_sequencenumber);
    f.register("delay", create_delay);
    f.register("throttle", create_throttle);
    f.register("ratelimiter", create_ratelimiter);
    f.register("timeout", create_timeout);
    f.register("debounce", create_debounce);
}

const SAMPLE_FORMATS: &[(&str, SampleFormat)] = &[
    ("u8", SampleFormat::U8),
    ("s16", SampleFormat::S16Le),
    ("s16le", SampleFormat::S16Le),
    ("s16be", SampleFormat::S16Be),
    ("s32", SampleFormat::S32Le),
    ("s32le", SampleFormat::S32Le),
    ("s32be", SampleFormat::S32Be),
    ("f32", SampleFormat::F32Le),
    ("f32le", SampleFormat::F32Le),
    ("f32be", SampleFormat::F32Be),
];

fn create_videotestsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::testing::{PixelFormat, VideoPattern, VideoTestSrc};

    let mut src = VideoTestSrc::new();

    if let Some(pattern) = props.get_enum(
        "pattern",
        &[
            ("smpte", VideoPattern::SmpteColorBars),
            ("smpte-color-bars", VideoPattern::SmpteColorBars),
            ("checkerboard", VideoPattern::Checkerboard),
            ("solid", VideoPattern::SolidColor),
            ("ball", VideoPattern::MovingBall),
            ("moving-ball", VideoPattern::MovingBall),
            ("gradient", VideoPattern::Gradient),
            ("black", VideoPattern::Black),
            ("white", VideoPattern::White),
            ("red", VideoPattern::Red),
            ("green", VideoPattern::Green),
            ("blue", VideoPattern::Blue),
            ("circular", VideoPattern::Circular),
            ("snow", VideoPattern::Snow),
        ],
    )? {
        src = src.with_pattern(pattern);
    }

    if let Some((w, h)) = props.get_size()? {
        src = src.with_resolution(w, h);
    }
    if let Some(count) = props.get_u64("num-buffers")? {
        src = src.with_num_frames(count);
    }
    if let Some((num, den)) = props.get_framerate("framerate")? {
        src = src.with_framerate(num, den);
    }

    // RGBA default for display compatibility, overridable via `format`.
    let format = props
        .get_enum(
            "format",
            &[
                ("rgba", PixelFormat::Rgba32),
                ("rgb", PixelFormat::Rgb24),
                ("bgra", PixelFormat::Bgra32),
                ("bgr", PixelFormat::Bgr24),
            ],
        )?
        .unwrap_or(PixelFormat::Rgba32);
    src = src.with_pixel_format(format);

    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

const PIXEL_FORMATS: &[(&str, crate::converters::PixelFormat)] = &[
    ("rgb", crate::converters::PixelFormat::Rgb24),
    ("rgb24", crate::converters::PixelFormat::Rgb24),
    ("rgba", crate::converters::PixelFormat::Rgba),
    ("bgr", crate::converters::PixelFormat::Bgr24),
    ("bgr24", crate::converters::PixelFormat::Bgr24),
    ("bgra", crate::converters::PixelFormat::Bgra),
    ("i420", crate::converters::PixelFormat::I420),
    ("nv12", crate::converters::PixelFormat::Nv12),
    ("yuyv", crate::converters::PixelFormat::Yuyv),
    ("uyvy", crate::converters::PixelFormat::Uyvy),
    ("gray8", crate::converters::PixelFormat::Gray8),
];

fn create_videoconvert(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::transform::VideoConvertElement;

    let mut element = VideoConvertElement::new();
    if let Some(fmt) = props.get_enum("in-format", PIXEL_FORMATS)? {
        element = element.with_input_format(fmt);
    }
    if let Some(fmt) = props.get_enum("out-format", PIXEL_FORMATS)? {
        element = element.with_output_format(fmt);
    }
    if let Some((w, h)) = props.get_size()? {
        element = element.with_size(w, h);
    }
    Ok(DynAsyncElement::new_box(TransformAdapter::new(element)))
}

fn create_videoscale(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let mut scale = VideoScale::new();
    if let Some(mode) = props.get_enum(
        "method",
        &[
            ("bilinear", ScaleMode::Bilinear),
            ("nearest", ScaleMode::NearestNeighbor),
        ],
    )? {
        scale = scale.with_mode(mode);
    }
    // Geometry-in-Metadata: no dimensions required — an unconfigured
    // videoscale passes frames through untouched.
    if let Some((w, h)) = props.get_size()? {
        scale.control().set_target(w, h);
    }
    if let Some(w) = props.get_u32("max-width")? {
        scale.control().set_max_width(w);
    }
    if let Some(h) = props.get_u32("max-height")? {
        scale.control().set_max_height(h);
    }
    Ok(DynAsyncElement::new_box(ElementAdapter::new(scale)))
}

fn create_audioconvert(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let mut element = AudioConvertElement::new();
    if let Some(fmt) = props.get_enum("from", SAMPLE_FORMATS)? {
        element = element.with_input_format(fmt);
    }
    if let Some(fmt) = props.get_enum("to", SAMPLE_FORMATS)? {
        element = element.with_output_format(fmt);
    }
    if let Some(ch) = props.get_u32("channels")? {
        element = element.with_channels(ch);
    }
    Ok(DynAsyncElement::new_box(ElementAdapter::new(element)))
}

fn create_audioresample(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::converters::ResampleQuality;

    let mut element = AudioResampleElement::new()
        .with_input_rate(props.req_u32("in-rate")?)
        .with_output_rate(props.req_u32("out-rate")?);
    if let Some(ch) = props.get_u32("channels")? {
        element = element.with_channels(ch);
    }
    if let Some(fmt) = props.get_enum("format", SAMPLE_FORMATS)? {
        element = element.with_format(fmt);
    }
    if let Some(q) = props.get_enum(
        "quality",
        &[
            ("fast", ResampleQuality::Fast),
            ("medium", ResampleQuality::Medium),
        ],
    )? {
        element = element.with_quality(q);
    }
    Ok(DynAsyncElement::new_box(ElementAdapter::new(element)))
}

fn create_audiodownmix(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        AudioDownmix::new(),
    )))
}

fn create_gain(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let factor = props.req_f64("gain")? as f32;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(Gain::new(
        factor,
    ))))
}

fn create_batch(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let max_count = props.req_usize("max-count")?;
    let max_bytes = props.get_usize("max-bytes")?.unwrap_or(usize::MAX);
    let mut batch = Batch::with_limits(max_count, max_bytes);
    if let Some(t) = props.get_ms("timeout-ms")? {
        batch = batch.with_timeout(t);
    }
    Ok(DynAsyncElement::new_box(ElementAdapter::new(batch)))
}

fn create_unbatch(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let chunk = props.req_usize("chunk-size")?;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(Unbatch::new(
        chunk,
    ))))
}

fn create_buffertrim(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let max = props.req_usize("max-size")?;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        BufferTrim::new(max),
    )))
}

fn create_bufferslice(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let offset = props.req_usize("offset")?;
    let length = props.req_usize("length")?;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        BufferSlice::new(offset, length),
    )))
}

fn create_bufferpad(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let min = props.req_usize("min-size")?;
    let fill = props.get_u8("fill")?.unwrap_or(0);
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        BufferPad::new(min, fill),
    )))
}

fn create_timestamper(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let mode = props
        .get_enum(
            "mode",
            &[
                ("system", TimestampMode::SystemTime),
                ("monotonic", TimestampMode::Monotonic),
                ("preserve", TimestampMode::Preserve),
                ("pts", TimestampMode::PtsOnly),
                ("dts", TimestampMode::DtsOnly),
            ],
        )?
        .unwrap_or_default();
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        Timestamper::new(mode),
    )))
}

fn create_sequencenumber(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let mut sn = SequenceNumber::new();
    if let Some(inc) = props.get_u64("increment")? {
        sn = sn.with_increment(inc);
    }
    Ok(DynAsyncElement::new_box(ElementAdapter::new(sn)))
}

fn create_delay(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let delay = props.req_ms("ms")?;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(Delay::new(
        delay,
    ))))
}

fn create_throttle(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let throttle = match (props.get_u64("interval-ms")?, props.get_f64("rate")?) {
        (Some(ms), None) => Throttle::from_millis(ms),
        (None, Some(rate)) if rate > 0.0 => {
            Throttle::new(std::time::Duration::from_secs_f64(1.0 / rate))
        }
        (None, Some(_)) => {
            return Err(Error::Parse(
                "throttle: 'rate' must be positive".to_string(),
            ));
        }
        _ => {
            return Err(Error::Parse(
                "throttle requires exactly one of 'interval-ms' or 'rate'".to_string(),
            ));
        }
    };
    Ok(DynAsyncElement::new_box(ElementAdapter::new(throttle)))
}

fn create_ratelimiter(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let limiter = match (
        props.get_f64("buffers-per-second")?,
        props.get_u64("bytes-per-second")?,
        props.get_ms("delay-ms")?,
    ) {
        (Some(r), None, None) => RateLimiter::buffers_per_second(r),
        (None, Some(r), None) => RateLimiter::bytes_per_second(r),
        (None, None, Some(d)) => RateLimiter::fixed_delay(d),
        _ => {
            return Err(Error::Parse(
                "ratelimiter requires exactly one of 'buffers-per-second', \
                 'bytes-per-second', or 'delay-ms'"
                    .to_string(),
            ));
        }
    };
    Ok(DynAsyncElement::new_box(ElementAdapter::new(limiter)))
}

fn create_timeout(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let t = props.req_ms("ms")?;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(Timeout::new(
        t,
    ))))
}

fn create_debounce(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let t = props.req_ms("ms")?;
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        Debounce::new(t),
    )))
}
