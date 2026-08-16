//! Factory registrations: device capture/playback elements (all gated).

use super::ElementFactory;
#[cfg(any(
    feature = "display",
    feature = "v4l2",
    feature = "alsa",
    feature = "screen-capture",
    feature = "pipewire",
    feature = "libcamera"
))]
use super::Props;
#[cfg(any(
    feature = "display",
    feature = "v4l2",
    feature = "alsa",
    feature = "screen-capture",
    feature = "pipewire",
    feature = "libcamera"
))]
use crate::element::DynAsyncElement;
#[cfg(any(
    feature = "display",
    feature = "v4l2",
    feature = "alsa",
    feature = "screen-capture",
    feature = "pipewire",
    feature = "libcamera"
))]
use crate::error::Result;

pub(super) fn register(f: &mut ElementFactory) {
    #[cfg(feature = "display")]
    f.register("autovideosink", create_autovideosink);
    #[cfg(feature = "v4l2")]
    f.register("v4l2src", create_v4l2src);
    #[cfg(feature = "alsa")]
    {
        f.register("alsasrc", create_alsasrc);
        f.register("alsasink", create_alsasink);
    }
    #[cfg(feature = "screen-capture")]
    f.register("screencapsrc", create_screencapsrc);
    #[cfg(feature = "pipewire")]
    {
        f.register("pipewiresrc", create_pipewiresrc);
        f.register("pipewiresink", create_pipewiresink);
    }
    #[cfg(feature = "libcamera")]
    f.register("libcamerasrc", create_libcamerasrc);
    let _ = f;
}

#[cfg(feature = "display")]
fn create_autovideosink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::element::AsyncSinkAdapter;
    use crate::elements::app::AutoVideoSink;

    let mut sink = AutoVideoSink::new();

    if let Some(title) = props.get_str("title") {
        sink = sink.with_title(title);
    }
    if let Some((w, h)) = props.get_size()? {
        sink = sink.with_size(w, h);
    }
    // `sync=true` plays the stream at its own speed instead of as fast as it
    // decodes. Off by default so capture previews are unaffected.
    if let Some(sync) = props.get_bool("sync")? {
        sink = sink.with_sync(sync);
    }
    if let Some(t) = props.get_ms("max-lateness-ms")? {
        sink = sink.with_max_lateness(t);
    }

    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(sink)))
}

#[cfg(feature = "v4l2")]
fn create_v4l2src(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::element::SourceAdapter;
    use crate::elements::device::v4l2::{V4l2Config, V4l2Src};

    let device = props
        .get_str("device")
        .unwrap_or_else(|| "/dev/video0".to_string());

    let (width, height) = props.get_size()?.unwrap_or((640, 480));

    // Default to YUYV for compatibility with videoconvert
    // MJPEG requires a decoder which we don't have yet
    let fourcc = props.get_str("format").or_else(|| Some("YUYV".to_string()));

    let buffer_count = props.get_u32("buffer-count")?.unwrap_or(4);
    let dmabuf_export = props.get_bool("dmabuf-export")?.unwrap_or(false);
    let framerate = props.get_framerate("framerate")?;

    let config = V4l2Config {
        width,
        height,
        fourcc,
        buffer_count,
        framerate,
        dmabuf_export,
    };

    let src = V4l2Src::with_config(&device, config)?;
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

#[cfg(feature = "alsa")]
fn alsa_format(props: &Props) -> Result<crate::elements::device::alsa::AlsaFormat> {
    use crate::elements::device::alsa::{AlsaFormat, AlsaSampleFormat};

    let mut format = AlsaFormat::default();
    if let Some(rate) = props.get_u32("rate")? {
        format.sample_rate = rate;
    }
    if let Some(ch) = props.get_u32("channels")? {
        format.channels = ch;
    }
    if let Some(fmt) = props.get_enum(
        "format",
        &[
            ("s16", AlsaSampleFormat::S16LE),
            ("s32", AlsaSampleFormat::S32LE),
            ("f32", AlsaSampleFormat::F32LE),
            ("u8", AlsaSampleFormat::U8),
        ],
    )? {
        format.format = fmt;
    }
    if let Some(n) = props.get_u32("buffer-frames")? {
        format.buffer_frames = n;
    }
    if let Some(n) = props.get_u32("period-frames")? {
        format.period_frames = n;
    }
    Ok(format)
}

#[cfg(feature = "alsa")]
fn create_alsasrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::element::AsyncSourceAdapter;
    use crate::elements::device::alsa::AlsaSrc;

    let device = props.get_str("device").unwrap_or_else(|| "default".into());
    let src = AlsaSrc::new(&device, alsa_format(props)?)?;
    Ok(DynAsyncElement::new_box(AsyncSourceAdapter::new(src)))
}

#[cfg(feature = "alsa")]
fn create_alsasink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::element::AsyncSinkAdapter;
    use crate::elements::device::alsa::AlsaSink;

    let device = props.get_str("device").unwrap_or_else(|| "default".into());
    let sink = AlsaSink::new(&device, alsa_format(props)?)?;
    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(sink)))
}

#[cfg(feature = "screen-capture")]
fn create_screencapsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::element::SourceAdapter;
    use crate::elements::device::screen_capture::{
        CaptureSourceType, ScreenCaptureConfig, ScreenCaptureSrc,
    };

    let mut config = ScreenCaptureConfig::default();
    if let Some(st) = props.get_enum(
        "source-type",
        &[
            ("monitor", CaptureSourceType::Monitor),
            ("window", CaptureSourceType::Window),
            ("any", CaptureSourceType::Any),
        ],
    )? {
        config.source_type = st;
    }
    if let Some(cursor) = props.get_bool("cursor")? {
        config.show_cursor = cursor;
    }
    if let Some(n) = props.get_u32("max-frames")? {
        config.max_frames = Some(n);
    }

    Ok(DynAsyncElement::new_box(SourceAdapter::new(
        ScreenCaptureSrc::new(config),
    )))
}

#[cfg(feature = "pipewire")]
fn create_pipewiresrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::element::AsyncSourceAdapter;
    use crate::elements::device::pipewire::PipeWireSrc;

    // Audio only: video capture needs a PipeWireTarget (a portal handle),
    // which a flat property string cannot express.
    let device = props.get_str("device");
    let src = PipeWireSrc::audio(device.as_deref())?;
    Ok(DynAsyncElement::new_box(AsyncSourceAdapter::new(src)))
}

#[cfg(feature = "pipewire")]
fn create_pipewiresink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::element::AsyncSinkAdapter;
    use crate::elements::device::pipewire::PipeWireSink;

    let device = props.get_str("device");
    let sink = PipeWireSink::audio(device.as_deref())?;
    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(sink)))
}

#[cfg(feature = "libcamera")]
fn create_libcamerasrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::element::AsyncSourceAdapter;
    use crate::elements::device::libcamera::LibCameraSrc;

    let src = match props.get_str("camera") {
        Some(id) => LibCameraSrc::with_camera(&id)?,
        None => LibCameraSrc::new()?,
    };
    Ok(DynAsyncElement::new_box(AsyncSourceAdapter::new(src)))
}
