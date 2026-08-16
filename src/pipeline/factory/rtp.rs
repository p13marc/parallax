//! Factory registrations: RTP elements (feature `rtp`).

use super::ElementFactory;
#[cfg(feature = "rtp")]
use super::Props;
#[cfg(feature = "rtp")]
use crate::element::{DynAsyncElement, ElementAdapter, SinkAdapter, SourceAdapter};
#[cfg(feature = "rtp")]
use crate::error::Result;

pub(super) fn register(f: &mut ElementFactory) {
    #[cfg(feature = "rtp")]
    {
        f.register("rtpsrc", create_rtpsrc);
        f.register("rtpsink", create_rtpsink);
        f.register("rtpjitterbuffer", create_rtpjitterbuffer);
        f.register("rtph264pay", |p| pay(p, Pay::H264));
        f.register("rtph264depay", |p| depay(p, Depay::H264));
        f.register("rtph265pay", |p| pay(p, Pay::H265));
        f.register("rtph265depay", |p| depay(p, Depay::H265));
        f.register("rtpvp8pay", |p| pay(p, Pay::Vp8));
        f.register("rtpvp8depay", |p| depay(p, Depay::Vp8));
        f.register("rtpvp9pay", |p| pay(p, Pay::Vp9));
        f.register("rtpvp9depay", |p| depay(p, Depay::Vp9));
        f.register("rtpopuspay", |p| pay(p, Pay::Opus));
        f.register("rtpopusdepay", |p| depay(p, Depay::Opus));
        f.register("rtpav1pay", |p| pay(p, Pay::Av1));
    }
    let _ = f;
}

#[cfg(feature = "rtp")]
#[derive(Clone, Copy)]
enum Pay {
    H264,
    H265,
    Vp8,
    Vp9,
    Opus,
    Av1,
}

#[cfg(feature = "rtp")]
#[derive(Clone, Copy)]
enum Depay {
    H264,
    H265,
    Vp8,
    Vp9,
    Opus,
}

#[cfg(feature = "rtp")]
fn pay(props: &Props, kind: Pay) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::rtp::{
        RtpAv1Pay, RtpH264Pay, RtpH265Pay, RtpOpusPay, RtpVp8Pay, RtpVp9Pay,
    };

    let mtu = props.get_usize("mtu")?;
    macro_rules! build {
        ($ty:ident) => {{
            let mut p = $ty::new();
            if let Some(mtu) = mtu {
                p = p.with_mtu(mtu);
            }
            DynAsyncElement::new_box(ElementAdapter::new(p))
        }};
    }
    Ok(match kind {
        Pay::H264 => build!(RtpH264Pay),
        Pay::H265 => build!(RtpH265Pay),
        Pay::Vp8 => build!(RtpVp8Pay),
        Pay::Vp9 => build!(RtpVp9Pay),
        Pay::Opus => build!(RtpOpusPay),
        Pay::Av1 => build!(RtpAv1Pay),
    })
}

#[cfg(feature = "rtp")]
fn depay(_props: &Props, kind: Depay) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::rtp::{
        RtpH264Depay, RtpH265Depay, RtpOpusDepay, RtpVp8Depay, RtpVp9Depay,
    };

    Ok(match kind {
        Depay::H264 => DynAsyncElement::new_box(ElementAdapter::new(RtpH264Depay::new())),
        Depay::H265 => DynAsyncElement::new_box(ElementAdapter::new(RtpH265Depay::new())),
        Depay::Vp8 => DynAsyncElement::new_box(ElementAdapter::new(RtpVp8Depay::new())),
        Depay::Vp9 => DynAsyncElement::new_box(ElementAdapter::new(RtpVp9Depay::new())),
        Depay::Opus => DynAsyncElement::new_box(ElementAdapter::new(RtpOpusDepay::new())),
    })
}

#[cfg(feature = "rtp")]
fn create_rtpsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::rtp::RtpSrc;

    let address = props.req_str("address")?;
    let mut src = RtpSrc::bind(address.as_str())?;
    if let Some(pt) = props.get_u8("payload-type")? {
        src = src.with_payload_type(pt);
    }
    if let Some(rate) = props.get_u32("clock-rate")? {
        src = src.with_clock_rate(rate);
    }
    if let Some(size) = props.get_usize("buffer-size")? {
        src = src.with_buffer_size(size);
    }
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

#[cfg(feature = "rtp")]
fn create_rtpsink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::rtp::RtpSink;

    let address = props.req_str("address")?;
    let mut sink = RtpSink::connect(address.as_str())?;
    if let Some(pt) = props.get_u8("payload-type")? {
        sink = sink.with_payload_type(pt);
    }
    if let Some(ssrc) = props.get_u32("ssrc")? {
        sink = sink.with_ssrc(ssrc);
    }
    if let Some(rate) = props.get_u32("clock-rate")? {
        sink = sink.with_clock_rate(rate);
    }
    Ok(DynAsyncElement::new_box(SinkAdapter::new(sink)))
}

#[cfg(feature = "rtp")]
fn create_rtpjitterbuffer(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::rtp::RtpJitterBuffer;

    let mut jb = RtpJitterBuffer::new();
    if let Some(ms) = props.get_u64("latency-ms")? {
        jb = jb.with_latency_ms(ms);
    }
    if let Some(n) = props.get_usize("max-packets")? {
        jb = jb.with_max_packets(n);
    }
    if let Some(rate) = props.get_u32("clock-rate")? {
        jb = jb.with_clock_rate(rate);
    }
    if let Some(drop) = props.get_bool("drop-late")? {
        jb = jb.with_drop_late(drop);
    }
    Ok(DynAsyncElement::new_box(ElementAdapter::new(jb)))
}
