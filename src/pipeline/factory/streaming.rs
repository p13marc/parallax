//! Factory registrations: adaptive-streaming sinks (HLS, DASH).

use super::{ElementFactory, Props};
use crate::element::{DynAsyncElement, PipelineElementAdapter, Snk};
use crate::elements::streaming::{DashConfig, DashSink, HlsConfig, HlsSink};
use crate::error::Result;

pub(super) fn register(f: &mut ElementFactory) {
    f.register("hlssink", create_hlssink);
    f.register("dashsink", create_dashsink);
}

fn create_hlssink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let mut config = HlsConfig::default();
    if let Some(dir) = props.get_str("location") {
        config.output_dir = dir.into();
    }
    if let Some(d) = props.get_f64("target-duration")? {
        config.segment_duration = d;
    }
    if let Some(n) = props.get_u32("playlist-length")? {
        config.playlist_length = n;
    }
    if let Some(name) = props.get_str("playlist-name") {
        config.playlist_name = name;
    }
    if let Some(prefix) = props.get_str("segment-prefix") {
        config.segment_prefix = prefix;
    }
    if let Some(vod) = props.get_bool("vod")? {
        config.is_vod = vod;
    }
    let sink = HlsSink::new(config)?;
    Ok(DynAsyncElement::new_box(PipelineElementAdapter::new(Snk(
        sink,
    ))))
}

fn create_dashsink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let mut config = DashConfig::default();
    if let Some(dir) = props.get_str("location") {
        config.output_dir = dir.into();
    }
    if let Some(d) = props.get_f64("segment-duration")? {
        config.segment_duration = d;
    }
    if let Some(n) = props.get_u32("segment-window")? {
        config.segment_window = n;
    }
    if let Some(name) = props.get_str("manifest-name") {
        config.manifest_name = name;
    }
    if let Some(prefix) = props.get_str("segment-prefix") {
        config.segment_prefix = prefix;
    }
    if let Some(live) = props.get_bool("live")? {
        config.is_live = live;
    }
    let sink = DashSink::new(config)?;
    Ok(DynAsyncElement::new_box(PipelineElementAdapter::new(Snk(
        sink,
    ))))
}
