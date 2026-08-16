//! Factory registrations: io, testing, util, and flow elements.

use super::{ElementFactory, Props};
use crate::element::{
    AsyncSinkAdapter, DynAsyncElement, ElementAdapter, SinkAdapter, SourceAdapter,
};
use crate::elements::{
    ConsoleFormat, ConsoleSink, DataSrc, FdSink, FdSrc, FileSink, FileSrc, Identity, Inspect,
    NullSink, NullSource, PassThrough, Queue2, TestPattern, TestSrc, Valve,
};
use crate::error::Result;

pub(super) fn register(f: &mut ElementFactory) {
    f.register("nullsource", create_nullsource);
    f.register("nullsink", create_nullsink);
    f.register("passthrough", create_passthrough);
    f.register("identity", create_identity);
    f.register("inspect", create_inspect);
    // Deprecated alias: "tee" never fanned out — it is a passthrough counter.
    // Kept so existing pipeline strings keep parsing.
    f.register("tee", create_inspect);
    f.register("filesrc", create_filesrc);
    f.register("filesink", create_filesink);
    f.register("fdsrc", create_fdsrc);
    f.register("fdsink", create_fdsink);
    f.register("consolesink", create_consolesink);
    f.register("datasrc", create_datasrc);
    f.register("testsrc", create_testsrc);
    f.register("valve", create_valve);
    f.register("queue2", create_queue2);
}

fn create_nullsource(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let count = props.get_u64("count")?.unwrap_or(100);
    let buffer_size = props.get_usize("buffer-size")?.unwrap_or(64);

    let source = NullSource::new(count).with_buffer_size(buffer_size);
    Ok(DynAsyncElement::new_box(SourceAdapter::new(source)))
}

fn create_nullsink(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    Ok(DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())))
}

fn create_passthrough(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        PassThrough::new(),
    )))
}

fn create_identity(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        Identity::new(),
    )))
}

fn create_inspect(_props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    Ok(DynAsyncElement::new_box(
        ElementAdapter::new(Inspect::new()),
    ))
}

fn create_filesrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let location = props.req_str("location")?;
    let mut src = FileSrc::new(&location);
    if let Some(size) = props.get_usize("chunk-size")? {
        src = src.with_chunk_size(size);
    }
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

fn create_filesink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let location = props.req_str("location")?;
    let sink = FileSink::new(&location);
    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(sink)))
}

fn create_fdsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let fd = props.req_i64("fd")? as std::os::fd::RawFd;
    Ok(DynAsyncElement::new_box(SourceAdapter::new(FdSrc::new(fd))))
}

fn create_fdsink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let fd = props.req_i64("fd")? as std::os::fd::RawFd;
    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(
        FdSink::new(fd),
    )))
}

fn create_consolesink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let format = props
        .get_enum(
            "format",
            &[
                ("metadata", ConsoleFormat::Metadata),
                ("hex", ConsoleFormat::Hex),
                ("text", ConsoleFormat::Text),
                ("full", ConsoleFormat::Full),
            ],
        )?
        .unwrap_or_default();
    let mut sink = ConsoleSink::new().format(format);
    if let Some(prefix) = props.get_str("prefix") {
        sink = sink.prefix(prefix);
    }
    Ok(DynAsyncElement::new_box(SinkAdapter::new(sink)))
}

fn create_datasrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let data = props.req_str("data")?;
    let mut src = DataSrc::from_string(&data);
    if let Some(size) = props.get_usize("chunk-size")? {
        src = src.with_chunk_size(size);
    }
    if let Some(n) = props.get_usize("repeat-count")? {
        src = src.repeat_n(n);
    } else if props.get_bool("repeat")?.unwrap_or(false) {
        src = src.repeat();
    }
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

fn create_testsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let mut src = TestSrc::new();
    if let Some(pattern) = props.get_enum(
        "pattern",
        &[
            ("zero", TestPattern::Zero),
            ("ones", TestPattern::Ones),
            ("counter", TestPattern::Counter),
            ("random", TestPattern::Random),
            ("alternating", TestPattern::Alternating),
            ("sequence", TestPattern::Sequence),
        ],
    )? {
        src = src.with_pattern(pattern);
    }
    if let Some(n) = props.get_u64("num-buffers")? {
        src = src.with_num_buffers(n);
    }
    if let Some(size) = props.get_usize("buffer-size")? {
        src = src.with_buffer_size(size);
    }
    if let Some(rate) = props.get_u64("rate")? {
        src = src.with_rate(rate);
    }
    if let Some(seed) = props.get_u64("seed")? {
        src = src.with_seed(seed);
    }
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

fn create_valve(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let open = props.get_bool("open")?.unwrap_or(true);
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        Valve::with_state(open),
    )))
}

fn create_queue2(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    const DEFAULT_MAX_BYTES: usize = 4 * 1024 * 1024;
    let max = props
        .get_usize("max-size-bytes")?
        .unwrap_or(DEFAULT_MAX_BYTES);
    let mut q = Queue2::stream(max);
    if let (Some(low), Some(high)) = (
        props.get_u32("low-percent")?,
        props.get_u32("high-percent")?,
    ) {
        q = q.with_watermarks(low, high);
    }
    Ok(DynAsyncElement::new_box(ElementAdapter::new(q)))
}
