//! Factory registrations: cross-process IPC elements.

use super::{ElementFactory, Props};
use crate::element::{AsyncSinkAdapter, AsyncSourceAdapter, DynAsyncElement};
use crate::elements::{IpcSink, IpcSrc};
use crate::error::Result;

pub(super) fn register(f: &mut ElementFactory) {
    f.register("ipcsrc", create_ipcsrc);
    f.register("ipcsink", create_ipcsink);
}

fn create_ipcsrc(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let path = props.req_str("path")?;
    Ok(DynAsyncElement::new_box(AsyncSourceAdapter::new(
        IpcSrc::new(path),
    )))
}

fn create_ipcsink(props: &Props) -> Result<Box<DynAsyncElement<'static>>> {
    let path = props.req_str("path")?;
    let mut sink = IpcSink::new(path);
    if let Some(cap) = props.get_u32("capacity")? {
        sink = sink.with_capacity(cap);
    }
    Ok(DynAsyncElement::new_box(AsyncSinkAdapter::new(sink)))
}
