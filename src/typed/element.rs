//! Typed element traits with compile-time type safety.

use std::marker::PhantomData;

use crate::buffer::Buffer;
use crate::error::Result;

/// A typed source that produces values of type `T`.
///
/// This trait is the typed counterpart to [`crate::element::Source`].
/// The output type is known at compile time.
pub trait TypedSource: Send {
    /// The type of data this source produces.
    type Output: Send + 'static;

    /// Produce the next item.
    ///
    /// Returns `Ok(None)` when the source is exhausted.
    fn produce(&mut self) -> Result<Option<Self::Output>>;

    /// Get the name of this source.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// A typed sink that consumes values of type `T`.
///
/// This trait is the typed counterpart to [`crate::element::Sink`].
/// The input type is known at compile time.
pub trait TypedSink: Send {
    /// The type of data this sink consumes.
    type Input: Send + 'static;

    /// Consume an item.
    fn consume(&mut self, item: Self::Input) -> Result<()>;

    /// Get the name of this sink.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// A typed transform element that converts `Input` to `Output`.
///
/// This trait is the typed counterpart to [`crate::element::Element`].
/// Both input and output types are known at compile time.
pub trait TypedTransform: Send {
    /// The type of data this transform accepts.
    type Input: Send + 'static;

    /// The type of data this transform produces.
    type Output: Send + 'static;

    /// Transform an input item into an output item.
    ///
    /// Returns `Ok(None)` to filter out (drop) the item.
    fn transform(&mut self, input: Self::Input) -> Result<Option<Self::Output>>;

    /// Get the name of this transform.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Marker trait for caps types.
///
/// Caps describe the format of data flowing through a pipeline.
/// In the typed API, caps are represented as Rust types.
pub trait Caps: Send + 'static {}

/// Raw bytes caps - untyped binary data.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bytes;
impl Caps for Bytes {}

/// Typed data caps - data with a known Rust type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Typed<T>(PhantomData<T>);
impl<T: Send + 'static> Caps for Typed<T> {}

/// Video caps with format, resolution, and framerate.
#[derive(Debug, Clone, Copy)]
pub struct Video<Fmt, const W: u32, const H: u32, const FPS: u32> {
    _fmt: PhantomData<Fmt>,
}
impl<Fmt: Send + 'static, const W: u32, const H: u32, const FPS: u32> Caps
    for Video<Fmt, W, H, FPS>
{
}

/// Audio caps with format and sample rate.
#[derive(Debug, Clone, Copy)]
pub struct Audio<Fmt, const RATE: u32> {
    _fmt: PhantomData<Fmt>,
}
impl<Fmt: Send + 'static, const RATE: u32> Caps for Audio<Fmt, RATE> {}

/// Telemetry data caps for sensor/monitoring data.
#[derive(Debug, Clone, Copy)]
pub struct Telemetry<T>(PhantomData<T>);
impl<T: Send + 'static> Caps for Telemetry<T> {}

// ============================================================================
// Typed Buffer Wrapper
// ============================================================================

/// A buffer with compile-time type information.
///
/// This wraps a dynamic [`Buffer`] but carries type information
/// about the data format.
pub struct TypedBuffer<C: Caps> {
    inner: Buffer,
    _caps: PhantomData<C>,
}

impl<C: Caps> TypedBuffer<C> {
    /// Create a typed buffer from a dynamic buffer.
    pub fn new(buffer: Buffer) -> Self {
        Self {
            inner: buffer,
            _caps: PhantomData,
        }
    }

    /// Get the underlying dynamic buffer.
    pub fn into_inner(self) -> Buffer {
        self.inner
    }

    /// Get a reference to the underlying buffer.
    pub fn as_buffer(&self) -> &Buffer {
        &self.inner
    }

    /// Get a mutable reference to the underlying buffer.
    pub fn as_buffer_mut(&mut self) -> &mut Buffer {
        &mut self.inner
    }

    /// Convert to a different caps type (unsafe - caller must ensure validity).
    pub fn cast<C2: Caps>(self) -> TypedBuffer<C2> {
        TypedBuffer {
            inner: self.inner,
            _caps: PhantomData,
        }
    }
}

impl<C: Caps> std::ops::Deref for TypedBuffer<C> {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<C: Caps> std::ops::DerefMut for TypedBuffer<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::CpuSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    #[test]
    fn test_caps_types_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Bytes>();
        assert_send::<Typed<u32>>();
        assert_send::<Telemetry<f32>>();
    }

    #[test]
    fn test_typed_buffer() {
        let segment = Arc::new(CpuSegment::new(4).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::default());

        let typed: TypedBuffer<Bytes> = TypedBuffer::new(buffer);
        assert_eq!(typed.len(), 4);

        let casted: TypedBuffer<Typed<u32>> = typed.cast();
        assert_eq!(casted.len(), 4);
    }
}
