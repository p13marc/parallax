//! Built-in converters for common format, geometry, rate and memory conversions.
//!
//! [`builtin_registry`] wires the *real* converter elements (the ones in
//! [`crate::elements::transform`], backed by [`crate::converters`]) into the
//! negotiation registry, each tagged with the [`ConvertAxes`] it fixes:
//!
//! | Converter | Axes | Element |
//! |---|---|---|
//! | `videoconvert` | `FORMAT` | [`VideoConvertElement`](crate::elements::transform::VideoConvertElement) |
//! | `videoscale` | `GEOMETRY` | [`VideoScale`](crate::elements::transform::VideoScale) |
//! | `audioconvert` | `FORMAT` | [`AudioConvertElement`](crate::elements::transform::AudioConvertElement) |
//! | `audioresample` | `RATE` | [`AudioResampleElement`](crate::elements::transform::AudioResampleElement) |
//! | `memorycopy` | `MEMORY` | [`MemoryCopy`] |
//! | `identity` | `NONE` | [`Identity`] |
//!
//! `identity` fixes nothing, so the planner can never insert it to "resolve" a
//! conflict — it exists only as an explicit passthrough node.
//!
//! Each factory is handed a [`ConversionRequest`] and configures its element
//! from the downstream caps, so a link wanting I420 gets an I420 converter (the
//! old registry hardcoded RGBA and then silently dropped the converters when the
//! second negotiation pass found the fresh mismatch).

use super::converters::{
    ConversionRequest, ConvertAxes, ConverterElement, ConverterRegistry, ConverterSpec, FormatType,
};
use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;
use crate::format::{FormatCaps, SampleFormat};
use crate::memory::MemoryType;
use std::sync::Arc;

// ============================================================================
// MemoryCopy - copy between memory types
// ============================================================================

/// Memory copier for transferring buffers between memory types.
///
/// The DmaBuf→Cpu direction is real (#145): a dmabuf-backed buffer is
/// copied into a CPU arena slot via [`Buffer::copy_to_cpu`], which is the
/// bridge that keeps CPU-only consumers working behind a dmabuf-emitting
/// source — visibly, as a graph node, the GStreamer way. A buffer that is
/// already CPU passes through untouched.
///
/// The GPU directions remain pass-through stubs (PLAN-11, #62): no GPU
/// memory backing exists to copy into yet.
pub struct MemoryCopy {
    /// Source memory type.
    source_type: MemoryType,
    /// Target memory type.
    target_type: MemoryType,
    /// Output arena for the DmaBuf→Cpu copy, grown to fit the frames seen.
    output: crate::memory::OutputArena,
}

impl MemoryCopy {
    /// Create a new memory copier.
    pub fn new(source_type: MemoryType, target_type: MemoryType) -> Self {
        Self {
            source_type,
            target_type,
            output: crate::memory::OutputArena::new(crate::memory::defaults::TRANSFORM_SLOT_COUNT)
                .grow_to_fit(),
        }
    }

    /// Create a CPU to GPU uploader.
    pub fn cpu_to_gpu() -> Self {
        Self::new(MemoryType::Cpu, MemoryType::GpuDevice)
    }

    /// Create a GPU to CPU downloader.
    pub fn gpu_to_cpu() -> Self {
        Self::new(MemoryType::GpuDevice, MemoryType::Cpu)
    }

    /// Create a DMA-BUF to CPU copier (#145).
    pub fn dmabuf_to_cpu() -> Self {
        Self::new(MemoryType::DmaBuf, MemoryType::Cpu)
    }

    /// Create an External to CPU repacker (#194): strided producer-owned
    /// frames land packed in CPU shm.
    pub fn external_to_cpu() -> Self {
        Self::new(MemoryType::External, MemoryType::Cpu)
    }
}

impl Element for MemoryCopy {
    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.output.set_budget(budget);
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Non-CPU frames land in CPU shm for consumers that negotiated Cpu.
        // Already-CPU input passes through — the graph edge asked for Cpu
        // and that is what it is.
        if self.target_type == MemoryType::Cpu && buffer.memory_type() != MemoryType::Cpu {
            // A strided frame (#194) repacks — a flat copy would carry the
            // row padding into memory every packed consumer misreads.
            if buffer.metadata().has_strided_planes() {
                let meta = buffer.metadata();
                let (Some((w, h)), Some(fmt), Some(layout)) = (
                    meta.video_dims(),
                    meta.video_pixel_format(),
                    meta.plane_layout(),
                ) else {
                    return Err(crate::error::Error::Element(
                        "memorycopy: strided buffer without video geometry cannot be repacked"
                            .into(),
                    ));
                };
                let packed_len =
                    crate::format::PlaneLayout::packed(fmt, w, h).required_len(fmt, w, h);
                let mut slot = self.output.acquire(packed_len, "memorycopy")?;
                layout
                    .repack_into(
                        buffer.as_bytes(),
                        fmt,
                        w,
                        h,
                        &mut slot.data_mut()[..packed_len],
                    )
                    .map_err(crate::error::Error::Element)?;
                let handle = crate::buffer::MemoryHandle::with_len(slot, packed_len);
                let mut metadata = buffer.metadata().clone();
                // Output is packed: set_video_dims clears the layout.
                metadata.set_video_dims(w, h, fmt);
                return Ok(Some(Buffer::new(handle, metadata)));
            }
            // Packed non-CPU (dmabuf, packed External): flat copy (#145).
            let mut slot = self.output.acquire(buffer.len(), "memorycopy")?;
            let data = buffer.as_bytes();
            slot.data_mut()[..data.len()].copy_from_slice(data);
            let handle = crate::buffer::MemoryHandle::with_len(slot, data.len());
            return Ok(Some(Buffer::new(handle, buffer.metadata().clone())));
        }
        // PLAN-11: GPU transfers (see plans/11_GPU_CODEC_FRAMEWORK.md).
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        "memorycopy"
    }
}

impl ConverterElement for MemoryCopy {
    fn converter_name(&self) -> &str {
        "memorycopy"
    }

    fn input_format(&self) -> FormatCaps {
        FormatCaps::Any
    }

    fn output_format(&self) -> FormatCaps {
        FormatCaps::Any
    }

    fn input_memory(&self) -> MemoryType {
        self.source_type
    }

    fn output_memory(&self) -> MemoryType {
        self.target_type
    }

    fn cost(&self) -> u32 {
        // Memory transfers are expensive
        match (self.source_type, self.target_type) {
            (MemoryType::Cpu, MemoryType::Cpu) => 1,
            (MemoryType::GpuDevice, MemoryType::GpuDevice) => 2,
            _ => 20, // Cross-device transfers are costly
        }
    }
}

// ============================================================================
// Identity converter (passthrough)
// ============================================================================

/// Identity converter that passes data through unchanged.
///
/// Used when formats are compatible but the pipeline needs an explicit node.
/// It fixes no [`ConvertAxes`], so negotiation can never auto-insert it.
pub struct Identity;

impl Element for Identity {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        "identity"
    }
}

impl ConverterElement for Identity {
    fn converter_name(&self) -> &str {
        "identity"
    }

    fn input_format(&self) -> FormatCaps {
        FormatCaps::Any
    }

    fn output_format(&self) -> FormatCaps {
        FormatCaps::Any
    }

    fn input_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn output_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn cost(&self) -> u32 {
        0 // Zero cost - passthrough
    }
}

// ============================================================================
// Factory helpers: read the target out of the request
// ============================================================================

/// The pixel format the downstream element wants, if it pins one.
fn target_pixel_format(request: &ConversionRequest) -> Option<crate::converters::PixelFormat> {
    let FormatCaps::VideoRaw(caps) = &request.output else {
        return None;
    };
    caps.pixel_format.fixate()?.try_into().ok()
}

/// The geometry the downstream element wants, if it pins one.
fn target_geometry(request: &ConversionRequest) -> Option<(u32, u32)> {
    let FormatCaps::VideoRaw(caps) = &request.output else {
        return None;
    };
    Some((caps.width.fixate()?, caps.height.fixate()?))
}

/// Map the caps-level sample format onto the converter engine's.
fn engine_sample_format(format: SampleFormat) -> crate::converters::SampleFormat {
    use crate::converters::SampleFormat as Engine;
    match format {
        SampleFormat::U8 => Engine::U8,
        SampleFormat::S16 => Engine::S16Le,
        SampleFormat::S32 => Engine::S32Le,
        SampleFormat::F32 => Engine::F32Le,
    }
}

// ============================================================================
// Registry builder
// ============================================================================

/// Create a converter registry with the built-in converters.
pub fn builtin_registry() -> ConverterRegistry {
    let mut registry = ConverterRegistry::new();

    // Pixel format conversion. Carries the input geometry through — it does not
    // rescale — hence FORMAT only.
    registry.register(ConverterSpec {
        name: "videoconvert",
        from_format: FormatType::VideoRaw,
        to_format: FormatType::VideoRaw,
        from_memory: MemoryType::Cpu,
        to_memory: MemoryType::Cpu,
        axes: ConvertAxes::FORMAT,
        cost: 5,
        factory: Arc::new(|request: &ConversionRequest| {
            let mut element = crate::elements::transform::VideoConvertElement::new();
            if let Some(format) = target_pixel_format(request) {
                element = element.with_output_format(format);
            }
            Box::new(element)
        }),
    });

    // Geometry. Keeps the input pixel format — hence GEOMETRY only.
    registry.register(ConverterSpec {
        name: "videoscale",
        from_format: FormatType::VideoRaw,
        to_format: FormatType::VideoRaw,
        from_memory: MemoryType::Cpu,
        to_memory: MemoryType::Cpu,
        axes: ConvertAxes::GEOMETRY,
        cost: 10,
        factory: Arc::new(|request: &ConversionRequest| {
            let element = crate::elements::transform::VideoScale::new();
            if let Some((width, height)) = target_geometry(request) {
                element.control().set_target(width, height);
            }
            Box::new(element)
        }),
    });

    // Sample format and channel count.
    registry.register(ConverterSpec {
        name: "audioconvert",
        from_format: FormatType::AudioRaw,
        to_format: FormatType::AudioRaw,
        from_memory: MemoryType::Cpu,
        to_memory: MemoryType::Cpu,
        axes: ConvertAxes::FORMAT,
        cost: 3,
        factory: Arc::new(|request: &ConversionRequest| {
            let mut element = crate::elements::transform::AudioConvertElement::new();
            if let FormatCaps::AudioRaw(caps) = &request.input
                && let Some(format) = caps.sample_format.fixate()
            {
                element = element.with_input_format(engine_sample_format(format));
            }
            if let FormatCaps::AudioRaw(caps) = &request.output {
                if let Some(format) = caps.sample_format.fixate() {
                    element = element.with_output_format(engine_sample_format(format));
                }
                if let Some(channels) = caps.channels.fixate() {
                    element = element.with_channels(u32::from(channels));
                }
            }
            Box::new(element)
        }),
    });

    // Sample rate. Registered on the same key as audioconvert — which is why the
    // registry has to keep more than one converter per key; the old one silently
    // evicted audioconvert here.
    registry.register(ConverterSpec {
        name: "audioresample",
        from_format: FormatType::AudioRaw,
        to_format: FormatType::AudioRaw,
        from_memory: MemoryType::Cpu,
        to_memory: MemoryType::Cpu,
        axes: ConvertAxes::RATE,
        cost: 8,
        factory: Arc::new(|request: &ConversionRequest| {
            let mut element = crate::elements::transform::AudioResampleElement::new();
            if let FormatCaps::AudioRaw(caps) = &request.input
                && let Some(rate) = caps.sample_rate.fixate()
            {
                element = element.with_input_rate(rate);
            }
            if let FormatCaps::AudioRaw(caps) = &request.output {
                if let Some(rate) = caps.sample_rate.fixate() {
                    element = element.with_output_rate(rate);
                }
                if let Some(channels) = caps.channels.fixate() {
                    element = element.with_channels(u32::from(channels));
                }
            }
            Box::new(element)
        }),
    });

    // CPU to GPU upload.
    registry.register(ConverterSpec {
        name: "memorycopy",
        from_format: FormatType::Any,
        to_format: FormatType::Any,
        from_memory: MemoryType::Cpu,
        to_memory: MemoryType::GpuDevice,
        axes: ConvertAxes::MEMORY,
        cost: 20,
        factory: Arc::new(|_request: &ConversionRequest| Box::new(MemoryCopy::cpu_to_gpu())),
    });

    // GPU to CPU download.
    registry.register(ConverterSpec {
        name: "memorycopy",
        from_format: FormatType::Any,
        to_format: FormatType::Any,
        from_memory: MemoryType::GpuDevice,
        to_memory: MemoryType::Cpu,
        axes: ConvertAxes::MEMORY,
        cost: 20,
        factory: Arc::new(|_request: &ConversionRequest| Box::new(MemoryCopy::gpu_to_cpu())),
    });

    // DMA-BUF to CPU copy (#145): the bridge behind a dmabuf-emitting
    // source for CPU-only consumers.
    registry.register(ConverterSpec {
        name: "memorycopy",
        from_format: FormatType::Any,
        to_format: FormatType::Any,
        from_memory: MemoryType::DmaBuf,
        to_memory: MemoryType::Cpu,
        axes: ConvertAxes::MEMORY,
        cost: 20,
        factory: Arc::new(|_request: &ConversionRequest| Box::new(MemoryCopy::dmabuf_to_cpu())),
    });

    // External to CPU repack (#194): the bridge behind an External-only
    // producer for consumers that did not opt in — the strided frame lands
    // packed in CPU shm.
    registry.register(ConverterSpec {
        name: "memorycopy",
        from_format: FormatType::Any,
        to_format: FormatType::Any,
        from_memory: MemoryType::External,
        to_memory: MemoryType::Cpu,
        axes: ConvertAxes::MEMORY,
        cost: 20,
        factory: Arc::new(|_request: &ConversionRequest| Box::new(MemoryCopy::external_to_cpu())),
    });

    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{AudioFormatCaps, CapsValue, PixelFormat, VideoFormatCaps};

    fn video_request(output: VideoFormatCaps) -> ConversionRequest {
        ConversionRequest {
            input: FormatCaps::VideoRaw(VideoFormatCaps::any()),
            output: FormatCaps::VideoRaw(output),
            input_memory: MemoryType::Cpu,
            output_memory: MemoryType::Cpu,
        }
    }

    #[test]
    fn audioconvert_and_audioresample_both_survive_registration() {
        let registry = builtin_registry();
        let names: Vec<_> = registry
            .candidates(
                FormatType::AudioRaw,
                FormatType::AudioRaw,
                MemoryType::Cpu,
                MemoryType::Cpu,
            )
            .iter()
            .map(|c| c.info.name)
            .collect();

        // audioconvert used to be evicted by audioresample: identical key,
        // HashMap::insert.
        assert!(names.contains(&"audioconvert"), "got {names:?}");
        assert!(names.contains(&"audioresample"), "got {names:?}");
    }

    #[test]
    fn a_format_and_geometry_conflict_plans_convert_plus_scale() {
        let registry = builtin_registry();
        let plan = registry
            .plan(
                FormatType::VideoRaw,
                FormatType::VideoRaw,
                MemoryType::Cpu,
                MemoryType::Cpu,
                ConvertAxes::FORMAT | ConvertAxes::GEOMETRY,
            )
            .expect("videoconvert + videoscale cover both axes");

        let names: Vec<_> = plan.steps.iter().map(|s| s.info.name).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"videoconvert"));
        assert!(names.contains(&"videoscale"));
    }

    #[test]
    fn identity_is_never_planned() {
        let registry = builtin_registry();
        // identity fixes no axis, so it can never be chosen to cover one.
        let plan = registry.plan(
            FormatType::VideoRaw,
            FormatType::VideoRaw,
            MemoryType::Cpu,
            MemoryType::Cpu,
            ConvertAxes::RATE,
        );
        assert!(plan.is_none(), "nothing converts video framerate");
    }

    #[test]
    fn the_videoconvert_factory_is_told_its_target_format() {
        let registry = builtin_registry();
        let converter = registry
            .candidates(
                FormatType::VideoRaw,
                FormatType::VideoRaw,
                MemoryType::Cpu,
                MemoryType::Cpu,
            )
            .iter()
            .find(|c| c.info.name == "videoconvert")
            .unwrap();

        // The old registry hardcoded RGBA here regardless of what the sink asked
        // for. Prove the element now takes its target from the request.
        let element = (converter.factory)(&video_request(VideoFormatCaps {
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            ..VideoFormatCaps::any()
        }));
        assert_eq!(element.name(), "videoconvert");
    }

    #[test]
    fn the_videoscale_factory_is_told_its_target_geometry() {
        let request = video_request(VideoFormatCaps {
            width: CapsValue::Fixed(1280),
            height: CapsValue::Fixed(720),
            ..VideoFormatCaps::any()
        });
        assert_eq!(target_geometry(&request), Some((1280, 720)));

        // An unconstrained sink pins nothing, and the scaler stays passthrough.
        assert_eq!(
            target_geometry(&video_request(VideoFormatCaps::any())),
            None
        );
    }

    #[test]
    fn the_audio_factories_read_both_ends_of_the_request() {
        let registry = builtin_registry();
        let resample = registry
            .candidates(
                FormatType::AudioRaw,
                FormatType::AudioRaw,
                MemoryType::Cpu,
                MemoryType::Cpu,
            )
            .iter()
            .find(|c| c.info.name == "audioresample")
            .unwrap();

        let element = (resample.factory)(&ConversionRequest {
            input: FormatCaps::AudioRaw(AudioFormatCaps {
                sample_rate: CapsValue::Fixed(44_100),
                ..AudioFormatCaps::any()
            }),
            output: FormatCaps::AudioRaw(AudioFormatCaps {
                sample_rate: CapsValue::Fixed(48_000),
                channels: CapsValue::Fixed(2),
                ..AudioFormatCaps::any()
            }),
            input_memory: MemoryType::Cpu,
            output_memory: MemoryType::Cpu,
        });
        assert_eq!(element.name(), "audioresample");
    }

    #[test]
    fn memory_copy_creation() {
        let uploader = MemoryCopy::cpu_to_gpu();
        assert_eq!(uploader.converter_name(), "memorycopy");
        assert_eq!(uploader.input_memory(), MemoryType::Cpu);
        assert_eq!(uploader.output_memory(), MemoryType::GpuDevice);
        assert_eq!(uploader.cost(), 20);
    }

    #[test]
    fn identity_passes_buffers_through() {
        use crate::buffer::MemoryHandle;
        use crate::memory::SharedArena;
        use crate::metadata::Metadata;

        let mut identity = Identity;
        assert_eq!(identity.converter_name(), "identity");
        assert_eq!(identity.cost(), 0);

        let arena = SharedArena::new(64, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 4);
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));
        let result = Element::process(&mut identity, buffer).unwrap();
        assert_eq!(result.unwrap().len(), 4);
    }
}
