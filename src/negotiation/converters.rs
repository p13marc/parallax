//! Converter registry for format, geometry, rate and memory conversions.
//!
//! # Axes
//!
//! Two elements can disagree along several independent *axes*: the pixel or
//! sample format, the geometry (width/height), the rate (framerate or sample
//! rate), and the memory type. A single converter usually fixes exactly one of
//! them — `videoconvert` changes the pixel format but carries the input
//! dimensions through; `videoscale` changes the dimensions but keeps the pixel
//! format. Bridging a link therefore sometimes needs a *chain*.
//!
//! The registry records which axes each converter can fix ([`ConvertAxes`]),
//! keeps every converter registered for a key (not just the last one), and
//! plans a cheapest chain that covers **all** the conflicting axes — or refuses
//! ([`ConverterRegistry::plan`] returns `None`). A partial chain is never
//! emitted: it would leave a running pipeline quietly producing wrong frames.

use crate::element::Element;
use crate::format::{FormatCaps, FormatMemoryCap, MediaFormat};
use crate::memory::MemoryType;
use std::collections::HashMap;
use std::sync::Arc;

/// What a converter is being asked to do.
///
/// Passed to a [`ConverterFactory`] so the converter can be *told its target*
/// instead of guessing one. Without this a registered `videoconvert` would have
/// to hardcode an output pixel format, and a pipeline wanting a different one
/// would silently get the wrong converter.
#[derive(Clone, Debug)]
pub struct ConversionRequest {
    /// Caps the upstream element produces.
    pub input: FormatCaps,
    /// Caps the downstream element requires.
    pub output: FormatCaps,
    /// Memory type on the upstream side.
    pub input_memory: MemoryType,
    /// Memory type on the downstream side.
    pub output_memory: MemoryType,
}

/// Factory function for creating converter elements.
///
/// Receives the [`ConversionRequest`] describing the link it is bridging.
pub type ConverterFactory =
    Arc<dyn Fn(&ConversionRequest) -> Box<dyn Element + Send> + Send + Sync>;

/// Trait for converter elements.
///
/// Converters transform buffers between formats or memory types.
/// This trait extends [`Element`] to provide both processing capability
/// and format/memory metadata for negotiation.
pub trait ConverterElement: Element + Send {
    /// Get the name of this converter.
    fn converter_name(&self) -> &str;

    /// Get input format this converter accepts.
    fn input_format(&self) -> FormatCaps;

    /// Get output format this converter produces.
    fn output_format(&self) -> FormatCaps;

    /// Get input memory type.
    fn input_memory(&self) -> MemoryType;

    /// Get output memory type.
    fn output_memory(&self) -> MemoryType;

    /// Get the cost of this conversion (lower is better).
    fn cost(&self) -> u32;
}

// ============================================================================
// ConvertAxes
// ============================================================================

/// The axes along which two caps can disagree, and which a converter can fix.
///
/// A hand-rolled bitflag set (the crate does not depend on `bitflags`;
/// [`MemoryLayout`](crate::format::MemoryLayout) is the same idiom).
///
/// # Example
///
/// ```rust
/// use parallax::negotiation::ConvertAxes;
///
/// let needed = ConvertAxes::FORMAT | ConvertAxes::GEOMETRY;
/// assert!(needed.contains(ConvertAxes::FORMAT));
/// assert!(!ConvertAxes::FORMAT.contains(needed));
/// assert_eq!(needed.without(ConvertAxes::FORMAT), ConvertAxes::GEOMETRY);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConvertAxes(u8);

impl ConvertAxes {
    /// Fixes nothing (an identity element).
    pub const NONE: Self = Self(0);
    /// Pixel format, or sample format and channel count.
    pub const FORMAT: Self = Self(1 << 0);
    /// Width and height.
    pub const GEOMETRY: Self = Self(1 << 1);
    /// Framerate, or sample rate.
    pub const RATE: Self = Self(1 << 2);
    /// Memory type (CPU, GPU, DMA-BUF...).
    pub const MEMORY: Self = Self(1 << 3);

    /// True when no axis is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// True when every axis in `other` is also in `self`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// True when `self` and `other` share at least one axis.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The axes in both sets.
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// The axes of `self` that are not in `other`.
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// How many axes are set.
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

impl std::ops::BitOr for ConvertAxes {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ConvertAxes {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Why two caps could not be intersected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapsConflict {
    /// Same media kind, but these axes fail to intersect. A converter chain
    /// covering exactly these axes would bridge the link.
    Axes(ConvertAxes),
    /// Fundamentally different media kinds (raw video vs. encoded audio, say).
    /// No converter in the registry can bridge this.
    Incompatible,
}

/// Report which axes prevent `source` and `sink` caps from intersecting.
///
/// Returns `None` when they already intersect — no converter is needed.
/// `Any` never conflicts, and `Fixed(1920)` against `Range { 1280..=1920 }`
/// does not conflict either (it fixates to 1920).
pub fn diff_caps(source: &FormatMemoryCap, sink: &FormatMemoryCap) -> Option<CapsConflict> {
    let mut axes = ConvertAxes::NONE;

    if source.memory.intersect(&sink.memory).is_none() {
        axes |= ConvertAxes::MEMORY;
    } else if let Some(fixated) = source.memory.fixate()
        && fixated.requires_explicit_optin()
        && !sink.memory.lists_memory(fixated)
        && !source.memory.lists_memory(MemoryType::Cpu)
    {
        // Opt-in rule (#194): the caps *intersect* (Fixed ∩ Any), but an
        // opt-in memory type (External) may not be delivered to a sink
        // that didn't name it, and this source cannot fall back to Cpu on
        // its own — a converter (memorycopy repack) must bridge.
        axes |= ConvertAxes::MEMORY;
    }

    match (&source.format, &sink.format) {
        (FormatCaps::Any, _) | (_, FormatCaps::Any) => {}
        (FormatCaps::Bytes, _) | (_, FormatCaps::Bytes) => {}
        (FormatCaps::VideoRaw(a), FormatCaps::VideoRaw(b)) => {
            if a.pixel_format.intersect(&b.pixel_format).is_none() {
                axes |= ConvertAxes::FORMAT;
            }
            if a.width.intersect(&b.width).is_none() || a.height.intersect(&b.height).is_none() {
                axes |= ConvertAxes::GEOMETRY;
            }
            if a.framerate.intersect(&b.framerate).is_none() {
                axes |= ConvertAxes::RATE;
            }
        }
        (FormatCaps::AudioRaw(a), FormatCaps::AudioRaw(b)) => {
            if a.sample_format.intersect(&b.sample_format).is_none()
                || a.channels.intersect(&b.channels).is_none()
            {
                axes |= ConvertAxes::FORMAT;
            }
            if a.sample_rate.intersect(&b.sample_rate).is_none() {
                axes |= ConvertAxes::RATE;
            }
        }
        (a, b) if a.intersect(b).is_some() => {}
        _ => return Some(CapsConflict::Incompatible),
    }

    if axes.is_empty() {
        None
    } else {
        Some(CapsConflict::Axes(axes))
    }
}

// ============================================================================
// Registry
// ============================================================================

/// Metadata about a converter for use in negotiation.
///
/// This is stored in the registry alongside the factory, so we can
/// query converter capabilities without creating instances.
#[derive(Clone, Debug)]
pub struct ConverterInfo {
    /// Name of the converter.
    pub name: &'static str,
    /// Input format type.
    pub from_format: FormatType,
    /// Output format type.
    pub to_format: FormatType,
    /// Input memory type.
    pub from_memory: MemoryType,
    /// Output memory type.
    pub to_memory: MemoryType,
    /// Which axes this converter can fix.
    pub axes: ConvertAxes,
    /// Cost of conversion (lower is better).
    pub cost: u32,
}

/// A converter as registered: its metadata plus the factory that builds it.
#[derive(Clone)]
pub struct RegisteredConverter {
    /// Converter metadata.
    pub info: ConverterInfo,
    /// Factory to instantiate it for a specific request.
    pub factory: ConverterFactory,
}

impl std::fmt::Debug for RegisteredConverter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredConverter")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

/// Everything needed to register a converter.
///
/// A struct rather than positional arguments: the positional form would reach
/// eight parameters.
pub struct ConverterSpec {
    /// Name of the converter (used in node names and negotiation errors).
    pub name: &'static str,
    /// Input format type.
    pub from_format: FormatType,
    /// Output format type.
    pub to_format: FormatType,
    /// Input memory type.
    pub from_memory: MemoryType,
    /// Output memory type.
    pub to_memory: MemoryType,
    /// Which axes this converter fixes.
    pub axes: ConvertAxes,
    /// Cost (lower is preferred when several converters cover the same axes).
    pub cost: u32,
    /// Factory to create the element.
    pub factory: ConverterFactory,
}

/// One element of a conversion chain.
#[derive(Clone)]
pub struct ConversionStep {
    /// Metadata of the converter to insert.
    pub info: ConverterInfo,
    /// Factory to instantiate it.
    pub factory: ConverterFactory,
}

impl std::fmt::Debug for ConversionStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversionStep")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

/// An ordered chain of converters that covers every conflicting axis.
#[derive(Clone, Debug)]
pub struct ConversionPlan {
    /// Converters in the order they should be spliced into the link.
    pub steps: Vec<ConversionStep>,
    /// Sum of the steps' costs.
    pub total_cost: u32,
}

/// Key for looking up converters.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConversionKey {
    /// Source format type (simplified for lookup).
    pub from_format: FormatType,
    /// Target format type.
    pub to_format: FormatType,
    /// Source memory type.
    pub from_memory: MemoryType,
    /// Target memory type.
    pub to_memory: MemoryType,
}

/// Simplified format type for converter lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FormatType {
    /// Raw video.
    VideoRaw,
    /// Encoded video.
    VideoEncoded,
    /// Raw audio.
    AudioRaw,
    /// Encoded audio.
    AudioEncoded,
    /// RTP.
    Rtp,
    /// MPEG-TS.
    MpegTs,
    /// Raw bytes.
    Bytes,
    /// Any format.
    Any,
}

impl From<&FormatCaps> for FormatType {
    fn from(caps: &FormatCaps) -> Self {
        match caps {
            FormatCaps::VideoRaw(_) => Self::VideoRaw,
            FormatCaps::Video(_) => Self::VideoEncoded,
            FormatCaps::AudioRaw(_) => Self::AudioRaw,
            FormatCaps::Audio(_) => Self::AudioEncoded,
            FormatCaps::Rtp(_) => Self::Rtp,
            FormatCaps::MpegTs => Self::MpegTs,
            FormatCaps::Bytes => Self::Bytes,
            FormatCaps::Any => Self::Any,
        }
    }
}

impl From<&MediaFormat> for FormatType {
    fn from(format: &MediaFormat) -> Self {
        match format {
            MediaFormat::VideoRaw(_) => Self::VideoRaw,
            MediaFormat::Video(_) => Self::VideoEncoded,
            MediaFormat::AudioRaw(_) => Self::AudioRaw,
            MediaFormat::Audio(_) => Self::AudioEncoded,
            MediaFormat::Rtp(_) => Self::Rtp,
            MediaFormat::MpegTs => Self::MpegTs,
            MediaFormat::Bytes => Self::Bytes,
        }
    }
}

/// Registry for format, geometry, rate and memory converters.
///
/// Several converters may share a key — `videoconvert` and `videoscale` are
/// both `VideoRaw → VideoRaw` on CPU — and are distinguished by the axes they
/// fix. [`plan`](Self::plan) picks the cheapest set covering the conflict.
#[derive(Default)]
pub struct ConverterRegistry {
    /// Direct converters: key -> converters, cheapest first.
    converters: HashMap<ConversionKey, Vec<RegisteredConverter>>,
}

impl ConverterRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a converter.
    ///
    /// Converters already registered under the same key are kept — the list
    /// stays sorted cheapest-first.
    pub fn register(&mut self, spec: ConverterSpec) {
        let key = ConversionKey {
            from_format: spec.from_format,
            to_format: spec.to_format,
            from_memory: spec.from_memory,
            to_memory: spec.to_memory,
        };
        let entry = RegisteredConverter {
            info: ConverterInfo {
                name: spec.name,
                from_format: spec.from_format,
                to_format: spec.to_format,
                from_memory: spec.from_memory,
                to_memory: spec.to_memory,
                axes: spec.axes,
                cost: spec.cost,
            },
            factory: spec.factory,
        };
        let bucket = self.converters.entry(key).or_default();
        bucket.push(entry);
        bucket.sort_by_key(|c| c.info.cost);
    }

    /// All converters registered under a key, cheapest first.
    pub fn candidates(
        &self,
        from_format: FormatType,
        to_format: FormatType,
        from_memory: MemoryType,
        to_memory: MemoryType,
    ) -> &[RegisteredConverter] {
        self.converters
            .get(&ConversionKey {
                from_format,
                to_format,
                from_memory,
                to_memory,
            })
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Find the cheapest converter registered for exactly this key.
    pub fn find_direct(
        &self,
        from_format: FormatType,
        to_format: FormatType,
        from_memory: MemoryType,
        to_memory: MemoryType,
    ) -> Option<&RegisteredConverter> {
        self.candidates(from_format, to_format, from_memory, to_memory)
            .first()
    }

    /// Plan a chain of converters covering every axis in `needed`.
    ///
    /// Greedy cheapest-cover: repeatedly take the converter covering the most
    /// remaining axes, breaking ties by cost. Returns `None` if the registry
    /// cannot cover **all** of `needed` — a partial chain is never returned,
    /// because it would leave the link still broken at runtime.
    ///
    /// A memory conversion combined with a format/geometry/rate conversion is
    /// not planned: the registry has no way to express which side of the
    /// transfer a CPU converter would run on. Such a link is a negotiation
    /// error.
    pub fn plan(
        &self,
        from_format: FormatType,
        to_format: FormatType,
        from_memory: MemoryType,
        to_memory: MemoryType,
        needed: ConvertAxes,
    ) -> Option<ConversionPlan> {
        if needed.is_empty() {
            return Some(ConversionPlan {
                steps: Vec::new(),
                total_cost: 0,
            });
        }

        if needed.contains(ConvertAxes::MEMORY) {
            if needed != ConvertAxes::MEMORY {
                return None; // mixed memory + data conversion: not modelled
            }
            let converter = self
                .candidates(FormatType::Any, FormatType::Any, from_memory, to_memory)
                .iter()
                .find(|c| c.info.axes.contains(ConvertAxes::MEMORY))?;
            return Some(ConversionPlan {
                steps: vec![ConversionStep {
                    info: converter.info.clone(),
                    factory: converter.factory.clone(),
                }],
                total_cost: converter.info.cost,
            });
        }

        // Data conversion: every step runs in the (single) memory domain.
        let candidates = self.candidates(from_format, to_format, from_memory, from_memory);

        let mut remaining = needed;
        let mut steps: Vec<ConversionStep> = Vec::new();
        let mut total_cost = 0u32;

        while !remaining.is_empty() {
            let best = candidates
                .iter()
                .filter(|c| c.info.axes.intersects(remaining))
                .max_by_key(|c| {
                    (
                        c.info.axes.intersection(remaining).count(),
                        std::cmp::Reverse(c.info.cost),
                    )
                })?;

            remaining = remaining.without(best.info.axes);
            total_cost += best.info.cost;
            steps.push(ConversionStep {
                info: best.info.clone(),
                factory: best.factory.clone(),
            });
        }

        Some(ConversionPlan { steps, total_cost })
    }

    /// Check whether a conversion covering `needed` is possible.
    pub fn can_convert(
        &self,
        from_format: FormatType,
        to_format: FormatType,
        from_memory: MemoryType,
        to_memory: MemoryType,
        needed: ConvertAxes,
    ) -> bool {
        self.plan(from_format, to_format, from_memory, to_memory, needed)
            .is_some()
    }

    /// Get the number of registered converters.
    pub fn len(&self) -> usize {
        self.converters.values().map(Vec::len).sum()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.converters.is_empty()
    }
}

impl std::fmt::Debug for ConverterRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConverterRegistry")
            .field("num_converters", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::error::Result;
    use crate::format::{CapsValue, MemoryCaps, PixelFormat, VideoFormatCaps};

    struct DummyConverter {
        name: String,
    }

    impl Element for DummyConverter {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            Ok(Some(buffer))
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    fn dummy(name: &'static str) -> ConverterFactory {
        Arc::new(move |_req: &ConversionRequest| {
            Box::new(DummyConverter {
                name: name.to_string(),
            }) as Box<dyn Element + Send>
        })
    }

    fn spec(name: &'static str, axes: ConvertAxes, cost: u32) -> ConverterSpec {
        ConverterSpec {
            name,
            from_format: FormatType::VideoRaw,
            to_format: FormatType::VideoRaw,
            from_memory: MemoryType::Cpu,
            to_memory: MemoryType::Cpu,
            axes,
            cost,
            factory: dummy(name),
        }
    }

    fn video(width: u32, height: u32, pf: PixelFormat) -> FormatMemoryCap {
        FormatMemoryCap::new(
            VideoFormatCaps {
                width: CapsValue::Fixed(width),
                height: CapsValue::Fixed(height),
                pixel_format: CapsValue::Fixed(pf),
                ..VideoFormatCaps::any()
            }
            .into(),
            MemoryCaps::cpu_only(),
        )
    }

    #[test]
    fn converters_sharing_a_key_all_survive_registration() {
        let mut registry = ConverterRegistry::new();
        registry.register(spec("videoconvert", ConvertAxes::FORMAT, 5));
        registry.register(spec("videoscale", ConvertAxes::GEOMETRY, 10));

        // The old registry was a HashMap<key, one converter>, so the second
        // register() silently evicted the first.
        assert_eq!(registry.len(), 2);
        assert_eq!(
            registry
                .candidates(
                    FormatType::VideoRaw,
                    FormatType::VideoRaw,
                    MemoryType::Cpu,
                    MemoryType::Cpu
                )
                .len(),
            2
        );
        assert_eq!(
            registry
                .find_direct(
                    FormatType::VideoRaw,
                    FormatType::VideoRaw,
                    MemoryType::Cpu,
                    MemoryType::Cpu
                )
                .unwrap()
                .info
                .name,
            "videoconvert",
            "find_direct returns the cheapest"
        );
    }

    #[test]
    fn a_two_axis_conflict_plans_a_two_element_chain() {
        let mut registry = ConverterRegistry::new();
        registry.register(spec("videoconvert", ConvertAxes::FORMAT, 5));
        registry.register(spec("videoscale", ConvertAxes::GEOMETRY, 10));

        let plan = registry
            .plan(
                FormatType::VideoRaw,
                FormatType::VideoRaw,
                MemoryType::Cpu,
                MemoryType::Cpu,
                ConvertAxes::FORMAT | ConvertAxes::GEOMETRY,
            )
            .expect("both axes are covered");

        let names: Vec<_> = plan.steps.iter().map(|s| s.info.name).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"videoconvert"));
        assert!(names.contains(&"videoscale"));
        assert_eq!(plan.total_cost, 15);
    }

    #[test]
    fn an_uncoverable_axis_yields_no_plan_rather_than_a_partial_chain() {
        let mut registry = ConverterRegistry::new();
        registry.register(spec("videoconvert", ConvertAxes::FORMAT, 5));

        // Nothing fixes geometry: refuse the plan rather than insert a converter
        // that leaves the link still broken.
        assert!(
            registry
                .plan(
                    FormatType::VideoRaw,
                    FormatType::VideoRaw,
                    MemoryType::Cpu,
                    MemoryType::Cpu,
                    ConvertAxes::FORMAT | ConvertAxes::GEOMETRY,
                )
                .is_none()
        );
    }

    #[test]
    fn diff_reports_exactly_the_conflicting_axes() {
        let src = video(1920, 1080, PixelFormat::Yuyv);

        let sink = video(1280, 720, PixelFormat::I420);
        assert_eq!(
            diff_caps(&src, &sink),
            Some(CapsConflict::Axes(
                ConvertAxes::FORMAT | ConvertAxes::GEOMETRY
            ))
        );

        let sink = video(1920, 1080, PixelFormat::I420);
        assert_eq!(
            diff_caps(&src, &sink),
            Some(CapsConflict::Axes(ConvertAxes::FORMAT))
        );

        assert_eq!(diff_caps(&src, &src), None);
    }

    #[test]
    fn any_and_ranges_never_conflict() {
        let src = video(1920, 1080, PixelFormat::I420);
        let sink = FormatMemoryCap::new(
            VideoFormatCaps {
                width: CapsValue::Range {
                    min: 1280,
                    max: 1920,
                },
                height: CapsValue::Range {
                    min: 720,
                    max: 1080,
                },
                pixel_format: CapsValue::Any,
                ..VideoFormatCaps::any()
            }
            .into(),
            MemoryCaps::cpu_only(),
        );
        assert_eq!(diff_caps(&src, &sink), None);
    }

    #[test]
    fn different_media_kinds_are_incompatible_not_convertible() {
        let video_cap = video(640, 480, PixelFormat::I420);
        let audio_cap = FormatMemoryCap::new(
            FormatCaps::AudioRaw(crate::format::AudioFormatCaps::any()),
            MemoryCaps::cpu_only(),
        );
        assert_eq!(
            diff_caps(&video_cap, &audio_cap),
            Some(CapsConflict::Incompatible)
        );
    }

    #[test]
    fn memory_conflicts_plan_a_memory_converter() {
        let mut registry = ConverterRegistry::new();
        registry.register(ConverterSpec {
            name: "memorycopy",
            from_format: FormatType::Any,
            to_format: FormatType::Any,
            from_memory: MemoryType::Cpu,
            to_memory: MemoryType::GpuDevice,
            axes: ConvertAxes::MEMORY,
            cost: 20,
            factory: dummy("memorycopy"),
        });

        let plan = registry
            .plan(
                FormatType::VideoRaw,
                FormatType::VideoRaw,
                MemoryType::Cpu,
                MemoryType::GpuDevice,
                ConvertAxes::MEMORY,
            )
            .unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].info.name, "memorycopy");

        // A memory transfer combined with a data conversion is not modelled.
        assert!(
            registry
                .plan(
                    FormatType::VideoRaw,
                    FormatType::VideoRaw,
                    MemoryType::Cpu,
                    MemoryType::GpuDevice,
                    ConvertAxes::MEMORY | ConvertAxes::FORMAT,
                )
                .is_none()
        );
    }

    #[test]
    fn format_type_from_caps() {
        use crate::format::AudioFormatCaps;

        let video = FormatCaps::VideoRaw(VideoFormatCaps::any());
        let audio = FormatCaps::AudioRaw(AudioFormatCaps::any());
        let bytes = FormatCaps::Bytes;

        assert_eq!(FormatType::from(&video), FormatType::VideoRaw);
        assert_eq!(FormatType::from(&audio), FormatType::AudioRaw);
        assert_eq!(FormatType::from(&bytes), FormatType::Bytes);
    }
}
