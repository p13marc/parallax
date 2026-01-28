//! Negotiation solver for pipeline caps.

use super::converters::{ConverterFactory, ConverterRegistry, FormatType};
use super::error::NegotiationError;
use crate::format::{ElementMediaCaps, MediaCaps, MediaFormat};
use crate::memory::MemoryType;
use std::collections::HashMap;

/// Result of caps negotiation.
#[derive(Debug, Default)]
pub struct NegotiationResult {
    /// Negotiated caps for each link (link_id -> negotiated caps).
    pub link_caps: HashMap<usize, LinkNegotiation>,
    /// Converters to insert.
    pub converters: Vec<ConverterInsertion>,
}

/// Negotiated caps for a single link.
#[derive(Debug, Clone)]
pub struct LinkNegotiation {
    /// Link identifier.
    pub link_id: usize,
    /// Negotiated media format.
    pub format: MediaFormat,
    /// Negotiated memory type.
    pub memory_type: MemoryType,
}

/// A converter to insert into the pipeline.
pub struct ConverterInsertion {
    /// Link where converter should be inserted.
    pub link_id: usize,
    /// Factory to create the converter element.
    pub factory: ConverterFactory,
    /// Reason for insertion.
    pub reason: String,
    /// Cost of this conversion.
    pub cost: u32,
}

impl std::fmt::Debug for ConverterInsertion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConverterInsertion")
            .field("link_id", &self.link_id)
            .field("reason", &self.reason)
            .field("cost", &self.cost)
            .finish_non_exhaustive()
    }
}

/// Element caps for negotiation.
///
/// Stores multiple format+memory combinations per pad, allowing the solver
/// to find the best matching format across all possibilities.
#[derive(Debug, Clone)]
pub struct ElementCaps {
    /// Element name.
    pub name: String,
    /// Caps for each sink (input) pad, keyed by pad name.
    pub sink_caps: HashMap<String, ElementMediaCaps>,
    /// Caps for each source (output) pad, keyed by pad name.
    pub source_caps: HashMap<String, ElementMediaCaps>,
}

impl ElementCaps {
    /// Create new element caps with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sink_caps: HashMap::new(),
            source_caps: HashMap::new(),
        }
    }

    /// Add caps for a sink (input) pad.
    ///
    /// For backward compatibility, accepts `MediaCaps` which is converted to
    /// `ElementMediaCaps` with a single entry.
    pub fn add_sink_pad(&mut self, pad_name: impl Into<String>, caps: MediaCaps) {
        self.sink_caps
            .insert(pad_name.into(), ElementMediaCaps::from(caps));
    }

    /// Add caps for a source (output) pad.
    ///
    /// For backward compatibility, accepts `MediaCaps` which is converted to
    /// `ElementMediaCaps` with a single entry.
    pub fn add_source_pad(&mut self, pad_name: impl Into<String>, caps: MediaCaps) {
        self.source_caps
            .insert(pad_name.into(), ElementMediaCaps::from(caps));
    }

    /// Add multi-format caps for a sink (input) pad.
    ///
    /// This allows declaring multiple supported format+memory combinations,
    /// ordered by preference.
    pub fn add_sink_pad_multi(&mut self, pad_name: impl Into<String>, caps: ElementMediaCaps) {
        self.sink_caps.insert(pad_name.into(), caps);
    }

    /// Add multi-format caps for a source (output) pad.
    ///
    /// This allows declaring multiple supported format+memory combinations,
    /// ordered by preference.
    pub fn add_source_pad_multi(&mut self, pad_name: impl Into<String>, caps: ElementMediaCaps) {
        self.source_caps.insert(pad_name.into(), caps);
    }
}

/// Link in the pipeline graph.
#[derive(Debug, Clone)]
pub struct LinkInfo {
    /// Link identifier.
    pub id: usize,
    /// Source element name.
    pub source_element: String,
    /// Source pad name.
    pub source_pad: String,
    /// Sink element name.
    pub sink_element: String,
    /// Sink pad name.
    pub sink_pad: String,
}

/// Solver for caps negotiation.
///
/// Takes element caps and link topology, finds compatible formats,
/// and optionally inserts converters.
pub struct NegotiationSolver {
    /// Element caps by name.
    elements: HashMap<String, ElementCaps>,
    /// Links in the pipeline.
    links: Vec<LinkInfo>,
    /// Converter registry for automatic conversion.
    converter_registry: Option<ConverterRegistry>,
}

impl NegotiationSolver {
    /// Create a new solver.
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            links: Vec::new(),
            converter_registry: None,
        }
    }

    /// Set the converter registry for automatic conversion insertion.
    pub fn with_converters(mut self, registry: ConverterRegistry) -> Self {
        self.converter_registry = Some(registry);
        self
    }

    /// Add an element with its caps.
    pub fn add_element(&mut self, caps: ElementCaps) {
        self.elements.insert(caps.name.clone(), caps);
    }

    /// Add a link between elements.
    pub fn add_link(&mut self, link: LinkInfo) {
        self.links.push(link);
    }

    /// Solve caps negotiation for all links.
    pub fn solve(&self) -> Result<NegotiationResult, NegotiationError> {
        let mut result = NegotiationResult::default();

        for link in &self.links {
            let link_result = self.negotiate_link(link)?;

            match link_result {
                LinkResult::Direct(negotiation) => {
                    result.link_caps.insert(link.id, negotiation);
                }
                LinkResult::NeedsConverter {
                    negotiation,
                    converter,
                } => {
                    result.link_caps.insert(link.id, negotiation);
                    result.converters.push(converter);
                }
            }
        }

        Ok(result)
    }

    /// Negotiate a single link.
    ///
    /// This method iterates over all format+memory combinations from both
    /// source and sink, finding the first (highest preference) match.
    fn negotiate_link(&self, link: &LinkInfo) -> Result<LinkResult, NegotiationError> {
        // Get source and sink caps (multi-format)
        let source_caps = self.get_source_caps(link)?;
        let sink_caps = self.get_sink_caps(link)?;

        // Try direct intersection across all format+memory combinations
        // ElementMediaCaps::intersect already handles finding the best match
        if let Some(intersected) = source_caps.intersect(&sink_caps) {
            // Can negotiate directly - fixate the intersected format+memory
            // Use fixate_with_defaults() to handle Any/Any cases gracefully
            let format = intersected.format.fixate_with_defaults();
            let memory_type = intersected.memory.fixate().unwrap_or(MemoryType::Cpu);

            return Ok(LinkResult::Direct(LinkNegotiation {
                link_id: link.id,
                format,
                memory_type,
            }));
        }

        // No direct match - try to find a converter
        // For converter lookup, we need to consider all source/sink format combinations
        if let Some(registry) = &self.converter_registry {
            // Try each source format against each sink format to find a converter path
            for source_cap in source_caps.iter() {
                let source_format_type = FormatType::from(&source_cap.format);
                let source_memory = source_cap.memory.fixate().unwrap_or(MemoryType::Cpu);

                for sink_cap in sink_caps.iter() {
                    let sink_format_type = FormatType::from(&sink_cap.format);
                    let sink_memory = sink_cap.memory.fixate().unwrap_or(MemoryType::Cpu);

                    if let Some((factories_with_info, total_cost)) = registry.find_path(
                        source_format_type,
                        sink_format_type,
                        source_memory,
                        sink_memory,
                    ) {
                        // Use the first factory (for now, we only support single-hop)
                        if let Some((factory, info)) = factories_with_info.into_iter().next() {
                            // Fixate source format for the link
                            let format = source_cap.format.fixate().ok_or_else(|| {
                                NegotiationError::CannotFixate {
                                    link_id: link.id,
                                    reason: "Could not fixate source format".into(),
                                }
                            })?;

                            return Ok(LinkResult::NeedsConverter {
                                negotiation: LinkNegotiation {
                                    link_id: link.id,
                                    format,
                                    memory_type: source_memory,
                                },
                                converter: ConverterInsertion {
                                    link_id: link.id,
                                    factory,
                                    reason: format!(
                                        "{}: {:?}/{:?} -> {:?}/{:?}",
                                        info.name,
                                        source_format_type,
                                        source_memory,
                                        sink_format_type,
                                        sink_memory
                                    ),
                                    cost: total_cost,
                                },
                            });
                        }
                    }
                }
            }
        }

        // No direct match and no converter available
        // Collect all formats for error message
        let source_formats: Vec<_> = source_caps
            .iter()
            .map(|c| format!("{:?}", c.format))
            .collect();
        let sink_formats: Vec<_> = sink_caps
            .iter()
            .map(|c| format!("{:?}", c.format))
            .collect();

        Err(NegotiationError::no_common_format(
            &link.source_element,
            &link.sink_element,
            &source_formats.join(" | "),
            &sink_formats.join(" | "),
        ))
    }

    /// Get source (output) caps for a link.
    fn get_source_caps(&self, link: &LinkInfo) -> Result<ElementMediaCaps, NegotiationError> {
        let element = self.elements.get(&link.source_element).ok_or_else(|| {
            NegotiationError::ElementNotFound {
                name: link.source_element.clone(),
            }
        })?;

        element
            .source_caps
            .get(&link.source_pad)
            .cloned()
            .ok_or_else(|| {
                NegotiationError::Internal(format!(
                    "Source pad '{}' not found on element '{}'",
                    link.source_pad, link.source_element
                ))
            })
    }

    /// Get sink (input) caps for a link.
    fn get_sink_caps(&self, link: &LinkInfo) -> Result<ElementMediaCaps, NegotiationError> {
        let element = self.elements.get(&link.sink_element).ok_or_else(|| {
            NegotiationError::ElementNotFound {
                name: link.sink_element.clone(),
            }
        })?;

        element
            .sink_caps
            .get(&link.sink_pad)
            .cloned()
            .ok_or_else(|| {
                NegotiationError::Internal(format!(
                    "Sink pad '{}' not found on element '{}'",
                    link.sink_pad, link.sink_element
                ))
            })
    }
}

impl Default for NegotiationSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of negotiating a single link.
enum LinkResult {
    /// Direct negotiation succeeded.
    Direct(LinkNegotiation),
    /// Needs a converter to be inserted.
    NeedsConverter {
        negotiation: LinkNegotiation,
        converter: ConverterInsertion,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        FormatCaps, Framerate, MemoryCaps, PixelFormat, VideoCodec, VideoFormat, VideoFormatCaps,
    };

    fn video_1080p() -> MediaFormat {
        MediaFormat::VideoRaw(VideoFormat::new(
            1920,
            1080,
            PixelFormat::I420,
            Framerate::FPS_30,
        ))
    }

    #[test]
    fn test_solver_direct_negotiation() {
        let mut solver = NegotiationSolver::new();

        // Source produces 1080p video
        let mut source_caps = ElementCaps::new("source");
        source_caps.add_source_pad("src", MediaCaps::from(video_1080p()));
        solver.add_element(source_caps);

        // Sink accepts any video
        let mut sink_caps = ElementCaps::new("sink");
        sink_caps.add_sink_pad(
            "sink",
            MediaCaps::from_format(FormatCaps::VideoRaw(VideoFormatCaps::any())),
        );
        solver.add_element(sink_caps);

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "source".into(),
            source_pad: "src".into(),
            sink_element: "sink".into(),
            sink_pad: "sink".into(),
        });

        let result = solver.solve().unwrap();

        assert_eq!(result.link_caps.len(), 1);
        assert!(result.converters.is_empty());

        let link_caps = result.link_caps.get(&0).unwrap();
        assert!(matches!(link_caps.format, MediaFormat::VideoRaw(_)));
    }

    #[test]
    fn test_solver_incompatible_formats() {
        let mut solver = NegotiationSolver::new();

        // Source produces raw video
        let mut source_caps = ElementCaps::new("source");
        source_caps.add_source_pad(
            "src",
            MediaCaps::from_format(FormatCaps::VideoRaw(VideoFormatCaps::any())),
        );
        solver.add_element(source_caps);

        // Sink only accepts H.264
        let mut sink_caps = ElementCaps::new("sink");
        sink_caps.add_sink_pad(
            "sink",
            MediaCaps::from_format(FormatCaps::Video(VideoCodec::H264)),
        );
        solver.add_element(sink_caps);

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "source".into(),
            source_pad: "src".into(),
            sink_element: "sink".into(),
            sink_pad: "sink".into(),
        });

        let result = solver.solve();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            NegotiationError::NoCommonFormat { .. }
        ));
    }

    #[test]
    fn test_solver_memory_negotiation() {
        let mut solver = NegotiationSolver::new();

        // Source produces CPU memory
        let mut source_caps = ElementCaps::new("source");
        source_caps.add_source_pad(
            "src",
            MediaCaps::new(
                FormatCaps::VideoRaw(VideoFormatCaps::any()),
                MemoryCaps::cpu_only(),
            ),
        );
        solver.add_element(source_caps);

        // Sink accepts CPU memory
        let mut sink_caps = ElementCaps::new("sink");
        sink_caps.add_sink_pad(
            "sink",
            MediaCaps::new(
                FormatCaps::VideoRaw(VideoFormatCaps::any()),
                MemoryCaps::cpu_only(),
            ),
        );
        solver.add_element(sink_caps);

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "source".into(),
            source_pad: "src".into(),
            sink_element: "sink".into(),
            sink_pad: "sink".into(),
        });

        let result = solver.solve().unwrap();
        let link_caps = result.link_caps.get(&0).unwrap();
        assert_eq!(link_caps.memory_type, MemoryType::Cpu);
    }

    #[test]
    fn test_solver_multi_link_pipeline() {
        let mut solver = NegotiationSolver::new();

        let format = video_1080p();

        // source -> filter -> sink
        let mut source_caps = ElementCaps::new("source");
        source_caps.add_source_pad("src", MediaCaps::from(format.clone()));
        solver.add_element(source_caps);

        let mut filter_caps = ElementCaps::new("filter");
        filter_caps.add_sink_pad(
            "sink",
            MediaCaps::from_format(FormatCaps::VideoRaw(VideoFormatCaps::any())),
        );
        filter_caps.add_source_pad(
            "src",
            MediaCaps::from_format(FormatCaps::VideoRaw(VideoFormatCaps::any())),
        );
        solver.add_element(filter_caps);

        let mut sink_caps = ElementCaps::new("sink");
        sink_caps.add_sink_pad(
            "sink",
            MediaCaps::from_format(FormatCaps::VideoRaw(VideoFormatCaps::any())),
        );
        solver.add_element(sink_caps);

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "source".into(),
            source_pad: "src".into(),
            sink_element: "filter".into(),
            sink_pad: "sink".into(),
        });

        solver.add_link(LinkInfo {
            id: 1,
            source_element: "filter".into(),
            source_pad: "src".into(),
            sink_element: "sink".into(),
            sink_pad: "sink".into(),
        });

        let result = solver.solve().unwrap();
        assert_eq!(result.link_caps.len(), 2);
        assert!(result.converters.is_empty());
    }

    #[test]
    fn test_solver_element_not_found() {
        let mut solver = NegotiationSolver::new();

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "nonexistent".into(),
            source_pad: "src".into(),
            sink_element: "sink".into(),
            sink_pad: "sink".into(),
        });

        let result = solver.solve();
        assert!(matches!(
            result.unwrap_err(),
            NegotiationError::ElementNotFound { .. }
        ));
    }

    #[test]
    fn test_solver_multi_output_element() {
        let mut solver = NegotiationSolver::new();

        // Source with multiple outputs (like a demuxer)
        let mut demux_caps = ElementCaps::new("demux");
        demux_caps.add_sink_pad("sink", MediaCaps::any());
        demux_caps.add_source_pad(
            "video",
            MediaCaps::from_format(FormatCaps::VideoRaw(VideoFormatCaps::any())),
        );
        demux_caps.add_source_pad(
            "audio",
            MediaCaps::from_format(FormatCaps::Audio(crate::format::AudioCodec::Aac)),
        );
        solver.add_element(demux_caps);

        // Video sink
        let mut video_sink = ElementCaps::new("video_sink");
        video_sink.add_sink_pad(
            "sink",
            MediaCaps::from_format(FormatCaps::VideoRaw(VideoFormatCaps::any())),
        );
        solver.add_element(video_sink);

        // Audio sink
        let mut audio_sink = ElementCaps::new("audio_sink");
        audio_sink.add_sink_pad(
            "sink",
            MediaCaps::from_format(FormatCaps::Audio(crate::format::AudioCodec::Aac)),
        );
        solver.add_element(audio_sink);

        // Link video output
        solver.add_link(LinkInfo {
            id: 0,
            source_element: "demux".into(),
            source_pad: "video".into(),
            sink_element: "video_sink".into(),
            sink_pad: "sink".into(),
        });

        // Link audio output
        solver.add_link(LinkInfo {
            id: 1,
            source_element: "demux".into(),
            source_pad: "audio".into(),
            sink_element: "audio_sink".into(),
            sink_pad: "sink".into(),
        });

        let result = solver.solve().unwrap();
        assert_eq!(result.link_caps.len(), 2);

        // Check video link negotiated correctly
        let video_link = result.link_caps.get(&0).unwrap();
        assert!(matches!(video_link.format, MediaFormat::VideoRaw(_)));

        // Check audio link negotiated correctly
        let audio_link = result.link_caps.get(&1).unwrap();
        assert!(matches!(audio_link.format, MediaFormat::Audio(_)));
    }

    #[test]
    fn test_solver_multi_format_source() {
        use crate::format::{CapsValue, FormatMemoryCap};

        let mut solver = NegotiationSolver::new();

        // Source supports multiple formats: YUYV (preferred), RGB, I420
        let yuyv_caps = VideoFormatCaps {
            width: CapsValue::Fixed(640),
            height: CapsValue::Fixed(480),
            pixel_format: CapsValue::Fixed(PixelFormat::Yuyv),
            framerate: CapsValue::Any,
        };
        let rgb_caps = VideoFormatCaps {
            width: CapsValue::Fixed(640),
            height: CapsValue::Fixed(480),
            pixel_format: CapsValue::Fixed(PixelFormat::Rgb24),
            framerate: CapsValue::Any,
        };
        let i420_caps = VideoFormatCaps {
            width: CapsValue::Fixed(640),
            height: CapsValue::Fixed(480),
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            framerate: CapsValue::Any,
        };

        let source_multi_caps = ElementMediaCaps::new(vec![
            FormatMemoryCap::new(yuyv_caps.into(), MemoryCaps::cpu_only()),
            FormatMemoryCap::new(rgb_caps.into(), MemoryCaps::cpu_only()),
            FormatMemoryCap::new(i420_caps.into(), MemoryCaps::cpu_only()),
        ]);

        let mut source_caps = ElementCaps::new("camera");
        source_caps.add_source_pad_multi("src", source_multi_caps);
        solver.add_element(source_caps);

        // Sink only accepts RGB
        let rgb_only = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Fixed(PixelFormat::Rgb24),
            framerate: CapsValue::Any,
        };
        let mut sink_caps = ElementCaps::new("display");
        sink_caps.add_sink_pad(
            "sink",
            MediaCaps::new(FormatCaps::VideoRaw(rgb_only), MemoryCaps::cpu_only()),
        );
        solver.add_element(sink_caps);

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "camera".into(),
            source_pad: "src".into(),
            sink_element: "display".into(),
            sink_pad: "sink".into(),
        });

        // Should negotiate to RGB (the common format)
        let result = solver.solve().unwrap();
        assert_eq!(result.link_caps.len(), 1);
        assert!(result.converters.is_empty());

        let link_caps = result.link_caps.get(&0).unwrap();
        if let MediaFormat::VideoRaw(vf) = &link_caps.format {
            assert_eq!(vf.pixel_format, PixelFormat::Rgb24);
            assert_eq!(vf.width, 640);
            assert_eq!(vf.height, 480);
        } else {
            panic!("Expected VideoRaw format");
        }
    }

    #[test]
    fn test_solver_multi_format_prefers_first_match() {
        use crate::format::{CapsValue, FormatMemoryCap};

        let mut solver = NegotiationSolver::new();

        // Source supports: RGB (preferred), I420
        let rgb_caps = VideoFormatCaps {
            width: CapsValue::Fixed(1920),
            height: CapsValue::Fixed(1080),
            pixel_format: CapsValue::Fixed(PixelFormat::Rgb24),
            framerate: CapsValue::Any,
        };
        let i420_caps = VideoFormatCaps {
            width: CapsValue::Fixed(1920),
            height: CapsValue::Fixed(1080),
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            framerate: CapsValue::Any,
        };

        let source_multi_caps = ElementMediaCaps::new(vec![
            FormatMemoryCap::new(rgb_caps.clone().into(), MemoryCaps::cpu_only()),
            FormatMemoryCap::new(i420_caps.clone().into(), MemoryCaps::cpu_only()),
        ]);

        let mut source_caps = ElementCaps::new("source");
        source_caps.add_source_pad_multi("src", source_multi_caps);
        solver.add_element(source_caps);

        // Sink accepts both RGB and I420
        let sink_multi_caps = ElementMediaCaps::new(vec![
            FormatMemoryCap::new(i420_caps.into(), MemoryCaps::cpu_only()), // I420 preferred by sink
            FormatMemoryCap::new(rgb_caps.into(), MemoryCaps::cpu_only()),
        ]);

        let mut sink_caps = ElementCaps::new("sink");
        sink_caps.add_sink_pad_multi("sink", sink_multi_caps);
        solver.add_element(sink_caps);

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "source".into(),
            source_pad: "src".into(),
            sink_element: "sink".into(),
            sink_pad: "sink".into(),
        });

        // Should negotiate to RGB (source's first choice that sink accepts)
        let result = solver.solve().unwrap();
        let link_caps = result.link_caps.get(&0).unwrap();
        if let MediaFormat::VideoRaw(vf) = &link_caps.format {
            assert_eq!(vf.pixel_format, PixelFormat::Rgb24);
        } else {
            panic!("Expected VideoRaw format");
        }
    }

    #[test]
    fn test_solver_multi_format_no_common_format() {
        use crate::format::{CapsValue, FormatMemoryCap};

        let mut solver = NegotiationSolver::new();

        // Source only supports YUYV
        let yuyv_caps = VideoFormatCaps {
            width: CapsValue::Fixed(640),
            height: CapsValue::Fixed(480),
            pixel_format: CapsValue::Fixed(PixelFormat::Yuyv),
            framerate: CapsValue::Any,
        };

        let source_multi_caps = ElementMediaCaps::new(vec![FormatMemoryCap::new(
            yuyv_caps.into(),
            MemoryCaps::cpu_only(),
        )]);

        let mut source_caps = ElementCaps::new("camera");
        source_caps.add_source_pad_multi("src", source_multi_caps);
        solver.add_element(source_caps);

        // Sink only accepts RGBA
        let rgba_caps = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Fixed(PixelFormat::Rgba),
            framerate: CapsValue::Any,
        };

        let mut sink_caps = ElementCaps::new("display");
        sink_caps.add_sink_pad(
            "sink",
            MediaCaps::new(FormatCaps::VideoRaw(rgba_caps), MemoryCaps::cpu_only()),
        );
        solver.add_element(sink_caps);

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "camera".into(),
            source_pad: "src".into(),
            sink_element: "display".into(),
            sink_pad: "sink".into(),
        });

        // Should fail - no common format and no converter
        let result = solver.solve();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            NegotiationError::NoCommonFormat { .. }
        ));
    }
}
