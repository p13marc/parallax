//! Negotiation solver for pipeline caps.

use super::converters::{ConverterFactory, ConverterRegistry, FormatType};
use super::error::NegotiationError;
use crate::format::{FormatCaps, MediaCaps, MediaFormat, MemoryCaps};
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
#[derive(Debug, Clone)]
pub struct ElementCaps {
    /// Element name.
    pub name: String,
    /// Caps for each sink (input) pad.
    pub sink_caps: Vec<MediaCaps>,
    /// Caps for each source (output) pad.
    pub source_caps: Vec<MediaCaps>,
}

/// Link in the pipeline graph.
#[derive(Debug, Clone)]
pub struct LinkInfo {
    /// Link identifier.
    pub id: usize,
    /// Source element name.
    pub source_element: String,
    /// Source pad index.
    pub source_pad: usize,
    /// Sink element name.
    pub sink_element: String,
    /// Sink pad index.
    pub sink_pad: usize,
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
    fn negotiate_link(&self, link: &LinkInfo) -> Result<LinkResult, NegotiationError> {
        // Get source and sink caps
        let source_caps = self.get_source_caps(link)?;
        let sink_caps = self.get_sink_caps(link)?;

        // Try direct intersection
        if let Some(intersected) = source_caps.intersect(&sink_caps) {
            // Can negotiate directly - use fixate_with_defaults to handle Any/Any case
            let format = intersected.fixate_format_with_defaults();
            let memory_type = intersected.fixate_memory().unwrap_or(MemoryType::Cpu); // Default to CPU if Any

            return Ok(LinkResult::Direct(LinkNegotiation {
                link_id: link.id,
                format,
                memory_type,
            }));
        }

        // Try to find a converter
        if let Some(registry) = &self.converter_registry {
            let source_format_type = FormatType::from(&source_caps.format);
            let sink_format_type = FormatType::from(&sink_caps.format);
            let source_memory = source_caps.memory.fixate().unwrap_or(MemoryType::Cpu);
            let sink_memory = sink_caps.memory.fixate().unwrap_or(MemoryType::Cpu);

            if let Some((factories, cost)) = registry.find_path(
                source_format_type,
                sink_format_type,
                source_memory,
                sink_memory,
            ) {
                // Use the first factory (for now, we only support single-hop)
                if let Some(factory) = factories.into_iter().next() {
                    // Fixate source format for the link
                    let format = source_caps.format.fixate().ok_or_else(|| {
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
                                "Convert {:?}/{:?} -> {:?}/{:?}",
                                source_format_type, source_memory, sink_format_type, sink_memory
                            ),
                            cost,
                        },
                    });
                }
            }
        }

        // No direct match and no converter available
        Err(NegotiationError::no_common_format(
            &link.source_element,
            &link.sink_element,
            &format!("{:?}", source_caps.format),
            &format!("{:?}", sink_caps.format),
        ))
    }

    /// Get source (output) caps for a link.
    fn get_source_caps(&self, link: &LinkInfo) -> Result<MediaCaps, NegotiationError> {
        let element = self.elements.get(&link.source_element).ok_or_else(|| {
            NegotiationError::ElementNotFound {
                name: link.source_element.clone(),
            }
        })?;

        element
            .source_caps
            .get(link.source_pad)
            .cloned()
            .ok_or_else(|| {
                NegotiationError::Internal(format!(
                    "Source pad {} not found on element {}",
                    link.source_pad, link.source_element
                ))
            })
    }

    /// Get sink (input) caps for a link.
    fn get_sink_caps(&self, link: &LinkInfo) -> Result<MediaCaps, NegotiationError> {
        let element = self.elements.get(&link.sink_element).ok_or_else(|| {
            NegotiationError::ElementNotFound {
                name: link.sink_element.clone(),
            }
        })?;

        element
            .sink_caps
            .get(link.sink_pad)
            .cloned()
            .ok_or_else(|| {
                NegotiationError::Internal(format!(
                    "Sink pad {} not found on element {}",
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
    use crate::format::{Framerate, PixelFormat, VideoCodec, VideoFormat, VideoFormatCaps};

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
        solver.add_element(ElementCaps {
            name: "source".into(),
            sink_caps: vec![],
            source_caps: vec![MediaCaps::from(video_1080p())],
        });

        // Sink accepts any video
        solver.add_element(ElementCaps {
            name: "sink".into(),
            sink_caps: vec![MediaCaps::from_format(FormatCaps::VideoRaw(
                VideoFormatCaps::any(),
            ))],
            source_caps: vec![],
        });

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "source".into(),
            source_pad: 0,
            sink_element: "sink".into(),
            sink_pad: 0,
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
        solver.add_element(ElementCaps {
            name: "source".into(),
            sink_caps: vec![],
            source_caps: vec![MediaCaps::from_format(FormatCaps::VideoRaw(
                VideoFormatCaps::any(),
            ))],
        });

        // Sink only accepts H.264
        solver.add_element(ElementCaps {
            name: "sink".into(),
            sink_caps: vec![MediaCaps::from_format(FormatCaps::Video(VideoCodec::H264))],
            source_caps: vec![],
        });

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "source".into(),
            source_pad: 0,
            sink_element: "sink".into(),
            sink_pad: 0,
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
        solver.add_element(ElementCaps {
            name: "source".into(),
            sink_caps: vec![],
            source_caps: vec![MediaCaps::new(
                FormatCaps::VideoRaw(VideoFormatCaps::any()),
                MemoryCaps::cpu_only(),
            )],
        });

        // Sink accepts CPU memory
        solver.add_element(ElementCaps {
            name: "sink".into(),
            sink_caps: vec![MediaCaps::new(
                FormatCaps::VideoRaw(VideoFormatCaps::any()),
                MemoryCaps::cpu_only(),
            )],
            source_caps: vec![],
        });

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "source".into(),
            source_pad: 0,
            sink_element: "sink".into(),
            sink_pad: 0,
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
        solver.add_element(ElementCaps {
            name: "source".into(),
            sink_caps: vec![],
            source_caps: vec![MediaCaps::from(format.clone())],
        });

        solver.add_element(ElementCaps {
            name: "filter".into(),
            sink_caps: vec![MediaCaps::from_format(FormatCaps::VideoRaw(
                VideoFormatCaps::any(),
            ))],
            source_caps: vec![MediaCaps::from_format(FormatCaps::VideoRaw(
                VideoFormatCaps::any(),
            ))],
        });

        solver.add_element(ElementCaps {
            name: "sink".into(),
            sink_caps: vec![MediaCaps::from_format(FormatCaps::VideoRaw(
                VideoFormatCaps::any(),
            ))],
            source_caps: vec![],
        });

        solver.add_link(LinkInfo {
            id: 0,
            source_element: "source".into(),
            source_pad: 0,
            sink_element: "filter".into(),
            sink_pad: 0,
        });

        solver.add_link(LinkInfo {
            id: 1,
            source_element: "filter".into(),
            source_pad: 0,
            sink_element: "sink".into(),
            sink_pad: 0,
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
            source_pad: 0,
            sink_element: "sink".into(),
            sink_pad: 0,
        });

        let result = solver.solve();
        assert!(matches!(
            result.unwrap_err(),
            NegotiationError::ElementNotFound { .. }
        ));
    }
}
