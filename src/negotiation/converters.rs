//! Converter registry for format and memory type conversions.

use crate::format::{FormatCaps, MediaFormat};
use crate::memory::MemoryType;
use std::collections::HashMap;
use std::sync::Arc;

/// Factory function for creating converter elements.
pub type ConverterFactory = Arc<dyn Fn() -> Box<dyn ConverterElement> + Send + Sync>;

/// Trait for converter elements.
///
/// Converters transform buffers between formats or memory types.
pub trait ConverterElement: Send {
    /// Get the name of this converter.
    fn name(&self) -> &str;

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

/// Registry for format and memory converters.
///
/// The registry maintains a collection of converter factories and can find
/// conversion paths between incompatible formats/memory types.
#[derive(Default)]
pub struct ConverterRegistry {
    /// Direct converters: key -> (factory, cost).
    converters: HashMap<ConversionKey, (ConverterFactory, u32)>,
}

impl ConverterRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a converter.
    pub fn register(
        &mut self,
        from_format: FormatType,
        to_format: FormatType,
        from_memory: MemoryType,
        to_memory: MemoryType,
        cost: u32,
        factory: ConverterFactory,
    ) {
        let key = ConversionKey {
            from_format,
            to_format,
            from_memory,
            to_memory,
        };
        self.converters.insert(key, (factory, cost));
    }

    /// Find a direct converter.
    pub fn find_direct(
        &self,
        from_format: FormatType,
        to_format: FormatType,
        from_memory: MemoryType,
        to_memory: MemoryType,
    ) -> Option<(&ConverterFactory, u32)> {
        let key = ConversionKey {
            from_format,
            to_format,
            from_memory,
            to_memory,
        };
        self.converters.get(&key).map(|(f, c)| (f, *c))
    }

    /// Find a conversion path (potentially multiple converters).
    ///
    /// Returns a list of converter factories and total cost.
    /// Uses simple BFS for now; could be optimized with Dijkstra.
    pub fn find_path(
        &self,
        from_format: FormatType,
        to_format: FormatType,
        from_memory: MemoryType,
        to_memory: MemoryType,
    ) -> Option<(Vec<ConverterFactory>, u32)> {
        // Check for direct conversion first
        if let Some((factory, cost)) =
            self.find_direct(from_format, to_format, from_memory, to_memory)
        {
            return Some((vec![factory.clone()], cost));
        }

        // If formats match, check for memory-only conversion
        if from_format == to_format
            || from_format == FormatType::Any
            || to_format == FormatType::Any
        {
            if let Some((factory, cost)) =
                self.find_direct(FormatType::Any, FormatType::Any, from_memory, to_memory)
            {
                return Some((vec![factory.clone()], cost));
            }
        }

        // TODO: Implement multi-hop path finding with Dijkstra
        // For now, only direct conversions are supported
        None
    }

    /// Check if a conversion is possible.
    pub fn can_convert(
        &self,
        from_format: FormatType,
        to_format: FormatType,
        from_memory: MemoryType,
        to_memory: MemoryType,
    ) -> bool {
        self.find_path(from_format, to_format, from_memory, to_memory)
            .is_some()
    }

    /// Get the number of registered converters.
    pub fn len(&self) -> usize {
        self.converters.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.converters.is_empty()
    }
}

impl std::fmt::Debug for ConverterRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConverterRegistry")
            .field("num_converters", &self.converters.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyConverter {
        name: String,
        cost: u32,
    }

    impl ConverterElement for DummyConverter {
        fn name(&self) -> &str {
            &self.name
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
            self.cost
        }
    }

    #[test]
    fn test_registry_register_and_find() {
        let mut registry = ConverterRegistry::new();

        let factory: ConverterFactory = Arc::new(|| {
            Box::new(DummyConverter {
                name: "test".to_string(),
                cost: 10,
            })
        });

        registry.register(
            FormatType::VideoRaw,
            FormatType::VideoRaw,
            MemoryType::Cpu,
            MemoryType::GpuDevice,
            10,
            factory,
        );

        assert_eq!(registry.len(), 1);

        let result = registry.find_direct(
            FormatType::VideoRaw,
            FormatType::VideoRaw,
            MemoryType::Cpu,
            MemoryType::GpuDevice,
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, 10);
    }

    #[test]
    fn test_registry_find_path_direct() {
        let mut registry = ConverterRegistry::new();

        let factory: ConverterFactory = Arc::new(|| {
            Box::new(DummyConverter {
                name: "gpu_upload".to_string(),
                cost: 5,
            })
        });

        registry.register(
            FormatType::VideoRaw,
            FormatType::VideoRaw,
            MemoryType::Cpu,
            MemoryType::GpuDevice,
            5,
            factory,
        );

        let path = registry.find_path(
            FormatType::VideoRaw,
            FormatType::VideoRaw,
            MemoryType::Cpu,
            MemoryType::GpuDevice,
        );

        assert!(path.is_some());
        let (factories, cost) = path.unwrap();
        assert_eq!(factories.len(), 1);
        assert_eq!(cost, 5);
    }

    #[test]
    fn test_registry_no_converter() {
        let registry = ConverterRegistry::new();

        let result = registry.find_direct(
            FormatType::VideoRaw,
            FormatType::AudioRaw,
            MemoryType::Cpu,
            MemoryType::Cpu,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_format_type_from_caps() {
        use crate::format::{AudioFormatCaps, VideoFormatCaps};

        let video = FormatCaps::VideoRaw(VideoFormatCaps::any());
        let audio = FormatCaps::AudioRaw(AudioFormatCaps::any());
        let bytes = FormatCaps::Bytes;

        assert_eq!(FormatType::from(&video), FormatType::VideoRaw);
        assert_eq!(FormatType::from(&audio), FormatType::AudioRaw);
        assert_eq!(FormatType::from(&bytes), FormatType::Bytes);
    }
}
