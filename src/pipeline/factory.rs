//! Element factory for creating elements from parsed descriptions.

use crate::element::{DynAsyncElement, ElementAdapter, SinkAdapter, SourceAdapter};
use crate::elements::{FileSink, FileSrc, NullSink, NullSource, PassThrough, Tee};
use crate::error::{Error, Result};
use crate::pipeline::parser::{ParsedElement, PropertyValue};
use crate::plugin::PluginRegistry;
use std::collections::HashMap;
use std::sync::Arc;

/// Type alias for element constructor functions.
type ElementConstructor =
    fn(&HashMap<String, PropertyValue>) -> Result<Box<DynAsyncElement<'static>>>;

/// Registry of element constructors.
pub struct ElementFactory {
    constructors: HashMap<String, ElementConstructor>,
    /// Optional plugin registry for dynamically loaded elements.
    plugin_registry: Option<Arc<PluginRegistry>>,
}

impl ElementFactory {
    /// Create a new factory with built-in elements registered.
    pub fn new() -> Self {
        let mut factory = Self {
            constructors: HashMap::new(),
            plugin_registry: None,
        };

        // Register built-in elements
        factory.register("nullsource", create_nullsource);
        factory.register("nullsink", create_nullsink);
        factory.register("passthrough", create_passthrough);
        factory.register("tee", create_tee);
        factory.register("filesrc", create_filesrc);
        factory.register("filesink", create_filesink);

        // Video display (feature-gated)
        #[cfg(feature = "display")]
        factory.register("autovideosink", create_autovideosink);

        // Video processing
        factory.register("videoconvert", create_videoconvert);

        // Test sources
        factory.register("videotestsrc", create_videotestsrc);

        // Device sources (feature-gated)
        #[cfg(feature = "v4l2")]
        factory.register("v4l2src", create_v4l2src);

        factory
    }

    /// Create a factory with a plugin registry.
    ///
    /// Elements from the plugin registry will be available in addition
    /// to built-in elements. Built-in elements take precedence.
    pub fn with_plugin_registry(registry: Arc<PluginRegistry>) -> Self {
        let mut factory = Self::new();
        factory.plugin_registry = Some(registry);
        factory
    }

    /// Set the plugin registry.
    pub fn set_plugin_registry(&mut self, registry: Arc<PluginRegistry>) {
        self.plugin_registry = Some(registry);
    }

    /// Register a custom element constructor.
    pub fn register(&mut self, name: &str, constructor: ElementConstructor) {
        self.constructors.insert(name.to_string(), constructor);
    }

    /// Create an element from a parsed description.
    pub fn create(&self, parsed: &ParsedElement) -> Result<Box<DynAsyncElement<'static>>> {
        // First try built-in constructors
        if let Some(constructor) = self.constructors.get(&parsed.name) {
            let props: HashMap<String, PropertyValue> = parsed.properties.iter().cloned().collect();
            return constructor(&props);
        }

        // Then try the plugin registry
        if let Some(ref registry) = self.plugin_registry {
            if registry.has_element(&parsed.name) {
                return registry.create_element(&parsed.name).map_err(|e| {
                    Error::InvalidSegment(format!(
                        "failed to create element '{}': {}",
                        parsed.name, e
                    ))
                });
            }
        }

        Err(Error::InvalidSegment(format!(
            "unknown element: {}",
            parsed.name
        )))
    }

    /// Check if an element type is registered.
    pub fn is_registered(&self, name: &str) -> bool {
        if self.constructors.contains_key(name) {
            return true;
        }
        if let Some(ref registry) = self.plugin_registry {
            return registry.has_element(name);
        }
        false
    }

    /// List all available element names.
    pub fn list_elements(&self) -> Vec<String> {
        let mut names: Vec<String> = self.constructors.keys().cloned().collect();
        if let Some(ref registry) = self.plugin_registry {
            names.extend(registry.list_elements());
        }
        names.sort();
        names.dedup();
        names
    }
}

impl Default for ElementFactory {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in element constructors

fn create_nullsource(
    props: &HashMap<String, PropertyValue>,
) -> Result<Box<DynAsyncElement<'static>>> {
    let count = props.get("count").and_then(|v| v.as_u64()).unwrap_or(100);

    let buffer_size = props
        .get("buffer-size")
        .and_then(|v| v.as_u64())
        .unwrap_or(64) as usize;

    let source = NullSource::new(count).with_buffer_size(buffer_size);
    Ok(DynAsyncElement::new_box(SourceAdapter::new(source)))
}

fn create_nullsink(
    _props: &HashMap<String, PropertyValue>,
) -> Result<Box<DynAsyncElement<'static>>> {
    Ok(DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())))
}

fn create_passthrough(
    _props: &HashMap<String, PropertyValue>,
) -> Result<Box<DynAsyncElement<'static>>> {
    Ok(DynAsyncElement::new_box(ElementAdapter::new(
        PassThrough::new(),
    )))
}

fn create_tee(_props: &HashMap<String, PropertyValue>) -> Result<Box<DynAsyncElement<'static>>> {
    Ok(DynAsyncElement::new_box(ElementAdapter::new(Tee::new())))
}

fn create_filesrc(props: &HashMap<String, PropertyValue>) -> Result<Box<DynAsyncElement<'static>>> {
    let location = props
        .get("location")
        .map(|v| v.as_string())
        .ok_or_else(|| Error::InvalidSegment("filesrc requires 'location' property".to_string()))?;

    let chunk_size = props
        .get("chunk-size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let mut src = FileSrc::new(&location);
    if let Some(size) = chunk_size {
        src = src.with_chunk_size(size);
    }

    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

fn create_filesink(
    props: &HashMap<String, PropertyValue>,
) -> Result<Box<DynAsyncElement<'static>>> {
    let location = props
        .get("location")
        .map(|v| v.as_string())
        .ok_or_else(|| {
            Error::InvalidSegment("filesink requires 'location' property".to_string())
        })?;

    let sink = FileSink::new(&location);
    Ok(DynAsyncElement::new_box(SinkAdapter::new(sink)))
}

#[cfg(feature = "display")]
fn create_autovideosink(
    props: &HashMap<String, PropertyValue>,
) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::app::AutoVideoSink;

    let mut sink = AutoVideoSink::new();

    if let Some(title) = props.get("title").map(|v| v.as_string()) {
        sink = sink.with_title(title);
    }

    if let (Some(w), Some(h)) = (
        props.get("width").and_then(|v| v.as_u64()),
        props.get("height").and_then(|v| v.as_u64()),
    ) {
        sink = sink.with_size(w as u32, h as u32);
    }

    Ok(DynAsyncElement::new_box(SinkAdapter::new(sink)))
}

fn create_videoconvert(
    _props: &HashMap<String, PropertyValue>,
) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::format::PixelFormat;
    use crate::negotiation::VideoConvert;

    // Default to RGBA output (most common for display)
    let element = VideoConvert::new(PixelFormat::Rgba);
    Ok(DynAsyncElement::new_box(ElementAdapter::new(element)))
}

fn create_videotestsrc(
    props: &HashMap<String, PropertyValue>,
) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::testing::{PixelFormat, VideoPattern, VideoTestSrc};

    let mut src = VideoTestSrc::new();

    // Pattern
    if let Some(pattern) = props.get("pattern").map(|v| v.as_string()) {
        src = src.with_pattern(match pattern.as_str() {
            "smpte" | "smpte-color-bars" => VideoPattern::SmpteColorBars,
            "checkerboard" => VideoPattern::Checkerboard,
            "solid" => VideoPattern::SolidColor,
            "ball" | "moving-ball" => VideoPattern::MovingBall,
            "gradient" => VideoPattern::Gradient,
            "black" => VideoPattern::Black,
            "white" => VideoPattern::White,
            "red" => VideoPattern::Red,
            "green" => VideoPattern::Green,
            "blue" => VideoPattern::Blue,
            "circular" => VideoPattern::Circular,
            "snow" => VideoPattern::Snow,
            _ => VideoPattern::SmpteColorBars,
        });
    }

    // Resolution
    if let (Some(w), Some(h)) = (
        props.get("width").and_then(|v| v.as_u64()),
        props.get("height").and_then(|v| v.as_u64()),
    ) {
        src = src.with_resolution(w as u32, h as u32);
    }

    // Frame count
    if let Some(count) = props.get("num-buffers").and_then(|v| v.as_u64()) {
        src = src.with_num_frames(count);
    }

    // FPS
    if let Some(fps) = props.get("framerate").and_then(|v| v.as_u64()) {
        src = src.with_framerate(fps as u32, 1);
    }

    // Use RGBA format for display compatibility
    src = src.with_pixel_format(PixelFormat::Rgba32);

    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

#[cfg(feature = "v4l2")]
fn create_v4l2src(props: &HashMap<String, PropertyValue>) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::device::V4l2Src;

    let device = props
        .get("device")
        .map(|v| v.as_string())
        .unwrap_or_else(|| "/dev/video0".to_string());

    let src = V4l2Src::new(&device)?;
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{AsyncElementDyn, ElementType};

    #[test]
    fn test_factory_creation() {
        let factory = ElementFactory::new();
        assert!(factory.is_registered("nullsource"));
        assert!(factory.is_registered("nullsink"));
        assert!(factory.is_registered("passthrough"));
        assert!(factory.is_registered("tee"));
        assert!(factory.is_registered("filesrc"));
        assert!(factory.is_registered("filesink"));
        assert!(!factory.is_registered("unknown"));
    }

    #[test]
    fn test_create_nullsource() {
        let factory = ElementFactory::new();
        let parsed = ParsedElement {
            name: "nullsource".to_string(),
            properties: vec![("count".to_string(), PropertyValue::Integer(50))],
        };

        let element = factory.create(&parsed).unwrap();
        assert_eq!(element.element_type(), ElementType::Source);
    }

    #[test]
    fn test_create_nullsink() {
        let factory = ElementFactory::new();
        let parsed = ParsedElement {
            name: "nullsink".to_string(),
            properties: vec![],
        };

        let element = factory.create(&parsed).unwrap();
        assert_eq!(element.element_type(), ElementType::Sink);
    }

    #[test]
    fn test_create_passthrough() {
        let factory = ElementFactory::new();
        let parsed = ParsedElement {
            name: "passthrough".to_string(),
            properties: vec![],
        };

        let element = factory.create(&parsed).unwrap();
        assert_eq!(element.element_type(), ElementType::Transform);
    }

    #[test]
    fn test_create_filesrc_requires_location() {
        let factory = ElementFactory::new();
        let parsed = ParsedElement {
            name: "filesrc".to_string(),
            properties: vec![],
        };

        let result = factory.create(&parsed);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_filesrc_with_location() {
        let factory = ElementFactory::new();
        let parsed = ParsedElement {
            name: "filesrc".to_string(),
            properties: vec![(
                "location".to_string(),
                PropertyValue::String("/path/to/file".to_string()),
            )],
        };

        let element = factory.create(&parsed).unwrap();
        assert_eq!(element.element_type(), ElementType::Source);
    }

    #[test]
    fn test_unknown_element() {
        let factory = ElementFactory::new();
        let parsed = ParsedElement {
            name: "unknown_element".to_string(),
            properties: vec![],
        };

        let result = factory.create(&parsed);
        assert!(result.is_err());
    }
}
