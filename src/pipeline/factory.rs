//! Element factory for creating elements from parsed descriptions.

use crate::element::{ElementAdapter, ElementDyn, SinkAdapter, SourceAdapter};
use crate::elements::{FileSink, FileSrc, NullSink, NullSource, PassThrough, Tee};
use crate::error::{Error, Result};
use crate::pipeline::parser::{ParsedElement, PropertyValue};
use std::collections::HashMap;

/// Type alias for element constructor functions.
type ElementConstructor = fn(&HashMap<String, PropertyValue>) -> Result<Box<dyn ElementDyn>>;

/// Registry of element constructors.
pub struct ElementFactory {
    constructors: HashMap<String, ElementConstructor>,
}

impl ElementFactory {
    /// Create a new factory with built-in elements registered.
    pub fn new() -> Self {
        let mut factory = Self {
            constructors: HashMap::new(),
        };

        // Register built-in elements
        factory.register("nullsource", create_nullsource);
        factory.register("nullsink", create_nullsink);
        factory.register("passthrough", create_passthrough);
        factory.register("tee", create_tee);
        factory.register("filesrc", create_filesrc);
        factory.register("filesink", create_filesink);

        factory
    }

    /// Register a custom element constructor.
    pub fn register(&mut self, name: &str, constructor: ElementConstructor) {
        self.constructors.insert(name.to_string(), constructor);
    }

    /// Create an element from a parsed description.
    pub fn create(&self, parsed: &ParsedElement) -> Result<Box<dyn ElementDyn>> {
        let constructor = self
            .constructors
            .get(&parsed.name)
            .ok_or_else(|| Error::InvalidSegment(format!("unknown element: {}", parsed.name)))?;

        let props: HashMap<String, PropertyValue> = parsed.properties.iter().cloned().collect();
        constructor(&props)
    }

    /// Check if an element type is registered.
    pub fn is_registered(&self, name: &str) -> bool {
        self.constructors.contains_key(name)
    }
}

impl Default for ElementFactory {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in element constructors

fn create_nullsource(props: &HashMap<String, PropertyValue>) -> Result<Box<dyn ElementDyn>> {
    let count = props.get("count").and_then(|v| v.as_u64()).unwrap_or(100);

    let buffer_size = props
        .get("buffer-size")
        .and_then(|v| v.as_u64())
        .unwrap_or(64) as usize;

    let source = NullSource::new(count).with_buffer_size(buffer_size);
    Ok(Box::new(SourceAdapter::new(source)))
}

fn create_nullsink(_props: &HashMap<String, PropertyValue>) -> Result<Box<dyn ElementDyn>> {
    Ok(Box::new(SinkAdapter::new(NullSink::new())))
}

fn create_passthrough(_props: &HashMap<String, PropertyValue>) -> Result<Box<dyn ElementDyn>> {
    Ok(Box::new(ElementAdapter::new(PassThrough::new())))
}

fn create_tee(_props: &HashMap<String, PropertyValue>) -> Result<Box<dyn ElementDyn>> {
    Ok(Box::new(ElementAdapter::new(Tee::new())))
}

fn create_filesrc(props: &HashMap<String, PropertyValue>) -> Result<Box<dyn ElementDyn>> {
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

    Ok(Box::new(SourceAdapter::new(src)))
}

fn create_filesink(props: &HashMap<String, PropertyValue>) -> Result<Box<dyn ElementDyn>> {
    let location = props
        .get("location")
        .map(|v| v.as_string())
        .ok_or_else(|| {
            Error::InvalidSegment("filesink requires 'location' property".to_string())
        })?;

    let sink = FileSink::new(&location);
    Ok(Box::new(SinkAdapter::new(sink)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::ElementType;

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
