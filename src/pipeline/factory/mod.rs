//! Element factory for creating elements from parsed descriptions.
//!
//! Registration is split per domain: [`basics`] (io/testing/util/flow),
//! [`transform`] (transforms + timing), [`network`], [`ipc`], [`streaming`],
//! [`codec`], [`device`], [`rtp`]. Feature-gated elements register under
//! their `cfg`; names whose feature is off get a "requires cargo feature"
//! error via [`GATED_ELEMENTS`] instead of "unknown element".
//!
//! Constructors receive a [`Props`] view; after a constructor succeeds the
//! factory calls [`Props::finish`], so a property no constructor consumed is
//! a hard error rather than a silent no-op.

mod basics;
#[cfg(any(
    feature = "h264",
    feature = "av1-encode",
    feature = "av1-decode",
    feature = "vpx",
    feature = "opus",
    feature = "image-jpeg",
    feature = "image-png"
))]
mod codec;
mod device;
mod ipc;
mod network;
mod props;
mod rtp;
mod streaming;
mod transform;

pub use props::Props;

use crate::element::DynAsyncElement;
use crate::error::{Error, Result};
use crate::pipeline::parser::{ParsedElement, PropertyValue};
use crate::plugin::PluginRegistry;
use std::collections::HashMap;
use std::sync::Arc;

/// Type alias for element constructor functions.
pub type ElementConstructor = fn(&Props) -> Result<Box<DynAsyncElement<'static>>>;

/// Factory names that exist but are compiled out without their cargo feature.
///
/// Kept unconditional so a build without the feature can say *why* a name is
/// unavailable. `--list-elements` in parallax-launch prints these too.
pub const GATED_ELEMENTS: &[(&str, &str)] = &[
    ("autovideosink", "display"),
    ("v4l2src", "v4l2"),
    ("alsasrc", "alsa"),
    ("alsasink", "alsa"),
    ("screencapsrc", "screen-capture"),
    ("pipewiresrc", "pipewire"),
    ("pipewiresink", "pipewire"),
    ("libcamerasrc", "libcamera"),
    ("httpsrc", "http"),
    ("httpcachesrc", "http"),
    ("httpsink", "http"),
    ("wssrc", "websocket"),
    ("wssink", "websocket"),
    ("h264enc", "h264"),
    ("h264dec", "h264"),
    ("av1enc", "av1-encode"),
    ("av1dec", "av1-decode"),
    ("vp8dec", "vpx"),
    ("vp9dec", "vpx"),
    ("opusenc", "opus"),
    ("opusdec", "opus"),
    ("jpegenc", "image-jpeg"),
    ("jpegdec", "image-jpeg"),
    ("pngenc", "image-png"),
    ("pngdec", "image-png"),
    ("rtpsrc", "rtp"),
    ("rtpsink", "rtp"),
    ("rtpjitterbuffer", "rtp"),
    ("rtph264pay", "rtp"),
    ("rtph264depay", "rtp"),
    ("rtph265pay", "rtp"),
    ("rtph265depay", "rtp"),
    ("rtpvp8pay", "rtp"),
    ("rtpvp8depay", "rtp"),
    ("rtpvp9pay", "rtp"),
    ("rtpvp9depay", "rtp"),
    ("rtpopuspay", "rtp"),
    ("rtpopusdepay", "rtp"),
    ("rtpav1pay", "rtp"),
    // Not yet registered even with the feature on (async constructor);
    // the error message is still more useful than "unknown element".
    ("zenohsrc", "zenoh"),
    ("zenohsink", "zenoh"),
    ("rtspsrc", "rtsp"),
];

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

        basics::register(&mut factory);
        transform::register(&mut factory);
        network::register(&mut factory);
        ipc::register(&mut factory);
        streaming::register(&mut factory);
        #[cfg(any(
            feature = "h264",
            feature = "av1-encode",
            feature = "av1-decode",
            feature = "vpx",
            feature = "opus",
            feature = "image-jpeg",
            feature = "image-png"
        ))]
        codec::register(&mut factory);
        device::register(&mut factory);
        rtp::register(&mut factory);

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
            let map: HashMap<String, PropertyValue> = parsed.properties.iter().cloned().collect();
            let props = Props::new(&parsed.name, &map);
            let element = constructor(&props)?;
            props.finish()?;
            return Ok(element);
        }

        // Then try the plugin registry
        if let Some(ref registry) = self.plugin_registry
            && registry.has_element(&parsed.name)
        {
            return registry.create_element(&parsed.name).map_err(|e| {
                Error::Parse(format!("failed to create element '{}': {}", parsed.name, e))
            });
        }

        // A known-but-compiled-out name gets a pointed error.
        if let Some((_, feature)) = GATED_ELEMENTS.iter().find(|(n, _)| *n == parsed.name) {
            return Err(Error::Parse(format!(
                "element '{}' requires cargo feature \"{feature}\" (rebuild with --features {feature})",
                parsed.name
            )));
        }

        Err(Error::Parse(format!("unknown element: {}", parsed.name)))
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

    /// Names in [`GATED_ELEMENTS`] whose feature is off in this build.
    pub fn unavailable_elements(&self) -> Vec<(&'static str, &'static str)> {
        GATED_ELEMENTS
            .iter()
            .filter(|(name, _)| !self.constructors.contains_key(*name))
            .copied()
            .collect()
    }
}

impl Default for ElementFactory {
    fn default() -> Self {
        Self::new()
    }
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
        assert!(factory.is_registered("inspect"));
        assert!(
            factory.is_registered("tee"),
            "deprecated alias still parses"
        );
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

    #[test]
    fn test_unknown_property_is_rejected() {
        let factory = ElementFactory::new();
        let parsed = ParsedElement {
            name: "nullsource".to_string(),
            properties: vec![("frobnicate".to_string(), PropertyValue::Integer(1))],
        };

        let err = match factory.create(&parsed) {
            Ok(_) => panic!("unknown property must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("frobnicate") && err.contains("nullsource"),
            "{err}"
        );
    }

    #[test]
    fn test_name_property_is_always_accepted() {
        let factory = ElementFactory::new();
        let parsed = ParsedElement {
            name: "nullsink".to_string(),
            properties: vec![(
                "name".to_string(),
                PropertyValue::String("mysink".to_string()),
            )],
        };
        factory.create(&parsed).unwrap();
    }

    #[cfg(not(feature = "alsa"))]
    #[test]
    fn test_gated_element_error_names_the_feature() {
        let factory = ElementFactory::new();
        let parsed = ParsedElement {
            name: "alsasink".to_string(),
            properties: vec![],
        };
        let err = match factory.create(&parsed) {
            Ok(_) => panic!("gated element must not construct without its feature"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("alsa") && err.contains("feature"), "{err}");
    }

    #[test]
    fn test_gated_names_never_overlap_registered() {
        // A name that registered under its cfg must not also be reported
        // unavailable; unavailable_elements() filters against the live map.
        let factory = ElementFactory::new();
        for (name, _) in factory.unavailable_elements() {
            assert!(!factory.is_registered(name), "{name} is both");
        }
    }
}
