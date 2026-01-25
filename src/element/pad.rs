//! Pad abstraction for element inputs and outputs.
//!
//! Pads represent the connection points of elements. Each element can have
//! multiple input and output pads, allowing for complex routing topologies.

use std::sync::Arc;

/// Direction of a pad (input or output).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadDirection {
    /// An input pad (receives buffers from upstream).
    Input,
    /// An output pad (sends buffers downstream).
    Output,
}

/// Template for creating pads.
///
/// Pad templates define the characteristics of pads that an element can have.
/// They are used during pipeline construction to validate connections.
#[derive(Debug, Clone)]
pub struct PadTemplate {
    /// Name pattern for this pad (e.g., "src", "sink", "src_%u").
    pub name: String,
    /// Direction of this pad.
    pub direction: PadDirection,
    /// Whether this pad is always present or created on demand.
    pub presence: PadPresence,
}

/// Whether a pad is always present or created dynamically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadPresence {
    /// Pad is always present on the element.
    Always,
    /// Pad is created on demand (e.g., for demuxers).
    Sometimes,
    /// Pad is created when requested.
    Request,
}

impl PadTemplate {
    /// Create a new pad template.
    pub fn new(name: impl Into<String>, direction: PadDirection, presence: PadPresence) -> Self {
        Self {
            name: name.into(),
            direction,
            presence,
        }
    }

    /// Create a template for an always-present input pad.
    pub fn input(name: impl Into<String>) -> Self {
        Self::new(name, PadDirection::Input, PadPresence::Always)
    }

    /// Create a template for an always-present output pad.
    pub fn output(name: impl Into<String>) -> Self {
        Self::new(name, PadDirection::Output, PadPresence::Always)
    }

    /// Create a template for a sometimes-present output pad.
    pub fn sometimes_output(name: impl Into<String>) -> Self {
        Self::new(name, PadDirection::Output, PadPresence::Sometimes)
    }
}

/// A pad instance on an element.
///
/// Pads are the actual connection points used at runtime. They are created
/// from pad templates when an element is instantiated.
#[derive(Debug, Clone)]
pub struct Pad {
    /// Unique name of this pad within the element.
    name: String,
    /// Direction of this pad.
    direction: PadDirection,
    /// The template this pad was created from (if any).
    template: Option<Arc<PadTemplate>>,
}

impl Pad {
    /// Create a new pad.
    pub fn new(name: impl Into<String>, direction: PadDirection) -> Self {
        Self {
            name: name.into(),
            direction,
            template: None,
        }
    }

    /// Create a pad from a template.
    pub fn from_template(template: Arc<PadTemplate>, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            direction: template.direction,
            template: Some(template),
        }
    }

    /// Create a standard input pad named "sink".
    pub fn sink() -> Self {
        Self::new("sink", PadDirection::Input)
    }

    /// Create a standard output pad named "src".
    pub fn src() -> Self {
        Self::new("src", PadDirection::Output)
    }

    /// Get the pad's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the pad's direction.
    pub fn direction(&self) -> PadDirection {
        self.direction
    }

    /// Check if this is an input pad.
    pub fn is_input(&self) -> bool {
        self.direction == PadDirection::Input
    }

    /// Check if this is an output pad.
    pub fn is_output(&self) -> bool {
        self.direction == PadDirection::Output
    }

    /// Get the template this pad was created from.
    pub fn template(&self) -> Option<&Arc<PadTemplate>> {
        self.template.as_ref()
    }
}

/// Collection of pads for an element.
#[derive(Debug, Default)]
pub struct PadList {
    pads: Vec<Pad>,
}

impl PadList {
    /// Create an empty pad list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pad to the list.
    pub fn add(&mut self, pad: Pad) {
        self.pads.push(pad);
    }

    /// Get a pad by name.
    pub fn get(&self, name: &str) -> Option<&Pad> {
        self.pads.iter().find(|p| p.name() == name)
    }

    /// Get all input pads.
    pub fn inputs(&self) -> impl Iterator<Item = &Pad> {
        self.pads.iter().filter(|p| p.is_input())
    }

    /// Get all output pads.
    pub fn outputs(&self) -> impl Iterator<Item = &Pad> {
        self.pads.iter().filter(|p| p.is_output())
    }

    /// Get all pads.
    pub fn iter(&self) -> impl Iterator<Item = &Pad> {
        self.pads.iter()
    }

    /// Get the number of pads.
    pub fn len(&self) -> usize {
        self.pads.len()
    }

    /// Check if the pad list is empty.
    pub fn is_empty(&self) -> bool {
        self.pads.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_creation() {
        let input = Pad::sink();
        assert_eq!(input.name(), "sink");
        assert!(input.is_input());
        assert!(!input.is_output());

        let output = Pad::src();
        assert_eq!(output.name(), "src");
        assert!(output.is_output());
        assert!(!output.is_input());
    }

    #[test]
    fn test_pad_template() {
        let template = PadTemplate::input("sink");
        assert_eq!(template.direction, PadDirection::Input);
        assert_eq!(template.presence, PadPresence::Always);

        let template = PadTemplate::sometimes_output("src_%u");
        assert_eq!(template.direction, PadDirection::Output);
        assert_eq!(template.presence, PadPresence::Sometimes);
    }

    #[test]
    fn test_pad_from_template() {
        let template = Arc::new(PadTemplate::output("src"));
        let pad = Pad::from_template(template.clone(), "src");

        assert_eq!(pad.name(), "src");
        assert!(pad.is_output());
        assert!(pad.template().is_some());
    }

    #[test]
    fn test_pad_list() {
        let mut list = PadList::new();
        list.add(Pad::sink());
        list.add(Pad::src());
        list.add(Pad::new("aux_out", PadDirection::Output));

        assert_eq!(list.len(), 3);
        assert_eq!(list.inputs().count(), 1);
        assert_eq!(list.outputs().count(), 2);

        assert!(list.get("sink").is_some());
        assert!(list.get("src").is_some());
        assert!(list.get("nonexistent").is_none());
    }
}
