//! Pipeline string parser using winnow.
//!
//! Parses GStreamer-like pipeline descriptions:
//!
//! ```text
//! filesrc location=/path/to/file ! passthrough ! filesink location=/output
//! nullsource count=100 ! tee ! nullsink
//! ```
//!
//! # Syntax
//!
//! - Elements are separated by `!`
//! - Properties are specified as `name=value` after the element name
//! - Values can be quoted strings, numbers, or bare identifiers
//! - Whitespace is optional around `!` and `=`

use crate::error::{Error, Result};
use winnow::Parser;
use winnow::ascii::{alpha1, digit1, multispace0};
use winnow::combinator::{alt, delimited, opt, repeat, separated};
use winnow::error::ContextError;
use winnow::token::{take_till, take_while};

type WResult<T> = std::result::Result<T, ContextError>;

/// A parsed element with its name and properties.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedElement {
    /// The element type name (e.g., "filesrc", "passthrough").
    pub name: String,
    /// Properties as key-value pairs.
    pub properties: Vec<(String, PropertyValue)>,
}

/// A property value in the pipeline description.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    /// A string value (quoted or unquoted).
    String(String),
    /// An integer value.
    Integer(i64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
}

impl PropertyValue {
    /// Get as a string, converting if necessary.
    pub fn as_string(&self) -> String {
        match self {
            PropertyValue::String(s) => s.clone(),
            PropertyValue::Integer(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::Bool(b) => b.to_string(),
        }
    }

    /// Try to get as an integer.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            PropertyValue::Integer(i) => Some(*i),
            PropertyValue::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Try to get as a u64.
    pub fn as_u64(&self) -> Option<u64> {
        self.as_i64().and_then(|i| u64::try_from(i).ok())
    }

    /// Try to get as a float.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PropertyValue::Float(f) => Some(*f),
            PropertyValue::Integer(i) => Some(*i as f64),
            PropertyValue::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Try to get as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PropertyValue::Bool(b) => Some(*b),
            PropertyValue::String(s) => match s.to_lowercase().as_str() {
                "true" | "yes" | "1" => Some(true),
                "false" | "no" | "0" => Some(false),
                _ => None,
            },
            PropertyValue::Integer(i) => Some(*i != 0),
            _ => None,
        }
    }
}

/// A parsed pipeline description.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedPipeline {
    /// The elements in order from source to sink.
    pub elements: Vec<ParsedElement>,
}

/// Parse a pipeline description string.
///
/// # Example
///
/// ```rust
/// use parallax::pipeline::parser::parse_pipeline;
///
/// let pipeline = parse_pipeline("nullsource count=10 ! passthrough ! nullsink").unwrap();
/// assert_eq!(pipeline.elements.len(), 3);
/// assert_eq!(pipeline.elements[0].name, "nullsource");
/// ```
pub fn parse_pipeline(input: &str) -> Result<ParsedPipeline> {
    pipeline
        .parse(input.trim())
        .map_err(|e| Error::InvalidSegment(format!("parse error: {e}")))
}

/// Parse a complete pipeline.
fn pipeline(input: &mut &str) -> WResult<ParsedPipeline> {
    let elements = separated(1.., element, link_separator).parse_next(input)?;

    // Ensure we consumed all input
    multispace0.parse_next(input)?;
    if !input.is_empty() {
        return Err(ContextError::new());
    }

    Ok(ParsedPipeline { elements })
}

/// Parse an element (name + optional properties).
fn element(input: &mut &str) -> WResult<ParsedElement> {
    let _ = multispace0.parse_next(input)?;
    let name: &str = identifier.parse_next(input)?;
    let _ = multispace0.parse_next(input)?;

    let properties: Vec<(String, PropertyValue)> = repeat(0.., property).parse_next(input)?;

    Ok(ParsedElement {
        name: name.to_string(),
        properties,
    })
}

/// Parse the link separator `!`.
fn link_separator(input: &mut &str) -> WResult<()> {
    let _ = multispace0.parse_next(input)?;
    let _ = '!'.parse_next(input)?;
    let _ = multispace0.parse_next(input)?;
    Ok(())
}

/// Parse an identifier (element name or property name).
fn identifier<'a>(input: &mut &'a str) -> WResult<&'a str> {
    (
        alt((alpha1::<_, ContextError>, "_")),
        take_while(0.., |c: char| c.is_alphanumeric() || c == '_' || c == '-'),
    )
        .take()
        .parse_next(input)
}

/// Parse a property (key=value).
fn property(input: &mut &str) -> WResult<(String, PropertyValue)> {
    let _ = multispace0.parse_next(input)?;

    // Check if this looks like a property (identifier followed by =)
    // If not, don't consume anything
    let checkpoint = *input;

    let key: &str = match identifier.parse_next(input) {
        Ok(k) => k,
        Err(_) => {
            *input = checkpoint;
            return Err(ContextError::new());
        }
    };

    let _ = multispace0.parse_next(input)?;

    if input.starts_with('=') {
        let _ = '='.parse_next(input)?;
    } else {
        // Not a property, backtrack
        *input = checkpoint;
        return Err(ContextError::new());
    }

    let _ = multispace0.parse_next(input)?;
    let value = property_value.parse_next(input)?;
    let _ = multispace0.parse_next(input)?;

    Ok((key.to_string(), value))
}

/// Parse a property value.
fn property_value(input: &mut &str) -> WResult<PropertyValue> {
    alt((
        quoted_string.map(PropertyValue::String),
        boolean.map(PropertyValue::Bool),
        float.map(PropertyValue::Float),
        integer.map(PropertyValue::Integer),
        bare_string.map(PropertyValue::String),
    ))
    .parse_next(input)
}

/// Parse a quoted string.
fn quoted_string(input: &mut &str) -> WResult<String> {
    alt((
        delimited('"', take_till(0.., '"'), '"'),
        delimited('\'', take_till(0.., '\''), '\''),
    ))
    .map(|s: &str| s.to_string())
    .parse_next(input)
}

/// Parse a boolean.
fn boolean(input: &mut &str) -> WResult<bool> {
    alt((
        "true".map(|_| true),
        "false".map(|_| false),
        "yes".map(|_| true),
        "no".map(|_| false),
    ))
    .parse_next(input)
}

/// Parse an integer.
fn integer(input: &mut &str) -> WResult<i64> {
    let negative = opt('-').parse_next(input)?;
    let digits: &str = digit1.parse_next(input)?;

    // Make sure this isn't a float (no decimal point follows)
    if input.starts_with('.') {
        return Err(ContextError::new());
    }

    let value: i64 = digits.parse().map_err(|_| ContextError::new())?;

    Ok(if negative.is_some() { -value } else { value })
}

/// Parse a float.
fn float(input: &mut &str) -> WResult<f64> {
    let negative = opt('-').parse_next(input)?;
    let int_part: &str = digit1.parse_next(input)?;
    let _ = '.'.parse_next(input)?;
    let frac_part: &str = digit1.parse_next(input)?;

    let s = format!(
        "{}{}.{}",
        if negative.is_some() { "-" } else { "" },
        int_part,
        frac_part
    );
    s.parse().map_err(|_| ContextError::new())
}

/// Parse a bare (unquoted) string value.
/// Stops at whitespace or `!`.
fn bare_string(input: &mut &str) -> WResult<String> {
    take_while(1.., |c: char| !c.is_whitespace() && c != '!' && c != '=')
        .map(|s: &str| s.to_string())
        .parse_next(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_element() {
        let result = parse_pipeline("nullsink").unwrap();
        assert_eq!(result.elements.len(), 1);
        assert_eq!(result.elements[0].name, "nullsink");
        assert!(result.elements[0].properties.is_empty());
    }

    #[test]
    fn test_parse_element_with_property() {
        let result = parse_pipeline("nullsource count=100").unwrap();
        assert_eq!(result.elements.len(), 1);
        assert_eq!(result.elements[0].name, "nullsource");
        assert_eq!(result.elements[0].properties.len(), 1);
        assert_eq!(result.elements[0].properties[0].0, "count");
        assert_eq!(
            result.elements[0].properties[0].1,
            PropertyValue::Integer(100)
        );
    }

    #[test]
    fn test_parse_pipeline_chain() {
        let result = parse_pipeline("nullsource ! passthrough ! nullsink").unwrap();
        assert_eq!(result.elements.len(), 3);
        assert_eq!(result.elements[0].name, "nullsource");
        assert_eq!(result.elements[1].name, "passthrough");
        assert_eq!(result.elements[2].name, "nullsink");
    }

    #[test]
    fn test_parse_with_properties() {
        let result =
            parse_pipeline("filesrc location=/path/to/file ! filesink location=/output").unwrap();
        assert_eq!(result.elements.len(), 2);
        assert_eq!(result.elements[0].name, "filesrc");
        assert_eq!(
            result.elements[0].properties[0],
            (
                "location".to_string(),
                PropertyValue::String("/path/to/file".to_string())
            )
        );
        assert_eq!(result.elements[1].name, "filesink");
        assert_eq!(
            result.elements[1].properties[0],
            (
                "location".to_string(),
                PropertyValue::String("/output".to_string())
            )
        );
    }

    #[test]
    fn test_parse_quoted_string() {
        let result = parse_pipeline(r#"filesrc location="/path with spaces/file.bin""#).unwrap();
        assert_eq!(
            result.elements[0].properties[0].1,
            PropertyValue::String("/path with spaces/file.bin".to_string())
        );
    }

    #[test]
    fn test_parse_single_quoted_string() {
        let result = parse_pipeline("filesrc location='/path/to/file'").unwrap();
        assert_eq!(
            result.elements[0].properties[0].1,
            PropertyValue::String("/path/to/file".to_string())
        );
    }

    #[test]
    fn test_parse_boolean_property() {
        let result = parse_pipeline("element enabled=true disabled=false").unwrap();
        assert_eq!(
            result.elements[0].properties[0],
            ("enabled".to_string(), PropertyValue::Bool(true))
        );
        assert_eq!(
            result.elements[0].properties[1],
            ("disabled".to_string(), PropertyValue::Bool(false))
        );
    }

    #[test]
    fn test_parse_float_property() {
        let result = parse_pipeline("element rate=1.5").unwrap();
        assert_eq!(
            result.elements[0].properties[0],
            ("rate".to_string(), PropertyValue::Float(1.5))
        );
    }

    #[test]
    fn test_parse_negative_integer() {
        let result = parse_pipeline("element offset=-100").unwrap();
        assert_eq!(
            result.elements[0].properties[0],
            ("offset".to_string(), PropertyValue::Integer(-100))
        );
    }

    #[test]
    fn test_parse_multiple_properties() {
        let result = parse_pipeline("nullsource count=50 buffer-size=1024").unwrap();
        assert_eq!(result.elements[0].properties.len(), 2);
        assert_eq!(result.elements[0].properties[0].0, "count");
        assert_eq!(result.elements[0].properties[1].0, "buffer-size");
    }

    #[test]
    fn test_parse_no_spaces() {
        let result = parse_pipeline("a!b!c").unwrap();
        assert_eq!(result.elements.len(), 3);
        assert_eq!(result.elements[0].name, "a");
        assert_eq!(result.elements[1].name, "b");
        assert_eq!(result.elements[2].name, "c");
    }

    #[test]
    fn test_parse_extra_spaces() {
        let result = parse_pipeline("  a   !   b   !   c  ").unwrap();
        assert_eq!(result.elements.len(), 3);
    }

    #[test]
    fn test_parse_underscore_in_name() {
        let result = parse_pipeline("null_source ! null_sink").unwrap();
        assert_eq!(result.elements[0].name, "null_source");
        assert_eq!(result.elements[1].name, "null_sink");
    }

    #[test]
    fn test_property_value_conversions() {
        let int_val = PropertyValue::Integer(42);
        assert_eq!(int_val.as_i64(), Some(42));
        assert_eq!(int_val.as_u64(), Some(42));
        assert_eq!(int_val.as_f64(), Some(42.0));
        assert_eq!(int_val.as_string(), "42");

        let float_val = PropertyValue::Float(3.14);
        assert_eq!(float_val.as_f64(), Some(3.14));
        assert_eq!(float_val.as_i64(), None);

        let bool_val = PropertyValue::Bool(true);
        assert_eq!(bool_val.as_bool(), Some(true));

        let str_val = PropertyValue::String("100".to_string());
        assert_eq!(str_val.as_i64(), Some(100));
    }

    #[test]
    fn test_parse_empty_fails() {
        let result = parse_pipeline("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_just_link_fails() {
        let result = parse_pipeline("!");
        assert!(result.is_err());
    }
}
