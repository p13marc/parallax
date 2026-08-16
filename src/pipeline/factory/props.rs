//! Typed, consumption-tracking property access for element constructors.
//!
//! Every getter records the key it looked at; [`Props::finish`] then rejects
//! any property the constructor never consumed, so a typo like `fps=30` on
//! `videotestsrc` (which spells it `framerate`) is a hard error instead of a
//! silent no-op. The `name` key is pre-consumed — `Pipeline::parse` uses it
//! for node naming, not construction.

use crate::error::{Error, Result};
use crate::pipeline::parser::PropertyValue;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// A view over one element's parsed properties.
pub struct Props<'a> {
    element: &'a str,
    map: &'a HashMap<String, PropertyValue>,
    consumed: RefCell<HashSet<String>>,
}

impl<'a> Props<'a> {
    /// Wrap a property map for the named element.
    pub fn new(element: &'a str, map: &'a HashMap<String, PropertyValue>) -> Self {
        let mut consumed = HashSet::new();
        consumed.insert("name".to_string());
        Self {
            element,
            map,
            consumed: RefCell::new(consumed),
        }
    }

    fn raw(&self, key: &str) -> Option<&'a PropertyValue> {
        self.consumed.borrow_mut().insert(key.to_string());
        self.map.get(key)
    }

    fn type_err(&self, key: &str, expected: &str) -> Error {
        Error::Parse(format!(
            "property '{key}' on element '{}' expects {expected}",
            self.element
        ))
    }

    fn missing(&self, key: &str) -> Error {
        Error::Parse(format!(
            "element '{}' requires property '{key}'",
            self.element
        ))
    }

    /// String value (any property converts).
    pub fn get_str(&self, key: &str) -> Option<String> {
        self.raw(key).map(|v| v.as_string())
    }

    /// Required string value.
    pub fn req_str(&self, key: &str) -> Result<String> {
        self.get_str(key).ok_or_else(|| self.missing(key))
    }

    /// Unsigned integer; errors if present but not an unsigned integer.
    pub fn get_u64(&self, key: &str) -> Result<Option<u64>> {
        match self.raw(key) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .map(Some)
                .ok_or_else(|| self.type_err(key, "an unsigned integer")),
        }
    }

    /// Required unsigned integer.
    pub fn req_u64(&self, key: &str) -> Result<u64> {
        self.get_u64(key)?.ok_or_else(|| self.missing(key))
    }

    /// Signed integer; errors if present but not an integer.
    pub fn get_i64(&self, key: &str) -> Result<Option<i64>> {
        match self.raw(key) {
            None => Ok(None),
            Some(v) => v
                .as_i64()
                .map(Some)
                .ok_or_else(|| self.type_err(key, "an integer")),
        }
    }

    /// Required signed integer.
    pub fn req_i64(&self, key: &str) -> Result<i64> {
        self.get_i64(key)?.ok_or_else(|| self.missing(key))
    }

    /// `u32` convenience (range-checked).
    pub fn get_u32(&self, key: &str) -> Result<Option<u32>> {
        match self.get_u64(key)? {
            None => Ok(None),
            Some(v) => u32::try_from(v)
                .map(Some)
                .map_err(|_| self.type_err(key, "an unsigned 32-bit integer")),
        }
    }

    /// Required `u32`.
    pub fn req_u32(&self, key: &str) -> Result<u32> {
        self.get_u32(key)?.ok_or_else(|| self.missing(key))
    }

    /// `u16` convenience (range-checked).
    pub fn get_u16(&self, key: &str) -> Result<Option<u16>> {
        match self.get_u64(key)? {
            None => Ok(None),
            Some(v) => u16::try_from(v)
                .map(Some)
                .map_err(|_| self.type_err(key, "an unsigned 16-bit integer")),
        }
    }

    /// Required `u16`.
    pub fn req_u16(&self, key: &str) -> Result<u16> {
        self.get_u16(key)?.ok_or_else(|| self.missing(key))
    }

    /// `u8` convenience (range-checked).
    pub fn get_u8(&self, key: &str) -> Result<Option<u8>> {
        match self.get_u64(key)? {
            None => Ok(None),
            Some(v) => u8::try_from(v)
                .map(Some)
                .map_err(|_| self.type_err(key, "an unsigned 8-bit integer")),
        }
    }

    /// `usize` convenience.
    pub fn get_usize(&self, key: &str) -> Result<Option<usize>> {
        Ok(self.get_u64(key)?.map(|v| v as usize))
    }

    /// Required `usize`.
    pub fn req_usize(&self, key: &str) -> Result<usize> {
        self.get_usize(key)?.ok_or_else(|| self.missing(key))
    }

    /// Float; errors if present but not numeric.
    pub fn get_f64(&self, key: &str) -> Result<Option<f64>> {
        match self.raw(key) {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| self.type_err(key, "a number")),
        }
    }

    /// Required float.
    pub fn req_f64(&self, key: &str) -> Result<f64> {
        self.get_f64(key)?.ok_or_else(|| self.missing(key))
    }

    /// `f32` convenience.
    pub fn get_f32(&self, key: &str) -> Result<Option<f32>> {
        Ok(self.get_f64(key)?.map(|v| v as f32))
    }

    /// Boolean; errors if present but not a boolean.
    pub fn get_bool(&self, key: &str) -> Result<Option<bool>> {
        match self.raw(key) {
            None => Ok(None),
            Some(v) => v
                .as_bool()
                .map(Some)
                .ok_or_else(|| self.type_err(key, "a boolean")),
        }
    }

    /// Millisecond duration from an unsigned integer property.
    pub fn get_ms(&self, key: &str) -> Result<Option<Duration>> {
        Ok(self.get_u64(key)?.map(Duration::from_millis))
    }

    /// Required millisecond duration.
    pub fn req_ms(&self, key: &str) -> Result<Duration> {
        self.get_ms(key)?.ok_or_else(|| self.missing(key))
    }

    /// Enum-by-name; errors list the accepted values.
    pub fn get_enum<T: Copy>(&self, key: &str, table: &[(&str, T)]) -> Result<Option<T>> {
        let Some(s) = self.get_str(key) else {
            return Ok(None);
        };
        for (n, v) in table {
            if s.eq_ignore_ascii_case(n) {
                return Ok(Some(*v));
            }
        }
        let accepted: Vec<&str> = table.iter().map(|(n, _)| *n).collect();
        Err(Error::Parse(format!(
            "property '{key}' on element '{}': unknown value '{s}' (accepted: {})",
            self.element,
            accepted.join(", ")
        )))
    }

    /// Required enum-by-name.
    pub fn req_enum<T: Copy>(&self, key: &str, table: &[(&str, T)]) -> Result<T> {
        self.get_enum(key, table)?.ok_or_else(|| self.missing(key))
    }

    /// `width`+`height` pair: both or neither; one alone is an error.
    pub fn get_size(&self) -> Result<Option<(u32, u32)>> {
        match (self.get_u32("width")?, self.get_u32("height")?) {
            (Some(w), Some(h)) => Ok(Some((w, h))),
            (None, None) => Ok(None),
            _ => Err(Error::Parse(format!(
                "element '{}': 'width' and 'height' must be given together",
                self.element
            ))),
        }
    }

    /// Framerate as integer fps or a `"num/den"` fraction.
    pub fn get_framerate(&self, key: &str) -> Result<Option<(u32, u32)>> {
        let Some(v) = self.raw(key) else {
            return Ok(None);
        };
        if let Some(fps) = v.as_u64() {
            let fps = u32::try_from(fps).map_err(|_| self.type_err(key, "a frame rate"))?;
            return Ok(Some((fps, 1)));
        }
        let s = v.as_string();
        if let Some((num, den)) = s.split_once('/')
            && let (Ok(n), Ok(d)) = (num.trim().parse(), den.trim().parse())
        {
            return Ok(Some((n, d)));
        }
        Err(self.type_err(key, "a frame rate (integer fps or \"num/den\")"))
    }

    /// Error on any property the constructor never consumed.
    pub fn finish(&self) -> Result<()> {
        let consumed = self.consumed.borrow();
        let mut unknown: Vec<&str> = self
            .map
            .keys()
            .filter(|k| !consumed.contains(*k))
            .map(String::as_str)
            .collect();
        if unknown.is_empty() {
            return Ok(());
        }
        unknown.sort_unstable();
        Err(Error::Parse(format!(
            "unknown propert{} {} on element '{}'",
            if unknown.len() == 1 { "y" } else { "ies" },
            unknown
                .iter()
                .map(|k| format!("'{k}'"))
                .collect::<Vec<_>>()
                .join(", "),
            self.element
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, PropertyValue)]) -> HashMap<String, PropertyValue> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn unknown_property_is_rejected() {
        let m = map(&[("fps", PropertyValue::Integer(30))]);
        let p = Props::new("videotestsrc", &m);
        assert!(p.finish().is_err(), "unconsumed key must fail finish()");
        let err = p.finish().unwrap_err().to_string();
        assert!(err.contains("fps") && err.contains("videotestsrc"), "{err}");
    }

    #[test]
    fn consumed_properties_pass_finish() {
        let m = map(&[("count", PropertyValue::Integer(5))]);
        let p = Props::new("nullsource", &m);
        assert_eq!(p.get_u64("count").unwrap(), Some(5));
        p.finish().unwrap();
    }

    #[test]
    fn name_is_preconsumed() {
        let m = map(&[("name", PropertyValue::String("x".into()))]);
        let p = Props::new("nullsink", &m);
        p.finish().unwrap();
    }

    #[test]
    fn type_mismatch_errors() {
        let m = map(&[("count", PropertyValue::String("many".into()))]);
        let p = Props::new("nullsource", &m);
        assert!(p.get_u64("count").is_err());
    }

    #[test]
    fn required_missing_errors() {
        let m = map(&[]);
        let p = Props::new("filesrc", &m);
        let err = p.req_str("location").unwrap_err().to_string();
        assert!(err.contains("location") && err.contains("filesrc"), "{err}");
    }

    #[test]
    fn enum_lookup_and_error() {
        let m = map(&[("mode", PropertyValue::String("Fast".into()))]);
        let p = Props::new("x", &m);
        let v = p.get_enum("mode", &[("fast", 1), ("slow", 2)]).unwrap();
        assert_eq!(v, Some(1));

        let m = map(&[("mode", PropertyValue::String("warp".into()))]);
        let p = Props::new("x", &m);
        let err = p
            .get_enum("mode", &[("fast", 1), ("slow", 2)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("warp") && err.contains("fast, slow"), "{err}");
    }

    #[test]
    fn size_pair_requires_both() {
        let m = map(&[("width", PropertyValue::Integer(640))]);
        let p = Props::new("x", &m);
        assert!(p.get_size().is_err());

        let m = map(&[
            ("width", PropertyValue::Integer(640)),
            ("height", PropertyValue::Integer(480)),
        ]);
        let p = Props::new("x", &m);
        assert_eq!(p.get_size().unwrap(), Some((640, 480)));
    }

    #[test]
    fn framerate_int_and_fraction() {
        let m = map(&[("framerate", PropertyValue::Integer(30))]);
        let p = Props::new("x", &m);
        assert_eq!(p.get_framerate("framerate").unwrap(), Some((30, 1)));

        let m = map(&[("framerate", PropertyValue::String("30000/1001".into()))]);
        let p = Props::new("x", &m);
        assert_eq!(p.get_framerate("framerate").unwrap(), Some((30000, 1001)));

        let m = map(&[("framerate", PropertyValue::String("fast".into()))]);
        let p = Props::new("x", &m);
        assert!(p.get_framerate("framerate").is_err());
    }
}
