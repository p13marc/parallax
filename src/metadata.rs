//! Buffer metadata types.

use rkyv::{Archive, Deserialize, Serialize};
use std::time::Duration;

/// Flags indicating buffer properties.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub struct BufferFlags {
    /// Buffer marks end of stream.
    pub eos: bool,
    /// Buffer contains a sync point (keyframe equivalent).
    pub sync_point: bool,
    /// Buffer is corrupted or incomplete.
    pub corrupted: bool,
    /// Buffer should not be displayed/processed (e.g., decode-only).
    pub decode_only: bool,
    /// Buffer was generated due to a timeout (fallback/heartbeat).
    pub timeout: bool,
    /// Buffer is a gap/discontinuity marker.
    pub gap: bool,
}

impl BufferFlags {
    /// Set the timeout flag.
    pub fn set_timeout(&mut self, value: bool) {
        self.timeout = value;
    }

    /// Check if timeout flag is set.
    pub fn is_timeout(&self) -> bool {
        self.timeout
    }

    /// Set the gap flag.
    pub fn set_gap(&mut self, value: bool) {
        self.gap = value;
    }

    /// Check if gap flag is set.
    pub fn is_gap(&self) -> bool {
        self.gap
    }
}

/// A key-value pair for extra metadata.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct ExtraField {
    /// Field name.
    pub key: String,
    /// Field value.
    pub value: MetadataValue,
}

/// Possible values for extra metadata fields.
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub enum MetadataValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Floating-point value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// Raw bytes.
    Bytes(Vec<u8>),
}

/// Metadata associated with a buffer.
///
/// Contains timing information, sequence numbers, flags, and extensible
/// key-value fields for domain-specific data.
#[derive(Debug, Clone, Default, Archive, Serialize, Deserialize)]
pub struct Metadata {
    /// Presentation timestamp (when this buffer should be processed/displayed).
    pub pts: Option<Duration>,

    /// Decode timestamp (when this buffer should be decoded).
    pub dts: Option<Duration>,

    /// Duration of this buffer's content.
    pub duration: Option<Duration>,

    /// Monotonic sequence number within a stream.
    pub sequence: u64,

    /// Stream identifier for demultiplexing.
    pub stream_id: Option<u64>,

    /// Byte offset in the original source.
    pub offset: Option<u64>,

    /// End byte offset in the original source.
    pub offset_end: Option<u64>,

    /// Buffer flags.
    pub flags: BufferFlags,

    /// Extra key-value metadata fields.
    /// Uses Vec for rkyv compatibility. For most buffers this is empty or small.
    pub extra: Vec<ExtraField>,
}

impl Metadata {
    /// Create new metadata with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create metadata with a sequence number.
    pub fn with_sequence(sequence: u64) -> Self {
        Self {
            sequence,
            ..Default::default()
        }
    }

    /// Set the presentation timestamp.
    pub fn with_pts(mut self, pts: Duration) -> Self {
        self.pts = Some(pts);
        self
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Mark as end-of-stream.
    pub fn with_eos(mut self) -> Self {
        self.flags.eos = true;
        self
    }

    /// Add an extra field.
    pub fn with_extra(mut self, key: impl Into<String>, value: MetadataValue) -> Self {
        self.extra.push(ExtraField {
            key: key.into(),
            value,
        });
        self
    }

    /// Get an extra field by key.
    pub fn get_extra(&self, key: &str) -> Option<&MetadataValue> {
        self.extra.iter().find(|f| f.key == key).map(|f| &f.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_builder() {
        let meta = Metadata::with_sequence(42)
            .with_pts(Duration::from_millis(100))
            .with_duration(Duration::from_millis(33))
            .with_extra("source", MetadataValue::String("camera1".into()));

        assert_eq!(meta.sequence, 42);
        assert_eq!(meta.pts, Some(Duration::from_millis(100)));
        assert_eq!(meta.duration, Some(Duration::from_millis(33)));
        assert_eq!(
            meta.get_extra("source"),
            Some(&MetadataValue::String("camera1".into()))
        );
    }

    #[test]
    fn test_multiple_extra_fields() {
        let meta = Metadata::new()
            .with_extra("a", MetadataValue::Int(1))
            .with_extra("b", MetadataValue::Int(2))
            .with_extra("c", MetadataValue::Int(3))
            .with_extra("d", MetadataValue::Int(4));

        assert_eq!(meta.extra.len(), 4);
        assert_eq!(meta.get_extra("c"), Some(&MetadataValue::Int(3)));
    }
}
