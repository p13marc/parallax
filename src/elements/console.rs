//! Console sink element for debugging output.

use crate::buffer::Buffer;
use crate::element::Sink;
use crate::error::Result;
use std::io::{self, Write};

/// Output format for ConsoleSink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsoleFormat {
    /// Print buffer metadata only (sequence, timestamp, length).
    #[default]
    Metadata,
    /// Print buffer content as hexadecimal.
    Hex,
    /// Print buffer content as UTF-8 string (lossy).
    Text,
    /// Print both metadata and hex content.
    Full,
}

/// A sink that prints buffer information to the console.
///
/// Useful for debugging pipelines and inspecting buffer contents.
///
/// # Example
///
/// ```rust,no_run
/// use parallax::elements::{ConsoleSink, ConsoleFormat};
/// use parallax::element::Sink;
/// # use parallax::buffer::{Buffer, MemoryHandle};
/// # use parallax::memory::HeapSegment;
/// # use parallax::metadata::Metadata;
/// # use std::sync::Arc;
///
/// // Create a console sink
/// let mut sink = ConsoleSink::new();
///
/// // Or with custom format
/// let mut sink = ConsoleSink::with_format(ConsoleFormat::Hex);
///
/// // Or with a prefix
/// let mut sink = ConsoleSink::with_prefix("[DEBUG]");
/// ```
pub struct ConsoleSink {
    name: String,
    format: ConsoleFormat,
    prefix: Option<String>,
    count: u64,
    total_bytes: u64,
    max_hex_bytes: usize,
}

impl ConsoleSink {
    /// Create a new ConsoleSink with default settings.
    pub fn new() -> Self {
        Self {
            name: "consolesink".to_string(),
            format: ConsoleFormat::default(),
            prefix: None,
            count: 0,
            total_bytes: 0,
            max_hex_bytes: 64,
        }
    }

    /// Create a ConsoleSink with a specific output format.
    pub fn with_format(format: ConsoleFormat) -> Self {
        Self {
            format,
            ..Self::new()
        }
    }

    /// Create a ConsoleSink with a prefix for each line.
    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        Self {
            prefix: Some(prefix.into()),
            ..Self::new()
        }
    }

    /// Set the output format.
    pub fn format(mut self, format: ConsoleFormat) -> Self {
        self.format = format;
        self
    }

    /// Set a prefix for each output line.
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Set the maximum number of bytes to display in hex mode.
    pub fn max_hex_bytes(mut self, max: usize) -> Self {
        self.max_hex_bytes = max;
        self
    }

    /// Set a custom name for the element.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the number of buffers consumed.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get the total bytes consumed.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    fn format_buffer(&self, buffer: &Buffer) -> String {
        let prefix = self.prefix.as_deref().unwrap_or("");
        let metadata = buffer.metadata();

        match self.format {
            ConsoleFormat::Metadata => {
                format!(
                    "{}Buffer #{}: seq={}, len={}, pts={:?}, flags={:?}",
                    prefix,
                    self.count,
                    metadata.sequence,
                    buffer.len(),
                    metadata.pts,
                    metadata.flags
                )
            }
            ConsoleFormat::Hex => {
                let bytes = buffer.as_bytes();
                let display_len = bytes.len().min(self.max_hex_bytes);
                let hex: String = bytes[..display_len]
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join(" ");
                let truncated = if bytes.len() > self.max_hex_bytes {
                    format!("... ({} more bytes)", bytes.len() - self.max_hex_bytes)
                } else {
                    String::new()
                };
                format!("{}[{}]{}", prefix, hex, truncated)
            }
            ConsoleFormat::Text => {
                let bytes = buffer.as_bytes();
                let text = String::from_utf8_lossy(bytes);
                format!("{}{}", prefix, text)
            }
            ConsoleFormat::Full => {
                let bytes = buffer.as_bytes();
                let display_len = bytes.len().min(self.max_hex_bytes);
                let hex: String = bytes[..display_len]
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join(" ");
                let truncated = if bytes.len() > self.max_hex_bytes {
                    format!("... ({} more)", bytes.len() - self.max_hex_bytes)
                } else {
                    String::new()
                };
                format!(
                    "{}Buffer #{}: seq={}, len={}, pts={:?}\n{}  Data: [{}]{}",
                    prefix,
                    self.count,
                    metadata.sequence,
                    buffer.len(),
                    metadata.pts,
                    prefix,
                    hex,
                    truncated
                )
            }
        }
    }
}

impl Default for ConsoleSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for ConsoleSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        self.count += 1;
        self.total_bytes += buffer.len() as u64;

        let output = self.format_buffer(&buffer);

        // Use writeln to stdout with proper error handling
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        writeln!(handle, "{}", output).ok();

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    fn create_test_buffer(data: &[u8], sequence: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(data.len()).unwrap());
        // Copy data into segment
        unsafe {
            use crate::memory::MemorySegment;
            std::ptr::copy_nonoverlapping(data.as_ptr(), segment.as_mut_ptr().unwrap(), data.len());
        }
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::with_sequence(sequence))
    }

    #[test]
    fn test_console_sink_creation() {
        let sink = ConsoleSink::new();
        assert_eq!(sink.count(), 0);
        assert_eq!(sink.total_bytes(), 0);
    }

    #[test]
    fn test_console_sink_with_format() {
        let sink = ConsoleSink::with_format(ConsoleFormat::Hex);
        assert_eq!(sink.format, ConsoleFormat::Hex);
    }

    #[test]
    fn test_console_sink_with_prefix() {
        let sink = ConsoleSink::with_prefix("[TEST]");
        assert_eq!(sink.prefix, Some("[TEST]".to_string()));
    }

    #[test]
    fn test_console_sink_consume() {
        let mut sink = ConsoleSink::new();
        let buffer = create_test_buffer(b"hello", 0);

        sink.consume(buffer).unwrap();

        assert_eq!(sink.count(), 1);
        assert_eq!(sink.total_bytes(), 5);
    }

    #[test]
    fn test_console_sink_format_metadata() {
        let sink = ConsoleSink::with_format(ConsoleFormat::Metadata);
        let buffer = create_test_buffer(b"test", 42);

        let output = sink.format_buffer(&buffer);
        assert!(output.contains("seq=42"));
        assert!(output.contains("len=4"));
    }

    #[test]
    fn test_console_sink_format_hex() {
        let sink = ConsoleSink::with_format(ConsoleFormat::Hex);
        let buffer = create_test_buffer(&[0x00, 0x01, 0xff], 0);

        let output = sink.format_buffer(&buffer);
        assert!(output.contains("00 01 ff"));
    }

    #[test]
    fn test_console_sink_format_text() {
        let sink = ConsoleSink::with_format(ConsoleFormat::Text);
        let buffer = create_test_buffer(b"Hello, World!", 0);

        let output = sink.format_buffer(&buffer);
        assert!(output.contains("Hello, World!"));
    }

    #[test]
    fn test_console_sink_builder_pattern() {
        let sink = ConsoleSink::new()
            .format(ConsoleFormat::Full)
            .prefix("[DEBUG] ")
            .max_hex_bytes(32)
            .name("my_console");

        assert_eq!(sink.format, ConsoleFormat::Full);
        assert_eq!(sink.prefix, Some("[DEBUG] ".to_string()));
        assert_eq!(sink.max_hex_bytes, 32);
        assert_eq!(sink.name, "my_console");
    }
}
