//! File-based source and sink elements.

use crate::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use crate::error::Result;
use crate::event::{Event, EventResult, SeekEvent, SeekType, SegmentFormat};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::pipeline::seek::{DurationQuery, PositionQuery};
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

/// A source element that reads from a file.
///
/// Reads the file in chunks and produces buffers until EOF.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::FileSrc;
/// use parallax::element::{Source, ProduceContext, ProduceResult};
/// use parallax::memory::SharedArena;
///
/// let mut src = FileSrc::open("input.bin")?;
/// let arena = SharedArena::new(64 * 1024, 4)?;
///
/// loop {
///     let slot = arena.acquire().unwrap();
///     let mut ctx = ProduceContext::new(slot);
///     match src.produce(&mut ctx)? {
///         ProduceResult::Produced(n) => {
///             let buffer = ctx.finalize(n);
///             println!("Read {} bytes", buffer.len());
///         }
///         ProduceResult::Eos => break,
///         _ => {}
///     }
/// }
/// ```
pub struct FileSrc {
    name: String,
    path: PathBuf,
    file: Option<File>,
    chunk_size: usize,
    sequence: u64,
    bytes_read: u64,
    output: OutputArena,
    /// Cached file size (set on first open).
    file_size: Option<u64>,
}

impl FileSrc {
    /// Default chunk size (64 KB).
    pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

    /// Create a new FileSrc that will read from the given path.
    ///
    /// The file is not opened until the first call to `produce()`.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref().to_path_buf();
        let name = format!("filesrc:{}", path.display());
        Self {
            name,
            path,
            file: None,
            chunk_size: Self::DEFAULT_CHUNK_SIZE,
            sequence: 0,
            bytes_read: 0,
            output: OutputArena::new(defaults::SOURCE_SLOT_COUNT),
            file_size: None,
        }
    }

    /// Open the file immediately and return a FileSrc.
    ///
    /// Returns an error if the file cannot be opened.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_size = file.metadata().ok().map(|m| m.len());
        let name = format!("filesrc:{}", path.display());
        Ok(Self {
            name,
            path,
            file: Some(file),
            chunk_size: Self::DEFAULT_CHUNK_SIZE,
            sequence: 0,
            bytes_read: 0,
            output: OutputArena::new(defaults::SOURCE_SLOT_COUNT),
            file_size,
        })
    }

    /// Set the chunk size for reading.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Get the path being read.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the total bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Ensure the file is open.
    fn ensure_open(&mut self) -> Result<&mut File> {
        if self.file.is_none() {
            let file = File::open(&self.path)?;
            self.file_size = file.metadata().ok().map(|m| m.len());
            self.file = Some(file);
        }
        Ok(self.file.as_mut().unwrap())
    }
}

impl Source for FileSrc {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        // Copy chunk_size before borrowing self mutably
        let chunk_size = self.chunk_size;

        // Ensure file is open first
        self.ensure_open()?;

        if ctx.has_buffer() {
            // Use the provided buffer
            let output = ctx.output();
            let read_len = output.len().min(chunk_size);
            let file = self.file.as_mut().unwrap();
            let bytes_read = file.read(&mut output[..read_len])?;

            if bytes_read == 0 {
                return Ok(ProduceResult::Eos);
            }

            self.bytes_read += bytes_read as u64;
            ctx.set_sequence(self.sequence);
            self.sequence += 1;

            Ok(ProduceResult::Produced(bytes_read))
        } else {
            // Fallback: create our own buffer using SharedArena
            use crate::buffer::{Buffer, MemoryHandle};
            use crate::metadata::Metadata;

            let Some(mut slot) = self.output.try_acquire(chunk_size, "filesrc")? else {
                return Ok(ProduceResult::WouldBlock);
            };

            let file = self.file.as_mut().unwrap();
            let bytes_read = file.read(&mut slot.data_mut()[..chunk_size])?;

            if bytes_read == 0 {
                return Ok(ProduceResult::Eos);
            }

            self.bytes_read += bytes_read as u64;
            let metadata = Metadata::from_sequence(self.sequence);
            self.sequence += 1;

            // Create buffer with only the bytes we read
            let handle = MemoryHandle::with_len(slot, bytes_read);
            let buffer = Buffer::new(handle, metadata);
            Ok(ProduceResult::OwnBuffer(buffer))
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(self.chunk_size)
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        match event {
            Event::Seek(seek) => self.handle_seek(seek),
            _ => EventResult::NotHandled,
        }
    }

    fn query_position(&self) -> Option<PositionQuery> {
        Some(PositionQuery {
            format: SegmentFormat::Bytes,
            position: Some(self.bytes_read),
        })
    }

    fn query_duration(&self) -> Option<DurationQuery> {
        Some(DurationQuery {
            format: SegmentFormat::Bytes,
            duration: self.file_size,
        })
    }
}

impl FileSrc {
    /// Handle a seek event by repositioning the file.
    fn handle_seek(&mut self, seek: &SeekEvent) -> EventResult {
        if seek.format != SegmentFormat::Bytes {
            return EventResult::NotHandled;
        }

        let file = match &mut self.file {
            Some(f) => f,
            None => return EventResult::Error,
        };

        let seek_pos = match seek.start.seek_type {
            SeekType::Set => {
                if seek.start.position < 0 {
                    return EventResult::Error;
                }
                std::io::SeekFrom::Start(seek.start.position as u64)
            }
            SeekType::Current => std::io::SeekFrom::Current(seek.start.position),
            SeekType::End => std::io::SeekFrom::End(seek.start.position),
            SeekType::None => return EventResult::handled(), // No position change
        };

        match file.seek(seek_pos) {
            Ok(new_pos) => {
                self.bytes_read = new_pos;
                // The absolute resulting offset — for Current/End seeks this
                // is the only party that knows it, and it feeds the honest
                // Segment/SeekDone (#162).
                EventResult::handled_at(new_pos as i64)
            }
            Err(_) => EventResult::Error,
        }
    }
}

/// A sink element that writes to a file.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::FileSink;
/// use parallax::element::{Sink, ConsumeContext};
///
/// let mut sink = FileSink::create("output.bin")?;
///
/// // Write buffers using ConsumeContext
/// let ctx = ConsumeContext::new(&buffer);
/// sink.consume(&ctx)?;
///
/// // Flush and close
/// sink.flush()?;
/// ```
pub struct FileSink {
    name: String,
    path: PathBuf,
    file: Option<File>,
    bytes_written: u64,
    buffers_written: u64,
}

impl FileSink {
    /// Create a new FileSink that will write to the given path.
    ///
    /// The file is not created until the first call to `consume()`.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref().to_path_buf();
        let name = format!("filesink:{}", path.display());
        Self {
            name,
            path,
            file: None,
            bytes_written: 0,
            buffers_written: 0,
        }
    }

    /// Create the file immediately and return a FileSink.
    ///
    /// Returns an error if the file cannot be created.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path)?;
        let name = format!("filesink:{}", path.display());
        Ok(Self {
            name,
            path,
            file: Some(file),
            bytes_written: 0,
            buffers_written: 0,
        })
    }

    /// Get the path being written to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the total bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Get the number of buffers written.
    pub fn buffers_written(&self) -> u64 {
        self.buffers_written
    }

    /// Flush the file.
    pub fn flush(&mut self) -> Result<()> {
        if let Some(file) = &mut self.file {
            file.flush()?;
        }
        Ok(())
    }

    /// Ensure the file is open.
    fn ensure_open(&mut self) -> Result<&mut File> {
        if self.file.is_none() {
            self.file = Some(File::create(&self.path)?);
        }
        Ok(self.file.as_mut().unwrap())
    }
}

impl Sink for FileSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let file = self.ensure_open()?;
        let data = ctx.input();
        file.write_all(data)?;
        self.bytes_written += data.len() as u64;
        self.buffers_written += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for FileSink {
    fn drop(&mut self) {
        // Best effort flush on drop
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{Buffer, MemoryHandle};
    use crate::element::{ConsumeContext, ProduceContext, ProduceResult};
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::io::Write;
    use std::sync::OnceLock;
    use tempfile::NamedTempFile;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(1024, 64).unwrap())
    }

    // Helper to produce a buffer using SharedArena
    fn produce_buffer(src: &mut FileSrc) -> Option<Vec<u8>> {
        let slot = test_arena().acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        match src.produce(&mut ctx).unwrap() {
            ProduceResult::Produced(n) => {
                let buffer = ctx.finalize(n);
                Some(buffer.as_bytes().to_vec())
            }
            ProduceResult::Eos => None,
            _ => None,
        }
    }

    // Helper to produce and get the buffer with metadata
    fn produce_buffer_with_meta(src: &mut FileSrc) -> Option<Buffer> {
        let slot = test_arena().acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        match src.produce(&mut ctx).unwrap() {
            ProduceResult::Produced(n) => Some(ctx.finalize(n)),
            ProduceResult::Eos => None,
            _ => None,
        }
    }

    #[test]
    fn test_filesrc_reads_file() {
        // Create a temp file with known content
        let mut temp = NamedTempFile::new().unwrap();
        let content = b"Hello, Parallax!";
        temp.write_all(content).unwrap();
        temp.flush().unwrap();

        // Read it back
        let mut src = FileSrc::open(temp.path()).unwrap();
        let buffer = produce_buffer_with_meta(&mut src).unwrap();

        assert_eq!(buffer.as_bytes(), content);
        assert_eq!(buffer.metadata().sequence, 0);
        assert_eq!(src.bytes_read(), content.len() as u64);

        // Next read should return None (EOF)
        let buffer = produce_buffer(&mut src);
        assert!(buffer.is_none());
    }

    #[test]
    fn test_filesrc_chunked_reading() {
        // Create a temp file larger than chunk size
        let mut temp = NamedTempFile::new().unwrap();
        let content = vec![0xABu8; 1000];
        temp.write_all(&content).unwrap();
        temp.flush().unwrap();

        // Read with small chunks
        let mut src = FileSrc::open(temp.path()).unwrap().with_chunk_size(100);

        let mut total_read = 0;
        let mut chunk_count = 0;

        while let Some(bytes) = produce_buffer(&mut src) {
            total_read += bytes.len();
            chunk_count += 1;
            assert!(bytes.len() <= 100);
        }

        assert_eq!(total_read, 1000);
        assert_eq!(chunk_count, 10);
    }

    #[test]
    fn test_filesink_writes_file() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        {
            let mut sink = FileSink::create(&path).unwrap();

            // Create a buffer with test data using SharedArena
            let arena = SharedArena::new(16, 4).unwrap();
            let mut slot = arena.acquire().unwrap();
            slot.data_mut()[..13].copy_from_slice(b"Hello, World!");
            let handle = MemoryHandle::with_len(slot, 13);
            let buffer = Buffer::new(handle, Metadata::from_sequence(0));

            // Use ConsumeContext
            let ctx = ConsumeContext::new(&buffer);
            sink.consume(&ctx).unwrap();
            assert_eq!(sink.bytes_written(), 13);
            assert_eq!(sink.buffers_written(), 1);
        }

        // Verify the file content
        let content = std::fs::read(&path).unwrap();
        assert_eq!(&content, b"Hello, World!");
    }

    #[test]
    fn test_filesrc_lazy_open() {
        // FileSrc::new doesn't open the file immediately
        let src = FileSrc::new("/nonexistent/path/file.bin");
        assert!(src.file.is_none());
    }

    #[test]
    fn test_filesink_lazy_create() {
        // FileSink::new doesn't create the file immediately
        let sink = FileSink::new("/nonexistent/path/file.bin");
        assert!(sink.file.is_none());
    }

    #[test]
    fn test_roundtrip() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        // Write data
        let original_data = b"Parallax streaming pipeline engine";
        {
            let mut sink = FileSink::create(&path).unwrap();
            let mut slot = test_arena().acquire().unwrap();
            slot.data_mut()[..original_data.len()].copy_from_slice(original_data);
            let handle = MemoryHandle::with_len(slot, original_data.len());
            let buffer = Buffer::new(handle, Metadata::from_sequence(0));

            // Use ConsumeContext
            let ctx = ConsumeContext::new(&buffer);
            sink.consume(&ctx).unwrap();
        }

        // Read it back
        let mut src = FileSrc::open(&path).unwrap();
        let buffer = produce_buffer_with_meta(&mut src).unwrap();
        assert_eq!(buffer.as_bytes(), original_data);
    }
}
