//! File-based source and sink elements.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Sink, Source};
use crate::error::Result;
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A source element that reads from a file.
///
/// Reads the file in chunks and produces buffers until EOF.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::FileSrc;
/// use parallax::element::Source;
///
/// let mut src = FileSrc::open("input.bin")?;
///
/// while let Some(buffer) = src.produce()? {
///     println!("Read {} bytes", buffer.len());
/// }
/// ```
pub struct FileSrc {
    name: String,
    path: PathBuf,
    file: Option<File>,
    chunk_size: usize,
    sequence: u64,
    bytes_read: u64,
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
        }
    }

    /// Open the file immediately and return a FileSrc.
    ///
    /// Returns an error if the file cannot be opened.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let name = format!("filesrc:{}", path.display());
        Ok(Self {
            name,
            path,
            file: Some(file),
            chunk_size: Self::DEFAULT_CHUNK_SIZE,
            sequence: 0,
            bytes_read: 0,
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
            self.file = Some(File::open(&self.path)?);
        }
        Ok(self.file.as_mut().unwrap())
    }
}

impl Source for FileSrc {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        // Copy chunk_size before borrowing self mutably
        let chunk_size = self.chunk_size;
        let file = self.ensure_open()?;

        // Allocate a buffer for reading
        let segment = Arc::new(HeapSegment::new(chunk_size)?);

        // Read into the segment
        let ptr = segment.as_mut_ptr().unwrap();
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, chunk_size) };

        let bytes_read = file.read(slice)?;

        if bytes_read == 0 {
            // EOF
            return Ok(None);
        }

        self.bytes_read += bytes_read as u64;

        // Create a handle that only covers the bytes actually read
        let handle = MemoryHandle::new(segment, 0, bytes_read);
        let metadata = Metadata::with_sequence(self.sequence);
        self.sequence += 1;

        Ok(Some(Buffer::new(handle, metadata)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A sink element that writes to a file.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::FileSink;
/// use parallax::element::Sink;
///
/// let mut sink = FileSink::create("output.bin")?;
///
/// // Write buffers
/// sink.consume(buffer)?;
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
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let file = self.ensure_open()?;
        let data = buffer.as_bytes();
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
    use crate::memory::MemorySegment;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_filesrc_reads_file() {
        // Create a temp file with known content
        let mut temp = NamedTempFile::new().unwrap();
        let content = b"Hello, Parallax!";
        temp.write_all(content).unwrap();
        temp.flush().unwrap();

        // Read it back
        let mut src = FileSrc::open(temp.path()).unwrap();
        let buffer = src.produce().unwrap().unwrap();

        assert_eq!(buffer.as_bytes(), content);
        assert_eq!(buffer.metadata().sequence, 0);
        assert_eq!(src.bytes_read(), content.len() as u64);

        // Next read should return None (EOF)
        let buffer = src.produce().unwrap();
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

        while let Some(buffer) = src.produce().unwrap() {
            total_read += buffer.len();
            chunk_count += 1;
            assert!(buffer.len() <= 100);
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

            // Create a buffer with test data
            let segment = Arc::new(HeapSegment::new(16).unwrap());
            let ptr = segment.as_mut_ptr().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(b"Hello, World!".as_ptr(), ptr, 13);
            }
            let handle = MemoryHandle::new(segment, 0, 13);
            let buffer = Buffer::new(handle, Metadata::with_sequence(0));

            sink.consume(buffer).unwrap();
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
            let segment = Arc::new(HeapSegment::new(original_data.len()).unwrap());
            let ptr = segment.as_mut_ptr().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(original_data.as_ptr(), ptr, original_data.len());
            }
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::new(handle, Metadata::with_sequence(0));
            sink.consume(buffer).unwrap();
        }

        // Read it back
        let mut src = FileSrc::open(&path).unwrap();
        let buffer = src.produce().unwrap().unwrap();
        assert_eq!(buffer.as_bytes(), original_data);
    }
}
