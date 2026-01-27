//! AV1 software decoder using dav1d.
//!
//! dav1d is a fast, cross-platform AV1 decoder developed by VideoLAN.
//! It's used in Firefox, VLC, and many other applications.
//!
//! # System Dependencies
//!
//! Requires the dav1d library to be installed:
//!
//! - **Fedora/RHEL**: `sudo dnf install libdav1d-devel`
//! - **Debian/Ubuntu**: `sudo apt install libdav1d-dev`
//! - **Arch**: `sudo pacman -S dav1d`
//! - **macOS**: `brew install dav1d`
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::Dav1dDecoder;
//!
//! let decoder = Dav1dDecoder::new()?;
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::element::{Element, ExecutionHints};
use crate::error::{Error, Result};
use crate::memory::{HeapSegment, MemorySegment};
use std::sync::Arc;

use super::common::{PixelFormat, VideoFrame};

/// AV1 software decoder using dav1d.
///
/// # Input
///
/// Expects OBU (Open Bitstream Unit) formatted AV1 data.
///
/// # Output
///
/// Produces raw video frames in I420 or I420p10 format.
///
/// # Example
///
/// ```rust,ignore
/// let decoder = Dav1dDecoder::new()?;
/// pipeline.add_node("av1dec", DynAsyncElement::new_box(ElementAdapter::new(decoder)));
/// ```
pub struct Dav1dDecoder {
    decoder: dav1d::Decoder,
    frame_count: u64,
}

impl Dav1dDecoder {
    /// Create a new dav1d decoder with default settings.
    pub fn new() -> Result<Self> {
        let settings = dav1d::Settings::new();
        let decoder = dav1d::Decoder::with_settings(&settings)
            .map_err(|e| Error::Config(format!("Failed to create dav1d decoder: {:?}", e)))?;

        Ok(Self {
            decoder,
            frame_count: 0,
        })
    }

    /// Create a decoder with custom settings.
    pub fn with_settings(settings: &dav1d::Settings) -> Result<Self> {
        let decoder = dav1d::Decoder::with_settings(settings)
            .map_err(|e| Error::Config(format!("Failed to create dav1d decoder: {:?}", e)))?;

        Ok(Self {
            decoder,
            frame_count: 0,
        })
    }

    /// Get the number of frames decoded.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Decode a single frame from the input buffer.
    fn decode_frame(&mut self, input: &[u8]) -> Result<Option<VideoFrame>> {
        // Send data to decoder
        self.decoder
            .send_data(input.to_vec(), None, None, None)
            .map_err(|e| Error::InvalidSegment(format!("dav1d send_data failed: {:?}", e)))?;

        // Try to get decoded picture
        match self.decoder.get_picture() {
            Ok(picture) => {
                let frame = self.picture_to_frame(&picture)?;
                self.frame_count += 1;
                Ok(Some(frame))
            }
            Err(dav1d::Error::Again) => Ok(None), // Need more data
            Err(e) => Err(Error::InvalidSegment(format!(
                "dav1d decode failed: {:?}",
                e
            ))),
        }
    }

    /// Convert dav1d Picture to our VideoFrame.
    fn picture_to_frame(&self, picture: &dav1d::Picture) -> Result<VideoFrame> {
        let width = picture.width() as u32;
        let height = picture.height() as u32;
        let bit_depth = picture.bit_depth();

        let format = match (picture.pixel_layout(), bit_depth) {
            (dav1d::PixelLayout::I420, 8) => PixelFormat::I420,
            (dav1d::PixelLayout::I420, 10) => PixelFormat::I420p10,
            (dav1d::PixelLayout::I422, 8) => PixelFormat::I422,
            (dav1d::PixelLayout::I444, 8) => PixelFormat::I444,
            _ => {
                return Err(Error::InvalidSegment(format!(
                    "Unsupported pixel format: {:?} {}bit",
                    picture.pixel_layout(),
                    bit_depth
                )));
            }
        };

        let mut frame = VideoFrame::new(width, height, format);
        frame.pts = picture.timestamp().unwrap_or(0);

        // Copy plane data
        let plane_y = picture.plane(dav1d::PlanarImageComponent::Y);
        let plane_u = picture.plane(dav1d::PlanarImageComponent::U);
        let plane_v = picture.plane(dav1d::PlanarImageComponent::V);

        frame.stride_y = picture.stride(dav1d::PlanarImageComponent::Y) as usize;
        frame.stride_u = picture.stride(dav1d::PlanarImageComponent::U) as usize;
        frame.stride_v = picture.stride(dav1d::PlanarImageComponent::V) as usize;

        // Calculate sizes
        let y_size = frame.stride_y * height as usize;
        let uv_height = match format {
            PixelFormat::I422 | PixelFormat::I444 => height as usize,
            _ => height as usize / 2,
        };
        let u_size = frame.stride_u * uv_height;
        let v_size = frame.stride_v * uv_height;

        // Allocate and copy
        frame.data = vec![0u8; y_size + u_size + v_size];
        frame.data[..y_size].copy_from_slice(&plane_y[..y_size]);
        frame.data[y_size..y_size + u_size].copy_from_slice(&plane_u[..u_size]);
        frame.data[y_size + u_size..].copy_from_slice(&plane_v[..v_size]);

        Ok(frame)
    }
}

impl Element for Dav1dDecoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input = buffer.as_bytes();

        match self.decode_frame(input)? {
            Some(frame) => {
                // Create output buffer with frame data
                let segment = Arc::new(HeapSegment::new(frame.data.len())?);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        frame.data.as_ptr(),
                        segment.as_mut_ptr().unwrap(),
                        frame.data.len(),
                    );
                }

                let mut metadata = buffer.metadata().clone();
                // Store frame timing
                metadata.pts = ClockTime::from_nanos(frame.pts as u64);
                // Note: width/height/format info could be added via MediaFormat if needed

                Ok(Some(Buffer::new(
                    MemoryHandle::from_segment(segment),
                    metadata,
                )))
            }
            None => Ok(None), // Need more data, no output yet
        }
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::native() // Native code (FFI), might crash on bad input
    }
}
