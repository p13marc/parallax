//! AV1 software encoder using rav1e.
//!
//! rav1e is an AV1 encoder written in Rust, known for its safety and quality.
//! It is pure Rust and does not require any system dependencies.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{Rav1eEncoder, Rav1eConfig};
//!
//! let config = Rav1eConfig::default()
//!     .dimensions(1920, 1080)
//!     .speed(6);
//! let encoder = Rav1eEncoder::new(config)?;
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Element, ExecutionHints};
use crate::error::{Error, Result};
use crate::memory::SharedArena;

use super::common::{PixelFormat, VideoFrame};
use super::traits::VideoEncoder;

/// Configuration for the rav1e AV1 encoder.
#[derive(Clone, Debug)]
///
/// Note there is **no width or height**: geometry travels in-band, in each
/// buffer's [`Metadata`](crate::metadata::Metadata), and the encoder builds its
/// rav1e context from the first frame it sees (#38). A default resolution here
/// would be a claim about data that has not arrived yet.
pub struct Rav1eConfig {
    /// Speed preset (0-10, higher = faster but lower quality).
    pub speed: usize,
    /// Quantizer (0-255, lower = higher quality).
    pub quantizer: usize,
    /// Bitrate in bits per second (0 = constant quality mode).
    pub bitrate: usize,
    /// Frames per second numerator.
    pub timebase_num: u64,
    /// Frames per second denominator.
    pub timebase_den: u64,
    /// Pixel format.
    pub pixel_format: PixelFormat,
    /// Bit depth (8 or 10).
    pub bit_depth: usize,
}

impl Default for Rav1eConfig {
    fn default() -> Self {
        Self {
            speed: 6,
            quantizer: 100,
            bitrate: 0,
            timebase_num: 1,
            timebase_den: 30,
            pixel_format: PixelFormat::I420,
            bit_depth: 8,
        }
    }
}

impl Rav1eConfig {
    /// Set speed preset (0-10).
    pub fn speed(mut self, speed: usize) -> Self {
        self.speed = speed.min(10);
        self
    }

    /// Set quantizer (0-255).
    pub fn quantizer(mut self, quantizer: usize) -> Self {
        self.quantizer = quantizer.min(255);
        self
    }

    /// Set target bitrate in bits per second.
    pub fn bitrate(mut self, bitrate: usize) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set frame rate.
    pub fn framerate(mut self, fps: u64) -> Self {
        self.timebase_num = 1;
        self.timebase_den = fps;
        self
    }

    /// Set fractional frame rate.
    pub fn framerate_rational(mut self, num: u64, den: u64) -> Self {
        self.timebase_num = num;
        self.timebase_den = den;
        self
    }

    /// Set bit depth (8 or 10).
    pub fn bit_depth(mut self, depth: usize) -> Self {
        self.bit_depth = if depth >= 10 { 10 } else { 8 };
        self
    }
}

/// AV1 software encoder using rav1e.
///
/// rav1e is an AV1 encoder written in Rust, known for its safety and quality.
///
/// # Input
///
/// Expects raw video frames in I420 format.
///
/// # Output
///
/// Produces AV1 OBU bitstream packets.
///
/// # Example
///
/// ```rust,ignore
/// let config = Rav1eConfig::default()
///     .dimensions(1920, 1080)
///     .speed(6)
///     .bitrate(5_000_000);
///
/// let encoder = Rav1eEncoder::new(config)?;
/// pipeline.add_node("av1enc", DynAsyncElement::new_box(ElementAdapter::new(encoder)));
/// ```
pub struct Rav1eEncoder {
    /// Built on the first frame, from that frame's dimensions.
    ///
    /// rav1e fixes the resolution when the context is created, so a
    /// mid-stream resize means a new context — see [`Self::ensure_context`].
    context: Option<rav1e::Context<u8>>,
    /// The geometry `context` was built for.
    dims: Option<(usize, usize)>,
    config: Rav1eConfig,
    frame_count: u64,
    /// Arena for output buffer allocation.
    arena: Option<SharedArena>,
    /// Packets drained at EOS, returned one per Element::flush call.
    pending_flush: std::collections::VecDeque<Vec<u8>>,
    /// Whether the rav1e context has been flushed.
    flushed: bool,
}

impl Rav1eEncoder {
    /// Create a new rav1e encoder.
    ///
    /// The rav1e context itself is built from the first frame, because that is
    /// the first point at which the resolution is actually known.
    pub fn new(config: Rav1eConfig) -> Result<Self> {
        Ok(Self {
            context: None,
            dims: None,
            config,
            frame_count: 0,
            arena: None,
            pending_flush: std::collections::VecDeque::new(),
            flushed: false,
        })
    }

    fn build_config(config: &Rav1eConfig, width: usize, height: usize) -> Result<rav1e::Config> {
        let mut enc = rav1e::EncoderConfig::default();

        enc.width = width;
        enc.height = height;
        enc.speed_settings = rav1e::config::SpeedSettings::from_preset(config.speed as u8);
        enc.quantizer = config.quantizer;
        enc.bitrate = config.bitrate as i32;
        enc.time_base = rav1e::data::Rational::new(config.timebase_num, config.timebase_den);

        let cfg = rav1e::Config::new()
            .with_encoder_config(enc)
            .with_threads(0); // Auto-detect thread count

        Ok(cfg)
    }

    /// Ensure a context exists and matches `width`x`height`.
    ///
    /// On a resize, the old context is drained first so its lookahead is not
    /// silently lost — those packets join `pending_flush` and come out of
    /// `Element::flush`.
    fn ensure_context(&mut self, width: usize, height: usize) -> Result<()> {
        if self.dims == Some((width, height)) && self.context.is_some() {
            return Ok(());
        }

        if self.context.is_some() {
            tracing::debug!(
                "rav1e input resized {:?} -> {width}x{height}, rebuilding context",
                self.dims
            );
            let tail = self.flush_internal()?;
            self.pending_flush.extend(tail);
        }

        let context = Self::build_config(&self.config, width, height)?
            .new_context()
            .map_err(|e| Error::Config(format!("Failed to create rav1e context: {:?}", e)))?;
        self.context = Some(context);
        self.dims = Some((width, height));
        self.flushed = false;
        Ok(())
    }

    /// The context, or an error naming why there isn't one.
    fn context_mut(&mut self) -> Result<&mut rav1e::Context<u8>> {
        self.context
            .as_mut()
            .ok_or_else(|| Error::Element("Rav1eEncoder: no frame has been encoded yet".into()))
    }

    /// Get the configuration.
    pub fn config(&self) -> &Rav1eConfig {
        &self.config
    }

    /// Get the number of frames encoded.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Encode a frame from raw I420 data at the given geometry.
    fn encode_frame(
        &mut self,
        input: &[u8],
        width: usize,
        height: usize,
        _pts: u64,
    ) -> Result<Option<Vec<u8>>> {
        self.ensure_context(width, height)?;

        // Create rav1e frame
        let mut frame = self.context_mut()?.new_frame();

        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);

        if input.len() < y_size + 2 * uv_size {
            return Err(Error::InvalidSegment(format!(
                "Input buffer too small: {} < {}",
                input.len(),
                y_size + 2 * uv_size
            )));
        }

        // Copy Y plane
        for y in 0..height {
            let src_offset = y * width;
            let dst_row = &mut frame.planes[0].rows_iter_mut().nth(y).unwrap();
            dst_row[..width].copy_from_slice(&input[src_offset..src_offset + width]);
        }

        // Copy U plane
        let u_start = y_size;
        let uv_width = width / 2;
        let uv_height = height / 2;
        for y in 0..uv_height {
            let src_offset = u_start + y * uv_width;
            let dst_row = &mut frame.planes[1].rows_iter_mut().nth(y).unwrap();
            dst_row[..uv_width].copy_from_slice(&input[src_offset..src_offset + uv_width]);
        }

        // Copy V plane
        let v_start = y_size + uv_size;
        for y in 0..uv_height {
            let src_offset = v_start + y * uv_width;
            let dst_row = &mut frame.planes[2].rows_iter_mut().nth(y).unwrap();
            dst_row[..uv_width].copy_from_slice(&input[src_offset..src_offset + uv_width]);
        }

        // Send frame to encoder
        self.context_mut()?
            .send_frame(frame)
            .map_err(|e| Error::InvalidSegment(format!("rav1e send_frame failed: {:?}", e)))?;

        // Try to receive encoded packet
        match self.context_mut()?.receive_packet() {
            Ok(packet) => {
                self.frame_count += 1;
                Ok(Some(packet.data))
            }
            Err(rav1e::EncoderStatus::NeedMoreData) => Ok(None),
            Err(rav1e::EncoderStatus::Encoded) => {
                // Frame was encoded but no packet ready yet
                Ok(None)
            }
            Err(e) => Err(Error::InvalidSegment(format!(
                "rav1e encode failed: {:?}",
                e
            ))),
        }
    }

    /// Flush remaining frames from the encoder (internal implementation).
    fn flush_internal(&mut self) -> Result<Vec<Vec<u8>>> {
        // No context means no frame ever arrived, so there is nothing buffered.
        let Some(context) = self.context.as_mut() else {
            return Ok(Vec::new());
        };
        context.flush();

        let mut packets = Vec::new();
        loop {
            match context.receive_packet() {
                Ok(packet) => {
                    self.frame_count += 1;
                    packets.push(packet.data);
                }
                Err(rav1e::EncoderStatus::LimitReached) => break,
                Err(rav1e::EncoderStatus::Encoded) => continue,
                Err(rav1e::EncoderStatus::NeedMoreData) => continue,
                Err(e) => {
                    return Err(Error::InvalidSegment(format!(
                        "rav1e flush failed: {:?}",
                        e
                    )));
                }
            }
        }
        Ok(packets)
    }
}

impl Element for Rav1eEncoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input = buffer.as_bytes();
        let pts = buffer.metadata().pts.nanos();

        // Geometry travels in-band. There is no constructor size to fall back
        // on, deliberately: a stale one would silently encode the wrong
        // rectangle the moment a scaler appeared upstream (#38).
        let (width, height) = buffer.metadata().video_dims().ok_or_else(|| {
            Error::Element(
                "Rav1eEncoder: buffer carries no video dimensions. Geometry travels \
                 in-band — set Metadata::set_video_dims() upstream, or insert an element \
                 that does (VideoScale, VideoConvert, a device source)."
                    .into(),
            )
        })?;

        match self.encode_frame(input, width as usize, height as usize, pts)? {
            Some(packet) => {
                // Initialize arena on first use
                if self.arena.is_none() {
                    // Allocate enough for typical compressed frames
                    let arena_size = packet.len().max(1024 * 1024); // At least 1MB
                    self.arena =
                        Some(SharedArena::new(arena_size, 16).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }

                let arena = self.arena.as_mut().unwrap();
                arena.reclaim();
                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
                slot.data_mut()[..packet.len()].copy_from_slice(&packet);

                let metadata = buffer.metadata().clone();
                // Note: codec info could be added via MediaFormat if needed

                Ok(Some(Buffer::new(
                    MemoryHandle::with_len(slot, packet.len()),
                    metadata,
                )))
            }
            None => Ok(None), // Encoder buffering, no output yet
        }
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        // rav1e buffers frames for lookahead; without draining here the
        // tail of the stream is silently lost at EOS. The executor calls
        // this repeatedly until it returns None.
        if !self.flushed {
            self.flushed = true;
            let packets = self.flush_internal()?;
            self.pending_flush.extend(packets);
        }

        let Some(packet) = self.pending_flush.pop_front() else {
            return Ok(None);
        };

        if self.arena.is_none() {
            let arena_size = packet.len().max(1024 * 1024);
            self.arena = Some(
                SharedArena::new(arena_size, 16)
                    .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
            );
        }
        let arena = self.arena.as_mut().unwrap();
        arena.reclaim();
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
        slot.data_mut()[..packet.len()].copy_from_slice(&packet);

        Ok(Some(Buffer::new(
            MemoryHandle::with_len(slot, packet.len()),
            crate::metadata::Metadata::new(),
        )))
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::cpu_intensive() // Pure Rust, memory-safe, CPU intensive
    }
}

impl Drop for Rav1eEncoder {
    fn drop(&mut self) {
        // Flush is called automatically, but we ignore remaining packets
        let _ = self.flush_internal();
    }
}

impl VideoEncoder for Rav1eEncoder {
    type Packet = Vec<u8>;

    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Self::Packet>> {
        // Validate frame format
        if frame.format != PixelFormat::I420 {
            return Err(Error::InvalidSegment(format!(
                "Rav1eEncoder only supports I420, got {:?}",
                frame.format
            )));
        }

        // No config dimensions to disagree with: the frame is authoritative,
        // and ensure_context rebuilds if it differs from the last one.

        // Encode the frame
        match self.encode_frame(
            &frame.data,
            frame.width as usize,
            frame.height as usize,
            frame.pts as u64,
        )? {
            Some(packet) => Ok(vec![packet]),
            None => Ok(vec![]), // Encoder buffering
        }
    }

    fn flush(&mut self) -> Result<Vec<Self::Packet>> {
        self.flush_internal()
    }

    fn has_pending(&self) -> bool {
        // rav1e may have frames in lookahead buffer
        // We can't easily query this, so assume true if we've sent any frames
        self.frame_count > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rav1e_config_builder() {
        let config = Rav1eConfig::default().speed(8).quantizer(150).framerate(60);

        assert_eq!(config.speed, 8);
        assert_eq!(config.quantizer, 150);
        assert_eq!(config.timebase_den, 60);
    }

    /// #38: the encoder takes its geometry from the frame, and says so when
    /// the frame does not carry any.
    #[test]
    fn geometry_comes_from_the_buffer_not_the_config() {
        use crate::buffer::MemoryHandle;
        use crate::memory::SharedArena;
        use crate::metadata::Metadata;

        const W: u32 = 64;
        const H: u32 = 64;
        let frame_size = (W * H) as usize * 3 / 2;

        let mut encoder = Rav1eEncoder::new(Rav1eConfig::default().speed(10)).unwrap();

        // A frame with no declared geometry is an error, not a guess.
        let arena = SharedArena::new(frame_size, 4).unwrap();
        let bare = Buffer::new(
            MemoryHandle::with_len(arena.acquire().unwrap(), frame_size),
            Metadata::from_sequence(0),
        );
        let err = encoder.process(bare).unwrap_err();
        assert!(
            err.to_string().contains("no video dimensions"),
            "expected a geometry complaint, got: {err}"
        );

        // With geometry in-band, it builds its context from the frame.
        let mut metadata = Metadata::from_sequence(1);
        metadata.set_video_dims(W, H, crate::format::PixelFormat::I420);
        let described = Buffer::new(
            MemoryHandle::with_len(arena.acquire().unwrap(), frame_size),
            metadata,
        );
        encoder
            .process(described)
            .expect("encode with in-band dims");
        assert_eq!(encoder.dims, Some((W as usize, H as usize)));
    }

    #[test]
    fn test_rav1e_config_clamp() {
        let config = Rav1eConfig::default().speed(100).quantizer(500);

        assert_eq!(config.speed, 10); // Clamped to max
        assert_eq!(config.quantizer, 255); // Clamped to max
    }
}
