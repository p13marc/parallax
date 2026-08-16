//! Common types for video codec elements.

/// Decode-but-drop threshold for video decoders (#165 ACCURATE clipping).
///
/// Tracks the Time segment the decoder is playing under: a decoded frame
/// whose PTS falls below `segment.start` (forward rate only) is dropped
/// *after* decoding — the demuxer legitimately starts data at the snapped
/// keyframe, but an ACCURATE seek's segment starts at the requested time,
/// and the executor makes that gap out-of-segment precisely so decoders
/// drop it here and the first shown frame is the request itself.
#[cfg(any(feature = "h264", feature = "av1-decode", feature = "vpx"))]
#[derive(Default)]
pub(crate) struct SegmentClip {
    below: Option<crate::clock::ClockTime>,
}

#[cfg(any(feature = "h264", feature = "av1-decode", feature = "vpx"))]
impl SegmentClip {
    /// Track a downstream event; Time segments update the threshold.
    pub(crate) fn observe(&mut self, event: &crate::event::Event) {
        if let crate::event::Event::Segment(seg) = event {
            self.below = (seg.format == crate::event::SegmentFormat::Time
                && seg.rate > 0.0
                && seg.start > 0)
                .then(|| crate::clock::ClockTime::from_nanos(seg.start as u64));
        }
    }

    /// Whether a decoded frame with this PTS is out-of-segment (drop it).
    pub(crate) fn clips(&self, pts: crate::clock::ClockTime) -> bool {
        match self.below {
            Some(start) => pts != crate::clock::ClockTime::NONE && pts < start,
            None => false,
        }
    }
}

/// Pixel format for video frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// Planar YUV 4:2:0, 8-bit
    I420,
    /// Planar YUV 4:2:0, 10-bit
    I420p10,
    /// Planar YUV 4:2:2, 8-bit
    I422,
    /// Planar YUV 4:4:4, 8-bit
    I444,
    /// NV12 (Y plane + interleaved UV)
    Nv12,
}

impl From<PixelFormat> for crate::format::PixelFormat {
    fn from(pf: PixelFormat) -> Self {
        use crate::format::PixelFormat as Caps;
        match pf {
            PixelFormat::I420 => Caps::I420,
            PixelFormat::I420p10 => Caps::I420_10Le,
            PixelFormat::I422 => Caps::I422,
            PixelFormat::I444 => Caps::I444,
            PixelFormat::Nv12 => Caps::Nv12,
        }
    }
}

impl PixelFormat {
    /// Get bytes per pixel component.
    pub fn bytes_per_component(&self) -> usize {
        match self {
            Self::I420p10 => 2,
            _ => 1,
        }
    }

    /// Calculate frame size in bytes.
    pub fn frame_size(&self, width: usize, height: usize) -> usize {
        let bpc = self.bytes_per_component();
        match self {
            Self::I420 | Self::I420p10 | Self::Nv12 => {
                // Y plane + UV planes (half resolution each dimension)
                width * height * bpc + 2 * (width / 2) * (height / 2) * bpc
            }
            Self::I422 => {
                // Y plane + UV planes (half width)
                width * height + 2 * (width / 2) * height
            }
            Self::I444 => {
                // Y plane + UV planes (full resolution)
                width * height * 3
            }
        }
    }
}

/// A decoded video frame.
#[derive(Clone, Debug)]
pub struct VideoFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Presentation timestamp (in timebase units).
    pub pts: i64,
    /// Frame data (planar layout).
    pub data: Vec<u8>,
    /// Stride for Y plane.
    pub stride_y: usize,
    /// Stride for U plane.
    pub stride_u: usize,
    /// Stride for V plane.
    pub stride_v: usize,
}

impl VideoFrame {
    /// Create a new video frame with allocated buffer.
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        let size = format.frame_size(width as usize, height as usize);
        let stride_y = width as usize * format.bytes_per_component();
        let stride_uv = match format {
            PixelFormat::I444 => stride_y,
            _ => stride_y / 2,
        };

        Self {
            width,
            height,
            format,
            pts: 0,
            data: vec![0u8; size],
            stride_y,
            stride_u: stride_uv,
            stride_v: stride_uv,
        }
    }

    /// Borrow this frame as the view type the encoder traits take.
    pub fn as_view(&self) -> VideoFrameRef<'_> {
        VideoFrameRef {
            width: self.width,
            height: self.height,
            format: self.format,
            pts: self.pts,
            data: &self.data,
            stride_y: self.stride_y,
            stride_u: self.stride_u,
            stride_v: self.stride_v,
        }
    }

    /// Get Y plane data.
    pub fn y_plane(&self) -> &[u8] {
        self.as_view().y_plane()
    }

    /// Get U plane data.
    pub fn u_plane(&self) -> &[u8] {
        self.as_view().u_plane()
    }

    /// Get V plane data.
    pub fn v_plane(&self) -> &[u8] {
        self.as_view().v_plane()
    }
}

/// A borrowed view of a raw video frame (see [`VideoFrame`] for the owned form).
///
/// This is what [`VideoEncoder::encode`](super::VideoEncoder::encode) takes:
/// `EncoderElement` builds one directly over the input buffer's bytes, so
/// encoding a frame copies nothing.
#[derive(Clone, Copy, Debug)]
pub struct VideoFrameRef<'a> {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Presentation timestamp (in timebase units).
    pub pts: i64,
    /// Frame data (planar layout), borrowed from the producer.
    pub data: &'a [u8],
    /// Stride for Y plane.
    pub stride_y: usize,
    /// Stride for U plane.
    pub stride_u: usize,
    /// Stride for V plane.
    pub stride_v: usize,
}

impl<'a> VideoFrameRef<'a> {
    /// Get Y plane data.
    pub fn y_plane(&self) -> &'a [u8] {
        let y_size = self.stride_y * self.height as usize;
        &self.data[..y_size]
    }

    /// Get U plane data.
    pub fn u_plane(&self) -> &'a [u8] {
        let y_size = self.stride_y * self.height as usize;
        let uv_height = match self.format {
            PixelFormat::I422 | PixelFormat::I444 => self.height as usize,
            _ => self.height as usize / 2,
        };
        let u_size = self.stride_u * uv_height;
        &self.data[y_size..y_size + u_size]
    }

    /// Get V plane data.
    pub fn v_plane(&self) -> &'a [u8] {
        let y_size = self.stride_y * self.height as usize;
        let uv_height = match self.format {
            PixelFormat::I422 | PixelFormat::I444 => self.height as usize,
            _ => self.height as usize / 2,
        };
        let u_size = self.stride_u * uv_height;
        let v_start = y_size + u_size;
        let v_size = self.stride_v * uv_height;
        &self.data[v_start..v_start + v_size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_size() {
        assert_eq!(
            PixelFormat::I420.frame_size(1920, 1080),
            1920 * 1080 * 3 / 2
        );
        assert_eq!(PixelFormat::I444.frame_size(1920, 1080), 1920 * 1080 * 3);
    }

    #[test]
    fn test_video_frame_planes() {
        let frame = VideoFrame::new(16, 16, PixelFormat::I420);
        assert_eq!(frame.y_plane().len(), 16 * 16);
        assert_eq!(frame.u_plane().len(), 8 * 8);
        assert_eq!(frame.v_plane().len(), 8 * 8);
    }

    #[test]
    fn view_planes_match_owned() {
        let mut frame = VideoFrame::new(16, 16, PixelFormat::I422);
        for (i, b) in frame.data.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let view = frame.as_view();
        assert_eq!(view.y_plane(), frame.y_plane());
        assert_eq!(view.u_plane(), frame.u_plane());
        assert_eq!(view.v_plane(), frame.v_plane());
    }
}
