//! Common types for video codec elements.

use crate::format::PlaneLayout;

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
    /// Frame data.
    pub data: Vec<u8>,
    /// Where each plane starts and how far apart its rows sit. Always the
    /// packed layout for an owned frame — it allocates its own buffer.
    pub layout: PlaneLayout,
}

impl VideoFrame {
    /// Create a new video frame with allocated buffer.
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        let size = format.frame_size(width as usize, height as usize);

        Self {
            width,
            height,
            format,
            pts: 0,
            data: vec![0u8; size],
            layout: PlaneLayout::packed(format.into(), width, height),
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
            layout: self.layout,
        }
    }

    /// Get Y plane data.
    pub fn y_plane(&self) -> &[u8] {
        let p = self
            .as_view()
            .plane(0)
            .expect("every format has a luma plane");
        &self.data[p.offset..p.offset + p.stride * p.rows]
    }

    /// Get U plane data.
    pub fn u_plane(&self) -> &[u8] {
        let p = self.as_view().plane(1).expect("no chroma plane");
        &self.data[p.offset..p.offset + p.stride * p.rows]
    }

    /// Get V plane data.
    pub fn v_plane(&self) -> &[u8] {
        let p = self.as_view().plane(2).expect("no second chroma plane");
        &self.data[p.offset..p.offset + p.stride * p.rows]
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
    /// Frame data, borrowed from the producer.
    pub data: &'a [u8],
    /// Where each plane starts and how far apart its rows sit (#196).
    ///
    /// Packed for an arena-backed frame, the producer's real strides for a
    /// codec-owned one. Read planes through [`plane`](Self::plane) — never
    /// by deriving an offset from width, which is what the three `stride_*`
    /// fields this replaced invited.
    pub layout: PlaneLayout,
}

/// One plane of a frame: its rows, and how to walk them.
#[derive(Clone, Copy, Debug)]
pub struct FramePlane<'a> {
    /// The plane's `stride * rows` bytes.
    pub data: &'a [u8],
    /// Byte offset of the plane's first row within the whole frame.
    pub offset: usize,
    /// Byte distance between consecutive rows.
    pub stride: usize,
    /// Number of rows.
    pub rows: usize,
    /// Used bytes per row (<= stride).
    pub row_bytes: usize,
}

impl<'a> FramePlane<'a> {
    /// Row `y`'s used bytes.
    pub fn row(&self, y: usize) -> &'a [u8] {
        let start = y * self.stride;
        &self.data[start..start + self.row_bytes]
    }
}

impl<'a> VideoFrameRef<'a> {
    /// Plane `index`, or `None` when the format has fewer planes.
    pub fn plane(&self, index: usize) -> Option<FramePlane<'a>> {
        let p = self
            .layout
            .resolved(self.format.into(), self.width, self.height)
            .nth(index)?;
        let end = p.offset + p.stride * p.rows;
        Some(FramePlane {
            data: self.data.get(p.offset..end)?,
            offset: p.offset,
            stride: p.stride,
            rows: p.rows,
            row_bytes: p.row_bytes,
        })
    }

    /// The luma plane. Every format this type describes has one.
    pub fn y_plane(&self) -> FramePlane<'a> {
        self.plane(0).expect("every format has a luma plane")
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
        assert_eq!(view.plane(0).unwrap().data, frame.y_plane());
        assert_eq!(view.plane(1).unwrap().data, frame.u_plane());
        assert_eq!(view.plane(2).unwrap().data, frame.v_plane());
    }

    /// The NV12 chroma plane is one interleaved plane at full row width,
    /// and there is no third plane.
    ///
    /// Two producers used to disagree about this: `VideoFrame::new` set
    /// `stride_u = stride_y / 2` while `EncoderElement` set
    /// `stride_u = stride_y, stride_v = 0`, and `v4l2_m2m` carried a comment
    /// and a test about reconciling them. `PlaneLayout` is now the single
    /// source, so the disagreement cannot be expressed.
    #[test]
    fn nv12_has_one_full_width_chroma_plane() {
        let frame = VideoFrame::new(16, 16, PixelFormat::Nv12);
        let view = frame.as_view();
        assert_eq!(view.y_plane().stride, 16);
        let uv = view.plane(1).expect("interleaved chroma plane");
        assert_eq!((uv.stride, uv.rows, uv.row_bytes), (16, 8, 16));
        assert!(view.plane(2).is_none(), "NV12 has exactly two planes");
    }

    /// A strided frame's planes are found by offset, not by summing the
    /// preceding planes' sizes.
    #[test]
    fn plane_views_follow_a_strided_layout() {
        use crate::format::{PixelFormat as Caps, PlaneDesc, PlaneLayout};
        // Y 16x16 at stride 24, then U and V 8x8 at stride 12.
        let layout = PlaneLayout::from_planes(&[
            PlaneDesc {
                offset: 0,
                stride: 24,
            },
            PlaneDesc {
                offset: 24 * 16,
                stride: 12,
            },
            PlaneDesc {
                offset: 24 * 16 + 12 * 8,
                stride: 12,
            },
        ]);
        let len = layout.full_span_len(Caps::I420, 16, 16);
        let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let view = VideoFrameRef {
            width: 16,
            height: 16,
            format: PixelFormat::I420,
            pts: 0,
            data: &data,
            layout,
        };
        let v = view.plane(2).unwrap();
        assert_eq!(v.offset, 24 * 16 + 12 * 8);
        assert_eq!((v.stride, v.rows, v.row_bytes), (12, 8, 8));
        // Row 3 of V starts at its own offset plus three strides — nowhere
        // near `stride * height` arithmetic over the whole buffer.
        assert_eq!(v.row(3), &data[v.offset + 3 * 12..v.offset + 3 * 12 + 8]);
    }
}
