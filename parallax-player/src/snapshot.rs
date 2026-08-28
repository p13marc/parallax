//! Saving the frame on screen.
//!
//! Deliberately not a branch in the pipeline. The issue (#283) proposes
//! `Valve → VideoConvert → PngEncoder → FileSink` off the decoder's src pad,
//! and each of those three has a problem the design does not mention: a
//! closed `Valve` *drops* rather than queues and has no one-shot mode, so
//! opening and closing it from the interface thread lets an unbounded number
//! of frames through; a `FileSink` cannot be re-targeted and appends, so N
//! snapshots concatenate into one broken PNG; and `Valve::passthrough()` is
//! true, which means the branch adds its whole downstream capacity to the
//! decoder's hold rather than costing nothing while idle.
//!
//! The host already has the frame. `FrameMailbox::last_presented()` hands
//! back the *decoder's own output* — the buffer it wrote, with its geometry,
//! plane layout and colour — which is exactly the "true frame grab, not what
//! I see" the issue asks for, and better than a widget screenshot, which
//! would capture the scaled, letterboxed, overlaid window instead.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use parallax::converters::{PixelFormat as ConvertFormat, VideoConvert};
use parallax::format::{PixelFormat, PlaneLayout};
use parallax::video::{PacedFrame, VideoFrame};

/// Write `frame` to a PNG beside `source`, and return where it went.
///
/// Blocking: it converts and compresses a full-resolution frame. Call it on
/// `spawn_blocking`, not on the interface's thread.
pub fn write_png(frame: &PacedFrame, source: &Path, at: u64) -> Result<PathBuf> {
    let view = frame.frame.view();
    let (width, height) = (view.width, view.height);
    if width == 0 || height == 0 {
        bail!("the frame on screen has no dimensions");
    }

    let path = destination(source, at);
    let rgb = to_rgb24(&frame.frame, width, height)?;

    let file =
        std::fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut writer| writer.write_image_data(&rgb))
        .with_context(|| format!("writing {}", path.display()))?;

    Ok(path)
}

/// The frame's pixels as packed 8-bit RGB.
///
/// The converter is the one the pipeline uses, so a snapshot and the picture
/// it was taken from agree about colour — including the matrix and range the
/// stream declared (#265). A frame that is already `Rgb24` skips it.
pub(crate) fn to_rgb24(frame: &VideoFrame, width: u32, height: u32) -> Result<Vec<u8>> {
    let view = frame.view();
    rgb24_from(
        frame.data.as_bytes(),
        view.format,
        view.layout,
        view.color,
        width,
        height,
    )
}

/// Packed 8-bit RGB from raw pixels of a known shape.
///
/// Split out of [`to_rgb24`] so the thumbnailer can convert bytes it has
/// already scaled, which are no longer inside a `Buffer` (#282). The colour
/// handling is the part worth sharing: the converter is the one the pipeline
/// uses, told what the stream declared, so every picture this player produces
/// agrees about matrix and range (#265).
pub(crate) fn rgb24_from(
    bytes: &[u8],
    format: PixelFormat,
    layout: Option<PlaneLayout>,
    color: Option<parallax::format::ColorInfo>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    convert_to(
        ConvertFormat::Rgb24,
        bytes,
        format,
        layout,
        color,
        width,
        height,
    )
}

/// Raw pixels of a known shape, converted to a packed RGB-family format.
///
/// The output format is a parameter because a PNG wants three bytes per pixel
/// and Iced's `image::Handle::from_rgba` wants four (#282). Everything else is
/// shared, and the part worth sharing is the colour handling: the converter is
/// the one the pipeline uses, told what the stream declared, so every picture
/// this player produces agrees about matrix and range (#265).
pub(crate) fn convert_to(
    target: ConvertFormat,
    bytes: &[u8],
    format: PixelFormat,
    layout: Option<PlaneLayout>,
    color: Option<parallax::format::ColorInfo>,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let target_as_caps = PixelFormat::from(target);
    if format == target_as_caps && layout.is_none() {
        return Ok(bytes.to_vec());
    }

    let input: ConvertFormat = format
        .try_into()
        .map_err(|_| anyhow::anyhow!("cannot convert {format:?} to {target:?}"))?;
    let mut converter =
        VideoConvert::new(input, target, width, height).context("building the converter")?;
    if let Some(color) = color {
        converter = converter.with_color_info(color);
    }

    // The converter addresses planes by whole rows, so a packed frame has to
    // say so explicitly rather than leaving the layout `None`.
    let layout = layout.unwrap_or_else(|| PlaneLayout::packed(format, width, height));
    let mut out = vec![0u8; target.buffer_size(width, height)];
    converter
        .convert(bytes, layout, &mut out)
        .context("converting the frame")?;
    Ok(out)
}

/// Where a snapshot of `source` taken at `at` nanoseconds goes.
///
/// Beside the video, named after it and the position, so a directory of
/// snapshots sorts into the order they were taken and nothing is overwritten
/// by accident.
pub fn destination(source: &Path, at: u64) -> PathBuf {
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "frame".into());
    let millis = at / 1_000_000;
    let name = format!(
        "{stem}-{:02}h{:02}m{:02}.{:03}s.png",
        millis / 3_600_000,
        (millis / 60_000) % 60,
        (millis / 1000) % 60,
        millis % 1000,
    );
    source.parent().unwrap_or(Path::new(".")).join(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    use parallax::buffer::{Buffer, MemoryHandle};
    use parallax::clock::ClockTime;
    use parallax::memory::SharedArena;
    use parallax::metadata::Metadata;
    use parallax::video::VideoFrame;

    /// A 64x64 I420 frame of limited-range black, the way a decoder emits it.
    fn black_i420() -> PacedFrame {
        const W: usize = 64;
        const H: usize = 64;
        let len = W * H * 3 / 2;
        let arena = SharedArena::new(len.next_power_of_two(), 2).expect("arena");
        let slot = arena.acquire().expect("slot");
        {
            let mut handle = MemoryHandle::with_len(slot.clone(), len);
            let bytes = handle.as_mut_slice().expect("writable");
            bytes[..W * H].fill(16); // Y: limited-range black
            bytes[W * H..].fill(128); // U and V: neutral chroma
        }
        let mut metadata = Metadata::default();
        metadata.set_video_dims(W as u32, H as u32, PixelFormat::I420);
        PacedFrame {
            frame: VideoFrame {
                data: Buffer::new(MemoryHandle::with_len(slot, len), metadata),
                width: W as u32,
                height: H as u32,
                format: PixelFormat::I420,
                layout: None,
                color: None,
                display: None,
            },
            target: ClockTime::ZERO,
            generation: 0,
            must_present: false,
        }
    }

    /// The file that lands is the source frame, at the source's size (#283).
    #[test]
    fn a_snapshot_is_the_decoded_frame_at_full_resolution() {
        let dir = std::env::temp_dir().join(format!("parallax-snapshot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let source = dir.join("clip.mkv");

        let path = write_png(&black_i420(), &source, 1_500_000_000).expect("it writes");
        assert_eq!(path, dir.join("clip-00h00m01.500s.png"));

        let decoder = png::Decoder::new(std::io::BufReader::new(
            std::fs::File::open(&path).expect("it exists"),
        ));
        let mut reader = decoder.read_info().expect("a valid PNG");
        assert_eq!(
            (reader.info().width, reader.info().height),
            (64, 64),
            "the source frame's size, not the window's"
        );
        assert_eq!(reader.info().color_type, png::ColorType::Rgb);

        let mut pixels = vec![0u8; reader.output_buffer_size().expect("a bounded size")];
        let frame = reader.next_frame(&mut pixels).expect("the image data");
        // Y=16 is limited-range black, and the converter is the pipeline's —
        // so the file holds black, not the dark grey a full-range reading of
        // the same bytes would give (#265).
        let brightest = pixels[..frame.buffer_size()]
            .iter()
            .copied()
            .max()
            .unwrap_or(255);
        assert!(
            brightest <= 4,
            "limited-range black came out at {brightest}, so the snapshot did \
             not go through the pipeline's own colour conversion"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_snapshot_is_named_after_the_film_and_the_moment() {
        assert_eq!(
            destination(Path::new("/films/holiday.mkv"), 3_723_400_000_000),
            PathBuf::from("/films/holiday-01h02m03.400s.png")
        );
        // Zero is a legitimate position, not a missing one.
        assert_eq!(
            destination(Path::new("clip.mp4"), 0),
            PathBuf::from("clip-00h00m00.000s.png")
        );
        // Two snapshots a second apart do not collide.
        assert_ne!(
            destination(Path::new("a.mkv"), 1_000_000_000),
            destination(Path::new("a.mkv"), 2_000_000_000)
        );
    }
}
