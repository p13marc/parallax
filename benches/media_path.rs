//! Media-path benchmarks: demux → decode → convert, the player's hot path.
//!
//! Baseline harness for the perf pass (#137) — every media-path perf PR
//! reports before/after numbers from here.
//!
//! Run with:
//!   just bench-media
//!   # or: cargo bench --features h264,mkv-demux --bench media_path
//!
//! The codec-dependent groups are feature-gated inside the file (the bench
//! target itself carries no `required-features`, like `colorspace`).

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "h264")]
mod encoded_fixture {
    use parallax::buffer::{Buffer, MemoryHandle};
    use parallax::element::Element;
    use parallax::elements::{H264Encoder, H264EncoderConfig};
    use parallax::memory::SharedArena;
    use parallax::metadata::Metadata;

    pub const WIDTH: u32 = 1280;
    pub const HEIGHT: u32 = 720;
    pub const FRAME_SIZE: usize = (WIDTH as usize * HEIGHT as usize * 3) / 2;
    pub const FRAMES: usize = 24;

    /// A raw I420 buffer with in-band geometry, frame `i` of a moving
    /// gradient (so inter frames have real motion to encode).
    pub fn raw_frame(arena: &SharedArena, i: usize) -> Buffer {
        arena.reclaim();
        let mut slot = arena.acquire().expect("bench arena slot");
        let data = slot.data_mut();
        let y_size = (WIDTH * HEIGHT) as usize;
        for (p, byte) in data[..y_size].iter_mut().enumerate() {
            *byte = ((p + i * 7) % 256) as u8;
        }
        for byte in data[y_size..FRAME_SIZE].iter_mut() {
            *byte = 128;
        }
        let mut metadata = Metadata::new();
        metadata.set_video_dims(WIDTH, HEIGHT, parallax::format::PixelFormat::I420);
        metadata.pts = parallax::clock::ClockTime::from_millis(i as u64 * 42);
        Buffer::new(MemoryHandle::with_len(slot, FRAME_SIZE), metadata)
    }

    /// Encode `FRAMES` frames of 720p test content to Annex-B access units.
    pub fn encoded_aus() -> Vec<Vec<u8>> {
        let arena = SharedArena::new(FRAME_SIZE, 4).expect("bench arena");
        let mut enc = H264Encoder::new(H264EncoderConfig::new()).expect("encoder");
        let mut aus = Vec::with_capacity(FRAMES);
        for i in 0..FRAMES {
            if let Some(out) = enc.process(raw_frame(&arena, i)).expect("encode") {
                aus.push(out.as_bytes().to_vec());
            }
        }
        while let Some(out) = enc.flush().expect("encoder flush") {
            aus.push(out.as_bytes().to_vec());
        }
        assert!(!aus.is_empty(), "encoder produced access units");
        aus
    }

    /// Wrap an AU in a Buffer (arena-backed, empty metadata + pts).
    pub fn au_buffers(aus: &[Vec<u8>]) -> Vec<Buffer> {
        let max = aus.iter().map(Vec::len).max().unwrap_or(1);
        let arena = SharedArena::new(max, aus.len() + 2).expect("au arena");
        aus.iter()
            .enumerate()
            .map(|(i, au)| {
                arena.reclaim();
                let mut slot = arena.acquire().expect("au slot");
                slot.data_mut()[..au.len()].copy_from_slice(au);
                let mut metadata = Metadata::new();
                metadata.pts = parallax::clock::ClockTime::from_millis(i as u64 * 42);
                Buffer::new(MemoryHandle::with_len(slot, au.len()), metadata)
            })
            .collect()
    }
}

/// Demux the checked-in H.264+AAC MKV fixture to EOS.
#[cfg(feature = "mkv-demux")]
fn bench_demux_mkv(c: &mut Criterion) {
    use criterion::{BatchSize, Throughput};
    use parallax::element::{Demuxer, DemuxerProduce};
    use parallax::elements::demux::MkvDemux;
    use std::io::Cursor;

    const FIXTURE: &[u8] = include_bytes!("../tests/fixtures/tiny_h264_aac.mkv");

    let mut group = c.benchmark_group("media_path/demux_mkv");
    group.throughput(Throughput::Bytes(FIXTURE.len() as u64));
    group.bench_function("h264_aac_to_eos", |b| {
        b.iter_batched(
            || MkvDemux::new(Cursor::new(FIXTURE.to_vec())).expect("fixture parses"),
            |mut demux| {
                let mut buffers = 0u32;
                loop {
                    match demux.produce().expect("produce") {
                        DemuxerProduce::Routed(routed) => {
                            for (_, buf) in routed {
                                buffers += 1;
                                std::hint::black_box(buf.as_bytes().len());
                            }
                        }
                        DemuxerProduce::Eos => break,
                        other => panic!("unexpected {other:?}"),
                    }
                }
                std::hint::black_box(buffers)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Decode 24 encoded 720p AUs with a fresh decoder per iteration.
#[cfg(feature = "h264")]
fn bench_decode_720p(c: &mut Criterion) {
    use criterion::{BatchSize, Throughput};
    use encoded_fixture as fx;
    use parallax::element::Element;
    use parallax::elements::H264Decoder;

    let aus = fx::encoded_aus();
    let decoded_bytes = (aus.len() * fx::FRAME_SIZE) as u64;

    let mut group = c.benchmark_group("media_path/decode_h264_720p");
    group.throughput(Throughput::Bytes(decoded_bytes));
    group.sample_size(20);
    group.bench_function("24_frames", |b| {
        b.iter_batched(
            || (H264Decoder::new().expect("decoder"), fx::au_buffers(&aus)),
            |(mut dec, buffers)| {
                let mut frames = 0u32;
                for buf in buffers {
                    if let Some(out) = dec.process(buf).expect("decode") {
                        frames += 1;
                        std::hint::black_box(out.as_bytes().len());
                    }
                }
                while let Some(out) = Element::flush(&mut dec).expect("flush") {
                    frames += 1;
                    std::hint::black_box(out.as_bytes().len());
                }
                std::hint::black_box(frames)
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

/// Decode + I420→RGBA convert — the player's whole video transform chain.
#[cfg(feature = "h264")]
fn bench_decode_convert_720p(c: &mut Criterion) {
    use criterion::{BatchSize, Throughput};
    use encoded_fixture as fx;
    use parallax::converters::PixelFormat as ConvPixelFormat;
    use parallax::element::Element;
    use parallax::elements::H264Decoder;
    use parallax::elements::transform::VideoConvertElement;

    let aus = fx::encoded_aus();
    let decoded_bytes = (aus.len() * fx::FRAME_SIZE) as u64;

    let mut group = c.benchmark_group("media_path/decode_convert_720p");
    group.throughput(Throughput::Bytes(decoded_bytes));
    group.sample_size(20);
    group.bench_function("24_frames_to_rgba", |b| {
        b.iter_batched(
            || {
                let convert = VideoConvertElement::new()
                    .with_input_format(ConvPixelFormat::I420)
                    .with_output_format(ConvPixelFormat::Rgba)
                    .with_size(fx::WIDTH, fx::HEIGHT);
                (
                    H264Decoder::new().expect("decoder"),
                    convert,
                    fx::au_buffers(&aus),
                )
            },
            |(mut dec, mut convert, buffers)| {
                let mut frames = 0u32;
                for buf in buffers {
                    if let Some(decoded) = dec.process(buf).expect("decode")
                        && let Some(rgba) = convert.process(decoded).expect("convert")
                    {
                        frames += 1;
                        std::hint::black_box(rgba.as_bytes().len());
                    }
                }
                std::hint::black_box(frames)
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn all(c: &mut Criterion) {
    #[cfg(feature = "mkv-demux")]
    bench_demux_mkv(c);
    #[cfg(feature = "h264")]
    bench_decode_720p(c);
    #[cfg(feature = "h264")]
    bench_decode_convert_720p(c);
    #[cfg(not(any(feature = "mkv-demux", feature = "h264")))]
    {
        let _ = c;
        eprintln!("media_path: enable --features h264,mkv-demux for the full suite");
    }
}

criterion_group!(benches, all);
criterion_main!(benches);
