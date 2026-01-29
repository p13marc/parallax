//! Benchmarks for colorspace conversion comparing SIMD vs scalar performance.
//!
//! Run with:
//!   cargo bench --features simd-colorspace -- colorspace
//!
//! Compare scalar (without feature):
//!   cargo bench -- colorspace

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parallax::converters::{PixelFormat, VideoConvert};

/// Common resolutions to benchmark
const RESOLUTIONS: &[(u32, u32, &str)] = &[
    (640, 480, "VGA"),
    (1280, 720, "720p"),
    (1920, 1080, "1080p"),
    (3840, 2160, "4K"),
];

fn bench_i420_to_rgba(c: &mut Criterion) {
    let mut group = c.benchmark_group("i420_to_rgba");

    for &(width, height, name) in RESOLUTIONS {
        let input_size = PixelFormat::I420.buffer_size(width, height);
        let output_size = PixelFormat::Rgba.buffer_size(width, height);

        group.throughput(Throughput::Bytes(input_size as u64));

        // Create test input with realistic YUV values
        let mut input = vec![0u8; input_size];
        // Fill Y plane with gradient
        for i in 0..(width * height) as usize {
            input[i] = ((i * 255) / (width * height) as usize) as u8;
        }
        // Fill U and V planes with neutral chroma
        let y_size = (width * height) as usize;
        let uv_size = (width / 2 * height / 2) as usize;
        for i in 0..uv_size {
            input[y_size + i] = 128; // U
            input[y_size + uv_size + i] = 128; // V
        }

        let mut output = vec![0u8; output_size];

        let converter = VideoConvert::new(PixelFormat::I420, PixelFormat::Rgba, width, height)
            .expect("Failed to create converter");

        group.bench_with_input(BenchmarkId::new("convert", name), &input, |b, input| {
            b.iter(|| {
                converter.convert(input, &mut output).unwrap();
                std::hint::black_box(&output);
            });
        });
    }

    group.finish();
}

fn bench_rgba_to_i420(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgba_to_i420");

    for &(width, height, name) in RESOLUTIONS {
        let input_size = PixelFormat::Rgba.buffer_size(width, height);
        let output_size = PixelFormat::I420.buffer_size(width, height);

        group.throughput(Throughput::Bytes(input_size as u64));

        // Create test input with gradient pattern
        let mut input = vec![0u8; input_size];
        for i in 0..(width * height) as usize {
            let idx = i * 4;
            input[idx] = ((i % 256) as u8); // R
            input[idx + 1] = ((i / 256 % 256) as u8); // G
            input[idx + 2] = ((i / 65536 % 256) as u8); // B
            input[idx + 3] = 255; // A
        }

        let mut output = vec![0u8; output_size];

        let converter = VideoConvert::new(PixelFormat::Rgba, PixelFormat::I420, width, height)
            .expect("Failed to create converter");

        group.bench_with_input(BenchmarkId::new("convert", name), &input, |b, input| {
            b.iter(|| {
                converter.convert(input, &mut output).unwrap();
                std::hint::black_box(&output);
            });
        });
    }

    group.finish();
}

fn bench_yuyv_to_rgba(c: &mut Criterion) {
    let mut group = c.benchmark_group("yuyv_to_rgba");

    for &(width, height, name) in RESOLUTIONS {
        let input_size = PixelFormat::Yuyv.buffer_size(width, height);
        let output_size = PixelFormat::Rgba.buffer_size(width, height);

        group.throughput(Throughput::Bytes(input_size as u64));

        // Create packed YUYV data
        let mut input = vec![0u8; input_size];
        for i in 0..(width * height / 2) as usize {
            let idx = i * 4;
            input[idx] = 128; // Y0
            input[idx + 1] = 128; // U
            input[idx + 2] = 128; // Y1
            input[idx + 3] = 128; // V
        }

        let mut output = vec![0u8; output_size];

        let converter = VideoConvert::new(PixelFormat::Yuyv, PixelFormat::Rgba, width, height)
            .expect("Failed to create converter");

        group.bench_with_input(BenchmarkId::new("convert", name), &input, |b, input| {
            b.iter(|| {
                converter.convert(input, &mut output).unwrap();
                std::hint::black_box(&output);
            });
        });
    }

    group.finish();
}

fn bench_nv12_to_rgba(c: &mut Criterion) {
    let mut group = c.benchmark_group("nv12_to_rgba");

    for &(width, height, name) in RESOLUTIONS {
        let input_size = PixelFormat::Nv12.buffer_size(width, height);
        let output_size = PixelFormat::Rgba.buffer_size(width, height);

        group.throughput(Throughput::Bytes(input_size as u64));

        // Create NV12 data (Y plane + interleaved UV)
        let mut input = vec![0u8; input_size];
        let y_size = (width * height) as usize;
        // Fill Y with gradient
        for i in 0..y_size {
            input[i] = ((i * 255) / y_size) as u8;
        }
        // Fill UV with neutral chroma
        for i in y_size..input_size {
            input[i] = 128;
        }

        let mut output = vec![0u8; output_size];

        let converter = VideoConvert::new(PixelFormat::Nv12, PixelFormat::Rgba, width, height)
            .expect("Failed to create converter");

        group.bench_with_input(BenchmarkId::new("convert", name), &input, |b, input| {
            b.iter(|| {
                converter.convert(input, &mut output).unwrap();
                std::hint::black_box(&output);
            });
        });
    }

    group.finish();
}

fn bench_rgba_to_nv12(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgba_to_nv12");

    for &(width, height, name) in RESOLUTIONS {
        let input_size = PixelFormat::Rgba.buffer_size(width, height);
        let output_size = PixelFormat::Nv12.buffer_size(width, height);

        group.throughput(Throughput::Bytes(input_size as u64));

        // Create RGBA data with gradient
        let mut input = vec![0u8; input_size];
        for i in 0..(width * height) as usize {
            let idx = i * 4;
            input[idx] = ((i % 256) as u8); // R
            input[idx + 1] = ((i / 256 % 256) as u8); // G
            input[idx + 2] = ((i / 65536 % 256) as u8); // B
            input[idx + 3] = 255; // A
        }

        let mut output = vec![0u8; output_size];

        let converter = VideoConvert::new(PixelFormat::Rgba, PixelFormat::Nv12, width, height)
            .expect("Failed to create converter");

        group.bench_with_input(BenchmarkId::new("convert", name), &input, |b, input| {
            b.iter(|| {
                converter.convert(input, &mut output).unwrap();
                std::hint::black_box(&output);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_i420_to_rgba,
    bench_rgba_to_i420,
    bench_yuyv_to_rgba,
    bench_nv12_to_rgba,
    bench_rgba_to_nv12,
);

criterion_main!(benches);
