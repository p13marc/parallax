//! Per-frame allocation budgets on the media hot path (#137).
//!
//! Own integration binary because `#[global_allocator]` is per-binary
//! (same pattern as `tests/rt_integration.rs`). The budgets are ceilings
//! that later perf PRs ratchet down; each assertion prints the measured
//! value so regressions and improvements are visible in test output.

use std::alloc::{GlobalAlloc, Layout, System};

struct TrackingAllocator;

std::thread_local! {
    static ALLOC_TRACKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ALLOC_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_TRACKING.with(|tracking| {
            if tracking.get() {
                ALLOC_COUNT.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn tracked<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let start = ALLOC_COUNT.with(|c| c.get());
    ALLOC_TRACKING.with(|t| t.set(true));
    let r = f();
    ALLOC_TRACKING.with(|t| t.set(false));
    (r, ALLOC_COUNT.with(|c| c.get()) - start)
}

// ============================================================================
// Metadata clone budget (always compiled)
// ============================================================================

/// `Metadata::clone` is paid per frame per hop. Since #138 the custom map
/// stores small scalars inline in a SmallVec, so cloning after
/// `set_video_dims` must not touch the heap at all.
#[test]
fn metadata_clone_budget() {
    use parallax::metadata::Metadata;

    let mut m = Metadata::new();
    m.set_video_dims(1280, 720, parallax::format::PixelFormat::I420);

    // Warm any lazy state.
    let _ = m.clone();

    let mut keep = Vec::with_capacity(8);
    let (_c, allocs) = tracked(|| {
        for _ in 0..8 {
            keep.push(m.clone());
        }
    });
    drop(keep);
    let per_clone = allocs as f64 / 8.0;
    println!("metadata_clone_budget: {per_clone:.1} allocs per clone");
    assert!(
        per_clone <= 0.0,
        "Metadata::clone after set_video_dims: {per_clone:.1} allocs (budget 0)"
    );
}

/// An empty Metadata must clone allocation-free (this already holds; pin it).
#[test]
fn empty_metadata_clone_is_alloc_free() {
    use parallax::metadata::Metadata;

    let m = Metadata::new();
    let _ = m.clone();
    let mut keep = Vec::with_capacity(8);
    let (_c, allocs) = tracked(|| {
        for _ in 0..8 {
            keep.push(m.clone());
        }
    });
    drop(keep);
    println!("empty_metadata_clone: {allocs} allocs for 8 clones");
    assert_eq!(allocs, 0, "empty Metadata clone must not allocate");
}

// ============================================================================
// Decode + convert steady-state budget (feature-gated)
// ============================================================================

#[cfg(feature = "h264")]
mod decode_convert {
    use super::tracked;
    use parallax::buffer::{Buffer, MemoryHandle};
    use parallax::converters::PixelFormat as ConvPixelFormat;
    use parallax::element::Element;
    use parallax::elements::transform::VideoConvertElement;
    use parallax::elements::{H264Decoder, H264Encoder, H264EncoderConfig};
    use parallax::memory::SharedArena;
    use parallax::metadata::Metadata;

    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;
    const FRAME_SIZE: usize = (WIDTH as usize * HEIGHT as usize * 3) / 2;

    fn encoded_aus(n: usize) -> Vec<Vec<u8>> {
        let arena = SharedArena::new(FRAME_SIZE, 4).unwrap();
        let mut enc = H264Encoder::new(H264EncoderConfig::new()).unwrap();
        let mut aus = Vec::new();
        for i in 0..n {
            arena.reclaim();
            let mut slot = arena.acquire().unwrap();
            let data = slot.data_mut();
            let y = (WIDTH * HEIGHT) as usize;
            for (p, b) in data[..y].iter_mut().enumerate() {
                *b = ((p + i * 7) % 256) as u8;
            }
            for b in data[y..FRAME_SIZE].iter_mut() {
                *b = 128;
            }
            let mut metadata = Metadata::new();
            metadata.set_video_dims(WIDTH, HEIGHT, parallax::format::PixelFormat::I420);
            metadata.pts = parallax::clock::ClockTime::from_millis(i as u64 * 42);
            let buf = Buffer::new(MemoryHandle::with_len(slot, FRAME_SIZE), metadata);
            if let Some(out) = enc.process(buf).unwrap() {
                aus.push(out.as_bytes().to_vec());
            }
        }
        while let Some(out) = enc.flush().unwrap() {
            aus.push(out.as_bytes().to_vec());
        }
        aus
    }

    fn au_buffer(arena: &SharedArena, au: &[u8], i: usize) -> Buffer {
        arena.reclaim();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..au.len()].copy_from_slice(au);
        let mut metadata = Metadata::new();
        metadata.pts = parallax::clock::ClockTime::from_millis(i as u64 * 42);
        Buffer::new(MemoryHandle::with_len(slot, au.len()), metadata)
    }

    /// Steady-state heap allocations per frame through decode + convert.
    ///
    /// Baseline was 12.0; #138 (inline metadata) brought it to 4.0 — the
    /// remaining allocs are H264Decoder's per-frame payload Vecs, which
    /// #139/#140 ratchet toward <= 2.
    #[test]
    fn decode_convert_steady_state_budget() {
        let aus = encoded_aus(24);
        assert!(aus.len() >= 20, "need enough AUs, got {}", aus.len());
        let max = aus.iter().map(Vec::len).max().unwrap();
        let arena = SharedArena::new(max, 30).unwrap();

        let mut dec = H264Decoder::new().unwrap();
        let mut convert = VideoConvertElement::new()
            .with_input_format(ConvPixelFormat::I420)
            .with_output_format(ConvPixelFormat::Rgba)
            .with_size(WIDTH, HEIGHT);

        // Warm-up: arenas built, converter cached, decoder primed.
        for (i, au) in aus.iter().take(10).enumerate() {
            if let Some(d) = dec.process(au_buffer(&arena, au, i)).unwrap() {
                let _ = convert.process(d).unwrap();
            }
        }

        // Pre-build the tracked-region inputs so input construction isn't
        // counted (the executor hands elements ready-made Buffers).
        let tracked_aus: Vec<Buffer> = aus
            .iter()
            .enumerate()
            .skip(10)
            .take(10)
            .map(|(i, au)| au_buffer(&arena, au, i))
            .collect();
        let n = tracked_aus.len() as f64;

        let (frames, allocs) = tracked(|| {
            let mut frames = 0u32;
            for buf in tracked_aus {
                if let Some(d) = dec.process(buf).unwrap()
                    && let Some(rgba) = convert.process(d).unwrap()
                {
                    frames += 1;
                    std::hint::black_box(rgba.as_bytes().len());
                }
            }
            frames
        });
        assert!(frames >= 8, "decoded {frames} of 10 tracked frames");

        let per_frame = allocs as f64 / n;
        println!("decode_convert_steady_state: {per_frame:.1} allocs per frame");
        assert!(
            per_frame <= 8.0,
            "decode+convert steady state: {per_frame:.1} allocs/frame (budget 8)"
        );
    }
}
