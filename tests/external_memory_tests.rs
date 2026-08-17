//! External-memory flow-through (#194): strided producer-owned buffers
//! travel the pipeline as first-class `Buffer`s, gated by explicit
//! consumer opt-in, with pool recycling driven by the release hook and a
//! memorycopy repack bridging non-opting consumers — no codec required.

use parallax::element::{ConsumeContext, Sink};
use parallax::elements::testing::{
    ExternalTestSrc, PAD_BYTE, TEST_HEIGHT, TEST_WIDTH, packed_reference_frame,
};
use parallax::elements::{AppSink, Pulled};
use parallax::error::Result;
use parallax::format::{
    ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, PixelFormat, VideoFormatCaps,
};
use parallax::memory::MemoryType;
use parallax::pipeline::{Executor, Pipeline};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A sink that opted into External and verifies every frame through the
/// declared plane layout.
struct OptInSink {
    seen: Arc<AtomicU64>,
    external: Arc<AtomicU64>,
    errors: Arc<Mutex<Vec<String>>>,
    /// The last buffer, held past pipeline end by the test.
    held: Arc<Mutex<Option<parallax::buffer::Buffer>>>,
}

impl Sink for OptInSink {
    fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let buffer = ctx.buffer();
        let seq = buffer.metadata().sequence;
        self.seen.fetch_add(1, Ordering::SeqCst);
        if buffer.memory_type() == MemoryType::External {
            self.external.fetch_add(1, Ordering::SeqCst);
        }

        let fail = |msg: String| self.errors.lock().unwrap().push(msg);

        let meta = buffer.metadata();
        if !meta.has_strided_planes() {
            fail(format!("seq {seq}: expected a strided layout"));
            return Ok(());
        }
        let (Some(layout), Some((w, h)), Some(fmt)) = (
            meta.plane_layout(),
            meta.video_dims(),
            meta.video_pixel_format(),
        ) else {
            fail(format!("seq {seq}: missing video geometry"));
            return Ok(());
        };

        // Padding bytes are visible in the raw span (proof the frame
        // really is strided)...
        let first = layout.resolved(fmt, w, h).next().unwrap();
        let pad_probe = buffer.as_bytes()[first.offset + first.row_bytes];
        if pad_probe != PAD_BYTE {
            fail(format!(
                "seq {seq}: expected padding sentinel, got {pad_probe:#x}"
            ));
        }

        // ...and layout-directed reads reconstruct the packed reference.
        let mut packed = vec![0u8; layout.required_len(fmt, w, h)];
        let n = layout
            .repack_into(buffer.as_bytes(), fmt, w, h, &mut packed)
            .expect("repack");
        packed.truncate(n);
        // Pool slots cycle: slot pattern = seq % pool (pool is 4 here).
        if packed != packed_reference_frame(seq % 4) {
            fail(format!("seq {seq}: layout-directed read mismatch"));
        }

        *self.held.lock().unwrap() = Some(buffer.clone());
        Ok(())
    }

    fn input_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(VideoFormatCaps::any()),
            MemoryCaps::external_or_cpu(),
        )])
    }

    fn name(&self) -> &str {
        "opt_in_sink"
    }
}

/// A sink that only accepts CPU memory and byte-checks the packed frames.
struct CpuOnlySink {
    seen: Arc<AtomicU64>,
    cpu_packed: Arc<AtomicU64>,
}

impl Sink for CpuOnlySink {
    fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let buffer = ctx.buffer();
        self.seen.fetch_add(1, Ordering::SeqCst);
        let seq = buffer.metadata().sequence;
        if buffer.memory_type() == MemoryType::Cpu
            && !buffer.metadata().has_strided_planes()
            && buffer.as_bytes() == packed_reference_frame(seq % 4)
        {
            self.cpu_packed.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    fn input_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(VideoFormatCaps::any()),
            MemoryCaps::cpu_only(),
        )])
    }

    fn name(&self) -> &str {
        "cpu_only_sink"
    }
}

/// External-only source → opted-in sink: External flows end to end,
/// strided layout intact, pool recycling past pool size, and the last
/// buffer stays valid after the pipeline is gone (producer memory is
/// pinned by the slot, not the element).
#[tokio::test(flavor = "multi_thread")]
async fn external_flows_end_to_end_and_pool_recycles() {
    const POOL: u32 = 4;
    const FRAMES: u64 = 12;

    let seen = Arc::new(AtomicU64::new(0));
    let external = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(Mutex::new(Vec::new()));
    let held = Arc::new(Mutex::new(None));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", ExternalTestSrc::new(POOL, FRAMES));
    let snk = pipeline.add_sink(
        "sink",
        OptInSink {
            seen: seen.clone(),
            external: external.clone(),
            errors: errors.clone(),
            held: held.clone(),
        },
    );
    pipeline.link(src, snk).unwrap();

    // Deny policy: External × [External, Cpu] must negotiate directly.
    pipeline.prepare().unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    assert_eq!(seen.load(Ordering::SeqCst), FRAMES);
    assert_eq!(
        external.load(Ordering::SeqCst),
        FRAMES,
        "every frame rode External"
    );
    assert!(
        FRAMES > POOL as u64,
        "test premise: more frames than pool slots"
    );
    let errs = errors.lock().unwrap();
    assert!(errs.is_empty(), "{errs:?}");
    drop(errs);

    // The held buffer outlives the pipeline: the ExternalSlot pins the
    // producer's memory, so reads stay valid until the last drop.
    let buffer = held.lock().unwrap().take().expect("sink held a buffer");
    let meta = buffer.metadata().clone();
    let (w, h) = meta.video_dims().unwrap();
    assert_eq!((w, h), (TEST_WIDTH, TEST_HEIGHT));
    let layout = meta.plane_layout().unwrap();
    let fmt = meta.video_pixel_format().unwrap();
    let mut packed = vec![0u8; layout.required_len(fmt, w, h)];
    let n = layout
        .repack_into(buffer.as_bytes(), fmt, w, h, &mut packed)
        .unwrap();
    packed.truncate(n);
    assert_eq!(
        packed,
        packed_reference_frame(meta.sequence % POOL as u64),
        "post-pipeline read through the pinned slot"
    );
}

/// External-only source against a CPU-only sink: Deny names memorycopy;
/// auto-converters insert the repack and the sink receives PACKED CPU
/// frames, byte-exact, with the strided layout cleared.
#[tokio::test(flavor = "multi_thread")]
async fn memorycopy_repacks_for_cpu_only_sink() {
    // Deny first.
    {
        let mut pipeline = Pipeline::new();
        let src = pipeline.add_source("src", ExternalTestSrc::new(2, 4));
        let snk = pipeline.add_sink(
            "sink",
            CpuOnlySink {
                seen: Arc::new(AtomicU64::new(0)),
                cpu_packed: Arc::new(AtomicU64::new(0)),
            },
        );
        pipeline.link(src, snk).unwrap();
        let err = match pipeline.prepare() {
            Ok(_) => panic!("Deny policy must refuse the memory mismatch"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("memorycopy"), "error names the fix: {err}");
    }

    let seen = Arc::new(AtomicU64::new(0));
    let cpu_packed = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", ExternalTestSrc::new(4, 6));
    let snk = pipeline.add_sink(
        "sink",
        CpuOnlySink {
            seen: seen.clone(),
            cpu_packed: cpu_packed.clone(),
        },
    );
    pipeline.link(src, snk).unwrap();
    pipeline.prepare_with_auto_converters().unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    assert_eq!(seen.load(Ordering::SeqCst), 6);
    assert_eq!(
        cpu_packed.load(Ordering::SeqCst),
        6,
        "memorycopy repacked every strided frame byte-exactly"
    );
}

/// An [External, Cpu] source against an Any-caps consumer (AppSink):
/// the opt-in rule keeps External away — the link negotiates Cpu and the
/// sink receives packed frames.
#[tokio::test(flavor = "multi_thread")]
async fn any_caps_sink_never_sees_external() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        ExternalTestSrc::new(2, 5).with_cpu_fallback_cap(true),
    );
    let appsink = AppSink::with_max_buffers(8);
    let handle_sink = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, snk).unwrap();

    // Default Deny policy: must succeed WITHOUT converters.
    pipeline.prepare().unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let mut received = 0u64;
    loop {
        match handle_sink.pull_buffer().await {
            Pulled::Buffer(b) => {
                assert_eq!(
                    b.memory_type(),
                    MemoryType::Cpu,
                    "Any-caps consumers get Cpu, never External"
                );
                assert!(!b.metadata().has_strided_planes());
                assert_eq!(
                    b.as_bytes(),
                    packed_reference_frame(b.metadata().sequence % 2),
                    "packed payload"
                );
                assert_eq!(b.metadata().video_pixel_format(), Some(PixelFormat::I420));
                received += 1;
            }
            Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
            Pulled::Ended(_) => break,
        }
    }
    handle.wait().await.unwrap();
    assert_eq!(received, 5);
}
