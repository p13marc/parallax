//! Regression tests for #92: a debugging aid must not be able to take down the
//! graph it is observing.
//!
//! Probe and tracer callbacks run inline on the data path, inside whichever
//! element task happens to invoke them. A panicking one therefore unwound out
//! through that task, killing the element and every sink below it — and naming
//! that element as the culprit, when it had done nothing wrong. #85 made the
//! failure visible instead of silently orphaning the pipeline; this makes it not
//! a failure at all.
//!
//! The offender is removed rather than retried: retrying re-panics on every
//! buffer, and removal leaves the pipeline behaving exactly as it did before the
//! aid was attached.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parallax::buffer::Buffer;
use parallax::elements::{NullSink, NullSource, PassThrough};
use parallax::pipeline::tracer::{Tracer, TracerRegistry};
use parallax::pipeline::{
    EndReason, Executor, PadRef, Pipeline, ProbeData, ProbeReturn, ProbeType,
};

const LIMIT: Duration = Duration::from_secs(10);

/// Panics on the first call it is given.
struct PanickingTracer {
    calls: Arc<AtomicU64>,
}

impl Tracer for PanickingTracer {
    fn on_buffer(&self, _element: &str, _buffer: &Buffer, _ts: Instant) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        panic!("a tracer fell over");
    }

    fn name(&self) -> &str {
        "panicking"
    }
}

/// Counts the buffers it sees, so a test can prove the *other* observers on the
/// same registry kept working after one of them was removed.
struct CountingTracer {
    seen: Arc<AtomicU64>,
}

impl Tracer for CountingTracer {
    fn on_buffer(&self, _element: &str, _buffer: &Buffer, _ts: Instant) {
        self.seen.fetch_add(1, Ordering::Relaxed);
    }

    fn name(&self) -> &str {
        "counting"
    }
}

fn five_buffer_pipeline() -> (Pipeline, parallax::pipeline::NodeId) {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let xfm = pipeline.add_transform("xfm", PassThrough::new());
    let snk = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();
    (pipeline, src)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panicking_probe_does_not_end_the_pipeline() {
    let (mut pipeline, src) = five_buffer_pipeline();
    let _probe = pipeline.add_probe(PadRef::src(src), ProbeType::BUFFER, |_| {
        panic!("a probe fell over");
    });

    let handle = Executor::new().start(&mut pipeline).unwrap();
    let ended = handle.ended();

    tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("a panicking probe wedged the pipeline")
        .expect("a panicking probe failed the pipeline");

    assert_eq!(
        tokio::time::timeout(LIMIT, ended).await.unwrap(),
        EndReason::Eos,
        "a panicking probe must not change how the run ends"
    );
}

/// The offender is gone, not merely survived — otherwise every subsequent buffer
/// pays for another panic and another `catch_unwind`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panicking_probe_is_removed() {
    let calls = Arc::new(AtomicU64::new(0));

    let (mut pipeline, src) = five_buffer_pipeline();
    let counter = calls.clone();
    let _probe = pipeline.add_probe(PadRef::src(src), ProbeType::BUFFER, move |_| {
        counter.fetch_add(1, Ordering::Relaxed);
        panic!("a probe fell over");
    });
    assert_eq!(pipeline.probe_registry().len(), 1);

    let handle = Executor::new().start(&mut pipeline).unwrap();
    tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("the pipeline never finished")
        .unwrap();

    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "the panicking probe was invoked more than once"
    );
    assert_eq!(
        pipeline.probe_registry().len(),
        0,
        "the panicking probe was left registered"
    );
}

/// One broken probe must not disable the pad's other probes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_surviving_probe_on_the_same_pad_keeps_running() {
    let seen = Arc::new(AtomicU64::new(0));

    let (mut pipeline, src) = five_buffer_pipeline();
    let _bad = pipeline.add_probe(PadRef::src(src), ProbeType::BUFFER, |_| {
        panic!("a probe fell over");
    });
    let counter = seen.clone();
    let _good = pipeline.add_probe(PadRef::src(src), ProbeType::BUFFER, move |data| {
        if matches!(data, ProbeData::Buffer(_)) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        ProbeReturn::Ok
    });

    let handle = Executor::new().start(&mut pipeline).unwrap();
    tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("the pipeline never finished")
        .unwrap();

    assert_eq!(
        seen.load(Ordering::Relaxed),
        5,
        "the healthy probe stopped seeing buffers"
    );
    assert_eq!(pipeline.probe_registry().len(), 1, "the wrong probe went");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panicking_tracer_does_not_end_the_pipeline() {
    let calls = Arc::new(AtomicU64::new(0));
    let seen = Arc::new(AtomicU64::new(0));

    let tracers = TracerRegistry::new();
    tracers.add(Box::new(PanickingTracer {
        calls: calls.clone(),
    }));
    tracers.add(Box::new(CountingTracer { seen: seen.clone() }));

    let (mut pipeline, _src) = five_buffer_pipeline();
    pipeline.set_tracer_registry(tracers.clone());

    let handle = Executor::new().start(&mut pipeline).unwrap();
    let ended = handle.ended();

    tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("a panicking tracer wedged the pipeline")
        .expect("a panicking tracer failed the pipeline");

    assert_eq!(
        tokio::time::timeout(LIMIT, ended).await.unwrap(),
        EndReason::Eos
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "the panicking tracer was invoked more than once"
    );
    assert_eq!(tracers.len(), 1, "the panicking tracer was left registered");
    assert!(
        seen.load(Ordering::Relaxed) > 0,
        "the healthy tracer stopped seeing buffers"
    );
}

/// Reports are collected at shutdown, so a panic there would lose every *other*
/// tracer's report as well as taking down the caller.
#[test]
fn a_tracer_that_panics_while_reporting_is_skipped() {
    struct BadReport;
    impl Tracer for BadReport {
        fn report(&self) -> Option<String> {
            panic!("report fell over");
        }
        fn name(&self) -> &str {
            "bad-report"
        }
    }

    struct GoodReport;
    impl Tracer for GoodReport {
        fn report(&self) -> Option<String> {
            Some("all fine".into())
        }
        fn name(&self) -> &str {
            "good-report"
        }
    }

    let tracers = TracerRegistry::new();
    tracers.add(Box::new(BadReport));
    tracers.add(Box::new(GoodReport));

    let reports = tracers.reports();
    assert_eq!(reports, vec![("good-report".into(), "all fine".into())]);
}
