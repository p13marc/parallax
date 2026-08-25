//! Rates and a latency distribution, beside the counters (#246).
//!
//! Every handle in [`control`](super) reports monotonic counters. A counter is
//! the right thing to difference over an arbitrary interval and the right thing
//! to hand an exporter, but it is not what a human reads: nobody wants
//! `bytes_encoded`, they want kbps. Before this module every consumer wrote its
//! own sampling loop with its own window — the zensight sensor did, the browser
//! tile would have — and three implementations of one EWMA are three chances to
//! disagree about what "encoded fps" means.
//!
//! So the window is a stated constant here, not a caller's choice.
//!
//! # What it costs on the hot path
//!
//! Nothing measurable. [`RateMeter::add`] is two relaxed `fetch_add`s and one
//! clock read, and at most once per [`RATE_WINDOW`] a `compare_exchange` and a
//! `swap`. No lock, no allocation — the same promise the rest of
//! [`control`](super) makes. The encoder call sites pay even less than that:
//! they already read the clock to measure their own encode time, and pass that
//! same `Instant` in.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The window a rate is folded over.
///
/// Also [`Queue2`](crate::elements::flow::Queue2)'s sampling cadence, and about
/// fifteen frames at 30 fps — long enough to average out per-frame size
/// variance, short enough that a bandwidth knob's effect is visible while you
/// are still looking at it.
pub const RATE_WINDOW: Duration = Duration::from_millis(500);

/// How long a rate takes to cover 63% of a step change, and 95% in three of
/// these.
///
/// `RATE_WINDOW / ln(1/(1-α))` for the private α of 0.5. Deliberately faster
/// than `Queue2`'s smoothing (τ ≈ 1.4 s): that one damps a download ETA, where
/// a jumpy number is the bug, while these feed a bandwidth-control loop and a
/// health panel, where lag is the bug.
pub const RATE_TIME_CONSTANT: Duration = Duration::from_millis(721);

/// The EWMA weight given to each closed window. Meaningless without
/// [`RATE_WINDOW`], which is why only the window and the time constant are public.
const ALPHA: f64 = 0.5;

/// A monotonic counter that also knows its own rate.
///
/// Events are added on the hot path; the rate is folded at most once per
/// [`RATE_WINDOW`] and read as a continuous-time EWMA, so a reader gets the
/// same answer whether or not a roll has just happened.
///
/// A stream that stops **decays toward zero** rather than freezing at its last
/// smoothed value — a stalled sender must read as stalled.
#[derive(Debug)]
pub struct RateMeter {
    /// Fixed reference for every timestamp below; `Instant` is not `Copy`-cheap
    /// to store atomically, and nothing ever compares two meters' epochs.
    epoch: Instant,
    total: AtomicU64,
    /// Events since the current window opened. The roll `swap(0)`s it, so
    /// closing a window and reading its contents is one operation and no event
    /// is ever counted twice or lost.
    window_accum: AtomicU64,
    /// Nanoseconds since `epoch` at which the current window opened.
    window_start_nanos: AtomicU64,
    /// The smoothed rate, as `f64::to_bits`. Always finite and non-negative.
    rate_bits: AtomicU64,
}

impl Default for RateMeter {
    fn default() -> Self {
        Self {
            epoch: Instant::now(),
            total: AtomicU64::new(0),
            window_accum: AtomicU64::new(0),
            window_start_nanos: AtomicU64::new(0),
            rate_bits: AtomicU64::new(0),
        }
    }
}

impl RateMeter {
    /// A meter with no events and a zero rate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `n` events now.
    pub fn add(&self, n: u64) {
        self.add_at(n, Instant::now());
    }

    /// Record `n` events at a given instant.
    ///
    /// Callers that already read the clock — every encoder does, to time its own
    /// encode — should pass that `Instant` rather than taking a second reading.
    pub fn add_at(&self, n: u64, at: Instant) {
        self.total.fetch_add(n, Ordering::Relaxed);
        self.window_accum.fetch_add(n, Ordering::Relaxed);

        // `Relaxed` throughout is deliberate: nothing here establishes a
        // happens-before anyone depends on, and an increment not yet visible to
        // whichever thread rolls the window lands in the *next* window rather
        // than vanishing — `fetch_add` is still atomic.
        let now = self.nanos_at(at);
        let start = self.window_start_nanos.load(Ordering::Relaxed);
        let elapsed = now.saturating_sub(start);
        if elapsed < RATE_WINDOW.as_nanos() as u64 {
            return;
        }

        // Elect exactly one roller. A loser's events are already in the
        // accumulator, so they are folded by whoever won.
        if self
            .window_start_nanos
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            let closed = self.window_accum.swap(0, Ordering::Relaxed);
            let stored = self.rate();
            self.store_rate(blend(stored, closed as f64, elapsed as f64 / 1e9));
        }
    }

    /// Events recorded since the meter was created.
    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// The smoothed rate in events per second, now.
    pub fn per_second(&self) -> f64 {
        self.per_second_at(Instant::now())
    }

    /// The smoothed rate in events per second, as of a given instant.
    ///
    /// A pure read: it stores nothing, so two calls at the same instant agree,
    /// and reading often does not change the answer.
    ///
    /// The in-progress window is folded in continuously rather than waiting for
    /// a roll, which is what makes a stopped stream decay smoothly to zero and
    /// bounds what a single burst can do to the figure.
    pub fn per_second_at(&self, at: Instant) -> f64 {
        let stored = self.rate();
        let start = self.window_start_nanos.load(Ordering::Relaxed);
        let elapsed = self.nanos_at(at).saturating_sub(start);
        if elapsed == 0 {
            return stored;
        }
        blend(
            stored,
            self.window_accum.load(Ordering::Relaxed) as f64,
            elapsed as f64 / 1e9,
        )
    }

    fn nanos_at(&self, at: Instant) -> u64 {
        at.saturating_duration_since(self.epoch).as_nanos() as u64
    }

    fn rate(&self) -> f64 {
        f64::from_bits(self.rate_bits.load(Ordering::Relaxed))
    }

    fn store_rate(&self, rate: f64) {
        let sane = if rate.is_finite() && rate >= 0.0 {
            rate
        } else {
            0.0
        };
        self.rate_bits.store(sane.to_bits(), Ordering::Relaxed);
    }
}

/// Blend `sample` into `stored` over `seconds` of elapsed time.
///
/// The continuous-time form of an EWMA: at exactly one [`RATE_WINDOW`] it is
/// the discrete step `α·sample + (1-α)·stored`, and at zero elapsed it is
/// `stored`. Being continuous is what makes a roll and a read at the same
/// instant agree, so no reader can see a step.
fn fold(stored: f64, sample: f64, seconds: f64) -> f64 {
    let weight = (1.0 - ALPHA).powf(seconds / RATE_WINDOW.as_secs_f64());
    sample + (stored - sample) * weight
}

/// Fold `events` observed over `seconds` into `stored`.
///
/// Inside one window there is nothing to infer about *when* the events landed,
/// so they are simply averaged over the elapsed time and folded continuously.
///
/// Past one window there is: a window only stays open because no event arrived
/// to roll it, and any event arriving at or after the boundary rolls it. So
/// every event still sitting in an over-long window provably landed within its
/// first [`RATE_WINDOW`], and everything after that was **silence**. Charging
/// the events to that first window and then decaying the silence is therefore
/// not a heuristic — it is what the timeline must have looked like. It is also
/// what makes a stopped stream fall off exponentially instead of trailing away
/// as `events / elapsed`, which is the difference between a health panel
/// reading zero and reading "1.6 fps" half a minute after the camera died.
///
/// Continuous at the boundary: at `seconds == RATE_WINDOW` the two branches are
/// the same expression.
fn blend(stored: f64, events: f64, seconds: f64) -> f64 {
    let window = RATE_WINDOW.as_secs_f64();
    if seconds <= window {
        return fold(stored, events / seconds, seconds);
    }
    let rolled = fold(stored, events / window, window);
    fold(rolled, 0.0, seconds - window)
}

// ---------------------------------------------------------------------------
// Latency distribution
// ---------------------------------------------------------------------------

/// Sub-buckets per octave, as a shift. Four of them bound the error at 19%.
const SUB_BITS: u32 = 2;
const SUB: usize = 1 << SUB_BITS;
/// The first bucket spans 1024 ns; finer than that is noise for an encode.
const BASE_SHIFT: u32 = 10;
const OCTAVES: usize = 24;
const BUCKETS: usize = OCTAVES * SUB;

/// A lock-free latency histogram: 96 buckets, four per octave, 768 bytes.
///
/// Plain power-of-two buckets would locate a percentile only to a factor of
/// two — "p95 is between 16 and 33 ms" does not answer whether a 33 ms frame
/// budget is being missed, which is the whole reason a single last-value sample
/// is not enough. Four sub-buckets per octave costs one shift and one mask and
/// takes that to 19%.
#[derive(Debug)]
pub struct LatencyHistogram {
    buckets: [AtomicU64; BUCKETS],
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        // `#[derive(Default)]` does not reach an array this long.
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

/// The bucket a duration falls in. Saturates at the top rather than wrapping.
const fn bucket_of(nanos: u64) -> usize {
    let units = nanos >> BASE_SHIFT;
    if units < SUB as u64 {
        // The first octave is linear, so tiny values do not all collapse into
        // bucket zero.
        return units as usize;
    }
    let octave = (63 - units.leading_zeros()) as usize;
    let sub = ((units >> (octave - SUB_BITS as usize)) as usize) & (SUB - 1);
    let index = (octave - SUB_BITS as usize + 1) * SUB + sub;
    if index >= BUCKETS { BUCKETS - 1 } else { index }
}

/// The exclusive upper bound of a bucket, in nanoseconds.
const fn bucket_upper_nanos(index: usize) -> u64 {
    if index < SUB {
        return ((index as u64) + 1) << BASE_SHIFT;
    }
    let octave = index / SUB + SUB_BITS as usize - 1;
    let units = ((SUB + index % SUB) as u64) + 1;
    (units << (octave - SUB_BITS as usize)) << BASE_SHIFT
}

impl LatencyHistogram {
    /// An empty histogram.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one observation, in nanoseconds.
    pub fn record(&self, nanos: u64) {
        self.buckets[bucket_of(nanos)].fetch_add(1, Ordering::Relaxed);
    }

    /// Summarise the distribution.
    ///
    /// One pass into a 768-byte stack array, so the count and the walk see the
    /// same numbers and nothing is allocated.
    pub fn snapshot(&self) -> EncodeLatency {
        let mut counts = [0u64; BUCKETS];
        let mut total = 0u64;
        for (slot, bucket) in counts.iter_mut().zip(self.buckets.iter()) {
            *slot = bucket.load(Ordering::Relaxed);
            total = total.saturating_add(*slot);
        }
        if total == 0 {
            return EncodeLatency::default();
        }

        let min_ns = counts
            .iter()
            .position(|&n| n > 0)
            .map_or(0, bucket_upper_nanos);
        let max_ns = counts
            .iter()
            .rposition(|&n| n > 0)
            .map_or(0, bucket_upper_nanos);

        EncodeLatency {
            count: total,
            min_ns,
            p50_ns: percentile(&counts, total, 0.50),
            p95_ns: percentile(&counts, total, 0.95),
            p99_ns: percentile(&counts, total, 0.99),
            max_ns,
        }
    }
}

/// The upper bound of the bucket holding the `fraction` quantile.
///
/// Deliberately no interpolation: guessing a position inside a log-spaced
/// bucket assumes a uniform distribution across a 1.19x span, which makes the
/// number look more precise than it is.
fn percentile(counts: &[u64; BUCKETS], total: u64, fraction: f64) -> u64 {
    let target = (total as f64 * fraction).ceil() as u64;
    let mut seen = 0u64;
    for (index, &count) in counts.iter().enumerate() {
        seen += count;
        if seen >= target.max(1) {
            return bucket_upper_nanos(index);
        }
    }
    bucket_upper_nanos(BUCKETS - 1)
}

/// A point-in-time summary of encode latency, in nanoseconds.
///
/// Percentiles are the **upper bound** of the bucket they fall in, so every
/// figure is at most 19% high and never low. All-time, not windowed: a rate
/// answers "how fast right now", a distribution answers "what does the tail
/// look like", and a tail needs history.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncodeLatency {
    /// Observations recorded.
    pub count: u64,
    /// The fastest observation, quantised up to its bucket.
    pub min_ns: u64,
    /// Median.
    pub p50_ns: u64,
    /// 95th percentile — the figure a frame budget is actually judged against.
    pub p95_ns: u64,
    /// 99th percentile.
    pub p99_ns: u64,
    /// The slowest observation, quantised up to its bucket.
    pub max_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic time, the way `AlsaPosition` does it: pass the instant in,
    /// so the arithmetic tests without sleeping.
    fn at(base: Instant, millis: u64) -> Instant {
        base + Duration::from_millis(millis)
    }

    #[test]
    fn a_meter_with_no_events_reports_a_zero_rate() {
        let meter = RateMeter::new();
        assert_eq!(meter.total(), 0);
        assert_eq!(meter.per_second(), 0.0);
    }

    #[test]
    fn the_total_counts_every_event_regardless_of_windowing() {
        let meter = RateMeter::new();
        let base = Instant::now();
        for i in 0..100 {
            meter.add_at(3, at(base, i * 40));
        }
        assert_eq!(meter.total(), 300, "the counter is exact, the rate is not");
    }

    #[test]
    fn a_steady_stream_converges_on_its_true_rate() {
        let meter = RateMeter::new();
        let base = Instant::now();
        // 30 events per second for 10 s.
        for i in 0..300u64 {
            meter.add_at(1, at(base, i * 1000 / 30));
        }
        let rate = meter.per_second_at(at(base, 10_000));
        assert!(
            (rate - 30.0).abs() < 3.0,
            "converged rate should be near 30/s, got {rate}"
        );
    }

    #[test]
    fn a_rate_change_is_tracked_within_a_few_time_constants() {
        let meter = RateMeter::new();
        let base = Instant::now();
        // 100/s for 5 s, then 10/s for 5 s.
        for i in 0..500u64 {
            meter.add_at(1, at(base, i * 10));
        }
        let fast = meter.per_second_at(at(base, 5_000));
        for i in 0..50u64 {
            meter.add_at(1, at(base, 5_000 + i * 100));
        }
        let slow = meter.per_second_at(at(base, 10_000));

        assert!((fast - 100.0).abs() < 10.0, "fast phase was {fast}");
        assert!((slow - 10.0).abs() < 3.0, "slow phase was {slow}");
    }

    #[test]
    fn a_stream_that_stops_decays_toward_zero_without_any_further_writes() {
        let meter = RateMeter::new();
        let base = Instant::now();
        for i in 0..300u64 {
            meter.add_at(1, at(base, i * 10));
        }
        let running = meter.per_second_at(at(base, 3_000));
        let after_1s = meter.per_second_at(at(base, 4_000));
        let after_5s = meter.per_second_at(at(base, 8_000));
        let after_30s = meter.per_second_at(at(base, 33_000));

        assert!(running > 50.0, "still running: {running}");
        assert!(after_1s < running, "must fall: {after_1s} vs {running}");
        assert!(after_5s < after_1s, "must keep falling: {after_5s}");
        assert!(after_30s < after_5s);
        assert!(
            after_30s < 1.0,
            "a stalled sender must read as stalled, got {after_30s}"
        );
    }

    #[test]
    fn reading_the_rate_twice_at_the_same_instant_gives_the_same_answer() {
        let meter = RateMeter::new();
        let base = Instant::now();
        for i in 0..50u64 {
            meter.add_at(7, at(base, i * 20));
        }
        let when = at(base, 1_234);
        assert_eq!(meter.per_second_at(when), meter.per_second_at(when));
        assert_eq!(meter.total(), 350, "reading must not consume");
    }

    /// Where inside a window the events landed does not move the reported rate.
    ///
    /// Naively dividing events by the span they arrived in makes a rate meter
    /// useless on a bursty producer: the same thousand frames read as 100k/s
    /// spread over 10 ms and as 1M/s packed into 1 ms. Folding the open window
    /// continuously against its own age instead makes the packing irrelevant,
    /// and bounds the whole burst's contribution at `count · ln2 / RATE_WINDOW`.
    #[test]
    fn a_burst_reads_the_same_however_tightly_it_is_packed() {
        // Both meters first, so a single base is at or after both epochs and
        // the test cannot depend on how fast the loops below happen to run.
        let loose = RateMeter::new();
        let tight = RateMeter::new();
        let base = Instant::now();

        // A thousand events spread over 10 ms...
        for i in 0..1_000u64 {
            loose.add_at(1, base + Duration::from_micros(i * 10));
        }
        // ...and the same thousand packed into 1 ms, ten times tighter.
        for i in 0..1_000u64 {
            tight.add_at(1, base + Duration::from_micros(i));
        }

        let read_at = base + Duration::from_millis(10);
        let loose_rate = loose.per_second_at(read_at);
        let tight_rate = tight.per_second_at(read_at);

        assert!(
            (loose_rate - tight_rate).abs() < 1.0,
            "packing must not move the rate: {loose_rate} vs {tight_rate}"
        );
        assert!(
            loose_rate < 2_000.0,
            "and a burst stays bounded by its size, not by 1/dt: {loose_rate}"
        );
        assert_eq!(loose.total(), 1_000, "while every event is still counted");
        assert_eq!(tight.total(), 1_000);
    }

    #[test]
    fn no_count_is_lost_when_many_threads_race_the_window_roll() {
        use std::sync::Arc;
        let meter = Arc::new(RateMeter::new());
        let base = Instant::now();

        let threads: Vec<_> = (0..8)
            .map(|_| {
                let meter = Arc::clone(&meter);
                std::thread::spawn(move || {
                    for i in 0..1_000u64 {
                        // Straddle many window boundaries so the roll is
                        // genuinely contended.
                        meter.add_at(1, at(base, i));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(meter.total(), 8_000, "the roll election must lose nothing");
    }

    #[test]
    fn every_latency_maps_into_a_bucket_whose_upper_bound_exceeds_it() {
        let mut last = 0;
        for shift in 0..34 {
            for step in 0..16u64 {
                let ns = (1u64 << shift).saturating_add(step * (1 << shift) / 16);
                let index = bucket_of(ns);
                assert!(
                    bucket_upper_nanos(index) > ns || index == BUCKETS - 1,
                    "bucket {index} upper {} does not exceed {ns}",
                    bucket_upper_nanos(index)
                );
                assert!(index >= last, "bucket_of must be monotonic");
                last = index;
            }
        }
        // The worked boundaries from the design.
        assert_eq!(bucket_of(4096), 4);
        assert_eq!(bucket_upper_nanos(4), 5120);
        assert_eq!(bucket_of(8192), 8);
        assert_eq!(bucket_upper_nanos(8), 10240);
    }

    #[test]
    fn a_percentile_is_the_top_of_its_bucket_and_never_an_interpolation() {
        let hist = LatencyHistogram::new();
        for _ in 0..100 {
            hist.record(5_000_000); // 5 ms
        }
        let snap = hist.snapshot();
        let upper = bucket_upper_nanos(bucket_of(5_000_000));
        assert_eq!(snap.p50_ns, upper);
        assert_eq!(snap.p99_ns, upper);
        assert!(
            upper > 5_000_000 && upper < 5_000_000 * 6 / 5,
            "and within 19% of the truth: {upper}"
        );
    }

    /// The argument for a distribution over `last_encode_ns`, as a test.
    #[test]
    fn one_startup_outlier_moves_the_max_but_leaves_the_p95_alone() {
        let hist = LatencyHistogram::new();
        hist.record(900_000_000); // a 900 ms first frame
        for _ in 0..999 {
            hist.record(4_000_000); // and 999 normal 4 ms ones
        }
        let snap = hist.snapshot();
        assert!(snap.max_ns > 500_000_000, "the outlier is visible");
        assert!(
            snap.p95_ns < 10_000_000,
            "but does not drown the tail figure: p95 = {}",
            snap.p95_ns
        );
        assert_eq!(snap.count, 1_000);
    }

    #[test]
    fn a_latency_past_the_top_bucket_saturates_rather_than_wrapping() {
        let hist = LatencyHistogram::new();
        hist.record(u64::MAX);
        let snap = hist.snapshot();
        assert_eq!(snap.count, 1);
        assert_eq!(snap.max_ns, bucket_upper_nanos(BUCKETS - 1));
        assert!(snap.max_ns > 30_000_000_000, "the top bucket is ~34 s");
    }

    #[test]
    fn an_empty_histogram_reports_zero_everywhere_rather_than_a_bogus_percentile() {
        assert_eq!(LatencyHistogram::new().snapshot(), EncodeLatency::default());
    }
}
