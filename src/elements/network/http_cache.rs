//! HTTP source with a local sparse cache — serve-from-disk seeks (#188).
//!
//! [`HttpCacheSrc`] fuses [`HttpSrc`](super::HttpSrc)'s ranged download with
//! download-mode `Queue2`'s sparse-cache bookkeeping into one **seekable
//! source**: everything downloaded is written through to a sparse cache file
//! at true stream offsets, and a seek into an already-downloaded span is
//! answered **from disk** — no reconnect, no re-download. This is the half
//! `Queue2` structurally cannot do (a transform cannot emit without input;
//! its backward seeks go `NotHandled` to the source and re-download).
//!
//! Semantics:
//! - Forward/hole seeks move the download cursor: the reader is dropped and
//!   the next `produce` reconnects with `Range: bytes=target-` (lazy, so a
//!   scrub storm coalesces into one connection — same as `HttpSrc`).
//! - Seeks into a downloaded span move the read cursor only; when the span
//!   runs out, the network resumes from there.
//! - Buffers are stamped with `Metadata.offset` (true stream offsets), and
//!   throttled [`MessageKind::DownloadProgress`] messages report the span
//!   map; a [`Queue2RangesHandle`] (via
//!   [`Controllable`](crate::control::Controllable), cloned before start)
//!   offers the same poll-anytime.
//!
//! The cache is an **unlinked** temp file by default (the kernel reclaims it
//! when the element drops — Linux-only crate); [`with_cache_file`] names a
//! visible path instead (always truncated at construction: a cache that
//! outlives a run would need URL/validator checks nothing implements).
//!
//! [`with_cache_file`]: HttpCacheSrc::with_cache_file
//! [`MessageKind::DownloadProgress`]: crate::pipeline::bus::MessageKind::DownloadProgress

use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::http::{HttpConnection, HttpSrc};
use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ProduceContext, ProduceResult, Source};
use crate::elements::flow::{Queue2RangesHandle, RangesShared};
use crate::error::{Error, Result};
use crate::event::{Event, EventResult, SeekEvent, SeekType, SegmentFormat};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;
use crate::pipeline::seek::{DurationQuery, PositionQuery};

/// Progress posts are throttled to this interval.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(500);

/// A seekable HTTP source with a sparse on-disk cache (#188).
///
/// See the [module docs](self) for the serve-from-disk contract.
pub struct HttpCacheSrc {
    name: String,
    url: String,
    chunk_size: usize,
    timeout: Option<Duration>,
    headers: Vec<(String, String)>,
    /// Same probe-under-lock pattern as `HttpSrc`: `is_seekable(&self)` and
    /// `query_duration(&self)` may connect; `produce` uses `get_mut`.
    conn: Mutex<HttpConnection>,
    /// Sparse cache, written at true stream offsets (`write_at`); holes are
    /// honest — `ranges` is the truth about what is local.
    cache: File,
    /// Downloaded-span map shared with [`Queue2RangesHandle`]s.
    ranges: Arc<RangesShared>,
    /// Stream offset of the next byte to emit.
    cursor: u64,
    sequence: u64,
    output: OutputArena,
    last_progress: Option<Instant>,
}

impl HttpCacheSrc {
    /// Create a cache-backed HTTP source over an unlinked temp file.
    pub fn new(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        Ok(Self {
            name: format!("httpcachesrc-{}", &url[..url.len().min(30)]),
            url,
            chunk_size: 64 * 1024,
            timeout: Some(Duration::from_secs(30)),
            headers: Vec::new(),
            conn: Mutex::new(HttpConnection::default()),
            cache: scratch_file()?,
            ranges: Arc::new(RangesShared::default()),
            cursor: 0,
            sequence: 0,
            output: OutputArena::new(defaults::SOURCE_SLOT_COUNT),
            last_progress: None,
        })
    }

    /// Use a named cache file instead of an unlinked temp file. Created (or
    /// truncated) at the given path and left on disk when the element drops.
    pub fn with_cache_file(mut self, path: impl AsRef<std::path::Path>) -> Result<Self> {
        self.cache = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(Error::Io)?;
        Ok(self)
    }

    /// Set the download/read chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(1);
        self
    }

    /// Set the request timeout (also bounds the pre-start probe).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add a custom header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// The URL this source downloads.
    pub fn url(&self) -> &str {
        &self.url
    }

    fn lock_conn(&self) -> std::sync::MutexGuard<'_, HttpConnection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Emit `n` bytes from `slot` as the buffer at the current cursor.
    fn emit(&mut self, slot: crate::memory::SharedSlotRef, n: usize) -> ProduceResult {
        let offset = self.cursor;
        self.cursor += n as u64;
        let seq = self.sequence;
        self.sequence += 1;
        let mut metadata = Metadata::from_sequence(seq);
        metadata.offset = Some(offset);
        ProduceResult::OwnBuffer(Buffer::new(MemoryHandle::with_len(slot, n), metadata))
    }

    /// Post a throttled `DownloadProgress`.
    fn post_progress(&mut self, ctx: &ProduceContext) {
        let due = self
            .last_progress
            .is_none_or(|t| t.elapsed() >= PROGRESS_INTERVAL);
        if !due {
            return;
        }
        self.last_progress = Some(Instant::now());
        let ranges = self.ranges.ranges.lock().unwrap().as_slice().to_vec();
        ctx.post_message(crate::pipeline::bus::MessageKind::DownloadProgress {
            ranges,
            total: match self.ranges.total.load(Ordering::Relaxed) {
                0 => None,
                n => Some(n),
            },
            write_pos: self.ranges.write_pos.load(Ordering::Relaxed),
        });
    }

    /// Handle a byte seek. The whole point of this element: a target inside
    /// a downloaded span moves the read cursor and touches no connection at
    /// all; anything else moves the download cursor with a lazy reconnect.
    fn handle_seek(&mut self, seek: &SeekEvent) -> EventResult {
        if seek.format != SegmentFormat::Bytes {
            return EventResult::NotHandled;
        }
        let cursor = self.cursor;
        let conn = self.conn.get_mut().unwrap_or_else(|e| e.into_inner());

        let target = match seek.start.seek_type {
            SeekType::Set => {
                if seek.start.position < 0 {
                    return EventResult::Error;
                }
                seek.start.position as u64
            }
            SeekType::Current => match cursor.checked_add_signed(seek.start.position) {
                Some(t) => t,
                None => return EventResult::Error,
            },
            SeekType::End => {
                if !conn.probed {
                    let (url, timeout, headers) = (&self.url, self.timeout, &self.headers);
                    if let Err(e) = HttpSrc::connect(url, timeout, headers, conn) {
                        tracing::warn!("httpcachesrc: probe for seek failed: {e}");
                        return EventResult::Error;
                    }
                }
                let Some(total) = conn.total_len else {
                    return EventResult::Error;
                };
                match total.checked_add_signed(seek.start.position) {
                    Some(t) => t,
                    None => return EventResult::Error,
                }
            }
            SeekType::None => return EventResult::handled(),
        };
        let target = match conn.total_len {
            Some(total) => target.min(total),
            None => target,
        };

        // Outside the cache the server must honor Range requests; inside,
        // it does not matter what the server can do.
        let cached = self.ranges.ranges.lock().unwrap().contains(target);
        if !cached {
            if !conn.probed {
                let (url, timeout, headers) = (&self.url, self.timeout, &self.headers);
                if let Err(e) = HttpSrc::connect(url, timeout, headers, conn) {
                    tracing::warn!("httpcachesrc: probe for seek failed: {e}");
                    return EventResult::Error;
                }
            }
            if !conn.seekable && target != 0 {
                tracing::warn!(
                    "httpcachesrc: server does not accept byte ranges and byte {target} \
                     is not cached; seek refused"
                );
                return EventResult::Error;
            }
        }

        // Reader repositioning is lazy: produce() drops a reader whose
        // position disagrees with the cursor, so a scrub storm coalesces
        // and a cached backward seek never disturbs a running download
        // more than necessary.
        self.cursor = target;
        EventResult::handled_at(target as i64)
    }
}

/// An unlinked temp file: reachable only through the returned handle, so the
/// kernel reclaims it when the element drops (Linux-only crate).
fn scratch_file() -> Result<File> {
    use std::sync::atomic::AtomicU64;
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "parallax-httpcache-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let file = File::options()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(Error::Io)?;
    std::fs::remove_file(&path).map_err(Error::Io)?;
    Ok(file)
}

impl Source for HttpCacheSrc {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        let conn = self.conn.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(total) = conn.total_len
            && self.cursor >= total
        {
            return Ok(ProduceResult::Eos);
        }

        // Serve from disk while the cursor sits inside a downloaded span —
        // the #188 contract. The network reader (if any) is left alone; it
        // is repositioned lazily when the span runs out.
        let run_end = self
            .ranges
            .ranges
            .lock()
            .unwrap()
            .contiguous_run_end(self.cursor);
        if let Some(end) = run_end
            && end > self.cursor
        {
            let want = (end - self.cursor).min(self.chunk_size as u64) as usize;
            let Some(mut slot) = self.output.try_acquire(self.chunk_size, "httpcachesrc")? else {
                return Ok(ProduceResult::WouldBlock);
            };
            let n = self
                .cache
                .read_at(&mut slot.data_mut()[..want], self.cursor)
                .map_err(Error::Io)?;
            if n == 0 {
                return Err(Error::Element(format!(
                    "httpcachesrc: cache file has no data at recorded offset {}",
                    self.cursor
                )));
            }
            return Ok(self.emit(slot, n));
        }

        // Network. The reader must sit exactly at the cursor: a seek or a
        // disk-served span may have moved us since the last read.
        if conn.reader.is_some() && conn.next_offset != self.cursor {
            conn.reader = None;
        }
        if conn.reader.is_none() {
            conn.next_offset = self.cursor;
            let (url, timeout, headers) = (&self.url, self.timeout, &self.headers);
            HttpSrc::connect(url, timeout, headers, conn)?;
            if let Some(total) = conn.total_len {
                self.ranges.total.store(total, Ordering::Relaxed);
                if self.cursor >= total {
                    return Ok(ProduceResult::Eos);
                }
            }
        }

        let Some(mut slot) = self.output.try_acquire(self.chunk_size, "httpcachesrc")? else {
            return Ok(ProduceResult::WouldBlock);
        };
        let reader = conn.reader.as_mut().expect("connected above");
        match reader.read(slot.data_mut()) {
            Ok(0) => {
                conn.reader = None;
                if conn.total_len.is_none() {
                    // Chunked/unknown length: EOF teaches us the total.
                    conn.total_len = Some(self.cursor);
                    self.ranges.total.store(self.cursor, Ordering::Relaxed);
                }
                Ok(ProduceResult::Eos)
            }
            Ok(n) => {
                conn.next_offset += n as u64;
                // Write-through at the true stream offset; holes stay holes.
                self.cache
                    .write_all_at(&slot.data_mut()[..n], self.cursor)
                    .map_err(Error::Io)?;
                self.ranges
                    .ranges
                    .lock()
                    .unwrap()
                    .add(self.cursor, n as u64);
                self.ranges
                    .write_pos
                    .store(self.cursor + n as u64, Ordering::Relaxed);
                self.post_progress(ctx);
                Ok(self.emit(slot, n))
            }
            Err(e) => {
                conn.reader = None;
                Err(Error::Io(e))
            }
        }
    }

    fn is_seekable(&self) -> bool {
        let mut conn = self.lock_conn();
        if !conn.probed
            && let Err(e) = HttpSrc::connect(&self.url, self.timeout, &self.headers, &mut conn)
        {
            tracing::warn!("httpcachesrc: probe failed, reporting unseekable: {e}");
            return false;
        }
        conn.seekable
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        match event {
            Event::Seek(seek) => self.handle_seek(seek),
            _ => EventResult::NotHandled,
        }
    }

    fn query_position(&self) -> Option<PositionQuery> {
        Some(PositionQuery {
            format: SegmentFormat::Bytes,
            position: Some(self.cursor),
        })
    }

    fn query_duration(&self) -> Option<DurationQuery> {
        let mut conn = self.lock_conn();
        if !conn.probed
            && let Err(e) = HttpSrc::connect(&self.url, self.timeout, &self.headers, &mut conn)
        {
            tracing::warn!("httpcachesrc: probe failed, duration unknown: {e}");
        }
        Some(DurationQuery {
            format: SegmentFormat::Bytes,
            duration: conn.total_len,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl crate::control::Controllable for HttpCacheSrc {
    type Control = Queue2RangesHandle;

    /// A poll-anytime view of download progress (ranges/total/write_pos) —
    /// the same handle shape download-mode `Queue2` hands out. Clone it
    /// *before* `executor.start()`.
    fn control(&self) -> Queue2RangesHandle {
        Queue2RangesHandle::from_shared(Arc::clone(&self.ranges))
    }
}

#[cfg(test)]
mod tests {
    use super::super::http::testserver::{ServerOpts, range_server_counted};
    use super::*;
    use std::net::SocketAddr;

    fn body_of(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn src_for(addr: SocketAddr) -> HttpCacheSrc {
        HttpCacheSrc::new(format!("http://{addr}/data"))
            .unwrap()
            .with_chunk_size(64)
            .with_timeout(Duration::from_secs(5))
    }

    /// Produce until `pred(bytes_so_far)`; panics on EOS first.
    fn drain_until(src: &mut HttpCacheSrc, out: &mut Vec<u8>, until: usize) {
        while out.len() < until {
            let mut ctx = crate::element::ProduceContext::without_buffer();
            match src.produce(&mut ctx).unwrap() {
                ProduceResult::OwnBuffer(buffer) => out.extend_from_slice(buffer.as_bytes()),
                ProduceResult::WouldBlock => std::thread::yield_now(),
                other => panic!("unexpected produce result: {other:?}"),
            }
        }
    }

    fn drain_to_eos(src: &mut HttpCacheSrc, out: &mut Vec<u8>) {
        loop {
            let mut ctx = crate::element::ProduceContext::without_buffer();
            match src.produce(&mut ctx).unwrap() {
                ProduceResult::OwnBuffer(buffer) => out.extend_from_slice(buffer.as_bytes()),
                ProduceResult::Eos => break,
                ProduceResult::WouldBlock => std::thread::yield_now(),
                other => panic!("unexpected produce result: {other:?}"),
            }
        }
    }

    /// THE #188 contract: a backward seek into a downloaded span is served
    /// from disk — bytes are correct and the request count does not move.
    #[test]
    fn backward_seek_serves_from_disk_without_a_request() {
        let body = body_of(10_000);
        let (addr, requests) = range_server_counted(body.clone(), ServerOpts::default());
        let mut src = src_for(addr);

        let mut all = Vec::new();
        drain_to_eos(&mut src, &mut all);
        assert_eq!(all, body);
        let after_download = requests.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after_download, 1, "one streaming GET downloads everything");

        // Backward into the fully-downloaded stream: disk only.
        let seek = SeekEvent::new_bytes(1_000);
        assert!(matches!(
            src.handle_upstream_event(&Event::Seek(seek)),
            EventResult::Handled { .. }
        ));
        let mut replay = Vec::new();
        drain_to_eos(&mut src, &mut replay);
        assert_eq!(replay, body[1_000..], "replayed bytes match the stream");
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            after_download,
            "serve-from-disk must not touch the network"
        );
    }

    /// A forward seek past the download cursor reconnects at the target and
    /// leaves an honest hole in the span map; playing on from a mid-span
    /// position serves disk first, then resumes the network exactly at the
    /// span's end.
    #[test]
    fn forward_seek_reconnects_and_leaves_an_honest_hole() {
        let body = body_of(10_000);
        let (addr, requests) = range_server_counted(body.clone(), ServerOpts::default());
        let mut src = src_for(addr);
        let handle = crate::control::Controllable::control(&src);

        let mut head = Vec::new();
        drain_until(&mut src, &mut head, 1_024);
        assert_eq!(head, body[..head.len()]);
        let downloaded = head.len() as u64;

        // Jump over a hole.
        let seek = SeekEvent::new_bytes(5_000);
        assert!(matches!(
            src.handle_upstream_event(&Event::Seek(seek)),
            EventResult::Handled { .. }
        ));
        let mut tail = Vec::new();
        drain_to_eos(&mut src, &mut tail);
        assert_eq!(tail, body[5_000..]);
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the hole forced exactly one reconnect"
        );
        let ranges = handle.ranges();
        assert_eq!(
            ranges,
            vec![(0, downloaded), (5_000, 10_000)],
            "two spans, one honest hole"
        );

        // Back into the first span: disk serves it, and when the span runs
        // out at `downloaded`, the network resumes there (request 3).
        let seek = SeekEvent::new_bytes(0);
        assert!(matches!(
            src.handle_upstream_event(&Event::Seek(seek)),
            EventResult::Handled { .. }
        ));
        let mut replay = Vec::new();
        drain_until(&mut src, &mut replay, downloaded as usize + 512);
        assert_eq!(replay, body[..replay.len()]);
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "disk first, then one resume request at the span end"
        );
    }

    /// End-relative and past-the-end seeks behave like HttpSrc's: End
    /// resolves against the probed total, past-the-end clamps and EOSes.
    #[test]
    fn end_relative_seek_and_clamp() {
        let body = body_of(2_000);
        let (addr, _requests) = range_server_counted(body.clone(), ServerOpts::default());
        let mut src = src_for(addr);

        let seek = SeekEvent::new(SegmentFormat::Bytes, crate::event::SeekPosition::end(-100));
        match src.handle_upstream_event(&Event::Seek(seek)) {
            EventResult::Handled { position } => assert_eq!(position, Some(1_900)),
            other => panic!("end-relative seek failed: {other:?}"),
        }
        let mut tail = Vec::new();
        drain_to_eos(&mut src, &mut tail);
        assert_eq!(tail, body[1_900..]);

        let seek = SeekEvent::new_bytes(1_000_000);
        match src.handle_upstream_event(&Event::Seek(seek)) {
            EventResult::Handled { position } => assert_eq!(position, Some(2_000)),
            other => panic!("past-the-end seek failed: {other:?}"),
        }
        let mut ctx = crate::element::ProduceContext::without_buffer();
        assert!(matches!(src.produce(&mut ctx).unwrap(), ProduceResult::Eos));
    }

    /// A cached span is servable even when the server refused ranges — but
    /// an uncached target is honestly refused.
    #[test]
    fn rangeless_server_still_replays_the_cache() {
        let body = body_of(3_000);
        let opts = ServerOpts {
            support_range: false,
            advertise_accept_ranges: false,
            ..Default::default()
        };
        let (addr, requests) = range_server_counted(body.clone(), opts);
        let mut src = src_for(addr);

        let mut all = Vec::new();
        drain_to_eos(&mut src, &mut all);
        assert_eq!(all, body);

        // Cached: served regardless of server capability.
        let seek = SeekEvent::new_bytes(500);
        assert!(matches!(
            src.handle_upstream_event(&Event::Seek(seek)),
            EventResult::Handled { .. }
        ));
        let mut replay = Vec::new();
        drain_to_eos(&mut src, &mut replay);
        assert_eq!(replay, body[500..]);
        assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// Full pipeline: a runtime backward byte seek is absorbed by the cache
    /// — SeekDone posts, playback restarts at the target from disk, and the
    /// request count does not move while replaying cached data.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pipeline_backward_seek_replays_from_cache() {
        use crate::elements::{AppSink, Pulled};
        use crate::pipeline::bus::MessageKind;
        use crate::pipeline::{Executor, LinkPolicy, Pipeline};

        let body = body_of(50_000);
        let (addr, requests) = range_server_counted(body.clone(), ServerOpts::default());

        let mut pipeline = Pipeline::new();
        let src = pipeline.add_source(
            "src",
            HttpCacheSrc::new(format!("http://{addr}/data"))
                .unwrap()
                .with_chunk_size(256)
                .with_timeout(Duration::from_secs(5)),
        );
        let sink = AppSink::with_max_buffers(4);
        let sink_handle = sink.handle();
        let snk = pipeline.add_async_sink("sink", sink);
        pipeline
            .link_pads_full(src, "src", snk, "sink", LinkPolicy::Block, Some(2))
            .unwrap();

        let executor = Executor::new();
        let mut handle = executor.start(&mut pipeline).unwrap();
        let mut bus = handle.take_bus().unwrap();

        // Pull a few KB so a prefix span exists.
        let mut pulled = 0usize;
        while pulled < 4_096 {
            match sink_handle.pull_buffer().await {
                Pulled::Buffer(b) => pulled += b.len(),
                Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
                other => panic!("stream ended early: {other:?}"),
            }
        }
        let before = requests.load(std::sync::atomic::Ordering::SeqCst);

        assert!(handle.seek_bytes(0).await, "the seek was dispatched");

        // The first post-seek buffer restarts at offset 0 with the right
        // bytes; pre-seek stragglers are recognizable by their offsets.
        let mut restarted = None;
        for _ in 0..1_000 {
            match sink_handle.pull_buffer().await {
                Pulled::Buffer(b) => {
                    if b.metadata().offset == Some(0) {
                        restarted = Some(b.as_bytes().to_vec());
                        break;
                    }
                }
                Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
                other => panic!("stream ended before the replay: {other:?}"),
            }
        }
        let restarted = restarted.expect("a buffer restarting at offset 0");
        assert_eq!(restarted, body[..restarted.len()]);
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            before,
            "the backward seek was served from the cache"
        );

        handle.stop();
        loop {
            match sink_handle.pull_buffer().await {
                Pulled::Buffer(_) => {}
                Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
                _ => break,
            }
        }
        handle.wait().await.unwrap();

        let mut seek_done = false;
        while let Some(msg) = bus.poll() {
            if let MessageKind::SeekDone { position, .. } = msg.kind {
                seek_done = true;
                assert_eq!(position, Some(0));
            }
        }
        assert!(seek_done, "SeekDone posted");
    }
}
