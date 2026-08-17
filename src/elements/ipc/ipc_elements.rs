//! IPC elements for cross-process pipelines.
//!
//! Zero-copy transfer between processes: payloads stay in shared-memory
//! arenas; per-buffer descriptors and acks ride a shared-memory SPSC ring
//! pair with eventfd doorbells ([`IpcChannel`], #179); the Unix socket is
//! the control plane only — registration (with fds via SCM_RIGHTS),
//! overflow metadata, teardown.
//!
//! ```text
//! Process A                                 Process B
//! ┌─────────┐    control socket            ┌─────────┐
//! │ IpcSink │ ────────────────────────────▶│ IpcSrc  │  RegisterChannel(+3 fds),
//! └────┬────┘                              └────┬────┘  RegisterArena(+1 fd),
//!      │   descriptor ring ───────────────────▶ │       MetaOverflow, Shutdown
//!      │ ◀─────────────────────────── ack ring  │
//!      └────────── shared memory arenas ────────┘
//!                  (payloads, zero-copy)
//! ```
//!
//! The pin protocol (#177): the sink holds a live `Buffer` clone for every
//! descriptor in flight — `slot_from_ipc` refuses to resurrect a released
//! slot, so the pin may only drop once the src's ack (sent *after* mapping)
//! comes back through the ack ring. In-flight is bounded at the ring
//! capacity, which is what makes both rings never-full by construction.

use super::protocol::{
    ControlMessage, KNOWN_CUSTOM_KEYS, MAX_CONTROL_MESSAGE_SIZE, frame_message, unframe_message,
};
use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{AsyncSink, AsyncSource, ConsumeContext, ProduceContext, ProduceResult};
use crate::error::{Error, Result};
use crate::event::Event;
use crate::format::Caps;
use crate::memory::ipc::{recv_fds_nonblocking, send_fds};
use crate::memory::{DEFAULT_IPC_RING_CAPACITY, IpcChannel, SharedArena};
use rustix::fd::OwnedFd;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Accept one connection without blocking; `Ok(None)` = nobody waiting.
///
/// A blocking `accept()` inside an element parks a tokio worker until the
/// peer shows up — which for an IPC pipeline may be never, and a handful of
/// those takes the whole runtime down (#172).
fn accept_nonblocking(listener: &UnixListener) -> Result<Option<UnixStream>> {
    listener.set_nonblocking(true).ok();
    let accepted = listener.accept();
    listener.set_nonblocking(false).ok();
    match accepted {
        Ok((socket, _addr)) => {
            socket.set_nonblocking(false).ok();
            Ok(Some(socket))
        }
        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
        Err(e) => Err(Error::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to accept IPC connection: {}", e),
        ))),
    }
}

/// Connect without failing on a peer that hasn't bound yet.
///
/// `Ok(None)` on ENOENT/ECONNREFUSED, so a client-mode element started
/// before its server-mode peer retries instead of erroring — start order
/// independence.
fn connect_nonfatal(path: &Path) -> Result<Option<UnixStream>> {
    match UnixStream::connect(path) {
        Ok(socket) => Ok(Some(socket)),
        Err(ref e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
            ) =>
        {
            Ok(None)
        }
        Err(e) => Err(Error::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to connect to IPC socket at {:?}: {}", path, e),
        ))),
    }
}

/// How long a sink waits on a silent peer before saying so.
const STALL_WARN_AFTER: Duration = Duration::from_secs(5);

/// Extract the custom-map entries that can cross the IPC boundary.
fn overflow_entries(meta: &crate::metadata::Metadata) -> Vec<(String, Vec<u8>)> {
    if meta.custom_is_empty() {
        return Vec::new();
    }
    KNOWN_CUSTOM_KEYS
        .iter()
        .filter_map(|key| {
            meta.get_bytes(key)
                .map(|b| ((*key).to_string(), b.to_vec()))
        })
        .collect()
}

// ============================================================================
// IpcSink
// ============================================================================

/// State shared between the sink element and its ack-reaper task.
struct SinkShared {
    channel: IpcChannel,
    /// In-flight pins: `(seq, buffer)` FIFO, popped as acks return. Each
    /// live `Buffer` clone keeps its slot's refcount above zero until the
    /// peer has mapped it (#177).
    pending: std::sync::Mutex<VecDeque<(u64, Buffer)>>,
    /// Signaled by the reaper after releasing pins; `consume`'s
    /// backpressure wait parks here (never on the ack doorbell — the
    /// reaper is that eventfd's single consumer, and two waiters on one
    /// eventfd steal each other's wakeups).
    reaped: tokio::sync::Notify,
}

/// The standing ack reaper: releases pins as acks arrive, independent of
/// `consume` being called.
///
/// This independence is load-bearing, not a nicety. Reaping only inside
/// `consume` deadlocks any pipeline whose source arena is no larger than
/// the in-flight window: all slots end up pinned → the source cannot
/// produce → `consume` is never called → the pins are never released. The
/// reaper breaks that cycle.
///
/// It is also the teardown path: once the channel goes terminal
/// (EOS/Error, set by `handle_downstream_event` or `Drop`), the reaper
/// keeps the pins alive until the peer has mapped everything — a pin
/// dropped early turns the peer's mapping into a refused stale ref
/// (#177) — then exits, bounded by a 5 s no-progress grace.
async fn sink_ack_reaper(shared: std::sync::Arc<SinkShared>) {
    let mut terminal_deadline: Option<tokio::time::Instant> = None;
    loop {
        let mut reaped_any = false;
        {
            let mut pending = shared.pending.lock().unwrap();
            while let Some(seq) = shared.channel.try_pop_ack() {
                match pending.pop_front() {
                    Some((expected, _buffer)) if expected == seq => reaped_any = true,
                    other => {
                        tracing::error!(
                            "ipcsink: ack {seq} does not match oldest in-flight {:?} — \
                             abandoning the ack stream",
                            other.map(|(s, _)| s)
                        );
                        return;
                    }
                }
            }
        }
        if reaped_any {
            shared.reaped.notify_waiters();
            terminal_deadline = None; // progress resets the teardown grace
        }

        let pending_empty = shared.pending.lock().unwrap().is_empty();
        if shared.channel.state() != crate::memory::IpcChannelState::Active {
            if pending_empty {
                return;
            }
            let deadline = *terminal_deadline
                .get_or_insert_with(|| tokio::time::Instant::now() + Duration::from_secs(5));
            if tokio::time::Instant::now() >= deadline {
                let left = shared.pending.lock().unwrap().len();
                tracing::warn!(
                    "ipcsink: dropping {left} unmapped in-flight buffers — peer stopped acking"
                );
                return;
            }
        }

        // Bounded so terminal-state transitions (which ring the *data*
        // doorbell, not this one) are noticed promptly.
        let _ = tokio::time::timeout(
            Duration::from_millis(500),
            shared.channel.ack_doorbell().wait_async(),
        )
        .await;
    }
}

/// IPC sink that sends buffers to another process (#179).
///
/// Publishes one 128-byte descriptor per buffer into the shared-memory data
/// ring — zero allocations, zero serialization in the common case — and
/// pins the buffer until the peer's ack returns through the ack ring.
/// Buffer arenas are registered on first sight: the first descriptor
/// referencing an arena is preceded by a `RegisterArena` + fd on the
/// control socket, so the sink forwards buffers from *any* upstream arena
/// (the old code registered a placeholder arena of its own and shipped
/// slot refs the peer could never resolve).
pub struct IpcSink {
    /// Path to the Unix socket.
    path: PathBuf,
    /// Connected control socket (write-only after the handshake).
    socket: Option<UnixStream>,
    /// Whether we're the server (created the socket).
    is_server: bool,
    /// Listener for incoming connections (server mode).
    listener: Option<UnixListener>,
    /// Channel + pin table, shared with the reaper task; set at connect.
    shared: Option<std::sync::Arc<SinkShared>>,
    /// Arena ids already sent to the peer.
    registered_arenas: HashSet<u64>,
    /// Next descriptor seq.
    next_seq: u64,
    /// Ring capacity == in-flight bound. Replaces the old `max_pending`.
    capacity: u32,
    /// Capabilities.
    caps: Caps,
}

impl IpcSink {
    /// Create a new IPC sink (server mode: binds the socket).
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            socket: None,
            is_server: true,
            listener: None,
            shared: None,
            registered_arenas: HashSet::new(),
            next_seq: 0,
            capacity: DEFAULT_IPC_RING_CAPACITY,
            caps: Caps::any(),
        }
    }

    /// Create a sink that connects to an existing socket (client mode).
    pub fn connect(path: impl AsRef<Path>) -> Self {
        let mut sink = Self::new(path);
        sink.is_server = false;
        sink
    }

    /// Set the ring capacity (power of two), which is also the in-flight
    /// buffer bound. Replaces the pre-#179 `with_max_pending`.
    pub fn with_capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// Set capabilities.
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = caps;
        self
    }

    /// Initialize the connection and the ring channel.
    ///
    /// `Ok(false)` = no peer yet; the caller re-polls rather than blocking.
    fn ensure_connected(&mut self) -> Result<bool> {
        if self.socket.is_some() {
            return Ok(true);
        }

        let socket = if self.is_server {
            if self.listener.is_none() {
                let _ = std::fs::remove_file(&self.path);
                self.listener = Some(UnixListener::bind(&self.path).map_err(|e| {
                    Error::Io(std::io::Error::new(
                        e.kind(),
                        format!("Failed to bind IPC socket at {:?}: {}", self.path, e),
                    ))
                })?);
            }
            let Some(socket) = accept_nonblocking(self.listener.as_ref().unwrap())? else {
                return Ok(false);
            };
            socket
        } else {
            let Some(socket) = connect_nonfatal(&self.path)? else {
                return Ok(false);
            };
            socket
        };

        // Build the channel and hand its three fds over in the handshake.
        let channel = IpcChannel::create(self.capacity)?;
        let msg = frame_message(&ControlMessage::RegisterChannel {
            capacity: self.capacity,
        });
        send_fds(&socket, &channel.fds(), &msg)?;

        let shared = std::sync::Arc::new(SinkShared {
            channel,
            pending: std::sync::Mutex::new(VecDeque::new()),
            reaped: tokio::sync::Notify::new(),
        });
        // consume() runs on the runtime, so the reaper can spawn here.
        tokio::spawn(sink_ack_reaper(shared.clone()));

        self.socket = Some(socket);
        self.shared = Some(shared);
        Ok(true)
    }

    /// Send a control message (no fds).
    fn send_message(&mut self, msg: &ControlMessage) -> Result<()> {
        let socket = self
            .socket
            .as_mut()
            .ok_or_else(|| Error::Element("Not connected".into()))?;
        socket.write_all(&frame_message(msg)).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to send IPC message: {}", e),
            ))
        })?;
        Ok(())
    }

    /// One poll interval of an unbounded wait, warning once it gets long.
    async fn stall_tick(since: &mut Option<Instant>, what: &str) {
        match since {
            Some(start) => {
                if start.elapsed() >= STALL_WARN_AFTER {
                    tracing::warn!(
                        "ipcsink: waiting {:.0}s — {what}",
                        start.elapsed().as_secs_f32()
                    );
                    *start = Instant::now();
                }
            }
            None => *since = Some(Instant::now()),
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

impl AsyncSink for IpcSink {
    /// The descriptor ring speaks arena identity, so only CPU (arena)
    /// buffers can cross (#145) — pinning it here turns a dmabuf link into
    /// a prepare()-time converter insertion instead of a runtime error.
    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        crate::format::ElementMediaCaps::new(vec![crate::format::FormatMemoryCap::new(
            crate::format::FormatCaps::Any,
            crate::format::MemoryCaps::cpu_only(),
        )])
    }

    /// Async because both waits are unbounded: a peer that never connects,
    /// and a peer that stops acknowledging (#172). The ack wait parks on
    /// the ack doorbell (cancel-safe), timeout-sliced to keep the 5 s
    /// stall warning.
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let mut waiting_since: Option<Instant> = None;
        while !self.ensure_connected()? {
            Self::stall_tick(&mut waiting_since, "no peer has connected").await;
        }
        let shared = self.shared.as_ref().unwrap().clone();

        // Backpressure: in-flight at capacity means every ring slot is
        // spoken for; wait for the reaper to release pins. Bounded slices
        // so a lost notify race costs 100 ms, not forever, and the 5 s
        // stall warning survives.
        let mut waiting_since: Option<Instant> = None;
        while shared.pending.lock().unwrap().len() >= self.capacity as usize {
            match waiting_since {
                Some(start) if start.elapsed() >= STALL_WARN_AFTER => {
                    tracing::warn!("ipcsink: waiting — peer is not acknowledging buffers");
                    waiting_since = Some(Instant::now());
                }
                None => waiting_since = Some(Instant::now()),
                _ => {}
            }
            let _ =
                tokio::time::timeout(Duration::from_millis(100), shared.reaped.notified()).await;
        }

        let buffer = ctx.buffer();
        // Packed is the IPC wire invariant (#194): strided layouts never
        // cross; nothing in-tree produces a strided Cpu buffer.
        debug_assert!(
            !buffer.metadata().has_strided_planes(),
            "ipcsink: strided plane layout on an IPC-bound buffer"
        );
        // The descriptor ring speaks arena identity — a dmabuf- or
        // external-backed buffer has none (#145/#194). Per-buffer
        // SCM_RIGHTS fd passing is the honest follow-up for dmabuf; until
        // then, negotiate Cpu or insert `memorycopy`.
        let (Some(slot), Some(ipc_ref)) = (buffer.memory().slot(), buffer.memory().ipc_ref())
        else {
            return Err(Error::Element(
                "ipcsink: dmabuf- or external-backed buffer cannot cross the descriptor \
                 ring; negotiate Cpu memory upstream or insert a memorycopy"
                    .into(),
            ));
        };

        // Register-on-first-sight: the descriptor names this arena, so its
        // fd must be with the peer before the descriptor is (socket FIFO +
        // ring publish = happens-before).
        let arena_id = slot.arena_id();
        if !self.registered_arenas.contains(&arena_id) {
            let msg = frame_message(&ControlMessage::RegisterArena { arena_id });
            let fd = unsafe { rustix::fd::BorrowedFd::borrow_raw(slot.arena_fd()) };
            let socket = self
                .socket
                .as_ref()
                .ok_or_else(|| Error::Element("Not connected".into()))?;
            send_fds(socket, &[fd], &msg)?;
            self.registered_arenas.insert(arena_id);
        }

        let seq = self.next_seq;
        self.next_seq += 1;

        // Rare custom metadata overflows through the socket, before the
        // descriptor that references it.
        let overflow = overflow_entries(buffer.metadata());
        let mut desc = crate::memory::IpcDescriptor::encode(seq, &ipc_ref, buffer.metadata());
        if !overflow.is_empty() {
            self.send_message(&ControlMessage::MetaOverflow {
                seq,
                entries: overflow,
            })?;
            desc.set_meta_overflow();
        }

        // Pin BEFORE publishing: once the descriptor is visible the peer
        // can map and ack it, and the concurrent reaper must find the pin
        // recorded (#177 plus the reaper's FIFO assertion).
        shared
            .pending
            .lock()
            .unwrap()
            .push_back((seq, buffer.clone()));

        // By the in-flight bound this can never be full — see the ring's
        // never-full invariant.
        if !shared.channel.try_push_desc(desc) {
            shared.pending.lock().unwrap().pop_back();
            return Err(Error::Element(
                "ipcsink: descriptor ring full despite in-flight accounting".into(),
            ));
        }
        Ok(())
    }

    fn input_caps(&self) -> Caps {
        self.caps.clone()
    }

    /// Terminal events move the ring to its terminal state so the peer's
    /// `IpcSrc` ends cleanly (EOS rides the shm state word, not the
    /// socket). The standing reaper sees the state change, keeps the
    /// in-flight pins alive until the peer has mapped them, then exits.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        match &event {
            Event::Eos => {
                if let Some(shared) = &self.shared {
                    shared.channel.set_eos();
                }
            }
            Event::Error(_) => {
                if let Some(shared) = &self.shared {
                    shared.channel.set_error();
                }
            }
            _ => {}
        }
        Some(event)
    }
}

impl Drop for IpcSink {
    fn drop(&mut self) {
        // Idempotent (first transition wins); covers abort-style teardown
        // where no terminal event was delivered. The reaper task holds its
        // own Arc to the shared state, so the pins survive this drop until
        // the peer has mapped everything or the reaper's grace expires —
        // never block here: a blocked worker's LIFO slot is non-stealable
        // (gotcha 15), and an earlier draft that drained synchronously in
        // Drop parked the *receiver's* task and manufactured the very
        // stale refs the drain existed to prevent.
        if let Some(shared) = &self.shared {
            shared.channel.set_eos();
        }
        if self.socket.is_some() {
            let _ = self.send_message(&ControlMessage::Shutdown);
        }
        if self.is_server {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

// ============================================================================
// IpcSrc
// ============================================================================

/// IPC source that receives buffers from another process (#179).
///
/// An [`AsyncSource`] — register with
/// [`Pipeline::add_async_source`](crate::pipeline::Pipeline::add_async_source).
/// Waits on the data doorbell instead of blocking a worker in a socket
/// read (the pre-#179 sync `Source` did exactly that once connected).
pub struct IpcSrc {
    /// Path to the Unix socket.
    path: PathBuf,
    /// Connected control socket (read via non-blocking recvmsg only —
    /// a plain `read` would discard SCM_RIGHTS fds).
    socket: Option<UnixStream>,
    /// Whether we're the server (created the socket).
    is_server: bool,
    /// Listener for incoming connections (server mode).
    listener: Option<UnixListener>,
    /// The ring channel, rebuilt from the handshake fds.
    channel: Option<IpcChannel>,
    /// Mapped arenas by id.
    arena_cache: HashMap<u64, SharedArena>,
    /// Overflow metadata waiting for its descriptor, by seq.
    meta_overflow: HashMap<u64, Vec<(String, Vec<u8>)>>,
    /// Bytes received but not yet parsed into a frame.
    ctrl_buf: Vec<u8>,
    /// Fd batches received but not yet claimed by a registration message.
    ///
    /// SOCK_STREAM can coalesce frames into one recvmsg, but the kernel
    /// never merges two SCM_RIGHTS blocks into one read — each batch was
    /// sent with exactly one fd-carrying message, and those messages are
    /// parsed in order, so FIFO matching is exact.
    pending_fds: VecDeque<Vec<OwnedFd>>,
    /// Parsed control messages not yet consumed.
    pending_ctrl: VecDeque<ControlMessage>,
    /// The peer closed the socket.
    peer_eof: bool,
    /// Capabilities.
    caps: Caps,
}

impl IpcSrc {
    /// Create a new IPC source (client mode: connects to the socket).
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            socket: None,
            is_server: false,
            listener: None,
            channel: None,
            arena_cache: HashMap::new(),
            meta_overflow: HashMap::new(),
            ctrl_buf: Vec::new(),
            pending_fds: VecDeque::new(),
            pending_ctrl: VecDeque::new(),
            peer_eof: false,
            caps: Caps::any(),
        }
    }

    /// Create a source that listens for connections (server mode).
    pub fn listen(path: impl AsRef<Path>) -> Self {
        let mut src = Self::new(path);
        src.is_server = true;
        src
    }

    /// Set capabilities.
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = caps;
        self
    }

    /// `Ok(false)` = no peer yet (`produce` turns it into `WouldBlock`).
    fn ensure_connected(&mut self) -> Result<bool> {
        if self.socket.is_some() {
            return Ok(true);
        }

        let socket = if self.is_server {
            if self.listener.is_none() {
                let _ = std::fs::remove_file(&self.path);
                self.listener = Some(UnixListener::bind(&self.path).map_err(|e| {
                    Error::Io(std::io::Error::new(
                        e.kind(),
                        format!("Failed to bind IPC socket at {:?}: {}", self.path, e),
                    ))
                })?);
            }
            let Some(socket) = accept_nonblocking(self.listener.as_ref().unwrap())? else {
                return Ok(false);
            };
            socket
        } else {
            let Some(socket) = connect_nonfatal(&self.path)? else {
                return Ok(false);
            };
            socket
        };

        self.socket = Some(socket);
        Ok(true)
    }

    /// Pull whatever the control socket holds into the parsed queues.
    /// Non-blocking; sets `peer_eof` on a closed socket.
    fn pump_control(&mut self) -> Result<()> {
        let Some(socket) = &self.socket else {
            return Ok(());
        };
        // Large enough for several coalesced frames; bounded regardless of
        // the peer's length prefixes.
        let mut buf = vec![0u8; MAX_CONTROL_MESSAGE_SIZE.min(16 * 1024)];
        loop {
            match recv_fds_nonblocking(socket, &mut buf)? {
                None => break,
                Some((0, _)) => {
                    self.peer_eof = true;
                    break;
                }
                Some((n, fds)) => {
                    if !fds.is_empty() {
                        self.pending_fds.push_back(fds);
                    }
                    self.ctrl_buf.extend_from_slice(&buf[..n]);
                }
            }
        }
        // Parse every complete frame out of the accumulator.
        let mut consumed = 0;
        while let Some((msg, used)) = unframe_message(&self.ctrl_buf[consumed..])? {
            self.pending_ctrl.push_back(msg);
            consumed += used;
        }
        if consumed > 0 {
            self.ctrl_buf.drain(..consumed);
        }
        Ok(())
    }

    /// Apply one parsed control message.
    fn handle_control(&mut self, msg: ControlMessage) -> Result<()> {
        match msg {
            ControlMessage::RegisterChannel { capacity } => {
                let fds = self
                    .pending_fds
                    .pop_front()
                    .ok_or_else(|| Error::Element("RegisterChannel without fds".into()))?;
                let mut it = fds.into_iter();
                let (Some(ring), Some(data_db), Some(ack_db)) = (it.next(), it.next(), it.next())
                else {
                    return Err(Error::Element(
                        "RegisterChannel needs [ring, data doorbell, ack doorbell] fds".into(),
                    ));
                };
                let channel = unsafe { IpcChannel::from_fds(ring, data_db, ack_db)? };
                if channel.capacity() != capacity {
                    return Err(Error::Element(format!(
                        "ipc ring capacity mismatch: message says {capacity}, segment says {}",
                        channel.capacity()
                    )));
                }
                self.channel = Some(channel);
            }
            ControlMessage::RegisterArena { arena_id } => {
                let fds = self
                    .pending_fds
                    .pop_front()
                    .ok_or_else(|| Error::Element("RegisterArena without an fd".into()))?;
                let fd = fds
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Element("RegisterArena with an empty fd batch".into()))?;
                let arena = unsafe { SharedArena::from_fd(fd)? };
                if arena.id() != arena_id {
                    return Err(Error::Element(format!(
                        "ipc arena id mismatch: message says {arena_id}, header says {}",
                        arena.id()
                    )));
                }
                self.arena_cache.insert(arena_id, arena);
            }
            ControlMessage::MetaOverflow { seq, entries } => {
                self.meta_overflow.insert(seq, entries);
            }
            ControlMessage::Shutdown => {
                self.peer_eof = true;
            }
        }
        Ok(())
    }

    /// Drain and apply every queued control message.
    fn drain_control(&mut self) -> Result<()> {
        self.pump_control()?;
        while let Some(msg) = self.pending_ctrl.pop_front() {
            self.handle_control(msg)?;
        }
        Ok(())
    }

    /// Wait (bounded) until `pred(self)` after control traffic — for state
    /// that is provably already in flight (registration/overflow messages
    /// are sent before the descriptor that needs them; socket FIFO + ring
    /// publish give the happens-before).
    fn await_control(&mut self, what: &str, pred: impl Fn(&Self) -> bool) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            self.drain_control()?;
            if pred(self) {
                return Ok(());
            }
            if self.peer_eof {
                return Err(Error::Element(format!(
                    "ipcsrc: peer closed while waiting for {what}"
                )));
            }
            if Instant::now() >= deadline {
                return Err(Error::Element(format!(
                    "ipcsrc: {what} did not arrive within 2s — protocol violation"
                )));
            }
            // The message is already in the socket in every conforming run;
            // this loop spins only across a scheduling gap.
            std::thread::yield_now();
        }
    }

    /// Map a descriptor into a Buffer and ack it.
    fn map_and_ack(&mut self, desc: crate::memory::IpcDescriptor) -> Result<ProduceResult> {
        // The registration is already in the socket if we've never seen
        // this arena (sent before the descriptor was pushed).
        if !self.arena_cache.contains_key(&desc.arena_id) {
            let id = desc.arena_id;
            self.await_control("arena registration", |s| s.arena_cache.contains_key(&id))?;
        }
        // Same for overflow metadata.
        if desc.has_meta_overflow() && !self.meta_overflow.contains_key(&desc.seq) {
            let seq = desc.seq;
            self.await_control("overflow metadata", |s| s.meta_overflow.contains_key(&seq))?;
        }

        let (slot_ref, mut meta) = desc.decode()?;
        if let Some(entries) = self.meta_overflow.remove(&desc.seq) {
            for (key, bytes) in entries {
                // Keys must intern to the compile-time list; unknown ones
                // were never supposed to be sent.
                if let Some(known) = KNOWN_CUSTOM_KEYS.iter().find(|k| **k == key) {
                    meta.set_bytes(known, bytes);
                }
            }
        }

        let arena = self
            .arena_cache
            .get(&slot_ref.arena_id)
            .ok_or_else(|| Error::Element(format!("unknown arena {}", slot_ref.arena_id)))?;
        let slot = arena.slot_from_ipc(&slot_ref).ok_or_else(|| {
            Error::Element(format!(
                "stale ipc slot ref: seq {} slot {} in arena {} was already released",
                desc.seq, slot_ref.slot_index, slot_ref.arena_id
            ))
        })?;
        let buffer = Buffer::new(MemoryHandle::with_len(slot, slot_ref.len), meta);

        // Ack AFTER mapping: our slot_from_ipc refcount is what lets the
        // sink drop its pin — this ordering IS the #177 contract.
        let channel = self.channel.as_ref().unwrap();
        channel.try_push_ack(desc.seq)?;

        Ok(ProduceResult::OwnBuffer(buffer))
    }
}

impl AsyncSource for IpcSrc {
    async fn produce(&mut self, _ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        if !self.ensure_connected()? {
            return Ok(ProduceResult::WouldBlock);
        }

        // Handshake: the peer's RegisterChannel may arrive long after the
        // socket connects (a server-mode sink sends it from its first
        // consume). Poll, don't block.
        if self.channel.is_none() {
            self.drain_control()?;
            if self.channel.is_none() {
                if self.peer_eof {
                    return Ok(ProduceResult::Eos);
                }
                return Ok(ProduceResult::WouldBlock);
            }
        }

        loop {
            if let Some(desc) = self.channel.as_ref().unwrap().try_pop_desc() {
                return self.map_and_ack(desc);
            }

            match self.channel.as_ref().unwrap().state() {
                crate::memory::IpcChannelState::Eos => {
                    // The final push and set_eos race: re-check the ring
                    // once after observing EOS.
                    if let Some(desc) = self.channel.as_ref().unwrap().try_pop_desc() {
                        return self.map_and_ack(desc);
                    }
                    return Ok(ProduceResult::Eos);
                }
                crate::memory::IpcChannelState::Error => {
                    return Err(Error::Pipeline(
                        "ipc peer signaled an error before EOS".into(),
                    ));
                }
                crate::memory::IpcChannelState::Active => {}
            }

            // Opportunistically apply early registrations/overflow and
            // notice a dead peer.
            self.drain_control()?;
            if self.peer_eof {
                // Socket gone without EOS/Error in the segment: treat the
                // remaining ring content as final, then end.
                if let Some(desc) = self.channel.as_ref().unwrap().try_pop_desc() {
                    return self.map_and_ack(desc);
                }
                return Ok(ProduceResult::Eos);
            }

            // Bounded wait: the executor's stop/pause/seek handling only
            // runs between produce calls, so hand control back on a quiet
            // channel instead of awaiting unboundedly.
            let bell = self.channel.as_ref().unwrap().data_doorbell();
            match tokio::time::timeout(Duration::from_millis(100), bell.wait_async()).await {
                Ok(result) => result?,
                Err(_elapsed) => return Ok(ProduceResult::WouldBlock),
            }
        }
    }

    fn output_caps(&self) -> Caps {
        self.caps.clone()
    }
}

impl Drop for IpcSrc {
    fn drop(&mut self) {
        if self.is_server {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_sink_creation() {
        let sink = IpcSink::new("/tmp/test-ipc-sink.sock");
        assert!(sink.socket.is_none());
        assert!(sink.is_server);
        assert_eq!(sink.capacity, DEFAULT_IPC_RING_CAPACITY);
    }

    #[test]
    fn test_ipc_src_creation() {
        let src = IpcSrc::new("/tmp/test-ipc-src.sock");
        assert!(src.socket.is_none());
        assert!(!src.is_server);
    }

    #[test]
    fn test_ipc_sink_with_caps() {
        use crate::format::{MediaFormat, VideoCodec};

        let caps = Caps::new(MediaFormat::Video(VideoCodec::H264));
        let sink = IpcSink::new("/tmp/test.sock").with_caps(caps.clone());
        assert_eq!(sink.caps, caps);
    }

    #[test]
    fn test_ipc_src_listen_mode() {
        let src = IpcSrc::listen("/tmp/test-listen.sock");
        assert!(src.is_server);
    }

    #[test]
    fn overflow_entries_picks_known_byte_keys_only() {
        let mut meta = crate::metadata::Metadata::new();
        assert!(overflow_entries(&meta).is_empty());

        meta.set_klv(vec![1, 2, 3]);
        meta.set("app/counter", 7u64); // inline primitive: not forwarded
        let entries = overflow_entries(&meta);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, "stanag/klv");
        assert_eq!(entries[0].1, vec![1, 2, 3]);
    }
}
