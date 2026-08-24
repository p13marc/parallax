//! Cross-process SPSC descriptor rings for the IPC data plane (#179).
//!
//! The IPC elements' payload path was always zero-copy (arena fd via
//! SCM_RIGHTS, slot descriptors); only the *message channel* was slow —
//! every BufferReady/BufferDone rkyv-framed through the Unix socket. This
//! module puts the descriptors in shared memory instead: one small memfd
//! segment carries two SPSC rings (data: [`IpcDescriptor`] entries,
//! sink→src; ack: `u64` seqs, src→sink) plus an EOS state word, and two
//! eventfd doorbells (passed by SCM_RIGHTS, rebuilt with
//! [`EventFd::from_owned_fd`]) provide the wakeups. The socket remains the
//! control plane: registration, fd passing, overflow metadata, shutdown —
//! the data-plane/signaling-plane split of design.md principle 8, applied
//! to the cross-process boundary exactly where the channel-architecture
//! report (§6.1) said a custom queue on the memory model is the right tool.
//!
//! # Why no per-entry commit word
//!
//! The report's sketch inherited a per-entry seq flag from `ReleaseQueue`,
//! which needs it because its push is MPSC two-phase (CAS-reserve, then
//! store) — the reservation is visible before the body. These rings are
//! SPSC per direction, so the `RingBuffer` protocol from `rt_bridge.rs`
//! transliterates directly: the producer writes the multi-word POD body
//! into `entries[head & mask]` — unpublished, the consumer never reads at
//! or past `head` — then publishes with a single `head.store(+1, Release)`.
//! Publication is atomic-or-nothing: a peer dying mid-write dies *before*
//! the sole release store, so a torn entry is unobservable. A `MAP_SHARED`
//! memfd mapping is ordinary coherent memory; the memory model applies
//! across processes unchanged.
//!
//! # The never-full invariant
//!
//! The sink bounds in-flight descriptors (its #177 pin table) at the ring
//! capacity. Every undrained ack corresponds to a distinct in-flight
//! descriptor, so acks-in-ring ≤ in-flight ≤ capacity — an ack push can
//! never find its ring full — and data-ring occupancy ≤ in-flight < capacity
//! at push time, so a descriptor push can never find its ring full either.
//! Both `false`/`Err` paths are protocol-violation assertions. Backpressure
//! lives in exactly one place: the sink awaiting the ack doorbell (which is
//! therefore also the space doorbell — no third fd).
//!
//! # Lifetime
//!
//! No cross-process refcount: each side owns its fd and mapping and unmaps
//! on drop; the kernel keeps the memfd pages while any fd or mapping lives,
//! so a peer's death never invalidates our view. Death is detected on the
//! control socket, not in the segment.

use super::eventfd::EventFd;
use crate::error::{Error, Result};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::mm::{MapFlags, ProtFlags};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Magic identifying a ring segment ("PLX_IPCR").
const IPC_RING_MAGIC: u64 = 0x504C585F49504352;

/// Ring segment format version, gated by exact equality in [`IpcChannel::from_fds`].
///
/// This is *mechanics*: the segment's layout, the descriptor's size and field
/// offsets, the handshake. It is **not** the schema version — that is
/// [`wire::WIRE_VERSION`](crate::wire::WIRE_VERSION), which changes when a
/// field's meaning or presence changes. The two move independently on purpose;
/// a doc saying "remember to bump both" is a doc that will be violated.
///
/// Bump this whenever [`IpcDescriptor`]'s layout changes. It no longer has to
/// track enum discriminants: since #219 the descriptor encodes through
/// [`wire::codes`](crate::wire::codes), whose own lock test pins every
/// assignment, so reordering `PixelFormat` is not a wire event at all.
///
/// 2: the descriptor grew 128 → 192 bytes to carry a `PlaneLayout` inline
/// (#219). 1: initial.
const IPC_RING_VERSION: u32 = 2;

/// Default entries per ring.
pub const DEFAULT_IPC_RING_CAPACITY: u32 = 64;

/// Sanity ceiling validated by `from_fds`.
const MAX_IPC_RING_CAPACITY: u32 = 4096;

/// Channel state, first transition from Active wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpcChannelState {
    /// Descriptors may still arrive.
    Active,
    /// The sink is done; drain the data ring, then stop.
    Eos,
    /// The sink failed; the reason travels over the control socket.
    Error,
}

const STATE_ACTIVE: u32 = 0;
const STATE_EOS: u32 = 1;
const STATE_ERROR: u32 = 2;

/// One producer- or consumer-counter on its own cache line, so the two
/// sides' stores never false-share.
#[repr(C, align(64))]
struct CounterLine {
    v: AtomicU64,
    _pad: [u8; 56],
}

const _: () = assert!(std::mem::size_of::<CounterLine>() == 64);

/// Segment header. 320 bytes: one line of identity/state, then the four
/// ring counters (data producer/consumer, ack producer/consumer), each on
/// its own line keyed to which *process role* writes it.
#[repr(C, align(64))]
struct SegmentHeader {
    magic: AtomicU64,
    version: AtomicU32,
    capacity: AtomicU32,
    state: AtomicU32,
    _reserved: [u8; 44],
    /// Data ring producer counter — written by the sink.
    data_head: CounterLine,
    /// Data ring consumer counter — written by the src.
    data_tail: CounterLine,
    /// Ack ring producer counter — written by the src.
    ack_head: CounterLine,
    /// Ack ring consumer counter — written by the sink.
    ack_tail: CounterLine,
}

const _: () = assert!(std::mem::size_of::<SegmentHeader>() == 320);

/// `(total, desc_offset, ack_offset)` for a given capacity.
fn layout(capacity: u32) -> (usize, usize, usize) {
    let cap = capacity as usize;
    let desc_offset = std::mem::size_of::<SegmentHeader>();
    let ack_offset = desc_offset + cap * std::mem::size_of::<IpcDescriptor>();
    let total = ack_offset + cap * std::mem::size_of::<u64>();
    (total, desc_offset, ack_offset)
}

// ============================================================================
// Descriptor
// ============================================================================

// The descriptor itself lives in `crate::wire::descriptor` (#219): it is a
// fixed-size POD *projection* of the same schema the zenoh attachment
// serializes, and keeping it here kept it drifting from that schema — `rtp`
// carried by one wire and not the other, `planes` by neither,
// `MediaFormat::Rtp` with a tag here and no mapping there. What stays here is
// the ring that carries it.
pub use crate::wire::descriptor::IpcDescriptor;

// ============================================================================
// Ring view + segment
// ============================================================================

/// One SPSC ring over raw shared-memory pointers.
///
/// SAFETY contract: exactly one process plays producer and one plays
/// consumer per ring (which end holds which role is fixed by construction:
/// the creating side is the data producer / ack consumer). The `RingBuffer`
/// release/acquire proof from `rt_bridge.rs` carries over verbatim.
struct RingView<T: Copy> {
    head: NonNull<CounterLine>,
    tail: NonNull<CounterLine>,
    entries: NonNull<T>,
    mask: u64,
    cap: u64,
}

impl<T: Copy> RingView<T> {
    fn head(&self) -> &AtomicU64 {
        unsafe { &self.head.as_ref().v }
    }

    fn tail(&self) -> &AtomicU64 {
        unsafe { &self.tail.as_ref().v }
    }

    fn try_push(&self, value: T) -> bool {
        let head = self.head().load(Ordering::Relaxed);
        // Acquire pairs with the consumer's tail Release: its entry READ
        // happens-before our overwrite of that slot.
        let tail = self.tail().load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= self.cap {
            return false;
        }
        unsafe {
            self.entries
                .as_ptr()
                .add((head & self.mask) as usize)
                .write(value);
        }
        // Publishes the body: the consumer's head Acquire sees every word.
        self.head().store(head.wrapping_add(1), Ordering::Release);
        true
    }

    fn try_pop(&self) -> Option<T> {
        let tail = self.tail().load(Ordering::Relaxed);
        // Acquire pairs with the producer's head Release: body visible.
        let head = self.head().load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let value = unsafe {
            self.entries
                .as_ptr()
                .add((tail & self.mask) as usize)
                .read()
        };
        // Returns the slot to the producer.
        self.tail().store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    fn len(&self) -> u64 {
        self.head()
            .load(Ordering::Acquire)
            .wrapping_sub(self.tail().load(Ordering::Acquire))
    }
}

/// The mmapped memfd; unmaps on drop.
struct Segment {
    #[allow(dead_code)]
    fd: OwnedFd,
    base: NonNull<u8>,
    total: usize,
}

impl Drop for Segment {
    fn drop(&mut self) {
        unsafe {
            let _ = rustix::mm::munmap(self.base.as_ptr().cast(), self.total);
        }
    }
}

impl Segment {
    fn map(fd: OwnedFd, total: usize) -> Result<Self> {
        let base = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                total,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &fd,
                0,
            )?
        };
        let base = NonNull::new(base.cast::<u8>())
            .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;
        Ok(Self { fd, base, total })
    }

    fn header(&self) -> &SegmentHeader {
        unsafe { &*self.base.as_ptr().cast::<SegmentHeader>() }
    }
}

// ============================================================================
// Channel
// ============================================================================

/// The IPC data-plane channel: two rings + state word in one memfd segment,
/// two doorbells beside it.
///
/// Created by the sink side ([`create`](Self::create)); the src side
/// rebuilds it from the three fds received over SCM_RIGHTS
/// ([`from_fds`](Self::from_fds)).
pub struct IpcChannel {
    seg: Segment,
    data: RingView<IpcDescriptor>,
    ack: RingView<u64>,
    data_doorbell: EventFd,
    ack_doorbell: EventFd,
}

// SAFETY: all shared state is atomics in the mapping; the SPSC role
// contract is documented on RingView and enforced by which process holds
// which end.
unsafe impl Send for IpcChannel {}
unsafe impl Sync for IpcChannel {}

impl IpcChannel {
    /// Create the segment + doorbells (sink side).
    ///
    /// `capacity` must be a power of two in `[1, 4096]`; it bounds both
    /// rings and therefore the sink's in-flight pin table.
    pub fn create(capacity: u32) -> Result<Self> {
        if capacity == 0 || !capacity.is_power_of_two() || capacity > MAX_IPC_RING_CAPACITY {
            return Err(Error::AllocationFailed(format!(
                "ipc ring capacity must be a power of two in [1, {MAX_IPC_RING_CAPACITY}], got {capacity}"
            )));
        }
        let (total, _, _) = layout(capacity);

        let cname = std::ffi::CString::new("parallax-ipc-ring")
            .map_err(|e| Error::AllocationFailed(e.to_string()))?;
        let fd = rustix::fs::memfd_create(&cname, rustix::fs::MemfdFlags::CLOEXEC)?;
        rustix::fs::ftruncate(&fd, total as u64)?;

        let seg = Segment::map(fd, total)?;
        {
            let h = seg.header();
            h.version.store(IPC_RING_VERSION, Ordering::Release);
            h.capacity.store(capacity, Ordering::Release);
            h.state.store(STATE_ACTIVE, Ordering::Release);
            // Counters are memfd zero-fill; store magic LAST so a peer
            // cannot validate a half-initialized header.
            h.magic.store(IPC_RING_MAGIC, Ordering::Release);
        }

        Ok(Self {
            data: Self::data_ring(&seg, capacity),
            ack: Self::ack_ring(&seg, capacity),
            seg,
            data_doorbell: EventFd::new()?,
            ack_doorbell: EventFd::new()?,
        })
    }

    /// Rebuild the channel from received fds (src side), in the order
    /// [`fds`](Self::fds) sends them: `[segment, data doorbell, ack doorbell]`.
    ///
    /// # Safety
    ///
    /// `ring` must be a ring-segment memfd. The header is validated (magic,
    /// version exact, capacity bounds, exact file size), but a hostile
    /// well-formed segment is indistinguishable from a real one — same
    /// trust model as [`SharedArena::from_fd`](super::SharedArena).
    pub unsafe fn from_fds(ring: OwnedFd, data_db: OwnedFd, ack_db: OwnedFd) -> Result<Self> {
        let stat = rustix::fs::fstat(&ring)?;
        let file_size = stat.st_size as usize;
        if file_size < std::mem::size_of::<SegmentHeader>() {
            return Err(Error::InvalidSegment(
                "ipc ring segment too small for header".into(),
            ));
        }

        let seg = Segment::map(ring, file_size)?;
        let (magic, version, capacity) = {
            let h = seg.header();
            (
                h.magic.load(Ordering::Acquire),
                h.version.load(Ordering::Acquire),
                h.capacity.load(Ordering::Acquire),
            )
        };
        if magic != IPC_RING_MAGIC {
            return Err(Error::InvalidSegment(format!(
                "invalid ipc ring magic: expected {IPC_RING_MAGIC:x}, got {magic:x}"
            )));
        }
        if version != IPC_RING_VERSION {
            return Err(Error::InvalidSegment(format!(
                "unsupported ipc ring version: expected {IPC_RING_VERSION}, got {version}"
            )));
        }
        if capacity == 0 || !capacity.is_power_of_two() || capacity > MAX_IPC_RING_CAPACITY {
            return Err(Error::InvalidSegment(format!(
                "ipc ring declares invalid capacity {capacity}"
            )));
        }
        let (expected_total, _, _) = layout(capacity);
        if file_size != expected_total {
            return Err(Error::InvalidSegment(format!(
                "ipc ring size {file_size} disagrees with capacity {capacity} (expected {expected_total})"
            )));
        }

        Ok(Self {
            data: Self::data_ring(&seg, capacity),
            ack: Self::ack_ring(&seg, capacity),
            seg,
            data_doorbell: EventFd::from_owned_fd(data_db)?,
            ack_doorbell: EventFd::from_owned_fd(ack_db)?,
        })
    }

    fn data_ring(seg: &Segment, capacity: u32) -> RingView<IpcDescriptor> {
        let (_, desc_offset, _) = layout(capacity);
        let h = seg.base.as_ptr().cast::<SegmentHeader>();
        unsafe {
            RingView {
                head: NonNull::new_unchecked(&raw mut (*h).data_head),
                tail: NonNull::new_unchecked(&raw mut (*h).data_tail),
                entries: NonNull::new_unchecked(
                    seg.base.as_ptr().add(desc_offset).cast::<IpcDescriptor>(),
                ),
                mask: capacity as u64 - 1,
                cap: capacity as u64,
            }
        }
    }

    fn ack_ring(seg: &Segment, capacity: u32) -> RingView<u64> {
        let (_, _, ack_offset) = layout(capacity);
        let h = seg.base.as_ptr().cast::<SegmentHeader>();
        unsafe {
            RingView {
                head: NonNull::new_unchecked(&raw mut (*h).ack_head),
                tail: NonNull::new_unchecked(&raw mut (*h).ack_tail),
                entries: NonNull::new_unchecked(seg.base.as_ptr().add(ack_offset).cast::<u64>()),
                mask: capacity as u64 - 1,
                cap: capacity as u64,
            }
        }
    }

    /// The three fds to pass over SCM_RIGHTS, in `from_fds` order.
    pub fn fds(&self) -> [BorrowedFd<'_>; 3] {
        [
            self.seg.fd.as_fd(),
            self.data_doorbell.fd(),
            self.ack_doorbell.fd(),
        ]
    }

    /// Entries per ring.
    pub fn capacity(&self) -> u32 {
        self.seg.header().capacity.load(Ordering::Relaxed)
    }

    /// Approximate data-ring occupancy (debugging/monitoring).
    pub fn len(&self) -> u64 {
        self.data.len()
    }

    /// Whether the data ring is empty (debugging/monitoring).
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    // ---- sink role -------------------------------------------------------

    /// Publish a descriptor and ring the data doorbell.
    ///
    /// Under the in-flight bound (< capacity pins) the ring can never be
    /// full — a `false` here is a protocol violation, not backpressure.
    pub fn try_push_desc(&self, desc: IpcDescriptor) -> bool {
        let pushed = self.data.try_push(desc);
        if pushed {
            let _ = self.data_doorbell.notify();
        }
        pushed
    }

    /// Drain one acknowledged seq, if any.
    pub fn try_pop_ack(&self) -> Option<u64> {
        self.ack.try_pop()
    }

    /// Await the next ack. Cancel-safe (the pop happens synchronously after
    /// the wake; the eventfd counter is sticky). Waits forever if the peer
    /// dies silently — callers wrap it in a timeout and watch the socket.
    pub async fn wait_ack(&self) -> Result<u64> {
        loop {
            if let Some(seq) = self.try_pop_ack() {
                return Ok(seq);
            }
            self.ack_doorbell.wait_async().await?;
        }
    }

    /// Move to EOS (first transition from Active wins) and wake the reader.
    pub fn set_eos(&self) {
        let _ = self.seg.header().state.compare_exchange(
            STATE_ACTIVE,
            STATE_EOS,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let _ = self.data_doorbell.notify();
    }

    /// Move to Error (first transition from Active wins) and wake the
    /// reader. The reason text travels over the control socket.
    pub fn set_error(&self) {
        let _ = self.seg.header().state.compare_exchange(
            STATE_ACTIVE,
            STATE_ERROR,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let _ = self.data_doorbell.notify();
    }

    // ---- src role --------------------------------------------------------

    /// Pop one descriptor, if any. The caller acks with
    /// [`try_push_ack`](Self::try_push_ack) *after* mapping the slot.
    pub fn try_pop_desc(&self) -> Option<IpcDescriptor> {
        self.data.try_pop()
    }

    /// Acknowledge a mapped descriptor and ring the ack doorbell.
    ///
    /// By the never-full invariant this cannot fail against a conforming
    /// peer; a full ring means the sink broke its in-flight bound.
    pub fn try_push_ack(&self, seq: u64) -> Result<()> {
        if !self.ack.try_push(seq) {
            return Err(Error::InvalidSegment(
                "ipc ack ring full — peer exceeded its in-flight bound".into(),
            ));
        }
        let _ = self.ack_doorbell.notify();
        Ok(())
    }

    /// Await the next descriptor; `Ok(None)` at EOS **after** the ring is
    /// drained (in-flight descriptors survive EOS), `Err` on Error state.
    /// Cancel-safe. Waits forever if the peer dies silently — callers wrap
    /// it in a timeout and watch the socket.
    pub async fn recv_desc(&self) -> Result<Option<IpcDescriptor>> {
        loop {
            if let Some(desc) = self.try_pop_desc() {
                return Ok(Some(desc));
            }
            match self.state() {
                // Re-check the ring once after observing EOS: the final
                // push and set_eos race, and drain-before-honoring-EOS is
                // the contract.
                IpcChannelState::Eos => {
                    return Ok(self.try_pop_desc());
                }
                IpcChannelState::Error => {
                    return Err(Error::Pipeline(
                        "ipc peer signaled an error (reason on the control socket)".into(),
                    ));
                }
                IpcChannelState::Active => {}
            }
            self.data_doorbell.wait_async().await?;
        }
    }

    /// Current channel state.
    pub fn state(&self) -> IpcChannelState {
        match self.seg.header().state.load(Ordering::Acquire) {
            STATE_EOS => IpcChannelState::Eos,
            STATE_ERROR => IpcChannelState::Error,
            _ => IpcChannelState::Active,
        }
    }

    /// The data doorbell, for timeout-wrapped waits in the elements.
    pub fn data_doorbell(&self) -> &EventFd {
        &self.data_doorbell
    }

    /// The ack doorbell, for timeout-wrapped waits in the elements.
    pub fn ack_doorbell(&self) -> &EventFd {
        &self.ack_doorbell
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ClockTime;
    use crate::format::{
        AudioCodec, AudioFormat, Framerate, MediaFormat, PixelFormat, RtpEncoding, RtpFormat,
        SampleFormat, VideoCodec, VideoFormat,
    };
    use crate::memory::SharedIpcSlotRef;
    use crate::metadata::{BufferFlags, Metadata, RtpMeta};

    /// Simulate the receiving process: dup all three fds and attach.
    fn attach(chan: &IpcChannel) -> IpcChannel {
        let [ring, data_db, ack_db] = chan.fds();
        let ring = rustix::io::fcntl_dupfd_cloexec(ring, 0).unwrap();
        let data_db = rustix::io::fcntl_dupfd_cloexec(data_db, 0).unwrap();
        let ack_db = rustix::io::fcntl_dupfd_cloexec(ack_db, 0).unwrap();
        unsafe { IpcChannel::from_fds(ring, data_db, ack_db).unwrap() }
    }

    fn desc(seq: u64) -> IpcDescriptor {
        let slot = SharedIpcSlotRef {
            arena_id: 0xA5A5,
            slot_index: seq as u32,
            data_offset: 4096 + seq as usize * 64,
            len: 64,
        };
        let mut meta = Metadata::new();
        meta.pts = ClockTime::from_nanos(seq * 1_000);
        meta.sequence = seq;
        IpcDescriptor::encode(seq, &slot, &meta)
    }

    #[test]
    fn create_attach_round_trip() {
        let sink = IpcChannel::create(16).unwrap();
        let src = attach(&sink);
        assert_eq!(src.capacity(), 16);
        assert_eq!(src.state(), IpcChannelState::Active);

        assert!(sink.try_push_desc(desc(7)));
        let got = src
            .try_pop_desc()
            .expect("descriptor visible across the attach");
        assert_eq!(got.seq, 7);
        assert_eq!(got.arena_id, 0xA5A5);
        assert_eq!(got.pts, 7_000);
        // The push rang the doorbell; the attach shares the eventfd.
        assert!(src.data_doorbell().try_wait().unwrap());

        src.try_push_ack(7).unwrap();
        assert_eq!(sink.try_pop_ack(), Some(7));
        assert!(sink.ack_doorbell().try_wait().unwrap());
    }

    #[test]
    fn fill_then_drain_capacity() {
        let sink = IpcChannel::create(8).unwrap();
        let src = attach(&sink);
        for i in 0..8 {
            assert!(sink.try_push_desc(desc(i)), "push {i} within capacity");
        }
        assert!(!sink.try_push_desc(desc(8)), "push past capacity must fail");
        for i in 0..8 {
            assert_eq!(src.try_pop_desc().unwrap().seq, i, "FIFO order");
        }
        assert!(src.try_pop_desc().is_none());
    }

    #[test]
    fn wrap_around_preserves_order() {
        let sink = IpcChannel::create(4).unwrap();
        let src = attach(&sink);
        for i in 0..40u64 {
            assert!(sink.try_push_desc(desc(i)));
            let got = src.try_pop_desc().unwrap();
            assert_eq!(got.seq, i);
            src.try_push_ack(got.seq).unwrap();
            assert_eq!(sink.try_pop_ack(), Some(i));
        }
    }

    #[test]
    fn ack_ring_never_full_at_max_in_flight() {
        // The invariant: in-flight <= capacity, so pushing capacity acks
        // must always succeed — across several wrap cycles.
        let sink = IpcChannel::create(8).unwrap();
        let src = attach(&sink);
        for _round in 0..3 {
            for i in 0..8 {
                assert!(sink.try_push_desc(desc(i)));
            }
            for _ in 0..8 {
                let d = src.try_pop_desc().unwrap();
                src.try_push_ack(d.seq)
                    .expect("ack ring must never be full");
            }
            for _ in 0..8 {
                assert!(sink.try_pop_ack().is_some());
            }
        }
    }

    #[tokio::test]
    async fn eos_delivers_after_drain() {
        let sink = IpcChannel::create(8).unwrap();
        let src = attach(&sink);
        assert!(sink.try_push_desc(desc(1)));
        assert!(sink.try_push_desc(desc(2)));
        sink.set_eos();

        // In-flight descriptors survive EOS.
        assert_eq!(src.recv_desc().await.unwrap().unwrap().seq, 1);
        assert_eq!(src.recv_desc().await.unwrap().unwrap().seq, 2);
        assert!(src.recv_desc().await.unwrap().is_none(), "then EOS");
    }

    #[tokio::test]
    async fn error_state_first_transition_wins() {
        let a = IpcChannel::create(4).unwrap();
        a.set_eos();
        a.set_error();
        assert_eq!(a.state(), IpcChannelState::Eos, "EOS came first");

        let b = IpcChannel::create(4).unwrap();
        b.set_error();
        b.set_eos();
        assert_eq!(b.state(), IpcChannelState::Error, "error came first");
        assert!(attach(&b).recv_desc().await.is_err());
    }

    #[test]
    fn from_fds_rejects_bad_magic_version_capacity_size() {
        let chan = IpcChannel::create(8).unwrap();

        // Bad magic.
        chan.seg.header().magic.store(0xDEAD, Ordering::Release);
        let [ring, d, a] = chan.fds();
        let dup = |fd| rustix::io::fcntl_dupfd_cloexec(fd, 0).unwrap();
        assert!(unsafe { IpcChannel::from_fds(dup(ring), dup(d), dup(a)) }.is_err());
        chan.seg
            .header()
            .magic
            .store(IPC_RING_MAGIC, Ordering::Release);

        // Bad version.
        chan.seg.header().version.store(999, Ordering::Release);
        let [ring, d, a] = chan.fds();
        assert!(unsafe { IpcChannel::from_fds(dup(ring), dup(d), dup(a)) }.is_err());
        chan.seg
            .header()
            .version
            .store(IPC_RING_VERSION, Ordering::Release);

        // Capacity not a power of two.
        chan.seg.header().capacity.store(7, Ordering::Release);
        let [ring, d, a] = chan.fds();
        assert!(unsafe { IpcChannel::from_fds(dup(ring), dup(d), dup(a)) }.is_err());
        chan.seg.header().capacity.store(8, Ordering::Release);

        // Size mismatch: capacity says 8 but the file only holds 4's worth.
        chan.seg.header().capacity.store(4, Ordering::Release);
        let [ring, d, a] = chan.fds();
        assert!(unsafe { IpcChannel::from_fds(dup(ring), dup(d), dup(a)) }.is_err());
    }

    #[test]
    fn create_rejects_bad_capacity() {
        assert!(IpcChannel::create(0).is_err());
        assert!(IpcChannel::create(3).is_err());
        assert!(IpcChannel::create(MAX_IPC_RING_CAPACITY * 2).is_err());
    }

    #[test]
    fn descriptor_round_trip_none_timestamps() {
        let slot = SharedIpcSlotRef {
            arena_id: 1,
            slot_index: 2,
            data_offset: 3,
            len: 4,
        };
        let meta = Metadata::new();
        let d = IpcDescriptor::encode(9, &slot, &meta);
        assert_eq!(d.seq, 9);
        let (slot2, meta2) = d.decode().unwrap();
        assert_eq!(slot2.arena_id, 1);
        assert_eq!(slot2.slot_index, 2);
        assert_eq!(slot2.data_offset, 3);
        assert_eq!(slot2.len, 4);
        assert_eq!(meta2.pts, meta.pts);
        assert_eq!(meta2.dts, meta.dts);
        assert_eq!(meta2.duration, meta.duration);
        assert_eq!(meta2.offset, None);
        assert_eq!(meta2.rtp, None);
        assert_eq!(meta2.format, None);

        // Explicit NONE round-trips exactly.
        let mut meta = Metadata::new();
        meta.pts = ClockTime::NONE;
        meta.dts = ClockTime::NONE;
        meta.duration = ClockTime::NONE;
        let (_, meta2) = IpcDescriptor::encode(1, &slot, &meta).decode().unwrap();
        assert!(meta2.pts.is_none());
        assert!(meta2.dts.is_none());
        assert!(meta2.duration.is_none());
    }

    #[test]
    fn descriptor_round_trip_all_fields() {
        let slot = SharedIpcSlotRef {
            arena_id: u64::MAX - 1,
            slot_index: u32::MAX,
            data_offset: usize::MAX / 2,
            len: 1 << 30,
        };
        let mut meta = Metadata::new();
        meta.pts = ClockTime::from_nanos(123);
        meta.dts = ClockTime::from_nanos(456);
        meta.duration = ClockTime::from_nanos(789);
        meta.sequence = 42;
        meta.stream_id = 7;
        meta.flags = BufferFlags::SYNC_POINT.insert(BufferFlags::DISCONT);
        meta.offset = Some(0xCAFE);
        meta.rtp = Some(RtpMeta {
            seq: 999,
            ts: 90_000,
            ssrc: 0xDEADBEEF,
            pt: 96,
            marker: true,
        });
        meta.format = Some(MediaFormat::VideoRaw(VideoFormat {
            width: 1920,
            height: 1080,
            pixel_format: PixelFormat::Nv12,
            framerate: Framerate::new(30_000, 1_001),
        }));

        let mut d = IpcDescriptor::encode(5, &slot, &meta);
        assert!(!d.has_meta_overflow());
        d.set_meta_overflow();
        assert!(d.has_meta_overflow());

        let (slot2, meta2) = d.decode().unwrap();
        assert_eq!(slot2.arena_id, slot.arena_id);
        assert_eq!(slot2.slot_index, slot.slot_index);
        assert_eq!(slot2.data_offset, slot.data_offset);
        assert_eq!(slot2.len, slot.len);
        assert_eq!(meta2.pts.nanos(), 123);
        assert_eq!(meta2.sequence, 42);
        assert_eq!(meta2.stream_id, 7);
        assert_eq!(meta2.flags, meta.flags);
        assert_eq!(meta2.offset, Some(0xCAFE));
        assert_eq!(meta2.rtp, meta.rtp);
        assert_eq!(meta2.format, meta.format);
    }

    #[test]
    fn coded_geometry_rides_the_descriptor() {
        let slot = SharedIpcSlotRef {
            arena_id: 1,
            slot_index: 0,
            data_offset: 0,
            len: 1,
        };
        let mut meta = Metadata::new();
        meta.set_coded_video_dims(VideoCodec::H264, 1920, 1080);

        let (_slot, back) = IpcDescriptor::encode(0, &slot, &meta).decode().unwrap();
        assert_eq!(back.format, Some(MediaFormat::Video(VideoCodec::H264)));
        assert_eq!(back.coded_video_dims(), Some((1920, 1080)));
        // Encoded video has no plane layout to inherit.
        assert_eq!(back.video_dims(), None);
    }

    #[test]
    fn a_codec_without_geometry_decodes_as_unknown_geometry() {
        // The RTP depayloaders know the codec and nothing else; that must not
        // come back as a 0x0 picture.
        let slot = SharedIpcSlotRef {
            arena_id: 1,
            slot_index: 0,
            data_offset: 0,
            len: 1,
        };
        let mut meta = Metadata::new();
        meta.format = Some(MediaFormat::Video(VideoCodec::H265));

        let (_slot, back) = IpcDescriptor::encode(0, &slot, &meta).decode().unwrap();
        assert_eq!(back.format, Some(MediaFormat::Video(VideoCodec::H265)));
        assert_eq!(back.coded_video_dims(), None);
        assert!(back.coded.is_none());
    }

    #[test]
    fn descriptor_round_trip_all_media_formats() {
        let slot = SharedIpcSlotRef {
            arena_id: 1,
            slot_index: 0,
            data_offset: 0,
            len: 1,
        };
        let formats = [
            MediaFormat::VideoRaw(VideoFormat {
                width: 640,
                height: 480,
                pixel_format: PixelFormat::Gray16Le,
                framerate: Framerate::new(25, 1),
            }),
            MediaFormat::Video(VideoCodec::Av1),
            MediaFormat::Video(VideoCodec::Mjpeg),
            MediaFormat::AudioRaw(AudioFormat {
                sample_rate: 48_000,
                channels: 2,
                sample_format: SampleFormat::F32,
            }),
            MediaFormat::Audio(AudioCodec::Eac3),
            MediaFormat::Rtp(RtpFormat {
                payload_type: 96,
                clock_rate: 90_000,
                encoding: RtpEncoding::Dynamic(99),
            }),
            MediaFormat::MpegTs,
            MediaFormat::Bytes,
        ];
        for format in formats {
            let mut meta = Metadata::new();
            meta.format = Some(format.clone());
            let (_, meta2) = IpcDescriptor::encode(0, &slot, &meta).decode().unwrap();
            assert_eq!(meta2.format, Some(format));
        }
    }

    /// The discriminant lock this test used to be is gone (#219): the
    /// descriptor no longer encodes enum discriminants, it encodes
    /// `wire::codes`, whose `every_code_is_pinned` test covers **both** wires
    /// rather than this one. What is left worth pinning here is that the
    /// descriptor really does go through that table.
    #[test]
    fn the_descriptor_encodes_through_the_shared_code_table() {
        use crate::wire::codes;

        let mut meta = Metadata::new();
        meta.format = Some(MediaFormat::VideoRaw(VideoFormat {
            width: 64,
            height: 48,
            pixel_format: PixelFormat::Gray16Le,
            framerate: Framerate::new(25, 1),
        }));
        let slot = SharedIpcSlotRef {
            arena_id: 1,
            slot_index: 0,
            data_offset: 0,
            len: 0,
        };
        let d = IpcDescriptor::encode(0, &slot, &meta);
        assert_eq!(
            d.fmt_c,
            codes::pixel_format_to_code(PixelFormat::Gray16Le) as u32
        );
    }

    #[test]
    fn descriptor_and_segment_layout_locked() {
        // 192 since #219 (was 128): the plane layout rides inline.
        assert_eq!(std::mem::size_of::<IpcDescriptor>(), 192);
        assert_eq!(std::mem::align_of::<IpcDescriptor>(), 8);
        assert_eq!(std::mem::size_of::<SegmentHeader>(), 320);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, arena_id), 0);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, seq), 8);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, len), 64);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, flags), 80);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, fmt_a), 96);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, plane_offset), 116);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, plane_stride), 132);
        assert_eq!(std::mem::offset_of!(SegmentHeader, data_head), 64);
        assert_eq!(std::mem::offset_of!(SegmentHeader, ack_tail), 256);
        let (total, desc_off, ack_off) = layout(64);
        assert_eq!((total, desc_off, ack_off), (13120, 320, 12608));
    }

    #[tokio::test]
    async fn recv_desc_wakes_on_doorbell() {
        let sink = std::sync::Arc::new(IpcChannel::create(8).unwrap());
        let src = std::sync::Arc::new(attach(&sink));

        let waiter = {
            let src = src.clone();
            tokio::spawn(async move { src.recv_desc().await.unwrap().unwrap().seq })
        };
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(sink.try_push_desc(desc(77)));

        let got = tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
            .await
            .expect("recv_desc never woke")
            .unwrap();
        assert_eq!(got, 77);
    }

    #[test]
    fn two_thread_stress() {
        const N: u64 = 10_000;
        let sink = std::sync::Arc::new(IpcChannel::create(16).unwrap());
        let src = std::sync::Arc::new(attach(&sink));

        let producer = {
            let sink = sink.clone();
            std::thread::spawn(move || {
                let mut in_flight = 0u64;
                let mut next = 0u64;
                let mut acked = 0u64;
                while acked < N {
                    while next < N && in_flight < 16 {
                        assert!(
                            sink.try_push_desc(desc(next)),
                            "push within in-flight bound"
                        );
                        next += 1;
                        in_flight += 1;
                    }
                    while let Some(seq) = sink.try_pop_ack() {
                        assert_eq!(seq, acked, "acks are FIFO");
                        acked += 1;
                        in_flight -= 1;
                    }
                    std::hint::spin_loop();
                }
            })
        };
        let consumer = {
            let src = src.clone();
            std::thread::spawn(move || {
                let mut expected = 0u64;
                while expected < N {
                    if let Some(d) = src.try_pop_desc() {
                        assert_eq!(d.seq, expected, "descriptors are FIFO");
                        let (slot, _) = d.decode().unwrap();
                        assert_eq!(slot.slot_index, expected as u32);
                        src.try_push_ack(d.seq).unwrap();
                        expected += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
            })
        };
        producer.join().unwrap();
        consumer.join().unwrap();
    }
}
