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
use super::shared_refcount::SharedIpcSlotRef;
use crate::clock::ClockTime;
use crate::error::{Error, Result};
use crate::format::{
    AudioCodec, AudioFormat, Framerate, MediaFormat, PixelFormat, RtpEncoding, RtpFormat,
    SampleFormat, VideoCodec, VideoFormat,
};
use crate::metadata::{BufferFlags, Metadata, RtpMeta};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::mm::{MapFlags, ProtFlags};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Magic identifying a ring segment ("PLX_IPCR").
const IPC_RING_MAGIC: u64 = 0x504C585F49504352;

/// Ring segment format version, gated by exact equality in [`IpcChannel::from_fds`].
///
/// Bump whenever [`IpcDescriptor`]'s layout changes **or any enum encoded by
/// discriminant changes its variant set or order** (`PixelFormat`,
/// `VideoCodec`, `SampleFormat`, `AudioCodec`, the `RtpEncoding` tag table).
/// The `descriptor_enum_discriminants_locked` test turns that rule into a CI
/// failure instead of a silent misdecode.
const IPC_RING_VERSION: u32 = 1;

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

/// One buffer's worth of wire state: the slot reference plus every fixed
/// metadata field. 128 bytes (two cache lines), plain old data.
///
/// The only part of [`Metadata`] that does not ride here is the custom map
/// (type-erased); its known byte-valued entries overflow through the control
/// socket (`ControlMessage::MetaOverflow`), flagged by presence bit 2.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IpcDescriptor {
    /// Arena the slot lives in (process-unique since #178).
    pub arena_id: u64,
    /// Channel-monotonic sequence — the ack correlation id.
    pub seq: u64,
    /// Raw `ClockTime` nanos; `u64::MAX` = NONE. Round-trips exactly.
    pub pts: u64,
    /// See `pts`.
    pub dts: u64,
    /// See `pts`.
    pub duration: u64,
    /// `Metadata.sequence`.
    pub sequence: u64,
    /// `Metadata.offset`, valid iff presence bit 0.
    pub byte_offset: u64,
    /// Slot data offset within the arena.
    pub data_offset: u64,
    /// Payload length.
    pub len: u64,
    /// `Metadata.stream_id`.
    pub stream_id: u32,
    /// Slot index within the arena.
    pub slot_index: u32,
    /// `BufferFlags::bits()`.
    pub flags: u8,
    /// Bit 0: byte_offset present. Bit 1: rtp present. Bit 2: a
    /// `MetaOverflow` control message precedes this descriptor.
    pub presence: u8,
    /// MediaFormat tag: 0 none, 1 VideoRaw, 2 Video, 3 AudioRaw, 4 Audio,
    /// 5 Rtp, 6 MpegTs, 7 Bytes.
    pub fmt_tag: u8,
    /// `RtpMeta.pt`.
    pub rtp_pt: u8,
    /// `RtpMeta.seq`.
    pub rtp_seq: u16,
    /// `RtpMeta.marker` (0/1).
    pub rtp_marker: u8,
    _pad0: u8,
    /// `RtpMeta.ts`.
    pub rtp_ts: u32,
    /// `RtpMeta.ssrc`.
    pub rtp_ssrc: u32,
    /// MediaFormat payload word 1; meaning per `fmt_tag` (see `encode`).
    pub fmt_a: u32,
    /// MediaFormat payload word 2.
    pub fmt_b: u32,
    /// MediaFormat payload word 3.
    pub fmt_c: u32,
    /// MediaFormat payload word 4.
    pub fmt_d: u32,
    /// MediaFormat payload word 5.
    pub fmt_e: u32,
    _pad1: [u8; 12],
}

const _: () = assert!(std::mem::size_of::<IpcDescriptor>() == 128);
const _: () = assert!(std::mem::align_of::<IpcDescriptor>() == 8);

const PRESENCE_OFFSET: u8 = 1 << 0;
const PRESENCE_RTP: u8 = 1 << 1;
const PRESENCE_META_OVERFLOW: u8 = 1 << 2;

const FMT_NONE: u8 = 0;
const FMT_VIDEO_RAW: u8 = 1;
const FMT_VIDEO: u8 = 2;
const FMT_AUDIO_RAW: u8 = 3;
const FMT_AUDIO: u8 = 4;
const FMT_RTP: u8 = 5;
const FMT_MPEG_TS: u8 = 6;
const FMT_BYTES: u8 = 7;

/// `RtpEncoding` tag table (it is not `repr(u8)` — it carries `Dynamic(u8)`).
fn rtp_encoding_to_tag(e: RtpEncoding) -> (u32, u32) {
    match e {
        RtpEncoding::H264 => (0, 0),
        RtpEncoding::H265 => (1, 0),
        RtpEncoding::Vp8 => (2, 0),
        RtpEncoding::Vp9 => (3, 0),
        RtpEncoding::Opus => (4, 0),
        RtpEncoding::Pcmu => (5, 0),
        RtpEncoding::Pcma => (6, 0),
        RtpEncoding::Av1 => (7, 0),
        RtpEncoding::Dynamic(pt) => (8, pt as u32),
    }
}

fn rtp_encoding_from_tag(tag: u32, dynamic_pt: u32) -> Result<RtpEncoding> {
    Ok(match tag {
        0 => RtpEncoding::H264,
        1 => RtpEncoding::H265,
        2 => RtpEncoding::Vp8,
        3 => RtpEncoding::Vp9,
        4 => RtpEncoding::Opus,
        5 => RtpEncoding::Pcmu,
        6 => RtpEncoding::Pcma,
        7 => RtpEncoding::Av1,
        8 => RtpEncoding::Dynamic(dynamic_pt as u8),
        other => return Err(bad_wire(format!("RtpEncoding tag {other}"))),
    })
}

fn pixel_format_from(v: u32) -> Result<PixelFormat> {
    use PixelFormat::*;
    Ok(match v {
        x if x == I420 as u32 => I420,
        x if x == Nv12 as u32 => Nv12,
        x if x == I420_10Le as u32 => I420_10Le,
        x if x == P010 as u32 => P010,
        x if x == I422 as u32 => I422,
        x if x == Yuyv as u32 => Yuyv,
        x if x == Uyvy as u32 => Uyvy,
        x if x == I444 as u32 => I444,
        x if x == Rgb24 as u32 => Rgb24,
        x if x == Rgba as u32 => Rgba,
        x if x == Bgr24 as u32 => Bgr24,
        x if x == Bgra as u32 => Bgra,
        x if x == Argb as u32 => Argb,
        x if x == Gray8 as u32 => Gray8,
        x if x == Gray16Le as u32 => Gray16Le,
        other => return Err(bad_wire(format!("PixelFormat {other}"))),
    })
}

fn video_codec_from(v: u32) -> Result<VideoCodec> {
    use VideoCodec::*;
    Ok(match v {
        x if x == H264 as u32 => H264,
        x if x == H265 as u32 => H265,
        x if x == Vp8 as u32 => Vp8,
        x if x == Vp9 as u32 => Vp9,
        x if x == Av1 as u32 => Av1,
        other => return Err(bad_wire(format!("VideoCodec {other}"))),
    })
}

fn sample_format_from(v: u32) -> Result<SampleFormat> {
    use SampleFormat::*;
    Ok(match v {
        x if x == S16 as u32 => S16,
        x if x == S32 as u32 => S32,
        x if x == F32 as u32 => F32,
        x if x == U8 as u32 => U8,
        other => return Err(bad_wire(format!("SampleFormat {other}"))),
    })
}

fn audio_codec_from(v: u32) -> Result<AudioCodec> {
    use AudioCodec::*;
    Ok(match v {
        x if x == Opus as u32 => Opus,
        x if x == Aac as u32 => Aac,
        x if x == Mp3 as u32 => Mp3,
        x if x == Pcmu as u32 => Pcmu,
        x if x == Pcma as u32 => Pcma,
        x if x == Vorbis as u32 => Vorbis,
        x if x == Eac3 as u32 => Eac3,
        other => return Err(bad_wire(format!("AudioCodec {other}"))),
    })
}

fn bad_wire(what: String) -> Error {
    // Unreachable when the version gate holds: both sides run the same
    // descriptor lineage. Reaching this means segment corruption.
    Error::InvalidSegment(format!("ipc descriptor: unknown {what}"))
}

impl IpcDescriptor {
    /// Encode a slot reference plus metadata. Drops the custom map — the
    /// element layer overflows its known entries through the control socket
    /// and flags them with [`set_meta_overflow`](Self::set_meta_overflow).
    pub fn encode(seq: u64, slot: &SharedIpcSlotRef, meta: &Metadata) -> Self {
        let mut d = Self {
            arena_id: slot.arena_id,
            seq,
            pts: meta.pts.nanos(),
            dts: meta.dts.nanos(),
            duration: meta.duration.nanos(),
            sequence: meta.sequence,
            byte_offset: meta.offset.unwrap_or(0),
            data_offset: slot.data_offset as u64,
            len: slot.len as u64,
            stream_id: meta.stream_id,
            slot_index: slot.slot_index,
            flags: meta.flags.bits(),
            presence: 0,
            fmt_tag: FMT_NONE,
            rtp_pt: 0,
            rtp_seq: 0,
            rtp_marker: 0,
            _pad0: 0,
            rtp_ts: 0,
            rtp_ssrc: 0,
            fmt_a: 0,
            fmt_b: 0,
            fmt_c: 0,
            fmt_d: 0,
            fmt_e: 0,
            _pad1: [0; 12],
        };
        if meta.offset.is_some() {
            d.presence |= PRESENCE_OFFSET;
        }
        if let Some(rtp) = meta.rtp {
            d.presence |= PRESENCE_RTP;
            d.rtp_pt = rtp.pt;
            d.rtp_seq = rtp.seq;
            d.rtp_marker = rtp.marker as u8;
            d.rtp_ts = rtp.ts;
            d.rtp_ssrc = rtp.ssrc;
        }
        match &meta.format {
            None => {}
            Some(MediaFormat::VideoRaw(v)) => {
                d.fmt_tag = FMT_VIDEO_RAW;
                d.fmt_a = v.width;
                d.fmt_b = v.height;
                d.fmt_c = v.pixel_format as u32;
                d.fmt_d = v.framerate.num;
                d.fmt_e = v.framerate.den;
            }
            Some(MediaFormat::Video(c)) => {
                d.fmt_tag = FMT_VIDEO;
                d.fmt_a = *c as u32;
            }
            Some(MediaFormat::AudioRaw(a)) => {
                d.fmt_tag = FMT_AUDIO_RAW;
                d.fmt_a = a.sample_rate;
                d.fmt_b = a.channels as u32;
                d.fmt_c = a.sample_format as u32;
            }
            Some(MediaFormat::Audio(c)) => {
                d.fmt_tag = FMT_AUDIO;
                d.fmt_a = *c as u32;
            }
            Some(MediaFormat::Rtp(r)) => {
                d.fmt_tag = FMT_RTP;
                d.fmt_a = r.payload_type as u32;
                d.fmt_b = r.clock_rate;
                let (tag, dynamic) = rtp_encoding_to_tag(r.encoding);
                d.fmt_c = tag;
                d.fmt_d = dynamic;
            }
            Some(MediaFormat::MpegTs) => d.fmt_tag = FMT_MPEG_TS,
            Some(MediaFormat::Bytes) => d.fmt_tag = FMT_BYTES,
        }
        d
    }

    /// Mark that a `MetaOverflow` control message precedes this descriptor.
    pub fn set_meta_overflow(&mut self) {
        self.presence |= PRESENCE_META_OVERFLOW;
    }

    /// Whether the element layer must collect overflow metadata first.
    pub fn has_meta_overflow(&self) -> bool {
        self.presence & PRESENCE_META_OVERFLOW != 0
    }

    /// Decode back into a slot reference and metadata (custom map empty —
    /// the element layer attaches overflow entries afterwards).
    pub fn decode(&self) -> Result<(SharedIpcSlotRef, Metadata)> {
        let slot = SharedIpcSlotRef {
            arena_id: self.arena_id,
            slot_index: self.slot_index,
            data_offset: self.data_offset as usize,
            len: self.len as usize,
        };
        // Field-by-field: `custom` is private, so no struct literal/FRU.
        let mut meta = Metadata::new();
        meta.pts = ClockTime::from_nanos(self.pts);
        meta.dts = ClockTime::from_nanos(self.dts);
        meta.duration = ClockTime::from_nanos(self.duration);
        meta.sequence = self.sequence;
        meta.stream_id = self.stream_id;
        meta.flags = BufferFlags::from_bits(self.flags);
        if self.presence & PRESENCE_OFFSET != 0 {
            meta.offset = Some(self.byte_offset);
        }
        if self.presence & PRESENCE_RTP != 0 {
            meta.rtp = Some(RtpMeta {
                seq: self.rtp_seq,
                ts: self.rtp_ts,
                ssrc: self.rtp_ssrc,
                pt: self.rtp_pt,
                marker: self.rtp_marker != 0,
            });
        }
        meta.format = match self.fmt_tag {
            FMT_NONE => None,
            FMT_VIDEO_RAW => Some(MediaFormat::VideoRaw(VideoFormat {
                width: self.fmt_a,
                height: self.fmt_b,
                pixel_format: pixel_format_from(self.fmt_c)?,
                framerate: Framerate::new(self.fmt_d, self.fmt_e),
            })),
            FMT_VIDEO => Some(MediaFormat::Video(video_codec_from(self.fmt_a)?)),
            FMT_AUDIO_RAW => Some(MediaFormat::AudioRaw(AudioFormat {
                sample_rate: self.fmt_a,
                channels: self.fmt_b as u16,
                sample_format: sample_format_from(self.fmt_c)?,
            })),
            FMT_AUDIO => Some(MediaFormat::Audio(audio_codec_from(self.fmt_a)?)),
            FMT_RTP => Some(MediaFormat::Rtp(RtpFormat {
                payload_type: self.fmt_a as u8,
                clock_rate: self.fmt_b,
                encoding: rtp_encoding_from_tag(self.fmt_c, self.fmt_d)?,
            })),
            FMT_MPEG_TS => Some(MediaFormat::MpegTs),
            FMT_BYTES => Some(MediaFormat::Bytes),
            other => return Err(bad_wire(format!("format tag {other}"))),
        };
        Ok((slot, meta))
    }
}

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

    /// Changing any of these discriminants requires bumping
    /// `IPC_RING_VERSION` — the descriptor encodes them raw.
    #[test]
    fn descriptor_enum_discriminants_locked() {
        assert_eq!(PixelFormat::I420 as u8, 0);
        assert_eq!(PixelFormat::Gray16Le as u8, 14);
        assert_eq!(VideoCodec::H264 as u8, 0);
        assert_eq!(VideoCodec::Av1 as u8, 4);
        assert_eq!(SampleFormat::S16 as u8, 0);
        assert_eq!(SampleFormat::U8 as u8, 3);
        assert_eq!(AudioCodec::Opus as u8, 0);
        assert_eq!(AudioCodec::Eac3 as u8, 6);
        assert_eq!(rtp_encoding_to_tag(RtpEncoding::Av1), (7, 0));
        assert_eq!(rtp_encoding_to_tag(RtpEncoding::Dynamic(50)), (8, 50));
    }

    #[test]
    fn descriptor_and_segment_layout_locked() {
        assert_eq!(std::mem::size_of::<IpcDescriptor>(), 128);
        assert_eq!(std::mem::align_of::<IpcDescriptor>(), 8);
        assert_eq!(std::mem::size_of::<SegmentHeader>(), 320);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, arena_id), 0);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, seq), 8);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, len), 64);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, flags), 80);
        assert_eq!(std::mem::offset_of!(IpcDescriptor, fmt_a), 96);
        assert_eq!(std::mem::offset_of!(SegmentHeader, data_head), 64);
        assert_eq!(std::mem::offset_of!(SegmentHeader, ack_tail), 256);
        let (total, desc_off, ack_off) = layout(64);
        assert_eq!((total, desc_off, ack_off), (9024, 320, 8512));
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
