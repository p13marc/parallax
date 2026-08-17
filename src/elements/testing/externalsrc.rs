//! An External-memory test source (#194).
//!
//! Exercises the strided/external flow without a codec: a fixed pool of
//! heap frames stands in for decoder-owned pictures, laid out STRIDED
//! (row padding filled with `0xEE`) so consumers must honor
//! `Metadata::plane_layout` — a packed misread sees the sentinel bytes.
//! The producer discipline is the real one: pinned frames, a release
//! channel fed by each slot's last-drop hook, `WouldBlock` while every
//! frame is in flight, and External emission gated on
//! `set_negotiated_memory` exactly like a real decoder.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ProduceContext, ProduceResult, Source};
use crate::error::Result;
use crate::format::{
    CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, PixelFormat, PlaneDesc,
    PlaneLayout, VideoFormatCaps,
};
use crate::memory::{ExternalSlot, MemoryType, OutputArena, defaults};
use crate::metadata::Metadata;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

/// Frame geometry every [`ExternalTestSrc`] uses.
pub const TEST_WIDTH: u32 = 64;
/// Frame geometry every [`ExternalTestSrc`] uses.
pub const TEST_HEIGHT: u32 = 48;
/// Row padding added to each plane's stride.
pub const TEST_PADDING: usize = 16;
/// The byte value filling the stride padding.
pub const PAD_BYTE: u8 = 0xEE;

/// The packed I420 reference frame for sequence number `seq` — what a
/// consumer that honors the layout must reconstruct byte-exactly.
pub fn packed_reference_frame(seq: u64) -> Vec<u8> {
    let len = PlaneLayout::packed(PixelFormat::I420, TEST_WIDTH, TEST_HEIGHT).required_len(
        PixelFormat::I420,
        TEST_WIDTH,
        TEST_HEIGHT,
    );
    (0..len).map(|i| ((i as u64 + seq) % 251) as u8).collect()
}

/// The strided layout every emitted External frame declares.
pub fn strided_test_layout() -> PlaneLayout {
    let packed = PlaneLayout::packed(PixelFormat::I420, TEST_WIDTH, TEST_HEIGHT);
    let mut descs = Vec::new();
    let mut offset = 0;
    for p in packed.resolved(PixelFormat::I420, TEST_WIDTH, TEST_HEIGHT) {
        let stride = p.row_bytes + TEST_PADDING;
        descs.push(PlaneDesc { offset, stride });
        offset += stride * p.rows;
    }
    PlaneLayout::from_planes(&descs)
}

/// Build the strided byte image of [`packed_reference_frame`]`(seq)`.
fn strided_frame(seq: u64) -> Box<[u8]> {
    let layout = strided_test_layout();
    let packed = packed_reference_frame(seq);
    let mut out = vec![PAD_BYTE; layout.required_len(PixelFormat::I420, TEST_WIDTH, TEST_HEIGHT)]
        .into_boxed_slice();
    let mut src = 0;
    for p in layout.resolved(PixelFormat::I420, TEST_WIDTH, TEST_HEIGHT) {
        for row in 0..p.rows {
            let dst = p.offset + row * p.stride;
            out[dst..dst + p.row_bytes].copy_from_slice(&packed[src..src + p.row_bytes]);
            src += p.row_bytes;
        }
    }
    out
}

/// Test source producing strided External buffers from a fixed pool.
pub struct ExternalTestSrc {
    /// The "decoder-owned" frames; an in-flight slot pins its Arc clone.
    frames: Vec<Arc<Box<[u8]>>>,
    /// Packed pattern seed per pool slot (rewritten on reuse).
    /// Pool slots cycle: frame content for sequence `seq` lives in slot
    /// `seq % pool`, so payloads are pre-built per slot from its first use
    /// and reused as-is — tests compare against `seq`'s reference by
    /// taking `seq % pool` into account via `slot_seed`.
    pool_size: u32,
    /// Indices currently free to hand out.
    available: VecDeque<u32>,
    /// Fed by each slot's last-drop release hook.
    release_rx: Receiver<u32>,
    release_tx: Sender<u32>,
    num_buffers: u64,
    produced: u64,
    /// Advertise a CPU cap besides the External one (`external_preferred`
    /// shape) instead of External-only.
    cpu_fallback_cap: bool,
    negotiated: Option<MemoryType>,
    /// CPU-path output arena.
    output: OutputArena,
}

impl ExternalTestSrc {
    /// A pool of `pool_size` strided frames, ending after `num_buffers`.
    ///
    /// Pool slot `i` carries the pattern of [`packed_reference_frame`]`(i)`
    /// and keeps it across recycles; the emitted sequence number maps to
    /// the slot as `seq % pool_size`, so consumers verify against
    /// `packed_reference_frame(seq % pool_size as u64)`.
    pub fn new(pool_size: u32, num_buffers: u64) -> Self {
        let frames = (0..pool_size)
            .map(|i| Arc::new(strided_frame(i as u64)))
            .collect();
        let (release_tx, release_rx) = channel();
        Self {
            frames,
            pool_size,
            available: (0..pool_size).collect(),
            release_rx,
            release_tx,
            num_buffers,
            produced: 0,
            cpu_fallback_cap: false,
            negotiated: None,
            output: OutputArena::new(defaults::SOURCE_SLOT_COUNT).grow_to_fit(),
        }
    }

    /// Also advertise a CPU cap (the `external_preferred` shape).
    pub fn with_cpu_fallback_cap(mut self, fallback: bool) -> Self {
        self.cpu_fallback_cap = fallback;
        self
    }

    /// Whether this source will emit External (post-negotiation).
    pub fn emits_external(&self) -> bool {
        self.negotiated == Some(MemoryType::External)
    }

    /// The reference pattern the buffer with sequence `seq` carries.
    pub fn reference_for_sequence(&self, seq: u64) -> Vec<u8> {
        packed_reference_frame(seq % self.pool_size as u64)
    }
}

impl Source for ExternalTestSrc {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.produced >= self.num_buffers {
            return Ok(ProduceResult::Eos);
        }

        // Recycle frames whose last reference dropped downstream.
        while let Ok(index) = self.release_rx.try_recv() {
            self.available.push_back(index);
        }

        let mut metadata = Metadata::from_sequence(self.produced);
        metadata.pts = crate::clock::ClockTime::from_millis(self.produced * 10);

        if self.negotiated == Some(MemoryType::External) {
            // Sequence maps to pool slot; refuse to reuse a pinned frame.
            let index = (self.produced % self.pool_size as u64) as u32;
            let Some(pos) = self.available.iter().position(|&i| i == index) else {
                return Ok(ProduceResult::WouldBlock);
            };
            self.available.remove(pos);

            let frame = Arc::clone(&self.frames[index as usize]);
            let ptr = frame.as_ptr();
            let len = frame.len();
            // SAFETY: `frame` (the owner clone) keeps the boxed slice
            // alive for the slot's whole life; External buffers are
            // read-only so the span is never written through.
            let slot = unsafe {
                ExternalSlot::with_release(
                    ptr,
                    len,
                    frame,
                    "external-test-src",
                    Box::new({
                        let tx = self.release_tx.clone();
                        move || {
                            let _ = tx.send(index);
                        }
                    }),
                )
            };
            metadata.set_video_planes(
                TEST_WIDTH,
                TEST_HEIGHT,
                PixelFormat::I420,
                strided_test_layout(),
            );
            self.produced += 1;
            return Ok(ProduceResult::OwnBuffer(Buffer::new(
                MemoryHandle::from_external(Arc::new(slot)),
                metadata,
            )));
        }

        // CPU path (no negotiation, or the link negotiated Cpu): packed.
        let packed = packed_reference_frame(self.produced % self.pool_size as u64);
        let Some(mut slot) = self.output.try_acquire(packed.len(), "externaltestsrc")? else {
            return Ok(ProduceResult::WouldBlock);
        };
        slot.data_mut()[..packed.len()].copy_from_slice(&packed);
        metadata.set_video_dims(TEST_WIDTH, TEST_HEIGHT, PixelFormat::I420);
        let len = packed.len();
        self.produced += 1;
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, len),
            metadata,
        )))
    }

    fn set_negotiated_memory(&mut self, memory: MemoryType) {
        self.negotiated = Some(memory);
    }

    fn output_media_caps(&self) -> ElementMediaCaps {
        let format = FormatCaps::VideoRaw(VideoFormatCaps {
            width: CapsValue::Fixed(TEST_WIDTH),
            height: CapsValue::Fixed(TEST_HEIGHT),
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            framerate: CapsValue::Any,
            layout: crate::format::MemoryLayout::NONE,
        });
        if self.cpu_fallback_cap {
            ElementMediaCaps::new([FormatMemoryCap::new(
                format,
                MemoryCaps::external_preferred(),
            )])
        } else {
            ElementMediaCaps::new([FormatMemoryCap::new(
                format,
                MemoryCaps {
                    types: CapsValue::Fixed(MemoryType::External),
                    can_import: vec![],
                    can_export: vec![MemoryType::External],
                },
            )])
        }
    }

    fn name(&self) -> &str {
        "externaltestsrc"
    }
}
