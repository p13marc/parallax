//! A dmabuf-emitting test source (#145).
//!
//! Exercises the DmaBuf flow-through path without hardware: a pool of
//! memfd-backed [`DmaBufSegment`]s stands in for a driver's exported
//! buffers (same mmap semantics; a real dma-buf fd differs only in its
//! kernel f_ops, covered by the udmabuf unit test). The producer-side
//! discipline is the real one — a fixed pool, a release channel fed by
//! each slot's last-drop hook, `WouldBlock` when every slot is in flight —
//! so tests can pin recycling, negotiation gating, and end-to-end
//! `MemoryType::DmaBuf` delivery.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ProduceContext, ProduceResult, Source};
use crate::error::Result;
use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
use crate::memory::{DmaBufSegment, DmaBufSlot, MemoryType, OutputArena, defaults};
use crate::metadata::Metadata;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

/// Test source producing dmabuf-backed buffers from a fixed pool.
///
/// Each pool slot's payload is pre-filled with its index byte, so a
/// consumer can verify which slot it received. Emits DmaBuf handles only
/// when the negotiated memory type says so (`set_negotiated_memory`);
/// otherwise it produces ordinary CPU buffers — exactly the gate a real
/// dmabuf-capable source applies.
pub struct DmaBufTestSrc {
    /// One long-lived mapping per pool slot, shared with in-flight slots.
    segments: Vec<Arc<DmaBufSegment>>,
    /// Indices currently free to hand out.
    available: VecDeque<u32>,
    /// Fed by each slot's last-drop release hook.
    release_rx: Receiver<u32>,
    release_tx: Sender<u32>,
    frame_len: usize,
    num_buffers: u64,
    produced: u64,
    /// Advertise a CPU cap besides the DmaBuf one (`dmabuf_preferred`
    /// shape) instead of DmaBuf-only.
    cpu_fallback_cap: bool,
    /// What the downstream link negotiated; `None` (no negotiation ran)
    /// means CPU, the safe default.
    negotiated: Option<MemoryType>,
    /// CPU-path output arena (negotiated CPU + no executor-provided slot).
    output: OutputArena,
}

impl DmaBufTestSrc {
    /// A pool of `pool_size` memfd-backed frames of `frame_len` bytes,
    /// ending after `num_buffers` produced frames.
    pub fn new(frame_len: usize, pool_size: u32, num_buffers: u64) -> Result<Self> {
        let mut segments = Vec::with_capacity(pool_size as usize);
        for index in 0..pool_size {
            let fd = rustix::fs::memfd_create("dmabuf_test_src", rustix::fs::MemfdFlags::CLOEXEC)
                .map_err(|e| crate::error::Error::AllocationFailed(e.to_string()))?;
            rustix::fs::ftruncate(&fd, frame_len as u64)
                .map_err(|e| crate::error::Error::AllocationFailed(e.to_string()))?;
            let mut segment = DmaBufSegment::from_fd(fd, frame_len)?;
            // Payload identifies the slot, so tests can see recycling.
            segment
                .as_mut_slice()
                .expect("fresh mapping is writable")
                .fill(index as u8);
            segments.push(Arc::new(segment));
        }
        let (release_tx, release_rx) = channel();
        Ok(Self {
            segments,
            available: (0..pool_size).collect(),
            release_rx,
            release_tx,
            frame_len,
            num_buffers,
            produced: 0,
            cpu_fallback_cap: false,
            negotiated: None,
            output: OutputArena::new(defaults::SOURCE_SLOT_COUNT).grow_to_fit(),
        })
    }

    /// Also advertise a CPU cap (the `dmabuf_preferred` shape): against a
    /// CPU-only consumer negotiation then picks CPU directly instead of
    /// inserting a `memorycopy`.
    pub fn with_cpu_fallback_cap(mut self, fallback: bool) -> Self {
        self.cpu_fallback_cap = fallback;
        self
    }

    /// Whether this source will emit dmabuf (post-negotiation).
    pub fn emits_dmabuf(&self) -> bool {
        self.negotiated == Some(MemoryType::DmaBuf)
    }
}

impl Source for DmaBufTestSrc {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.produced >= self.num_buffers {
            return Ok(ProduceResult::Eos);
        }

        // Recycle slots whose last reference dropped downstream.
        while let Ok(index) = self.release_rx.try_recv() {
            self.available.push_back(index);
        }

        let mut metadata = Metadata::from_sequence(self.produced);
        metadata.pts = crate::clock::ClockTime::from_millis(self.produced * 10);

        if self.negotiated == Some(MemoryType::DmaBuf) {
            // Pool exhausted downstream: the consumers hold every slot.
            let Some(index) = self.available.pop_front() else {
                return Ok(ProduceResult::WouldBlock);
            };
            let slot = Arc::new(DmaBufSlot::with_release(
                Arc::clone(&self.segments[index as usize]),
                index,
                Box::new({
                    let tx = self.release_tx.clone();
                    move |idx| {
                        let _ = tx.send(idx);
                    }
                }),
            ));
            self.produced += 1;
            return Ok(ProduceResult::OwnBuffer(Buffer::new(
                MemoryHandle::from_dmabuf(slot),
                metadata,
            )));
        }

        // CPU path (no negotiation, or the link negotiated Cpu).
        let Some(slot) = self.output.try_acquire(self.frame_len, "dmabuftestsrc")? else {
            return Ok(ProduceResult::WouldBlock);
        };
        self.produced += 1;
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, self.frame_len),
            metadata,
        )))
    }

    fn set_negotiated_memory(&mut self, memory: MemoryType) {
        self.negotiated = Some(memory);
    }

    fn output_media_caps(&self) -> ElementMediaCaps {
        let mut caps = vec![FormatMemoryCap::new(
            FormatCaps::Any,
            MemoryCaps::dmabuf_only(),
        )];
        if self.cpu_fallback_cap {
            caps.push(FormatMemoryCap::new(
                FormatCaps::Any,
                MemoryCaps::cpu_only(),
            ));
        }
        ElementMediaCaps::new(caps)
    }

    fn name(&self) -> &str {
        "dmabuftestsrc"
    }
}
