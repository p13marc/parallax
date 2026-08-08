//! H.264 picture-order-count and reference-picture bookkeeping.
//!
//! Pure decode-order logic — no Vulkan handles anywhere — so all of it is
//! unit-testable without a video-capable device. The decoder feeds it parsed
//! slice headers; it answers the three questions hardware decode needs
//! answered per picture:
//!
//! 1. what is this picture's POC (spec 8.2.1, types 0/1/2),
//! 2. which DPB slots make up reference lists L0/L1 (8.2.4.2, default
//!    initialisation only — `ref_pic_list_modification` is rejected upstream),
//! 3. which references survive after this picture (8.2.5: IDR handling,
//!    sliding window, and adaptive MMCO marking).
//!
//! Scope matches the decoder: progressive frames only. Fields/MBAFF never get
//! here — the slice parser rejects `field_pic_flag` before this module runs.

use h264_reader::nal::slice::{DecRefPicMarking, MemoryManagementControlOperation};

/// SPS-derived constants POC computation needs, extracted once per SPS.
#[derive(Debug, Clone, Default)]
pub struct PocParams {
    /// `pic_order_cnt_type` (0, 1 or 2).
    pub poc_type: u8,
    /// `1 << (log2_max_pic_order_cnt_lsb_minus4 + 4)` — POC type 0 only.
    pub max_pic_order_cnt_lsb: u32,
    /// `1 << (log2_max_frame_num_minus4 + 4)`.
    pub max_frame_num: u32,
    /// POC type 1: `delta_pic_order_always_zero_flag`.
    pub delta_pic_order_always_zero: bool,
    /// POC type 1: `offset_for_non_ref_pic`.
    pub offset_for_non_ref_pic: i32,
    /// POC type 1: `offset_for_top_to_bottom_field`.
    pub offset_for_top_to_bottom_field: i32,
    /// POC type 1: `offset_for_ref_frame[]`.
    pub offsets_for_ref_frame: Vec<i32>,
}

/// What the POC machinery needs to know about the current picture.
#[derive(Debug, Clone, Copy, Default)]
pub struct PictureId {
    /// `frame_num` from the slice header.
    pub frame_num: u32,
    /// Whether this is an IDR picture.
    pub is_idr: bool,
    /// Whether `nal_ref_idc != 0` (the picture becomes a reference).
    pub is_reference: bool,
    /// POC type 0: `pic_order_cnt_lsb`.
    pub pic_order_cnt_lsb: u32,
    /// POC type 0: `delta_pic_order_cnt_bottom` (0 unless the PPS sends it).
    pub delta_pic_order_cnt_bottom: i32,
    /// POC type 1: `delta_pic_order_cnt[0]`.
    pub delta_pic_order_cnt_0: i32,
    /// POC type 1: `delta_pic_order_cnt[1]`.
    pub delta_pic_order_cnt_1: i32,
}

/// Computed picture order counts for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FramePoc {
    /// `TopFieldOrderCnt`.
    pub top: i32,
    /// `BottomFieldOrderCnt`.
    pub bottom: i32,
}

impl FramePoc {
    /// `PicOrderCnt` of a frame: `min(top, bottom)` (spec 8.2.1 note).
    pub fn poc(&self) -> i32 {
        self.top.min(self.bottom)
    }
}

/// Rolling POC decoder state (spec 8.2.1).
#[derive(Debug, Default)]
pub struct PocState {
    prev_pic_order_cnt_msb: i32,
    prev_pic_order_cnt_lsb: u32,
    prev_frame_num: u32,
    prev_frame_num_offset: u32,
}

impl PocState {
    /// Fresh state (as after an IDR).
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset, as `reset()`/IDR-with-MMCO5 requires.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Compute this frame's POC and update the rolling state.
    ///
    /// Must be called exactly once per picture, in decode order.
    pub fn advance(&mut self, params: &PocParams, pic: &PictureId) -> FramePoc {
        match params.poc_type {
            0 => self.advance_type0(params, pic),
            1 => self.advance_type1(params, pic),
            _ => self.advance_type2(params, pic),
        }
    }

    fn advance_type0(&mut self, params: &PocParams, pic: &PictureId) -> FramePoc {
        if pic.is_idr {
            self.prev_pic_order_cnt_msb = 0;
            self.prev_pic_order_cnt_lsb = 0;
        }

        let max_lsb = params.max_pic_order_cnt_lsb as i32;
        let lsb = pic.pic_order_cnt_lsb as i32;
        let prev_lsb = self.prev_pic_order_cnt_lsb as i32;

        let msb = if lsb < prev_lsb && (prev_lsb - lsb) >= max_lsb / 2 {
            self.prev_pic_order_cnt_msb + max_lsb
        } else if lsb > prev_lsb && (lsb - prev_lsb) > max_lsb / 2 {
            self.prev_pic_order_cnt_msb - max_lsb
        } else {
            self.prev_pic_order_cnt_msb
        };

        let top = msb + lsb;
        let bottom = top + pic.delta_pic_order_cnt_bottom;

        if pic.is_reference {
            self.prev_pic_order_cnt_msb = msb;
            self.prev_pic_order_cnt_lsb = pic.pic_order_cnt_lsb;
        }

        FramePoc { top, bottom }
    }

    fn frame_num_offset(&mut self, params: &PocParams, pic: &PictureId) -> u32 {
        let offset = if pic.is_idr {
            0
        } else if self.prev_frame_num > pic.frame_num {
            self.prev_frame_num_offset + params.max_frame_num
        } else {
            self.prev_frame_num_offset
        };
        self.prev_frame_num_offset = offset;
        self.prev_frame_num = pic.frame_num;
        offset
    }

    fn advance_type1(&mut self, params: &PocParams, pic: &PictureId) -> FramePoc {
        let frame_num_offset = self.frame_num_offset(params, pic);
        let num_refs = params.offsets_for_ref_frame.len() as u32;

        let mut abs_frame_num = if num_refs != 0 {
            frame_num_offset + pic.frame_num
        } else {
            0
        };
        if !pic.is_reference && abs_frame_num > 0 {
            abs_frame_num -= 1;
        }

        let expected_delta_per_cycle: i32 = params.offsets_for_ref_frame.iter().sum();

        let mut expected_poc = 0i32;
        if abs_frame_num > 0 {
            let cycle = (abs_frame_num - 1) / num_refs;
            let in_cycle = (abs_frame_num - 1) % num_refs;
            expected_poc = cycle as i32 * expected_delta_per_cycle;
            for offset in &params.offsets_for_ref_frame[..=in_cycle as usize] {
                expected_poc += offset;
            }
        }
        if !pic.is_reference {
            expected_poc += params.offset_for_non_ref_pic;
        }

        let (d0, d1) = if params.delta_pic_order_always_zero {
            (0, 0)
        } else {
            (pic.delta_pic_order_cnt_0, pic.delta_pic_order_cnt_1)
        };

        let top = expected_poc + d0;
        let bottom = top + params.offset_for_top_to_bottom_field + d1;
        FramePoc { top, bottom }
    }

    fn advance_type2(&mut self, params: &PocParams, pic: &PictureId) -> FramePoc {
        let frame_num_offset = self.frame_num_offset(params, pic);
        let temp_poc = if pic.is_idr {
            0
        } else {
            let base = 2 * (frame_num_offset + pic.frame_num) as i32;
            if pic.is_reference { base } else { base - 1 }
        };
        FramePoc {
            top: temp_poc,
            bottom: temp_poc,
        }
    }
}

/// A reference picture the tracker knows about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefPicture {
    /// DPB slot holding its pixels.
    pub slot_index: u32,
    /// `frame_num` as coded.
    pub frame_num: u32,
    /// Frame POC (top field order count; frames only).
    pub poc_top: i32,
    /// Bottom field order count.
    pub poc_bottom: i32,
    /// `LongTermFrameIdx` when marked long-term.
    pub long_term_frame_idx: Option<u32>,
}

impl RefPicture {
    /// Whether the picture is a long-term reference.
    pub fn is_long_term(&self) -> bool {
        self.long_term_frame_idx.is_some()
    }

    /// Frame POC (`min(top, bottom)`).
    pub fn poc(&self) -> i32 {
        self.poc_top.min(self.poc_bottom)
    }

    fn frame_num_wrap(&self, current_frame_num: u32, max_frame_num: u32) -> i32 {
        if self.frame_num > current_frame_num {
            self.frame_num as i32 - max_frame_num as i32
        } else {
            self.frame_num as i32
        }
    }
}

/// Everything the tracker needs to know about the just-decoded picture in
/// order to mark references (spec 8.2.5).
#[derive(Debug, Clone, Copy)]
pub struct CurrentPicture {
    /// DPB slot the picture was decoded into.
    pub slot_index: u32,
    /// `frame_num`.
    pub frame_num: u32,
    /// Computed POC.
    pub poc: FramePoc,
    /// Whether `nal_ref_idc != 0`.
    pub is_reference: bool,
    /// Whether the picture is an IDR.
    pub is_idr: bool,
    /// IDR only: `long_term_reference_flag`.
    pub idr_long_term: bool,
}

/// Decoded-reference bookkeeping: who is a reference, and in what order they
/// enter prediction lists.
#[derive(Debug, Default)]
pub struct RefTracker {
    refs: Vec<RefPicture>,
    max_long_term_frame_idx: Option<u32>,
}

impl RefTracker {
    /// Empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// The current reference set (unspecified order).
    pub fn references(&self) -> &[RefPicture] {
        &self.refs
    }

    /// Drop everything (decoder reset).
    pub fn clear(&mut self) -> Vec<u32> {
        self.max_long_term_frame_idx = None;
        self.refs.drain(..).map(|r| r.slot_index).collect()
    }

    /// Default reference list for a P slice (8.2.4.2.1): short-term by
    /// descending `PicNum` (= `FrameNumWrap` for frames), then long-term by
    /// ascending `LongTermFrameIdx`.
    pub fn list_p(&self, current_frame_num: u32, max_frame_num: u32) -> Vec<RefPicture> {
        let mut short: Vec<_> = self
            .refs
            .iter()
            .copied()
            .filter(|r| !r.is_long_term())
            .collect();
        short
            .sort_by_key(|r| std::cmp::Reverse(r.frame_num_wrap(current_frame_num, max_frame_num)));

        let mut long: Vec<_> = self
            .refs
            .iter()
            .copied()
            .filter(|r| r.is_long_term())
            .collect();
        long.sort_by_key(|r| r.long_term_frame_idx);

        short.into_iter().chain(long).collect()
    }

    /// Default reference lists for a B slice (8.2.4.2.3).
    ///
    /// L0: short-term with POC < current (descending POC), then POC > current
    /// (ascending), then long-term. L1: the mirror. Frames only.
    pub fn lists_b(&self, current_poc: i32) -> (Vec<RefPicture>, Vec<RefPicture>) {
        let mut before: Vec<_> = self
            .refs
            .iter()
            .copied()
            .filter(|r| !r.is_long_term() && r.poc() <= current_poc)
            .collect();
        before.sort_by_key(|r| std::cmp::Reverse(r.poc()));

        let mut after: Vec<_> = self
            .refs
            .iter()
            .copied()
            .filter(|r| !r.is_long_term() && r.poc() > current_poc)
            .collect();
        after.sort_by_key(|r| r.poc());

        let mut long: Vec<_> = self
            .refs
            .iter()
            .copied()
            .filter(|r| r.is_long_term())
            .collect();
        long.sort_by_key(|r| r.long_term_frame_idx);

        let l0: Vec<_> = before
            .iter()
            .chain(after.iter())
            .chain(long.iter())
            .copied()
            .collect();
        let l1: Vec<_> = after
            .iter()
            .chain(before.iter())
            .chain(long.iter())
            .copied()
            .collect();

        // 8.2.4.2.3: if L1 would equal L0 and has more than one entry, swap
        // its first two entries.
        let mut l1 = l1;
        if l1.len() > 1 && l1 == l0 {
            l1.swap(0, 1);
        }
        (l0, l1)
    }

    /// Apply decoded-reference-picture marking after decoding `current`
    /// (spec 8.2.5). Returns the DPB slots that stopped being references and
    /// can be released.
    ///
    /// `max_num_ref_frames` comes from the SPS; `max_frame_num` is
    /// `1 << (log2_max_frame_num_minus4 + 4)`.
    pub fn mark(
        &mut self,
        current: CurrentPicture,
        marking: Option<&DecRefPicMarking>,
        max_num_ref_frames: u32,
        max_frame_num: u32,
    ) -> Vec<u32> {
        let mut released = Vec::new();

        if current.is_idr {
            released.extend(self.refs.drain(..).map(|r| r.slot_index));
            self.max_long_term_frame_idx = None;
            if current.is_reference {
                self.refs.push(RefPicture {
                    slot_index: current.slot_index,
                    frame_num: current.frame_num,
                    poc_top: current.poc.top,
                    poc_bottom: current.poc.bottom,
                    long_term_frame_idx: current.idr_long_term.then_some(0),
                });
                if current.idr_long_term {
                    self.max_long_term_frame_idx = Some(0);
                }
            }
            return released;
        }

        if !current.is_reference {
            return released;
        }

        let mut current_long_term_idx = None;
        match marking {
            Some(DecRefPicMarking::Adaptive(ops)) => {
                for op in ops {
                    self.apply_mmco(
                        op,
                        &current,
                        max_frame_num,
                        &mut released,
                        &mut current_long_term_idx,
                    );
                }
                // MMCO alone may still leave the window over-full.
                self.sliding_window(
                    max_num_ref_frames,
                    current.frame_num,
                    max_frame_num,
                    &mut released,
                );
            }
            _ => {
                // SlidingWindow (or an implied one when no marking is present).
                self.sliding_window(
                    max_num_ref_frames,
                    current.frame_num,
                    max_frame_num,
                    &mut released,
                );
            }
        }

        self.refs.push(RefPicture {
            slot_index: current.slot_index,
            frame_num: current.frame_num,
            poc_top: current.poc.top,
            poc_bottom: current.poc.bottom,
            long_term_frame_idx: current_long_term_idx,
        });

        released
    }

    /// 8.2.5.3: evict the short-term reference with the smallest
    /// `FrameNumWrap` while the window is full.
    fn sliding_window(
        &mut self,
        max_num_ref_frames: u32,
        current_frame_num: u32,
        max_frame_num: u32,
        released: &mut Vec<u32>,
    ) {
        let capacity = max_num_ref_frames.max(1) as usize;
        // The current picture is about to be inserted, so make room down to
        // capacity - 1.
        while self.refs.len() >= capacity {
            let Some((pos, _)) = self
                .refs
                .iter()
                .enumerate()
                .filter(|(_, r)| !r.is_long_term())
                .min_by_key(|(_, r)| r.frame_num_wrap(current_frame_num, max_frame_num))
            else {
                // Only long-term references left: nothing the sliding window
                // may remove. Bail rather than loop forever.
                break;
            };
            released.push(self.refs.remove(pos).slot_index);
        }
    }

    fn apply_mmco(
        &mut self,
        op: &MemoryManagementControlOperation,
        current: &CurrentPicture,
        max_frame_num: u32,
        released: &mut Vec<u32>,
        current_long_term_idx: &mut Option<u32>,
    ) {
        match op {
            MemoryManagementControlOperation::ShortTermUnusedForRef {
                difference_of_pic_nums_minus1,
            } => {
                // 8.2.5.4.1: PicNumX = CurrPicNum - (difference + 1).
                let pic_num_x =
                    current.frame_num as i64 - (*difference_of_pic_nums_minus1 as i64 + 1);
                if let Some(pos) = self.refs.iter().position(|r| {
                    !r.is_long_term()
                        && r.frame_num_wrap(current.frame_num, max_frame_num) as i64 == pic_num_x
                }) {
                    released.push(self.refs.remove(pos).slot_index);
                }
            }
            MemoryManagementControlOperation::LongTermUnusedForRef { long_term_pic_num } => {
                if let Some(pos) = self
                    .refs
                    .iter()
                    .position(|r| r.long_term_frame_idx == Some(*long_term_pic_num))
                {
                    released.push(self.refs.remove(pos).slot_index);
                }
            }
            MemoryManagementControlOperation::ShortTermUsedForLongTerm {
                difference_of_pic_nums_minus1,
                long_term_frame_idx,
            } => {
                let pic_num_x =
                    current.frame_num as i64 - (*difference_of_pic_nums_minus1 as i64 + 1);
                // A long-term ref already holding the target index is replaced.
                if let Some(pos) = self
                    .refs
                    .iter()
                    .position(|r| r.long_term_frame_idx == Some(*long_term_frame_idx))
                {
                    released.push(self.refs.remove(pos).slot_index);
                }
                if let Some(r) = self.refs.iter_mut().find(|r| {
                    !r.is_long_term()
                        && r.frame_num_wrap(current.frame_num, max_frame_num) as i64 == pic_num_x
                }) {
                    r.long_term_frame_idx = Some(*long_term_frame_idx);
                }
            }
            MemoryManagementControlOperation::MaxUsedLongTermFrameRef {
                max_long_term_frame_idx_plus1,
            } => {
                self.max_long_term_frame_idx = max_long_term_frame_idx_plus1.checked_sub(1);
                let max = self.max_long_term_frame_idx;
                self.refs.retain(|r| match r.long_term_frame_idx {
                    Some(idx) => match max {
                        Some(m) if idx <= m => true,
                        _ => {
                            released.push(r.slot_index);
                            false
                        }
                    },
                    None => true,
                });
            }
            MemoryManagementControlOperation::AllRefPicturesUnused => {
                released.extend(self.refs.drain(..).map(|r| r.slot_index));
                self.max_long_term_frame_idx = None;
            }
            MemoryManagementControlOperation::CurrentUsedForLongTerm {
                long_term_frame_idx,
            } => {
                if let Some(pos) = self
                    .refs
                    .iter()
                    .position(|r| r.long_term_frame_idx == Some(*long_term_frame_idx))
                {
                    released.push(self.refs.remove(pos).slot_index);
                }
                *current_long_term_idx = Some(*long_term_frame_idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params_type0(log2_lsb: u8) -> PocParams {
        PocParams {
            poc_type: 0,
            max_pic_order_cnt_lsb: 1 << (log2_lsb + 4),
            max_frame_num: 1 << 8,
            ..Default::default()
        }
    }

    fn pic(frame_num: u32, lsb: u32, is_idr: bool, is_reference: bool) -> PictureId {
        PictureId {
            frame_num,
            is_idr,
            is_reference,
            pic_order_cnt_lsb: lsb,
            ..Default::default()
        }
    }

    #[test]
    fn poc_type0_monotonic_sequence() {
        let params = params_type0(4); // max_lsb = 256
        let mut state = PocState::new();

        assert_eq!(state.advance(&params, &pic(0, 0, true, true)).poc(), 0);
        assert_eq!(state.advance(&params, &pic(1, 2, false, true)).poc(), 2);
        assert_eq!(state.advance(&params, &pic(2, 4, false, true)).poc(), 4);
    }

    #[test]
    fn poc_type0_lsb_wraparound_moves_msb_up() {
        let params = params_type0(0); // max_lsb = 16
        let mut state = PocState::new();

        // Forward steps smaller than max/2, then a wrap: 0 → 6 → 12 → 2.
        state.advance(&params, &pic(0, 0, true, true));
        state.advance(&params, &pic(1, 6, false, true));
        state.advance(&params, &pic(2, 12, false, true));
        // prev - new = 10 >= max/2 = 8 → msb += 16.
        let poc = state.advance(&params, &pic(3, 2, false, true));
        assert_eq!(poc.poc(), 16 + 2);
    }

    #[test]
    fn poc_type0_backward_jump_moves_msb_down() {
        let params = params_type0(0); // max_lsb = 16
        let mut state = PocState::new();

        // Wrap forward to msb = 16 first (0 → 6 → 12 → 2)...
        state.advance(&params, &pic(0, 0, true, true));
        state.advance(&params, &pic(1, 6, false, true));
        state.advance(&params, &pic(2, 12, false, true));
        state.advance(&params, &pic(3, 2, false, true));
        // ...then a backward reorder to lsb 15: 15 - 2 = 13 > 8 → msb -= 16.
        let poc = state.advance(&params, &pic(4, 15, false, true));
        assert_eq!(poc.poc(), 15);
    }

    #[test]
    fn poc_type0_idr_resets_state() {
        let params = params_type0(0);
        let mut state = PocState::new();

        state.advance(&params, &pic(0, 0, true, true));
        state.advance(&params, &pic(1, 8, false, true));
        let poc = state.advance(&params, &pic(0, 0, true, true));
        assert_eq!(poc.poc(), 0, "IDR restarts POC from zero");
    }

    #[test]
    fn poc_type0_non_reference_does_not_update_prev() {
        let params = params_type0(0); // max 16
        let mut state = PocState::new();

        state.advance(&params, &pic(0, 0, true, true));
        // Non-reference picture at lsb 14 — must not shift the prev anchor.
        state.advance(&params, &pic(1, 14, false, false));
        // From the anchor at 0, lsb 4 is a plain forward step, no wrap.
        assert_eq!(state.advance(&params, &pic(1, 4, false, true)).poc(), 4);
    }

    #[test]
    fn poc_type1_with_ref_offsets() {
        // One offset of +2 per cycle: POC advances by 2 per reference frame.
        let params = PocParams {
            poc_type: 1,
            max_frame_num: 16,
            offsets_for_ref_frame: vec![2],
            ..Default::default()
        };
        let mut state = PocState::new();

        assert_eq!(state.advance(&params, &pic(0, 0, true, true)).poc(), 0);
        assert_eq!(state.advance(&params, &pic(1, 0, false, true)).poc(), 2);
        assert_eq!(state.advance(&params, &pic(2, 0, false, true)).poc(), 4);
    }

    #[test]
    fn poc_type2_is_twice_decode_order() {
        let params = PocParams {
            poc_type: 2,
            max_frame_num: 16,
            ..Default::default()
        };
        let mut state = PocState::new();

        assert_eq!(state.advance(&params, &pic(0, 0, true, true)).poc(), 0);
        assert_eq!(state.advance(&params, &pic(1, 0, false, true)).poc(), 2);
        // Non-reference pictures sit one below 2*frame_num (8.2.1.3).
        assert_eq!(state.advance(&params, &pic(2, 0, false, false)).poc(), 3);
    }

    #[test]
    fn poc_type2_frame_num_wrap_keeps_counting() {
        let params = PocParams {
            poc_type: 2,
            max_frame_num: 4,
            ..Default::default()
        };
        let mut state = PocState::new();
        for (frame_num, expected) in [(0u32, 0i32), (1, 2), (2, 4), (3, 6), (0, 8), (1, 10)] {
            let p = pic(frame_num, 0, frame_num == 0 && expected == 0, true);
            assert_eq!(state.advance(&params, &p).poc(), expected);
        }
    }

    fn strf(slot: u32, frame_num: u32, poc: i32) -> RefPicture {
        RefPicture {
            slot_index: slot,
            frame_num,
            poc_top: poc,
            poc_bottom: poc,
            long_term_frame_idx: None,
        }
    }

    fn cur(slot: u32, frame_num: u32, poc: i32) -> CurrentPicture {
        CurrentPicture {
            slot_index: slot,
            frame_num,
            poc: FramePoc {
                top: poc,
                bottom: poc,
            },
            is_reference: true,
            is_idr: false,
            idr_long_term: false,
        }
    }

    #[test]
    fn p_list_orders_short_term_by_descending_pic_num() {
        let mut t = RefTracker::new();
        t.refs = vec![strf(0, 1, 2), strf(1, 3, 6), strf(2, 2, 4)];

        let l0 = t.list_p(4, 256);
        let nums: Vec<_> = l0.iter().map(|r| r.frame_num).collect();
        assert_eq!(nums, vec![3, 2, 1], "most recent first");
    }

    #[test]
    fn p_list_puts_long_term_last_by_index() {
        let mut t = RefTracker::new();
        t.refs = vec![
            RefPicture {
                long_term_frame_idx: Some(1),
                ..strf(0, 0, 0)
            },
            strf(1, 3, 6),
            RefPicture {
                long_term_frame_idx: Some(0),
                ..strf(2, 1, 2)
            },
        ];

        let l0 = t.list_p(4, 256);
        let slots: Vec<_> = l0.iter().map(|r| r.slot_index).collect();
        assert_eq!(
            slots,
            vec![1, 2, 0],
            "short-term, then LT idx 0, then LT idx 1"
        );
    }

    #[test]
    fn b_lists_split_around_current_poc() {
        let mut t = RefTracker::new();
        t.refs = vec![strf(0, 1, 2), strf(1, 2, 4), strf(2, 3, 8), strf(3, 4, 10)];

        let (l0, l1) = t.lists_b(6);
        let l0_pocs: Vec<_> = l0.iter().map(|r| r.poc()).collect();
        let l1_pocs: Vec<_> = l1.iter().map(|r| r.poc()).collect();
        assert_eq!(l0_pocs, vec![4, 2, 8, 10], "past desc, then future asc");
        assert_eq!(l1_pocs, vec![8, 10, 4, 2], "future asc, then past desc");
    }

    #[test]
    fn sliding_window_evicts_oldest_short_term() {
        let mut t = RefTracker::new();
        t.refs = vec![strf(0, 1, 2), strf(1, 2, 4)];

        let released = t.mark(cur(2, 3, 6), Some(&DecRefPicMarking::SlidingWindow), 2, 256);
        assert_eq!(released, vec![0], "frame_num 1 was the oldest");
        assert_eq!(t.references().len(), 2);
        assert!(t.references().iter().any(|r| r.slot_index == 2));
    }

    #[test]
    fn sliding_window_never_evicts_long_term() {
        let mut t = RefTracker::new();
        t.refs = vec![
            RefPicture {
                long_term_frame_idx: Some(0),
                ..strf(0, 0, 0)
            },
            strf(1, 2, 4),
        ];

        let released = t.mark(cur(2, 3, 6), Some(&DecRefPicMarking::SlidingWindow), 2, 256);
        assert_eq!(
            released,
            vec![1],
            "the short-term ref goes, not the long-term"
        );
    }

    #[test]
    fn idr_flushes_everything() {
        let mut t = RefTracker::new();
        t.refs = vec![strf(0, 1, 2), strf(1, 2, 4)];

        let mut idr = cur(2, 0, 0);
        idr.is_idr = true;
        let mut released = t.mark(idr, None, 4, 256);
        released.sort_unstable();
        assert_eq!(released, vec![0, 1]);
        assert_eq!(t.references().len(), 1);
        assert_eq!(t.references()[0].slot_index, 2);
    }

    #[test]
    fn non_reference_picture_changes_nothing() {
        let mut t = RefTracker::new();
        t.refs = vec![strf(0, 1, 2)];

        let mut b = cur(5, 2, 3);
        b.is_reference = false;
        let released = t.mark(b, None, 4, 256);
        assert!(released.is_empty());
        assert_eq!(t.references().len(), 1);
    }

    #[test]
    fn mmco1_removes_the_named_short_term() {
        let mut t = RefTracker::new();
        t.refs = vec![strf(0, 1, 2), strf(1, 2, 4)];

        // CurrPicNum = 3; difference_of_pic_nums_minus1 = 1 → PicNumX = 1.
        let ops = DecRefPicMarking::Adaptive(vec![
            MemoryManagementControlOperation::ShortTermUnusedForRef {
                difference_of_pic_nums_minus1: 1,
            },
        ]);
        let released = t.mark(cur(2, 3, 6), Some(&ops), 4, 256);
        assert_eq!(released, vec![0]);
    }

    #[test]
    fn mmco6_marks_current_long_term() {
        let mut t = RefTracker::new();
        let ops = DecRefPicMarking::Adaptive(vec![
            MemoryManagementControlOperation::CurrentUsedForLongTerm {
                long_term_frame_idx: 0,
            },
        ]);
        t.mark(cur(0, 0, 0), Some(&ops), 4, 256);
        assert_eq!(t.references()[0].long_term_frame_idx, Some(0));
    }

    #[test]
    fn mmco5_clears_all_references() {
        let mut t = RefTracker::new();
        t.refs = vec![strf(0, 1, 2), strf(1, 2, 4)];
        let ops = DecRefPicMarking::Adaptive(vec![
            MemoryManagementControlOperation::AllRefPicturesUnused,
        ]);
        let mut released = t.mark(cur(2, 3, 6), Some(&ops), 4, 256);
        released.sort_unstable();
        assert_eq!(released, vec![0, 1]);
        assert_eq!(t.references().len(), 1, "only the current picture remains");
    }
}
