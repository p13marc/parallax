//! An H.265 RTP depacketizer that actually produces NAL units (RFC 7798).
//!
//! # Why this exists
//!
//! `rtp` 0.17's `H265Packet::depacketize` parses the packet into `self.payload`
//! for inspection and then returns `Ok(payload.clone())` on **every** branch
//! (`rtp-0.17.2/src/codecs/h265/mod.rs:999-1040`). So what came out of
//! `RtpH265Depay` was the RTP payload verbatim: the two-byte payload header
//! still attached, no start codes, and fragmentation units never reassembled —
//! each fragment emitted as its own "frame". Nothing downstream could decode
//! it, and the element advertised `MediaFormat::Video(H265)` on it. Its H.264
//! sibling (`h264/mod.rs:227-291`) does the real work; the H.265 side is a
//! stub, `//TODO:` and all.
//!
//! # Scope
//!
//! The three packet shapes a real sender emits (RFC 7798 §4.4):
//!
//! - **Single NAL unit** (types 0–47): the payload *is* the NAL unit.
//! - **AP**, type 48 (§4.4.2): aggregation, like H.264's STAP-A —
//!   length-prefixed NAL units back to back.
//! - **FU**, type 49 (§4.4.3): fragmentation across packets, reassembled on
//!   the S/E bits with the original NAL header reconstructed from the payload
//!   header and the FU header's type field.
//!
//! **PACI** (type 50) is refused rather than mis-parsed: it wraps a payload in
//! a PHES header this does not decode, and emitting its bytes as a NAL unit is
//! how the bug being fixed here got shipped in the first place.
//!
//! # Loss is not an error
//!
//! A malformed or orphaned packet is **dropped**, never returned as `Err`. An
//! `Err` out of a transform kills the element task and (since #191) winds the
//! whole run down — which for a UDP element would mean one corrupt datagram
//! ends a live stream. The caller counts the drop instead.

/// Annex-B start code. Four bytes, matching what the demuxers in this tree
/// emit, so a consumer sees one framing whatever the source.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Aggregation Packet (RFC 7798 §4.4.2).
const NAL_TYPE_AP: u8 = 48;
/// Fragmentation Unit (RFC 7798 §4.4.3).
const NAL_TYPE_FU: u8 = 49;
/// PACI-carrying packet (RFC 7798 §4.4.4).
const NAL_TYPE_PACI: u8 = 50;

/// Why a packet produced nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Discard {
    /// Truncated, or a length prefix that runs past the payload.
    Malformed,
    /// A fragment arrived with no start fragment before it — the S packet was
    /// lost, so the NAL unit cannot be reconstructed.
    OrphanFragment,
    /// A packet shape this does not decode (PACI).
    Unsupported,
}

/// Reassembles RFC 7798 payloads into Annex-B NAL units.
pub(crate) struct H265Depacketizer {
    /// The NAL unit being reassembled across FU packets, start code included.
    fragment: Vec<u8>,
    /// Whether a start fragment has been seen and not yet completed.
    in_fragment: bool,
    /// Whether the sender negotiated `sprop-max-don-diff > 0`, which puts
    /// DONL/DOND fields in AP and FU packets. Off by default: it is off in
    /// every sender this tree talks to, and reading it when it is absent
    /// shifts every length prefix.
    donl: bool,
}

impl H265Depacketizer {
    pub(crate) fn new() -> Self {
        Self {
            fragment: Vec::new(),
            in_fragment: false,
            donl: false,
        }
    }

    /// Expect DONL/DOND fields (`sprop-max-don-diff > 0` in the SDP).
    pub(crate) fn with_donl(mut self, donl: bool) -> Self {
        self.donl = donl;
        self
    }

    /// Drop any partial fragment — after a flush or a seek, the next packet
    /// starts a new NAL unit.
    pub(crate) fn reset(&mut self) {
        self.fragment.clear();
        self.in_fragment = false;
    }

    /// Depacketize one RTP payload.
    ///
    /// `Ok(Some(bytes))` is one or more complete Annex-B NAL units.
    /// `Ok(None)` means the packet was consumed into a fragment that is not
    /// complete yet — the normal case for every FU packet but the last.
    pub(crate) fn push(&mut self, payload: &[u8]) -> Result<Option<Vec<u8>>, Discard> {
        if payload.len() < 2 {
            return Err(Discard::Malformed);
        }
        let nal_type = (payload[0] >> 1) & 0x3F;

        match nal_type {
            NAL_TYPE_AP => {
                // A fragment cannot span an aggregation packet, so anything
                // half-assembled was lost.
                self.reset();
                self.aggregation(payload).map(Some)
            }
            NAL_TYPE_FU => self.fragmentation(payload),
            NAL_TYPE_PACI => {
                self.reset();
                Err(Discard::Unsupported)
            }
            // Single NAL unit packet: the payload is the NAL unit, header and
            // all. Only the start code is missing.
            _ => {
                self.reset();
                let mut out = Vec::with_capacity(START_CODE.len() + payload.len());
                out.extend_from_slice(&START_CODE);
                out.extend_from_slice(payload);
                Ok(Some(out))
            }
        }
    }

    /// RFC 7798 §4.4.2. After the payload header, each aggregation unit is an
    /// optional DONL (first) or DOND (subsequent), a 16-bit size, then the NAL
    /// unit itself.
    fn aggregation(&self, payload: &[u8]) -> Result<Vec<u8>, Discard> {
        let mut out = Vec::with_capacity(payload.len() + 16);
        let mut i = 2;
        let mut first = true;

        while i < payload.len() {
            if self.donl {
                let skip = if first { 2 } else { 1 };
                if i + skip > payload.len() {
                    return Err(Discard::Malformed);
                }
                i += skip;
            }
            if i + 2 > payload.len() {
                return Err(Discard::Malformed);
            }
            let size = u16::from_be_bytes([payload[i], payload[i + 1]]) as usize;
            i += 2;
            if size == 0 || i + size > payload.len() {
                return Err(Discard::Malformed);
            }
            out.extend_from_slice(&START_CODE);
            out.extend_from_slice(&payload[i..i + size]);
            i += size;
            first = false;
        }

        if out.is_empty() {
            return Err(Discard::Malformed);
        }
        Ok(out)
    }

    /// RFC 7798 §4.4.3. The FU header's `S`/`E` bits bracket the NAL unit, and
    /// its 6-bit type is what the reconstructed NAL header must carry — the
    /// payload header says 49 (FU), which is not a NAL type at all.
    fn fragmentation(&mut self, payload: &[u8]) -> Result<Option<Vec<u8>>, Discard> {
        if payload.len() < 3 {
            self.reset();
            return Err(Discard::Malformed);
        }
        let fu_header = payload[2];
        let start = fu_header & 0x80 != 0;
        let end = fu_header & 0x40 != 0;
        let fu_type = fu_header & 0x3F;

        let mut offset = 3;
        // DONL rides only the first fragment of a NAL unit.
        if self.donl && start {
            offset += 2;
        }
        if offset > payload.len() {
            self.reset();
            return Err(Discard::Malformed);
        }

        if start {
            // A start arriving mid-fragment means the previous NAL unit's end
            // was lost. Discard it: half a NAL unit is not decodable, and
            // emitting it would corrupt everything to the next keyframe.
            self.fragment.clear();
            self.fragment.extend_from_slice(&START_CODE);
            // Rebuild the original two-byte NAL header: keep F and the layer
            // id, put the real type back.
            self.fragment
                .push((payload[0] & 0b1000_0001) | (fu_type << 1));
            self.fragment.push(payload[1]);
            self.in_fragment = true;
        } else if !self.in_fragment {
            return Err(Discard::OrphanFragment);
        }

        self.fragment.extend_from_slice(&payload[offset..]);

        if end {
            self.in_fragment = false;
            return Ok(Some(std::mem::take(&mut self.fragment)));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::annexb::{NalCodec, is_entry_point, nal_units};

    /// A two-byte H.265 NAL header for `nal_type`, layer 0, TID 1.
    fn nal_header(nal_type: u8) -> [u8; 2] {
        [nal_type << 1, 0x01]
    }

    fn single(nal_type: u8, body: &[u8]) -> Vec<u8> {
        let mut v = nal_header(nal_type).to_vec();
        v.extend_from_slice(body);
        v
    }

    #[test]
    fn a_single_nal_packet_gains_a_start_code_and_keeps_its_header() {
        let mut d = H265Depacketizer::new();
        // Type 1 = TRAIL_R.
        let packet = single(1, &[0xAA, 0xBB]);
        let out = d.push(&packet).unwrap().expect("one NAL unit");

        assert_eq!(&out[..4], &START_CODE);
        assert_eq!(&out[4..], &packet[..], "the NAL header must survive");
        assert_eq!(nal_units(&out).count(), 1);
    }

    /// The shape a keyframe actually takes: VPS, SPS, PPS and the IRAP slice
    /// aggregated into one packet.
    #[test]
    fn an_aggregation_packet_splits_into_its_units() {
        let vps = single(32, &[0x01]);
        let sps = single(33, &[0x02, 0x03]);
        let pps = single(34, &[0x04]);
        let idr = single(19, &[0x05, 0x06, 0x07]); // IDR_W_RADL

        let mut packet = nal_header(NAL_TYPE_AP).to_vec();
        for nal in [&vps, &sps, &pps, &idr] {
            packet.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            packet.extend_from_slice(nal);
        }

        let mut d = H265Depacketizer::new();
        let out = d.push(&packet).unwrap().expect("four NAL units");

        let units: Vec<&[u8]> = nal_units(&out).map(|n| n.data).collect();
        assert_eq!(units.len(), 4);
        assert_eq!(units[0], &vps[..]);
        assert_eq!(units[3], &idr[..]);
        // And the whole thing reads as a decoder entry point, which is the
        // point of aggregating it.
        assert!(is_entry_point(&out, NalCodec::H265));
    }

    #[test]
    fn a_fragmented_nal_unit_is_reassembled_with_its_real_type() {
        // A 300-byte IDR (type 19) split across three FU packets.
        let body: Vec<u8> = (0..300u32).map(|i| (i % 251) as u8).collect();
        let mut d = H265Depacketizer::new();

        let mut emitted = None;
        for (i, chunk) in body.chunks(100).enumerate() {
            let start = i == 0;
            let end = i == 2;
            let mut packet = nal_header(NAL_TYPE_FU).to_vec();
            packet.push((start as u8) << 7 | (end as u8) << 6 | 19);
            packet.extend_from_slice(chunk);

            let out = d.push(&packet).unwrap();
            if end {
                emitted = out;
            } else {
                assert!(
                    out.is_none(),
                    "a fragment that is not the last emits nothing"
                );
            }
        }

        let out = emitted.expect("the end fragment completes the NAL unit");
        let unit = nal_units(&out).next().expect("one NAL unit").data;
        // The payload header said 49 (FU), which is not a NAL type; the FU
        // header's type is what must come back.
        assert_eq!((unit[0] >> 1) & 0x3F, 19, "the real NAL type is restored");
        assert_eq!(&unit[2..], &body[..], "every byte, in order");
        assert!(is_entry_point(&out, NalCodec::H265));
    }

    /// The old behaviour, stated as a test so it cannot come back: the raw
    /// payload is not a NAL unit.
    #[test]
    fn the_payload_header_is_never_emitted_as_data() {
        let mut d = H265Depacketizer::new();
        let mut packet = nal_header(NAL_TYPE_FU).to_vec();
        packet.push(0x80 | 19); // S, type 19
        packet.extend_from_slice(&[0xDE, 0xAD]);
        assert_eq!(d.push(&packet).unwrap(), None);

        let mut packet = nal_header(NAL_TYPE_FU).to_vec();
        packet.push(0x40 | 19); // E
        packet.extend_from_slice(&[0xBE, 0xEF]);
        let out = d.push(&packet).unwrap().unwrap();

        // Two header bytes + four payload bytes, not 2+3 + 2+3.
        assert_eq!(&out[4..], &[19 << 1, 0x01, 0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn loss_is_dropped_rather_than_reassembled_wrong() {
        let mut d = H265Depacketizer::new();

        // A middle fragment with no start before it cannot be reconstructed.
        let mut orphan = nal_header(NAL_TYPE_FU).to_vec();
        orphan.push(19); // neither S nor E
        orphan.extend_from_slice(&[1, 2, 3]);
        assert_eq!(d.push(&orphan), Err(Discard::OrphanFragment));

        // A start whose end never arrives is discarded when the next start
        // comes, rather than being glued onto it.
        let mut first = nal_header(NAL_TYPE_FU).to_vec();
        first.push(0x80 | 19);
        first.extend_from_slice(&[0xAA; 8]);
        assert_eq!(d.push(&first).unwrap(), None);

        let mut second = nal_header(NAL_TYPE_FU).to_vec();
        second.push(0x80 | 0x40 | 19); // S and E: a whole NAL in one packet
        second.extend_from_slice(&[0xBB, 0xCC]);
        let out = d.push(&second).unwrap().unwrap();
        assert_eq!(
            &out[4..],
            &[19 << 1, 0x01, 0xBB, 0xCC],
            "the abandoned fragment must not be prepended"
        );
    }

    #[test]
    fn malformed_and_unsupported_packets_are_refused_not_guessed() {
        let mut d = H265Depacketizer::new();

        assert_eq!(d.push(&[]), Err(Discard::Malformed));
        assert_eq!(d.push(&[0x00]), Err(Discard::Malformed));

        // An AP whose length prefix runs past the payload.
        let mut ap = nal_header(NAL_TYPE_AP).to_vec();
        ap.extend_from_slice(&100u16.to_be_bytes());
        ap.extend_from_slice(&[1, 2, 3]);
        assert_eq!(d.push(&ap), Err(Discard::Malformed));

        // PACI: a shape this does not decode. Emitting its bytes as a NAL
        // unit is exactly the bug being fixed.
        let paci = single(NAL_TYPE_PACI, &[1, 2, 3, 4]);
        assert_eq!(d.push(&paci), Err(Discard::Unsupported));
    }

    #[test]
    fn donl_fields_are_skipped_when_the_sender_negotiated_them() {
        let vps = single(32, &[0x01]);
        let sps = single(33, &[0x02]);

        let mut packet = nal_header(NAL_TYPE_AP).to_vec();
        packet.extend_from_slice(&[0x00, 0x07]); // DONL, first unit only
        packet.extend_from_slice(&(vps.len() as u16).to_be_bytes());
        packet.extend_from_slice(&vps);
        packet.push(0x01); // DOND
        packet.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        packet.extend_from_slice(&sps);

        let mut d = H265Depacketizer::new().with_donl(true);
        let out = d.push(&packet).unwrap().expect("two NAL units");
        let units: Vec<&[u8]> = nal_units(&out).map(|n| n.data).collect();
        assert_eq!(units, vec![&vps[..], &sps[..]]);

        // …and reading it without the flag shifts every length prefix, which
        // is why the flag is not guessed.
        let mut naive = H265Depacketizer::new();
        let wrong = naive.push(&packet);
        assert!(
            wrong.is_err() || wrong.as_ref().unwrap().as_ref().unwrap() != &out,
            "a DONL stream must not silently decode as a non-DONL one"
        );
    }
}
