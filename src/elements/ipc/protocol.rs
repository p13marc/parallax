//! Control-plane protocol for the IPC elements.
//!
//! Since #179 the hot path — per-buffer descriptors and acks — rides the
//! shared-memory ring ([`crate::memory::IpcChannel`]); the Unix socket
//! carries only what genuinely needs a stream: registration (with fds via
//! SCM_RIGHTS), rare overflow metadata, and teardown. That is the
//! data-plane/signaling-plane split of design.md principle 8.
//!
//! Messages are rkyv-encoded with a 4-byte LE length prefix. The length is
//! peer-controlled, so it is bounded ([`MAX_CONTROL_MESSAGE_SIZE`]) and a
//! malformed body is an error, never a panic.

use crate::error::{Error, Result};

/// Upper bound on one framed control message.
///
/// Registration and shutdown are tiny; the only variable-size message is
/// `MetaOverflow`, whose entries are bounded custom-metadata payloads (KLV
/// packets, SEI). 64 KiB is generous for those and small enough that a
/// hostile length prefix cannot balloon an allocation.
pub const MAX_CONTROL_MESSAGE_SIZE: usize = 64 * 1024;

/// Custom-metadata keys forwarded across the IPC boundary.
///
/// Keys are `&'static str` in [`crate::metadata::Metadata`], so only keys
/// known at compile time can be re-attached on the receiving side. Mirrors
/// `zenoh_wire::KNOWN_CUSTOM_KEYS` (feature-gated, hence the small
/// duplication); keep the two lists in sync when adding keys.
pub const KNOWN_CUSTOM_KEYS: &[&str] = &["stanag/klv", "h264/sei"];

/// Control messages between IpcSink and IpcSrc.
///
/// Every variant is sink→src; the src's only signal back to the sink is
/// the ack ring (and closing the socket).
#[derive(Clone, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ControlMessage {
    /// The ring channel handshake. Accompanied by three fds via SCM_RIGHTS,
    /// in [`crate::memory::IpcChannel::fds`] order:
    /// `[ring segment, data doorbell, ack doorbell]`. Sent once, first.
    RegisterChannel {
        /// Ring capacity, for validation (the segment header self-describes;
        /// a mismatch means a protocol bug).
        capacity: u32,
    },

    /// A buffer arena the sink is about to reference. Accompanied by one fd
    /// (the arena memfd). Sent register-on-first-sight, always *before* the
    /// first descriptor whose `arena_id` matches — socket FIFO plus the
    /// ring publish give the src a happens-before it can rely on.
    RegisterArena {
        /// The arena's process-unique id (#178), as the descriptors carry it.
        arena_id: u64,
    },

    /// Custom-metadata overflow for the descriptor with this `seq`. Sent
    /// *before* that descriptor is pushed (same ordering argument as
    /// `RegisterArena`); the descriptor carries the meta-overflow presence
    /// bit so the src knows to collect this first.
    MetaOverflow {
        /// The descriptor's ack-correlation seq.
        seq: u64,
        /// `(key, bytes)` pairs; only [`KNOWN_CUSTOM_KEYS`] survive the trip.
        entries: Vec<(String, Vec<u8>)>,
    },

    /// Teardown backstop for abnormal paths. Graceful EOS rides the ring
    /// segment's state word, not the socket.
    Shutdown,
}

/// Frame a message: 4-byte LE length prefix + rkyv bytes.
///
/// Panics only on rkyv failing to serialize our own value, which is a bug,
/// not an input condition.
pub fn frame_message(msg: &ControlMessage) -> Vec<u8> {
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(msg).expect("serialization failed");
    let mut framed = Vec::with_capacity(4 + bytes.len());
    framed.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    framed.extend_from_slice(&bytes);
    framed
}

/// Parse one framed message from the front of `data`.
///
/// - `Ok(Some((msg, consumed)))` — one whole frame parsed.
/// - `Ok(None)` — the frame is incomplete; read more and retry.
/// - `Err` — the peer sent garbage (oversized length, undecodable body).
pub fn unframe_message(data: &[u8]) -> Result<Option<(ControlMessage, usize)>> {
    if data.len() < 4 {
        return Ok(None);
    }
    let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if len > MAX_CONTROL_MESSAGE_SIZE {
        return Err(Error::Element(format!(
            "ipc control message length {len} exceeds the {MAX_CONTROL_MESSAGE_SIZE} bound"
        )));
    }
    if data.len() < 4 + len {
        return Ok(None);
    }

    // rkyv wants aligned bytes; the frame arrives at arbitrary offset.
    let mut aligned = rkyv::util::AlignedVec::<8>::with_capacity(len);
    aligned.extend_from_slice(&data[4..4 + len]);
    let msg = rkyv::from_bytes::<ControlMessage, rkyv::rancor::Error>(&aligned)
        .map_err(|e| Error::Element(format!("undecodable ipc control message: {e}")))?;
    Ok(Some((msg, 4 + len)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_round_trips_every_variant() {
        let messages = [
            ControlMessage::RegisterChannel { capacity: 64 },
            ControlMessage::RegisterArena { arena_id: u64::MAX },
            ControlMessage::MetaOverflow {
                seq: 42,
                entries: vec![
                    ("stanag/klv".into(), vec![1, 2, 3]),
                    ("h264/sei".into(), vec![0xFF; 100]),
                ],
            },
            ControlMessage::Shutdown,
        ];
        for msg in messages {
            let framed = frame_message(&msg);
            let (parsed, consumed) = unframe_message(&framed).unwrap().unwrap();
            assert_eq!(parsed, msg);
            assert_eq!(consumed, framed.len());
        }
    }

    #[test]
    fn partial_frames_ask_for_more() {
        let framed = frame_message(&ControlMessage::Shutdown);
        assert!(unframe_message(&framed[..2]).unwrap().is_none());
        assert!(unframe_message(&framed[..4]).unwrap().is_none());
        assert!(
            unframe_message(&framed[..framed.len() - 1])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn two_frames_parse_in_sequence() {
        let a = frame_message(&ControlMessage::RegisterArena { arena_id: 1 });
        let b = frame_message(&ControlMessage::Shutdown);
        let mut joined = a.clone();
        joined.extend_from_slice(&b);

        let (first, consumed) = unframe_message(&joined).unwrap().unwrap();
        assert_eq!(first, ControlMessage::RegisterArena { arena_id: 1 });
        assert_eq!(consumed, a.len());
        let (second, consumed2) = unframe_message(&joined[consumed..]).unwrap().unwrap();
        assert_eq!(second, ControlMessage::Shutdown);
        assert_eq!(consumed2, b.len());
    }

    #[test]
    fn hostile_input_errors_instead_of_panicking() {
        // Oversized length prefix: must not allocate 4 GB or panic.
        let mut oversized = Vec::new();
        oversized.extend_from_slice(&u32::MAX.to_le_bytes());
        oversized.extend_from_slice(&[0u8; 64]);
        assert!(unframe_message(&oversized).is_err());

        // Well-sized garbage body: an error, never a panic.
        let mut garbage = Vec::new();
        garbage.extend_from_slice(&16u32.to_le_bytes());
        garbage.extend_from_slice(&[0xA5u8; 16]);
        assert!(unframe_message(&garbage).is_err());
    }
}
