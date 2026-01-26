//! Control protocol for element processes.
//!
//! Defines the messages exchanged between the supervisor and element processes
//! over Unix sockets. Uses rkyv for zero-copy serialization.

use crate::format::MediaCaps;
use crate::memory::IpcSlotRef;
use crate::metadata::Metadata;

/// Control message sent between supervisor and element processes.
///
/// All messages are serialized with rkyv and sent over Unix sockets with
/// length-prefixed framing.
#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub enum ControlMessage {
    /// Initialize element with negotiated caps.
    ///
    /// Sent from supervisor to element after spawning.
    Init {
        /// Negotiated media capabilities.
        caps: SerializableCaps,
    },

    /// A buffer is ready for processing.
    ///
    /// Sent from supervisor to element (for sinks) or from element to
    /// supervisor (for sources).
    BufferReady {
        /// Reference to the buffer in shared memory.
        slot: IpcSlotRef,
        /// Buffer metadata.
        metadata: SerializableMetadata,
    },

    /// Buffer processing is complete, slot can be reused.
    ///
    /// Sent from element to supervisor after processing a BufferReady message.
    BufferDone {
        /// The slot that was processed.
        slot: IpcSlotRef,
    },

    /// Request a state change.
    ///
    /// Sent from supervisor to element.
    StateChange {
        /// The new state to transition to.
        new_state: ElementState,
    },

    /// Acknowledge a state change.
    ///
    /// Sent from element to supervisor after completing a state transition.
    StateChanged {
        /// The new current state.
        state: ElementState,
    },

    /// Report an error.
    ///
    /// Can be sent by either side.
    Error {
        /// Error code (0 = success, non-zero = error).
        code: u32,
        /// Human-readable error message.
        message: String,
    },

    /// End of stream signal.
    ///
    /// Sent when a source has no more data.
    Eos,

    /// Request graceful shutdown.
    ///
    /// Sent from supervisor to element.
    Shutdown,

    /// Acknowledge shutdown is complete.
    ///
    /// Sent from element to supervisor before exiting.
    ShutdownAck,

    /// Heartbeat ping (liveness check).
    ///
    /// Sent from supervisor to element.
    Ping {
        /// Sequence number for matching responses.
        seq: u64,
    },

    /// Heartbeat pong (liveness response).
    ///
    /// Sent from element to supervisor in response to Ping.
    Pong {
        /// Sequence number from the Ping.
        seq: u64,
    },

    /// Arena registration.
    ///
    /// Sent when a new shared memory arena needs to be mapped.
    /// The arena fd is sent via SCM_RIGHTS alongside this message.
    RegisterArena {
        /// Unique arena identifier.
        arena_id: u64,
        /// Size of the arena in bytes.
        size: usize,
        /// Size of each slot in the arena.
        slot_size: usize,
        /// Number of slots in the arena.
        slot_count: usize,
    },

    /// Statistics report.
    ///
    /// Periodically sent from element to supervisor.
    Stats {
        /// Number of buffers processed.
        buffers_processed: u64,
        /// Total bytes processed.
        bytes_processed: u64,
        /// Average processing time in nanoseconds.
        avg_process_time_ns: u64,
        /// Current memory usage in bytes.
        memory_usage: u64,
    },
}

/// Element state.
///
/// Elements transition through these states during their lifecycle.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[rkyv(derive(Debug))]
#[repr(u8)]
pub enum ElementState {
    /// Initial state - element is created but not initialized.
    #[default]
    Null = 0,

    /// Element is initialized and ready to process.
    Ready = 1,

    /// Element is actively processing data.
    Playing = 2,

    /// Element is paused (can resume quickly).
    Paused = 3,
}

impl ElementState {
    /// Check if state allows processing buffers.
    pub fn can_process(&self) -> bool {
        matches!(self, Self::Playing)
    }

    /// Check if state transition is valid.
    pub fn can_transition_to(&self, new_state: Self) -> bool {
        use ElementState::*;
        matches!(
            (self, new_state),
            // From Null
            (Null, Ready) |
            // From Ready
            (Ready, Playing) | (Ready, Null) |
            // From Playing
            (Playing, Paused) | (Playing, Ready) |
            // From Paused
            (Paused, Playing) | (Paused, Ready)
        )
    }
}

impl std::fmt::Display for ElementState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Null => write!(f, "Null"),
            Self::Ready => write!(f, "Ready"),
            Self::Playing => write!(f, "Playing"),
            Self::Paused => write!(f, "Paused"),
        }
    }
}

/// Serializable version of MediaCaps for IPC.
///
/// This is a simplified representation that can be serialized with rkyv.
#[derive(Clone, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct SerializableCaps {
    /// Format type identifier.
    pub format_type: u32,
    /// Format-specific data (JSON or binary).
    pub format_data: Vec<u8>,
    /// Memory type.
    pub memory_type: u32,
}

impl SerializableCaps {
    /// Create from MediaCaps.
    pub fn from_caps(_caps: &MediaCaps) -> Self {
        // TODO: Proper serialization
        Self {
            format_type: 0,
            format_data: Vec::new(),
            memory_type: 0,
        }
    }
}

/// Serializable version of Metadata for IPC.
#[derive(Clone, Debug, Default, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct SerializableMetadata {
    /// Buffer flags.
    pub flags: u8,
    /// Presentation timestamp in nanoseconds.
    pub pts: Option<u64>,
    /// Decode timestamp in nanoseconds.
    pub dts: Option<u64>,
    /// Duration in nanoseconds.
    pub duration: Option<u64>,
    /// Sequence number.
    pub sequence: u64,
    /// Stream identifier.
    pub stream_id: Option<u32>,
}

impl SerializableMetadata {
    /// Create from Metadata.
    pub fn from_metadata(meta: &Metadata) -> Self {
        Self {
            flags: meta.flags.bits(),
            pts: if meta.pts.is_some() {
                Some(meta.pts.nanos())
            } else {
                None
            },
            dts: if meta.dts.is_some() {
                Some(meta.dts.nanos())
            } else {
                None
            },
            duration: if meta.duration.is_some() {
                Some(meta.duration.nanos())
            } else {
                None
            },
            sequence: meta.sequence,
            stream_id: if meta.stream_id != 0 {
                Some(meta.stream_id)
            } else {
                None
            },
        }
    }

    /// Convert to Metadata.
    pub fn to_metadata(&self) -> Metadata {
        use crate::clock::ClockTime;
        use crate::metadata::BufferFlags;

        let mut meta = Metadata::from_sequence(self.sequence);

        if let Some(pts) = self.pts {
            meta.pts = ClockTime::from_nanos(pts);
        }
        if let Some(dts) = self.dts {
            meta.dts = ClockTime::from_nanos(dts);
        }
        if let Some(duration) = self.duration {
            meta.duration = ClockTime::from_nanos(duration);
        }
        if let Some(stream_id) = self.stream_id {
            meta.stream_id = stream_id;
        }

        // Set flags
        meta.flags = BufferFlags::from_bits(self.flags);

        meta
    }
}

/// Frame a message for sending.
///
/// Returns a buffer with length prefix followed by serialized message.
pub fn frame_message(msg: &ControlMessage) -> Vec<u8> {
    let serialized = rkyv::to_bytes::<rkyv::rancor::Error>(msg).expect("serialization failed");
    let len = serialized.len() as u32;

    let mut framed = Vec::with_capacity(4 + serialized.len());
    framed.extend_from_slice(&len.to_le_bytes());
    framed.extend_from_slice(&serialized);
    framed
}

/// Unframe a message from a buffer.
///
/// Returns the message and the number of bytes consumed.
/// Returns None if the buffer doesn't contain a complete message.
pub fn unframe_message(buf: &[u8]) -> Option<(ControlMessage, usize)> {
    if buf.len() < 4 {
        return None;
    }

    let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if buf.len() < 4 + len {
        return None;
    }

    // Copy to aligned buffer for rkyv
    let mut aligned = rkyv::util::AlignedVec::<8>::new();
    aligned.extend_from_slice(&buf[4..4 + len]);

    let msg: ControlMessage = rkyv::from_bytes::<ControlMessage, rkyv::rancor::Error>(&aligned)
        .expect("deserialization failed");

    Some((msg, 4 + len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_state_transitions() {
        use ElementState::*;

        assert!(Null.can_transition_to(Ready));
        assert!(!Null.can_transition_to(Playing));
        assert!(!Null.can_transition_to(Paused));

        assert!(Ready.can_transition_to(Playing));
        assert!(Ready.can_transition_to(Null));
        assert!(!Ready.can_transition_to(Paused));

        assert!(Playing.can_transition_to(Paused));
        assert!(Playing.can_transition_to(Ready));
        assert!(!Playing.can_transition_to(Null));

        assert!(Paused.can_transition_to(Playing));
        assert!(Paused.can_transition_to(Ready));
        assert!(!Paused.can_transition_to(Null));
    }

    #[test]
    fn test_element_state_can_process() {
        assert!(!ElementState::Null.can_process());
        assert!(!ElementState::Ready.can_process());
        assert!(ElementState::Playing.can_process());
        assert!(!ElementState::Paused.can_process());
    }

    #[test]
    fn test_message_framing() {
        let msg = ControlMessage::Ping { seq: 42 };
        let framed = frame_message(&msg);

        let (decoded, consumed) = unframe_message(&framed).unwrap();
        assert_eq!(consumed, framed.len());

        if let ControlMessage::Ping { seq } = decoded {
            assert_eq!(seq, 42);
        } else {
            panic!("Wrong message type");
        }
    }

    #[test]
    fn test_message_framing_partial() {
        let msg = ControlMessage::Pong { seq: 123 };
        let framed = frame_message(&msg);

        // Incomplete buffer
        assert!(unframe_message(&framed[..2]).is_none());
        assert!(unframe_message(&framed[..4]).is_none());
        assert!(unframe_message(&framed[..framed.len() - 1]).is_none());

        // Complete buffer
        assert!(unframe_message(&framed).is_some());
    }

    #[test]
    fn test_serializable_metadata_roundtrip() {
        use crate::clock::ClockTime;
        use crate::metadata::BufferFlags;

        let original = Metadata::from_sequence(42)
            .with_pts(ClockTime::from_millis(1000))
            .with_dts(ClockTime::from_millis(900))
            .with_duration(ClockTime::from_millis(33))
            .with_stream_id(1)
            .with_flags(BufferFlags::SYNC_POINT | BufferFlags::DISCONT);

        let serializable = SerializableMetadata::from_metadata(&original);
        let recovered = serializable.to_metadata();

        assert_eq!(recovered.sequence, 42);
        assert_eq!(recovered.pts, ClockTime::from_millis(1000));
        assert_eq!(recovered.dts, ClockTime::from_millis(900));
        assert_eq!(recovered.duration, ClockTime::from_millis(33));
        assert_eq!(recovered.stream_id, 1);
        assert!(recovered.flags.contains(BufferFlags::SYNC_POINT));
        assert!(recovered.flags.contains(BufferFlags::DISCONT));
    }

    #[test]
    fn test_control_message_variants() {
        // Test each variant serializes correctly
        let messages = vec![
            ControlMessage::Init {
                caps: SerializableCaps {
                    format_type: 1,
                    format_data: vec![1, 2, 3],
                    memory_type: 0,
                },
            },
            ControlMessage::BufferReady {
                slot: IpcSlotRef::new(1, 0, 1024),
                metadata: SerializableMetadata::default(),
            },
            ControlMessage::BufferDone {
                slot: IpcSlotRef::new(1, 0, 1024),
            },
            ControlMessage::StateChange {
                new_state: ElementState::Playing,
            },
            ControlMessage::StateChanged {
                state: ElementState::Playing,
            },
            ControlMessage::Error {
                code: 1,
                message: "test error".into(),
            },
            ControlMessage::Eos,
            ControlMessage::Shutdown,
            ControlMessage::ShutdownAck,
            ControlMessage::Ping { seq: 1 },
            ControlMessage::Pong { seq: 1 },
            ControlMessage::RegisterArena {
                arena_id: 1,
                size: 1024 * 1024,
                slot_size: 4096,
                slot_count: 256,
            },
            ControlMessage::Stats {
                buffers_processed: 100,
                bytes_processed: 1000000,
                avg_process_time_ns: 5000,
                memory_usage: 10000,
            },
        ];

        for msg in messages {
            let framed = frame_message(&msg);
            let (decoded, _) = unframe_message(&framed).unwrap();
            // Just verify it decodes without panic
            let _ = format!("{:?}", decoded);
        }
    }
}
