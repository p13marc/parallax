//! `FrameMeta` — the per-frame attachment on the `@media` plane.

use crate::clock::ClockTime;
use crate::metadata::{BufferFlags, Metadata};
use serde::{Deserialize, Serialize};

/// Per-frame metadata riding as the attachment on every `@media` sample.
///
/// The payload stays opaque encoded bytes; this sidecar is what lets a
/// consumer gate on keyframes, detect gaps and time frames without parsing the
/// bitstream. Encoded as **CBOR**, not as a telemetry envelope.
///
/// # The omitted-`None` rule is wire shape, not style
///
/// The three timing fields carry `skip_serializing_if`, so an absent timestamp
/// is *missing from the CBOR map* rather than present-and-null. A consumer
/// reading the map sees the difference, and the reference implementation pins
/// it with a test. Do not "tidy" the attribute away.
///
/// # Why parallax defines its own
///
/// `zensight-common` has a byte-compatible twin. Neither crate can import the
/// other's: zensight depends on `parallax-pipeline`, and the reverse would
/// drag zenoh and the whole engine into a crate that every sensor links —
/// including the ones that never touch video. The binding artifact is a
/// conformance corpus of canonical CBOR vectors, checked into both.
///
/// ```
/// use parallax::wire::FrameMeta;
/// use parallax::metadata::{BufferFlags, Metadata};
/// use parallax::clock::ClockTime;
///
/// let mut meta = Metadata::new();
/// meta.pts = ClockTime::from_nanos(1_000_000);
/// meta.dts = ClockTime::from_nanos(1_000_000);
/// meta.flags |= BufferFlags::SYNC_POINT;
/// meta.sequence = 7;
///
/// let frame = FrameMeta::from_metadata(&meta, 1920, 1080);
/// assert!(frame.keyframe);
/// assert_eq!(frame.pts_ns, Some(1_000_000));
/// // dts equals pts, so it is omitted — the spec says "if distinct".
/// assert_eq!(frame.dts_ns, None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FrameMeta {
    /// Whether this frame is independently decodable (H.264 IDR / any JPEG).
    ///
    /// A byte-level promise: a fresh decoder must be able to start here. See
    /// [`annexb::ParamSetCache`](crate::codec::annexb::ParamSetCache).
    pub keyframe: bool,
    /// Presentation timestamp in nanoseconds, if the pipeline stamped one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pts_ns: Option<u64>,
    /// Decode timestamp in nanoseconds, if distinct from `pts_ns`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dts_ns: Option<u64>,
    /// Frame duration in nanoseconds, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ns: Option<u64>,
    /// Monotonic per-stream sequence number; a gap means frames were dropped.
    pub sequence: u64,
    /// Encoded frame width in pixels.
    pub width: u32,
    /// Encoded frame height in pixels.
    pub height: u32,
}

/// `ClockTime::NONE` is the `u64::MAX` sentinel; the wire says "absent".
fn clock_ns(t: ClockTime) -> Option<u64> {
    (!t.is_none()).then(|| t.nanos())
}

impl FrameMeta {
    /// Capture a buffer's metadata for the wire.
    ///
    /// `width`/`height` are passed in rather than read off `metadata` because
    /// the publisher may know the geometry when the buffer does not — and
    /// because a `0` there is meaningful ("unknown"), not an error.
    ///
    /// `dts_ns` is omitted when it equals `pts_ns`: the specification says
    /// "if distinct from `pts_ns`", and every frame without B-frames would
    /// otherwise carry a redundant field.
    ///
    /// Note that a `ClockTime` of ZERO is a *real* timestamp, not an absent
    /// one — only `ClockTime::NONE` means absent. A producer that stamps `pts`
    /// and leaves `dts` at its default therefore publishes `dts_ns: Some(0)`,
    /// which is faithful to the buffer it was handed. If that is wrong, it is
    /// wrong at the producer.
    pub fn from_metadata(metadata: &Metadata, width: u32, height: u32) -> Self {
        let pts_ns = clock_ns(metadata.pts);
        let dts_ns = clock_ns(metadata.dts).filter(|dts| Some(*dts) != pts_ns);

        Self {
            keyframe: metadata.flags.is_keyframe(),
            pts_ns,
            dts_ns,
            duration_ns: clock_ns(metadata.duration),
            sequence: metadata.sequence,
            width,
            height,
        }
    }

    /// Rebuild in-memory metadata from the wire representation.
    ///
    /// An absent `dts_ns` becomes `pts_ns` rather than `ClockTime::NONE` —
    /// the field is omitted precisely when the two are equal, and decoders
    /// want a value.
    pub fn to_metadata(&self) -> Metadata {
        let mut metadata = Metadata::new();
        metadata.pts = self.pts_ns.map_or(ClockTime::NONE, ClockTime::from_nanos);
        metadata.dts = self
            .dts_ns
            .or(self.pts_ns)
            .map_or(ClockTime::NONE, ClockTime::from_nanos);
        metadata.duration = self
            .duration_ns
            .map_or(ClockTime::NONE, ClockTime::from_nanos);
        metadata.sequence = self.sequence;
        if self.keyframe {
            metadata.flags |= BufferFlags::SYNC_POINT;
        }
        metadata
    }

    /// Encode as CBOR for the sample attachment.
    #[cfg(feature = "zenoh")]
    pub fn to_cbor(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64);
        ciborium::into_writer(self, &mut out).expect("FrameMeta serialization cannot fail");
        out
    }

    /// Decode from a sample attachment.
    ///
    /// `None` for anything that is not a `FrameMeta` — a foreign publisher's
    /// attachment, a truncated one, a parallax-plane rkyv blob. Callers fall
    /// back; they never error.
    #[cfg(feature = "zenoh")]
    pub fn from_cbor(bytes: &[u8]) -> Option<Self> {
        ciborium::from_reader(bytes).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_survives_the_round_trip() {
        let mut meta = Metadata::new();
        meta.pts = ClockTime::from_nanos(1_000);
        meta.dts = ClockTime::from_nanos(500);
        meta.duration = ClockTime::from_nanos(33_366_667);
        meta.sequence = 99;
        meta.flags |= BufferFlags::SYNC_POINT;

        let frame = FrameMeta::from_metadata(&meta, 640, 480);
        assert_eq!(frame.pts_ns, Some(1_000));
        assert_eq!(frame.dts_ns, Some(500));
        assert_eq!(frame.duration_ns, Some(33_366_667));

        let back = frame.to_metadata();
        assert_eq!(back.pts, meta.pts);
        assert_eq!(back.dts, meta.dts);
        assert_eq!(back.duration, meta.duration);
        assert_eq!(back.sequence, 99);
        assert!(back.flags.is_keyframe());
    }

    #[test]
    fn dts_is_omitted_when_it_equals_pts() {
        let mut meta = Metadata::new();
        meta.pts = ClockTime::from_nanos(42);
        meta.dts = ClockTime::from_nanos(42);

        let frame = FrameMeta::from_metadata(&meta, 1, 1);
        assert_eq!(frame.dts_ns, None, "the spec says 'if distinct from pts'");
        // …and comes back equal, because a decoder wants a value.
        assert_eq!(frame.to_metadata().dts, ClockTime::from_nanos(42));
    }

    #[test]
    fn an_unstamped_buffer_omits_its_timing() {
        // ClockTime::NONE is u64::MAX; it must not travel as a real timestamp.
        // Note `Metadata::new()` starts at ZERO, not NONE — a real distinction
        // the pipeline relies on elsewhere — so set the sentinel explicitly.
        let mut meta = Metadata::new();
        meta.pts = ClockTime::NONE;
        meta.dts = ClockTime::NONE;
        meta.duration = ClockTime::NONE;

        let frame = FrameMeta::from_metadata(&meta, 0, 0);
        assert_eq!(frame.pts_ns, None);
        assert_eq!(frame.dts_ns, None);
        assert_eq!(frame.duration_ns, None);
        assert!(frame.to_metadata().pts.is_none());

        // A zero timestamp is a real one and must survive as zero.
        let zeroed = Metadata::new();
        assert_eq!(FrameMeta::from_metadata(&zeroed, 0, 0).pts_ns, Some(0));
    }

    #[cfg(feature = "zenoh")]
    #[test]
    fn absent_timing_is_missing_from_the_map_not_null() {
        // The load-bearing wire detail: a consumer reading the CBOR map must
        // see three keys, not six-with-nulls.
        let frame = FrameMeta {
            keyframe: true,
            sequence: 1,
            width: 16,
            height: 16,
            ..Default::default()
        };
        let cbor = frame.to_cbor();

        let value: ciborium::Value = ciborium::from_reader(&cbor[..]).unwrap();
        let map = value.as_map().expect("a CBOR map");
        let keys: Vec<String> = map
            .iter()
            .filter_map(|(k, _)| k.as_text().map(str::to_string))
            .collect();

        assert_eq!(keys, vec!["keyframe", "sequence", "width", "height"]);
        assert_eq!(FrameMeta::from_cbor(&cbor), Some(frame));
    }

    #[cfg(feature = "zenoh")]
    #[test]
    fn a_foreign_attachment_decodes_to_none_rather_than_erroring() {
        // A parallax-plane rkyv attachment, a truncated blob, and noise.
        assert_eq!(FrameMeta::from_cbor(&[0x50, 0x58, 0x02, 0xFF]), None);
        assert_eq!(FrameMeta::from_cbor(&[]), None);
        assert_eq!(FrameMeta::from_cbor(b"not cbor at all"), None);
    }

    #[cfg(feature = "zenoh")]
    #[test]
    fn a_full_frame_round_trips_through_cbor() {
        let frame = FrameMeta {
            keyframe: false,
            pts_ns: Some(1_234_567_890),
            dts_ns: Some(1_234_000_000),
            duration_ns: Some(16_683_333),
            sequence: u64::MAX,
            width: 3840,
            height: 2160,
        };
        assert_eq!(FrameMeta::from_cbor(&frame.to_cbor()), Some(frame));
    }
}
