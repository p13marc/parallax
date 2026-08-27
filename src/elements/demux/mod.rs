//! Demultiplexing elements.
//!
//! - [`StreamIdDemux`]: Demultiplex by stream ID
//! - `TsDemux`: MPEG Transport Stream demultiplexer (requires `mpeg-ts` feature)
//! - `Mp4Demux`: MP4/MOV container demultiplexer (requires `mp4-demux` feature)
//! - `MkvDemux`: Matroska/WebM container demultiplexer (requires `mkv-demux` feature)

/// What a container says about a track, beyond its codec (#278).
///
/// Both file demuxers used to carry a codec, a geometry and nothing else, so
/// a track picker had nothing to label a row with — no language, no title, no
/// idea which track the file itself considers the default. This is that, and
/// it is deliberately one type across containers: the two surfaces were
/// already asymmetric enough (`Mp4Demux` has `track(id)`, `MkvDemux` does
/// not) without a second vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackLabel {
    /// ISO-639 language code, when the container states one.
    ///
    /// MKV gives ISO-639-2 bibliographic (and a BCP-47 field that overrides
    /// it when present); MP4 packs a 5-bit-per-letter code into `mdhd`.
    /// `None` and `"und"` both mean the file did not say.
    pub language: Option<String>,
    /// Human-readable track name, when the container states one.
    pub name: Option<String>,
    /// The track the file considers the default for its type.
    pub default: bool,
    /// A track meant to be shown regardless of preference — foreign-dialogue
    /// subtitles, typically. MKV only; MP4 has no equivalent flag.
    pub forced: bool,
    /// Whether the track is enabled at all. A disabled track is present in
    /// the file and not meant to be played.
    pub enabled: bool,
}

impl Default for TrackLabel {
    fn default() -> Self {
        // Matches both containers' own defaults: a track with nothing said
        // about it is an ordinary, enabled, non-forced track.
        Self {
            language: None,
            name: None,
            default: false,
            forced: false,
            enabled: true,
        }
    }
}

impl TrackLabel {
    /// The language, with `"und"` normalised to "not stated".
    ///
    /// Both containers write `und` for "undetermined", which is information
    /// about the *encoder* rather than about the track. A picker showing
    /// "und" next to every row is worse than showing nothing.
    pub fn spoken_language(&self) -> Option<&str> {
        self.language
            .as_deref()
            .filter(|code| !code.is_empty() && *code != "und")
    }

    /// What to show for this track in a chooser.
    ///
    /// The name if there is one, else the language, else nothing — the caller
    /// falls back to the codec, which it knows and this does not.
    pub fn describe(&self) -> Option<String> {
        match (self.name.as_deref(), self.spoken_language()) {
            (Some(name), Some(language)) => Some(format!("{name} [{language}]")),
            (Some(name), None) => Some(name.to_string()),
            (None, Some(language)) => Some(language.to_string()),
            (None, None) => None,
        }
    }

    /// Fold this into a [`TagList`](crate::pipeline::tags::TagList).
    ///
    /// The bus's `StreamInfo` has no language or flag fields — it carries a
    /// `TagList` and expects the vocabulary to live there, which is why these
    /// keys exist.
    pub fn to_tags(&self) -> crate::pipeline::tags::TagList {
        use crate::pipeline::tags::{TagValue, tag_keys};
        let mut tags = crate::pipeline::tags::TagList::new();
        if let Some(language) = self.spoken_language() {
            tags.set(tag_keys::LANGUAGE_CODE, TagValue::String(language.into()));
        }
        if let Some(name) = &self.name {
            tags.set(tag_keys::TITLE, TagValue::String(name.clone()));
        }
        tags.set(tag_keys::TRACK_DEFAULT, TagValue::Bool(self.default));
        tags.set(tag_keys::TRACK_FORCED, TagValue::Bool(self.forced));
        tags
    }
}

/// Build a bus stream collection from a container's tracks (#278).
///
/// Only the file demuxers report streams, so this compiles with them.
///
/// Shared by the file demuxers so a picker sees one vocabulary whichever
/// container it opened. `StreamInfo` has no "which one is playing" field, so
/// that rides `tag_keys::TRACK_SELECTED` — a collection that cannot say what
/// is selected is only half an answer.
#[cfg(any(feature = "mkv-demux", feature = "mp4-demux"))]
pub(crate) struct StreamReport {
    /// A stable identifier for the stream, unique within the container.
    pub id: String,
    /// Video, audio, subtitle or data.
    pub kind: crate::element::muxer::StreamType,
    /// Language, name and flags.
    pub label: TrackLabel,
    /// A one-line description of what it holds.
    pub caps: String,
    /// Whether this is the track the demuxer routed.
    pub selected: bool,
}

#[cfg(any(feature = "mkv-demux", feature = "mp4-demux"))]
impl StreamReport {
    fn into_stream_info(self) -> crate::pipeline::bus::StreamInfo {
        use crate::pipeline::tags::{TagValue, tag_keys};
        let mut tags = self.label.to_tags();
        tags.set(tag_keys::TRACK_SELECTED, TagValue::Bool(self.selected));
        tags.set(tag_keys::CODEC, TagValue::String(self.caps.clone()));
        crate::pipeline::bus::StreamInfo {
            stream_id: self.id,
            stream_type: self.kind,
            tags,
            caps: Some(self.caps),
        }
    }
}

/// Post what a container holds, once.
///
/// Both messages, because they answer different questions: the collection is
/// "what tracks exist and which is playing", the tag list is "what is this
/// file". Neither had ever been posted by anything.
#[cfg(any(feature = "mkv-demux", feature = "mp4-demux"))]
pub(crate) fn post_streams(
    bus: &crate::pipeline::bus::BusHandle,
    container: &str,
    duration: crate::clock::ClockTime,
    reports: Vec<StreamReport>,
) {
    use crate::pipeline::bus::MessageKind;
    use crate::pipeline::tags::{TagList, TagValue, tag_keys};

    let mut tags = TagList::new();
    tags.set(
        tag_keys::CONTAINER_FORMAT,
        TagValue::String(container.into()),
    );
    if duration.to_option().is_some() {
        tags.set(tag_keys::DURATION, TagValue::ClockTime(duration));
    }
    // The selected tracks' codecs, which is what a title bar wants.
    for report in reports.iter().filter(|r| r.selected) {
        let key = match report.kind {
            crate::element::muxer::StreamType::Video => tag_keys::VIDEO_CODEC,
            crate::element::muxer::StreamType::Audio => tag_keys::AUDIO_CODEC,
            _ => continue,
        };
        tags.set(key, TagValue::String(report.caps.clone()));
    }
    bus.post_tags(tags);

    bus.post(MessageKind::StreamCollection {
        streams: reports.into_iter().map(|r| r.into_stream_info()).collect(),
    });
}

mod streamid_demux;

#[cfg(feature = "mpeg-ts")]
mod mpegts;

#[cfg(feature = "mp4-demux")]
mod mp4;

#[cfg(feature = "mp4-demux")]
mod mp4_extra;

#[cfg(feature = "mkv-demux")]
mod mkv;

#[cfg(feature = "mkv-demux")]
mod mkv_cues;

pub use streamid_demux::{StreamIdDemux, StreamIdDemuxStats, StreamOutput};

#[cfg(feature = "mpeg-ts")]
pub use mpegts::{
    TS_PACKET_SIZE, TsByteIndex, TsDemux, TsDemuxElement, TsDemuxStats, TsElementaryStream,
    TsFrame, TsProgram, TsStreamType,
};

#[cfg(feature = "mp4-demux")]
pub use mp4::{
    Mp4AudioInfo, Mp4Codec, Mp4Demux, Mp4DemuxSource, Mp4DemuxStats, Mp4Sample, Mp4SeekPoint,
    Mp4Track, Mp4TrackType, Mp4VideoInfo,
};

#[cfg(feature = "mkv-demux")]
pub use mkv::{MkvAudioInfo, MkvCodec, MkvDemux, MkvTrack, MkvTrackType, MkvVideoInfo};
