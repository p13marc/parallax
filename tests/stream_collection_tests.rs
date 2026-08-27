//! What a container tells the application about itself (#278).
//!
//! `MessageKind::StreamCollection`, `MessageKind::Tag` and the whole of
//! `pipeline::tags` shipped with no producer: the types existed, the bus
//! carried them, and nothing in the tree had ever posted one. These are the
//! tests for the first producers.

#![cfg(feature = "mkv-demux")]

use std::io::Cursor;

use parallax::element::Demuxer;
use parallax::elements::demux::{MkvDemux, MkvTrackType};
use parallax::elements::{AppSink, Pulled};
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::tags::tag_keys;
use parallax::pipeline::{Executor, Pipeline};

const H264_AAC_MKV: &[u8] = include_bytes!("fixtures/tiny_h264_aac.mkv");

/// Drain the bus for the one collection and the one tag list.
async fn collect(data: &[u8]) -> (Vec<parallax::pipeline::bus::StreamInfo>, Vec<MessageKind>) {
    let demux = MkvDemux::new(Cursor::new(data.to_vec())).expect("fixture parses");

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_demuxer("demux", demux);
    let video = AppSink::with_max_buffers(4);
    let video_handle = video.handle();
    let sink = pipeline.add_async_sink("video", video);
    pipeline
        .link_pads(src, "video", sink, "sink")
        .expect("video links");

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).expect("it starts");
    let mut bus = handle.take_bus().expect("the bus");

    // Drain until the stream ends; the sink is bounded, so somebody has to.
    while let Pulled::Buffer(_) = video_handle.pull_buffer().await {}
    handle.stop();
    let _ = handle.wait().await;

    let mut streams = Vec::new();
    let mut kinds = Vec::new();
    while let Some(message) = bus.poll() {
        if let MessageKind::StreamCollection { streams: found } = &message.kind {
            streams = found.clone();
        }
        kinds.push(message.kind);
    }
    (streams, kinds)
}

#[tokio::test]
async fn a_matroska_file_says_what_tracks_it_has() {
    let (streams, _) = collect(H264_AAC_MKV).await;
    assert_eq!(
        streams.len(),
        2,
        "the fixture has one video and one audio track"
    );

    let video = streams
        .iter()
        .find(|s| s.stream_type == parallax::element::muxer::StreamType::Video)
        .expect("a video stream");
    assert_eq!(video.caps.as_deref(), Some("V_MPEG4/ISO/AVC"));
    assert_eq!(
        video.tags.get_bool(tag_keys::TRACK_SELECTED),
        Some(true),
        "the routed video track says it is the one playing"
    );

    let audio = streams
        .iter()
        .find(|s| s.stream_type == parallax::element::muxer::StreamType::Audio)
        .expect("an audio stream");
    // Only the video pad is linked here, but the collection describes the
    // *file*, not the graph — an unrouted track still has to be listed or a
    // picker cannot offer it.
    assert_eq!(audio.caps.as_deref(), Some("A_AAC"));

    // Stream ids are stable and distinct, which is what a selection API
    // would key on.
    assert_ne!(video.stream_id, audio.stream_id);
}

#[tokio::test]
async fn the_container_posts_its_own_tags() {
    let (_, kinds) = collect(H264_AAC_MKV).await;
    let tags = kinds
        .iter()
        .find_map(|kind| match kind {
            MessageKind::Tag { tags } => Some(tags),
            _ => None,
        })
        .expect("a tag list was posted");

    assert_eq!(
        tags.get_string(tag_keys::CONTAINER_FORMAT),
        Some("matroska")
    );
    assert_eq!(
        tags.get_string(tag_keys::VIDEO_CODEC),
        Some("V_MPEG4/ISO/AVC"),
        "the selected video track's codec, for a title bar"
    );
    assert!(
        tags.get(tag_keys::DURATION).is_some(),
        "the fixture states a duration"
    );
}

#[tokio::test]
async fn the_collection_goes_out_once() {
    let (_, kinds) = collect(H264_AAC_MKV).await;
    let collections = kinds
        .iter()
        .filter(|kind| matches!(kind, MessageKind::StreamCollection { .. }))
        .count();
    assert_eq!(
        collections, 1,
        "the track table is fixed at construction; repeating it says nothing"
    );
}

/// Selecting a track that is not there is an error, not a silent default.
#[test]
fn a_track_that_does_not_exist_is_refused() {
    let open = || MkvDemux::new(Cursor::new(H264_AAC_MKV.to_vec())).expect("fixture parses");

    let tracks: Vec<_> = open()
        .tracks()
        .iter()
        .map(|t| (t.id, t.track_type))
        .collect();
    let video = tracks
        .iter()
        .find(|(_, kind)| *kind == MkvTrackType::Video)
        .expect("a video track")
        .0;
    let audio = tracks
        .iter()
        .find(|(_, kind)| *kind == MkvTrackType::Audio)
        .expect("an audio track")
        .0;

    // Naming the tracks that are there is the same as naming none.
    assert!(open().with_tracks(Some(video), Some(audio)).is_ok());
    assert!(open().with_tracks(None, None).is_ok());

    // A track number the file does not have.
    let missing = open().with_tracks(Some(99), None);
    assert!(missing.is_err(), "track 99 does not exist");

    // The right track, asked for on the wrong pad. Silently falling back
    // would put audio on the video pad, which fails much later and much
    // more confusingly.
    let swapped = open().with_tracks(Some(audio), None);
    assert!(swapped.is_err(), "the audio track is not a video track");
}

/// "No audio" and "the usual audio" are different requests (#278).
///
/// `with_tracks(video, None)` means *the container's default* audio track.
/// Reading it as "no audio" leaves the pad declared and unlinked, and every
/// audio buffer is then dropped into it — which is what a player with no
/// sound card does for the whole file.
#[test]
fn no_audio_is_not_the_same_as_the_default_audio() {
    let open = || MkvDemux::new(Cursor::new(H264_AAC_MKV.to_vec())).expect("fixture parses");
    let pads = |demux: &MkvDemux<Cursor<Vec<u8>>>| demux.outputs().len();

    let defaulted = open().with_tracks(None, None).expect("defaults route");
    assert_eq!(pads(&defaulted), 2, "video and audio");

    let explicit_none = open()
        .with_tracks(None, None)
        .expect("defaults route")
        .video_only();
    assert_eq!(pads(&explicit_none), 1, "video only, and no audio pad");
}

/// Language, name and flags reach the track table (#278).
#[test]
fn tracks_carry_what_a_picker_needs_to_label_them() {
    let demux = MkvDemux::new(Cursor::new(H264_AAC_MKV.to_vec())).expect("fixture parses");
    let video = demux
        .tracks()
        .iter()
        .find(|t| t.track_type == MkvTrackType::Video)
        .expect("a video track");

    // ffmpeg writes FlagDefault on the tracks it muxes, and the crate
    // defaults it true regardless — either way a picker can read it.
    assert!(video.label.enabled, "a muxed track is enabled");
    assert!(
        !video.label.forced,
        "video is never forced; that flag is for subtitles"
    );

    // The fixture states no language, and "not stated" must not read as a
    // language called "und".
    assert_eq!(video.label.spoken_language(), None);
    // With neither a name nor a language there is nothing to show, and the
    // caller falls back to the codec — which it knows and this does not.
    assert_eq!(video.label.describe(), None);
}
