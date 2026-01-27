//! Integration tests for container format elements (MP4 demuxer/muxer).
//!
//! These tests verify the container format elements work correctly.

// ============================================================================
// MP4 Demuxer Tests
// ============================================================================

#[cfg(feature = "mp4-demux")]
mod mp4_demux_tests {
    use parallax::elements::demux::{Mp4Codec, Mp4TrackType};

    #[test]
    fn test_mp4_codec_is_video() {
        assert!(Mp4Codec::H264.is_video());
        assert!(Mp4Codec::H265.is_video());
        assert!(Mp4Codec::Vp9.is_video());
        assert!(!Mp4Codec::Aac.is_video());
        assert!(!Mp4Codec::Ttxt.is_video());
    }

    #[test]
    fn test_mp4_codec_is_audio() {
        assert!(!Mp4Codec::H264.is_audio());
        assert!(!Mp4Codec::H265.is_audio());
        assert!(Mp4Codec::Aac.is_audio());
    }

    #[test]
    fn test_mp4_track_type_debug() {
        assert_eq!(format!("{:?}", Mp4TrackType::Video), "Video");
        assert_eq!(format!("{:?}", Mp4TrackType::Audio), "Audio");
        assert_eq!(format!("{:?}", Mp4TrackType::Subtitle), "Subtitle");
        assert_eq!(format!("{:?}", Mp4TrackType::Unknown), "Unknown");
    }

    #[test]
    fn test_mp4_codec_display() {
        assert_eq!(format!("{}", Mp4Codec::H264), "H.264/AVC");
        assert_eq!(format!("{}", Mp4Codec::H265), "H.265/HEVC");
        assert_eq!(format!("{}", Mp4Codec::Vp9), "VP9");
        assert_eq!(format!("{}", Mp4Codec::Aac), "AAC");
        assert_eq!(format!("{}", Mp4Codec::Ttxt), "TTXT");
        assert_eq!(format!("{}", Mp4Codec::Unknown), "Unknown");
    }
}

// ============================================================================
// MP4 Muxer Tests
// ============================================================================

#[cfg(feature = "mp4-demux")]
mod mp4_mux_tests {
    use super::*;
    use parallax::elements::mux::{
        AudioCodecConfig, Mp4AudioTrackConfig, Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig,
        VideoCodecConfig,
    };

    #[test]
    fn test_mp4_mux_config_h264() {
        let config = Mp4MuxConfig::h264();
        assert_eq!(config.major_brand, "isom");
        assert_eq!(config.timescale, 1000);
        assert!(config.compatible_brands.contains(&"avc1".to_string()));
    }

    #[test]
    fn test_mp4_mux_config_hevc() {
        let config = Mp4MuxConfig::hevc();
        assert_eq!(config.major_brand, "isom");
        assert!(config.compatible_brands.contains(&"hev1".to_string()));
    }

    #[test]
    fn test_mp4_mux_config_vp9() {
        let config = Mp4MuxConfig::vp9();
        assert_eq!(config.major_brand, "isom");
        // VP9 config should not include codec-specific brands
        assert!(!config.compatible_brands.contains(&"avc1".to_string()));
    }

    #[test]
    fn test_video_track_config_h264() {
        let sps = vec![0x67, 0x42, 0x00, 0x1f]; // Example SPS
        let pps = vec![0x68, 0xce, 0x3c, 0x80]; // Example PPS

        let config = Mp4VideoTrackConfig::h264(1920, 1080, &sps, &pps);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);

        match config.codec {
            VideoCodecConfig::H264 { sps: s, pps: p } => {
                assert_eq!(s, sps);
                assert_eq!(p, pps);
            }
            _ => panic!("Expected H264 codec config"),
        }
    }

    #[test]
    fn test_video_track_config_hevc() {
        let config = Mp4VideoTrackConfig::hevc(3840, 2160);
        assert_eq!(config.width, 3840);
        assert_eq!(config.height, 2160);
        assert!(matches!(config.codec, VideoCodecConfig::H265));
    }

    #[test]
    fn test_video_track_config_vp9() {
        let config = Mp4VideoTrackConfig::vp9(1280, 720);
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert!(matches!(config.codec, VideoCodecConfig::Vp9));
    }

    #[test]
    fn test_audio_track_config_aac() {
        let config = Mp4AudioTrackConfig::aac(48000, 2);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);

        match config.codec {
            AudioCodecConfig::Aac { profile } => {
                assert_eq!(profile, 2); // AAC-LC
            }
        }
    }

    #[test]
    fn test_audio_track_config_aac_custom_profile() {
        let config = Mp4AudioTrackConfig::aac_with_profile(44100, 1, 5);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 1);

        match config.codec {
            AudioCodecConfig::Aac { profile } => {
                assert_eq!(profile, 5); // HE-AAC
            }
        }
    }

    #[test]
    fn test_mp4_mux_create_and_add_tracks() {
        let buffer = Cursor::new(Vec::new());
        let mut mux = Mp4Mux::new(buffer, Mp4MuxConfig::default()).expect("Should create muxer");

        // Add video track
        let sps = vec![0x67, 0x42, 0x00, 0x1f];
        let pps = vec![0x68, 0xce, 0x3c, 0x80];
        let video_config = Mp4VideoTrackConfig::h264(1920, 1080, &sps, &pps);
        let video_track_id = mux
            .add_video_track(video_config)
            .expect("Should add video track");
        assert_eq!(video_track_id, 1);

        // Add audio track
        let audio_config = Mp4AudioTrackConfig::aac(48000, 2);
        let audio_track_id = mux
            .add_audio_track(audio_config)
            .expect("Should add audio track");
        assert_eq!(audio_track_id, 2);
    }

    #[test]
    fn test_mp4_mux_write_samples() {
        let buffer = Cursor::new(Vec::new());
        let mut mux = Mp4Mux::new(buffer, Mp4MuxConfig::default()).expect("Should create muxer");

        // Add video track
        let sps = vec![0x67, 0x42, 0x00, 0x1f];
        let pps = vec![0x68, 0xce, 0x3c, 0x80];
        let video_config = Mp4VideoTrackConfig::h264(640, 480, &sps, &pps);
        let video_track = mux
            .add_video_track(video_config)
            .expect("Should add video track");

        // Write some video samples
        let frame_data = vec![0u8; 1000]; // Dummy frame data
        mux.write_video_sample(video_track, &frame_data, 0, true)
            .expect("Should write keyframe");
        mux.write_video_sample(video_track, &frame_data, 33, false)
            .expect("Should write P-frame");
        mux.write_video_sample(video_track, &frame_data, 66, false)
            .expect("Should write P-frame");

        // Check stats
        let stats = mux.stats();
        assert_eq!(stats.video_samples, 3);
        assert_eq!(stats.keyframes, 1);
        assert_eq!(stats.samples_written, 3);
        assert_eq!(stats.bytes_written, 3000);
    }

    #[test]
    fn test_mp4_mux_write_audio_samples() {
        let buffer = Cursor::new(Vec::new());
        let mut mux = Mp4Mux::new(buffer, Mp4MuxConfig::default()).expect("Should create muxer");

        // Add audio track
        let audio_config = Mp4AudioTrackConfig::aac(44100, 2);
        let audio_track = mux
            .add_audio_track(audio_config)
            .expect("Should add audio track");

        // Write some audio samples
        let audio_data = vec![0u8; 512]; // Dummy AAC frame
        for i in 0..10 {
            mux.write_audio_sample(audio_track, &audio_data, i * 23)
                .expect("Should write audio sample");
        }

        // Check stats
        let stats = mux.stats();
        assert_eq!(stats.audio_samples, 10);
        assert_eq!(stats.samples_written, 10);
        // Audio samples are always sync points
        assert_eq!(stats.keyframes, 10);
    }

    #[test]
    fn test_mp4_mux_finish() {
        let buffer = Cursor::new(Vec::new());
        let mut mux = Mp4Mux::new(buffer, Mp4MuxConfig::default()).expect("Should create muxer");

        // Add a track and write a sample
        let sps = vec![0x67, 0x42, 0x00, 0x1f];
        let pps = vec![0x68, 0xce, 0x3c, 0x80];
        let video_config = Mp4VideoTrackConfig::h264(320, 240, &sps, &pps);
        let video_track = mux
            .add_video_track(video_config)
            .expect("Should add video track");

        let frame_data = vec![0u8; 100];
        mux.write_video_sample(video_track, &frame_data, 0, true)
            .expect("Should write frame");

        // Finish the file
        let output = mux.finish().expect("Should finalize MP4");

        // Verify the output has MP4 data
        let data = output.into_inner();
        assert!(!data.is_empty(), "Output should not be empty");

        // Check for ftyp box (should start near the beginning)
        let ftyp_found = data.windows(4).any(|w| w == b"ftyp");
        assert!(ftyp_found, "Should contain ftyp box");

        // Check for moov box (metadata)
        let moov_found = data.windows(4).any(|w| w == b"moov");
        assert!(moov_found, "Should contain moov box");
    }

    #[test]
    fn test_mp4_mux_multiple_tracks() {
        let buffer = Cursor::new(Vec::new());
        let mut mux = Mp4Mux::new(buffer, Mp4MuxConfig::default()).expect("Should create muxer");

        // Add multiple tracks
        let sps = vec![0x67, 0x42, 0x00, 0x1f];
        let pps = vec![0x68, 0xce, 0x3c, 0x80];

        let video1 = mux
            .add_video_track(Mp4VideoTrackConfig::h264(1920, 1080, &sps, &pps))
            .expect("Track 1");
        let video2 = mux
            .add_video_track(Mp4VideoTrackConfig::hevc(1920, 1080))
            .expect("Track 2");
        let audio1 = mux
            .add_audio_track(Mp4AudioTrackConfig::aac(48000, 2))
            .expect("Track 3");
        let audio2 = mux
            .add_audio_track(Mp4AudioTrackConfig::aac(44100, 1))
            .expect("Track 4");

        assert_eq!(video1, 1);
        assert_eq!(video2, 2);
        assert_eq!(audio1, 3);
        assert_eq!(audio2, 4);
    }
}

// ============================================================================
// MP4 Round-trip Tests (Mux then Demux)
// ============================================================================

#[cfg(feature = "mp4-demux")]
mod mp4_roundtrip_tests {
    use super::*;
    use parallax::elements::demux::{Mp4Codec, Mp4Demux, Mp4TrackType};
    use parallax::elements::mux::{Mp4AudioTrackConfig, Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};

    #[test]
    fn test_mp4_roundtrip_video_only() {
        // Create an MP4 with video track
        let buffer = Cursor::new(Vec::new());
        let mut mux = Mp4Mux::new(buffer, Mp4MuxConfig::default()).expect("Should create muxer");

        let sps = vec![0x67, 0x42, 0x00, 0x1f, 0xe9, 0x01, 0x40, 0x7a, 0xc1, 0x00];
        let pps = vec![0x68, 0xce, 0x3c, 0x80];
        let video_config = Mp4VideoTrackConfig::h264(320, 240, &sps, &pps);
        let video_track = mux
            .add_video_track(video_config)
            .expect("Should add video track");

        // Write some frames
        let frame1 = vec![0x00, 0x00, 0x00, 0x01, 0x65]; // IDR frame marker
        let frame2 = vec![0x00, 0x00, 0x00, 0x01, 0x41]; // P frame marker
        let frame3 = vec![0x00, 0x00, 0x00, 0x01, 0x41]; // P frame marker

        mux.write_video_sample(video_track, &frame1, 0, true)
            .unwrap();
        mux.write_video_sample(video_track, &frame2, 33, false)
            .unwrap();
        mux.write_video_sample(video_track, &frame3, 66, false)
            .unwrap();

        let output = mux.finish().expect("Should finalize");
        let mp4_data = output.into_inner();

        // Now demux the MP4
        let reader = Cursor::new(mp4_data.clone());
        let mut demux =
            Mp4Demux::new(reader, mp4_data.len() as u64).expect("Should create demuxer");

        // Check tracks
        let tracks = demux.tracks();
        assert_eq!(tracks.len(), 1, "Should have 1 track");
        assert_eq!(tracks[0].track_type, Mp4TrackType::Video);
        assert_eq!(tracks[0].codec, Mp4Codec::H264);
        assert_eq!(tracks[0].video_info.as_ref().unwrap().width, 320);
        assert_eq!(tracks[0].video_info.as_ref().unwrap().height, 240);

        // Read samples
        let video_id = demux.video_track_id().expect("Should have video track");

        let sample1 = demux.read_sample(video_id).expect("Should read sample 1");
        assert!(sample1.is_some());
        let s1 = sample1.unwrap();
        assert!(s1.is_keyframe);

        let sample2 = demux.read_sample(video_id).expect("Should read sample 2");
        assert!(sample2.is_some());
        let s2 = sample2.unwrap();
        assert!(!s2.is_keyframe);

        let sample3 = demux.read_sample(video_id).expect("Should read sample 3");
        assert!(sample3.is_some());

        // Should be end of track
        let sample4 = demux.read_sample(video_id).expect("Should handle end");
        assert!(sample4.is_none());
    }

    #[test]
    fn test_mp4_roundtrip_audio_only() {
        // Create an MP4 with audio track
        let buffer = Cursor::new(Vec::new());
        let mut mux = Mp4Mux::new(buffer, Mp4MuxConfig::default()).expect("Should create muxer");

        let audio_config = Mp4AudioTrackConfig::aac(44100, 2);
        let audio_track = mux
            .add_audio_track(audio_config)
            .expect("Should add audio track");

        // Write some audio frames (dummy AAC data)
        for i in 0..5 {
            let audio_data = vec![0xFF, 0xF1, 0x50, 0x80]; // AAC ADTS header start
            mux.write_audio_sample(audio_track, &audio_data, i * 23)
                .unwrap();
        }

        let output = mux.finish().expect("Should finalize");
        let mp4_data = output.into_inner();

        // Now demux the MP4
        let reader = Cursor::new(mp4_data.clone());
        let mut demux =
            Mp4Demux::new(reader, mp4_data.len() as u64).expect("Should create demuxer");

        // Check tracks
        let tracks = demux.tracks();
        assert_eq!(tracks.len(), 1, "Should have 1 track");
        assert_eq!(tracks[0].track_type, Mp4TrackType::Audio);
        assert_eq!(tracks[0].codec, Mp4Codec::Aac);

        let audio_info = tracks[0]
            .audio_info
            .as_ref()
            .expect("Should have audio info");
        assert_eq!(audio_info.sample_rate, 44100);
        assert_eq!(audio_info.channels, 2);

        // Read all samples
        let audio_id = demux.audio_track_id().expect("Should have audio track");
        let samples = demux
            .read_all_samples(audio_id)
            .expect("Should read samples");
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_mp4_roundtrip_av() {
        // Create an MP4 with both video and audio
        let buffer = Cursor::new(Vec::new());
        let mut mux = Mp4Mux::new(buffer, Mp4MuxConfig::default()).expect("Should create muxer");

        let sps = vec![0x67, 0x42, 0x00, 0x1f];
        let pps = vec![0x68, 0xce, 0x3c, 0x80];
        let video_config = Mp4VideoTrackConfig::h264(640, 480, &sps, &pps);
        let video_track = mux
            .add_video_track(video_config)
            .expect("Should add video track");

        let audio_config = Mp4AudioTrackConfig::aac(48000, 2);
        let audio_track = mux
            .add_audio_track(audio_config)
            .expect("Should add audio track");

        // Interleave video and audio samples
        let video_data = vec![0u8; 100];
        let audio_data = vec![0u8; 50];

        mux.write_video_sample(video_track, &video_data, 0, true)
            .unwrap();
        mux.write_audio_sample(audio_track, &audio_data, 0).unwrap();
        mux.write_video_sample(video_track, &video_data, 33, false)
            .unwrap();
        mux.write_audio_sample(audio_track, &audio_data, 21)
            .unwrap();
        mux.write_video_sample(video_track, &video_data, 66, false)
            .unwrap();
        mux.write_audio_sample(audio_track, &audio_data, 42)
            .unwrap();

        let output = mux.finish().expect("Should finalize");
        let mp4_data = output.into_inner();

        // Demux
        let reader = Cursor::new(mp4_data.clone());
        let mut demux =
            Mp4Demux::new(reader, mp4_data.len() as u64).expect("Should create demuxer");

        // Check we have both tracks
        let tracks = demux.tracks();
        assert_eq!(tracks.len(), 2, "Should have 2 tracks");

        let video_track_id = demux.video_track_id().expect("Should have video");
        let audio_track_id = demux.audio_track_id().expect("Should have audio");

        // Read video samples
        let video_samples = demux
            .read_all_samples(video_track_id)
            .expect("Should read video");
        assert_eq!(video_samples.len(), 3);
        assert!(video_samples[0].is_keyframe);
        assert!(!video_samples[1].is_keyframe);

        // Read audio samples
        demux.seek_to_start(audio_track_id); // Reset position
        let audio_samples = demux
            .read_all_samples(audio_track_id)
            .expect("Should read audio");
        assert_eq!(audio_samples.len(), 3);
    }
}
