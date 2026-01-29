//! Streaming protocol elements for adaptive bitrate delivery.
//!
//! This module provides elements for outputting media streams via
//! HTTP-based adaptive streaming protocols:
//!
//! - **HLS (HTTP Live Streaming)** - Apple's protocol, widely supported
//! - **DASH (Dynamic Adaptive Streaming over HTTP)** - MPEG standard
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Streaming Output                              │
//! │                                                                  │
//! │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
//! │  │ Encoder  │───▶│ Muxer    │───▶│ HlsSink  │───▶│ Files    │ │
//! │  │ (H.264)  │    │ (TS/fMP4)│    │ or       │    │ + M3U8/  │ │
//! │  └──────────┘    └──────────┘    │ DashSink │    │ MPD      │ │
//! │                                   └──────────┘    └──────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # HLS Output
//!
//! HLS (HTTP Live Streaming) segments media into `.ts` files and generates
//! M3U8 playlists that players use to download and play segments.
//!
//! ```rust,ignore
//! use parallax::elements::streaming::{HlsSink, HlsConfig};
//!
//! let sink = HlsSink::new(HlsConfig {
//!     output_dir: "/var/www/stream".into(),
//!     segment_duration: 6.0,
//!     playlist_length: 5,
//!     ..Default::default()
//! })?;
//!
//! // Add to pipeline after TS muxer
//! pipeline.add_node("hls", sink);
//! ```
//!
//! # DASH Output
//!
//! DASH (Dynamic Adaptive Streaming over HTTP) uses fragmented MP4 segments
//! and XML manifests (MPD - Media Presentation Description).
//!
//! ```rust,ignore
//! use parallax::elements::streaming::{DashSink, DashConfig};
//!
//! let sink = DashSink::new(DashConfig {
//!     output_dir: "/var/www/stream".into(),
//!     segment_duration: 4.0,
//!     ..Default::default()
//! })?;
//! ```
//!
//! # Adaptive Bitrate (ABR)
//!
//! For multi-bitrate streaming, create parallel encoding pipelines:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    ABR Pipeline                                  │
//! │                                                                  │
//! │  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
//! │  │ Source   │───▶│ Enc 1080p│───▶│ HlsSink  │───▶ 1080p/*.ts  │
//! │  │ (Camera) │    └──────────┘    └──────────┘                  │
//! │  │          │                                                   │
//! │  │          │    ┌──────────┐    ┌──────────┐                  │
//! │  │          │───▶│ Enc 720p │───▶│ HlsSink  │───▶ 720p/*.ts   │
//! │  │          │    └──────────┘    └──────────┘                  │
//! │  │          │                                                   │
//! │  │          │    ┌──────────┐    ┌──────────┐                  │
//! │  │          │───▶│ Enc 480p │───▶│ HlsSink  │───▶ 480p/*.ts   │
//! │  └──────────┘    └──────────┘    └──────────┘                  │
//! │                                                                  │
//! │                  Master Playlist: master.m3u8                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Feature Flags
//!
//! - `hls` - Enable HLS output (requires `mpeg-ts` feature)
//! - `dash` - Enable DASH output (requires `mp4-demux` feature)
//! - `streaming-protocols` - Enable both HLS and DASH

mod dash;
mod hls;
mod segment;

pub use dash::{DashAdaptationSet, DashConfig, DashRepresentation, DashSink, DashStats};
pub use hls::{HlsConfig, HlsSink, HlsStats, HlsVariant};
pub use segment::{SegmentBoundaryDetector, SegmentInfo, SegmentWriter};
