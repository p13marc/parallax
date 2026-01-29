//! Media codec elements using pure Rust implementations.
//!
//! This module provides encoding and decoding elements for video, audio, and images.
//! All codecs are implemented in pure Rust with no C dependencies (except where noted).
//!
//! # Video Codecs
//!
//! | Codec | Feature | Decoder | Encoder | Pure Rust |
//! |-------|---------|---------|---------|-----------|
//! | H.264 | `h264` | [`H264Decoder`] | [`H264Encoder`] | No (OpenH264) |
//! | AV1 | `av1-decode` | [`Dav1dDecoder`] | - | No (C lib) |
//! | AV1 | `av1-encode` | - | [`Rav1eEncoder`] | Yes |
//!
//! # Audio Codecs
//!
//! | Codec | Feature | Decoder | Encoder | Pure Rust |
//! |-------|---------|---------|---------|-----------|
//! | Opus | `opus` | [`OpusDecoder`] | [`OpusEncoder`] | No (libopus) |
//! | AAC | `aac-encode` | - | [`AacEncoder`] | No (FDK-AAC) |
//! | FLAC | `audio-flac` | [`SymphoniaDecoder`] | - | Yes |
//! | MP3 | `audio-mp3` | [`SymphoniaDecoder`] | - | Yes |
//! | AAC | `audio-aac` | [`SymphoniaDecoder`] | - | Yes |
//! | Vorbis | `audio-vorbis` | [`SymphoniaDecoder`] | - | Yes |
//!
//! # Image Codecs
//!
//! | Format | Feature | Decoder | Encoder | Pure Rust |
//! |--------|---------|---------|---------|-----------|
//! | JPEG | `image-jpeg` | [`JpegDecoder`] | - | Yes |
//! | PNG | `image-png` | [`PngDecoder`] | [`PngEncoder`] | Yes |
//!
//! # Feature Flags
//!
//! ```toml
//! # Video codecs
//! parallax = { version = "0.1", features = ["h264"] }        # H.264 encoder/decoder (OpenH264)
//! parallax = { version = "0.1", features = ["av1-encode"] }  # AV1 encoder
//! parallax = { version = "0.1", features = ["av1-decode"] }  # AV1 decoder (needs libdav1d)
//!
//! # Audio codecs
//! parallax = { version = "0.1", features = ["opus"] }          # Opus encoder/decoder (needs libopus)
//! parallax = { version = "0.1", features = ["aac-encode"] }    # AAC encoder (FDK-AAC, license restrictions)
//! parallax = { version = "0.1", features = ["audio-codecs"] }  # All Symphonia decoders
//! parallax = { version = "0.1", features = ["audio-flac"] }    # FLAC decoder only
//! parallax = { version = "0.1", features = ["audio-mp3"] }     # MP3 decoder only
//!
//! # Image codecs (all pure Rust)
//! parallax = { version = "0.1", features = ["image-codecs"] }  # All image codecs
//! parallax = { version = "0.1", features = ["image-jpeg"] }    # JPEG only
//! parallax = { version = "0.1", features = ["image-png"] }     # PNG only
//! ```
//!
//! # Build Dependencies
//!
//! Most codecs are pure Rust and require no external dependencies.
//!
//! ## opus
//!
//! Requires the **libopus** system library:
//!
//! - **Fedora/RHEL**: `sudo dnf install opus-devel`
//! - **Debian/Ubuntu**: `sudo apt install libopus-dev`
//! - **Arch**: `sudo pacman -S opus`
//! - **macOS**: `brew install opus`
//!
//! ## aac-encode (FDK-AAC)
//!
//! Requires **FDK-AAC** library. **Note: License restrictions for commercial use.**
//!
//! - **Fedora/RHEL**: `sudo dnf install fdk-aac-devel` (from RPM Fusion)
//! - **Debian/Ubuntu**: `sudo apt install libfdk-aac-dev`
//! - **Arch**: `sudo pacman -S libfdk-aac`
//! - **macOS**: `brew install fdk-aac`
//!
//! ## av1-encode (rav1e)
//!
//! Optionally install **nasm** for x86_64 SIMD optimizations:
//!
//! - **Fedora/RHEL**: `sudo dnf install nasm`
//! - **Debian/Ubuntu**: `sudo apt install nasm`
//! - **Arch**: `sudo pacman -S nasm`
//! - **macOS**: `brew install nasm`
//!
//! ## h264 (OpenH264)
//!
//! Requires a **C++ compiler** to build OpenH264 from source:
//!
//! - **Fedora/RHEL**: `sudo dnf install gcc-c++`
//! - **Debian/Ubuntu**: `sudo apt install g++`
//! - **Arch**: `sudo pacman -S gcc`
//! - **macOS**: `xcode-select --install`
//!
//! ## av1-decode (dav1d)
//!
//! Requires the **libdav1d** system library:
//!
//! - **Fedora/RHEL**: `sudo dnf install libdav1d-devel`
//! - **Debian/Ubuntu**: `sudo apt install libdav1d-dev`
//! - **Arch**: `sudo pacman -S dav1d`
//! - **macOS**: `brew install dav1d`

// Common types (video frames, pixel formats)
mod common;
pub use common::{PixelFormat, VideoFrame};

// Video codec traits
mod traits;
pub use traits::{FrameType, VideoDecoder, VideoEncoder};

// Audio codec traits
mod audio_traits;
pub use audio_traits::{AudioDecoder, AudioEncoder, AudioSampleFormat, AudioSamples};

// Video element wrappers
mod decoder_element;
mod encoder_element;
pub use decoder_element::DecoderElement;
pub use encoder_element::EncoderElement;

// Hardware codec element wrappers (Vulkan Video)
#[cfg(feature = "vulkan-video")]
mod hw_decoder;
#[cfg(feature = "vulkan-video")]
mod hw_encoder;
#[cfg(feature = "vulkan-video")]
pub use hw_decoder::HwDecoderElement;
#[cfg(feature = "vulkan-video")]
pub use hw_encoder::HwEncoderElement;

// Audio element wrappers
mod audio_decoder_element;
mod audio_encoder_element;
pub use audio_decoder_element::AudioDecoderElement;
pub use audio_encoder_element::AudioEncoderElement;

// H.264 video codec
#[cfg(feature = "h264")]
mod h264;
#[cfg(feature = "h264")]
pub use h264::{DecodedFrame, H264Decoder, H264Encoder, H264EncoderConfig};

// AV1 video codecs
#[cfg(feature = "av1-decode")]
mod decoder;
#[cfg(feature = "av1-decode")]
pub use decoder::Dav1dDecoder;

#[cfg(feature = "av1-encode")]
mod encoder;
#[cfg(feature = "av1-encode")]
pub use encoder::{Rav1eConfig, Rav1eEncoder};

// Audio codecs - Symphonia (decode only)
#[cfg(any(
    feature = "audio-flac",
    feature = "audio-mp3",
    feature = "audio-aac",
    feature = "audio-vorbis"
))]
mod audio;

#[cfg(any(
    feature = "audio-flac",
    feature = "audio-mp3",
    feature = "audio-aac",
    feature = "audio-vorbis"
))]
pub use audio::{AudioFormat, AudioFrameInfo, SampleFormat, SymphoniaDecoder};

// Opus audio codec (encode + decode)
#[cfg(feature = "opus")]
mod opus;
#[cfg(feature = "opus")]
pub use opus::{OpusApplication, OpusDecoder, OpusEncoder};

// AAC audio encoder (FDK-AAC)
#[cfg(feature = "aac-encode")]
mod aac;
#[cfg(feature = "aac-encode")]
pub use aac::AacEncoder;

// Image codecs
#[cfg(any(feature = "image-jpeg", feature = "image-png"))]
mod image;

#[cfg(any(feature = "image-jpeg", feature = "image-png"))]
pub use image::{ColorType, ImageFrame};

#[cfg(feature = "image-jpeg")]
pub use image::JpegDecoder;

#[cfg(feature = "image-png")]
pub use image::{PngDecoder, PngEncoder};
