//! Reading a codec out of an `@media` key expression (#223).
//!
//! The `@media` plane names the codec in the key:
//!
//! ```text
//! <base>/v1/<origin>/@media/<producer>/<stream>/video/<codec>/<tier>
//! <base>/v1/<origin>/@media/<producer>/<stream>/preview/<format>
//! ```
//!
//! That is what makes a subscriber's caps *deterministic*: `prepare()` runs
//! before a single sample arrives and is synchronous, so caps derived from
//! received data would be `Video(H264)` on one run and `Any` on the next
//! depending on network timing — and under `ConverterPolicy::Deny` the same
//! pipeline string would prepare once and fail the next time. The key is known
//! at construction, so the answer is stable.
//!
//! Parsing is **positional from the right**, three chunks at most. parallax
//! deliberately does not parse the base, origin or producer chunks, does not
//! validate an origin, and does not link the keyspace crate: it subscribes to
//! a key it was handed, and only needs to know what is in it.

use crate::format::VideoCodec;

/// What an `@media` key expression says about its payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MediaKeyShape {
    /// `…/video/<codec>/<tier>` — an encoded video tier.
    Video {
        /// The codec named in the key.
        codec: VideoCodec,
        /// The tier chunk (`low`/`medium`/`high`, deployment-defined).
        tier: String,
    },
    /// `…/preview/<format>` — a preview stream, typically JPEG.
    Preview {
        /// The codec the format chunk names.
        codec: VideoCodec,
    },
    /// A key that names no codec parallax recognises.
    ///
    /// Not an error: a caller may legitimately subscribe to a key shape this
    /// module has never heard of, and the sample's own `Encoding` still says
    /// what arrived.
    Opaque,
}

/// Map a key-expression codec chunk to a [`VideoCodec`].
fn codec_chunk(chunk: &str) -> Option<VideoCodec> {
    Some(match chunk {
        "h264" | "avc" => VideoCodec::H264,
        "h265" | "hevc" => VideoCodec::H265,
        "vp8" => VideoCodec::Vp8,
        "vp9" => VideoCodec::Vp9,
        "av1" => VideoCodec::Av1,
        "jpeg" | "mjpeg" => VideoCodec::Mjpeg,
        _ => return None,
    })
}

impl MediaKeyShape {
    /// Read the trailing chunks of an `@media` key expression.
    pub fn parse(key: &str) -> Self {
        let chunks: Vec<&str> = key.split('/').filter(|c| !c.is_empty()).collect();

        // …/video/<codec>/<tier>
        if chunks.len() >= 3 {
            let n = chunks.len();
            if chunks[n - 3] == "video"
                && let Some(codec) = codec_chunk(chunks[n - 2])
            {
                return Self::Video {
                    codec,
                    tier: chunks[n - 1].to_string(),
                };
            }
        }

        // …/preview/<format>
        if chunks.len() >= 2 {
            let n = chunks.len();
            if chunks[n - 2] == "preview"
                && let Some(codec) = codec_chunk(chunks[n - 1])
            {
                return Self::Preview { codec };
            }
        }

        Self::Opaque
    }

    /// The codec this key names, if any.
    pub fn codec(&self) -> Option<VideoCodec> {
        match self {
            Self::Video { codec, .. } | Self::Preview { codec } => Some(*codec),
            Self::Opaque => None,
        }
    }
}

/// Whether a key expression contains a wildcard.
///
/// The `@media` plane forbids them on both sides. A publisher must always use
/// its concrete origin — a `*` there is a lie about identity. A subscriber
/// wildcarding the tier would receive every tier of a stream interleaved on
/// one subscriber, unseparable except by re-parsing the key per sample, which
/// is the opposite of demand-driven simulcast; wildcarding the origin would
/// pull the same stream from every host in the fleet, on the plane that
/// carries the most bytes per second.
pub fn has_wildcard(key: &str) -> bool {
    key.split('/').any(|chunk| chunk == "*" || chunk == "**")
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = "zensight/v1/h-3fa9c2d41b7e/@media/parallax/cam0/video/h264/high";

    #[test]
    fn a_real_tier_key_names_its_codec_and_tier() {
        assert_eq!(
            MediaKeyShape::parse(REAL),
            MediaKeyShape::Video {
                codec: VideoCodec::H264,
                tier: "high".to_string(),
            }
        );
    }

    #[test]
    fn parsing_is_positional_from_the_right() {
        // No base, no origin, no producer: parallax must not require them.
        assert_eq!(
            MediaKeyShape::parse("cam0/video/vp9/low").codec(),
            Some(VideoCodec::Vp9)
        );
        // …and an arbitrarily deep prefix must not disturb it.
        assert_eq!(
            MediaKeyShape::parse("a/b/c/d/e/f/g/video/av1/medium").codec(),
            Some(VideoCodec::Av1)
        );
    }

    #[test]
    fn a_preview_key_names_jpeg() {
        assert_eq!(
            MediaKeyShape::parse("zensight/v1/h-abc/@media/parallax/cam0/preview/jpeg"),
            MediaKeyShape::Preview {
                codec: VideoCodec::Mjpeg
            }
        );
    }

    #[test]
    fn an_unrecognised_key_is_opaque_not_an_error() {
        for key in [
            "some/other/plane/telemetry/fps",
            "cam0/video/theora/high", // codec we do not know
            "cam0/audio/opus/high",   // not a video shape
            "cam0",
            "",
        ] {
            assert_eq!(
                MediaKeyShape::parse(key),
                MediaKeyShape::Opaque,
                "key {key:?} should be opaque"
            );
            assert_eq!(MediaKeyShape::parse(key).codec(), None);
        }
    }

    #[test]
    fn wildcards_are_detected_anywhere_in_the_key() {
        assert!(has_wildcard(
            "zensight/v1/*/@media/parallax/cam0/video/h264/high"
        ));
        assert!(has_wildcard("cam0/video/h264/*"));
        assert!(has_wildcard("**"));
        assert!(!has_wildcard(REAL));
        // A chunk merely *containing* a star is not a zenoh wildcard chunk.
        assert!(!has_wildcard("cam0/video/h264/high*low"));
    }
}
