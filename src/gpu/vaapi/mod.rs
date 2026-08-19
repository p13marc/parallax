//! VA-API hardware video decode (#193).
//!
//! The hardware-decode backend that is actually validatable on commodity
//! Intel and AMD graphics, unlike Vulkan Video ([`crate::gpu::vulkan`], #3,
//! which needs Gen12+ / RADV). Decode work runs on the GPU's fixed-function
//! video engine; the CPU cost of a decoded frame collapses to the readback.
//!
//! # What lives here
//!
//! [`VaDisplay`] is the capability layer: opening a display, asking which
//! codecs the driver will actually decode, and answering
//! [`vaapi_available`](crate::gpu::vaapi_available). The decoder elements
//! themselves are in [`crate::elements::codec`], per the #160 codec-surface
//! rule (video decoders implement `Element` directly).
//!
//! # Probe first, always
//!
//! Every entry point here is a *question*, not an assertion: whether a
//! display opens, whether a profile exists, whether an entrypoint is
//! supported. A VA-API decoder must answer all of them before it constructs
//! anything, and fall back to software when any answer is no. Two reasons
//! this is not optional:
//!
//! - Drivers ship different codec sets. Fedora's `libva-intel-media-driver`
//!   is built without H.264 and HEVC (patent-encumbered); the same hardware
//!   exposes them under RPM Fusion's `intel-media-driver-freeworld`. The
//!   same GPU therefore has different capabilities depending on packaging.
//! - A driver can advertise a profile it will not actually build a config
//!   for, so the only honest probe is to build one — see
//!   [`VaDisplay::decode_config_works`].

use std::sync::Arc;

use cros_codecs::libva;

mod frame;
pub use frame::{VaFrame, VaFrameDescriptor, VaFramePool};

use super::Codec;

/// A VA-API display, plus what it can decode.
///
/// Opening one costs a driver load, so callers keep it and share it across
/// decoders (it is `Arc`-shared internally — `cros-codecs` takes
/// `Arc<Display>`).
pub struct VaDisplay {
    display: Arc<libva::Display>,
    profiles: Vec<libva::VAProfile::Type>,
}

impl VaDisplay {
    /// Open the default render node, falling back to whatever DRM device
    /// answers.
    ///
    /// `/dev/dri/renderD128` first because it is the unprivileged node —
    /// it needs no session, no seat and no compositor, which is what makes
    /// hardware decode usable from a headless process. `Display::open`
    /// walks every DRM device and is the fallback for boxes that number
    /// their render nodes differently.
    pub fn open() -> Option<Self> {
        let display = libva::Display::open_drm_display("/dev/dri/renderD128")
            .ok()
            .or_else(libva::Display::open)?;
        let profiles = display.query_config_profiles().ok()?;
        Some(Self { display, profiles })
    }

    /// The underlying display, for handing to a decoder backend.
    pub fn handle(&self) -> Arc<libva::Display> {
        Arc::clone(&self.display)
    }

    /// The driver's self-reported name, e.g.
    /// `"Intel iHD driver for Intel(R) Gen Graphics - 26.2.4"`.
    pub fn vendor(&self) -> String {
        self.display
            .query_vendor_string()
            .unwrap_or_else(|_| "unknown".to_string())
    }

    /// Whether this display will decode `codec`.
    ///
    /// Both halves are checked: that the profile exists at all, and that it
    /// offers `VAEntrypointVLD` (decode). A driver can list a profile for
    /// encode only, and treating that as decode support is how you get a
    /// failure at the first frame instead of at construction.
    pub fn supports_decode(&self, codec: Codec) -> bool {
        self.decode_profiles(codec).next().is_some()
    }

    /// The profiles this display will decode `codec` with, in the order the
    /// driver reported them.
    pub fn decode_profiles(
        &self,
        codec: Codec,
    ) -> impl Iterator<Item = libva::VAProfile::Type> + '_ {
        let wanted = profiles_for(codec);
        self.profiles
            .iter()
            .copied()
            .filter(move |p| wanted.contains(p))
            .filter(|p| {
                self.display
                    .query_config_entrypoints(*p)
                    .is_ok_and(|e| e.contains(&libva::VAEntrypoint::VAEntrypointVLD))
            })
    }

    /// Whether a decode config for `codec` can actually be *created*, not
    /// merely advertised.
    ///
    /// [`supports_decode`](Self::supports_decode) reads the driver's own
    /// profile and entrypoint tables. This asks the driver to build the
    /// thing, which is a strictly stronger question: a profile can be listed
    /// with a VLD entrypoint and still refuse the config (an unsupported
    /// `RT_FORMAT`, a disabled engine, a driver that reports optimistically).
    /// Creating a config is cheap and destroying it is immediate, so asking
    /// costs nothing next to discovering it at the first frame.
    ///
    /// Returns true on the first profile that answers, so a driver offering
    /// only the 10-bit profile of a codec still reads as capable.
    pub fn decode_config_works(&self, codec: Codec) -> bool {
        self.decode_profiles(codec).any(|profile| {
            self.display
                .create_config(
                    vec![libva::VAConfigAttrib {
                        type_: libva::VAConfigAttribType::VAConfigAttribRTFormat,
                        value: libva::VA_RT_FORMAT_YUV420,
                    }],
                    profile,
                    libva::VAEntrypoint::VAEntrypointVLD,
                )
                .is_ok()
        })
    }

    /// Every codec this display can decode — for diagnostics and for the
    /// `--hwdec` report.
    pub fn decodable(&self) -> Vec<Codec> {
        [Codec::H264, Codec::H265, Codec::Vp9, Codec::Av1]
            .into_iter()
            .filter(|c| self.supports_decode(*c))
            .collect()
    }
}

impl std::fmt::Debug for VaDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VaDisplay")
            .field("vendor", &self.vendor())
            .field("decodable", &self.decodable())
            .finish()
    }
}

/// The VA profiles that mean "this codec", per codec.
///
/// A profile list rather than one profile: hardware advertises the coding
/// profiles it implements (Main, High, …), and any of them decoding is
/// enough to say the codec is supported. Deliberately no `ConstrainedBaseline`
/// for H.264 — every decoder that has it also has `Main`, and listing it
/// alone would claim support for streams the hardware may refuse.
fn profiles_for(codec: Codec) -> &'static [libva::VAProfile::Type] {
    use libva::VAProfile::*;
    match codec {
        Codec::H264 => &[VAProfileH264Main, VAProfileH264High],
        Codec::H265 => &[VAProfileHEVCMain, VAProfileHEVCMain10],
        Codec::Vp9 => &[VAProfileVP9Profile0, VAProfileVP9Profile2],
        Codec::Av1 => &[VAProfileAV1Profile0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe answers rather than panicking, on any machine.
    ///
    /// This is the whole contract of this module: on a box with no DRM
    /// device, no driver, or a driver without video support, `open()`
    /// returns `None` and every caller falls back to software.
    #[test]
    fn opening_a_display_never_panics() {
        match VaDisplay::open() {
            None => eprintln!("no VA display here — software fallback is the answer"),
            Some(d) => {
                eprintln!("VA-API: {} decodes {:?}", d.vendor(), d.decodable());
                // Whatever it reports, it must be self-consistent — and a
                // codec it advertises must survive the stronger probe, or the
                // decoder would construct and then fail at the first frame.
                for codec in d.decodable() {
                    assert!(d.supports_decode(codec));
                    assert!(
                        d.decode_config_works(codec),
                        "{} advertises {codec} but refuses a decode config for it",
                        d.vendor()
                    );
                }
            }
        }
    }

    /// Every codec maps to at least one profile, so `supports_decode` can
    /// never be vacuously false because the table forgot an arm.
    #[test]
    fn every_codec_has_profiles() {
        for codec in [Codec::H264, Codec::H265, Codec::Vp9, Codec::Av1] {
            assert!(!profiles_for(codec).is_empty(), "{codec} has no profiles");
        }
    }
}
