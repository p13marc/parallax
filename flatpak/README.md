# Flatpak packaging

`com.github.p13marc.ParallaxPlayer` — the `parallax-player` workspace member
as a flatpak, on freedesktop Platform/Sdk 25.08 with the rust-stable SDK
extension (fleet-standard manifest shape, mirroring tcgui/zensight).

Built by the `flatpak` job in `.forgejo/workflows/release.yml` on every
release tag: an unsigned export is force-pushed to the `flatpak-export`
branch (vm-edge signs and publishes it to https://flatpak.marcpardo.eu), and
a versioned `parallax-player-<tag>.flatpak` bundle is attached to the
release.

**CI builds with software rendering only** — the runner has no GPU. wgpu
falls back to Mesa llvmpipe from the runtime and VA-API probing fails over
to the software decoders; a "no suitable adapter"/black-window symptom in a
GPU-less environment is expected, not a bug. GPU paths are validated on the
workstation hardware lane (`PARALLAX_FORCE_GPU=1`).

Codec system libraries (libvpx, dav1d, libopus, alsa-lib, libva) come from
the 25.08 Platform; their headers and the libclang that bindgen needs come
from the Sdk. Nothing GPU-related is bundled.

Local build (needs flatpak-builder + the 25.08 runtime/Sdk + rust-stable
extension):

    flatpak-builder --user --force-clean --install-deps-from=flathub \
      builddir flatpak/com.github.p13marc.ParallaxPlayer.yml
