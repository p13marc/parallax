# Parallax - Justfile
# Run `just --list` to see available recipes

# Default recipe: run tests
default: test

# Build the project
build:
    cargo build

# Build in release mode
build-release:
    cargo build --release

# Run all tests with nextest
test:
    cargo nextest run

# Run tests with verbose output
test-verbose:
    cargo nextest run --no-capture

# Run a specific test
test-one NAME:
    cargo nextest run {{NAME}}

# Run tests matching a pattern
test-filter PATTERN:
    cargo nextest run -E 'test({{PATTERN}})'

# Run tests with coverage (requires cargo-llvm-cov)
coverage:
    cargo llvm-cov nextest

# Run clippy lints
lint:
    cargo clippy -- -D warnings

# Feature combo the zensight video sensor builds against (mirrors CI)
sensor_features := "zenoh,h264,v4l2,rtp,rtsp,image-jpeg,hotplug"

# Check + test + lint the sensor feature combo (mirrors CI's test-sensor/clippy jobs)
check-sensor:
    cargo check --all-targets --features {{sensor_features}}
    cargo nextest run --features {{sensor_features}}
    cargo clippy --all-targets --features {{sensor_features}},image-codecs,zenoh-unstable -- -D warnings

# Pure-Rust media/container feature combo (mirrors CI's media checks)
media_features := "mp4-demux,mkv-demux,mpeg-ts,audio-aac,audio-vorbis,http"

# Check + test + lint the media/container combo (mirrors CI)
check-media:
    cargo check --all-targets --features {{media_features}}
    cargo nextest run --features {{media_features}}
    cargo clippy --all-targets --features {{media_features}} -- -D warnings

# Media combo plus the system-library codecs (libvpx/libdav1d/libopus dev
# packages + libclang required, and nasm for rav1e; not in CI). This is the
# only recipe that clippies `av1-encode` — no CI job does.
media_full_features := media_features + ",vpx,av1-decode,av1-encode,opus,h264"

check-media-full:
    cargo nextest run --features {{media_full_features}}
    cargo clippy --all-targets --features {{media_full_features}} -- -D warnings

# Media-path benchmarks (demux/decode/convert) — needs the codec system libs
bench-media:
    cargo bench --features h264,mkv-demux --bench media_path

# Check + test + lint the display combo, GPU backend included (#190).
# Compiles headless (wgpu loads drivers at runtime); the GPU path itself
# needs a real session to exercise.
#
# `display` ALONE is linted too, and not for completeness: everything under
# `present/` behind `display-gpu` is invisible to the combined build, so a
# module that loses its cfg gate compiles here and breaks CI's display-check.
# That has happened.
check-display:
    cargo nextest run --features display,display-gpu
    cargo clippy --all-targets --features display,display-gpu -- -D warnings
    cargo clippy --all-targets --features display -- -D warnings

# Vulkan Video (#3). Mirrors CI's vulkan-check, and exists because of root
# CLAUDE.md gotcha 11b: `elements::codec::hw_encoder`/`hw_decoder` need
# `vulkan-video` AND a codec feature, so this combination is the only one
# that compiles them. They had rotted before; a `gpu::Codec` variant added
# for VA-API broke them again.
check-vulkan:
    cargo check --all-targets --features image-jpeg,vulkan-video

# Colorspace SIMD combo (the `yuv` crate) — what parallax-player actually
# builds with. The scalar arms are the default-feature fallback, so both
# families need running: a stride or format change touches two twins.
check-simd:
    cargo nextest run --features simd-colorspace
    cargo clippy --all-targets --features simd-colorspace -- -D warnings

# Feature gates no other recipe or CI job compiled before this existed.
# `aac-encode` had rotted un-compiled against an fdk-aac signature that never
# shipped; mirrors CI's feature-gates job. Needs g++ (fdk-aac builds FDK from
# source), libclang + kernel headers (v4l2r bindgen) and libasound2/alsa-lib.
check-feature-gates:
    cargo nextest run --features aac-encode,websocket,alsa,v4l2-m2m
    cargo clippy --all-targets --features aac-encode,websocket,alsa,v4l2-m2m -- -D warnings

# Check + test + lint VA-API hardware decode (#193). Build needs libva-devel
# + libclang (cros-libva runs bindgen); the tests green-skip when no VA driver
# is present, so this is meaningful on any machine.
#
# NOTE what your driver actually decodes: Fedora's libva-intel-media-driver is
# built WITHOUT H.264/HEVC (patent-encumbered) — the same hardware exposes them
# under RPM Fusion's intel-media-driver-freeworld. `vainfo` is the ground truth.
# `mkv-demux` rides along because the decode integration tests need a
# container to read a fixture out of; `vpx` and `h264` because the strongest
# test there is a bit-exact comparison against the software decoder, and video
# decoding is normative enough that "close enough" is not a thing.
check-vaapi:
    cargo nextest run --features vaapi,mkv-demux,mp4-demux,vpx,h264
    cargo clippy --all-targets --features vaapi,mkv-demux,mp4-demux,vpx,h264 -- -D warnings

# Check + test + lint the V4L2 M2M hardware encoder (mirrors CI's test-v4l2-m2m job).
# Needs libclang + kernel headers (on immutable Fedora: run inside a toolbox).
# Live queue test: modprobe vicodec, then PARALLAX_VICODEC_TEST_DEVICE=auto just check-m2m
check-m2m:
    cargo clippy --all-targets --features v4l2-m2m -- -D warnings
    cargo nextest run --features v4l2-m2m -E 'test(v4l2_m2m)'

# Run clippy with all features
lint-all:
    cargo clippy --all-features -- -D warnings

# Format code
fmt:
    cargo fmt

# Check formatting without modifying
fmt-check:
    cargo fmt --check

# Run all checks (format, lint, test)
check: fmt-check lint test

# Run benchmarks
bench:
    cargo bench

# Run a specific benchmark
bench-one NAME:
    cargo bench --bench {{NAME}}

# Clean build artifacts
clean:
    cargo clean

# Generate documentation
doc:
    cargo doc --no-deps

# Open documentation in browser
doc-open:
    cargo doc --no-deps --open

# Watch for changes and run tests
watch:
    cargo watch -x 'nextest run'

# Watch for changes and run clippy
watch-lint:
    cargo watch -x 'clippy -- -D warnings'

# Install development dependencies
dev-deps:
    cargo install cargo-nextest cargo-watch cargo-llvm-cov

# Show project statistics
stats:
    @echo "Lines of code:"
    @tokei src/
    @echo ""
    @echo "Dependencies:"
    @cargo tree --depth 1

# Run memory pool benchmark
bench-pool:
    cargo bench --bench memory_pool

# Run throughput benchmark
bench-throughput:
    cargo bench --bench throughput

# Serve a local H.264 RTSP test stream (for the RTSP examples; needs
# python3-gobject + gstreamer1-rtsp-server)
rtsp-server:
    ./scripts/rtsp_test_server.py

# Decode-perf matrix (#192): build the player, run interleaved A/B/A
# measurement passes over FILE. Extra configs via the script directly.
decode-matrix FILE="~/Videos/boss_pokemon.webm":
    cargo build -p parallax-player --release
    ./scripts/decode_matrix.py --file {{FILE}} \
        --config "baseline:" \
        --config "t4:--dav1d-threads 4" \
        --config "ahead16:--decode-ahead 16" \
        --config "t4-ahead16:--dav1d-threads 4 --decode-ahead 16"
