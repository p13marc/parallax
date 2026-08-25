//! The toolchain pin is written down twice; this is what keeps the two honest.
//!
//! `rust-toolchain.toml` is what rustup applies to every `cargo` invocation in
//! the repo, so it governs what a developer builds and lints with.
//! `.forgejo/workflows/ci.yml`'s `RUST_VERSION` is what CI installs, and it has
//! to be spelled out separately because `dtolnay/rust-toolchain` requires an
//! explicit `toolchain:` input and will not read the file.
//!
//! Two sources of truth for one fact is a drift waiting to happen — and the
//! drift it invites is exactly the one the pin exists to prevent: CI linting
//! with a different compiler than anyone can reproduce. So the two are
//! compared here, in the same spirit as `every_code_is_pinned` and
//! `test_plugin_abi_version`: a number that must not move quietly gets a test
//! that fails when it does.
//!
//! The MSRV is deliberately *not* this number. `Cargo.toml`'s `rust-version`
//! is the oldest compiler the crate promises to build on, enforced by the
//! `msrv` CI job; the pin is what development happens on. They move
//! independently, and the job overrides the file with `RUSTUP_TOOLCHAIN` so
//! that stays true.

const TOOLCHAIN_TOML: &str = include_str!("../rust-toolchain.toml");
const CI_YML: &str = include_str!("../.forgejo/workflows/ci.yml");
const CARGO_TOML: &str = include_str!("../Cargo.toml");

/// The value of `key = "value"` on the first uncommented line that sets it.
fn scalar(haystack: &str, key: &str) -> Option<String> {
    haystack.lines().find_map(|line| {
        let line = line.trim();
        if line.starts_with('#') {
            return None;
        }
        let rest = line.strip_prefix(key)?.trim_start();
        let rest = rest.strip_prefix(['=', ':'])?.trim();
        Some(rest.trim_matches(['"', '\'']).to_string())
    })
}

#[test]
fn the_pin_and_ci_name_the_same_toolchain() {
    let pinned = scalar(TOOLCHAIN_TOML, "channel").expect("rust-toolchain.toml has a channel");
    let ci = scalar(CI_YML, "RUST_VERSION").expect("ci.yml has a RUST_VERSION");

    assert_eq!(
        pinned, ci,
        "rust-toolchain.toml pins {pinned} but CI installs {ci}. \
         Whichever is right, both have to say it — a developer whose clippy is \
         not CI's clippy cannot reproduce a red build, which is the whole \
         reason the pin exists."
    );
}

/// A pin older than the MSRV would mean the crate is developed on a compiler
/// it does not claim to support, and the `msrv` job would be testing something
/// *newer* than the pin — the check inverted.
#[test]
fn the_pin_is_not_older_than_the_msrv() {
    fn parts(v: &str) -> Vec<u32> {
        v.split('.').filter_map(|p| p.parse().ok()).collect()
    }

    let pinned = scalar(TOOLCHAIN_TOML, "channel").expect("a channel");
    let msrv = scalar(CARGO_TOML, "rust-version").expect("Cargo.toml has a rust-version");

    assert!(
        parts(&pinned) >= parts(&msrv),
        "the toolchain pin ({pinned}) is older than the MSRV ({msrv})"
    );
}

/// The pin only binds CI if every job that installs a toolchain names it.
/// A single `@stable` left behind is a job that silently goes on drifting —
/// and one such job is all it takes, since they all lint the same tree.
#[test]
fn no_ci_job_installs_a_floating_toolchain() {
    let floating: Vec<&str> = CI_YML
        .lines()
        .filter(|l| l.contains("rust-toolchain@") && !l.trim_start().starts_with('#'))
        .filter(|l| !l.contains("rust-toolchain@master"))
        .collect();

    assert!(
        floating.is_empty(),
        "these steps install a moving toolchain instead of the pin: {floating:#?}"
    );
}

/// The action ref is only half the line. `rust-toolchain@master` pins the
/// *action*; `toolchain:` picks the *compiler*, and a job that says
/// `toolchain: stable` floats exactly as badly as `rust-toolchain@stable`
/// would — while sailing through the check above.
///
/// So every `toolchain:` input must name a version this repo has decided on:
/// the pin (usually through `RUST_VERSION`), or the MSRV, which the `msrv` job
/// installs deliberately. Anything else — `stable`, `beta`, `nightly` — is the
/// moving target the pin exists to remove.
#[test]
fn every_ci_toolchain_input_names_a_version_we_chose() {
    let pinned = scalar(TOOLCHAIN_TOML, "channel").expect("rust-toolchain.toml has a channel");
    let msrv = scalar(CARGO_TOML, "rust-version").expect("Cargo.toml has a rust-version");

    let floating: Vec<&str> = CI_YML
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("toolchain:") && !l.starts_with('#'))
        .filter(|l| {
            let value = l.trim_start_matches("toolchain:").trim().trim_matches('"');
            value != "${{ env.RUST_VERSION }}" && value != pinned && value != msrv
        })
        .collect();

    assert!(
        floating.is_empty(),
        "these steps select a toolchain that is neither the pin ({pinned}) nor \
         the MSRV ({msrv}): {floating:#?}"
    );
}
