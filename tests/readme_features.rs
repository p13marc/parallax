//! The README's feature table has to agree with `Cargo.toml`.
//!
//! It did not, in both directions at once: three features it documented had
//! been deleted — `huge-pages` and `rdma` with the segment abstraction, `gpu`
//! with the unused `gpu-allocator` dependency — so anyone copying a row into
//! their own manifest got a hard cargo error. And six real ones were missing,
//! including `cli`, which gates the `parallax-launch` binary the README is
//! meant to introduce.
//!
//! Nothing checked it, so it drifted for as long as it took someone to notice.
//! `toolchain_pin.rs` is the precedent: a consistency assertion over files
//! cargo does not compile is worth the twenty lines it costs.

const README: &str = include_str!("../README.md");
const CARGO_TOML: &str = include_str!("../Cargo.toml");

/// Feature names declared in `Cargo.toml`'s `[features]` section.
fn declared_features() -> Vec<String> {
    CARGO_TOML
        .split("[features]")
        .nth(1)
        .expect("Cargo.toml has a [features] section")
        // Stop at the next section header.
        .split("\n[")
        .next()
        .expect("non-empty")
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.starts_with('#') || !line.contains('=') {
                return None;
            }
            Some(line.split('=').next()?.trim().to_string())
        })
        .filter(|name| !name.is_empty())
        .collect()
}

/// The backticked names in the **first column** of the README's feature table.
///
/// First column only: the Description column is full of type names
/// (`AutoVideoSink`, `DeviceMonitor`) that are not features and never will be.
fn documented_names() -> Vec<String> {
    let start = README
        .find("| Feature | Description")
        .expect("README has a feature table");
    let table = &README[start..];
    let end = table.find("\n## ").unwrap_or(table.len());

    let mut out = Vec::new();
    for row in table[..end].lines() {
        let row = row.trim();
        if !row.starts_with('|') {
            continue;
        }
        // `| `name` / `other` | description | deps |` -> the first cell.
        let Some(first) = row.trim_start_matches('|').split('|').next() else {
            continue;
        };
        let mut rest = first;
        while let Some(open) = rest.find('`') {
            rest = &rest[open + 1..];
            let Some(close) = rest.find('`') else { break };
            out.push(rest[..close].to_string());
            rest = &rest[close + 1..];
        }
    }
    out
}

#[test]
fn the_readme_documents_every_feature_that_exists() {
    let documented = documented_names();
    let missing: Vec<String> = declared_features()
        .into_iter()
        .filter(|f| f != "default")
        .filter(|f| !documented.contains(f))
        .collect();

    assert!(
        missing.is_empty(),
        "these features exist but the README's table does not mention them: {missing:#?}"
    );
}

#[test]
fn the_readme_documents_no_feature_that_does_not_exist() {
    let declared = declared_features();
    // The first column holds feature names and nothing else, so anything here
    // that `[features]` does not declare is a stale row.
    let known_non_features: [&str; 0] = [];
    let phantom: Vec<String> = documented_names()
        .into_iter()
        .filter(|n| !declared.contains(n))
        .filter(|n| !known_non_features.contains(&n.as_str()))
        .collect();

    assert!(
        phantom.is_empty(),
        "the README's table documents these, and `[features]` does not declare them: {phantom:#?}"
    );
}
