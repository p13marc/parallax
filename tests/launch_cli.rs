//! End-to-end tests for the `parallax-launch` binary (feature `cli`).

use std::process::Command;

fn launch() -> Command {
    Command::new(env!("CARGO_BIN_EXE_parallax-launch"))
}

#[test]
fn runs_a_pipeline_to_eos() {
    let out = launch()
        .args([
            "-q",
            "videotestsrc",
            "num-buffers=25",
            "!",
            "videoconvert",
            "!",
            "nullsink",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn pipeline_as_single_quoted_arg_works() {
    let out = launch()
        .args(["-q", "videotestsrc num-buffers=5 ! nullsink"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn unknown_element_exits_1_with_message() {
    let out = launch()
        .args(["nosuchelement", "!", "nullsink"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown element"), "{stderr}");
}

#[test]
fn unknown_property_exits_1_with_message() {
    let out = launch()
        .args(["videotestsrc", "fps=30", "!", "nullsink"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("fps") && stderr.contains("videotestsrc"),
        "{stderr}"
    );
}

#[cfg(not(feature = "alsa"))]
#[test]
fn gated_element_error_names_the_feature() {
    let out = launch()
        .args(["nullsource", "!", "alsasink"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("alsa") && stderr.contains("feature"),
        "{stderr}"
    );
}

#[test]
fn list_elements_prints_registered_names() {
    let out = launch().arg("--list-elements").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("videotestsrc"), "{stdout}");
    assert!(stdout.contains("nullsink"), "{stdout}");
}

#[test]
fn dot_prints_graph_without_running() {
    let out = launch()
        .args(["--dot", "nullsource", "!", "nullsink"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("digraph"), "{stdout}");
}

#[test]
fn no_pipeline_is_a_usage_error() {
    let out = launch().output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn missing_required_property_exits_1() {
    let out = launch()
        .args(["filesrc", "!", "nullsink"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("location"), "{stderr}");
}
