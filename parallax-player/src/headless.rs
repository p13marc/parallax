//! Playing with no window at all.
//!
//! The pipeline is the one the window drives, and this module is the host:
//! it asks the clock for running time, takes the frames that are due, and
//! comes back. That is the same policy [`parallax_iced::VideoView`] runs —
//! [`FrameMailbox::take_due`] with the same lateness budget — with a 5 ms
//! poll standing in for the compositor's vertical blank.
//!
//! What it deliberately does *not* exercise is the shader, the upload and the
//! swapchain. `just play-headless` runs the real widget under weston and
//! remains the end-to-end proof; this is `just play-smoke`, which needs no
//! display server, no GPU and no sound card, so CI can run it.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use parallax::clock::ClockTime;
use parallax::elements::EndReason;
use parallax_iced::DEFAULT_MAX_LATENESS;

use crate::engine::{self, Engine};
use crate::{Args, stats_line};

/// How often the null host looks for a frame.
///
/// A real host is woken by the compositor; this one polls, which is the same
/// shape with a worse clock. Deliberately not the mailbox doorbell: it is rung
/// with `notify_waiters`, which leaves no permit, so a waiter that registers
/// after the last ring parks forever on a stream that has already ended. The
/// sink handles that with a registered-then-checked `Notified`; five
/// milliseconds of polling handles it with nothing.
const POLL: Duration = Duration::from_millis(5);

/// Exit code for a run that ended in a pipeline error.
const EXIT_ERROR: i32 = 1;
/// Exit code for a run the user interrupted, as the terminal player used.
const EXIT_INTERRUPTED: i32 = 130;

/// Play `args.file` with no window, and return the process exit code.
pub fn run(args: Args) -> Result<i32> {
    // Multi-threaded, not current-thread: the drain loop *is* the main future,
    // and the element tasks `Executor::start` spawns need workers of their own.
    let runtime = tokio::runtime::Runtime::new().context("starting the async runtime")?;
    runtime.block_on(play(args))
}

async fn play(args: Args) -> Result<i32> {
    let path = args
        .file
        .clone()
        .context("--headless needs a file to play")?;
    let options = args.options();

    let probed = tokio::task::spawn_blocking(move || engine::probe(&path))
        .await
        .context("the probe task panicked")??;
    let engine = Engine::build(probed, &options)?;
    tracing::info!("headless: {}", engine.source.summary());

    // The deadline is a hard stop in both hosts, so their numbers compare:
    // the windowed player prints and exits at `--exit-after` with the
    // pipeline still running, and `scripts/decode_matrix.py` reads exactly
    // those numbers across configurations. A natural end-of-stream drains in
    // both.
    let deadline = args
        .exit_after
        .map(|s| Instant::now() + Duration::from_secs(s));
    let mut interrupted = false;

    let end = loop {
        // `clock()` is `None` until the first buffer — an element only sees
        // the pipeline clock through its consume context. `ClockTime::NONE`
        // makes `schedule` answer `Present`, which is "draw on arrival", and
        // is also what `--no-sync` looks like for the whole run.
        let now = engine
            .mailbox
            .clock()
            .map(|clock| clock.now())
            .unwrap_or(ClockTime::NONE);

        // Drain rather than take one: frames between the last shown and now
        // are already superseded, and showing them in sequence would play the
        // video fast.
        while engine.mailbox.take_due(now, DEFAULT_MAX_LATENESS).is_some() {}

        if let Some(reason) = engine.mailbox.end_reason()
            && !engine.mailbox.has_frame()
        {
            break Some(reason);
        }
        if deadline.is_some_and(|at| Instant::now() >= at) {
            break None;
        }

        tokio::select! {
            _ = tokio::time::sleep(POLL) => {}
            // An interrupted measurement should still yield its line.
            _ = tokio::signal::ctrl_c() => {
                interrupted = true;
                break None;
            }
        }
    };

    println!(
        "{}",
        stats_line(engine.position(), engine.dropped(), &engine.mailbox.stats())
    );
    engine.teardown().await;

    Ok(exit_code(end.as_ref(), interrupted))
}

/// What the process exits with, given how the run ended.
///
/// A pipeline error is a failed run even though frames were presented; an
/// interrupt is neither success nor failure and says so with the shell's own
/// number. Reaching `--exit-after` is a success: bounding the run is what it
/// was asked to do.
fn exit_code(end: Option<&EndReason>, interrupted: bool) -> i32 {
    if interrupted {
        return EXIT_INTERRUPTED;
    }
    match end {
        Some(EndReason::Error(_)) => EXIT_ERROR,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parallax::pipeline::StreamError;

    #[test]
    fn only_a_pipeline_error_makes_the_run_a_failure() {
        assert_eq!(exit_code(Some(&EndReason::Eos), false), 0);
        // The deadline was reached: bounding the run is not a failure.
        assert_eq!(exit_code(None, false), 0);
        assert_eq!(exit_code(Some(&EndReason::Aborted), false), 0);
        assert_eq!(
            exit_code(
                Some(&EndReason::Error(StreamError::new("decode", "boom"))),
                false
            ),
            EXIT_ERROR
        );
        // An interrupt outranks whatever the pipeline was doing.
        assert_eq!(exit_code(Some(&EndReason::Eos), true), EXIT_INTERRUPTED);
    }
}
