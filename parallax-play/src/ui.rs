//! What is drawn over the video.
//!
//! The overlay is **unmounted** when hidden, not merely transparent. Every
//! redraw runs Iced's full `view()` + layout + text shaping, and a video
//! player redraws at the frame rate — so a playing-fullscreen tree should be
//! one `Shader` widget and nothing else.

use iced::widget::{Space, button, column, container, row, slider, text};
use iced::{Alignment, Element, Length};

use parallax::clock::ClockTime;

use crate::{Message, Player};

/// `mm:ss`, or `--:--` when the container never said how long the file is.
///
/// An unknown duration is not a zero one: clamping a seek against zero sends
/// every one to the start, which is what an MKV with no Segment Duration used
/// to do.
pub fn timecode(t: ClockTime) -> String {
    match t.to_option() {
        Some(t) => {
            let secs = t.nanos() / 1_000_000_000;
            format!("{:02}:{:02}", secs / 60, secs % 60)
        }
        None => "--:--".into(),
    }
}

/// Everything drawn over the video.
pub fn overlay(player: &Player) -> Element<'_, Message> {
    let mut stack = column![].spacing(8);

    if let Some(error) = &player.error {
        stack = stack.push(banner(error));
    }
    if player.show_stats {
        stack = stack.push(stats(player));
    }

    // Push the controls to the bottom.
    stack = stack.push(Space::new().width(Length::Fill).height(Length::Fill));

    if player.controls_visible() {
        stack = stack.push(controls(player));
    }

    container(stack)
        .padding(12)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn banner(message: &str) -> Element<'_, Message> {
    container(text(message.to_string()).size(14))
        .padding(10)
        .width(Length::Fill)
        .style(|theme: &iced::Theme| {
            container::background(theme.palette().danger).color(iced::Color::WHITE)
        })
        .into()
}

fn panel<'a>(content: impl Into<Element<'a, Message>>) -> container::Container<'a, Message> {
    container(content).padding(12).style(|_: &iced::Theme| {
        container::background(iced::Color::from_rgba(0.0, 0.0, 0.0, 0.72))
            .border(iced::border::rounded(8))
    })
}

/// The transport: position, a seek bar, volume, and the two toggles.
fn controls(player: &Player) -> Element<'_, Message> {
    let position = player.position.to_option().map(|p| p.nanos()).unwrap_or(0);
    let duration = player.duration();

    // While a seek is in flight the bar shows where the transport is
    // *heading*, not where it is: `position()` does not move until the seek's
    // Segment lands, and a bar that snaps back under the user's finger is
    // worse than one that leads.
    let shown = player.seek.target().unwrap_or(position);

    let scrubber: Element<'_, Message> = match duration.to_option() {
        Some(total) if total.nanos() > 0 => {
            let fraction = (shown as f64 / total.nanos() as f64).clamp(0.0, 1.0) as f32;
            slider(0.0..=1.0f32, fraction, Message::SeekTo)
                .step(0.0005f32)
                .width(Length::Fill)
                .into()
        }
        // No duration means the container never said — a live stream, or an
        // MKV without one. There is nothing to scrub against, so say so
        // rather than draw a bar that cannot mean anything.
        _ => container(text("live / unknown length").size(12))
            .width(Length::Fill)
            .into(),
    };

    let transport = row![
        button(text(if player.paused { "Play" } else { "Pause" }).size(13))
            .on_press(Message::TogglePause)
            .padding([4, 10]),
        text(timecode(ClockTime::from_nanos(shown))).size(13),
        scrubber,
        text(timecode(duration)).size(13),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    let mut tools = row![].spacing(8).align_y(Alignment::Center);
    if player.has_audio() {
        tools = tools
            .push(
                button(text(if player.muted { "Unmute" } else { "Mute" }).size(12))
                    .on_press(Message::ToggleMute)
                    .padding([4, 8]),
            )
            .push(
                slider(-40.0..=6.0f32, player.volume_db, Message::VolumeTo)
                    .step(1.0f32)
                    .width(110),
            )
            .push(text(format!("{:+.0} dB", player.volume_db)).size(12));
    }
    tools = tools
        .push(Space::new().width(Length::Fill))
        .push(speed(player))
        .push(
            button(text("Stats").size(12))
                .on_press(Message::ToggleStats)
                .padding([4, 8]),
        )
        .push(
            button(
                text(if player.fullscreen {
                    "Windowed"
                } else {
                    "Fullscreen"
                })
                .size(12),
            )
            .on_press(Message::ToggleFullscreen)
            .padding([4, 8]),
        );

    panel(column![transport, tools].spacing(8))
        .width(Length::Fill)
        .into()
}

/// The playback speed (#280), as one button per rate.
///
/// The current one is labelled with a dot rather than styled: a `Button`
/// style closure allocates per redraw, and this widget is rebuilt at the
/// frame rate.
///
/// Audio is muted away from 1x — `AlsaSink` plays at the hardware rate and
/// this tree has no pitch-preserving time-stretch — so the label says so
/// rather than leaving the user to wonder whether the sound broke.
fn speed(player: &Player) -> Element<'_, Message> {
    let mut row = row![].spacing(4).align_y(Alignment::Center);
    if player.speed != 1.0 && player.has_audio() {
        row = row.push(text("muted").size(11));
    }
    for rate in crate::SPEEDS {
        let current = (player.speed - rate).abs() < 1e-6;
        let label = if current {
            format!("• {rate}x")
        } else {
            format!("{rate}x")
        };
        row = row.push(
            button(text(label).size(12))
                .on_press(Message::SpeedTo(rate))
                .padding([4, 6]),
        );
    }
    row.into()
}

/// What the pipeline is actually doing. Toggled with `i`, or the button.
fn stats(player: &Player) -> Element<'_, Message> {
    let mut lines = column![text(player.status.clone()).size(13)].spacing(2);

    if let Some(source) = player.source()
        && source.subtitles > 0
    {
        // Listed rather than rendered — #279 is where they start being drawn,
        // and saying so beats appearing not to have noticed.
        lines = lines.push(
            text(format!(
                "{} subtitle track(s) — not rendered yet",
                source.subtitles
            ))
            .size(12),
        );
    }

    if let Some(engine) = player.engine() {
        let frames = engine.mailbox.stats();
        lines = lines
            .push(
                text(format!(
                    "presented {} · queued {}/{} · dropped {} late, {} stale, {} host-too-slow",
                    frames.presented,
                    frames.queued,
                    engine.mailbox.capacity(),
                    frames.dropped_late,
                    frames.dropped_stale,
                    frames.dropped_full,
                ))
                .size(12),
            )
            .push(
                text(format!(
                    "pipeline dropped {} · clipped {} · {}",
                    engine.dropped(),
                    frames.clipped,
                    match engine.mailbox.end_reason() {
                        Some(reason) => format!("ended: {reason:?}"),
                        None => "playing".into(),
                    }
                ))
                .size(12),
            );
        if let Some(latency) = engine.handle().latency() {
            lines = lines.push(text(format!("declared latency: {:?}", latency.max)).size(12));
        }
    }

    panel(lines).into()
}
