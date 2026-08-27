//! What is drawn over the video.
//!
//! The overlay is **unmounted** when hidden, not merely transparent. Every
//! redraw runs Iced's full `view()` + layout + text shaping, and a video
//! player redraws at the frame rate — so a playing-fullscreen tree should be
//! one `Shader` widget and nothing else.

use iced::widget::{Space, button, column, container, row, scrollable, slider, text};
use iced::{Alignment, Element, Length};

use parallax::clock::ClockTime;

use crate::{Level, Message, Notice, Player};

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

    for (index, notice) in player.notices.iter().enumerate() {
        stack = stack.push(toast(index, notice));
    }
    if player.show_stats {
        stack = stack.push(stats(player));
    }
    if player.show_playlist {
        stack = stack.push(playlist(player));
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

/// One thing the pipeline said, with a way to make it go away.
///
/// A warning fades on its own; an error waits to be dismissed, because the
/// run is over and a message nobody read is the same as no message.
fn toast(index: usize, notice: &Notice) -> Element<'_, Message> {
    let level = notice.level;
    let body = row![
        text(notice.text.clone()).size(14).width(Length::Fill),
        button(text("×").size(14))
            .on_press(Message::Dismiss(index))
            .padding([0, 6]),
    ]
    .spacing(8)
    .align_y(Alignment::Center);

    container(body)
        .padding(10)
        .width(Length::Fill)
        .style(move |theme: &iced::Theme| {
            let palette = theme.palette();
            let background = match level {
                Level::Error => palette.danger,
                Level::Warning => palette.warning,
                Level::Info => palette.primary,
            };
            container::background(background).color(iced::Color::WHITE)
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

    // Frame stepping only when the container said how long a frame is; a
    // guess would step 33 ms and call it a frame (#281).
    let steppable = player
        .source()
        .and_then(|s| s.video.as_ref())
        .and_then(|v| v.frame_duration())
        .is_some();

    let mut transport = row![
        button(text(if player.paused { "Play" } else { "Pause" }).size(13))
            .on_press(Message::TogglePause)
            .padding([4, 10]),
        text(timecode(ClockTime::from_nanos(shown))).size(13),
        scrubber,
        text(timecode(duration)).size(13),
    ]
    .spacing(10)
    .align_y(Alignment::Center);

    if steppable {
        transport = transport
            .push(
                button(text("◀|").size(12))
                    .on_press(Message::StepBy(-1))
                    .padding([4, 6]),
            )
            .push(
                button(text("|▶").size(12))
                    .on_press(Message::StepBy(1))
                    .padding([4, 6]),
            );
    }

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
            button(text("Queue").size(12))
                .on_press(Message::TogglePlaylist)
                .padding([4, 8]),
        )
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

/// What an empty window shows.
///
/// Not a slogan: the two things a player with nothing open can offer are a
/// way in and the files it opened last time.
pub fn welcome(player: &Player) -> Element<'_, Message> {
    let mut body = column![
        text("parallax").size(28),
        button(text("Open… (o)").size(14))
            .on_press(Message::OpenDialog)
            .padding([6, 14]),
    ]
    .spacing(14)
    .align_x(Alignment::Center);

    if !player.recent.is_empty() {
        let mut list = column![text("recently played").size(12)].spacing(4);
        for path in &player.recent {
            let label = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            list = list.push(
                button(text(label).size(13))
                    .on_press(Message::Enqueue(vec![path.clone()]))
                    .padding([3, 8]),
            );
        }
        body = body.push(list.align_x(Alignment::Center));
    }

    body.into()
}

/// What is queued. Toggled with `p`, or the button.
///
/// Scrollable and capped in height: a playlist is allowed to be long, and the
/// video is what the window is for.
fn playlist(player: &Player) -> Element<'_, Message> {
    let mut rows = column![].spacing(2);
    for (index, entry) in player.playlist.entries().iter().enumerate() {
        let current = player.playlist.current() == Some(index);
        let mut label = entry.label();
        if entry.failed {
            // Marked rather than removed: a file the user put in the list is
            // still something they chose, and silently dropping it looks like
            // the drop never happened.
            label.push_str("  (could not be opened)");
        }
        rows = rows.push(
            row![
                text(if current { "▶" } else { " " }).size(12),
                button(text(label).size(12))
                    .on_press(Message::PlayIndex(index))
                    .padding([3, 6])
                    .width(Length::Fill),
                button(text("×").size(12))
                    .on_press(Message::Remove(index))
                    .padding([3, 6]),
            ]
            .spacing(6)
            .align_y(Alignment::Center),
        );
    }

    let body: Element<'_, Message> = if player.playlist.is_empty() {
        text("nothing queued — drop files on the window, or pass one on the command line")
            .size(12)
            .into()
    } else {
        scrollable(rows).height(Length::Fixed(180.0)).into()
    };

    let controls = row![
        button(text("Previous (b)").size(11))
            .on_press(Message::Previous)
            .padding([3, 8]),
        button(text("Next (n)").size(11))
            .on_press(Message::Next)
            .padding([3, 8]),
        button(text(player.playlist.repeat.label()).size(11))
            .on_press(Message::CycleRepeat)
            .padding([3, 8]),
    ]
    .spacing(6);

    panel(column![body, controls].spacing(8))
        .width(Length::Fill)
        .into()
}

/// What the pipeline is actually doing. Toggled with `i`, or the button.
///
/// Everything here is read from something that already existed — the mailbox
/// counters, the bus, the pipeline handle — and most of it used to be thrown
/// away in a `_ => {}` arm. It is unmounted when hidden, like the rest of the
/// overlay, so a playing-fullscreen tree stays one `Shader` widget.
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
        if frames.worst_late_ns > 0 {
            lines = lines.push(
                text(format!(
                    "worst lateness: {:.1} ms",
                    frames.worst_late_ns as f64 / 1e6
                ))
                .size(12),
            );
        }
        // Which decoder actually won, which is the question `--hwdec auto`
        // leaves open.
        lines = lines.push(text(format!("decode: {}", engine.decoder)).size(12));
    }

    let diagnostics = &player.diagnostics;
    // Presentation is stated rather than queried: `wgpu::Device` cannot name
    // its own adapter, and the custom-primitive path is handed nothing else,
    // so Iced's own answer is the only honest one — and the upload-vs-import
    // half is a fact about this build, not about the machine (#269).
    let path = match &diagnostics.graphics {
        Some((adapter, backend)) => format!("present: upload · {backend} · {adapter}"),
        None => "present: upload".into(),
    };
    lines = lines.push(text(path).size(12));

    if let Some(clock) = &diagnostics.clock {
        lines = lines.push(text(format!("clock: {clock}")).size(12));
    }
    if let Some(percent) = diagnostics.buffering {
        lines = lines.push(text(format!("buffering: {percent}%")).size(12));
    }
    if let Some((have, total)) = diagnostics.downloaded {
        let total = match total {
            Some(total) => format!(" of {:.1} MB", total as f64 / 1e6),
            None => String::new(),
        };
        lines =
            lines.push(text(format!("downloaded: {:.1} MB{total}", have as f64 / 1e6)).size(12));
    }
    if let Some((proportion, jitter_ns)) = diagnostics.qos {
        // 1.0 is keeping up; 2.0 means only half the frames were presentable
        // in time. Saying so beats printing a bare number.
        lines = lines.push(
            text(format!(
                "qos: {proportion:.2}× required ({:.1} ms late)",
                jitter_ns as f64 / 1e6
            ))
            .size(12),
        );
    }

    lines = lines.push(
        button(text("Write graph (d)").size(11))
            .on_press(Message::DumpGraph)
            .padding([3, 8]),
    );

    panel(lines).into()
}
