//! What is drawn over the video.

use iced::widget::{column, container, text};
use iced::{Element, Length};

use parallax::clock::ClockTime;

use crate::{Message, Player};

/// `mm:ss`, or `--:--` when the container never said how long the file is.
pub fn timecode(t: ClockTime) -> String {
    match t.to_option() {
        Some(t) => {
            let secs = t.nanos() / 1_000_000_000;
            format!("{:02}:{:02}", secs / 60, secs % 60)
        }
        None => "--:--".into(),
    }
}

/// The status line, and an error when there is one.
pub fn overlay(player: &Player) -> Element<'_, Message> {
    let mut lines = column![].spacing(4);

    if let Some(error) = &player.error {
        lines = lines.push(
            text(error.clone())
                .size(14)
                .color(iced::Color::from_rgb(1.0, 0.45, 0.4)),
        );
    }

    let position = timecode(player.position);
    let duration = timecode(player.duration());
    let heading = player
        .seek
        .target()
        .map(|t| format!(" → {}", timecode(ClockTime::from_nanos(t))))
        .unwrap_or_default();
    let paused = if player.paused { " ⏸" } else { "" };

    // Subtitle tracks are listed rather than rendered — #279 is where they
    // start being drawn, and saying so beats appearing not to have noticed.
    let subtitles = match player.source().map(|s| s.subtitles).unwrap_or(0) {
        0 => String::new(),
        n => format!(" · {n} subtitle track(s), not rendered"),
    };

    lines = lines.push(
        text(format!(
            "{position} / {duration}{heading}{paused} · {}{subtitles}",
            player.status
        ))
        .size(14),
    );

    container(lines)
        .padding(12)
        .width(Length::Fill)
        .height(Length::Fill)
        .align_y(iced::alignment::Vertical::Bottom)
        .into()
}
