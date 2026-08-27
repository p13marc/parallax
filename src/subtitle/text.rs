//! Turning what a container hands over into text a host can draw.

/// The plain text of an ASS/SSA event body.
///
/// Matroska stores an ASS event as the `Dialogue:` line with its prefix and
/// its two timestamps removed — the timing became the block's own — leaving
/// `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`. Eight
/// commas, and everything after the eighth is the text (which may itself
/// contain commas, so this counts rather than splits).
///
/// Override tags in `{...}` are dropped and `\N` / `\n` become real line
/// breaks. That is a *degradation*, and a deliberate one: rendering ASS
/// positioning, karaoke and per-glyph animation is a project of its own, and
/// showing the dialogue in the player's own style beats showing
/// `{\pos(190,270)}` to the viewer.
pub fn ass_dialogue(body: &str) -> String {
    // Nine fields means eight separators; a line with fewer is malformed, and
    // showing it raw is more useful than showing nothing.
    let text = match body.match_indices(',').nth(8 - 1) {
        Some((at, _)) => &body[at + 1..],
        None => body,
    };
    strip_ass_markup(text)
}

/// Remove `{...}` override blocks and convert ASS line breaks.
fn strip_ass_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut depth = 0usize;
    while let Some(c) = chars.next() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth > 0 => {}
            // `\N` is a hard break, `\n` a soft one; a player that wraps
            // anyway can treat both as a break. `\h` is a non-breaking space.
            '\\' => match chars.peek() {
                Some('N') | Some('n') => {
                    chars.next();
                    out.push('\n');
                }
                Some('h') => {
                    chars.next();
                    out.push('\u{a0}');
                }
                _ => out.push('\\'),
            },
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}

/// The plain text of a WebVTT cue payload.
///
/// Only the inline tags, because a Matroska WebVTT block is the cue payload
/// and nothing else — settings and identifiers live in `CodecPrivate` and
/// `BlockAdditions`, which parallax does not route.
pub fn webvtt_payload(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut in_tag = false;
    for c in body.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if in_tag => {}
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ass_event_body_is_its_ninth_field_onward() {
        assert_eq!(
            ass_dialogue("0,0,Default,,0,0,0,,Hello there"),
            "Hello there"
        );
        // The text may contain commas of its own; the split counts, it does
        // not partition.
        assert_eq!(
            ass_dialogue("1,0,Default,Bob,0,0,0,,Well, yes, quite"),
            "Well, yes, quite"
        );
    }

    #[test]
    fn override_tags_are_dropped_and_breaks_are_kept() {
        assert_eq!(
            ass_dialogue(r"0,0,Default,,0,0,0,,{\pos(190,270)}Up here"),
            "Up here"
        );
        assert_eq!(
            ass_dialogue(r"0,0,Default,,0,0,0,,First line\NSecond line"),
            "First line\nSecond line"
        );
        assert_eq!(
            ass_dialogue(r"0,0,Default,,0,0,0,,soft\nbreak"),
            "soft\nbreak"
        );
        // A nested or unbalanced brace must not eat the rest of the line
        // forever — a malformed file should lose styling, not dialogue.
        assert_eq!(
            ass_dialogue(r"0,0,Default,,0,0,0,,{\i1}italic{\i0} plain"),
            "italic plain"
        );
        // A lone backslash is text.
        assert_eq!(ass_dialogue(r"0,0,Default,,0,0,0,,a\b"), r"a\b");
    }

    #[test]
    fn a_malformed_event_shows_its_text_rather_than_nothing() {
        // Fewer than eight commas: there is no field to take, so show it all.
        assert_eq!(ass_dialogue("just some text"), "just some text");
    }

    #[test]
    fn webvtt_inline_tags_are_dropped() {
        assert_eq!(webvtt_payload("<v Roger>Hello <b>there</b>"), "Hello there");
        assert_eq!(webvtt_payload("plain"), "plain");
    }
}
