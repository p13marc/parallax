//! Reading a SubRip file from beside the video.
//!
//! This is the *only* place SRT is parsed, and it exists for sidecar files.
//! Matroska never stores SRT as SRT: a muxer converts it to `S_TEXT/UTF8`
//! with the timing in the container, so a subtitle track inside an MKV needs
//! no parsing at all — the block payload *is* the cue text.

use crate::clock::ClockTime;
use crate::error::{Error, Result};

use super::Cue;

/// Parse a SubRip document into cues.
///
/// Tolerant on purpose: subtitles are downloaded from anywhere, and a file
/// with a malformed block in the middle should lose that block rather than
/// the film's dialogue. Blocks that do not parse are skipped, and the caller
/// gets whatever did.
///
/// An empty result is an error, though — a file that produced no cues at all
/// is not an SRT file, and silently showing nothing looks like a bug in the
/// player.
pub fn parse(text: &str) -> Result<Vec<Cue>> {
    // A BOM at the front is common and would make the first index unparsable.
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut cues = Vec::new();
    let mut skipped = 0usize;

    for block in text.split("\n\n").flat_map(|b| b.split("\r\n\r\n")) {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        match parse_block(block) {
            Some(cue) => cues.push(cue),
            None => skipped += 1,
        }
    }

    if cues.is_empty() {
        return Err(Error::Parse(
            "no subtitle cues found — is this a SubRip file?".into(),
        ));
    }
    if skipped > 0 {
        tracing::warn!("skipped {skipped} malformed subtitle block(s)");
    }
    Ok(cues)
}

/// One `index / timing / text...` block.
fn parse_block(block: &str) -> Option<Cue> {
    let mut lines = block.lines();
    let first = lines.next()?.trim();
    // The index line is optional in the wild — some tools omit it — so if the
    // first line is already a timing line, use it.
    let timing = if first.contains("-->") {
        first
    } else {
        lines.next()?.trim()
    };
    let (start, end) = parse_timing(timing)?;

    let text: Vec<&str> = lines.map(str::trim_end).collect();
    let text = text.join("\n").trim().to_string();
    if text.is_empty() {
        return None;
    }

    let duration = ClockTime::from_nanos(end.nanos().saturating_sub(start.nanos()));
    Some(Cue::new(text, start, duration))
}

/// `00:00:01,000 --> 00:00:04,000`, with the fractional separator either way.
fn parse_timing(line: &str) -> Option<(ClockTime, ClockTime)> {
    let (start, rest) = line.split_once("-->")?;
    // Anything after the end stamp is a position setting (WebVTT-style), and
    // parallax does not place text by file directive.
    let end = rest.split_whitespace().next()?;
    Some((parse_stamp(start.trim())?, parse_stamp(end)?))
}

/// `HH:MM:SS,mmm` — or with a `.`, or with the hours left off.
fn parse_stamp(stamp: &str) -> Option<ClockTime> {
    let (clock, fraction) = match stamp.rsplit_once([',', '.']) {
        Some((clock, fraction)) => (clock, fraction),
        None => (stamp, "0"),
    };
    // Milliseconds, but tolerate two or four digits rather than refusing.
    let millis: u64 = fraction.parse().ok()?;
    let millis = match fraction.len() {
        0 => 0,
        1 => millis * 100,
        2 => millis * 10,
        3 => millis,
        _ => millis / 10u64.pow(fraction.len() as u32 - 3),
    };

    let mut seconds = 0u64;
    for part in clock.split(':') {
        seconds = seconds
            .checked_mul(60)?
            .checked_add(part.trim().parse().ok()?)?;
    }
    Some(ClockTime::from_nanos(
        seconds
            .checked_mul(1_000_000_000)?
            .checked_add(millis * 1_000_000)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "1\n\
        00:00:01,000 --> 00:00:04,000\n\
        Hello there\n\
        \n\
        2\n\
        00:00:05,500 --> 00:00:07,250\n\
        Two lines\n\
        of dialogue\n";

    #[test]
    fn a_subrip_file_becomes_cues() {
        let cues = parse(SAMPLE).expect("it parses");
        assert_eq!(cues.len(), 2);

        assert_eq!(cues[0].text, "Hello there");
        assert_eq!(cues[0].start, ClockTime::from_secs(1));
        assert_eq!(cues[0].duration, ClockTime::from_secs(3));

        assert_eq!(cues[1].text, "Two lines\nof dialogue");
        assert_eq!(cues[1].start, ClockTime::from_millis(5_500));
        assert_eq!(cues[1].duration, ClockTime::from_millis(1_750));
    }

    #[test]
    fn the_shapes_real_files_come_in() {
        // A byte-order mark, a dot instead of a comma, no index line, and
        // trailing position settings — all seen in the wild.
        let odd = "\u{feff}00:00:02.000 --> 00:00:03.000 X1:100 X2:200\nOdd but fine\n";
        let cues = parse(odd).expect("it parses");
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "Odd but fine");
        assert_eq!(cues[0].start, ClockTime::from_secs(2));

        // Hours omitted, which some tools do for short clips.
        let short = "1\n01:30,000 --> 01:31,000\nlate\n";
        let cues = parse(short).expect("it parses");
        assert_eq!(cues[0].start, ClockTime::from_secs(90));
    }

    #[test]
    fn one_bad_block_does_not_lose_the_film() {
        let mangled = "1\nnot a timing line\nlost\n\n\
                       2\n00:00:05,000 --> 00:00:06,000\nkept\n";
        let cues = parse(mangled).expect("the good block survives");
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "kept");
    }

    #[test]
    fn a_file_that_is_not_subrip_is_an_error() {
        // Showing nothing and reporting success looks like a player bug.
        assert!(parse("").is_err());
        assert!(parse("<html><body>404</body></html>").is_err());
    }
}
