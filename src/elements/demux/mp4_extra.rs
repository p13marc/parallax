//! Supplemental ISO-BMFF sample-entry scan for codecs the `mp4` crate
//! does not model.
//!
//! mp4-rs 0.14 parses `stsd` by matching a fixed set of sample entries
//! (`avc1`, `hev1`, `vp09`, `mp4a`, `tx3g`) and silently drops everything
//! else — the track survives with working (codec-agnostic) sample tables
//! but reports an unknown codec, and the raw `stsd` bytes are not
//! retained anywhere reachable. This walker re-reads just enough of the
//! `moov` tree from the raw reader *before* `Mp4Reader::read_header` to
//! recover, per track: the first sample-entry fourcc, the Opus `dOps`
//! configuration, and the AV1 `av1C` configuration record.

use crate::error::{Error, Result};
use std::io::{Read, Seek, SeekFrom};

/// What the scan recovered for one track.
#[derive(Debug, Clone, Default)]
pub(crate) struct StsdExtra {
    /// Fourcc of the first sample entry (e.g. `av01`, `Opus`).
    pub fourcc: [u8; 4],
    /// Opus decoder configuration (`dOps` payload), when present.
    pub dops: Option<DopsInfo>,
    /// AV1 configuration record (`av1C` payload), when present.
    pub av1c: Option<Vec<u8>>,
    /// HEVC configuration record (`hvcC` payload), when present.
    pub hvcc: Option<Vec<u8>>,
    /// What the `colr` box said about colour (#265), when present and
    /// in a form this tree models.
    pub color: Option<crate::format::ColorInfo>,
}

/// Sample-entry fourccs that are `VisualSampleEntry`s.
///
/// Needed because the fixed-field run before the child boxes is 70 bytes for
/// video and 20 for audio, so `colr` cannot be looked for without first
/// knowing which kind of entry this is. Listed rather than inferred: a fourcc
/// this scan does not recognise gets no children read at all, which is the
/// safe answer — reading at the wrong offset finds boxes that are not there.
const VISUAL_ENTRIES: [&[u8; 4]; 10] = [
    b"av01", b"avc1", b"avc3", b"hvc1", b"hev1", b"vp08", b"vp09", b"mp4v", b"jpeg", b"raw ",
];

/// Parse a `ColourInformationBox` payload (ISO 14496-12 §12.1.5).
///
/// Handles `nclx` (ISO) and `nclc` (the QuickTime spelling, which is the same
/// three indices without the full-range byte). `rICC`/`prof` carry an ICC
/// profile, which this tree has nowhere to put — `None`, so the consumer's
/// fallback applies rather than a wrong guess.
fn parse_colr(payload: &[u8]) -> Option<crate::format::ColorInfo> {
    use crate::format::{ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, TransferFunction};

    let kind: &[u8; 4] = payload.get(0..4)?.try_into().ok()?;
    if kind != b"nclx" && kind != b"nclc" {
        return None;
    }
    let at = |i: usize| -> Option<u16> {
        Some(u16::from_be_bytes([*payload.get(i)?, *payload.get(i + 1)?]))
    };
    // The three indices are H.273 codes, so they are this tree's discriminants
    // and no translation table is needed — only the narrowing to u8, which is
    // lossless for every code either standard assigns.
    let narrow = |v: u16| u8::try_from(v).unwrap_or(0);
    let info = ColorInfo {
        primaries: ColorPrimaries::from_h273(narrow(at(4)?)),
        transfer: TransferFunction::from_h273(narrow(at(6)?)),
        matrix: ColorMatrix::from_h273(narrow(at(8)?)),
        // `nclc` has no range byte at all, and an `nclx` truncated before it
        // is malformed — either way, unsignalled.
        range: match payload.get(10) {
            Some(b) if kind == b"nclx" => ColorRange::from_code(1 + (b >> 7)),
            _ => ColorRange::Unspecified,
        },
    };
    (!info.is_unspecified()).then_some(info)
}

/// Parsed `dOps` (OpusSpecificBox) fields.
///
/// `pre_skip`/`input_sample_rate` are parsed for the tests and future
/// consumers (pre-skip trimming); the demuxer itself only needs
/// channels + raw today.
#[derive(Debug, Clone)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct DopsInfo {
    /// Output channel count.
    pub channels: u8,
    /// Samples (at 48 kHz) to discard from the start of the stream.
    pub pre_skip: u16,
    /// Original input sample rate (informational; Opus decodes at 48 kHz).
    pub input_sample_rate: u32,
    /// Raw dOps payload, decoder-config style.
    pub raw: Vec<u8>,
}

/// One box header: (fourcc, payload_start, payload_end).
fn read_box_header<R: Read + Seek>(
    r: &mut R,
    pos: u64,
    end: u64,
) -> Result<Option<([u8; 4], u64, u64)>> {
    if pos + 8 > end {
        return Ok(None);
    }
    r.seek(SeekFrom::Start(pos)).map_err(io_err)?;
    let mut head = [0u8; 8];
    r.read_exact(&mut head).map_err(io_err)?;
    let size32 = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as u64;
    let fourcc = [head[4], head[5], head[6], head[7]];
    let (payload_start, box_size) = if size32 == 1 {
        let mut large = [0u8; 8];
        r.read_exact(&mut large).map_err(io_err)?;
        (pos + 16, u64::from_be_bytes(large))
    } else if size32 == 0 {
        (pos + 8, end - pos) // box extends to the end of the enclosure
    } else {
        (pos + 8, size32)
    };
    let box_end = pos.checked_add(box_size).filter(|e| *e <= end);
    match box_end {
        Some(box_end) if payload_start <= box_end => Ok(Some((fourcc, payload_start, box_end))),
        _ => Err(Error::Config("malformed MP4 box size".into())),
    }
}

fn io_err(e: std::io::Error) -> Error {
    Error::Config(format!("MP4 stsd scan: {e}"))
}

/// Find the first child box named `name` between `pos` and `end`.
fn find_child<R: Read + Seek>(
    r: &mut R,
    name: &[u8; 4],
    mut pos: u64,
    end: u64,
) -> Result<Option<(u64, u64)>> {
    while let Some((fourcc, payload, box_end)) = read_box_header(r, pos, end)? {
        if &fourcc == name {
            return Ok(Some((payload, box_end)));
        }
        pos = box_end;
    }
    Ok(None)
}

fn read_bytes<R: Read + Seek>(r: &mut R, pos: u64, len: usize) -> Result<Vec<u8>> {
    r.seek(SeekFrom::Start(pos)).map_err(io_err)?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(io_err)?;
    Ok(buf)
}

/// Scan the file for each track's stsd extras. Rewinds the reader to 0.
///
/// Tolerant by design: a track whose boxes cannot be walked is simply
/// absent from the result (the mp4 crate will report it Unknown, exactly
/// as before this scan existed).
pub(crate) fn scan<R: Read + Seek>(r: &mut R) -> Result<Vec<(u32, StsdExtra)>> {
    let file_end = r.seek(SeekFrom::End(0)).map_err(io_err)?;
    let mut out = Vec::new();

    // Top level: find moov.
    let Some((moov_start, moov_end)) = find_child(r, b"moov", 0, file_end)? else {
        r.seek(SeekFrom::Start(0)).map_err(io_err)?;
        return Ok(out);
    };

    // Every trak inside moov.
    let mut pos = moov_start;
    while let Some((fourcc, payload, box_end)) = read_box_header(r, pos, moov_end)? {
        if &fourcc == b"trak"
            && let Some(entry) = scan_trak(r, payload, box_end)?
        {
            out.push(entry);
        }
        pos = box_end;
    }

    r.seek(SeekFrom::Start(0)).map_err(io_err)?;
    Ok(out)
}

fn scan_trak<R: Read + Seek>(
    r: &mut R,
    trak_start: u64,
    trak_end: u64,
) -> Result<Option<(u32, StsdExtra)>> {
    // tkhd → track_id (fullbox: version decides the offset).
    let Some((tkhd, _)) = find_child(r, b"tkhd", trak_start, trak_end)? else {
        return Ok(None);
    };
    let ver = read_bytes(r, tkhd, 1)?[0];
    let id_off = tkhd + 4 + if ver == 1 { 16 } else { 8 };
    let id_bytes = read_bytes(r, id_off, 4)?;
    let track_id = u32::from_be_bytes([id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3]]);

    // trak → mdia → minf → stbl → stsd.
    let mut cur = (trak_start, trak_end);
    for name in [b"mdia", b"minf", b"stbl", b"stsd"] {
        match find_child(r, name, cur.0, cur.1)? {
            Some(next) => cur = next,
            None => return Ok(None),
        }
    }
    let (stsd, stsd_end) = cur;

    // stsd is a fullbox with an entry count; take the first sample entry.
    let first_entry = stsd + 8;
    let Some((fourcc, entry_payload, entry_end)) = read_box_header(r, first_entry, stsd_end)?
    else {
        return Ok(None);
    };
    let mut extra = StsdExtra {
        fourcc,
        ..Default::default()
    };

    if VISUAL_ENTRIES.contains(&&fourcc) {
        // VisualSampleEntry: 6 reserved + 2 dref + 70 bytes of fixed
        // fields before the child boxes (ISO 14496-12 §12.1.3).
        let children = entry_payload + 6 + 2 + 70;
        match &fourcc {
            b"av01" => {
                if let Some((av1c, av1c_end)) = find_child(r, b"av1C", children, entry_end)? {
                    extra.av1c = Some(read_bytes(r, av1c, (av1c_end - av1c) as usize)?);
                }
            }
            // HEVC, under either sample-entry name: `hvc1` keeps the parameter
            // sets out of band, `hev1` permits them in band as well. Both carry
            // an `hvcC`, and the mp4 crate models neither in a way that exposes
            // it, so the record is recovered here like `av1C` is.
            b"hvc1" | b"hev1" => {
                if let Some((hvcc, hvcc_end)) = find_child(r, b"hvcC", children, entry_end)? {
                    extra.hvcc = Some(read_bytes(r, hvcc, (hvcc_end - hvcc) as usize)?);
                }
            }
            _ => {}
        }
        // `colr` is codec-independent (#265), so it is read for every visual
        // entry — including `avc1`, whose H.264 VUI this tree does not parse,
        // which makes the container the only place the colour can come from.
        if let Some((colr, colr_end)) = find_child(r, b"colr", children, entry_end)? {
            extra.color = parse_colr(&read_bytes(r, colr, (colr_end - colr) as usize)?);
        }
    } else if &fourcc == b"Opus" {
        // AudioSampleEntry: 6 reserved + 2 dref + 20 bytes of fixed
        // fields before the child boxes (§12.2.3).
        {
            let children = entry_payload + 6 + 2 + 20;
            if let Some((dops, dops_end)) = find_child(r, b"dOps", children, entry_end)? {
                let raw = read_bytes(r, dops, (dops_end - dops) as usize)?;
                if raw.len() >= 11 {
                    extra.dops = Some(DopsInfo {
                        channels: raw[1],
                        pre_skip: u16::from_be_bytes([raw[2], raw[3]]),
                        input_sample_rate: u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]),
                        raw,
                    });
                }
            }
        }
    }

    Ok(Some((track_id, extra)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn boxed(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut b = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        b.extend_from_slice(fourcc);
        b.extend_from_slice(payload);
        b
    }

    #[test]
    fn scans_opus_and_av1_entries() {
        // dOps payload: version 0, 2 ch, pre-skip 312, 48000 Hz, gain 0, family 0.
        let dops = boxed(
            b"dOps",
            &[0, 2, 0x01, 0x38, 0x00, 0x00, 0xBB, 0x80, 0, 0, 0],
        );
        let mut audio_entry_payload = vec![0u8; 6 + 2 + 20];
        audio_entry_payload.extend_from_slice(&dops);
        let opus_entry = boxed(b"Opus", &audio_entry_payload);

        let mut stsd_payload = vec![0, 0, 0, 0, 0, 0, 0, 1]; // fullbox + count
        stsd_payload.extend_from_slice(&opus_entry);
        let stsd = boxed(b"stsd", &stsd_payload);
        let stbl = boxed(b"stbl", &stsd);
        let minf = boxed(b"minf", &stbl);
        let mdia = boxed(b"mdia", &minf);
        let mut tkhd_payload = vec![0u8; 4 + 8]; // v0 fullbox + ctime/mtime
        tkhd_payload.extend_from_slice(&7u32.to_be_bytes()); // track_id 7
        tkhd_payload.extend_from_slice(&[0u8; 8]);
        let tkhd = boxed(b"tkhd", &tkhd_payload);
        let mut trak_payload = tkhd;
        trak_payload.extend_from_slice(&mdia);
        let trak = boxed(b"trak", &trak_payload);
        let moov = boxed(b"moov", &trak);
        let mut file = boxed(b"ftyp", b"isom");
        file.extend_from_slice(&moov);

        let scanned = scan(&mut Cursor::new(file)).unwrap();
        assert_eq!(scanned.len(), 1);
        let (id, extra) = &scanned[0];
        assert_eq!(*id, 7);
        assert_eq!(&extra.fourcc, b"Opus");
        let dops = extra.dops.as_ref().unwrap();
        assert_eq!(dops.channels, 2);
        assert_eq!(dops.pre_skip, 312);
        assert_eq!(dops.input_sample_rate, 48000);
    }

    /// A one-track file whose single visual sample entry has these children.
    fn visual_file(fourcc: &[u8; 4], children: &[u8]) -> Vec<u8> {
        let mut entry_payload = vec![0u8; 6 + 2 + 70];
        entry_payload.extend_from_slice(children);
        let entry = boxed(fourcc, &entry_payload);

        let mut stsd_payload = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd_payload.extend_from_slice(&entry);
        let stsd = boxed(b"stsd", &stsd_payload);
        let mdia = boxed(b"mdia", &boxed(b"minf", &boxed(b"stbl", &stsd)));

        let mut tkhd_payload = vec![0u8; 4 + 8];
        tkhd_payload.extend_from_slice(&1u32.to_be_bytes());
        tkhd_payload.extend_from_slice(&[0u8; 8]);
        let mut trak_payload = boxed(b"tkhd", &tkhd_payload);
        trak_payload.extend_from_slice(&mdia);
        let moov = boxed(b"moov", &boxed(b"trak", &trak_payload));

        let mut file = boxed(b"ftyp", b"isom");
        file.extend_from_slice(&moov);
        file
    }

    /// `nclx` payload: primaries, transfer, matrix, then the full-range flag
    /// in the top bit of a byte.
    fn nclx(primaries: u16, transfer: u16, matrix: u16, full: bool) -> Vec<u8> {
        let mut p = b"nclx".to_vec();
        p.extend_from_slice(&primaries.to_be_bytes());
        p.extend_from_slice(&transfer.to_be_bytes());
        p.extend_from_slice(&matrix.to_be_bytes());
        p.push(if full { 0x80 } else { 0x00 });
        p
    }

    #[test]
    fn colr_is_read_for_any_visual_entry() {
        use crate::format::{ColorMatrix, ColorPrimaries, ColorRange, TransferFunction};

        // BT.709 across the board, limited range — and under `avc1`, the
        // entry whose codec configuration this scan does not otherwise
        // touch. H.264's VUI is not parsed anywhere in this tree, so the
        // container is the only place its colour can come from (#265).
        let colr = boxed(b"colr", &nclx(1, 1, 1, false));
        let scanned = scan(&mut Cursor::new(visual_file(b"avc1", &colr))).unwrap();
        let color = scanned[0].1.color.expect("colr was read");
        assert_eq!(color.primaries, ColorPrimaries::Bt709);
        assert_eq!(color.transfer, TransferFunction::Bt709);
        assert_eq!(color.matrix, ColorMatrix::Bt709);
        assert_eq!(color.range, ColorRange::Limited);

        // The full-range bit is the one that changes the picture most
        // visibly, and it is a single bit in a byte most files leave at zero.
        let colr = boxed(b"colr", &nclx(5, 6, 6, true));
        let scanned = scan(&mut Cursor::new(visual_file(b"vp09", &colr))).unwrap();
        let color = scanned[0].1.color.unwrap();
        assert_eq!(color.matrix, ColorMatrix::Bt601);
        assert_eq!(color.range, ColorRange::Full);
    }

    #[test]
    fn an_unreadable_colr_leaves_the_colour_unsignalled() {
        // `nclc` is QuickTime's spelling: the same three indices, no range
        // byte. It must be read, and must not invent a range.
        let mut nclc = b"nclc".to_vec();
        nclc.extend_from_slice(&[0, 1, 0, 1, 0, 1]);
        let scanned = scan(&mut Cursor::new(visual_file(
            b"avc1",
            &boxed(b"colr", &nclc),
        )))
        .unwrap();
        let color = scanned[0].1.color.unwrap();
        assert_eq!(color.matrix, crate::format::ColorMatrix::Bt709);
        assert_eq!(color.range, crate::format::ColorRange::Unspecified);

        // An ICC profile has nowhere to go in this tree — better unsignalled
        // than guessed at.
        let scanned = scan(&mut Cursor::new(visual_file(
            b"avc1",
            &boxed(b"colr", b"prof\x00\x01\x02\x03"),
        )))
        .unwrap();
        assert!(scanned[0].1.color.is_none());

        // And an entry with no `colr` at all says nothing, rather than
        // reading whatever bytes happen to follow.
        let scanned = scan(&mut Cursor::new(visual_file(b"avc1", &[]))).unwrap();
        assert!(scanned[0].1.color.is_none());
    }

    #[test]
    fn missing_moov_yields_empty() {
        let mut file = boxed(b"ftyp", b"isom");
        file.extend_from_slice(&boxed(b"mdat", &[0u8; 16]));
        assert!(scan(&mut Cursor::new(file)).unwrap().is_empty());
    }
}
