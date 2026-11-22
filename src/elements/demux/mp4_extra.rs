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
    /// Pixel aspect ratio from the `pasp` box, and rotation from the `tkhd`
    /// display matrix (#266). `None` when neither said anything.
    pub display: Option<crate::format::DisplayGeometry>,
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

/// Read the rotation out of a `tkhd` display matrix.
///
/// The matrix is nine 32-bit fixed-point values `{a b u; c d v; x y w}` at
/// +40 (version 0) or +52 (version 1) from the start of the box payload —
/// fullbox 4, times 8 or 16, track_ID 4, reserved 4, duration 4 or 8,
/// reserved 8, then layer/alternate_group/volume/reserved 8.
///
/// Only the four right angles are recognised, by the sign pattern of
/// `a`/`b`/`c`/`d`; an arbitrary affine (a shear, a flip, a non-right
/// rotation) is refused rather than approximated, because approximating it
/// would silently show the wrong picture where refusing shows the stored one.
///
/// `a` and `d` are 16.16 fixed point, `b` and `c` are too; the standard
/// rotations use only 0 and ±1.0 (`0x0001_0000`), so exact comparison is
/// right here and a tolerance would only admit transforms this cannot express.
fn rotation_from_matrix(m: &[u8]) -> Option<crate::format::Rotation> {
    use crate::format::Rotation;
    const ONE: i32 = 0x0001_0000;

    let at = |i: usize| -> Option<i32> {
        Some(i32::from_be_bytes(
            m.get(i * 4..i * 4 + 4)?.try_into().ok()?,
        ))
    };
    let quad = (at(0)?, at(1)?, at(3)?, at(4)?);
    // Written as comparisons rather than patterns because `-ONE` is not one:
    // a negated constant cannot appear in a pattern position.
    for (want, rotation) in [
        ((ONE, 0, 0, ONE), Rotation::None),
        ((0, ONE, -ONE, 0), Rotation::Cw90),
        ((-ONE, 0, 0, -ONE), Rotation::Cw180),
        ((0, -ONE, ONE, 0), Rotation::Cw270),
    ] {
        if quad == want {
            return Some(rotation);
        }
    }
    None
}

/// Parse a `PixelAspectRatioBox` payload: two 32-bit integers.
///
/// A zero in either half is refused — `hSpacing:vSpacing` is a ratio, and a
/// zero would make every downstream aspect calculation a division by nothing.
fn parse_pasp(payload: &[u8]) -> Option<(u32, u32)> {
    let at = |i: usize| -> Option<u32> {
        Some(u32::from_be_bytes(
            payload.get(i * 4..i * 4 + 4)?.try_into().ok()?,
        ))
    };
    match (at(0)?, at(1)?) {
        (0, _) | (_, 0) => None,
        pair => Some(pair),
    }
}

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

/// A `trak`'s first sample entry, as `(fourcc, payload_start, end)`.
///
/// trak → mdia → minf → stbl → stsd, then past the stsd fullbox header and
/// its entry count. The first entry is the only one mp4-rs reads.
fn first_sample_entry<R: Read + Seek>(
    r: &mut R,
    trak_start: u64,
    trak_end: u64,
) -> Result<Option<([u8; 4], u64, u64)>> {
    let mut cur = (trak_start, trak_end);
    for name in [b"mdia", b"minf", b"stbl", b"stsd"] {
        match find_child(r, name, cur.0, cur.1)? {
            Some(next) => cur = next,
            None => return Ok(None),
        }
    }
    let (stsd, stsd_end) = cur;
    read_box_header(r, stsd + 8, stsd_end)
}

/// The configuration box mp4-rs needs as the **first** child of each sample
/// entry it parses, and the size of the entry's fixed fields before its
/// children (#305).
///
/// `Avc1Box`, `Hev1Box`, `Vp09Box` and `Mp4aBox` each read exactly one child
/// after their fixed fields and take it for the configuration. `avc1` and
/// `hev1` then error ("avcc not found"), `vp09` parses whatever came first as
/// a `vpcC`, and `mp4a` drops its `esds` without a word. ISO/IEC 14496-12
/// does not fix the order of a sample entry's children, and some muxers put
/// `pasp` or `colr` first — so a whole file failed `Mp4Reader::read_header`
/// over the position of one box.
const CONFIG_FIRST: [(&[u8; 4], &[u8; 4], u64); 4] = [
    (b"avc1", b"avcC", 6 + 2 + 70),
    (b"hev1", b"hvcC", 6 + 2 + 70),
    (b"vp09", b"vpcC", 6 + 2 + 70),
    (b"mp4a", b"esds", 6 + 2 + 20),
];

/// Same-length patches that move each track's configuration box to the
/// front of its first sample entry (`CONFIG_FIRST`), as `(offset, bytes)`
/// for a [`PatchedReader`]. Rewinds the reader to 0.
///
/// Reordering a box's children leaves its size alone, so nothing else in the
/// file moves: not the sizes up the tree, not the chunk offsets into `mdat`.
///
/// Tolerant **per track**: a `trak` this walk cannot read contributes no
/// patch and costs the others nothing. The caller already treats a failed
/// scan as "no patches", so letting one truncated audio `trak` fail the
/// whole scan would take the video track's patch down with it and bring
/// back the very error the patch exists to remove — over a track the crate
/// itself might have read without complaint.
pub(crate) fn config_first_patches<R: Read + Seek>(r: &mut R) -> Result<Vec<(u64, Vec<u8>)>> {
    let file_end = r.seek(SeekFrom::End(0)).map_err(io_err)?;
    let mut patches = Vec::new();

    if let Some((moov_start, moov_end)) = find_child(r, b"moov", 0, file_end)? {
        let mut pos = moov_start;
        // A box header that does not parse ends the walk — there is no
        // finding the next `trak` past it — but keeps what came before.
        while let Ok(Some((fourcc, payload, box_end))) = read_box_header(r, pos, moov_end) {
            if &fourcc == b"trak"
                && let Ok(Some(patch)) = config_first_patch(r, payload, box_end)
            {
                patches.push(patch);
            }
            pos = box_end;
        }
    }

    r.seek(SeekFrom::Start(0)).map_err(io_err)?;
    Ok(patches)
}

/// The patch for one `trak`, if its first sample entry needs one.
///
/// Conservative: the entry is touched only when its children parse as boxes
/// that run to its end (bar fewer than eight bytes of padding, which stay
/// where they are) and the configuration box is among them but not first.
/// Anything this walk cannot read the way the crate would is left exactly as
/// it is, so the worst case is the old failure.
fn config_first_patch<R: Read + Seek>(
    r: &mut R,
    trak_start: u64,
    trak_end: u64,
) -> Result<Option<(u64, Vec<u8>)>> {
    let Some((fourcc, payload, entry_end)) = first_sample_entry(r, trak_start, trak_end)? else {
        return Ok(None);
    };
    let Some(&(_, config, fixed)) = CONFIG_FIRST.iter().find(|(entry, ..)| **entry == fourcc)
    else {
        return Ok(None);
    };
    // An `mp4a` has 20 bytes of fixed fields only in version 0. QuickTime's
    // versions 1 and 2 add more, which the crate does not read either.
    if &fourcc == b"mp4a" && read_bytes(r, payload + 8, 2)? != [0u8; 2] {
        return Ok(None);
    }

    let children = payload + fixed;
    let mut boxes = Vec::new();
    let mut pos = children;
    loop {
        match read_box_header(r, pos, entry_end) {
            Ok(Some((name, _, end))) => {
                boxes.push((name, pos, end));
                pos = end;
            }
            Ok(None) => break,
            Err(_) => return Ok(None),
        }
    }
    let Some(first) = boxes.iter().position(|(name, ..)| name == config) else {
        return Ok(None);
    };
    if first == 0 {
        return Ok(None);
    }

    let original = read_bytes(r, children, (pos - children) as usize)?;
    let slice = |&(_, start, end): &([u8; 4], u64, u64)| {
        &original[(start - children) as usize..(end - children) as usize]
    };
    let mut reordered = slice(&boxes[first]).to_vec();
    for (i, child) in boxes.iter().enumerate() {
        if i != first {
            reordered.extend_from_slice(slice(child));
        }
    }
    Ok(Some((children, reordered)))
}

/// `inner`, read with some byte ranges replaced (#305).
///
/// Every patch is a same-length replacement (see [`config_first_patches`]),
/// so offsets and lengths are those of the file underneath. The crate reads
/// a patched range differently and every other byte as it is.
#[derive(Debug)]
pub(crate) struct PatchedReader<R> {
    inner: R,
    pos: u64,
    patches: Vec<(u64, Vec<u8>)>,
}

impl<R: Seek> PatchedReader<R> {
    pub(crate) fn new(mut inner: R, patches: Vec<(u64, Vec<u8>)>) -> std::io::Result<Self> {
        let pos = inner.stream_position()?;
        Ok(Self {
            inner,
            pos,
            patches,
        })
    }
}

impl<R: Read> Read for PatchedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        let (start, end) = (self.pos, self.pos + n as u64);
        for (at, bytes) in &self.patches {
            let from = start.max(*at);
            let to = end.min(at + bytes.len() as u64);
            if from < to {
                buf[(from - start) as usize..(to - start) as usize]
                    .copy_from_slice(&bytes[(from - at) as usize..(to - at) as usize]);
            }
        }
        self.pos = end;
        Ok(n)
    }
}

impl<R: Seek> Seek for PatchedReader<R> {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        self.pos = self.inner.seek(to)?;
        Ok(self.pos)
    }
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

    // The display matrix (#266). Version 1 widens creation/modification time
    // and duration to 64 bits, which moves everything after them by 12 bytes.
    // mp4-rs reads this box only for the track id, which is how a portrait
    // phone video played sideways.
    let matrix_off = tkhd + if ver == 1 { 52 } else { 40 };
    let rotation = read_bytes(r, matrix_off, 36)
        .ok()
        .as_deref()
        .and_then(rotation_from_matrix);

    let Some((fourcc, entry_payload, entry_end)) = first_sample_entry(r, trak_start, trak_end)?
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
        // `pasp` is codec-independent too (#266). Non-square pixels are why
        // anamorphic DVD rips come out squished: 720x480 stored, 16:9 shown.
        let par = match find_child(r, b"pasp", children, entry_end)? {
            Some((pasp, pasp_end)) => parse_pasp(&read_bytes(r, pasp, (pasp_end - pasp) as usize)?),
            None => None,
        };
        // One `Some` when either half said something. Square pixels and no
        // rotation is the default a consumer would apply anyway, so stamping
        // it would only cost a wire field per buffer.
        if par.is_some() || rotation.is_some_and(|r| r != crate::format::Rotation::None) {
            extra.display = Some(crate::format::DisplayGeometry {
                par: par.unwrap_or((1, 1)),
                rotation: rotation.unwrap_or_default(),
            });
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
        visual_file_with_matrix(fourcc, children, &identity_matrix())
    }

    /// The unity display matrix: `{1 0 0; 0 1 0; 0 0 1}` in 16.16 / 2.30.
    fn identity_matrix() -> Vec<u8> {
        matrix_for(0x0001_0000, 0, 0, 0x0001_0000)
    }

    /// A display matrix with these four rotation cells; the rest is unity.
    fn matrix_for(a: i32, b: i32, c: i32, d: i32) -> Vec<u8> {
        let mut m = Vec::new();
        for v in [a, b, 0, c, d, 0, 0, 0, 0x4000_0000i32] {
            m.extend_from_slice(&v.to_be_bytes());
        }
        m
    }

    /// [`visual_file`] with an explicit `tkhd` display matrix.
    fn visual_file_with_matrix(fourcc: &[u8; 4], children: &[u8], matrix: &[u8]) -> Vec<u8> {
        let mut entry_payload = vec![0u8; 6 + 2 + 70];
        entry_payload.extend_from_slice(children);
        file_with_entry(&boxed(fourcc, &entry_payload), matrix)
    }

    /// A one-track file whose `stsd` holds this one sample entry.
    fn file_with_entry(entry: &[u8], matrix: &[u8]) -> Vec<u8> {
        let mut stsd_payload = vec![0, 0, 0, 0, 0, 0, 0, 1];
        stsd_payload.extend_from_slice(entry);
        let stsd = boxed(b"stsd", &stsd_payload);
        let mdia = boxed(b"mdia", &boxed(b"minf", &boxed(b"stbl", &stsd)));

        // v0 tkhd: fullbox(4) + ctime/mtime(8) + track_id(4) + reserved(4)
        // + duration(4) + reserved(8) + layer/alt/vol/reserved(8) puts the
        // matrix at +40 from the payload start.
        let mut tkhd_payload = vec![0u8; 4 + 8];
        tkhd_payload.extend_from_slice(&1u32.to_be_bytes());
        tkhd_payload.extend_from_slice(&[0u8; 4 + 4 + 8 + 8]);
        assert_eq!(tkhd_payload.len(), 40, "the matrix goes at +40");
        tkhd_payload.extend_from_slice(matrix);
        tkhd_payload.extend_from_slice(&[0u8; 8]); // width/height
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
    fn pasp_and_the_display_matrix_become_display_geometry() {
        use crate::format::Rotation;

        // Anamorphic NTSC: 32:27 pixels, upright.
        let pasp = boxed(b"pasp", &[0, 0, 0, 32, 0, 0, 0, 27]);
        let scanned = scan(&mut Cursor::new(visual_file(b"avc1", &pasp))).unwrap();
        let display = scanned[0].1.display.expect("pasp was read");
        assert_eq!(display.par, (32, 27));
        assert_eq!(display.rotation, Rotation::None);

        // Every right angle, by the sign pattern the standard rotations use.
        const ONE: i32 = 0x0001_0000;
        for (matrix, want) in [
            (matrix_for(ONE, 0, 0, ONE), Rotation::None),
            (matrix_for(0, ONE, -ONE, 0), Rotation::Cw90),
            (matrix_for(-ONE, 0, 0, -ONE), Rotation::Cw180),
            (matrix_for(0, -ONE, ONE, 0), Rotation::Cw270),
        ] {
            let file = visual_file_with_matrix(b"avc1", &[], &matrix);
            let got = scan(&mut Cursor::new(file)).unwrap()[0].1.display;
            match want {
                // Upright and square is the default a consumer applies
                // anyway, so nothing is stamped for it.
                Rotation::None => assert!(got.is_none(), "unity matrix says nothing"),
                _ => assert_eq!(got.unwrap().rotation, want, "{matrix:?}"),
            }
        }
    }

    #[test]
    fn a_matrix_that_is_not_a_right_angle_is_refused() {
        use crate::format::Rotation;

        // A horizontal flip: expressible in the box, not in `Rotation`, and
        // approximating it would silently mirror the picture.
        const ONE: i32 = 0x0001_0000;
        let flipped = visual_file_with_matrix(b"avc1", &[], &matrix_for(-ONE, 0, 0, ONE));
        assert!(
            scan(&mut Cursor::new(flipped)).unwrap()[0]
                .1
                .display
                .is_none()
        );

        // A 45-degree rotation: cos/sin in 16.16, and neither cell matches.
        let skewed =
            visual_file_with_matrix(b"avc1", &[], &matrix_for(46341, 46341, -46341, 46341));
        assert!(
            scan(&mut Cursor::new(skewed)).unwrap()[0]
                .1
                .display
                .is_none()
        );

        // A zero in the aspect ratio is a division waiting to happen. The
        // rotation still comes through, because they are separate boxes.
        let zero = boxed(b"pasp", &[0, 0, 0, 0, 0, 0, 0, 27]);
        let file = visual_file_with_matrix(b"avc1", &zero, &matrix_for(0, ONE, -ONE, 0));
        let display = scan(&mut Cursor::new(file)).unwrap()[0].1.display.unwrap();
        assert_eq!(display.par, (1, 1), "a zero ratio is refused, not used");
        assert_eq!(display.rotation, Rotation::Cw90);
    }

    #[test]
    fn missing_moov_yields_empty() {
        let mut file = boxed(b"ftyp", b"isom");
        file.extend_from_slice(&boxed(b"mdat", &[0u8; 16]));
        assert!(scan(&mut Cursor::new(file)).unwrap().is_empty());
    }

    /// Where the box with this fourcc starts: four bytes before the name.
    fn box_at(file: &[u8], fourcc: &[u8; 4]) -> u64 {
        (file.windows(4).position(|w| w == fourcc).unwrap() - 4) as u64
    }

    fn pasp_1_1() -> Vec<u8> {
        boxed(b"pasp", &[0, 0, 0, 1, 0, 0, 0, 1])
    }

    fn avcc() -> Vec<u8> {
        boxed(b"avcC", &[1, 0x64, 0, 0x1f, 0xff, 0xe0, 0])
    }

    fn patches_of(file: &[u8]) -> Vec<(u64, Vec<u8>)> {
        config_first_patches(&mut Cursor::new(file)).unwrap()
    }

    /// #305: the `avc1` children the failing files carry, in their order.
    #[test]
    fn a_config_box_behind_pasp_is_moved_to_the_front() {
        let btrt = boxed(b"btrt", &[0u8; 12]);
        let children = [pasp_1_1(), avcc(), btrt.clone()].concat();
        let file = visual_file(b"avc1", &children);

        let patches = patches_of(&file);
        assert_eq!(patches.len(), 1);
        let (at, bytes) = &patches[0];
        assert_eq!(*at, box_at(&file, b"pasp"), "from the first child on");
        assert_eq!(bytes, &[avcc(), pasp_1_1(), btrt].concat());
    }

    #[test]
    fn an_entry_already_in_order_is_left_alone() {
        let file = visual_file(b"avc1", &[avcc(), pasp_1_1()].concat());
        assert!(patches_of(&file).is_empty());
        // No configuration box at all: nothing to move.
        assert!(patches_of(&visual_file(b"avc1", &pasp_1_1())).is_empty());
    }

    #[test]
    fn an_entry_whose_children_do_not_parse_is_left_alone() {
        // An avcC that claims more bytes than the entry holds.
        let mut liar = avcc();
        liar[..4].copy_from_slice(&500u32.to_be_bytes());
        let file = visual_file(b"avc1", &[pasp_1_1(), liar].concat());
        assert!(patches_of(&file).is_empty());
    }

    /// One track's damage is that track's problem. A `trak` whose `mdia`
    /// holds a box claiming more bytes than its enclosure makes the walk
    /// error — and that error used to fail the scan for every track, so the
    /// caller's `unwrap_or_default` threw away the `avc1` patch with it.
    #[test]
    fn a_trak_that_cannot_be_walked_does_not_cost_the_others_their_patch() {
        let good = visual_file(b"avc1", &[pasp_1_1(), avcc()].concat());
        let moov = box_at(&good, b"moov") as usize;
        let good_trak = &good[moov + 8..];

        let mut liar = boxed(b"minf", &[]);
        liar[..4].copy_from_slice(&256u32.to_be_bytes());
        let bad_trak = boxed(b"trak", &boxed(b"mdia", &liar));

        // The unreadable track first, so the walk has to go on past it; then
        // last, so a later failure cannot take back an earlier patch.
        for traks in [
            [bad_trak.as_slice(), good_trak].concat(),
            [good_trak, bad_trak.as_slice()].concat(),
        ] {
            let mut file = boxed(b"ftyp", b"isom");
            file.extend_from_slice(&boxed(b"moov", &traks));

            let patches = patches_of(&file);
            assert_eq!(patches.len(), 1);
            assert_eq!(patches[0].0, box_at(&file, b"pasp"));
            assert_eq!(patches[0].1, [avcc(), pasp_1_1()].concat());
        }
    }

    #[test]
    fn only_the_entries_the_crate_parses_are_touched() {
        // `av01` is read by this module's own scan, which finds `av1C`
        // wherever it is.
        let colr = boxed(b"colr", &nclx(1, 1, 1, false));
        let file = visual_file(b"av01", &[colr, boxed(b"av1C", &[0x81, 0, 0, 0])].concat());
        assert!(patches_of(&file).is_empty());
    }

    /// An `mp4a` entry of this QuickTime sound version with these children.
    fn mp4a_entry(version: u16, children: &[u8]) -> Vec<u8> {
        let mut payload = vec![0u8; 6 + 2];
        payload.extend_from_slice(&version.to_be_bytes());
        payload.extend_from_slice(&[0u8; 18]);
        payload.extend_from_slice(children);
        boxed(b"mp4a", &payload)
    }

    #[test]
    fn an_mp4a_esds_is_moved_only_in_the_version_0_layout() {
        let children = [boxed(b"btrt", &[0u8; 12]), boxed(b"esds", &[0u8; 8])].concat();
        let v0 = file_with_entry(&mp4a_entry(0, &children), &identity_matrix());
        assert_eq!(patches_of(&v0).len(), 1);
        // Version 1 has 16 more bytes of fixed fields: read at the version-0
        // offset, its children would be boxes that are not there.
        let v1 = file_with_entry(&mp4a_entry(1, &children), &identity_matrix());
        assert!(patches_of(&v1).is_empty());
    }

    #[test]
    fn a_patched_reader_overlays_its_ranges_and_nothing_else() {
        let data: Vec<u8> = (0..32).collect();
        let patches = vec![(10, vec![0xAA; 4])];
        let mut r = PatchedReader::new(Cursor::new(data.clone()), patches).unwrap();

        let mut whole = Vec::new();
        r.read_to_end(&mut whole).unwrap();
        let mut want = data;
        want[10..14].fill(0xAA);
        assert_eq!(whole, want);

        // A read that starts inside the patch, after a seek.
        r.seek(SeekFrom::Start(12)).unwrap();
        let mut four = [0u8; 4];
        r.read_exact(&mut four).unwrap();
        assert_eq!(four, [0xAA, 0xAA, 14, 15]);

        // One byte at a time, across both edges of the patch.
        r.seek(SeekFrom::Start(8)).unwrap();
        let mut bytes = Vec::new();
        for _ in 0..8 {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).unwrap();
            bytes.push(b[0]);
        }
        assert_eq!(bytes, [8, 9, 0xAA, 0xAA, 0xAA, 0xAA, 14, 15]);
    }
}
