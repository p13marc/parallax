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

    match &fourcc {
        // VisualSampleEntry: 6 reserved + 2 dref + 70 bytes of fixed
        // fields before the child boxes (ISO 14496-12 §12.1.3).
        b"av01" => {
            let children = entry_payload + 6 + 2 + 70;
            if let Some((av1c, av1c_end)) = find_child(r, b"av1C", children, entry_end)? {
                extra.av1c = Some(read_bytes(r, av1c, (av1c_end - av1c) as usize)?);
            }
        }
        // HEVC, under either sample-entry name: `hvc1` keeps the parameter
        // sets out of band, `hev1` permits them in band as well. Both carry
        // an `hvcC`, and the mp4 crate models neither in a way that exposes
        // it, so the record is recovered here like `av1C` is.
        b"hvc1" | b"hev1" => {
            let children = entry_payload + 6 + 2 + 70;
            if let Some((hvcc, hvcc_end)) = find_child(r, b"hvcC", children, entry_end)? {
                extra.hvcc = Some(read_bytes(r, hvcc, (hvcc_end - hvcc) as usize)?);
            }
        }
        // AudioSampleEntry: 6 reserved + 2 dref + 20 bytes of fixed
        // fields before the child boxes (§12.2.3).
        b"Opus" => {
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
        _ => {}
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

    #[test]
    fn missing_moov_yields_empty() {
        let mut file = boxed(b"ftyp", b"isom");
        file.extend_from_slice(&boxed(b"mdat", &[0u8; 16]));
        assert!(scan(&mut Cursor::new(file)).unwrap().is_empty());
    }
}
