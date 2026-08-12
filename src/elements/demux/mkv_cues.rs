//! Parallax-owned Matroska cue index + shared reader for [`MkvDemux`].
//!
//! matroska-demuxer 0.8 mis-resolves `CueRelativePosition` against the
//! *first* cluster in its seek broad phase (ffmpeg-muxed files carry that
//! element), so its cue seek either errors or silently lands wrong. The old
//! fallback — rewind to byte 0 and skip below the target — re-parsed the
//! whole file per seek. Instead, this module reads the `Cues` element
//! itself with a small EBML walker *before* the crate ever sees the reader
//! (the same sidestep `mp4_extra.rs` performs on the mp4 crate), and
//! [`SharedReader`] keeps a second handle to the reader so `MkvDemux` can
//! reposition it at a cue's cluster directly.
//!
//! `CueRelativePosition` is deliberately ignored: landing mid-cluster would
//! skip the cluster's own Timestamp element and leave the parser's notion
//! of time stale, while landing at the cluster element start costs at most
//! one cluster of intra-cluster skipping — which the seek's per-track skip
//! performs anyway.
//!
//! [`MkvDemux`]: super::MkvDemux

use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex, PoisonError};

// Element IDs, raw (marker bit kept), per the Matroska spec.
const EBML_HEADER: u64 = 0x1A45_DFA3;
const SEGMENT: u64 = 0x1853_8067;
const SEEK_HEAD: u64 = 0x114D_9B74;
const SEEK: u64 = 0x4DBB;
const SEEK_ID: u64 = 0x53AB;
const SEEK_POSITION: u64 = 0x53AC;
const CUES: u64 = 0x1C53_BB6B;
const CUE_POINT: u64 = 0xBB;
const CUE_TIME: u64 = 0xB3;
const CUE_TRACK_POSITIONS: u64 = 0xB7;
const CUE_CLUSTER_POSITION: u64 = 0xF1;
const CLUSTER: u64 = 0x1F43_B675;
const VOID: u64 = 0xEC;
const CRC32: u64 = 0xBF;

/// One cue: a stream time and the absolute file offset of the Cluster
/// element that contains it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CueEntry {
    /// Cue time in Matroska ticks (the same unit as block timestamps).
    pub time_ticks: u64,
    /// Absolute offset of the target Cluster *element* (ID byte), i.e.
    /// `segment data offset + CueClusterPosition`.
    pub cluster_offset: u64,
}

/// Read the file's cue table, tolerating anything.
///
/// Infallible by policy: any structural surprise returns the entries found
/// so far (usually none), and the seek path falls back to the crate. The
/// reader is rewound to the start before returning, whatever happened.
pub(crate) fn scan_cues<R: Read + Seek>(r: &mut R) -> Vec<CueEntry> {
    let entries = scan_cues_inner(r).unwrap_or_default();
    let _ = r.seek(SeekFrom::Start(0));
    entries
}

fn scan_cues_inner<R: Read + Seek>(r: &mut R) -> Option<Vec<CueEntry>> {
    let file_len = r.seek(SeekFrom::End(0)).ok()?;
    r.seek(SeekFrom::Start(0)).ok()?;

    // EBML header, skipped whole.
    let (id, size) = read_element_header(r)?;
    if id != EBML_HEADER {
        return None;
    }
    skip(r, size?)?;

    // Segment: everything below is offset-relative to its data start.
    let (id, size) = read_element_header(r)?;
    if id != SEGMENT {
        return None;
    }
    // An unknown segment size (live streams) still has a defined data
    // offset; the walk below is bounded by hitting the first Cluster.
    let _ = size;
    let segment_data_offset = pos(r)?;

    // Walk the segment's top level. A SeekHead may point straight at the
    // Cues (ffmpeg writes them at the end of the file); reaching a Cluster
    // without either means scanning the whole file — give up instead.
    let mut cues_offset: Option<u64> = None;
    loop {
        let at = pos(r)?;
        if at >= file_len {
            return None;
        }
        let (id, size) = read_element_header(r)?;
        match id {
            SEEK_HEAD => {
                if let Some(offset) = parse_seek_head(r, size?, segment_data_offset)? {
                    cues_offset = Some(offset);
                }
            }
            CUES => {
                return parse_cues(r, size?, segment_data_offset, file_len);
            }
            CLUSTER => match cues_offset {
                Some(offset) if offset < file_len => {
                    r.seek(SeekFrom::Start(offset)).ok()?;
                    let (id, size) = read_element_header(r)?;
                    if id != CUES {
                        return None;
                    }
                    return parse_cues(r, size?, segment_data_offset, file_len);
                }
                _ => return None,
            },
            _ => skip(r, size?)?,
        }
    }
}

/// SeekHead → the absolute offset of the Cues element, if indexed.
fn parse_seek_head<R: Read + Seek>(
    r: &mut R,
    size: u64,
    segment_data_offset: u64,
) -> Option<Option<u64>> {
    let end = pos(r)? + size;
    let mut cues = None;
    while pos(r)? < end {
        let (id, sz) = read_element_header(r)?;
        let sz = sz?;
        match id {
            SEEK => {
                let seek_end = pos(r)? + sz;
                let mut target_id = 0u64;
                let mut position = None;
                while pos(r)? < seek_end {
                    let (id, sz) = read_element_header(r)?;
                    let sz = sz?;
                    match id {
                        SEEK_ID => target_id = read_uint(r, sz)?,
                        SEEK_POSITION => position = Some(read_uint(r, sz)?),
                        _ => skip(r, sz)?,
                    }
                }
                if target_id == CUES
                    && let Some(p) = position
                {
                    cues = Some(segment_data_offset + p);
                }
            }
            _ => skip(r, sz)?,
        }
    }
    Some(cues)
}

/// Cues → sorted, bounds-checked cue table.
fn parse_cues<R: Read + Seek>(
    r: &mut R,
    size: u64,
    segment_data_offset: u64,
    file_len: u64,
) -> Option<Vec<CueEntry>> {
    let end = pos(r)? + size;
    let mut entries = Vec::new();
    while pos(r)? < end {
        let (id, sz) = read_element_header(r)?;
        let sz = sz?;
        if id != CUE_POINT {
            skip(r, sz)?;
            continue;
        }
        let point_end = pos(r)? + sz;
        let mut time = None;
        // Several CueTrackPositions may share a point; the smallest cluster
        // position covers every track from one landing.
        let mut cluster_pos: Option<u64> = None;
        while pos(r)? < point_end {
            let (id, sz) = read_element_header(r)?;
            let sz = sz?;
            match id {
                CUE_TIME => time = Some(read_uint(r, sz)?),
                CUE_TRACK_POSITIONS => {
                    let tp_end = pos(r)? + sz;
                    while pos(r)? < tp_end {
                        let (id, sz) = read_element_header(r)?;
                        let sz = sz?;
                        match id {
                            CUE_CLUSTER_POSITION => {
                                let p = read_uint(r, sz)?;
                                cluster_pos = Some(cluster_pos.map_or(p, |c: u64| c.min(p)));
                            }
                            _ => skip(r, sz)?,
                        }
                    }
                }
                _ => skip(r, sz)?,
            }
        }
        if let (Some(time_ticks), Some(p)) = (time, cluster_pos) {
            let cluster_offset = segment_data_offset + p;
            if cluster_offset < file_len {
                entries.push(CueEntry {
                    time_ticks,
                    cluster_offset,
                });
            }
        }
    }
    entries.sort_by_key(|e| e.time_ticks);
    Some(entries)
}

/// One element header: raw ID (marker kept) and size (`None` = unknown).
fn read_element_header<R: Read>(r: &mut R) -> Option<(u64, Option<u64>)> {
    let id = read_vint(r, false)?;
    let size = read_vint_size(r)?;
    Some((id, size))
}

/// EBML variable-length integer. `strip_marker` removes the length-marker
/// bit (sizes); IDs keep it (that is how they are written in the spec).
fn read_vint<R: Read>(r: &mut R, strip_marker: bool) -> Option<u64> {
    let mut first = [0u8; 1];
    r.read_exact(&mut first).ok()?;
    let byte = first[0];
    if byte == 0 {
        return None;
    }
    let extra = byte.leading_zeros() as usize;
    if extra > 7 {
        return None;
    }
    let mut value = if strip_marker {
        u64::from(byte & (0x7F >> extra))
    } else {
        u64::from(byte)
    };
    for _ in 0..extra {
        let mut next = [0u8; 1];
        r.read_exact(&mut next).ok()?;
        value = (value << 8) | u64::from(next[0]);
    }
    Some(value)
}

/// A size vint; all-ones payload means "unknown size" (`None`).
fn read_vint_size<R: Read>(r: &mut R) -> Option<Option<u64>> {
    let mut first = [0u8; 1];
    r.read_exact(&mut first).ok()?;
    let byte = first[0];
    if byte == 0 {
        return None;
    }
    let extra = byte.leading_zeros() as usize;
    if extra > 7 {
        return None;
    }
    let mut value = u64::from(byte & (0x7F >> extra));
    let mut all_ones = value == u64::from(0x7Fu8 >> extra);
    for _ in 0..extra {
        let mut next = [0u8; 1];
        r.read_exact(&mut next).ok()?;
        all_ones &= next[0] == 0xFF;
        value = (value << 8) | u64::from(next[0]);
    }
    Some(if all_ones { None } else { Some(value) })
}

/// Big-endian unsigned integer payload (0–8 bytes).
fn read_uint<R: Read>(r: &mut R, size: u64) -> Option<u64> {
    if size > 8 {
        return None;
    }
    let mut value = 0u64;
    for _ in 0..size {
        let mut b = [0u8; 1];
        r.read_exact(&mut b).ok()?;
        value = (value << 8) | u64::from(b[0]);
    }
    Some(value)
}

fn skip<R: Seek>(r: &mut R, size: u64) -> Option<()> {
    r.seek(SeekFrom::Current(size as i64)).ok()?;
    Some(())
}

fn pos<R: Seek>(r: &mut R) -> Option<u64> {
    r.stream_position().ok()
}

/// A cloneable `Read + Seek` handle over one underlying reader.
///
/// One clone goes into `MatroskaFile::open`, the other stays on `MkvDemux`
/// so a cue seek can reposition the stream directly. Both handles live on
/// the single demuxer task, so the mutex is never contended — it exists to
/// make the type `Send` and the sharing sound.
#[derive(Debug)]
pub struct SharedReader<R>(Arc<Mutex<R>>);

impl<R> SharedReader<R> {
    pub(crate) fn new(inner: R) -> Self {
        Self(Arc::new(Mutex::new(inner)))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, R> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<R> Clone for SharedReader<R> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<R: Read> Read for SharedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.lock().read(buf)
    }
}

impl<R: Seek> Seek for SharedReader<R> {
    fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
        self.lock().seek(from)
    }
}

// The 0xEC/0xBF constants document the IDs the walker tolerates by falling
// into the generic skip arm; referenced here so they stay named.
const _: [u64; 2] = [VOID, CRC32];

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const H264_AAC_MKV: &[u8] = include_bytes!("../../../tests/fixtures/tiny_h264_aac.mkv");
    const VP9_OPUS_WEBM: &[u8] = include_bytes!("../../../tests/fixtures/tiny_vp9_opus.webm");
    const VP8_VORBIS_WEBM: &[u8] = include_bytes!("../../../tests/fixtures/tiny_vp8_vorbis.webm");

    fn cues(data: &[u8]) -> Vec<CueEntry> {
        scan_cues(&mut Cursor::new(data.to_vec()))
    }

    #[test]
    fn h264_fixture_cue_table_is_exact() {
        // ffmpeg wrote 4 cue points over 2 clusters; the offsets are the
        // Cluster element starts (verified against a raw EBML dump).
        let entries = cues(H264_AAC_MKV);
        assert_eq!(entries.len(), 4, "{entries:?}");
        let times: Vec<u64> = entries.iter().map(|e| e.time_ticks).collect();
        assert_eq!(times, [0, 500, 1000, 1500]);
        assert_eq!(entries[0].cluster_offset, entries[1].cluster_offset);
        assert_eq!(entries[2].cluster_offset, entries[3].cluster_offset);
        assert!(entries[1].cluster_offset < entries[2].cluster_offset);
    }

    #[test]
    fn webm_fixtures_have_cues() {
        for (name, data) in [("vp9", VP9_OPUS_WEBM), ("vp8", VP8_VORBIS_WEBM)] {
            let entries = cues(data);
            assert!(!entries.is_empty(), "{name}: no cues found");
            assert_eq!(entries[0].time_ticks, 0, "{name}: first cue at 0");
            for pair in entries.windows(2) {
                assert!(pair[0].time_ticks <= pair[1].time_ticks, "{name}: sorted");
            }
        }
    }

    #[test]
    fn scan_rewinds_the_reader_and_tolerates_garbage() {
        let mut cursor = Cursor::new(H264_AAC_MKV.to_vec());
        let _ = scan_cues(&mut cursor);
        assert_eq!(cursor.position(), 0, "reader handed onward at the start");

        assert!(cues(b"not matroska at all").is_empty());
        assert!(cues(&[]).is_empty());
        // Truncated header: valid EBML magic, then nothing.
        assert!(cues(&[0x1A, 0x45, 0xDF, 0xA3]).is_empty());
    }

    #[test]
    fn shared_reader_clones_share_position() {
        let mut a = SharedReader::new(Cursor::new(vec![1u8, 2, 3, 4]));
        let mut b = a.clone();
        let mut buf = [0u8; 2];
        a.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [1, 2]);
        b.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [3, 4], "clone continues where the other stopped");
        b.seek(SeekFrom::Start(1)).unwrap();
        a.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [2, 3], "seek through one handle moves both");
    }
}
