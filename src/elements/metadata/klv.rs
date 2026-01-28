//! KLV (Key-Length-Value) encoder for STANAG 4609 / MISB metadata.
//!
//! This module provides utilities for creating KLV-encoded metadata packets
//! that can be multiplexed into MPEG-TS streams. Commonly used for:
//!
//! - STANAG 4609 (NATO motion imagery)
//! - MISB ST 0601 (UAS Datalink Local Set)
//! - MISB ST 0102 (Security Metadata Local Set)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::metadata::{KlvEncoder, KlvTag, Uls};
//!
//! let mut encoder = KlvEncoder::new();
//!
//! // Add platform location
//! encoder.add_tag(KlvTag::SensorLatitude, 37.2350_f64.to_be_bytes().to_vec());
//! encoder.add_tag(KlvTag::SensorLongitude, (-115.8111_f64).to_be_bytes().to_vec());
//!
//! // Get encoded KLV packet (with MISB ST 0601 ULS)
//! let klv_data = encoder.encode_with_uls(Uls::MisbSt0601);
//! ```

use crate::buffer::{Buffer, MemoryHandle};

use crate::error::{Error, Result};
use crate::memory::{CpuSegment, MemorySegment};
use crate::metadata::Metadata;

use std::collections::BTreeMap;
use std::sync::Arc;

// ============================================================================
// Universal Label (UL) / Universal Label Set (ULS)
// ============================================================================

/// Universal Label Set identifiers for KLV data.
///
/// These 16-byte keys identify the type of KLV data according to SMPTE and MISB standards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uls {
    /// MISB ST 0601 - UAS Datalink Local Set (most common).
    MisbSt0601,
    /// MISB ST 0102 - Security Metadata Local Set.
    MisbSt0102,
    /// MISB ST 0104 - Predator UAV Basic Universal Set.
    MisbSt0104,
    /// MISB ST 0903 - Video Moving Target Indicator Local Set.
    MisbSt0903,
    /// SMPTE ST 336 generic.
    SmpteGeneric,
    /// Custom ULS (16 bytes).
    Custom([u8; 16]),
}

impl Uls {
    /// Get the 16-byte Universal Label for this ULS.
    pub fn as_bytes(&self) -> &[u8; 16] {
        match self {
            // MISB ST 0601.17 UAS Datalink Local Set
            Uls::MisbSt0601 => &[
                0x06, 0x0E, 0x2B, 0x34, // SMPTE designator
                0x02, 0x0B, 0x01, 0x01, // Registry category + designation
                0x0E, 0x01, 0x03, 0x01, // Organization (MISB)
                0x01, 0x00, 0x00, 0x00, // ST 0601 Local Set
            ],
            // MISB ST 0102 Security Metadata Local Set
            Uls::MisbSt0102 => &[
                0x06, 0x0E, 0x2B, 0x34, 0x02, 0x01, 0x01, 0x01, 0x0E, 0x01, 0x03, 0x03, 0x02, 0x00,
                0x00, 0x00,
            ],
            // MISB ST 0104 Predator Basic
            Uls::MisbSt0104 => &[
                0x06, 0x0E, 0x2B, 0x34, 0x02, 0x01, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x02, 0x01, 0x01,
                0x00, 0x00,
            ],
            // MISB ST 0903 VMTI
            Uls::MisbSt0903 => &[
                0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01, 0x0E, 0x01, 0x03, 0x03, 0x06, 0x00,
                0x00, 0x00,
            ],
            // SMPTE ST 336 generic key
            Uls::SmpteGeneric => &[
                0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00,
            ],
            Uls::Custom(bytes) => bytes,
        }
    }
}

// ============================================================================
// KLV Tags (MISB ST 0601)
// ============================================================================

/// Common KLV tag identifiers from MISB ST 0601.
///
/// These are the local set tags for UAS Datalink metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum KlvTag {
    /// Checksum (tag 1) - calculated automatically.
    Checksum = 1,
    /// Unix timestamp (tag 2) - microseconds since epoch.
    UnixTimeStamp = 2,
    /// Mission ID (tag 3).
    MissionId = 3,
    /// Platform tail number (tag 4).
    PlatformTailNumber = 4,
    /// Platform heading angle (tag 5).
    PlatformHeadingAngle = 5,
    /// Platform pitch angle (tag 6).
    PlatformPitchAngle = 6,
    /// Platform roll angle (tag 7).
    PlatformRollAngle = 7,
    /// Platform true airspeed (tag 8).
    PlatformTrueAirspeed = 8,
    /// Platform indicated airspeed (tag 9).
    PlatformIndicatedAirspeed = 9,
    /// Platform designation (tag 10).
    PlatformDesignation = 10,
    /// Image source sensor (tag 11).
    ImageSourceSensor = 11,
    /// Image coordinate system (tag 12).
    ImageCoordinateSystem = 12,
    /// Sensor latitude (tag 13).
    SensorLatitude = 13,
    /// Sensor longitude (tag 14).
    SensorLongitude = 14,
    /// Sensor true altitude (tag 15).
    SensorTrueAltitude = 15,
    /// Sensor horizontal FOV (tag 16).
    SensorHorizontalFov = 16,
    /// Sensor vertical FOV (tag 17).
    SensorVerticalFov = 17,
    /// Sensor relative azimuth angle (tag 18).
    SensorRelativeAzimuth = 18,
    /// Sensor relative elevation angle (tag 19).
    SensorRelativeElevation = 19,
    /// Sensor relative roll angle (tag 20).
    SensorRelativeRoll = 20,
    /// Slant range (tag 21).
    SlantRange = 21,
    /// Target width (tag 22).
    TargetWidth = 22,
    /// Frame center latitude (tag 23).
    FrameCenterLatitude = 23,
    /// Frame center longitude (tag 24).
    FrameCenterLongitude = 24,
    /// Frame center elevation (tag 25).
    FrameCenterElevation = 25,
    /// Offset corner latitude point 1 (tag 26).
    OffsetCornerLatitude1 = 26,
    /// Offset corner longitude point 1 (tag 27).
    OffsetCornerLongitude1 = 27,
    /// Offset corner latitude point 2 (tag 28).
    OffsetCornerLatitude2 = 28,
    /// Offset corner longitude point 2 (tag 29).
    OffsetCornerLongitude2 = 29,
    /// Offset corner latitude point 3 (tag 30).
    OffsetCornerLatitude3 = 30,
    /// Offset corner longitude point 3 (tag 31).
    OffsetCornerLongitude3 = 31,
    /// Offset corner latitude point 4 (tag 32).
    OffsetCornerLatitude4 = 32,
    /// Offset corner longitude point 4 (tag 33).
    OffsetCornerLongitude4 = 33,
    /// Security local set (tag 48).
    SecurityLocalSet = 48,
    /// UAS local set version (tag 65).
    UasLsVersion = 65,
    /// Custom tag.
    Custom(u8),
}

impl From<KlvTag> for u8 {
    fn from(tag: KlvTag) -> u8 {
        match tag {
            KlvTag::Checksum => 1,
            KlvTag::UnixTimeStamp => 2,
            KlvTag::MissionId => 3,
            KlvTag::PlatformTailNumber => 4,
            KlvTag::PlatformHeadingAngle => 5,
            KlvTag::PlatformPitchAngle => 6,
            KlvTag::PlatformRollAngle => 7,
            KlvTag::PlatformTrueAirspeed => 8,
            KlvTag::PlatformIndicatedAirspeed => 9,
            KlvTag::PlatformDesignation => 10,
            KlvTag::ImageSourceSensor => 11,
            KlvTag::ImageCoordinateSystem => 12,
            KlvTag::SensorLatitude => 13,
            KlvTag::SensorLongitude => 14,
            KlvTag::SensorTrueAltitude => 15,
            KlvTag::SensorHorizontalFov => 16,
            KlvTag::SensorVerticalFov => 17,
            KlvTag::SensorRelativeAzimuth => 18,
            KlvTag::SensorRelativeElevation => 19,
            KlvTag::SensorRelativeRoll => 20,
            KlvTag::SlantRange => 21,
            KlvTag::TargetWidth => 22,
            KlvTag::FrameCenterLatitude => 23,
            KlvTag::FrameCenterLongitude => 24,
            KlvTag::FrameCenterElevation => 25,
            KlvTag::OffsetCornerLatitude1 => 26,
            KlvTag::OffsetCornerLongitude1 => 27,
            KlvTag::OffsetCornerLatitude2 => 28,
            KlvTag::OffsetCornerLongitude2 => 29,
            KlvTag::OffsetCornerLatitude3 => 30,
            KlvTag::OffsetCornerLongitude3 => 31,
            KlvTag::OffsetCornerLatitude4 => 32,
            KlvTag::OffsetCornerLongitude4 => 33,
            KlvTag::SecurityLocalSet => 48,
            KlvTag::UasLsVersion => 65,
            KlvTag::Custom(v) => v,
        }
    }
}

impl From<u8> for KlvTag {
    fn from(v: u8) -> Self {
        match v {
            1 => KlvTag::Checksum,
            2 => KlvTag::UnixTimeStamp,
            3 => KlvTag::MissionId,
            4 => KlvTag::PlatformTailNumber,
            5 => KlvTag::PlatformHeadingAngle,
            6 => KlvTag::PlatformPitchAngle,
            7 => KlvTag::PlatformRollAngle,
            8 => KlvTag::PlatformTrueAirspeed,
            9 => KlvTag::PlatformIndicatedAirspeed,
            10 => KlvTag::PlatformDesignation,
            11 => KlvTag::ImageSourceSensor,
            12 => KlvTag::ImageCoordinateSystem,
            13 => KlvTag::SensorLatitude,
            14 => KlvTag::SensorLongitude,
            15 => KlvTag::SensorTrueAltitude,
            16 => KlvTag::SensorHorizontalFov,
            17 => KlvTag::SensorVerticalFov,
            18 => KlvTag::SensorRelativeAzimuth,
            19 => KlvTag::SensorRelativeElevation,
            20 => KlvTag::SensorRelativeRoll,
            21 => KlvTag::SlantRange,
            22 => KlvTag::TargetWidth,
            23 => KlvTag::FrameCenterLatitude,
            24 => KlvTag::FrameCenterLongitude,
            25 => KlvTag::FrameCenterElevation,
            26 => KlvTag::OffsetCornerLatitude1,
            27 => KlvTag::OffsetCornerLongitude1,
            28 => KlvTag::OffsetCornerLatitude2,
            29 => KlvTag::OffsetCornerLongitude2,
            30 => KlvTag::OffsetCornerLatitude3,
            31 => KlvTag::OffsetCornerLongitude3,
            32 => KlvTag::OffsetCornerLatitude4,
            33 => KlvTag::OffsetCornerLongitude4,
            48 => KlvTag::SecurityLocalSet,
            65 => KlvTag::UasLsVersion,
            v => KlvTag::Custom(v),
        }
    }
}

// ============================================================================
// BER Length Encoding
// ============================================================================

/// Encode a length using BER (Basic Encoding Rules).
///
/// - 0-127: Single byte
/// - 128-255: 0x81 + 1 byte
/// - 256-65535: 0x82 + 2 bytes
/// - 65536+: 0x84 + 4 bytes
fn encode_ber_length(length: usize) -> Vec<u8> {
    if length < 128 {
        vec![length as u8]
    } else if length < 256 {
        vec![0x81, length as u8]
    } else if length < 65536 {
        vec![0x82, (length >> 8) as u8, (length & 0xFF) as u8]
    } else {
        vec![
            0x84,
            (length >> 24) as u8,
            (length >> 16) as u8,
            (length >> 8) as u8,
            (length & 0xFF) as u8,
        ]
    }
}

/// Decode a BER-encoded length, returning (length, bytes_consumed).
pub fn decode_ber_length(data: &[u8]) -> Option<(usize, usize)> {
    if data.is_empty() {
        return None;
    }

    let first = data[0];
    if first < 128 {
        Some((first as usize, 1))
    } else {
        let num_bytes = (first & 0x7F) as usize;
        if data.len() < 1 + num_bytes {
            return None;
        }

        let mut length = 0usize;
        for i in 0..num_bytes {
            length = (length << 8) | data[1 + i] as usize;
        }
        Some((length, 1 + num_bytes))
    }
}

// ============================================================================
// KLV Encoder
// ============================================================================

/// KLV encoder for creating STANAG 4609 / MISB metadata packets.
#[derive(Debug, Default)]
pub struct KlvEncoder {
    /// Tags and their values (BTreeMap for deterministic ordering).
    tags: BTreeMap<u8, Vec<u8>>,
}

impl KlvEncoder {
    /// Create a new empty KLV encoder.
    pub fn new() -> Self {
        Self {
            tags: BTreeMap::new(),
        }
    }

    /// Add a tag with raw bytes value.
    pub fn add_tag(&mut self, tag: KlvTag, value: Vec<u8>) -> &mut Self {
        self.tags.insert(tag.into(), value);
        self
    }

    /// Add a tag with a u8 value.
    pub fn add_u8(&mut self, tag: KlvTag, value: u8) -> &mut Self {
        self.tags.insert(tag.into(), vec![value]);
        self
    }

    /// Add a tag with a u16 value (big-endian).
    pub fn add_u16(&mut self, tag: KlvTag, value: u16) -> &mut Self {
        self.tags.insert(tag.into(), value.to_be_bytes().to_vec());
        self
    }

    /// Add a tag with a u32 value (big-endian).
    pub fn add_u32(&mut self, tag: KlvTag, value: u32) -> &mut Self {
        self.tags.insert(tag.into(), value.to_be_bytes().to_vec());
        self
    }

    /// Add a tag with a u64 value (big-endian).
    pub fn add_u64(&mut self, tag: KlvTag, value: u64) -> &mut Self {
        self.tags.insert(tag.into(), value.to_be_bytes().to_vec());
        self
    }

    /// Add a tag with a string value.
    pub fn add_string(&mut self, tag: KlvTag, value: &str) -> &mut Self {
        self.tags.insert(tag.into(), value.as_bytes().to_vec());
        self
    }

    /// Add Unix timestamp (microseconds since epoch).
    pub fn add_timestamp(&mut self, micros: u64) -> &mut Self {
        self.add_u64(KlvTag::UnixTimeStamp, micros)
    }

    /// Add current Unix timestamp.
    pub fn add_current_timestamp(&mut self) -> &mut Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        self.add_timestamp(micros)
    }

    /// Add sensor latitude (degrees, -90 to +90).
    ///
    /// Encoded as IMAPB per MISB ST 0601.
    pub fn add_sensor_latitude(&mut self, degrees: f64) -> &mut Self {
        let encoded = encode_latitude(degrees);
        self.add_tag(KlvTag::SensorLatitude, encoded.to_be_bytes().to_vec())
    }

    /// Add sensor longitude (degrees, -180 to +180).
    ///
    /// Encoded as IMAPB per MISB ST 0601.
    pub fn add_sensor_longitude(&mut self, degrees: f64) -> &mut Self {
        let encoded = encode_longitude(degrees);
        self.add_tag(KlvTag::SensorLongitude, encoded.to_be_bytes().to_vec())
    }

    /// Add sensor altitude (meters HAE).
    ///
    /// Encoded as IMAPB per MISB ST 0601.
    pub fn add_sensor_altitude(&mut self, meters: f64) -> &mut Self {
        let encoded = encode_altitude(meters);
        self.add_tag(KlvTag::SensorTrueAltitude, encoded.to_be_bytes().to_vec())
    }

    /// Add frame center latitude (degrees).
    pub fn add_frame_center_latitude(&mut self, degrees: f64) -> &mut Self {
        let encoded = encode_latitude(degrees);
        self.add_tag(KlvTag::FrameCenterLatitude, encoded.to_be_bytes().to_vec())
    }

    /// Add frame center longitude (degrees).
    pub fn add_frame_center_longitude(&mut self, degrees: f64) -> &mut Self {
        let encoded = encode_longitude(degrees);
        self.add_tag(KlvTag::FrameCenterLongitude, encoded.to_be_bytes().to_vec())
    }

    /// Add platform heading angle (degrees, 0 to 360).
    pub fn add_platform_heading(&mut self, degrees: f64) -> &mut Self {
        let encoded = encode_heading(degrees);
        self.add_tag(KlvTag::PlatformHeadingAngle, encoded.to_be_bytes().to_vec())
    }

    /// Add platform pitch angle (degrees, -90 to +90).
    pub fn add_platform_pitch(&mut self, degrees: f64) -> &mut Self {
        let encoded = encode_pitch(degrees);
        self.add_tag(KlvTag::PlatformPitchAngle, encoded.to_be_bytes().to_vec())
    }

    /// Add platform roll angle (degrees, -90 to +90).
    pub fn add_platform_roll(&mut self, degrees: f64) -> &mut Self {
        let encoded = encode_roll(degrees);
        self.add_tag(KlvTag::PlatformRollAngle, encoded.to_be_bytes().to_vec())
    }

    /// Clear all tags.
    pub fn clear(&mut self) {
        self.tags.clear();
    }

    /// Get the number of tags.
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Encode the local set (tags only, without ULS key).
    pub fn encode_local_set(&self) -> Vec<u8> {
        let mut local_set = Vec::new();

        for (&tag, value) in &self.tags {
            if tag == u8::from(KlvTag::Checksum) {
                continue; // Checksum is added at the end
            }

            // Tag (1 byte for local set)
            local_set.push(tag);
            // Length (BER encoded)
            local_set.extend(encode_ber_length(value.len()));
            // Value
            local_set.extend(value);
        }

        local_set
    }

    /// Encode as a complete KLV packet with the specified ULS.
    ///
    /// Format: [16-byte ULS key] [BER length] [local set with checksum]
    pub fn encode_with_uls(&self, uls: Uls) -> Vec<u8> {
        let mut packet = Vec::new();

        // ULS key (16 bytes)
        packet.extend_from_slice(uls.as_bytes());

        // Encode local set
        let local_set = self.encode_local_set();

        // Calculate checksum over local set
        let checksum = calculate_checksum(&local_set);

        // Total value length = local set + checksum tag (1) + checksum length (1) + checksum (2)
        let total_length = local_set.len() + 1 + 1 + 2;

        // BER-encoded length
        packet.extend(encode_ber_length(total_length));

        // Local set
        packet.extend(local_set);

        // Checksum (tag 1, length 2, value 2 bytes)
        packet.push(u8::from(KlvTag::Checksum));
        packet.push(2); // Length
        packet.extend(checksum.to_be_bytes());

        packet
    }

    /// Encode as a MISB ST 0601 packet (most common).
    pub fn encode_st0601(&self) -> Vec<u8> {
        self.encode_with_uls(Uls::MisbSt0601)
    }

    /// Create a Buffer containing the encoded KLV data.
    pub fn to_buffer(&self, uls: Uls) -> Result<Buffer> {
        let data = self.encode_with_uls(uls);
        create_klv_buffer(data)
    }

    /// Create a MISB ST 0601 Buffer.
    pub fn to_st0601_buffer(&self) -> Result<Buffer> {
        self.to_buffer(Uls::MisbSt0601)
    }
}

// ============================================================================
// IMAPB Encoding (MISB ST 1201)
// ============================================================================

/// Encode latitude (-90 to +90 degrees) as 4-byte IMAPB.
fn encode_latitude(degrees: f64) -> i32 {
    let clamped = degrees.clamp(-90.0, 90.0);
    // Scale to [-2^31+1, 2^31-1] range
    let scaled = (clamped / 90.0) * (i32::MAX as f64);
    scaled.round() as i32
}

/// Encode longitude (-180 to +180 degrees) as 4-byte IMAPB.
fn encode_longitude(degrees: f64) -> i32 {
    let clamped = degrees.clamp(-180.0, 180.0);
    // Scale to [-2^31+1, 2^31-1] range
    let scaled = (clamped / 180.0) * (i32::MAX as f64);
    scaled.round() as i32
}

/// Encode altitude (-900 to +19000 meters) as 2-byte IMAPB.
fn encode_altitude(meters: f64) -> u16 {
    let clamped = meters.clamp(-900.0, 19000.0);
    // Offset and scale
    let offset = clamped + 900.0;
    let scaled = (offset / 19900.0) * (u16::MAX as f64);
    scaled.round() as u16
}

/// Encode heading (0 to 360 degrees) as 2-byte IMAPB.
fn encode_heading(degrees: f64) -> u16 {
    let clamped = degrees.clamp(0.0, 360.0);
    let scaled = (clamped / 360.0) * (u16::MAX as f64);
    scaled.round() as u16
}

/// Encode pitch (-90 to +90 degrees) as 2-byte IMAPB.
fn encode_pitch(degrees: f64) -> i16 {
    let clamped = degrees.clamp(-90.0, 90.0);
    let scaled = (clamped / 90.0) * (i16::MAX as f64);
    scaled.round() as i16
}

/// Encode roll (-90 to +90 degrees) as 2-byte IMAPB.
fn encode_roll(degrees: f64) -> i16 {
    let clamped = degrees.clamp(-90.0, 90.0);
    let scaled = (clamped / 90.0) * (i16::MAX as f64);
    scaled.round() as i16
}

// ============================================================================
// Checksum Calculation
// ============================================================================

/// Calculate MISB checksum (running 16-bit sum).
fn calculate_checksum(data: &[u8]) -> u16 {
    let mut sum: u16 = 0;
    for (i, &byte) in data.iter().enumerate() {
        // Checksum includes position-weighted bytes
        sum = sum.wrapping_add((byte as u16) << (8 * ((i + 1) % 2)));
    }
    sum
}

// ============================================================================
// Buffer Creation
// ============================================================================

/// Create a Buffer from KLV data.
fn create_klv_buffer(data: Vec<u8>) -> Result<Buffer> {
    if data.is_empty() {
        return Err(Error::Element("Empty KLV data".into()));
    }

    let segment = Arc::new(
        CpuSegment::new(data.len())
            .map_err(|e| Error::Element(format!("Failed to allocate buffer: {}", e)))?,
    );

    let ptr = segment
        .as_mut_ptr()
        .ok_or_else(|| Error::Element("Failed to get segment pointer".into()))?;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }

    let handle = MemoryHandle::from_segment_with_len(segment, data.len());
    let metadata = Metadata::new();

    Ok(Buffer::new(handle, metadata))
}

// ============================================================================
// Convenience Builder
// ============================================================================

/// Builder for creating STANAG 4609 / MISB ST 0601 metadata.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::metadata::StanagMetadataBuilder;
///
/// let klv = StanagMetadataBuilder::new()
///     .timestamp_now()
///     .sensor_position(37.2350, -115.8111, 1500.0)
///     .frame_center(37.2300, -115.8100)
///     .platform_attitude(180.0, 5.0, 0.0)
///     .build_st0601();
/// ```
#[derive(Debug, Default)]
pub struct StanagMetadataBuilder {
    encoder: KlvEncoder,
}

impl StanagMetadataBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            encoder: KlvEncoder::new(),
        }
    }

    /// Add current timestamp.
    pub fn timestamp_now(mut self) -> Self {
        self.encoder.add_current_timestamp();
        self
    }

    /// Add Unix timestamp in microseconds.
    pub fn timestamp(mut self, micros: u64) -> Self {
        self.encoder.add_timestamp(micros);
        self
    }

    /// Add mission ID.
    pub fn mission_id(mut self, id: &str) -> Self {
        self.encoder.add_string(KlvTag::MissionId, id);
        self
    }

    /// Add platform designation.
    pub fn platform_designation(mut self, designation: &str) -> Self {
        self.encoder
            .add_string(KlvTag::PlatformDesignation, designation);
        self
    }

    /// Add sensor position (lat, lon, alt in degrees/meters).
    pub fn sensor_position(mut self, lat: f64, lon: f64, alt: f64) -> Self {
        self.encoder.add_sensor_latitude(lat);
        self.encoder.add_sensor_longitude(lon);
        self.encoder.add_sensor_altitude(alt);
        self
    }

    /// Add frame center coordinates.
    pub fn frame_center(mut self, lat: f64, lon: f64) -> Self {
        self.encoder.add_frame_center_latitude(lat);
        self.encoder.add_frame_center_longitude(lon);
        self
    }

    /// Add platform attitude (heading, pitch, roll in degrees).
    pub fn platform_attitude(mut self, heading: f64, pitch: f64, roll: f64) -> Self {
        self.encoder.add_platform_heading(heading);
        self.encoder.add_platform_pitch(pitch);
        self.encoder.add_platform_roll(roll);
        self
    }

    /// Add slant range (meters).
    pub fn slant_range(mut self, meters: f64) -> Self {
        // Encoded as 4-byte unsigned, 0 to 5,000,000 meters
        let clamped = meters.clamp(0.0, 5_000_000.0);
        let scaled = (clamped / 5_000_000.0) * (u32::MAX as f64);
        self.encoder
            .add_u32(KlvTag::SlantRange, scaled.round() as u32);
        self
    }

    /// Add UAS local set version.
    pub fn version(mut self, version: u8) -> Self {
        self.encoder.add_u8(KlvTag::UasLsVersion, version);
        self
    }

    /// Add a raw tag.
    pub fn raw_tag(mut self, tag: KlvTag, value: Vec<u8>) -> Self {
        self.encoder.add_tag(tag, value);
        self
    }

    /// Build as MISB ST 0601 encoded bytes.
    pub fn build_st0601(self) -> Vec<u8> {
        self.encoder.encode_st0601()
    }

    /// Build as a Buffer containing MISB ST 0601 data.
    pub fn build_st0601_buffer(self) -> Result<Buffer> {
        self.encoder.to_st0601_buffer()
    }

    /// Build with a custom ULS.
    pub fn build_with_uls(self, uls: Uls) -> Vec<u8> {
        self.encoder.encode_with_uls(uls)
    }

    /// Get the underlying encoder.
    pub fn into_encoder(self) -> KlvEncoder {
        self.encoder
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ber_length_encoding() {
        assert_eq!(encode_ber_length(0), vec![0x00]);
        assert_eq!(encode_ber_length(127), vec![0x7F]);
        assert_eq!(encode_ber_length(128), vec![0x81, 0x80]);
        assert_eq!(encode_ber_length(255), vec![0x81, 0xFF]);
        assert_eq!(encode_ber_length(256), vec![0x82, 0x01, 0x00]);
        assert_eq!(encode_ber_length(65535), vec![0x82, 0xFF, 0xFF]);
    }

    #[test]
    fn test_ber_length_decoding() {
        assert_eq!(decode_ber_length(&[0x00]), Some((0, 1)));
        assert_eq!(decode_ber_length(&[0x7F]), Some((127, 1)));
        assert_eq!(decode_ber_length(&[0x81, 0x80]), Some((128, 2)));
        assert_eq!(decode_ber_length(&[0x82, 0x01, 0x00]), Some((256, 3)));
    }

    #[test]
    fn test_klv_encoder_basic() {
        let mut encoder = KlvEncoder::new();
        encoder.add_u8(KlvTag::UasLsVersion, 17);
        encoder.add_string(KlvTag::MissionId, "TEST");

        assert_eq!(encoder.len(), 2);

        let local_set = encoder.encode_local_set();
        assert!(!local_set.is_empty());
    }

    #[test]
    fn test_klv_encoder_st0601() {
        let mut encoder = KlvEncoder::new();
        encoder
            .add_timestamp(1234567890_000_000)
            .add_sensor_latitude(37.2350)
            .add_sensor_longitude(-115.8111);

        let packet = encoder.encode_st0601();

        // Should start with ST 0601 ULS
        assert_eq!(&packet[0..4], &[0x06, 0x0E, 0x2B, 0x34]);

        // Should have content
        assert!(packet.len() > 20);
    }

    #[test]
    fn test_latitude_encoding() {
        // Test 0 degrees
        let zero = encode_latitude(0.0);
        assert_eq!(zero, 0);

        // Test positive
        let pos = encode_latitude(45.0);
        assert!(pos > 0);

        // Test negative
        let neg = encode_latitude(-45.0);
        assert!(neg < 0);

        // Test clamping
        let clamped_hi = encode_latitude(100.0);
        let clamped_max = encode_latitude(90.0);
        assert_eq!(clamped_hi, clamped_max);
    }

    #[test]
    fn test_longitude_encoding() {
        let zero = encode_longitude(0.0);
        assert_eq!(zero, 0);

        let pos = encode_longitude(90.0);
        assert!(pos > 0);

        let neg = encode_longitude(-90.0);
        assert!(neg < 0);
    }

    #[test]
    fn test_altitude_encoding() {
        // -900m should be 0
        let min = encode_altitude(-900.0);
        assert_eq!(min, 0);

        // 19000m should be max
        let max = encode_altitude(19000.0);
        assert_eq!(max, u16::MAX);

        // 0m should be somewhere in between
        let zero = encode_altitude(0.0);
        assert!(zero > 0 && zero < u16::MAX);
    }

    #[test]
    fn test_stanag_builder() {
        let klv = StanagMetadataBuilder::new()
            .version(17)
            .mission_id("TEST_MISSION")
            .sensor_position(37.2350, -115.8111, 1500.0)
            .frame_center(37.2300, -115.8100)
            .platform_attitude(180.0, 5.0, 0.0)
            .build_st0601();

        // Verify it starts with ULS
        assert_eq!(&klv[0..4], &[0x06, 0x0E, 0x2B, 0x34]);

        // Should have reasonable length
        assert!(klv.len() > 40);
    }

    #[test]
    fn test_checksum_calculation() {
        let data = vec![0x01, 0x02, 0x03, 0x04];
        let checksum = calculate_checksum(&data);
        assert_ne!(checksum, 0);
    }

    #[test]
    fn test_uls_keys() {
        assert_eq!(Uls::MisbSt0601.as_bytes().len(), 16);
        assert_eq!(Uls::MisbSt0102.as_bytes().len(), 16);

        // All start with SMPTE designator
        assert_eq!(&Uls::MisbSt0601.as_bytes()[0..4], &[0x06, 0x0E, 0x2B, 0x34]);
    }

    #[test]
    fn test_klv_tag_conversion() {
        assert_eq!(u8::from(KlvTag::UnixTimeStamp), 2);
        assert_eq!(u8::from(KlvTag::SensorLatitude), 13);

        let tag: KlvTag = 13.into();
        assert_eq!(tag, KlvTag::SensorLatitude);

        let custom: KlvTag = 100.into();
        assert_eq!(custom, KlvTag::Custom(100));
    }

    #[test]
    fn test_klv_buffer_creation() {
        let encoder = KlvEncoder::new();
        let buffer = encoder.to_st0601_buffer().unwrap();
        assert!(buffer.len() > 16); // At least ULS + length + checksum
    }

    #[test]
    fn test_heading_encoding() {
        let zero = encode_heading(0.0);
        assert_eq!(zero, 0);

        let half = encode_heading(180.0);
        assert!(half > 0 && half < u16::MAX);

        let full = encode_heading(360.0);
        assert_eq!(full, u16::MAX);
    }

    #[test]
    fn test_pitch_roll_encoding() {
        let zero_pitch = encode_pitch(0.0);
        assert_eq!(zero_pitch, 0);

        let max_pitch = encode_pitch(90.0);
        assert_eq!(max_pitch, i16::MAX);

        let min_pitch = encode_pitch(-90.0);
        // Allow for rounding
        assert!(min_pitch < 0);
    }
}
