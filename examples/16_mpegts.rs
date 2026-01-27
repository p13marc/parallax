//! # MPEG-TS Muxing
//!
//! Multiplex video and metadata into MPEG-TS container.
//!
//! ```text
//! [VideoSrc] ──┐
//!              ├→ [TsMux] → [FileSink]
//! [DataSrc]  ──┘
//! ```
//!
//! Run: `cargo run --example 16_mpegts --features mpeg-ts`

use parallax::clock::ClockTime;
use parallax::element::{DynAsyncElement, ProduceContext, ProduceResult, Source};
use parallax::elements::FileSink;
use parallax::elements::mux::{TsMuxConfig, TsMuxElement, TsMuxStreamType, TsMuxTrack};
use parallax::error::Result;
use parallax::memory::CpuArena;
use parallax::pipeline::Pipeline;

/// Simulates H.264 NAL units (simplified)
struct VideoSource {
    frame: u32,
    max_frames: u32,
}

impl Source for VideoSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.frame >= self.max_frames {
            return Ok(ProduceResult::Eos);
        }

        // Fake H.264 NAL unit (SPS-like header)
        let nal = [0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1f];
        ctx.output()[..nal.len()].copy_from_slice(&nal);

        // Set PTS (40ms per frame = 25fps)
        ctx.metadata_mut().pts = ClockTime::from_millis(self.frame as u64 * 40);
        ctx.metadata_mut().sequence = self.frame as u64;

        self.frame += 1;
        Ok(ProduceResult::Produced(nal.len()))
    }
}

/// Simulates KLV metadata
struct DataSource {
    packet: u32,
    max_packets: u32,
}

impl Source for DataSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.packet >= self.max_packets {
            return Ok(ProduceResult::Eos);
        }

        // Fake KLV data (MISB key prefix)
        let klv = [0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x01];
        ctx.output()[..klv.len()].copy_from_slice(&klv);

        // Set PTS (align with video)
        ctx.metadata_mut().pts = ClockTime::from_millis(self.packet as u64 * 40);
        ctx.metadata_mut().sequence = self.packet as u64;

        self.packet += 1;
        Ok(ProduceResult::Produced(klv.len()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== MPEG-TS Muxing Pipeline ===\n");

    let output_path = "/tmp/output.ts";

    // Configure TS muxer with video and data tracks
    let mux_config = TsMuxConfig::new()
        .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
        .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());

    let mux = TsMuxElement::new(mux_config)?;
    println!("TsMux configured with {} tracks", mux.inputs().len());

    let mut pipeline = Pipeline::new();

    // Video source
    let video_arena = CpuArena::new(256, 8)?;
    let video_src = pipeline.add_source_with_arena(
        "video_src",
        VideoSource {
            frame: 0,
            max_frames: 10,
        },
        video_arena,
    );

    // Data source
    let data_arena = CpuArena::new(256, 8)?;
    let data_src = pipeline.add_source_with_arena(
        "data_src",
        DataSource {
            packet: 0,
            max_packets: 10,
        },
        data_arena,
    );

    // Muxer (special element type - needs DynAsyncElement wrapping)
    let muxer = pipeline.add_element("tsmux", mux);

    // File sink
    let sink = pipeline.add_sink("filesink", FileSink::new(output_path));

    // Link: video and data → mux → file
    pipeline.link(video_src, muxer)?;
    pipeline.link(data_src, muxer)?;
    pipeline.link(muxer, sink)?;

    println!("Muxing 10 video frames + 10 data packets...\n");
    pipeline.run().await?;

    let file_size = std::fs::metadata(output_path)?.len();
    println!("Output: {}", output_path);
    println!("Size: {} bytes", file_size);
    println!("TS packets: {}", file_size / 188);

    Ok(())
}
