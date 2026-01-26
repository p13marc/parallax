//! Video display example using AsyncVideoTestSrc and IcedVideoSink.
//!
//! This example demonstrates displaying video test patterns in a GUI window
//! using a proper pipeline structure with an async video source.
//!
//! Run with: cargo run --example video_display --features iced-sink
//!
//! Controls:
//! - Close the window to exit

use parallax::element::{AsyncSourceAdapter, DynAsyncElement, SinkAdapter};
use parallax::elements::{
    AsyncVideoTestSrc, IcedVideoSink, IcedVideoSinkConfig, InputPixelFormat, VideoPattern,
};
use parallax::pipeline::{Pipeline, PipelineExecutor};
use std::thread;

fn main() -> iced::Result {
    println!("=== Video Display Example ===");
    println!("Displaying moving ball pattern at 640x480 @ 30fps");
    println!("Using AsyncVideoTestSrc with tokio timer for precise framerate");
    println!("Close the window to exit.\n");

    // Configuration
    let width = 640;
    let height = 480;
    let pattern = VideoPattern::MovingBall;

    // Create the async video source with live mode (tokio timer for framerate)
    let src = AsyncVideoTestSrc::new()
        .with_pattern(pattern)
        .with_resolution(width, height)
        .with_framerate(30, 1)
        .with_name("test-pattern")
        .live(true); // Enable tokio timer for precise framerate control

    // Create the video sink
    let config = IcedVideoSinkConfig {
        title: format!("Parallax Video - {:?}", pattern),
        width,
        height,
        show_stats: true,
        pixel_format: InputPixelFormat::Rgb24,
    };

    let (sink, window_handle) = IcedVideoSink::with_config(config);

    // Build the pipeline
    let mut pipeline = Pipeline::new();

    // Use AsyncSourceAdapter to bridge the async source to the pipeline
    let src_node = pipeline.add_node(
        "videotestsrc",
        DynAsyncElement::new_box(AsyncSourceAdapter::new(src)),
    );

    let sink_node = pipeline.add_node(
        "iced_sink",
        DynAsyncElement::new_box(SinkAdapter::new(sink)),
    );

    pipeline
        .link(src_node, sink_node)
        .expect("Failed to link pipeline");

    // Run the pipeline in a background thread with a multi-threaded tokio runtime
    // (required for block_in_place used by AsyncSourceAdapter)
    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        rt.block_on(async {
            let executor = PipelineExecutor::new();
            if let Err(e) = executor.run(&mut pipeline).await {
                eprintln!("Pipeline error: {}", e);
            }
        });
        println!("Pipeline finished");
    });

    // Run the Iced window on the main thread (blocks until closed)
    window_handle.run()
}
