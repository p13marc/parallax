//! Display video frames in a GUI window using Iced.
//!
//! Run with: cargo run --example 16_video_display --features iced-sink

use parallax::element::{AsyncSourceAdapter, DynAsyncElement, SinkAdapter};
use parallax::elements::{
    AsyncVideoTestSrc, IcedVideoSink, IcedVideoSinkConfig, InputPixelFormat, VideoPattern,
};
use parallax::pipeline::Pipeline;
use std::thread;

fn main() -> iced::Result {
    println!("Video display: 640x480 @ 30fps");
    println!("Close window to exit");

    let width = 640;
    let height = 480;

    let src = AsyncVideoTestSrc::new()
        .with_resolution(width, height)
        .with_pattern(VideoPattern::MovingBall)
        .live(true);

    let config = IcedVideoSinkConfig {
        width,
        height,
        title: "Video Display Example".to_string(),
        input_format: InputPixelFormat::Rgba,
    };
    let (sink, handle) = IcedVideoSink::new(config);

    thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut pipeline = Pipeline::new();
            let src_node = pipeline.add_node(
                "src",
                DynAsyncElement::new_box(AsyncSourceAdapter::new(src)),
            );
            let sink_node =
                pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(sink)));
            pipeline.link(src_node, sink_node).unwrap();
            let _ = pipeline.run().await;
        });
    });

    handle.run()
}
