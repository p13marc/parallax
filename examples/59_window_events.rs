//! Window input from AutoVideoSink: keys, mouse, fullscreen (#74).
//!
//! Displays a test pattern and reacts to the video window itself:
//! - **Space** — pause / resume the pipeline
//! - **f** or **Enter** — toggle borderless fullscreen
//! - **Escape** or **q** — quit
//! - Left click prints the position
//!
//! ```bash
//! cargo run --example 59_window_events --features display
//! ```

use parallax::elements::{AutoVideoSink, VideoKey, VideoTestSrc, VideoWindowEvent};
use parallax::pipeline::{Executor, Pipeline};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", VideoTestSrc::new().with_resolution(640, 480));
    let mut sink = AutoVideoSink::new().with_title("parallax — window events demo");

    // The window handle must be taken BEFORE Executor::start moves the sink.
    let window = sink.handle();
    let sink_id = pipeline.add_sink("display", sink);
    pipeline.link(src, sink_id)?;

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline)?;
    let mut ended = handle.ended();

    println!("Space = pause/resume, f/Enter = fullscreen, Esc/q = quit");

    // The window opens lazily on the first frame; only a false AFTER a true
    // means the user closed it.
    let mut window_seen = false;
    loop {
        tokio::select! {
            reason = &mut ended => {
                println!("pipeline ended: {reason:?}");
                break;
            }
            _ = tokio::time::sleep(Duration::from_millis(20)) => {
                while let Some(event) = window.try_event() {
                    match event {
                        VideoWindowEvent::KeyPressed(VideoKey::Space) => {
                            if handle.is_paused() {
                                println!("resume");
                                handle.resume();
                            } else {
                                println!("pause");
                                handle.pause();
                            }
                        }
                        VideoWindowEvent::KeyPressed(VideoKey::Enter) => {
                            window.set_fullscreen(!window.is_fullscreen());
                        }
                        VideoWindowEvent::KeyPressed(VideoKey::Escape) => {
                            println!("quit");
                            handle.stop();
                        }
                        VideoWindowEvent::KeyPressed(VideoKey::Character(c)) => match c.as_str() {
                            "f" => window.set_fullscreen(!window.is_fullscreen()),
                            "q" => {
                                println!("quit");
                                handle.stop();
                            }
                            other => println!("key: {other}"),
                        },
                        VideoWindowEvent::MousePressed { x, y } => {
                            println!("click at {x:.0},{y:.0}");
                        }
                        VideoWindowEvent::CloseRequested => println!("window closed"),
                        VideoWindowEvent::Resized { width, height } => {
                            println!("resized to {width}x{height}");
                        }
                        other => println!("{other:?}"),
                    }
                }
                if window.is_open() {
                    window_seen = true;
                } else if window_seen {
                    handle.stop();
                }
            }
        }
    }

    handle.wait().await?;
    Ok(())
}
