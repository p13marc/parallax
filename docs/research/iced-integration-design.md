# Plan: Proper Video Display Sink (autovideosink)

## Problem Statement

We want to express pipelines like:
```rust
Pipeline::parse("v4l2src ! videoconvert ! autovideosink")?.run().await?;
```

And have it "just work" - automatically creating a window and displaying video.

## How GStreamer Does It

GStreamer's `autovideosink` selects an appropriate sink (like `xvimagesink`). Looking at [xvimagesink.c](https://github.com/GStreamer/gst-plugins-base/blob/master/sys/xvimage/xvimagesink.c):

1. **Creates its own X11 window** via `gst_xvcontext_create_xwindow()`
2. **Runs its own event thread** (`gst_xv_image_sink_event_thread`) that:
   - Loops while `running` is true
   - Calls `gst_xv_image_sink_handle_xevents()` to process X11 events
   - Sleeps ~50ms between iterations
3. **Rendering is non-blocking** - `show_frame()` just blits to the window
4. **No external main loop required** - the sink is self-contained

Key insight: **GStreamer video sinks don't need special traits or lifecycle management - they're just regular sinks that happen to create windows and run their own event threads.**

## The Parallax Solution

We can do the same thing using:
- **[winit](https://github.com/rust-windowing/winit)** - Cross-platform window creation and event handling
- **[softbuffer](https://github.com/rust-windowing/softbuffer)** - GPU-less 2D buffer display

### Architecture

```
Pipeline Thread (Tokio):              Display Thread:
┌─────────────────────────────────┐   ┌──────────────────────────────┐
│ v4l2src → videoconvert → sink   │   │ winit EventLoop              │
│                           │     │   │   - Handle resize/close      │
│                           ▼     │   │   - Redraw on request        │
│                    ┌──────────┐ │   │                              │
│                    │ Channel  │─┼───│→ softbuffer Surface          │
│                    │ (frames) │ │   │   - Blit RGBA pixels         │
│                    └──────────┘ │   │                              │
└─────────────────────────────────┘   └──────────────────────────────┘
```

The `AutoVideoSink`:
1. **On creation**: Spawns a display thread with winit event loop
2. **On `consume()`**: Sends frame via channel (non-blocking with bounded channel)
3. **Display thread**: Receives frames, blits to softbuffer surface
4. **On drop**: Signals display thread to close

### Why This Works

- `AutoVideoSink` is a **regular `Sink`** - no special trait needed
- The sink spawns its own display thread (like GStreamer)
- `pipeline.run().await` works normally
- No API changes required

---

## Implementation

### Phase 1: Core AutoVideoSink

**File:** `src/elements/app/autovideosink.rs`

```rust
use crate::element::{Sink, ConsumeContext};
use crate::error::{Error, Result};
use std::sync::mpsc::{self, SyncSender, Receiver};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Frame data sent to the display thread.
struct DisplayFrame {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

/// A video sink that automatically creates a window and displays frames.
///
/// This sink spawns its own display thread with a winit event loop,
/// similar to how GStreamer's xvimagesink works. No special lifecycle
/// management is required - it's just a regular sink.
///
/// # Example
///
/// ```rust,ignore
/// Pipeline::parse("videotestsrc ! videoconvert ! autovideosink")?.run().await?;
/// ```
pub struct AutoVideoSink {
    /// Channel sender for frames
    sender: Option<SyncSender<DisplayFrame>>,
    /// Handle to the display thread
    display_thread: Option<JoinHandle<()>>,
    /// Flag to signal shutdown
    running: Arc<AtomicBool>,
    /// Window title
    title: String,
    /// Expected dimensions (0 = auto-detect from first frame)
    width: u32,
    height: u32,
}

impl AutoVideoSink {
    /// Create a new auto video sink with default settings.
    pub fn new() -> Self {
        Self {
            sender: None,
            display_thread: None,
            running: Arc::new(AtomicBool::new(false)),
            title: "Parallax Video".to_string(),
            width: 0,
            height: 0,
        }
    }

    /// Set the window title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set expected dimensions (optional, auto-detected from first frame).
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Start the display thread.
    fn start_display(&mut self) -> Result<()> {
        if self.display_thread.is_some() {
            return Ok(()); // Already started
        }

        let (sender, receiver) = mpsc::sync_channel::<DisplayFrame>(4);
        let running = Arc::clone(&self.running);
        let title = self.title.clone();
        let initial_width = if self.width > 0 { self.width } else { 640 };
        let initial_height = if self.height > 0 { self.height } else { 480 };

        running.store(true, Ordering::SeqCst);

        let handle = thread::spawn(move || {
            if let Err(e) = run_display_loop(receiver, running, &title, initial_width, initial_height) {
                eprintln!("Display error: {}", e);
            }
        });

        self.sender = Some(sender);
        self.display_thread = Some(handle);

        Ok(())
    }

    /// Stop the display thread.
    fn stop_display(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        
        // Drop sender to unblock receiver
        self.sender.take();

        // Wait for thread to finish
        if let Some(handle) = self.display_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Default for AutoVideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AutoVideoSink {
    fn drop(&mut self) {
        self.stop_display();
    }
}

impl Sink for AutoVideoSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        // Start display thread on first frame
        if self.sender.is_none() {
            self.start_display()?;
        }

        let sender = self.sender.as_ref()
            .ok_or_else(|| Error::Element("Display not started".into()))?;

        // Get frame dimensions from metadata or use configured size
        let (width, height) = if let Some(format) = ctx.metadata().format.as_ref() {
            (format.width(), format.height())
        } else {
            (self.width.max(640), self.height.max(480))
        };

        let frame = DisplayFrame {
            data: ctx.buffer().to_vec(),
            width,
            height,
        };

        // Send frame (blocks if display is slow - natural backpressure)
        sender.send(frame)
            .map_err(|_| Error::Element("Display closed".into()))
    }

    fn name(&self) -> &str {
        "autovideosink"
    }
}

/// Run the winit display loop in the display thread.
fn run_display_loop(
    receiver: Receiver<DisplayFrame>,
    running: Arc<AtomicBool>,
    title: &str,
    initial_width: u32,
    initial_height: u32,
) -> Result<()> {
    use winit::event::{Event, WindowEvent};
    use winit::event_loop::{ControlFlow, EventLoop};
    use winit::window::WindowBuilder;
    use winit::dpi::LogicalSize;

    // Create event loop and window
    let event_loop = EventLoop::new()
        .map_err(|e| Error::Element(format!("Failed to create event loop: {}", e)))?;
    
    let window = WindowBuilder::new()
        .with_title(title)
        .with_inner_size(LogicalSize::new(initial_width, initial_height))
        .build(&event_loop)
        .map_err(|e| Error::Element(format!("Failed to create window: {}", e)))?;

    // Create softbuffer context and surface
    let context = softbuffer::Context::new(&window)
        .map_err(|e| Error::Element(format!("Failed to create softbuffer context: {}", e)))?;
    let mut surface = softbuffer::Surface::new(&context, &window)
        .map_err(|e| Error::Element(format!("Failed to create surface: {}", e)))?;

    // Current frame buffer
    let mut current_frame: Option<DisplayFrame> = None;

    event_loop.run(move |event, elwt| {
        // Check if we should exit
        if !running.load(Ordering::SeqCst) {
            elwt.exit();
            return;
        }

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                running.store(false, Ordering::SeqCst);
                elwt.exit();
            }

            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                // Try to receive a new frame (non-blocking)
                while let Ok(frame) = receiver.try_recv() {
                    current_frame = Some(frame);
                }

                // Render current frame
                if let Some(ref frame) = current_frame {
                    let size = window.inner_size();
                    let width = size.width as usize;
                    let height = size.height as usize;

                    // Resize surface if needed
                    let _ = surface.resize(
                        std::num::NonZeroU32::new(size.width).unwrap(),
                        std::num::NonZeroU32::new(size.height).unwrap(),
                    );

                    if let Ok(mut buffer) = surface.buffer_mut() {
                        // Scale/copy frame to surface
                        blit_frame(&frame, &mut buffer, width, height);
                        let _ = buffer.present();
                    }
                }
            }

            Event::AboutToWait => {
                // Check for new frames periodically
                if let Ok(frame) = receiver.try_recv() {
                    current_frame = Some(frame);
                    window.request_redraw();
                } else {
                    // Poll again after a short delay
                    elwt.set_control_flow(ControlFlow::WaitUntil(
                        std::time::Instant::now() + std::time::Duration::from_millis(16)
                    ));
                }
            }

            _ => {}
        }
    }).map_err(|e| Error::Element(format!("Event loop error: {}", e)))
}

/// Blit an RGBA frame to the softbuffer surface.
fn blit_frame(frame: &DisplayFrame, buffer: &mut [u32], dst_width: usize, dst_height: usize) {
    let src_width = frame.width as usize;
    let src_height = frame.height as usize;

    // Simple nearest-neighbor scaling
    for dst_y in 0..dst_height {
        let src_y = (dst_y * src_height) / dst_height;
        for dst_x in 0..dst_width {
            let src_x = (dst_x * src_width) / dst_width;
            
            let src_idx = (src_y * src_width + src_x) * 4;
            if src_idx + 3 < frame.data.len() {
                let r = frame.data[src_idx] as u32;
                let g = frame.data[src_idx + 1] as u32;
                let b = frame.data[src_idx + 2] as u32;
                // softbuffer expects 0RGB format
                buffer[dst_y * dst_width + dst_x] = (r << 16) | (g << 8) | b;
            }
        }
    }
}
```

### Phase 2: Factory Registration

**File:** `src/pipeline/factory.rs` (modify)

```rust
impl ElementFactory {
    pub fn new() -> Self {
        let mut factory = Self { creators: HashMap::new() };
        
        // ... existing registrations ...

        // Display sinks
        factory.register("autovideosink", create_autovideosink);
        
        factory
    }
}

fn create_autovideosink(props: &HashMap<String, PropertyValue>) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::app::AutoVideoSink;
    
    let mut sink = AutoVideoSink::new();
    
    if let Some(title) = props.get("title").map(|v| v.as_string()) {
        sink = sink.with_title(title);
    }
    if let (Some(w), Some(h)) = (
        props.get("width").and_then(|v| v.as_u64()),
        props.get("height").and_then(|v| v.as_u64()),
    ) {
        sink = sink.with_size(w as u32, h as u32);
    }
    
    Ok(DynAsyncElement::new_box(SinkAdapter::new(sink)))
}
```

### Phase 3: Device Source Registration

**File:** `src/pipeline/factory.rs` (modify)

```rust
// Register v4l2src (feature-gated)
#[cfg(feature = "v4l2")]
factory.register("v4l2src", create_v4l2src);

// Register videoconvert
factory.register("videoconvert", create_videoconvert);

#[cfg(feature = "v4l2")]
fn create_v4l2src(props: &HashMap<String, PropertyValue>) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::device::V4l2Src;
    
    let device = props.get("device")
        .map(|v| v.as_string())
        .unwrap_or_else(|| "/dev/video0".to_string());
    
    let src = V4l2Src::new(&device)?;
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

fn create_videoconvert(_props: &HashMap<String, PropertyValue>) -> Result<Box<DynAsyncElement<'static>>> {
    use crate::elements::transform::VideoConvertElement;
    Ok(DynAsyncElement::new_box(TransformAdapter::new(VideoConvertElement::new())))
}
```

---

## Implementation Summary

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | AutoVideoSink with winit/softbuffer | Medium |
| 2 | Factory registration for autovideosink | Small |
| 3 | Factory registration for v4l2src, videoconvert | Small |

### Dependencies

Add to `Cargo.toml`:
```toml
[dependencies]
winit = "0.29"
softbuffer = "0.4"

[features]
display = ["dep:winit", "dep:softbuffer"]
```

---

## Why This Is Better Than the Previous Plan

| Aspect | Previous Plan | This Plan |
|--------|---------------|-----------|
| Special traits | `DisplaySink` trait needed | No special traits |
| API changes | `run()` behavior changed | No API changes |
| Pipeline execution | Special case for display sinks | Normal execution |
| Main thread requirement | Yes | No (display runs in own thread) |
| Complexity | High | Low |
| GStreamer parity | Different model | Same model |

---

## Testing

```rust
// This should "just work"
#[tokio::main]
async fn main() -> Result<()> {
    Pipeline::parse("v4l2src ! videoconvert ! autovideosink")?.run().await
}
```

---

## Future Enhancements

1. **GPU acceleration**: Use wgpu instead of softbuffer for hardware-accelerated rendering
2. **GstVideoOverlay equivalent**: Allow application to provide a window handle
3. **Multiple displays**: Support for multiple video windows
4. **Fullscreen**: Support for fullscreen mode
5. **OSD**: Overlay text/graphics on video

---

## References

- [GStreamer xvimagesink source](https://github.com/GStreamer/gst-plugins-base/blob/master/sys/xvimage/xvimagesink.c)
- [winit](https://github.com/rust-windowing/winit) - Window handling library
- [softbuffer](https://github.com/rust-windowing/softbuffer) - GPU-less 2D display
- [GstVideoOverlay](https://gstreamer.freedesktop.org/documentation/video/gstvideooverlay.html) - For future window handle support
