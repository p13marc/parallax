# Plan: Proper Iced Pipeline Integration

## Problem Statement

Currently, Parallax cannot express a simple pipeline like:
```rust
Pipeline::parse("v4l2src ! videoconvert ! icedsink")?.run()?;
```

The current `IcedVideoSink` requires manual threading because:
1. `iced::application(...).run()` blocks the thread to run the event loop
2. The pipeline executor runs elements as Tokio tasks
3. These two execution models conflict

Additionally, there are broader infrastructure gaps:
- **No auto-negotiation**: Converters must be manually inserted
- **Missing factory registrations**: Device elements like `v4l2src` aren't registered

## Root Cause Analysis

### How GStreamer Solves This

GStreamer video sinks (like `gtksink`, `gtkglsink`) work differently:
1. The **pipeline runs in background threads** managed by GStreamer
2. The **GUI runs its own event loop** on the main thread
3. Video sinks use **GstVideoOverlay** interface to render to a GUI widget
4. Communication happens via **GLib message bus** for thread-safe updates

Key insight: **GStreamer video sinks don't control the event loop** - they just receive frames and render them to a widget that the GUI toolkit manages.

### How Iced Works

Iced follows The Elm Architecture:
1. **State**: Application state
2. **Messages**: Events that trigger state changes
3. **Update**: Handle messages and update state
4. **View**: Render state to widgets

For async work, Iced provides:
- `Task::run` - Run a future and get a message when done
- `Subscription::run` - Create a stream of messages from async work
- `iced::stream::channel` - Bridge between futures and streams

With `features = ["tokio"]`, Iced uses Tokio as its executor and can run alongside other Tokio tasks.

### The Real Issue

The current `IcedVideoSink` design is **inverted**:
- It expects to be called by the pipeline executor (push model)
- But Iced's architecture requires the application to **pull** data via subscriptions

## Solution: Flip the Model

Instead of a "sink" that receives frames, we need:
1. A **channel** that the pipeline writes frames to
2. An **Iced subscription** that reads from this channel
3. An **Iced application** that displays frames

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Iced Application                              │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ Subscription::run(frame_receiver)                            ││
│  │   └── Receives: Message::Frame(rgba_data)                    ││
│  └──────────────────────────────────────────────────────────────┘│
│                           ▲                                      │
│                           │ mpsc channel                         │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │ fn subscription(&self) -> Subscription<Message>              ││
│  │   └── Returns the frame subscription                         ││
│  └──────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ frames via channel
┌─────────────────────────────────────────────────────────────────┐
│               Parallax Pipeline (Tokio tasks)                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐           │
│  │ v4l2src  │───▶│ convert  │───▶│ ChannelSink      │           │
│  └──────────┘    └──────────┘    │ (sends to mpsc)  │           │
│                                  └──────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Element Factory Registration

**Goal:** Enable `Pipeline::parse("v4l2src ! ...")` to work.

Currently, the factory only registers: `nullsource`, `nullsink`, `passthrough`, `tee`, `filesrc`, `filesink`.

#### 1.1 Register Device Elements

**File:** `src/pipeline/factory.rs`

```rust
impl ElementFactory {
    pub fn new() -> Self {
        let mut factory = Self { /* ... */ };

        // Existing
        factory.register("nullsource", create_nullsource);
        factory.register("nullsink", create_nullsink);
        // ...

        // NEW: Device sources (feature-gated)
        #[cfg(feature = "v4l2")]
        factory.register("v4l2src", create_v4l2src);
        
        #[cfg(feature = "pipewire")]
        factory.register("pipewiresrc", create_pipewiresrc);
        
        #[cfg(feature = "libcamera")]
        factory.register("libcamerasrc", create_libcamerasrc);

        // NEW: Video elements
        factory.register("videoconvert", create_videoconvert);
        factory.register("videoscale", create_videoscale);

        // NEW: Display sinks
        factory.register("channelsink", create_channelsink);
        #[cfg(feature = "iced-sink")]
        factory.register("icedsink", create_icedsink);  // Alias for channelsink

        factory
    }
}
```

#### 1.2 Implement Constructors

```rust
#[cfg(feature = "v4l2")]
fn create_v4l2src(props: &HashMap<String, PropertyValue>) -> Result<Box<DynAsyncElement<'static>>> {
    let device = props
        .get("device")
        .map(|v| v.as_string())
        .unwrap_or_else(|| "/dev/video0".to_string());
    
    let width = props.get("width").and_then(|v| v.as_u64()).map(|v| v as u32);
    let height = props.get("height").and_then(|v| v.as_u64()).map(|v| v as u32);
    
    let mut config = V4l2Config::default();
    if let Some(w) = width { config.width = w; }
    if let Some(h) = height { config.height = h; }
    
    let src = V4l2Src::with_config(&device, config)?;
    Ok(DynAsyncElement::new_box(SourceAdapter::new(src)))
}

fn create_videoconvert(props: &HashMap<String, PropertyValue>) -> Result<Box<DynAsyncElement<'static>>> {
    // VideoConvert element that auto-negotiates format
    let convert = VideoConvertElement::new();
    Ok(DynAsyncElement::new_box(ElementAdapter::new(convert)))
}
```

---

### Phase 2: Auto-Negotiation and Converter Insertion

**Goal:** Pipeline automatically inserts `videoconvert` when formats don't match.

This builds on the existing caps negotiation infrastructure in `docs/PLAN_CAPS_NEGOTIATION.md`.

#### 2.1 Elements Declare Their Caps

Each element must declare what formats it produces/accepts:

```rust
impl Source for V4l2Src {
    fn output_caps(&self) -> Caps {
        // V4L2 typically outputs YUYV, MJPG, etc.
        Caps::video_raw(VideoFormatCaps {
            width: CapsValue::Fixed(self.width),
            height: CapsValue::Fixed(self.height),
            pixel_format: CapsValue::Fixed(self.pixel_format()), // e.g., YUYV
            framerate: CapsValue::Any,
        })
    }
}

impl Sink for ChannelSink {
    fn input_caps(&self) -> Caps {
        // Display sinks require RGBA
        Caps::video_raw(VideoFormatCaps {
            width: CapsValue::Fixed(self.width),
            height: CapsValue::Fixed(self.height),
            pixel_format: CapsValue::Fixed(PixelFormat::Rgba),
            framerate: CapsValue::Any,
        })
    }
}
```

#### 2.2 Converter Registry

**File:** `src/pipeline/converter_registry.rs`

```rust
/// Registry of format converters
pub struct ConverterRegistry {
    /// Map: (from_format, to_format) -> converter factory
    converters: HashMap<(MediaType, MediaType), ConverterFactory>,
}

impl ConverterRegistry {
    pub fn new() -> Self {
        let mut registry = Self { converters: HashMap::new() };
        
        // Register video format converters
        registry.register_video_converter(
            |from, to| VideoConvertElement::can_convert(from, to),
            |from, to| Box::new(VideoConvertElement::new_for(from, to)),
        );
        
        // Register video scalers
        registry.register_video_scaler(
            |from, to| VideoScaleElement::can_scale(from, to),
            |from, to| Box::new(VideoScaleElement::new_for(from, to)),
        );
        
        registry
    }
    
    /// Find a converter (or chain) from src to dst format
    pub fn find_path(&self, src: &Caps, dst: &Caps) -> Option<Vec<Box<dyn Element>>> {
        // Use Dijkstra or BFS to find shortest conversion path
        // Returns empty vec if formats are compatible
        // Returns None if no conversion path exists
    }
}
```

#### 2.3 Auto-Insert Converters During Pipeline Construction

**File:** `src/pipeline/graph.rs`

```rust
impl Pipeline {
    /// Link elements with automatic converter insertion
    pub fn link_with_negotiation(&mut self, from: NodeId, to: NodeId) -> Result<()> {
        let from_caps = self.get_node(from)?.output_caps();
        let to_caps = self.get_node(to)?.input_caps();
        
        // Check if formats are compatible
        if from_caps.intersects(&to_caps) {
            // Direct link
            return self.link(from, to);
        }
        
        // Find converter path
        let converters = self.converter_registry.find_path(&from_caps, &to_caps)
            .ok_or_else(|| Error::NegotiationFailed {
                explanation: NegotiationErrorExplanation {
                    path: vec![
                        self.get_node(from)?.name().to_string(),
                        self.get_node(to)?.name().to_string(),
                    ],
                    upstream_format: from_caps.clone(),
                    downstream_format: to_caps.clone(),
                    suggestions: vec![
                        "No converter available for this format pair".to_string(),
                    ],
                },
            })?;
        
        // Insert converters
        let mut current = from;
        for (i, converter) in converters.into_iter().enumerate() {
            let name = format!("__auto_convert_{}_{}", from, i);
            let node_id = self.add_node(&name, converter);
            self.link(current, node_id)?;
            current = node_id;
        }
        self.link(current, to)?;
        
        Ok(())
    }
}
```

#### 2.4 Pipeline::parse Uses Auto-Negotiation

```rust
impl Pipeline {
    pub fn parse(description: &str) -> Result<Self> {
        let parsed = parser::parse(description)?;
        let mut pipeline = Pipeline::new();
        
        // Create elements
        let mut node_ids = Vec::new();
        for element in &parsed.elements {
            let elem = pipeline.factory.create(element)?;
            let id = pipeline.add_node(&element.name, elem);
            node_ids.push(id);
        }
        
        // Link with auto-negotiation
        for window in node_ids.windows(2) {
            pipeline.link_with_negotiation(window[0], window[1])?;
        }
        
        Ok(pipeline)
    }
}
```

---

### Phase 3: Core Iced Integration Components

#### 3.1 FrameData Struct

**File:** `src/elements/app/channel_sink.rs`

```rust
/// Frame data sent through the channel
#[derive(Debug, Clone)]
pub struct FrameData {
    /// RGBA pixel data
    pub pixels: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height  
    pub height: u32,
    /// Presentation timestamp
    pub pts: ClockTime,
    /// Sequence number
    pub sequence: u64,
}
```

#### 3.2 ChannelSink Element

```rust
use tokio::sync::mpsc;

/// A sink that sends frames to an mpsc channel
pub struct ChannelSink {
    sender: mpsc::Sender<FrameData>,
    width: u32,
    height: u32,
    name: String,
}

impl ChannelSink {
    pub fn new(sender: mpsc::Sender<FrameData>, width: u32, height: u32) -> Self {
        Self {
            sender,
            width,
            height,
            name: "channelsink".to_string(),
        }
    }
}

impl AsyncSink for ChannelSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let frame = FrameData {
            pixels: ctx.input().to_vec(),
            width: self.width,
            height: self.height,
            pts: ctx.metadata().pts,
            sequence: ctx.metadata().sequence,
        };
        
        self.sender.send(frame).await
            .map_err(|_| Error::Element("receiver dropped".into()))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn input_caps(&self) -> Caps {
        Caps::video_raw(VideoFormatCaps {
            width: CapsValue::Fixed(self.width),
            height: CapsValue::Fixed(self.height),
            pixel_format: CapsValue::Fixed(PixelFormat::Rgba),
            framerate: CapsValue::Any,
        })
    }
}
```

#### 3.3 VideoDisplay Component

**File:** `src/elements/app/video_display.rs`

```rust
use iced::{widget::image, Element, Subscription};
use iced::stream;
use tokio::sync::mpsc;

/// Iced component for displaying video frames
pub struct VideoDisplay {
    /// Channel receiver (wrapped for sharing with subscription)
    receiver: Arc<Mutex<mpsc::Receiver<FrameData>>>,
    /// Current frame as Iced image handle
    current_frame: Option<image::Handle>,
    /// Dimensions
    width: u32,
    height: u32,
    /// Stats
    frames_displayed: u64,
}

/// Messages from the video subscription
#[derive(Debug, Clone)]
pub enum VideoMessage {
    /// New frame received
    NewFrame(FrameData),
    /// Channel closed (pipeline stopped)
    Closed,
}

impl VideoDisplay {
    /// Create a new video display and return the sender for the pipeline
    pub fn new(width: u32, height: u32) -> (Self, mpsc::Sender<FrameData>) {
        let (tx, rx) = mpsc::channel(4);  // Small buffer for backpressure
        
        let display = Self {
            receiver: Arc::new(Mutex::new(rx)),
            current_frame: None,
            width,
            height,
            frames_displayed: 0,
        };
        
        (display, tx)
    }
    
    /// Handle a video message
    pub fn update(&mut self, message: VideoMessage) {
        match message {
            VideoMessage::NewFrame(frame) => {
                self.current_frame = Some(image::Handle::from_rgba(
                    frame.width,
                    frame.height,
                    frame.pixels,
                ));
                self.frames_displayed += 1;
            }
            VideoMessage::Closed => {
                // Pipeline stopped
            }
        }
    }
    
    /// Get the subscription for receiving frames
    pub fn subscription(&self) -> Subscription<VideoMessage> {
        let receiver = Arc::clone(&self.receiver);
        
        Subscription::run_with_id(
            std::any::TypeId::of::<Self>(),
            stream::channel(4, move |mut output| {
                let receiver = Arc::clone(&receiver);
                async move {
                    loop {
                        let frame = {
                            let mut rx = receiver.lock().unwrap();
                            rx.recv().await
                        };
                        
                        match frame {
                            Some(f) => {
                                let _ = output.send(VideoMessage::NewFrame(f)).await;
                            }
                            None => {
                                let _ = output.send(VideoMessage::Closed).await;
                                break;
                            }
                        }
                    }
                }
            })
        )
    }
    
    /// Render as an Iced widget
    pub fn view(&self) -> Element<'_, VideoMessage> {
        use iced::widget::{container, text};
        use iced::Length;
        
        if let Some(handle) = &self.current_frame {
            image::Image::new(handle.clone())
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            container(text("Waiting for video..."))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into()
        }
    }
}
```

---

### Phase 4: High-Level API

#### 4.1 Pipeline Builder with Display

**File:** `src/elements/app/mod.rs`

```rust
/// Create a video pipeline with an Iced display
/// 
/// # Example
/// 
/// ```rust,ignore
/// let (pipeline, display) = VideoPipeline::new()
///     .source("v4l2src device=/dev/video0")
///     .display(640, 480)
///     .build()?;
/// 
/// // Start pipeline
/// let handle = pipeline.start()?;
/// 
/// // Run Iced app with the display
/// MyApp::run_with_video(display)?;
/// ```
pub struct VideoPipeline {
    source_desc: Option<String>,
    transforms: Vec<String>,
    width: u32,
    height: u32,
}

impl VideoPipeline {
    pub fn new() -> Self {
        Self {
            source_desc: None,
            transforms: Vec::new(),
            width: 640,
            height: 480,
        }
    }
    
    pub fn source(mut self, desc: &str) -> Self {
        self.source_desc = Some(desc.to_string());
        self
    }
    
    pub fn transform(mut self, desc: &str) -> Self {
        self.transforms.push(desc.to_string());
        self
    }
    
    pub fn display(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    
    pub fn build(self) -> Result<(Pipeline, VideoDisplay)> {
        let (display, sender) = VideoDisplay::new(self.width, self.height);
        
        // Build pipeline string
        let mut parts = Vec::new();
        if let Some(src) = &self.source_desc {
            parts.push(src.clone());
        }
        parts.extend(self.transforms.clone());
        parts.push("channelsink".to_string());
        
        let pipeline_str = parts.join(" ! ");
        let mut pipeline = Pipeline::parse(&pipeline_str)?;
        
        // Replace the channelsink with our configured one
        let sink_id = pipeline.find_sink()?;
        let channel_sink = ChannelSink::new(sender, self.width, self.height);
        pipeline.replace_element(sink_id, channel_sink)?;
        
        Ok((pipeline, display))
    }
}
```

#### 4.2 Simplified API

```rust
/// One-liner to create video pipeline and display
pub fn video_pipeline(
    pipeline_desc: &str,
    width: u32,
    height: u32,
) -> Result<(Pipeline, VideoDisplay)> {
    let (display, sender) = VideoDisplay::new(width, height);
    
    // Parse and modify pipeline
    let mut pipeline = Pipeline::parse(pipeline_desc)?;
    
    // Find and configure the display sink
    if let Some(sink_id) = pipeline.find_node_by_name("icedsink") {
        let channel_sink = ChannelSink::new(sender, width, height);
        pipeline.replace_element(sink_id, channel_sink)?;
    } else if let Some(sink_id) = pipeline.find_node_by_name("channelsink") {
        let channel_sink = ChannelSink::new(sender, width, height);
        pipeline.replace_element(sink_id, channel_sink)?;
    } else {
        return Err(Error::Config("Pipeline must end with 'icedsink' or 'channelsink'".into()));
    }
    
    Ok((pipeline, display))
}
```

---

### Phase 5: Complete Example

**File:** `examples/24_v4l2_iced_simple.rs`

```rust
//! Simple V4L2 to Iced display using the new pipeline API.
//!
//! Run with: `cargo run --example 24_v4l2_iced_simple --features "v4l2,iced-sink"`

use iced::{Element, Subscription, Task};
use parallax::elements::app::{video_pipeline, VideoDisplay, VideoMessage};
use parallax::pipeline::PipelineHandle;

fn main() -> iced::Result {
    iced::application("V4L2 Camera", App::update, App::view)
        .subscription(App::subscription)
        .run_with(App::new)
}

struct App {
    video: VideoDisplay,
    pipeline_handle: Option<PipelineHandle>,
}

#[derive(Debug, Clone)]
enum Message {
    Video(VideoMessage),
    PipelineReady(PipelineHandle),
    PipelineError(String),
}

impl App {
    fn new() -> (Self, Task<Message>) {
        // Create pipeline and display
        let result = video_pipeline(
            "v4l2src ! videoconvert ! icedsink",
            640, 480
        );
        
        match result {
            Ok((mut pipeline, display)) => {
                // Start pipeline
                match pipeline.start() {
                    Ok(handle) => {
                        (
                            Self {
                                video: display,
                                pipeline_handle: Some(handle),
                            },
                            Task::none()
                        )
                    }
                    Err(e) => {
                        eprintln!("Failed to start pipeline: {}", e);
                        (
                            Self {
                                video: display,
                                pipeline_handle: None,
                            },
                            Task::none()
                        )
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to create pipeline: {}", e);
                // Create dummy display
                let (display, _) = VideoDisplay::new(640, 480);
                (
                    Self {
                        video: display,
                        pipeline_handle: None,
                    },
                    Task::none()
                )
            }
        }
    }
    
    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Video(video_msg) => {
                self.video.update(video_msg);
            }
            Message::PipelineReady(handle) => {
                self.pipeline_handle = Some(handle);
            }
            Message::PipelineError(err) => {
                eprintln!("Pipeline error: {}", err);
            }
        }
        Task::none()
    }
    
    fn view(&self) -> Element<'_, Message> {
        self.video.view().map(Message::Video)
    }
    
    fn subscription(&self) -> Subscription<Message> {
        self.video.subscription().map(Message::Video)
    }
}
```

---

## Implementation Phases Summary

| Phase | Description | Effort |
|-------|-------------|--------|
| **1** | Factory registration for device elements | Small |
| **2** | Auto-negotiation and converter insertion | Medium |
| **3** | Core Iced integration (ChannelSink, VideoDisplay) | Medium |
| **4** | High-level API (VideoPipeline builder) | Small |
| **5** | Examples and documentation | Small |

### Dependencies

- Phase 2 depends on caps negotiation infrastructure from `PLAN_CAPS_NEGOTIATION.md`
- Phase 3 depends on Phase 1 (factory registration)
- Phase 4 depends on Phase 3

### Feature Flags

```toml
[features]
# Device capture
v4l2 = ["dep:v4l"]
pipewire = ["dep:pipewire"]
libcamera = ["dep:libcamera"]

# Display
iced-sink = ["dep:iced"]

# Video processing
video-convert = []  # Pure Rust converters (always available)
```

---

## Why This Works

1. **No blocking**: Pipeline runs in Tokio tasks, Iced runs its event loop, channel bridges them
2. **Standard Iced pattern**: Uses `Subscription::run` which is the idiomatic way to receive external events
3. **Decoupled**: Pipeline and GUI are independent, connected only by channel
4. **Works with `#[tokio::main]`**: Because Iced with `features = ["tokio"]` uses Tokio
5. **Auto-negotiation**: Converters inserted automatically when formats don't match
6. **GStreamer-like syntax**: `v4l2src ! videoconvert ! icedsink`

## Comparison with Current Approach

| Aspect | Current | New |
|--------|---------|-----|
| Threading | Manual, error-prone | Automatic via channel |
| Pipeline syntax | Not possible | `v4l2src ! convert ! icedsink` |
| Iced integration | Requires `handle.run()` | Standard subscription pattern |
| Format conversion | Manual in example | Auto via `videoconvert` element |
| Device release | Had bugs | Pipeline owns lifecycle |
| Factory | Missing device elements | All elements registered |
| Caps negotiation | Manual | Automatic with error messages |

## References

- [Iced Subscriptions](https://docs.rs/iced/latest/iced/struct.Subscription.html)
- [iced::stream::channel](https://docs.rs/iced/latest/iced/stream/fn.channel.html)
- [GStreamer GUI Integration Tutorial](https://gstreamer.freedesktop.org/documentation/tutorials/basic/toolkit-integration.html)
- [iced_video_player](https://github.com/jazzfool/iced_video_player) - Uses GStreamer internally
- [PLAN_CAPS_NEGOTIATION.md](./PLAN_CAPS_NEGOTIATION.md) - Caps negotiation infrastructure
