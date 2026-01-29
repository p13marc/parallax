//! PipeWire audio/video capture and playback.
//!
//! PipeWire is the modern multimedia server for Linux, replacing PulseAudio
//! and JACK for audio, and providing video capture (cameras, screen capture).
//!
//! ## Features
//!
//! - Audio capture from microphones
//! - Audio playback to speakers
//! - Video capture from cameras
//! - Screen capture via XDG portal (Wayland)
//!
//! ## Example
//!
//! ```rust,ignore
//! use parallax::elements::device::pipewire::{PipeWireSrc, PipeWireTarget};
//!
//! // Audio capture
//! let mic = PipeWireSrc::audio(None)?;  // Default microphone
//!
//! // Camera capture
//! let camera = PipeWireSrc::video(PipeWireTarget::DefaultCamera)?;
//!
//! // Screen capture (requires user permission via portal)
//! let screen = PipeWireSrc::screen_capture().await?;
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use kanal::{Receiver, Sender, bounded};
use pipewire as pw;

use crate::element::{
    Affinity, AsyncSink, AsyncSource, ExecutionHints, ProduceContext, ProduceResult,
};
use crate::error::Result;

use super::DeviceError;

/// Check if PipeWire is available on this system.
pub fn is_available() -> bool {
    // Initialize PipeWire - this always succeeds in pipewire 0.8+
    pw::init();
    // Try to create a main loop to verify PipeWire is available
    pw::main_loop::MainLoop::new(None).is_ok()
}

/// PipeWire capture target.
#[derive(Debug, Clone)]
pub enum PipeWireTarget {
    /// Default camera device.
    DefaultCamera,
    /// Specific camera by name or serial.
    Camera(String),
    /// Default audio input (microphone).
    DefaultMicrophone,
    /// Specific audio input by name.
    AudioInput(String),
    /// Screen capture (requires portal).
    Screen,
    /// Specific window (requires portal).
    Window(u32),
}

/// PipeWire node information.
#[derive(Debug, Clone)]
pub struct PipeWireNodeInfo {
    /// Node ID.
    pub id: u32,
    /// Node name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Media class (e.g., "Audio/Source", "Video/Source").
    pub media_class: String,
    /// Whether this is a capture device.
    pub is_capture: bool,
    /// Whether this is a playback device.
    pub is_playback: bool,
}

/// Enumerate audio nodes available via PipeWire.
pub fn enumerate_audio_nodes() -> Result<Vec<PipeWireNodeInfo>> {
    // Initialize PipeWire
    pw::init();

    // Create main loop and context in a thread
    // PipeWire requires its own main loop
    let (tx, rx) = bounded::<Vec<PipeWireNodeInfo>>(1);

    thread::spawn(move || {
        let main_loop = match pw::main_loop::MainLoop::new(None) {
            Ok(ml) => ml,
            Err(_) => {
                let _ = tx.send(Vec::new());
                return;
            }
        };

        let context = match pw::context::Context::new(&main_loop) {
            Ok(ctx) => ctx,
            Err(_) => {
                let _ = tx.send(Vec::new());
                return;
            }
        };

        let core = match context.connect(None) {
            Ok(c) => c,
            Err(_) => {
                let _ = tx.send(Vec::new());
                return;
            }
        };

        let registry = match core.get_registry() {
            Ok(r) => r,
            Err(_) => {
                let _ = tx.send(Vec::new());
                return;
            }
        };
        let collected_nodes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let nodes_clone = collected_nodes.clone();
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();

        let _listener = registry
            .add_listener_local()
            .global(move |global| {
                if global.type_ == pw::types::ObjectType::Node {
                    if let Some(props) = &global.props {
                        let name = props.get("node.name").unwrap_or("unknown").to_string();
                        let description =
                            props.get("node.description").unwrap_or(&name).to_string();
                        let media_class = props.get("media.class").unwrap_or("").to_string();

                        // Filter to audio nodes
                        if media_class.contains("Audio") {
                            let is_capture =
                                media_class.contains("Source") || media_class.contains("Input");
                            let is_playback =
                                media_class.contains("Sink") || media_class.contains("Output");

                            if let Ok(mut nodes) = nodes_clone.lock() {
                                nodes.push(PipeWireNodeInfo {
                                    id: global.id,
                                    name,
                                    description,
                                    media_class,
                                    is_capture,
                                    is_playback,
                                });
                            }
                        }
                    }
                }
            })
            .register();

        // Sync to ensure we've received all objects
        let _pending = core.sync(0);

        // Run main loop briefly to collect objects
        for _ in 0..10 {
            main_loop
                .loop_()
                .iterate(std::time::Duration::from_millis(10));
            if done_clone.load(Ordering::SeqCst) {
                break;
            }
        }

        let result = collected_nodes
            .lock()
            .map(|n| n.clone())
            .unwrap_or_default();
        let _ = tx.send(result);
    });

    // Wait for enumeration with timeout
    let nodes = rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .unwrap_or_default();

    Ok(nodes)
}

/// PipeWire audio/video capture source.
pub struct PipeWireSrc {
    /// Receiver for captured buffers.
    receiver: Receiver<Vec<u8>>,
    /// Sender to request shutdown.
    _shutdown: Sender<()>,
    /// Thread handle.
    _thread: Option<thread::JoinHandle<()>>,
    /// Target being captured.
    target: PipeWireTarget,
}

impl PipeWireSrc {
    /// Create an audio capture source.
    ///
    /// # Arguments
    ///
    /// * `device` - Device name, or None for default microphone.
    pub fn audio(device: Option<&str>) -> Result<Self> {
        let target = match device {
            Some(name) => PipeWireTarget::AudioInput(name.to_string()),
            None => PipeWireTarget::DefaultMicrophone,
        };

        Self::new(target)
    }

    /// Create a video capture source.
    ///
    /// # Arguments
    ///
    /// * `target` - The capture target (camera, screen, etc.).
    pub fn video(target: PipeWireTarget) -> Result<Self> {
        Self::new(target)
    }

    /// Create a screen capture source.
    ///
    /// This will request permission via the XDG portal on Wayland.
    #[cfg(feature = "screen-capture")]
    pub async fn screen_capture() -> Result<Self> {
        // Use ashpd to request screen capture permission
        use ashpd::desktop::PersistMode;
        use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};

        let proxy = Screencast::new()
            .await
            .map_err(|e| DeviceError::PipeWire(e.to_string()))?;

        let session = proxy
            .create_session()
            .await
            .map_err(|e| DeviceError::PipeWire(e.to_string()))?;

        proxy
            .select_sources(
                &session,
                CursorMode::Embedded,
                SourceType::Monitor | SourceType::Window,
                true, // multiple
                None, // restore_token
                PersistMode::DoNot,
            )
            .await
            .map_err(|e| DeviceError::PipeWire(e.to_string()))?;

        let response = proxy
            .start(&session, None)
            .await
            .map_err(|e| DeviceError::PipeWire(e.to_string()))?
            .response()
            .map_err(|_| DeviceError::PortalDenied)?;

        let streams = response.streams();
        if streams.is_empty() {
            return Err(DeviceError::PortalDenied.into());
        }

        let node_id = streams[0].pipe_wire_node_id();
        Self::new_with_node_id(node_id)
    }

    /// Create a source for a specific PipeWire node ID.
    fn new_with_node_id(node_id: u32) -> Result<Self> {
        Self::new(PipeWireTarget::Camera(format!("node:{}", node_id)))
    }

    /// Internal constructor.
    fn new(target: PipeWireTarget) -> Result<Self> {
        pw::init();

        let (buffer_tx, buffer_rx) = bounded::<Vec<u8>>(16);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);

        let target_clone = target.clone();
        let thread = thread::spawn(move || {
            Self::capture_thread(target_clone, buffer_tx, shutdown_rx);
        });

        Ok(Self {
            receiver: buffer_rx,
            _shutdown: shutdown_tx,
            _thread: Some(thread),
            target,
        })
    }

    /// Main capture thread.
    fn capture_thread(
        target: PipeWireTarget,
        buffer_tx: Sender<Vec<u8>>,
        shutdown_rx: Receiver<()>,
    ) {
        let main_loop = match pw::main_loop::MainLoop::new(None) {
            Ok(ml) => ml,
            Err(e) => {
                tracing::error!("Failed to create PipeWire main loop: {}", e);
                return;
            }
        };

        let context = match pw::context::Context::new(&main_loop) {
            Ok(ctx) => ctx,
            Err(e) => {
                tracing::error!("Failed to create PipeWire context: {}", e);
                return;
            }
        };

        let core = match context.connect(None) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Failed to connect to PipeWire: {}", e);
                return;
            }
        };

        // Determine stream properties based on target
        let (media_type, media_category) = match &target {
            PipeWireTarget::DefaultMicrophone | PipeWireTarget::AudioInput(_) => {
                ("Audio", "Capture")
            }
            PipeWireTarget::DefaultCamera | PipeWireTarget::Camera(_) => ("Video", "Capture"),
            PipeWireTarget::Screen | PipeWireTarget::Window(_) => ("Video", "Capture"),
        };

        let props = pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => media_type,
            *pw::keys::MEDIA_CATEGORY => media_category,
            *pw::keys::MEDIA_ROLE => "Production",
        };

        let stream = match pw::stream::Stream::new(&core, "parallax-capture", props) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to create PipeWire stream: {}", e);
                return;
            }
        };

        let buffer_tx_clone = buffer_tx.clone();

        let _listener = stream
            .add_local_listener_with_user_data(())
            .process(move |stream, _| {
                // Get buffer from stream
                if let Some(mut pw_buffer) = stream.dequeue_buffer() {
                    if let Some(data) = pw_buffer.datas_mut().first_mut() {
                        let chunk = data.chunk();
                        let size = chunk.size() as usize;
                        if size > 0 {
                            // Copy data to a Vec for sending
                            if let Some(slice) = data.data() {
                                let bytes = slice[..size].to_vec();
                                let _ = buffer_tx_clone.try_send(bytes);
                            }
                        }
                    }
                }
            })
            .register();

        // Connect stream to default device
        // Empty params - accept any format
        let params: &mut [&pw::spa::pod::Pod] = &mut [];

        if let Err(e) = stream.connect(
            pw::spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            params,
        ) {
            tracing::error!("Failed to connect PipeWire stream: {:?}", e);
            return;
        }

        // Run main loop until shutdown
        loop {
            // Check for shutdown
            if shutdown_rx.try_recv().is_ok() {
                break;
            }

            main_loop
                .loop_()
                .iterate(std::time::Duration::from_millis(10));
        }
    }
}

impl AsyncSource for PipeWireSrc {
    async fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        match self.receiver.as_async().recv().await {
            Ok(data) => {
                let len = data.len();
                if len > 0 {
                    ctx.output()[..len].copy_from_slice(&data);
                    Ok(ProduceResult::Produced(len))
                } else {
                    Ok(ProduceResult::WouldBlock)
                }
            }
            Err(_) => Ok(ProduceResult::Eos),
        }
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        match &self.target {
            PipeWireTarget::DefaultMicrophone | PipeWireTarget::AudioInput(_) => {
                Some(4096) // Audio buffer
            }
            _ => Some(1920 * 1080 * 3), // Video buffer (1080p RGB)
        }
    }

    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    fn is_rt_safe(&self) -> bool {
        false
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

/// PipeWire audio playback sink.
pub struct PipeWireSink {
    /// Sender for buffers to play.
    sender: Sender<Vec<u8>>,
    /// Sender to request shutdown.
    _shutdown: Sender<()>,
    /// Thread handle.
    _thread: Option<thread::JoinHandle<()>>,
}

impl PipeWireSink {
    /// Create an audio playback sink.
    ///
    /// # Arguments
    ///
    /// * `device` - Device name, or None for default speaker.
    pub fn audio(device: Option<&str>) -> Result<Self> {
        pw::init();

        let (buffer_tx, buffer_rx) = bounded::<Vec<u8>>(16);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);

        let device = device.map(|s| s.to_string());
        let thread = thread::spawn(move || {
            Self::playback_thread(device, buffer_rx, shutdown_rx);
        });

        Ok(Self {
            sender: buffer_tx,
            _shutdown: shutdown_tx,
            _thread: Some(thread),
        })
    }

    /// Main playback thread.
    fn playback_thread(
        _device: Option<String>,
        buffer_rx: Receiver<Vec<u8>>,
        shutdown_rx: Receiver<()>,
    ) {
        let main_loop = match pw::main_loop::MainLoop::new(None) {
            Ok(ml) => ml,
            Err(e) => {
                tracing::error!("Failed to create PipeWire main loop: {}", e);
                return;
            }
        };

        let context = match pw::context::Context::new(&main_loop) {
            Ok(ctx) => ctx,
            Err(e) => {
                tracing::error!("Failed to create PipeWire context: {}", e);
                return;
            }
        };

        let core = match context.connect(None) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Failed to connect to PipeWire: {}", e);
                return;
            }
        };

        let props = pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::MEDIA_ROLE => "Music",
        };

        let stream = match pw::stream::Stream::new(&core, "parallax-playback", props) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to create PipeWire stream: {}", e);
                return;
            }
        };

        let buffer_rx_clone = buffer_rx.clone();

        let _listener = stream
            .add_local_listener_with_user_data(())
            .process(move |stream, _| {
                // Get buffer from stream to fill
                if let Some(mut pw_buffer) = stream.dequeue_buffer() {
                    if let Some(data) = pw_buffer.datas_mut().first_mut() {
                        // Check if we have data to play
                        if let Ok(Some(audio_data)) = buffer_rx_clone.try_recv() {
                            if let Some(slice) = data.data() {
                                let copy_len = audio_data.len().min(slice.len());
                                slice[..copy_len].copy_from_slice(&audio_data[..copy_len]);
                                let chunk = data.chunk_mut();
                                *chunk.size_mut() = copy_len as u32;
                            }
                        }
                    }
                }
            })
            .register();

        // Connect stream
        let params: &mut [&pw::spa::pod::Pod] = &mut [];
        if let Err(e) = stream.connect(
            pw::spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            params,
        ) {
            tracing::error!("Failed to connect PipeWire stream: {:?}", e);
            return;
        }

        // Run main loop until shutdown
        loop {
            if shutdown_rx.try_recv().is_ok() {
                break;
            }
            main_loop
                .loop_()
                .iterate(std::time::Duration::from_millis(10));
        }
    }
}

impl AsyncSink for PipeWireSink {
    async fn consume(&mut self, ctx: &crate::element::ConsumeContext<'_>) -> Result<()> {
        let data = ctx.input().to_vec();
        self.sender
            .as_async()
            .send(data)
            .await
            .map_err(|e| DeviceError::PipeWire(e.to_string()))?;
        Ok(())
    }

    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    fn is_rt_safe(&self) -> bool {
        false
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        // Just verify this doesn't panic
        let available = is_available();
        println!("PipeWire available: {}", available);
    }

    #[test]
    fn test_enumerate_audio_nodes() {
        if !is_available() {
            println!("PipeWire not available, skipping");
            return;
        }

        match enumerate_audio_nodes() {
            Ok(nodes) => {
                println!("Found {} audio nodes:", nodes.len());
                for node in &nodes {
                    println!(
                        "  [{}] {} - {} (capture: {}, playback: {})",
                        node.id, node.name, node.media_class, node.is_capture, node.is_playback
                    );
                }
            }
            Err(e) => {
                println!("Failed to enumerate: {}", e);
            }
        }
    }
}
