//! Video device hotplug monitoring via udev.
//!
//! [`DeviceMonitor`] watches the kernel's `video4linux` subsystem for devices
//! appearing and disappearing (USB webcam plugged in / unplugged) and emits
//! [`DeviceEvent`]s. Discovery stays scan-once via
//! [`enumerate_video_devices`](super::enumerate_video_devices); the monitor is
//! what keeps a catalog fresh afterwards.
//!
//! Requires the `hotplug` feature (udev + v4l2).
//!
//! # Example: catalog refresh loop
//!
//! ```rust,ignore
//! use parallax::elements::device::{DeviceMonitor, DeviceEvent};
//!
//! let mut monitor = DeviceMonitor::new()?;
//! // async consumer (tokio):
//! while let Some(event) = monitor.recv().await {
//!     match event {
//!         DeviceEvent::Added(device) => println!("camera appeared: {}", device.id),
//!         DeviceEvent::Removed { id } => println!("camera removed: {id}"),
//!     }
//! }
//! ```
//!
//! Only nodes with the `VIDEO_CAPTURE` capability produce [`DeviceEvent::Added`];
//! metadata-only nodes (e.g. the second `/dev/video*` node UVC cameras expose)
//! are filtered out. Removal events cannot be capability-checked (the device
//! is already gone), so consumers may see `Removed` for ids they never saw
//! `Added` for — match against your own catalog.

use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::error::{Error, Result};

use super::{CaptureBackend, VideoCaptureDevice};

/// A video device hotplug event.
#[derive(Debug, Clone)]
pub enum DeviceEvent {
    /// A capture-capable video device appeared.
    Added(VideoCaptureDevice),
    /// A video device disappeared.
    Removed {
        /// The device identifier (its `/dev/video*` path).
        id: String,
    },
}

/// Watches udev for `video4linux` hotplug events on a background thread.
///
/// Events arrive on an unbounded channel: consume with the async
/// [`recv`](Self::recv) from tokio tasks, or [`blocking_recv`](Self::blocking_recv) /
/// [`try_recv`](Self::try_recv) from plain threads. Dropping the monitor
/// stops the background thread.
pub struct DeviceMonitor {
    receiver: tokio::sync::mpsc::UnboundedReceiver<DeviceEvent>,
    shutdown: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DeviceMonitor {
    /// Start monitoring `video4linux` hotplug events.
    ///
    /// Fails if the udev monitor socket cannot be created (no udev on the
    /// system, or insufficient permissions).
    pub fn new() -> Result<Self> {
        // The builder is Send; the listening socket is created inside the
        // monitor thread and its startup outcome reported back.
        let builder = udev::MonitorBuilder::new()
            .and_then(|b| b.match_subsystem("video4linux"))
            .map_err(|e| Error::Element(format!("udev monitor setup failed: {e}")))?;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let (startup_tx, startup_rx) =
            std::sync::mpsc::channel::<std::result::Result<(), String>>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = shutdown.clone();

        let thread = std::thread::Builder::new()
            .name("parallax-devmon".into())
            .spawn(move || {
                let socket = match builder.listen() {
                    Ok(socket) => {
                        let _ = startup_tx.send(Ok(()));
                        socket
                    }
                    Err(e) => {
                        let _ = startup_tx.send(Err(e.to_string()));
                        return;
                    }
                };
                monitor_loop(socket, tx, shutdown_thread);
            })
            .map_err(|e| Error::Element(format!("failed to spawn monitor thread: {e}")))?;

        match startup_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => Ok(Self {
                receiver: rx,
                shutdown,
                thread: Some(thread),
            }),
            Ok(Err(e)) => Err(Error::Element(format!("udev listen failed: {e}"))),
            Err(_) => Err(Error::Element("udev monitor startup timed out".into())),
        }
    }

    /// Receive the next event (async).
    pub async fn recv(&mut self) -> Option<DeviceEvent> {
        self.receiver.recv().await
    }

    /// Receive the next event, blocking the current thread.
    ///
    /// Must not be called from within an async runtime.
    pub fn blocking_recv(&mut self) -> Option<DeviceEvent> {
        self.receiver.blocking_recv()
    }

    /// Receive an event without blocking.
    pub fn try_recv(&mut self) -> Option<DeviceEvent> {
        self.receiver.try_recv().ok()
    }
}

impl Drop for DeviceMonitor {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join(); // wakes within one poll interval
        }
    }
}

impl std::fmt::Debug for DeviceMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMonitor").finish_non_exhaustive()
    }
}

/// Poll interval — also the shutdown latency bound.
const POLL_INTERVAL_MS: i32 = 500;

fn monitor_loop(
    socket: udev::MonitorSocket,
    tx: tokio::sync::mpsc::UnboundedSender<DeviceEvent>,
    shutdown: Arc<AtomicBool>,
) {
    let fd = socket.as_raw_fd();
    while !shutdown.load(Ordering::Acquire) {
        let mut pollfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pollfd points to a valid struct for the duration of the call.
        let ready = unsafe { libc::poll(&mut pollfd, 1, POLL_INTERVAL_MS) };
        if ready < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            tracing::warn!("device monitor poll failed: {err}; stopping");
            return;
        }
        if ready == 0 {
            continue; // timeout: re-check shutdown
        }

        for event in socket.iter() {
            let Some(devnode) = event.devnode() else {
                continue;
            };
            let id = devnode.display().to_string();
            match event.event_type() {
                udev::EventType::Add => {
                    if let Some(device) = probe_capture_device(devnode, &id) {
                        tracing::debug!("video device added: {id}");
                        if tx.send(DeviceEvent::Added(device)).is_err() {
                            return; // receiver dropped
                        }
                    } else {
                        tracing::trace!("ignoring non-capture video node {id}");
                    }
                }
                udev::EventType::Remove => {
                    tracing::debug!("video device removed: {id}");
                    if tx.send(DeviceEvent::Removed { id }).is_err() {
                        return;
                    }
                }
                _ => {} // Change/Bind/Unbind: not catalog-relevant
            }
        }
    }
}

/// Probe a freshly-added node; `Some` only for capture-capable devices.
fn probe_capture_device(path: &Path, id: &str) -> Option<VideoCaptureDevice> {
    let device = v4l::Device::with_path(path).ok()?;
    let caps = device.query_caps().ok()?;
    if !caps
        .capabilities
        .contains(v4l::capability::Flags::VIDEO_CAPTURE)
    {
        return None;
    }
    Some(VideoCaptureDevice {
        id: id.to_string(),
        name: caps.card.clone(),
        backend: CaptureBackend::V4l2,
        model: Some(caps.card),
        location: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The monitor starts, idles without events, and shuts down cleanly.
    /// (Actual hotplug needs physical plug/unplug or `modprobe vivid` —
    /// exercised manually, not in CI.)
    #[test]
    fn monitor_starts_and_stops() {
        let mut monitor = match DeviceMonitor::new() {
            Ok(monitor) => monitor,
            Err(e) => {
                // Sandboxed environments may lack udev socket permissions.
                println!("skipping: udev monitor unavailable ({e})");
                return;
            }
        };
        assert!(monitor.try_recv().is_none(), "no events while idle");
        drop(monitor); // must join the thread promptly
    }

    /// Capability probing filters out nodes that aren't capture devices.
    #[test]
    fn probe_rejects_missing_nodes() {
        assert!(probe_capture_device(Path::new("/dev/video-nonexistent"), "x").is_none());
    }
}
