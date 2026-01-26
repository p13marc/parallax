//! Pipeline event system for async event handling.
//!
//! Events are emitted by the pipeline during execution and can be
//! received asynchronously by the caller.

use crate::element::PadDirection;
use crate::format::Caps;
use std::fmt;
use tokio::sync::broadcast;

/// Events emitted by the pipeline during execution.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    /// Pipeline state has changed.
    StateChanged {
        /// Previous state.
        from: super::PipelineState,
        /// New state.
        to: super::PipelineState,
    },

    /// End of stream reached (all sources exhausted).
    Eos,

    /// An error occurred in the pipeline.
    Error {
        /// The error message.
        message: String,
        /// The node where the error occurred (if known).
        node: Option<String>,
    },

    /// A buffer was processed by a node.
    BufferProcessed {
        /// The node that processed the buffer.
        node: String,
        /// Sequence number of the buffer.
        sequence: u64,
    },

    /// A node started processing.
    NodeStarted {
        /// The node that started.
        node: String,
    },

    /// A node finished processing (EOS reached for that node).
    NodeFinished {
        /// The node that finished.
        node: String,
        /// Number of buffers processed.
        buffers_processed: u64,
    },

    /// Pipeline execution started.
    Started,

    /// Pipeline execution stopped.
    Stopped,

    /// Warning (non-fatal issue).
    Warning {
        /// The warning message.
        message: String,
        /// The node that emitted the warning (if known).
        node: Option<String>,
    },

    /// Custom user-defined event.
    Custom {
        /// Event name.
        name: String,
        /// Event payload (opaque bytes).
        payload: Vec<u8>,
    },

    /// A new pad was added to an element.
    ///
    /// This event is emitted when a demuxer or other dynamic element
    /// creates a new output pad at runtime.
    PadAdded {
        /// The node that has the new pad.
        node: String,
        /// The name of the new pad.
        pad_name: String,
        /// The direction of the pad (Input or Output).
        direction: PadDirection,
        /// The caps (format) of the new pad.
        caps: Caps,
    },

    /// A pad was removed from an element.
    ///
    /// This event is emitted when a dynamic pad is removed from an element.
    PadRemoved {
        /// The node that had the pad removed.
        node: String,
        /// The name of the removed pad.
        pad_name: String,
        /// The direction of the pad (Input or Output).
        direction: PadDirection,
    },

    /// A link was dynamically created between pads.
    ///
    /// This event is emitted when pads are connected at runtime.
    LinkCreated {
        /// Source node name.
        source_node: String,
        /// Source pad name.
        source_pad: String,
        /// Sink node name.
        sink_node: String,
        /// Sink pad name.
        sink_pad: String,
    },

    /// A link was dynamically removed.
    ///
    /// This event is emitted when a connection is removed at runtime.
    LinkRemoved {
        /// Source node name.
        source_node: String,
        /// Source pad name.
        source_pad: String,
        /// Sink node name.
        sink_node: String,
        /// Sink pad name.
        sink_pad: String,
    },

    /// Re-negotiation is required.
    ///
    /// This event is emitted when the pipeline needs to re-negotiate
    /// caps due to dynamic changes (e.g., new pads added).
    NegotiationRequired {
        /// The node that triggered the re-negotiation.
        node: String,
        /// Reason for re-negotiation.
        reason: String,
    },
}

impl fmt::Display for PipelineEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineEvent::StateChanged { from, to } => {
                write!(f, "StateChanged: {:?} -> {:?}", from, to)
            }
            PipelineEvent::Eos => write!(f, "EOS"),
            PipelineEvent::Error { message, node } => {
                if let Some(n) = node {
                    write!(f, "Error in {}: {}", n, message)
                } else {
                    write!(f, "Error: {}", message)
                }
            }
            PipelineEvent::BufferProcessed { node, sequence } => {
                write!(f, "Buffer {} processed by {}", sequence, node)
            }
            PipelineEvent::NodeStarted { node } => write!(f, "Node {} started", node),
            PipelineEvent::NodeFinished {
                node,
                buffers_processed,
            } => {
                write!(f, "Node {} finished ({} buffers)", node, buffers_processed)
            }
            PipelineEvent::Started => write!(f, "Pipeline started"),
            PipelineEvent::Stopped => write!(f, "Pipeline stopped"),
            PipelineEvent::Warning { message, node } => {
                if let Some(n) = node {
                    write!(f, "Warning in {}: {}", n, message)
                } else {
                    write!(f, "Warning: {}", message)
                }
            }
            PipelineEvent::Custom { name, payload } => {
                write!(f, "Custom event '{}' ({} bytes)", name, payload.len())
            }
            PipelineEvent::PadAdded {
                node,
                pad_name,
                direction,
                caps: _,
            } => {
                write!(f, "Pad added: {}.{} ({:?})", node, pad_name, direction)
            }
            PipelineEvent::PadRemoved {
                node,
                pad_name,
                direction,
            } => {
                write!(f, "Pad removed: {}.{} ({:?})", node, pad_name, direction)
            }
            PipelineEvent::LinkCreated {
                source_node,
                source_pad,
                sink_node,
                sink_pad,
            } => {
                write!(
                    f,
                    "Link created: {}.{} -> {}.{}",
                    source_node, source_pad, sink_node, sink_pad
                )
            }
            PipelineEvent::LinkRemoved {
                source_node,
                source_pad,
                sink_node,
                sink_pad,
            } => {
                write!(
                    f,
                    "Link removed: {}.{} -> {}.{}",
                    source_node, source_pad, sink_node, sink_pad
                )
            }
            PipelineEvent::NegotiationRequired { node, reason } => {
                write!(f, "Re-negotiation required ({}): {}", node, reason)
            }
        }
    }
}

/// Sender for pipeline events.
///
/// This is held by the pipeline executor and used to emit events.
#[derive(Clone)]
pub struct EventSender {
    sender: broadcast::Sender<PipelineEvent>,
}

impl EventSender {
    /// Create a new event sender with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Send an event.
    ///
    /// Returns the number of receivers that received the event.
    /// Returns 0 if there are no receivers (which is fine).
    pub fn send(&self, event: PipelineEvent) -> usize {
        self.sender.send(event).unwrap_or(0)
    }

    /// Send an EOS event.
    pub fn send_eos(&self) {
        self.send(PipelineEvent::Eos);
    }

    /// Send an error event.
    pub fn send_error(&self, message: impl Into<String>, node: Option<String>) {
        self.send(PipelineEvent::Error {
            message: message.into(),
            node,
        });
    }

    /// Send a state changed event.
    pub fn send_state_changed(&self, from: super::PipelineState, to: super::PipelineState) {
        self.send(PipelineEvent::StateChanged { from, to });
    }

    /// Send a node started event.
    pub fn send_node_started(&self, node: impl Into<String>) {
        self.send(PipelineEvent::NodeStarted { node: node.into() });
    }

    /// Send a node finished event.
    pub fn send_node_finished(&self, node: impl Into<String>, buffers_processed: u64) {
        self.send(PipelineEvent::NodeFinished {
            node: node.into(),
            buffers_processed,
        });
    }

    /// Send a pad added event.
    pub fn send_pad_added(
        &self,
        node: impl Into<String>,
        pad_name: impl Into<String>,
        direction: PadDirection,
        caps: Caps,
    ) {
        self.send(PipelineEvent::PadAdded {
            node: node.into(),
            pad_name: pad_name.into(),
            direction,
            caps,
        });
    }

    /// Send a pad removed event.
    pub fn send_pad_removed(
        &self,
        node: impl Into<String>,
        pad_name: impl Into<String>,
        direction: PadDirection,
    ) {
        self.send(PipelineEvent::PadRemoved {
            node: node.into(),
            pad_name: pad_name.into(),
            direction,
        });
    }

    /// Send a link created event.
    pub fn send_link_created(
        &self,
        source_node: impl Into<String>,
        source_pad: impl Into<String>,
        sink_node: impl Into<String>,
        sink_pad: impl Into<String>,
    ) {
        self.send(PipelineEvent::LinkCreated {
            source_node: source_node.into(),
            source_pad: source_pad.into(),
            sink_node: sink_node.into(),
            sink_pad: sink_pad.into(),
        });
    }

    /// Send a link removed event.
    pub fn send_link_removed(
        &self,
        source_node: impl Into<String>,
        source_pad: impl Into<String>,
        sink_node: impl Into<String>,
        sink_pad: impl Into<String>,
    ) {
        self.send(PipelineEvent::LinkRemoved {
            source_node: source_node.into(),
            source_pad: source_pad.into(),
            sink_node: sink_node.into(),
            sink_pad: sink_pad.into(),
        });
    }

    /// Send a negotiation required event.
    pub fn send_negotiation_required(&self, node: impl Into<String>, reason: impl Into<String>) {
        self.send(PipelineEvent::NegotiationRequired {
            node: node.into(),
            reason: reason.into(),
        });
    }

    /// Create a receiver for events.
    pub fn subscribe(&self) -> EventReceiver {
        EventReceiver {
            receiver: self.sender.subscribe(),
        }
    }
}

impl Default for EventSender {
    fn default() -> Self {
        Self::new(256) // Default capacity
    }
}

/// Receiver for pipeline events.
///
/// Multiple receivers can be created from a single sender.
pub struct EventReceiver {
    receiver: broadcast::Receiver<PipelineEvent>,
}

impl EventReceiver {
    /// Receive the next event.
    ///
    /// Returns `None` if the sender has been dropped.
    pub async fn recv(&mut self) -> Option<PipelineEvent> {
        loop {
            match self.receiver.recv().await {
                Ok(event) => return Some(event),
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    // We missed some events, continue to get the next one
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    }

    /// Try to receive an event without blocking.
    ///
    /// Returns `None` if no event is available or the sender has been dropped.
    pub fn try_recv(&mut self) -> Option<PipelineEvent> {
        loop {
            match self.receiver.try_recv() {
                Ok(event) => return Some(event),
                Err(broadcast::error::TryRecvError::Lagged(_)) => {
                    // We missed some events, try again
                    continue;
                }
                Err(_) => return None,
            }
        }
    }

    /// Wait for EOS or an error.
    ///
    /// Returns `Ok(())` on EOS, `Err(message)` on error.
    pub async fn wait_eos(&mut self) -> Result<(), String> {
        while let Some(event) = self.recv().await {
            match event {
                PipelineEvent::Eos => return Ok(()),
                PipelineEvent::Error { message, node } => {
                    let full_msg = if let Some(n) = node {
                        format!("Error in {}: {}", n, message)
                    } else {
                        message
                    };
                    return Err(full_msg);
                }
                _ => continue,
            }
        }
        Err("Event channel closed unexpectedly".to_string())
    }
}

/// A stream adapter for receiving events.
///
/// Implements `Stream` for use with async iteration.
pub struct EventStream {
    receiver: EventReceiver,
}

impl EventStream {
    /// Create a new event stream from a receiver.
    pub fn new(receiver: EventReceiver) -> Self {
        Self { receiver }
    }
}

impl futures::Stream for EventStream {
    type Item = PipelineEvent;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Use a pinned future for the async recv
        let fut = self.receiver.recv();
        tokio::pin!(fut);
        fut.poll(cx)
    }
}

impl EventSender {
    /// Create a stream of events.
    pub fn stream(&self) -> EventStream {
        EventStream::new(self.subscribe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PipelineState;

    #[tokio::test]
    async fn test_event_send_recv() {
        let sender = EventSender::new(16);
        let mut receiver = sender.subscribe();

        sender.send_eos();

        let event = receiver.recv().await.unwrap();
        assert!(matches!(event, PipelineEvent::Eos));
    }

    #[tokio::test]
    async fn test_multiple_receivers() {
        let sender = EventSender::new(16);
        let mut receiver1 = sender.subscribe();
        let mut receiver2 = sender.subscribe();

        sender.send_state_changed(PipelineState::Suspended, PipelineState::Running);

        // Both receivers should get the event
        let e1 = receiver1.recv().await.unwrap();
        let e2 = receiver2.recv().await.unwrap();

        assert!(matches!(e1, PipelineEvent::StateChanged { .. }));
        assert!(matches!(e2, PipelineEvent::StateChanged { .. }));
    }

    #[tokio::test]
    async fn test_wait_eos() {
        let sender = EventSender::new(16);
        let mut receiver = sender.subscribe();

        // Spawn a task to send events
        let sender_clone = sender.clone();
        tokio::spawn(async move {
            sender_clone.send(PipelineEvent::Started);
            sender_clone.send(PipelineEvent::NodeStarted {
                node: "src".to_string(),
            });
            sender_clone.send_eos();
        });

        let result = receiver.wait_eos().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_eos_error() {
        let sender = EventSender::new(16);
        let mut receiver = sender.subscribe();

        // Spawn a task to send an error
        let sender_clone = sender.clone();
        tokio::spawn(async move {
            sender_clone.send_error("Something went wrong", Some("sink".to_string()));
        });

        let result = receiver.wait_eos().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Something went wrong"));
    }

    #[test]
    fn test_event_display() {
        let event = PipelineEvent::Error {
            message: "test error".to_string(),
            node: Some("node1".to_string()),
        };
        assert_eq!(format!("{}", event), "Error in node1: test error");

        let event = PipelineEvent::Eos;
        assert_eq!(format!("{}", event), "EOS");
    }
}
