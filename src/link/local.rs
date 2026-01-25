//! Local (in-process) links using kanal channels.

use crate::buffer::Buffer;
use crate::error::{Error, Result};

/// A local link for passing buffers between elements in the same process.
///
/// This is just a thin wrapper around kanal channels, providing a consistent
/// API with other link types.
///
/// # Example
///
/// ```rust
/// use parallax::link::LocalLink;
/// use parallax::buffer::{Buffer, MemoryHandle};
/// use parallax::memory::HeapSegment;
/// use parallax::metadata::Metadata;
/// use std::sync::Arc;
///
/// let (tx, rx) = LocalLink::bounded(16);
///
/// // Send a buffer
/// let segment = Arc::new(HeapSegment::new(1024).unwrap());
/// let buffer = Buffer::<()>::new(
///     MemoryHandle::from_segment(segment),
///     Metadata::default(),
/// );
/// tx.send(buffer).unwrap();
///
/// // Receive it
/// let received = rx.recv().unwrap();
/// ```
pub struct LocalLink;

impl LocalLink {
    /// Create a bounded local link with the specified capacity.
    ///
    /// The returned sender and receiver can be used to pass buffers between
    /// elements running in the same process.
    pub fn bounded(capacity: usize) -> (LocalSender, LocalReceiver) {
        let (tx, rx) = kanal::bounded(capacity);
        (LocalSender { inner: tx }, LocalReceiver { inner: rx })
    }

    /// Create an unbounded local link.
    ///
    /// Use with caution - can consume unbounded memory if producer is faster
    /// than consumer.
    pub fn unbounded() -> (LocalSender, LocalReceiver) {
        let (tx, rx) = kanal::unbounded();
        (LocalSender { inner: tx }, LocalReceiver { inner: rx })
    }
}

/// Sender half of a local link.
#[derive(Clone)]
pub struct LocalSender {
    inner: kanal::Sender<Buffer>,
}

impl LocalSender {
    /// Send a buffer through the link.
    ///
    /// Blocks if the channel is full (for bounded links).
    pub fn send(&self, buffer: Buffer) -> Result<()> {
        self.inner
            .send(buffer)
            .map_err(|_| Error::Pipeline("channel closed".into()))
    }

    /// Try to send without blocking.
    ///
    /// Returns `Err` if the channel is full or closed.
    pub fn try_send(&self, buffer: Buffer) -> Result<()> {
        match self.inner.try_send(buffer) {
            Ok(true) => Ok(()),
            Ok(false) => Err(Error::Pipeline("channel full".into())),
            Err(_) => Err(Error::Pipeline("channel closed".into())),
        }
    }

    /// Send asynchronously.
    pub async fn send_async(&self, buffer: Buffer) -> Result<()> {
        self.inner
            .as_async()
            .send(buffer)
            .await
            .map_err(|_| Error::Pipeline("channel closed".into()))
    }

    /// Check if the channel is closed.
    pub fn is_closed(&self) -> bool {
        self.inner.is_disconnected()
    }

    /// Get the number of pending messages in the channel.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the channel is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Receiver half of a local link.
pub struct LocalReceiver {
    inner: kanal::Receiver<Buffer>,
}

impl LocalReceiver {
    /// Receive a buffer from the link.
    ///
    /// Blocks until a buffer is available or the channel is closed.
    /// Returns `None` if the channel is closed and empty.
    pub fn recv(&self) -> Option<Buffer> {
        self.inner.recv().ok()
    }

    /// Try to receive without blocking.
    ///
    /// Returns `None` if no buffer is available.
    pub fn try_recv(&self) -> Option<Buffer> {
        match self.inner.try_recv() {
            Ok(Some(buf)) => Some(buf),
            _ => None,
        }
    }

    /// Receive asynchronously.
    pub async fn recv_async(&self) -> Option<Buffer> {
        self.inner.as_async().recv().await.ok()
    }

    /// Check if the channel is closed.
    pub fn is_closed(&self) -> bool {
        self.inner.is_disconnected()
    }

    /// Get the number of pending messages in the channel.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the channel is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Create an iterator over received buffers.
    pub fn iter(&self) -> impl Iterator<Item = Buffer> + '_ {
        std::iter::from_fn(|| self.recv())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;
    use std::thread;

    fn make_buffer(seq: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(64).unwrap());
        Buffer::new(
            MemoryHandle::from_segment(segment),
            Metadata::from_sequence(seq),
        )
    }

    #[test]
    fn test_local_link_basic() {
        let (tx, rx) = LocalLink::bounded(16);

        tx.send(make_buffer(1)).unwrap();
        tx.send(make_buffer(2)).unwrap();

        let b1 = rx.recv().unwrap();
        let b2 = rx.recv().unwrap();

        assert_eq!(b1.metadata().sequence, 1);
        assert_eq!(b2.metadata().sequence, 2);
    }

    #[test]
    fn test_local_link_threaded() {
        let (tx, rx) = LocalLink::bounded(16);
        let count = 100;

        let producer = thread::spawn(move || {
            for i in 0..count {
                tx.send(make_buffer(i)).unwrap();
            }
        });

        let consumer = thread::spawn(move || {
            let mut received = Vec::new();
            for buf in rx.iter().take(count as usize) {
                received.push(buf.metadata().sequence);
            }
            received
        });

        producer.join().unwrap();
        let received = consumer.join().unwrap();

        assert_eq!(received.len(), count as usize);
        for (i, seq) in received.iter().enumerate() {
            assert_eq!(*seq, i as u64);
        }
    }

    #[test]
    fn test_local_link_closed() {
        let (tx, rx) = LocalLink::bounded(16);

        tx.send(make_buffer(1)).unwrap();
        drop(tx);

        // Can still receive pending
        assert!(rx.recv().is_some());
        // Now closed
        assert!(rx.recv().is_none());
        assert!(rx.is_closed());
    }

    #[test]
    fn test_local_link_try_send() {
        let (tx, rx) = LocalLink::bounded(2);

        assert!(tx.try_send(make_buffer(1)).is_ok());
        assert!(tx.try_send(make_buffer(2)).is_ok());
        // Channel full
        assert!(tx.try_send(make_buffer(3)).is_err());

        // Drain one
        rx.recv();
        // Now can send
        assert!(tx.try_send(make_buffer(3)).is_ok());
    }

    #[tokio::test]
    async fn test_local_link_async() {
        let (tx, rx) = LocalLink::bounded(16);

        tx.send_async(make_buffer(42)).await.unwrap();
        let buf = rx.recv_async().await.unwrap();
        assert_eq!(buf.metadata().sequence, 42);
    }

    #[test]
    fn test_unbounded_link() {
        let (tx, rx) = LocalLink::unbounded();

        // Can send many without blocking
        for i in 0..1000 {
            tx.send(make_buffer(i)).unwrap();
        }

        assert_eq!(tx.len(), 1000);

        for i in 0..1000 {
            let buf = rx.recv().unwrap();
            assert_eq!(buf.metadata().sequence, i);
        }
    }
}
