//! A drop-oldest ("leaky-downstream") channel for lossy pipeline links.
//!
//! [`LinkPolicy::DropOldest`](crate::pipeline::LinkPolicy::DropOldest) needs a
//! sender that can evict the oldest *queued* buffer when the channel is full.
//! `tokio::sync::mpsc` cannot do that — its `Sender` has no pop, and the
//! `Receiver` is owned by the consumer task — so those links get this channel
//! instead: a mutex-guarded `VecDeque` with a [`Notify`] for receiver wakeup.
//!
//! Two invariants the executor relies on:
//!
//! - **Sends never await.** A drop-oldest sender by definition never waits for
//!   room, so both send paths are synchronous. That also keeps every send
//!   trivially safe to call from any context (no `select!` cancellation
//!   hazard, per the executor's no-send-in-select rule).
//! - **Control entries are never evicted.** Each entry carries a `droppable`
//!   flag; eviction skips non-droppable entries, and [`LeakySender::send_control`]
//!   enqueues past capacity rather than dropping or blocking. Control traffic
//!   (EOS, errors, flush pairs, segments) is rare and bounded, so the
//!   overshoot is transient; buffers are what a leaky link sheds.
//!
//! `recv` is cancel-safe: the `Notify` future is registered (`enable`) *before*
//! the queue is checked, the pop happens synchronously in the poll that
//! returns the value, and nothing is stashed across the single await. A
//! cancelled `recv` may have consumed a stored `notify_one` permit, but the
//! next call re-registers before checking the queue, so a non-empty queue is
//! always seen without needing a permit (the AppSink idiom).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;
use tokio::sync::mpsc::error::TryRecvError;

/// Create a leaky channel bounded (for buffers) at `capacity` entries.
pub(crate) fn channel<T>(capacity: usize) -> (LeakySender<T>, LeakyReceiver<T>) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            queue: VecDeque::with_capacity(capacity),
            rx_alive: true,
            tx_count: 1,
        }),
        notify: Notify::new(),
        capacity: capacity.max(1),
        len: AtomicUsize::new(0),
    });
    (
        LeakySender {
            shared: Arc::clone(&shared),
        },
        LeakyReceiver { shared },
    )
}

struct Shared<T> {
    /// Held only for O(1) / short-scan operations, never across an await.
    state: Mutex<State<T>>,
    /// Wakes the receiver on enqueue; `notify_one` only, so a stored permit
    /// exists for a not-yet-registered receiver.
    notify: Notify,
    /// The buffer bound. Control entries may push past it (see module docs).
    capacity: usize,
    /// Occupancy snapshot for flow monitors — readable without the lock.
    len: AtomicUsize,
}

struct State<T> {
    /// `(item, droppable)`: only droppable entries may be evicted.
    queue: VecDeque<(T, bool)>,
    rx_alive: bool,
    tx_count: usize,
}

/// The outcome of a lossy push.
pub(crate) enum LossyPush<T> {
    /// Enqueued; there was room.
    Sent,
    /// Enqueued after evicting the oldest droppable entry, returned here so
    /// the caller can count/report the loss.
    SentEvictedOldest(T),
    /// The receiver is gone; the item comes back untouched.
    Closed(T),
}

pub(crate) struct LeakySender<T> {
    shared: Arc<Shared<T>>,
}

impl<T> Clone for LeakySender<T> {
    fn clone(&self) -> Self {
        self.shared.state.lock().unwrap().tx_count += 1;
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<T> Drop for LeakySender<T> {
    fn drop(&mut self) {
        let last = {
            let mut st = self.shared.state.lock().unwrap();
            st.tx_count -= 1;
            st.tx_count == 0
        };
        if last {
            // Wake a receiver waiting on a channel that will never fill again.
            self.shared.notify.notify_one();
        }
    }
}

impl<T> LeakySender<T> {
    /// Push a buffer, evicting the oldest droppable entry when full.
    ///
    /// Synchronous — never waits. If the queue is at capacity and holds only
    /// control entries, the item is pushed anyway (transient overshoot;
    /// control is rare and bounded).
    pub(crate) fn send_lossy(&self, item: T) -> LossyPush<T> {
        let mut st = self.shared.state.lock().unwrap();
        if !st.rx_alive {
            return LossyPush::Closed(item);
        }
        let evicted = if st.queue.len() >= self.shared.capacity {
            match st.queue.iter().position(|(_, droppable)| *droppable) {
                Some(idx) => st.queue.remove(idx).map(|(item, _)| item),
                None => None,
            }
        } else {
            None
        };
        st.queue.push_back((item, true));
        self.shared.len.store(st.queue.len(), Ordering::Release);
        drop(st);
        self.shared.notify.notify_one();
        match evicted {
            Some(old) => LossyPush::SentEvictedOldest(old),
            None => LossyPush::Sent,
        }
    }

    /// Push a control message (EOS/error/event): always enqueued, even past
    /// capacity, never evicted, never blocks.
    ///
    /// Returns `false` when the receiver is gone.
    pub(crate) fn send_control(&self, item: T) -> bool {
        let mut st = self.shared.state.lock().unwrap();
        if !st.rx_alive {
            return false;
        }
        st.queue.push_back((item, false));
        self.shared.len.store(st.queue.len(), Ordering::Release);
        drop(st);
        self.shared.notify.notify_one();
        true
    }

    /// Current queue occupancy (for flow monitors).
    pub(crate) fn len(&self) -> usize {
        self.shared.len.load(Ordering::Acquire)
    }
}

pub(crate) struct LeakyReceiver<T> {
    shared: Arc<Shared<T>>,
}

impl<T> Drop for LeakyReceiver<T> {
    fn drop(&mut self) {
        let mut st = self.shared.state.lock().unwrap();
        st.rx_alive = false;
        // Release queued payloads (arena slots) immediately, like dropping a
        // tokio Receiver does. Senders observe `Closed` on their next push.
        st.queue.clear();
        self.shared.len.store(0, Ordering::Release);
    }
}

impl<T> LeakyReceiver<T> {
    /// Receive the next message, waiting if the queue is empty.
    ///
    /// Returns `None` once every sender is gone and the queue is drained.
    ///
    /// Cancel-safe: the pop happens synchronously in the poll that returns
    /// the value; a cancelled call consumes no message.
    pub(crate) async fn recv(&mut self) -> Option<T> {
        loop {
            // Register before checking the queue: a push that lands between
            // the check and the await must find this waiter (or leave a
            // permit via notify_one, which `enable` also observes).
            let mut notified = std::pin::pin!(self.shared.notify.notified());
            notified.as_mut().enable();
            {
                let mut st = self.shared.state.lock().unwrap();
                if let Some((item, _)) = st.queue.pop_front() {
                    self.shared.len.store(st.queue.len(), Ordering::Release);
                    return Some(item);
                }
                if st.tx_count == 0 {
                    return None;
                }
            }
            notified.await;
        }
    }

    /// Non-blocking receive, mirroring tokio's semantics: `Disconnected` only
    /// when the queue is empty *and* every sender is gone.
    pub(crate) fn try_recv(&mut self) -> Result<T, TryRecvError> {
        let mut st = self.shared.state.lock().unwrap();
        match st.queue.pop_front() {
            Some((item, _)) => {
                self.shared.len.store(st.queue.len(), Ordering::Release);
                Ok(item)
            }
            None if st.tx_count == 0 => Err(TryRecvError::Disconnected),
            None => Err(TryRecvError::Empty),
        }
    }

    /// Current queue occupancy (for flow monitors).
    pub(crate) fn len(&self) -> usize {
        self.shared.len.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_removes_the_oldest_droppable() {
        let (tx, mut rx) = channel::<u32>(3);
        for i in 0..3 {
            assert!(matches!(tx.send_lossy(i), LossyPush::Sent));
        }
        // Full: pushing 3 evicts 0, pushing 4 evicts 1.
        assert!(matches!(tx.send_lossy(3), LossyPush::SentEvictedOldest(0)));
        assert!(matches!(tx.send_lossy(4), LossyPush::SentEvictedOldest(1)));
        let mut got = Vec::new();
        while let Ok(v) = rx.try_recv() {
            got.push(v);
        }
        assert_eq!(got, [2, 3, 4], "survivors keep FIFO order, newest kept");
    }

    #[test]
    fn control_is_never_evicted_and_pushes_past_capacity() {
        let (tx, mut rx) = channel::<i32>(2);
        assert!(tx.send_control(-1));
        assert!(tx.send_control(-2));
        // Queue is at capacity with only control entries: a lossy push finds
        // nothing droppable and overshoots instead of dropping control.
        assert!(matches!(tx.send_lossy(1), LossyPush::Sent));
        assert_eq!(tx.len(), 3);
        // A second lossy push evicts the droppable one, not control.
        assert!(matches!(tx.send_lossy(2), LossyPush::SentEvictedOldest(1)));
        // Control keeps enqueueing past capacity.
        assert!(tx.send_control(-3));
        let mut got = Vec::new();
        while let Ok(v) = rx.try_recv() {
            got.push(v);
        }
        assert_eq!(got, [-1, -2, 2, -3], "control kept, in order");
    }

    #[test]
    fn interleaved_control_and_buffers_keep_fifo_order() {
        let (tx, mut rx) = channel::<i32>(8);
        tx.send_lossy(1);
        tx.send_control(-1);
        tx.send_lossy(2);
        tx.send_control(-2);
        let mut got = Vec::new();
        while let Ok(v) = rx.try_recv() {
            got.push(v);
        }
        assert_eq!(got, [1, -1, 2, -2]);
    }

    #[test]
    fn receiver_drop_closes_and_clears() {
        let (tx, rx) = channel::<u32>(4);
        tx.send_lossy(1);
        drop(rx);
        assert!(matches!(tx.send_lossy(2), LossyPush::Closed(2)));
        assert!(!tx.send_control(3));
        assert_eq!(tx.len(), 0, "queued payloads released on receiver drop");
    }

    #[tokio::test]
    async fn all_senders_dropped_ends_recv() {
        let (tx, mut rx) = channel::<u32>(4);
        let tx2 = tx.clone();
        tx.send_lossy(7);
        drop(tx);
        drop(tx2);
        // Drains what is queued, then reports the close.
        assert_eq!(rx.recv().await, Some(7));
        assert_eq!(rx.recv().await, None);
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Disconnected)));
    }

    #[tokio::test]
    async fn recv_wakes_on_send() {
        let (tx, mut rx) = channel::<u32>(4);
        let handle = tokio::spawn(async move { rx.recv().await });
        tokio::task::yield_now().await;
        tx.send_lossy(42);
        assert_eq!(handle.await.unwrap(), Some(42));
    }

    /// A `recv` future dropped by a `select!` loser must not lose a message:
    /// the pop happens in the returning poll, and a consumed permit cannot
    /// hide a non-empty queue from the next call.
    #[tokio::test]
    async fn cancelled_recv_loses_nothing() {
        let (tx, mut rx) = channel::<u32>(64);
        let expected: Vec<u32> = (0..50).collect();

        let consumer = tokio::spawn(async move {
            let mut got = Vec::new();
            loop {
                let mut tick = std::pin::pin!(tokio::task::yield_now());
                tokio::select! {
                    biased;
                    msg = rx.recv() => match msg {
                        Some(v) => got.push(v),
                        None => break,
                    },
                    // A perpetually-ready competitor: recv's future is created
                    // and dropped on every loop iteration where yield wins.
                    _ = &mut tick => {}
                }
            }
            got
        });

        for i in 0..50u32 {
            tx.send_lossy(i);
            if i % 7 == 0 {
                tokio::task::yield_now().await;
            }
        }
        drop(tx);
        let got = consumer.await.unwrap();
        assert_eq!(got, expected, "every message survives select cancellation");
    }

    #[test]
    fn len_tracks_pushes_and_pops() {
        let (tx, mut rx) = channel::<u32>(4);
        assert_eq!(tx.len(), 0);
        tx.send_lossy(1);
        tx.send_lossy(2);
        assert_eq!(tx.len(), 2);
        assert_eq!(rx.len(), 2);
        let _ = rx.try_recv();
        assert_eq!(rx.len(), 1);
    }
}
