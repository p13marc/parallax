//! Thin wrapper over the Linux eventfd syscall.
//!
//! One `EventFd` is a wait-free wake-up channel: any thread `notify()`s
//! (a single `write(2)`, safe from RT threads), and a consumer waits with
//! `try_wait` (non-blocking), `wait_timeout` (blocking, for pool threads),
//! or `wait_async` (tokio). Used by the async↔RT bridges, the RT drivers,
//! and the arena release-queue doorbell (#180).
//!
//! Lives in `memory` because the doorbell sits below the pipeline layer;
//! the pipeline re-uses it from here.

use crate::error::{Error, Result};
use rustix::event::{EventfdFlags, eventfd};
use rustix::fd::OwnedFd;
use std::time::Duration;

/// Linux eventfd: a kernel counter usable as a cross-thread wake-up.
pub struct EventFd {
    /// Cached tokio registration, created lazily by the first `wait_async`
    /// (#181 — a fresh `AsyncFd` per wait cost an epoll ADD+DEL syscall
    /// pair per wakeup cycle). Declared BEFORE `fd`: fields drop in
    /// declaration order, so the registration deregisters from the reactor
    /// before the fd closes. Binds to the first runtime that awaits it —
    /// fine for every current consumer (bridges/drivers/doorbells are
    /// created and consumed within one executor run); `notify`/`try_wait`
    /// bypass it entirely (raw fd I/O), so RT threads are unaffected.
    async_fd: std::sync::OnceLock<tokio::io::unix::AsyncFd<std::os::unix::io::RawFd>>,
    fd: OwnedFd,
}

impl EventFd {
    /// Create a new eventfd with initial value 0.
    pub fn new() -> Result<Self> {
        let fd = eventfd(0, EventfdFlags::NONBLOCK | EventfdFlags::CLOEXEC)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eventfd: {}", e))))?;
        Ok(Self {
            async_fd: std::sync::OnceLock::new(),
            fd,
        })
    }

    /// Signal the eventfd (increment counter).
    ///
    /// This is safe to call from any thread, including RT threads.
    pub fn notify(&self) -> Result<()> {
        let val: u64 = 1;
        let bytes = val.to_ne_bytes();
        rustix::io::write(&self.fd, &bytes)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eventfd write: {}", e))))?;
        Ok(())
    }

    /// Try to read from the eventfd (non-blocking).
    ///
    /// Returns `true` if the eventfd was signaled, `false` if it would block.
    /// Reading drains the counter, so one `try_wait` observes every notify
    /// since the last read.
    pub fn try_wait(&self) -> Result<bool> {
        let mut buf = [0u8; 8];
        match rustix::io::read(&self.fd, &mut buf) {
            Ok(8) => Ok(true),
            Ok(_) => Ok(false),
            Err(rustix::io::Errno::WOULDBLOCK) => Ok(false),
            Err(e) => Err(Error::Io(std::io::Error::other(format!(
                "eventfd read: {}",
                e
            )))),
        }
    }

    /// Block until signaled or `timeout` elapses; drains the counter on wake.
    ///
    /// Returns `Ok(true)` if signaled, `Ok(false)` on timeout. Restarts on
    /// EINTR. The fd stays NONBLOCK — `try_wait` does the draining and
    /// `poll(2)` supplies the blocking.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        use rustix::event::{PollFd, PollFlags};
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if self.try_wait()? {
                return Ok(true);
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok(false);
            }
            let ts = rustix::event::Timespec {
                tv_sec: remaining.as_secs() as _,
                tv_nsec: remaining.subsec_nanos() as _,
            };
            let mut fds = [PollFd::new(&self.fd, PollFlags::IN)];
            match rustix::event::poll(&mut fds, Some(&ts)) {
                // Readable or timed out — the next iteration's try_wait /
                // deadline check decides which.
                Ok(_) => continue,
                Err(rustix::io::Errno::INTR) => continue,
                Err(e) => {
                    return Err(Error::Io(std::io::Error::other(format!(
                        "eventfd poll: {}",
                        e
                    ))));
                }
            }
        }
    }

    /// Wait asynchronously for the eventfd to be signaled.
    ///
    /// This uses tokio's async file descriptor support. The epoll
    /// registration is created once and cached (#181); `clear_ready()` on
    /// the would-block path keeps the edge-triggered readiness honest, and
    /// readiness is deliberately NOT cleared after a successful read — the
    /// worst case is one spurious poll on the next wait, while clearing
    /// could swallow the edge of a concurrent `notify`.
    pub async fn wait_async(&self) -> Result<()> {
        use tokio::io::Interest;
        use tokio::io::unix::AsyncFd;

        let async_fd = match self.async_fd.get() {
            Some(afd) => afd,
            None => {
                let afd = AsyncFd::with_interest(self.as_raw_fd(), Interest::READABLE)
                    .map_err(Error::Io)?;
                // Single-consumer in practice; on a theoretical race the
                // winner's registration is kept and ours drops (deregisters).
                self.async_fd.get_or_init(|| afd)
            }
        };

        loop {
            let mut guard = async_fd.readable().await.map_err(Error::Io)?;

            match self.try_wait() {
                Ok(true) => return Ok(()),
                Ok(false) => {
                    guard.clear_ready();
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Get the raw file descriptor for use with epoll/select.
    pub fn as_raw_fd(&self) -> std::os::unix::io::RawFd {
        use std::os::unix::io::AsRawFd;
        self.fd.as_raw_fd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_timeout_times_out_on_silence() {
        let efd = EventFd::new().unwrap();
        let start = std::time::Instant::now();
        assert!(!efd.wait_timeout(Duration::from_millis(20)).unwrap());
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(18),
            "returned early: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(500),
            "overslept: {elapsed:?}"
        );
    }

    #[test]
    fn wait_timeout_wakes_on_notify_and_drains() {
        let efd = EventFd::new().unwrap();
        efd.notify().unwrap();
        efd.notify().unwrap();
        assert!(efd.wait_timeout(Duration::from_secs(1)).unwrap());
        // Both notifies drained by the one read: a second wait times out.
        assert!(!efd.wait_timeout(Duration::from_millis(10)).unwrap());
    }

    #[test]
    fn wait_timeout_wakes_from_another_thread() {
        let efd = std::sync::Arc::new(EventFd::new().unwrap());
        let ringer = efd.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            ringer.notify().unwrap();
        });
        let start = std::time::Instant::now();
        assert!(efd.wait_timeout(Duration::from_secs(5)).unwrap());
        assert!(start.elapsed() < Duration::from_secs(1));
        t.join().unwrap();
    }
}
