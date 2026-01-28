//! IPC utilities for passing file descriptors between processes.
//!
//! This module provides utilities for sending and receiving file descriptors
//! over Unix domain sockets using `SCM_RIGHTS` ancillary messages.

use crate::error::{Error, Result};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
};
use std::io::{IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::os::unix::net::UnixStream;

/// Maximum number of file descriptors that can be sent in a single message.
pub const MAX_FDS_PER_MESSAGE: usize = 4;

/// Send file descriptors over a Unix socket.
///
/// # Arguments
///
/// * `socket` - The Unix socket to send over.
/// * `fds` - Slice of borrowed file descriptors to send.
/// * `data` - Optional data payload to send along with the fds.
///
/// # Example
///
/// ```rust,ignore
/// use std::os::unix::net::UnixStream;
/// use parallax::memory::{SharedArena, ipc::send_fds};
///
/// let (sender, receiver) = UnixStream::pair()?;
/// let arena = SharedArena::new(4096, 4)?;
///
/// send_fds(&sender, &[arena.fd()], b"hello")?;
/// ```
pub fn send_fds<Fd: AsFd>(socket: &UnixStream, fds: &[Fd], data: &[u8]) -> Result<()> {
    if fds.is_empty() {
        return Err(Error::InvalidSegment("no file descriptors to send".into()));
    }
    if fds.len() > MAX_FDS_PER_MESSAGE {
        return Err(Error::InvalidSegment(format!(
            "too many fds: {} > {}",
            fds.len(),
            MAX_FDS_PER_MESSAGE
        )));
    }

    // We need to send at least one byte of data for SCM_RIGHTS to work
    let data = if data.is_empty() { &[0u8] } else { data };

    // Convert to borrowed fds
    let borrowed_fds: Vec<BorrowedFd<'_>> = fds.iter().map(|fd| fd.as_fd()).collect();

    // Create the ancillary data buffer with MaybeUninit
    let mut ancillary_space: [MaybeUninit<u8>; 64] = [const { MaybeUninit::uninit() }; 64];
    let mut ancillary = SendAncillaryBuffer::new(&mut ancillary_space);

    // Add the file descriptors
    if !ancillary.push(SendAncillaryMessage::ScmRights(&borrowed_fds)) {
        return Err(Error::InvalidSegment(
            "failed to add fds to ancillary buffer".into(),
        ));
    }

    // Send the message
    let iov = [IoSlice::new(data)];
    sendmsg(socket, &iov, &mut ancillary, SendFlags::empty())?;

    Ok(())
}

/// Receive file descriptors from a Unix socket.
///
/// # Arguments
///
/// * `socket` - The Unix socket to receive from.
/// * `data_buf` - Buffer to receive the data payload.
///
/// # Returns
///
/// A tuple of (bytes_read, `Vec<OwnedFd>`) containing the data length and
/// received file descriptors.
///
/// # Example
///
/// ```rust,ignore
/// use std::os::unix::net::UnixStream;
/// use parallax::memory::ipc::recv_fds;
///
/// let (sender, receiver) = UnixStream::pair()?;
/// // ... sender sends fds ...
///
/// let mut buf = [0u8; 1024];
/// let (len, fds) = recv_fds(&receiver, &mut buf)?;
/// ```
pub fn recv_fds(socket: &UnixStream, data_buf: &mut [u8]) -> Result<(usize, Vec<OwnedFd>)> {
    // Ensure we have at least one byte for the mandatory data
    if data_buf.is_empty() {
        return Err(Error::InvalidSegment("data buffer cannot be empty".into()));
    }

    // Create the ancillary data buffer for receiving with MaybeUninit
    let mut ancillary_space: [MaybeUninit<u8>; 64] = [const { MaybeUninit::uninit() }; 64];
    let mut ancillary = RecvAncillaryBuffer::new(&mut ancillary_space);

    // Receive the message
    let mut iov = [IoSliceMut::new(data_buf)];
    let result = recvmsg(socket, &mut iov, &mut ancillary, RecvFlags::empty())?;

    // Extract file descriptors from ancillary messages
    let mut fds = Vec::new();
    for msg in ancillary.drain() {
        if let RecvAncillaryMessage::ScmRights(rights) = msg {
            for fd in rights {
                fds.push(fd);
            }
        }
    }

    Ok((result.bytes, fds))
}

/// Helper to send a single file descriptor with size metadata.
///
/// This is a convenience function for the common case of sending a single
/// shared memory segment's fd along with its size.
pub fn send_segment_handle<Fd: AsFd>(socket: &UnixStream, fd: Fd, size: usize) -> Result<()> {
    let size_bytes = size.to_le_bytes();
    send_fds(socket, &[fd], &size_bytes)
}

/// Helper to receive a single file descriptor with size metadata.
///
/// Returns the fd and size.
pub fn recv_segment_handle(socket: &UnixStream) -> Result<(OwnedFd, usize)> {
    let mut size_buf = [0u8; std::mem::size_of::<usize>()];
    let (bytes_read, fds) = recv_fds(socket, &mut size_buf)?;

    if bytes_read != size_buf.len() {
        return Err(Error::InvalidSegment(format!(
            "expected {} bytes for size, got {}",
            size_buf.len(),
            bytes_read
        )));
    }

    if fds.len() != 1 {
        return Err(Error::InvalidSegment(format!(
            "expected 1 fd, got {}",
            fds.len()
        )));
    }

    let size = usize::from_le_bytes(size_buf);
    let fd = fds.into_iter().next().unwrap();

    Ok((fd, size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SharedArena;

    #[test]
    fn test_send_recv_fds() {
        let (sender, receiver) = UnixStream::pair().unwrap();

        // Create a shared memory arena
        let arena = SharedArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().expect("arena not exhausted");

        // Write some data to it
        slot.data_mut()[0] = 42;
        slot.data_mut()[1000] = 123;

        // Send the fd
        send_fds(&sender, &[arena.fd()], b"hello").unwrap();

        // Receive the fd
        let mut buf = [0u8; 16];
        let (len, fds) = recv_fds(&receiver, &mut buf).unwrap();

        assert_eq!(len, 5);
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(fds.len(), 1);

        // Map the received fd as a new arena view
        let received_fd = fds.into_iter().next().unwrap();
        let received_arena = unsafe { SharedArena::from_fd(received_fd).unwrap() };

        // Get the same slot (by index) and verify the data
        let ipc_ref = slot.ipc_ref();
        let received_slot = received_arena.slot_from_ipc(&ipc_ref).unwrap();

        assert_eq!(received_slot.data()[0], 42);
        assert_eq!(received_slot.data()[1000], 123);
    }

    #[test]
    fn test_send_recv_segment_handle() {
        let (sender, receiver) = UnixStream::pair().unwrap();

        let arena = SharedArena::new(8192, 4).unwrap();
        let mut slot = arena.acquire().expect("arena not exhausted");

        // Write test data
        slot.data_mut()[0] = 99;

        // Send using helper
        send_segment_handle(&sender, arena.fd(), arena.total_size()).unwrap();

        // Receive using helper
        let (fd, size) = recv_segment_handle(&receiver).unwrap();

        assert_eq!(size, arena.total_size());

        // Map and verify
        let received_arena = unsafe { SharedArena::from_fd(fd).unwrap() };

        let ipc_ref = slot.ipc_ref();
        let received_slot = received_arena.slot_from_ipc(&ipc_ref).unwrap();
        assert_eq!(received_slot.data()[0], 99);
    }

    #[test]
    fn test_send_empty_fds_fails() {
        let (sender, _receiver) = UnixStream::pair().unwrap();
        let empty: &[BorrowedFd<'_>] = &[];
        let result = send_fds(&sender, empty, b"data");
        assert!(result.is_err());
    }

    #[test]
    fn test_modifications_visible_across_processes() {
        // This tests the core zero-copy property:
        // modifications made through one mapping are visible through another

        let (sender, receiver) = UnixStream::pair().unwrap();
        let arena1 = SharedArena::new(4096, 4).unwrap();
        let mut slot1 = arena1.acquire().expect("arena not exhausted");

        // Send the fd
        send_segment_handle(&sender, arena1.fd(), arena1.total_size()).unwrap();

        // Receive and open
        let (fd, _size) = recv_segment_handle(&receiver).unwrap();
        let arena2 = unsafe { SharedArena::from_fd(fd).unwrap() };

        // Get the same slot via IPC ref
        let ipc_ref = slot1.ipc_ref();
        let mut slot2 = arena2.slot_from_ipc(&ipc_ref).unwrap();

        // Write through slot1
        slot1.data_mut()[0] = 111;

        // Read through slot2 - should see the change
        assert_eq!(slot2.data()[0], 111);

        // Write through slot2
        slot2.data_mut()[100] = 222;

        // Read through slot1 - should see the change
        assert_eq!(slot1.data()[100], 222);
    }
}
