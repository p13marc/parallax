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
/// use parallax::memory::ipc::send_fds;
///
/// let (sender, receiver) = UnixStream::pair()?;
/// let segment = SharedMemorySegment::new("test", 4096)?;
///
/// send_fds(&sender, &[segment.as_fd()], b"hello")?;
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
/// A tuple of (bytes_read, Vec<OwnedFd>) containing the data length and
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
    use crate::memory::{MemorySegment, SharedMemorySegment};

    #[test]
    fn test_send_recv_fds() {
        let (sender, receiver) = UnixStream::pair().unwrap();

        // Create a shared memory segment
        let segment = SharedMemorySegment::new("test-ipc", 4096).unwrap();

        // Write some data to it
        unsafe {
            let slice = std::slice::from_raw_parts_mut(segment.as_mut_ptr().unwrap(), 4096);
            slice[0] = 42;
            slice[1000] = 123;
        }

        // Send the fd
        send_fds(&sender, &[segment.as_fd()], b"hello").unwrap();

        // Receive the fd
        let mut buf = [0u8; 16];
        let (len, fds) = recv_fds(&receiver, &mut buf).unwrap();

        assert_eq!(len, 5);
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(fds.len(), 1);

        // Open the received fd as a segment
        let received =
            unsafe { SharedMemorySegment::from_fd(fds.into_iter().next().unwrap(), 4096).unwrap() };

        // Verify we can read the same data
        unsafe {
            assert_eq!(*received.as_ptr(), 42);
            assert_eq!(*received.as_ptr().add(1000), 123);
        }
    }

    #[test]
    fn test_send_recv_segment_handle() {
        let (sender, receiver) = UnixStream::pair().unwrap();

        let segment = SharedMemorySegment::new("test-handle", 8192).unwrap();

        // Write test data
        unsafe {
            *segment.as_mut_ptr().unwrap() = 99;
        }

        // Send using helper
        send_segment_handle(&sender, segment.as_fd(), segment.len()).unwrap();

        // Receive using helper
        let (fd, size) = recv_segment_handle(&receiver).unwrap();

        assert_eq!(size, 8192);

        // Open and verify
        let received = unsafe { SharedMemorySegment::from_fd(fd, size).unwrap() };
        assert_eq!(received.len(), 8192);
        unsafe {
            assert_eq!(*received.as_ptr(), 99);
        }
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
        let segment1 = SharedMemorySegment::new("test-visibility", 4096).unwrap();

        // Send the fd
        send_segment_handle(&sender, segment1.as_fd(), segment1.len()).unwrap();

        // Receive and open
        let (fd, size) = recv_segment_handle(&receiver).unwrap();
        let segment2 = unsafe { SharedMemorySegment::from_fd(fd, size).unwrap() };

        // Write through segment1
        unsafe {
            *segment1.as_mut_ptr().unwrap() = 111;
        }

        // Read through segment2 - should see the change
        unsafe {
            assert_eq!(*segment2.as_ptr(), 111);
        }

        // Write through segment2
        unsafe {
            *segment2.as_mut_ptr().unwrap().add(100) = 222;
        }

        // Read through segment1 - should see the change
        unsafe {
            assert_eq!(*segment1.as_ptr().add(100), 222);
        }
    }
}
