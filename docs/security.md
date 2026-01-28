# Security Sandbox Documentation

This document describes the security sandbox implementation in Parallax, designed for isolating untrusted pipeline elements.

## Overview

Parallax provides per-element process isolation using Linux security primitives:
- **seccomp-bpf**: System call filtering
- **Linux namespaces**: Resource isolation (mount, network, PID)
- **cgroups v2**: Resource limits (memory, CPU, PIDs)
- **Privilege dropping**: setuid/setgid to unprivileged user

## Sandbox Configurations

### Default Sandbox

```rust
ElementSandbox::default()
```

| Feature | Setting |
|---------|---------|
| Seccomp | `MinimalCompute` (restricted syscall allowlist) |
| Mount namespace | Enabled |
| Network namespace | Enabled (no network access) |
| PID namespace | Enabled |
| Privilege drop | Disabled |
| cgroup limits | None |

### Strict Sandbox

```rust
ElementSandbox::strict()
```

Maximum isolation for untrusted code:
- All namespaces enabled
- Drops to `nobody:nogroup` (65534:65534)
- Memory limit: 512 MB
- CPU quota: 100% of one core
- PID limit: 64 processes/threads

### Custom Configurations

```rust
ElementSandbox::default()
    .drop_privileges(1000, 1000)
    .with_limits(CgroupLimits::default().with_memory(1 << 30))
    .allow_path(AllowedPath::read_only("/usr/lib"))
    .with_env("RUST_LOG", "debug")
```

## Seccomp Policies

### MinimalCompute (Default)

Allows only syscalls needed for pure computation and IPC via passed file descriptors:

**Memory Management:**
- `brk`, `mmap`, `munmap`, `mprotect`, `mremap`, `madvise`

**I/O (for passed FDs only):**
- `read`, `write`, `readv`, `writev`, `pread64`, `pwrite64`

**Synchronization:**
- `futex`, `futex_waitv`

**Process Control:**
- `exit`, `exit_group`, `rt_sigreturn`, `rt_sigaction`, `rt_sigprocmask`

**Time:**
- `clock_gettime`, `gettimeofday`, `nanosleep`, `clock_nanosleep`

**Info:**
- `getpid`, `gettid`, `getuid`, `getgid`, `geteuid`, `getegid`

**I/O Multiplexing:**
- `poll`, `ppoll`, `epoll_create1`, `epoll_ctl`, `epoll_wait`, `epoll_pwait`

**Misc:**
- `getrandom`, `close`, `dup`, `dup2`, `dup3`, `fcntl`, `ioctl`, `eventfd2`, `pipe2`

**Explicitly DENIED:**
- `open`, `openat` - no filesystem access
- `socket`, `connect`, `bind` - no network access
- `fork`, `clone`, `execve` - no process creation
- `ptrace` - no debugging other processes
- `mount`, `umount` - no filesystem modification

### WithNetwork

Extends `MinimalCompute` with:
- `socket`, `bind`, `listen`, `accept`, `accept4`, `connect`
- `sendto`, `recvfrom`, `sendmsg`, `recvmsg`
- `shutdown`, `setsockopt`, `getsockopt`
- `getsockname`, `getpeername`

### WithFilesystem

Extends `MinimalCompute` with:
- `open`, `openat`, `stat`, `fstat`, `lstat`, `fstatat`
- `access`, `faccessat`, `readlink`, `readlinkat`
- `getcwd`, `chdir`, `fchdir`
- `rename`, `renameat`, `unlink`, `unlinkat`
- `mkdir`, `mkdirat`, `rmdir`
- `lseek`, `getdents64`

### Custom Policy

```rust
SeccompPolicy::Custom(vec![
    SeccompRule::allow("read"),
    SeccompRule::allow("write"),
    SeccompRule::with_args("ioctl", vec![
        ArgConstraint { arg: 1, op: ArgOp::Eq, value: TIOCGWINSZ as u64 }
    ]),
])
```

## Namespace Isolation

### Mount Namespace

When enabled, the element runs in a private mount namespace with:
- Minimal root filesystem (only required libraries)
- `/proc` mounted (read-only)
- `/dev/null`, `/dev/zero`, `/dev/urandom` available
- Explicitly allowed paths bind-mounted

### Network Namespace

When enabled, the element has:
- No access to host network interfaces
- Loopback interface only (127.0.0.1)
- No ability to create external connections

**Exception:** Elements that need network access use `ElementSandbox::with_network()`.

### PID Namespace

When enabled, the element:
- Cannot see host processes
- Has PID 1 in its namespace
- Cannot send signals to host processes

## cgroup Resource Limits

### Default Limits

| Resource | Limit |
|----------|-------|
| `memory.max` | 512 MB |
| `memory.high` | 256 MB (soft limit, triggers reclaim) |
| `cpu.max` | 100000/100000 (100% of one CPU) |
| `pids.max` | 64 |
| `io.weight` | 100 (default) |

### Memory Limits

```rust
CgroupLimits::default().with_memory(1 << 30)  // 1 GB
```

When `memory.max` is exceeded, the kernel OOM-kills the process.
When `memory.high` is exceeded, the kernel aggressively reclaims memory.

### CPU Limits

```rust
CgroupLimits::default().with_cpu(0.5)  // 50% of one CPU
```

CPU quota is enforced via `cpu.max` in cgroups v2:
- Format: `quota period` (microseconds)
- Example: `50000 100000` = 50ms every 100ms = 50%

### PID Limits

```rust
CgroupLimits::default().with_pids(16)  // Max 16 threads
```

Prevents fork bombs and thread exhaustion attacks.

## Privilege Dropping

```rust
ElementSandbox::default().drop_privileges(65534, 65534)  // nobody:nogroup
```

After sandbox setup, the process calls `setgid()` then `setuid()` to drop to an unprivileged user. This prevents:
- Modifying system files
- Binding to privileged ports (< 1024)
- Accessing files owned by other users

## Security Considerations

### What IS Protected

1. **Host filesystem** - Elements cannot access files outside allowed paths
2. **Host network** - Elements cannot make network connections (unless explicitly allowed)
3. **Host processes** - Elements cannot see or signal other processes
4. **System resources** - Memory, CPU, PIDs are limited
5. **Syscall surface** - Only required syscalls are permitted

### What IS NOT Protected

1. **Shared memory** - Elements share memory via memfd for zero-copy IPC. A malicious element could corrupt data.
2. **Timing attacks** - Elements can measure time and potentially extract information via timing side-channels.
3. **Resource exhaustion within limits** - An element can use its full allocation.
4. **Information in passed data** - Elements can read any data passed to them.

### Recommendations

1. **Use strict sandbox for untrusted code:**
   ```rust
   pipeline.run_isolated().await?;
   ```

2. **Isolate codec elements** (common source of vulnerabilities):
   ```rust
   pipeline.run_isolating(vec!["*dec*", "*demux*"]).await?;
   ```

3. **Validate inputs** before passing to sandboxed elements

4. **Monitor resource usage** via cgroup statistics

5. **Use separate user IDs** for different trust levels

## Implementation Notes

### File Descriptor Passing

Isolated elements communicate via:
- `memfd_create()` for shared memory (data buffers)
- `eventfd()` for wakeup notifications
- Unix domain sockets with `SCM_RIGHTS` for FD passing

All FDs are passed at startup; the element cannot create new ones.

### Seccomp Enforcement

The seccomp filter uses `SECCOMP_RET_KILL_PROCESS` for denied syscalls. This immediately terminates the process rather than returning an error, preventing probing attacks.

### Namespace Setup Order

1. `unshare(CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET)`
2. Pivot root to minimal filesystem
3. Set up allowed bind mounts
4. Apply cgroup limits
5. Apply seccomp filter (must be last - limits further syscalls)
6. Drop privileges via `setgid()` + `setuid()`
7. Execute element main loop

### Recovery from Crashes

The supervisor monitors child processes via `waitpid()`. On crash:
1. Collect exit status / signal
2. Clean up shared memory references
3. Optionally restart with `RestartPolicy`
4. Propagate error to pipeline if unrecoverable

## Testing the Sandbox

```bash
# Run isolated example
cargo run --example 13_isolate_all

# Verify seccomp is active (requires strace)
strace -f cargo run --example 13_isolate_all 2>&1 | grep seccomp
```

## Future Improvements

1. **Landlock** - Additional filesystem sandboxing on Linux 5.13+
2. **io_uring restrictions** - When io_uring support is added
3. **Capability dropping** - Fine-grained capability control
4. **Audit logging** - Log denied syscalls for debugging
