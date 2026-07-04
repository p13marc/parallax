# Plan 16: Process Isolation

**Priority:** Low (deferred)  
**Effort:** Large (3-4 weeks, from scratch)  
**Depends on:** None (memory/IPC primitives still work)

> **⚠️ Status update (2026-07):** This plan predates commit `da6df59`, which **removed the entire process-isolation scaffolding** (`src/execution/` — `isolated_executor.rs`, `mode.rs`, `sandbox.rs`, `supervisor.rs`, `protocol.rs` — plus `examples/12_isolation.rs`) because the fork-based supervisor carried a fork-bomb risk and was not production-ready. The "What Actually Works" table below describes **deleted code** and is kept only as a design record; the surviving pieces are the memory layer (`memory/ipc.rs` SCM_RIGHTS, `SharedArena` cross-process refcounting) and the informational `ExecutionHints::trust_level`/`uses_native_code` fields. Any new implementation should be **spawn-based** (fork+exec of a worker binary, or element-group processes bridged by `IpcSrc`/`IpcSink`), not fork-per-element, and should rebase the file/example references below (e.g. example slot 51 is now taken by `51_bus_messages.rs`). Current guidance for users is OS-level sandboxing — see `docs/security.md`.

---

## Problem Statement (historical, pre-da6df59)

Parallax's original design promised "Security-First: Inter-process isolation by default." In reality, calling `pipeline.run_isolated()` logged a warning and silently fell back to in-process execution — and the scaffolding has since been removed entirely.

### What Actually Works

| Component | File | Status |
|-----------|------|--------|
| `ExecutionMode` enum | `execution/mode.rs` | Working, tested |
| Pattern matching (`*dec*`) | `execution/mode.rs` | Working, tested |
| Execution planning (graph analysis) | `execution/isolated_executor.rs` | Working, tested |
| Group assignment | `execution/isolated_executor.rs` | Working, tested |
| Control protocol (13 message types) | `execution/protocol.rs` | Working, rkyv serialization tested |
| SCM_RIGHTS fd passing | `memory/ipc.rs` | Working, integration-tested |
| SharedArena cross-process refcounting | `memory/shared_refcount.rs` | Working, stress-tested |
| Supervisor state tracking | `execution/supervisor.rs` | Data structures only, no process control |
| Sandbox configuration | `execution/sandbox.rs` | Config structs only, no enforcement |
| IpcSink/IpcSrc | `elements/ipc/ipc_elements.rs` | Same-process only, doesn't use SCM_RIGHTS |

### What's Missing

1. **Process spawning** — No `Command::new()`, no `fork()`, no `clone()`. The supervisor has no code to actually create child processes.
2. **Sandbox enforcement** — `SeccompPolicy`, namespace flags, and `CgroupLimits` are config structs with zero enforcement code. No seccomp BPF filters, no `unshare()`, no cgroup writes.
3. **IPC channel setup** — IpcSink/IpcSrc don't use the existing `send_fds()`/`recv_fds()` from `memory/ipc.rs`. Arena fd is never sent across process boundaries.
4. **Supervisor event loop** — No `waitpid()`, no crash detection, no restart implementation.
5. **Automatic IPC injection** — The executor identifies boundaries but never injects IpcSrc/IpcSink elements.
6. **Child process entry point** — No binary or function that a child process would run after being spawned.

### The Gap in Numbers

- Memory layer: 90% complete (arena, fd passing, refcounting all work)
- Protocol layer: 80% complete (message types, serialization)
- Orchestration layer: 5% (planning only, no execution)
- Security layer: 0% (pure configuration)

---

## Design Approach

### Phased Implementation

This plan is split into three sub-phases because process isolation has a long dependency chain. Each phase produces a working, testable system:

- **Phase A: Basic Process Isolation** — Spawn child processes, connect via Unix sockets, pass data. No sandboxing. This alone is valuable for crash isolation.
- **Phase B: Sandbox Enforcement** — Add seccomp, namespaces, cgroups. This requires Phase A.
- **Phase C: Crash Recovery** — Implement supervisor watchdog, restart policies, state recovery. This requires Phase A.

### Architecture

```
                     SUPERVISOR PROCESS
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Pipeline: filesrc ! h264dec ! displaysink                       │
│                                                                  │
│  After planning:                                                 │
│  ┌──────────┐     ┌──────────────────────┐     ┌────────────┐   │
│  │ filesrc  │────▶│ IpcSink (injected)   │     │ IpcSrc     │──▶│ displaysink │
│  │ (Group 0)│     │ (boundary to Group 1)│     │ (boundary) │   │ (Group 0)   │
│  └──────────┘     └──────────┬───────────┘     └─────┬──────┘   │
│                              │                       │           │
│                              │ Unix Socket Pair      │           │
│                              │ + SharedArena fd      │           │
│                              │ via SCM_RIGHTS        │           │
└──────────────────────────────┼───────────────────────┼───────────┘
                               │                       │
                               ▼                       ▲
                     CHILD PROCESS (Group 1)
┌──────────────────────────────────────────────────────────────────┐
│  ┌──────────┐     ┌──────────┐     ┌──────────────────────┐     │
│  │ IpcSrc   │────▶│ h264dec  │────▶│ IpcSink (to parent)  │     │
│  │ (from    │     │          │     │                      │     │
│  │  parent) │     │          │     │                      │     │
│  └──────────┘     └──────────┘     └──────────────────────┘     │
│                                                                  │
│  Sandbox: seccomp (MinimalCompute), PID namespace, no network    │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decision: Self-Spawning vs. Fork

**Choice: Self-spawning via `Command::new(std::env::current_exe())`**

Rationale:
- `fork()` is unsafe in multithreaded Rust programs (Tokio). Post-fork, only the forking thread survives — other threads' mutexes become permanently locked.
- Self-spawning with a `--parallax-worker` flag is safe, predictable, and compatible with namespace setup (can `unshare()` before exec).
- PipeWire uses this approach for its filter processes.

The child process re-executes the current binary with special arguments:
```
/path/to/binary --parallax-worker --socket-fd=5 --arena-fd=6 --element=h264dec
```

For library users (who don't control `main()`), we provide a helper that should be called early in `main()`:

```rust
fn main() {
    // This returns immediately in normal mode, exits in worker mode
    parallax::worker::check_worker_mode();
    
    // Normal application code...
}
```

---

## Implementation Steps

### Phase A: Basic Process Isolation (Core — 2 weeks)

#### Step A1: Child Process Entry Point (Medium)

**New file:** `src/execution/worker.rs`

Create the worker process entry point that child processes execute:

```rust
/// Check if this process was spawned as a Parallax worker.
/// Call this early in main(). In worker mode, this function never returns.
pub fn check_worker_mode() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--parallax-worker") {
        run_worker(args);
        std::process::exit(0);
    }
}

/// Worker main loop.
fn run_worker(args: Vec<String>) {
    // Parse args: --socket-fd=N, --arena-fd=N, --element-factory=NAME, --element-props=JSON
    let socket_fd = parse_arg(&args, "--socket-fd");
    let arena_fd = parse_arg(&args, "--arena-fd");
    let factory_name = parse_str_arg(&args, "--element-factory");
    let props_json = parse_str_arg(&args, "--element-props");
    
    // Reconstruct Unix socket from inherited fd
    let socket = unsafe { UnixStream::from_raw_fd(socket_fd) };
    
    // Map shared memory arena from inherited fd
    let arena = SharedArena::from_fd(arena_fd, slot_size, slot_count)?;
    
    // Create element via factory
    let element = ElementFactory::create(&factory_name, &props_json)?;
    
    // Run worker loop: receive buffers, process, send results
    run_worker_loop(socket, arena, element);
}
```

**Key requirement:** The element must be creatable from its factory name and serialized properties. This means the `ElementFactory` (already exists in `pipeline/factory.rs`) must be available in the child process.

#### Step A2: Fix IpcSink/IpcSrc to Use SCM_RIGHTS (Medium)

**File:** `src/elements/ipc/ipc_elements.rs`

The IPC elements already handle Unix sockets and control messages. The missing piece is sending the arena fd via `send_fds()` from `memory/ipc.rs`.

Changes to `IpcSink::send_arena_registration()`:
```rust
fn send_arena_registration(&mut self) -> Result<()> {
    let arena = self.arena.as_ref().unwrap();
    let socket = self.socket.as_ref().unwrap();
    
    // Send arena fd via SCM_RIGHTS
    let msg = ControlMessage::RegisterArena {
        arena_id: arena.id(),
        size: arena.total_size(),
        slot_size: arena.slot_size(),
        slot_count: arena.slot_count(),
    };
    let msg_bytes = frame_message(&msg);
    
    // Send fd + message in one call
    crate::memory::ipc::send_fds(socket, &[arena.fd()], &msg_bytes)?;
    
    Ok(())
}
```

Changes to `IpcSrc` to receive the arena fd and map it:
```rust
fn receive_arena_registration(&mut self) -> Result<()> {
    let socket = self.socket.as_ref().unwrap();
    
    // Receive fd + message
    let mut data_buf = vec![0u8; 4096];
    let (fds, n) = crate::memory::ipc::recv_fds(socket, &mut data_buf)?;
    
    if fds.is_empty() {
        return Err(Error::Element("No arena fd received".into()));
    }
    
    // Parse the control message from data_buf[..n]
    let (msg, _) = unframe_message(&data_buf[..n])
        .ok_or_else(|| Error::Element("Invalid arena registration".into()))?;
    
    if let ControlMessage::RegisterArena { arena_id, slot_size, slot_count, .. } = msg {
        // Map the arena from the received fd
        let arena = SharedArena::from_fd(fds[0], slot_size, slot_count)?;
        self.arena_cache.insert(arena_id, arena);
    }
    
    Ok(())
}
```

**Prerequisite:** Add `SharedArena::from_fd()` constructor that maps an existing arena fd instead of creating a new one. Check if this already exists; if not, add it to `memory/shared_refcount.rs`.

#### Step A3: Add `SharedArena::from_fd()` (Small)

**File:** `src/memory/shared_refcount.rs`

Add a constructor that maps an existing arena from a received file descriptor:

```rust
impl SharedArena {
    /// Map an existing arena from a file descriptor received via SCM_RIGHTS.
    ///
    /// This is used by child processes to access arenas created by the parent.
    /// The fd should have been received via `ipc::recv_fds()`.
    pub fn from_fd(fd: OwnedFd, slot_size: usize, slot_count: usize) -> Result<Self> {
        let total_size = Self::compute_layout(slot_size, slot_count);
        
        // mmap the received fd
        let ptr = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                total_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &fd,
                0,
            )?
        };
        
        // Read and validate header
        let header = unsafe { &*(ptr as *const ArenaHeader) };
        if header.magic != ARENA_MAGIC {
            return Err(Error::InvalidSegment("Invalid arena magic".into()));
        }
        
        Ok(Self {
            fd,
            ptr,
            size: total_size,
            slot_size,
            slot_count,
            is_owner: false,  // We don't own this arena, just mapped it
            // ...
        })
    }
}
```

#### Step A4: Implement Process Spawning in Supervisor (Large — Core Work)

**File:** `src/execution/supervisor.rs`

Add the actual process spawning logic:

```rust
impl Supervisor {
    /// Spawn a child process for an element group.
    pub fn spawn_group(
        &mut self,
        group: &ElementGroup,
        arena: &SharedArena,
        factory_names: &HashMap<NodeId, String>,
        element_props: &HashMap<NodeId, String>,
    ) -> Result<SpawnedProcess> {
        // Create Unix socket pair for IPC
        let (parent_sock, child_sock) = UnixStream::pair()?;
        
        // Build command
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.arg("--parallax-worker");
        
        // Pass socket fd to child (will be inherited)
        // Use fd 3 by convention (after stdin/stdout/stderr)
        let child_sock_fd = child_sock.as_raw_fd();
        cmd.arg(format!("--socket-fd={}", child_sock_fd));
        
        // Pass arena fd
        let arena_fd = arena.fd().as_raw_fd();
        cmd.arg(format!("--arena-fd={}", arena_fd));
        
        // For each element in the group, pass factory info
        // (For simplicity in v1, one element per child)
        let node_id = group.nodes[0];
        if let Some(factory) = factory_names.get(&node_id) {
            cmd.arg(format!("--element-factory={}", factory));
        }
        if let Some(props) = element_props.get(&node_id) {
            cmd.arg(format!("--element-props={}", props));
        }
        
        // Ensure fds are not closed on exec
        unsafe {
            cmd.pre_exec(move || {
                // Keep child_sock_fd and arena_fd open
                rustix::io::fcntl_setfd(
                    BorrowedFd::borrow_raw(child_sock_fd),
                    FdFlags::empty(), // Clear CLOEXEC
                )?;
                rustix::io::fcntl_setfd(
                    BorrowedFd::borrow_raw(arena_fd),
                    FdFlags::empty(),
                )?;
                Ok(())
            });
        }
        
        let child = cmd.spawn()?;
        let pid = child.id();
        
        Ok(SpawnedProcess {
            child,
            pid,
            socket: parent_sock,
            group_id: group.id,
        })
    }
}

/// A spawned child process.
pub struct SpawnedProcess {
    pub child: std::process::Child,
    pub pid: u32,
    pub socket: UnixStream,
    pub group_id: GroupId,
}
```

#### Step A5: Implement `run_isolated()` in IsolatedExecutor (Large — Core Work)

**File:** `src/execution/isolated_executor.rs`

Replace the stub `run_isolated()` with actual implementation:

```rust
async fn run_isolated(&self, pipeline: &mut Pipeline, plan: ExecutionPlan) -> Result<()> {
    // 1. Create shared memory arena
    let arena = SharedArena::new(self.config.slot_size, self.config.arena_slots)?;
    
    // 2. Collect factory names and properties for each element
    //    (needed to recreate elements in child processes)
    let factory_info = self.collect_factory_info(pipeline, &plan)?;
    
    // 3. Spawn child processes for non-supervisor groups
    let mut children = Vec::new();
    for (group_id, group) in &plan.groups {
        if group.is_supervisor {
            continue;
        }
        
        let spawned = self.supervisor.spawn_group(
            group, &arena,
            &factory_info.names, &factory_info.props,
        )?;
        children.push(spawned);
    }
    
    // 4. Send arena fd to each child via SCM_RIGHTS
    for child in &children {
        crate::memory::ipc::send_fds(
            &child.socket,
            &[arena.fd()],
            &frame_message(&ControlMessage::RegisterArena {
                arena_id: arena.id(),
                size: arena.total_size(),
                slot_size: arena.slot_size(),
                slot_count: arena.slot_count(),
            }),
        )?;
    }
    
    // 5. For the supervisor group, run elements in-process
    //    but connect boundaries to child sockets via IPC
    let supervisor_group = plan.groups.get(&GroupId::SUPERVISOR)
        .ok_or_else(|| Error::Pipeline("No supervisor group".into()))?;
    
    // 6. Build the supervisor sub-pipeline with IPC elements at boundaries
    let mut supervisor_pipeline = self.build_supervisor_pipeline(
        pipeline, &plan, &supervisor_group, &children, &arena,
    )?;
    
    // 7. Run supervisor pipeline
    let supervisor_handle = supervisor_pipeline.run();
    
    // 8. Wait for completion, monitoring children
    self.supervisor_loop(supervisor_handle, &mut children).await
}

/// Monitor children and the supervisor pipeline until completion.
async fn supervisor_loop(
    &self,
    supervisor_handle: impl Future<Output = Result<()>>,
    children: &mut Vec<SpawnedProcess>,
) -> Result<()> {
    tokio::select! {
        result = supervisor_handle => {
            // Pipeline completed, shut down children
            for child in children.iter_mut() {
                let msg = frame_message(&ControlMessage::Shutdown);
                let _ = child.socket.write_all(&msg);
                let _ = child.child.wait();
            }
            result
        }
        // Could also monitor children for crashes here
    }
}
```

#### Step A6: Element Factory Serialization (Medium)

**File:** `src/pipeline/factory.rs` (modify)

For process isolation to work, elements need to be recreatable in child processes from their factory name and serialized properties. This is already partially supported by the `ElementFactory` and pipeline parser.

Add a method to extract factory info from pipeline nodes:

```rust
impl Pipeline {
    /// Get the factory name and properties for an element node.
    ///
    /// Returns None if the element wasn't created via factory (e.g., programmatic API).
    pub fn factory_info(&self, node_id: NodeId) -> Option<(String, HashMap<String, String>)> {
        let node = self.get_node(node_id)?;
        Some((node.factory_name()?.to_string(), node.properties().clone()))
    }
}
```

Elements created programmatically (not via pipeline parser) may not have factory names. In that case, isolation isn't possible for those elements — log an error and keep them in the supervisor process.

#### Step A7: Worker Loop Implementation (Medium)

**File:** `src/execution/worker.rs`

The worker loop in the child process:

```rust
fn run_worker_loop(
    mut socket: UnixStream,
    arena: SharedArena,
    mut element: Box<DynAsyncElement<'static>>,
) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    
    rt.block_on(async move {
        loop {
            // Read control message from supervisor
            let msg = read_message(&mut socket)?;
            
            match msg {
                ControlMessage::BufferReady { slot, metadata } => {
                    // Map slot to buffer via arena
                    let buffer = arena.slot_to_buffer(&slot, metadata)?;
                    
                    // Process
                    let result = element.process(Some(buffer)).await?;
                    
                    // Send result back
                    if let Some(output_buf) = result {
                        let (out_slot, out_meta) = arena.buffer_to_slot(&output_buf)?;
                        let reply = ControlMessage::BufferReady {
                            slot: out_slot,
                            metadata: out_meta,
                        };
                        write_message(&mut socket, &reply)?;
                    }
                    
                    // Acknowledge the input slot is done
                    let ack = ControlMessage::BufferDone { slot };
                    write_message(&mut socket, &ack)?;
                }
                ControlMessage::Eos => {
                    // Flush element
                    if let Some(flushed) = element.flush().await? {
                        let (slot, meta) = arena.buffer_to_slot(&flushed)?;
                        write_message(&mut socket, &ControlMessage::BufferReady {
                            slot, metadata: meta,
                        })?;
                    }
                    write_message(&mut socket, &ControlMessage::Eos)?;
                    break;
                }
                ControlMessage::Shutdown => {
                    write_message(&mut socket, &ControlMessage::ShutdownAck)?;
                    break;
                }
                ControlMessage::Ping { seq } => {
                    write_message(&mut socket, &ControlMessage::Pong { seq })?;
                }
                _ => {}
            }
        }
        Ok(())
    })
}
```

#### Step A8: Integration Test — Basic Isolation (Medium)

**File:** `tests/isolation_integration.rs`

```rust
#[tokio::test]
async fn test_basic_process_isolation() {
    // Create a simple pipeline: TestSrc -> PassThrough -> NullSink
    // Isolate the PassThrough element
    let pipeline = Pipeline::parse("testsrc count=10 ! passthrough ! nullsink").unwrap();
    
    // Run with PassThrough isolated
    pipeline.run_isolating(vec!["passthrough*"]).await.unwrap();
    
    // Verify: all 10 buffers should have been processed
}

#[tokio::test]
async fn test_full_isolation() {
    let pipeline = Pipeline::parse("testsrc count=5 ! passthrough ! nullsink").unwrap();
    pipeline.run_isolated().await.unwrap();
}

#[tokio::test]
async fn test_isolation_crash_detection() {
    // Create a pipeline with an element that intentionally crashes
    // Verify the supervisor detects the crash and returns an error
}
```

#### Step A9: Example — Process Isolation (Small)

**File:** `examples/51_process_isolation.rs`

```rust
//! Example 51: Process Isolation
//!
//! Demonstrates running pipeline elements in separate processes:
//! - filesrc runs in the supervisor process (I/O, trusted)
//! - decoder runs in isolated process (untrusted, crash-safe)
//! - filesink runs in the supervisor process (I/O, trusted)
//!
//! cargo run --example 51_process_isolation
```

---

### Phase B: Sandbox Enforcement (Security — 1 week)

Requires Phase A to be complete and tested.

#### Step B1: seccomp BPF Filter Installation (Medium)

**New file:** `src/execution/seccomp.rs`

Add actual seccomp filter installation using `rustix` (already a dependency):

```rust
use rustix::thread::prctl;

/// Install a seccomp-bpf filter in the current process.
///
/// This should be called in the child process AFTER setup but BEFORE
/// processing any untrusted data.
pub fn install_seccomp(policy: &SeccompPolicy) -> Result<()> {
    let filter = compile_filter(policy)?;
    
    // PR_SET_NO_NEW_PRIVS is required before SECCOMP_MODE_FILTER
    unsafe {
        prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)?;
    }
    
    // Install BPF filter
    unsafe {
        libc::prctl(
            libc::PR_SET_SECCOMP,
            libc::SECCOMP_MODE_FILTER,
            &filter as *const _,
        );
    }
    
    Ok(())
}

/// Compile a SeccompPolicy into a BPF program.
fn compile_filter(policy: &SeccompPolicy) -> Result<libc::sock_fprog> {
    match policy {
        SeccompPolicy::Permissive => Ok(allow_all_filter()),
        SeccompPolicy::MinimalCompute => compile_minimal_compute(),
        SeccompPolicy::WithNetwork => compile_with_network(),
        SeccompPolicy::WithFileSystem { paths } => compile_with_fs(paths),
        SeccompPolicy::Custom { allowed_syscalls } => compile_custom(allowed_syscalls),
    }
}
```

**Alternative:** Use the `seccompiler` crate (Firecracker's seccomp library, pure Rust, well-tested). Add it as an optional dependency behind a `sandbox` feature flag.

#### Step B2: Namespace Setup (Medium)

**File:** `src/execution/worker.rs` (modify)

Add namespace isolation in the child process before running the worker loop:

```rust
fn setup_namespaces(sandbox: &ElementSandbox) -> Result<()> {
    let mut flags = 0;
    
    if sandbox.pid_namespace {
        flags |= libc::CLONE_NEWPID;
    }
    if sandbox.network_namespace {
        flags |= libc::CLONE_NEWNET;
    }
    if sandbox.mount_namespace {
        flags |= libc::CLONE_NEWNS;
    }
    
    if flags != 0 {
        // unshare() creates new namespaces for the calling process
        let ret = unsafe { libc::unshare(flags) };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            // Non-fatal: log warning if we don't have CAP_SYS_ADMIN
            tracing::warn!("Failed to create namespaces: {} (may need CAP_SYS_ADMIN)", err);
        }
    }
    
    Ok(())
}
```

**Note:** PID namespace requires a fork after `unshare()` for the child to be PID 1 in the new namespace. For Phase B, mount and network namespaces are sufficient and simpler.

#### Step B3: Cgroup Limits (Small)

**File:** `src/execution/cgroup.rs` (new)

```rust
/// Apply cgroup limits to the current process.
pub fn apply_cgroup_limits(limits: &CgroupLimits) -> Result<()> {
    // Create a cgroup for this process
    let cgroup_path = format!("/sys/fs/cgroup/parallax-{}", std::process::id());
    std::fs::create_dir_all(&cgroup_path)?;
    
    // Write memory limit
    if let Some(mem_limit) = limits.memory_limit {
        std::fs::write(
            format!("{}/memory.max", cgroup_path),
            mem_limit.to_string(),
        )?;
    }
    
    // Write CPU limit (as period + quota)
    if let Some(cpu_percent) = limits.cpu_percent {
        let period = 100_000; // 100ms
        let quota = (period as f64 * cpu_percent / 100.0) as u64;
        std::fs::write(format!("{}/cpu.max", cgroup_path), format!("{} {}", quota, period))?;
    }
    
    // Write PID limit
    if let Some(max_pids) = limits.max_pids {
        std::fs::write(format!("{}/pids.max", cgroup_path), max_pids.to_string())?;
    }
    
    // Move current process into the cgroup
    std::fs::write(format!("{}/cgroup.procs", cgroup_path), std::process::id().to_string())?;
    
    Ok(())
}
```

**Note:** cgroupv2 is assumed (standard on modern Linux). Requires appropriate permissions.

#### Step B4: Sandbox Setup in Worker Entry Point (Small)

**File:** `src/execution/worker.rs` (modify)

Wire the sandbox setup into the worker entry point:

```rust
fn run_worker(args: Vec<String>) {
    let socket_fd = parse_arg(&args, "--socket-fd");
    let arena_fd = parse_arg(&args, "--arena-fd");
    let sandbox_json = parse_str_arg(&args, "--sandbox");
    
    // Apply sandbox BEFORE processing any data
    if let Some(sandbox_json) = sandbox_json {
        let sandbox: ElementSandbox = serde_json::from_str(&sandbox_json)?;
        
        // 1. Set up namespaces (must be first)
        setup_namespaces(&sandbox)?;
        
        // 2. Apply cgroup limits
        if let Some(ref limits) = sandbox.cgroup_limits {
            apply_cgroup_limits(limits)?;
        }
        
        // 3. Drop privileges
        if let Some((uid, gid)) = sandbox.uid_gid {
            unsafe {
                libc::setgid(gid);
                libc::setuid(uid);
            }
        }
        
        // 4. Install seccomp (must be LAST - after all privileged operations)
        install_seccomp(&sandbox.seccomp)?;
    }
    
    // Now run the worker loop (seccomp is active)
    run_worker_loop(socket, arena, element);
}
```

The order matters: namespaces and privileges must be set up before seccomp locks down syscalls.

#### Step B5: Tests for Sandbox Enforcement (Medium)

**File:** `tests/sandbox_integration.rs`

```rust
#[tokio::test]
async fn test_seccomp_blocks_network() {
    // Spawn a worker with MinimalCompute policy
    // Verify that socket() syscall fails in the child
}

#[tokio::test] 
async fn test_network_namespace_isolation() {
    // Spawn a worker with network namespace
    // Verify the child can't reach the host network
}

#[tokio::test]
async fn test_cgroup_memory_limit() {
    // Spawn a worker with 10MB memory limit
    // Verify the child is killed when it exceeds the limit
}
```

---

### Phase C: Crash Recovery (Resilience — 1 week)

Requires Phase A to be complete and tested.

#### Step C1: Supervisor Watchdog Loop (Medium)

**File:** `src/execution/supervisor.rs` (modify)

Add an async event loop that monitors child processes:

```rust
impl Supervisor {
    /// Run the supervisor event loop.
    ///
    /// Monitors child processes for crashes and applies restart policies.
    pub async fn run_loop(&mut self, children: &mut Vec<SpawnedProcess>) -> Result<()> {
        loop {
            // Check each child process
            for child in children.iter_mut() {
                match child.child.try_wait() {
                    Ok(Some(status)) => {
                        if status.success() {
                            tracing::info!("Child {} exited normally", child.pid);
                        } else {
                            tracing::error!(
                                "Child {} crashed with status {:?}",
                                child.pid, status.code()
                            );
                            
                            // Apply restart policy
                            if let Some(element) = self.elements.get_mut(&child.element_id) {
                                if element.restart_count < self.restart_policy.max_restarts {
                                    let delay = self.restart_policy
                                        .delay_for_restart(element.restart_count);
                                    element.restart_count += 1;
                                    
                                    tokio::time::sleep(delay).await;
                                    
                                    // Respawn
                                    *child = self.respawn(child)?;
                                } else {
                                    return Err(Error::Pipeline(format!(
                                        "Element {} exceeded max restarts ({})",
                                        element.name, self.restart_policy.max_restarts
                                    )));
                                }
                            }
                        }
                    }
                    Ok(None) => { /* still running */ }
                    Err(e) => {
                        tracing::error!("Failed to check child {}: {}", child.pid, e);
                    }
                }
            }
            
            // Check every 100ms
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}
```

#### Step C2: Graceful Shutdown Protocol (Small)

**File:** `src/execution/supervisor.rs` (modify)

Implement clean shutdown with timeout:

```rust
impl Supervisor {
    /// Gracefully shut down all children.
    pub async fn shutdown_all(&mut self, children: &mut Vec<SpawnedProcess>) -> Result<()> {
        // Send Shutdown to all children
        for child in children.iter_mut() {
            let msg = frame_message(&ControlMessage::Shutdown);
            let _ = child.socket.write_all(&msg);
        }
        
        // Wait up to 5 seconds for graceful exit
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        
        for child in children.iter_mut() {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            match tokio::time::timeout(remaining, async {
                loop {
                    if let Ok(Some(_)) = child.child.try_wait() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }).await {
                Ok(()) => { /* exited gracefully */ }
                Err(_) => {
                    // Timeout — kill
                    tracing::warn!("Child {} didn't exit gracefully, killing", child.pid);
                    let _ = child.child.kill();
                    let _ = child.child.wait();
                }
            }
        }
        
        Ok(())
    }
}
```

#### Step C3: Heartbeat Monitor (Small)

Add periodic ping/pong to detect frozen children:

```rust
impl Supervisor {
    /// Send a heartbeat ping to a child and wait for pong.
    async fn heartbeat(&self, child: &mut SpawnedProcess, timeout: Duration) -> Result<bool> {
        let seq = self.next_ping_seq.fetch_add(1, Ordering::Relaxed);
        let msg = frame_message(&ControlMessage::Ping { seq });
        child.socket.write_all(&msg)?;
        
        // Wait for Pong with matching seq
        match tokio::time::timeout(timeout, async {
            loop {
                if let Some(ControlMessage::Pong { seq: resp_seq }) = read_message_async(&child.socket).await? {
                    if resp_seq == seq {
                        return Ok::<bool, Error>(true);
                    }
                }
            }
        }).await {
            Ok(Ok(true)) => Ok(true),
            _ => Ok(false), // Timeout or error = unhealthy
        }
    }
}
```

#### Step C4: Crash Recovery Test (Medium)

**File:** `tests/crash_recovery_integration.rs`

```rust
#[tokio::test]
async fn test_child_crash_and_restart() {
    // Create a pipeline with an element that crashes after 5 buffers
    // Configure restart policy: max_restarts=2
    // Verify:
    // 1. First crash is detected
    // 2. Element is restarted
    // 3. Processing continues from the restart point
}

#[tokio::test]
async fn test_max_restarts_exceeded() {
    // Configure max_restarts=1
    // Element crashes twice
    // Verify the pipeline returns an error after exceeding max restarts
}
```

---

## File Changes Summary

### Phase A (Basic Isolation)

| File | Change Type | Description |
|------|-------------|-------------|
| `src/execution/worker.rs` | Add | Child process entry point |
| `src/execution/mod.rs` | Modify | Export `worker` module |
| `src/execution/supervisor.rs` | Modify | Add `spawn_group()`, `SpawnedProcess` |
| `src/execution/isolated_executor.rs` | Modify | Replace stub with real `run_isolated()` |
| `src/memory/shared_refcount.rs` | Modify | Add `SharedArena::from_fd()` |
| `src/elements/ipc/ipc_elements.rs` | Modify | Use `send_fds()`/`recv_fds()` for arena fd |
| `src/pipeline/factory.rs` | Modify | Add factory info extraction |
| `examples/51_process_isolation.rs` | Add | Process isolation example |
| `tests/isolation_integration.rs` | Add | Integration tests |

### Phase B (Sandboxing)

| File | Change Type | Description |
|------|-------------|-------------|
| `src/execution/seccomp.rs` | Add | seccomp BPF filter compilation and installation |
| `src/execution/cgroup.rs` | Add | cgroup limit application |
| `src/execution/worker.rs` | Modify | Wire sandbox setup |
| `src/execution/sandbox.rs` | Modify | Add `serde::Serialize/Deserialize` for CLI passing |
| `Cargo.toml` | Modify | Optional `seccompiler` dependency behind `sandbox` feature |
| `tests/sandbox_integration.rs` | Add | Sandbox enforcement tests |

### Phase C (Crash Recovery)

| File | Change Type | Description |
|------|-------------|-------------|
| `src/execution/supervisor.rs` | Modify | Add watchdog loop, shutdown, heartbeat |
| `tests/crash_recovery_integration.rs` | Add | Crash recovery tests |

---

## Dependencies

### New Crate Dependencies

| Crate | Feature Flag | Purpose | Notes |
|-------|-------------|---------|-------|
| `seccompiler` | `sandbox` | seccomp BPF filter compilation | Pure Rust, from Firecracker. Optional. |

All other required functionality (`UnixStream::pair()`, `Command::new()`, `mmap`, `unshare`, `fcntl`) is available via already-present dependencies (`rustix`, `libc`, `std`).

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `current_exe()` fails on some systems | Fall back to `argv[0]`; document requirement |
| Elements not factory-creatable (programmatic API) | Keep in supervisor process, log warning |
| fd inheritance across `exec` is tricky | Use `pre_exec` hook with explicit `CLOEXEC` clearing |
| seccomp filter blocks needed syscalls | Start with `Permissive` policy, tighten iteratively |
| cgroupv2 not available | Check `/sys/fs/cgroup/cgroup.controllers`, fall back gracefully |
| Multithreaded fork safety | Use `Command::new()` (exec), not `fork()` |
| Worker needs access to element factory | Statically-linked factory registry; dynamic plugins require `LD_LIBRARY_PATH` |
| Performance overhead of IPC | Benchmark: shared-memory slot reference + Unix socket control msg should be <10us per buffer |

---

## Success Criteria

### Phase A
1. `pipeline.run_isolated()` actually spawns child processes (visible in `ps aux`)
2. Data flows correctly: source → IPC → child process → IPC → sink
3. `pipeline.run_isolating(vec!["*passthrough*"])` isolates only matched elements
4. SharedArena memory is truly shared (same physical pages, verified via `/proc/<pid>/maps`)
5. Integration test passes with `testsrc count=100 ! passthrough ! nullsink`

### Phase B
6. seccomp filter prevents `socket()` syscall in `MinimalCompute` mode
7. Network namespace prevents child from reaching host network
8. Memory cgroup kills child that allocates too much

### Phase C
9. Crashed child is restarted and pipeline continues
10. Max restart limit is enforced
11. Graceful shutdown completes within timeout
12. Frozen child (heartbeat timeout) is detected and killed

---

## Open Questions

1. **Single element per child, or multiple?** — Phase A implements one element per child for simplicity. Multiple elements per child (grouped mode) is a natural extension but adds complexity around internal routing.

2. **Feature flag?** — Should process isolation be behind a feature flag (e.g., `isolation`)? Pros: smaller binary for users who don't need it. Cons: more conditional compilation. **Recommendation:** No feature flag for Phase A (it's core functionality). `sandbox` feature flag for Phase B (seccomp/cgroups are Linux-specific and require extra deps).

3. **Static vs dynamic element creation in children** — Factory-based element creation works for parsed pipelines. For programmatic pipelines, we may need to serialize element state. This is a known limitation for v1; document it.
