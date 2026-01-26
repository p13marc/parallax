//! Element sandboxing configuration.
//!
//! Provides security isolation for element processes using Linux namespaces,
//! seccomp filters, and cgroup limits.

/// Sandbox configuration for element processes.
///
/// Defines the isolation and resource limits applied to sandboxed elements.
#[derive(Clone, Debug)]
pub struct ElementSandbox {
    /// Seccomp filter (syscall allowlist).
    pub seccomp: SeccompPolicy,

    /// Drop to unprivileged user/group.
    ///
    /// If set, the process will setuid/setgid to these IDs after setup.
    pub uid_gid: Option<(u32, u32)>,

    /// Enable filesystem namespace isolation.
    ///
    /// When true, the element runs in a private mount namespace with
    /// a minimal filesystem (only required libs and devices).
    pub mount_namespace: bool,

    /// Enable network namespace isolation.
    ///
    /// When true, the element has no network access (loopback only).
    pub network_namespace: bool,

    /// Enable PID namespace isolation.
    ///
    /// When true, the element can only see its own processes.
    pub pid_namespace: bool,

    /// Memory and CPU limits via cgroups.
    pub cgroup_limits: Option<CgroupLimits>,

    /// Allow the element to access specific paths.
    ///
    /// These paths are bind-mounted into the element's filesystem namespace.
    pub allowed_paths: Vec<AllowedPath>,

    /// Environment variables to set in the sandboxed process.
    pub environment: Vec<(String, String)>,
}

impl Default for ElementSandbox {
    fn default() -> Self {
        Self {
            seccomp: SeccompPolicy::MinimalCompute,
            uid_gid: None, // Don't change user by default
            mount_namespace: true,
            network_namespace: true,
            pid_namespace: true,
            cgroup_limits: None,
            allowed_paths: Vec::new(),
            environment: Vec::new(),
        }
    }
}

impl ElementSandbox {
    /// Create a sandbox with minimal restrictions (for debugging).
    pub fn permissive() -> Self {
        Self {
            seccomp: SeccompPolicy::Permissive,
            uid_gid: None,
            mount_namespace: false,
            network_namespace: false,
            pid_namespace: false,
            cgroup_limits: None,
            allowed_paths: Vec::new(),
            environment: Vec::new(),
        }
    }

    /// Create a strict sandbox suitable for untrusted code.
    pub fn strict() -> Self {
        Self {
            seccomp: SeccompPolicy::MinimalCompute,
            uid_gid: Some((65534, 65534)), // nobody:nogroup
            mount_namespace: true,
            network_namespace: true,
            pid_namespace: true,
            cgroup_limits: Some(CgroupLimits::default()),
            allowed_paths: Vec::new(),
            environment: Vec::new(),
        }
    }

    /// Create a sandbox that allows network access.
    pub fn with_network() -> Self {
        Self {
            seccomp: SeccompPolicy::WithNetwork,
            network_namespace: false,
            ..Self::default()
        }
    }

    /// Create a sandbox that allows filesystem access.
    pub fn with_filesystem() -> Self {
        Self {
            seccomp: SeccompPolicy::WithFilesystem,
            mount_namespace: false,
            ..Self::default()
        }
    }

    /// Set the user/group to drop privileges to.
    pub fn drop_privileges(mut self, uid: u32, gid: u32) -> Self {
        self.uid_gid = Some((uid, gid));
        self
    }

    /// Set cgroup limits.
    pub fn with_limits(mut self, limits: CgroupLimits) -> Self {
        self.cgroup_limits = Some(limits);
        self
    }

    /// Allow access to a specific path.
    pub fn allow_path(mut self, path: AllowedPath) -> Self {
        self.allowed_paths.push(path);
        self
    }

    /// Add an environment variable.
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment.push((key.into(), value.into()));
        self
    }

    /// Check if this sandbox uses any namespaces.
    pub fn uses_namespaces(&self) -> bool {
        self.mount_namespace || self.network_namespace || self.pid_namespace
    }
}

/// Seccomp filter policy.
///
/// Defines which syscalls are allowed in the sandboxed process.
#[derive(Clone, Debug, Default)]
pub enum SeccompPolicy {
    /// Allow all syscalls (no filtering).
    Permissive,

    /// Minimal syscalls for pure computation.
    ///
    /// Allows: read, write, mmap, mprotect, brk, exit, futex, etc.
    /// Denies: open, socket, fork, exec, etc.
    #[default]
    MinimalCompute,

    /// Allow network syscalls (socket, connect, etc.).
    WithNetwork,

    /// Allow filesystem syscalls (open, stat, etc.).
    WithFilesystem,

    /// Allow both network and filesystem.
    Full,

    /// Custom syscall filter.
    Custom(Vec<SeccompRule>),
}

impl SeccompPolicy {
    /// Get the list of allowed syscalls for this policy.
    pub fn allowed_syscalls(&self) -> Vec<&'static str> {
        match self {
            Self::Permissive => vec![], // Empty means allow all

            Self::MinimalCompute => vec![
                // Memory management
                "brk",
                "mmap",
                "munmap",
                "mprotect",
                "mremap",
                "madvise",
                // Basic I/O (for pipes/sockets passed in)
                "read",
                "write",
                "readv",
                "writev",
                "pread64",
                "pwrite64",
                // Synchronization
                "futex",
                "futex_waitv",
                // Process control
                "exit",
                "exit_group",
                "rt_sigreturn",
                "rt_sigaction",
                "rt_sigprocmask",
                // Time
                "clock_gettime",
                "gettimeofday",
                "nanosleep",
                "clock_nanosleep",
                // Info
                "getpid",
                "gettid",
                "getuid",
                "getgid",
                "geteuid",
                "getegid",
                // Misc
                "getrandom",
                "close",
                "dup",
                "dup2",
                "dup3",
                "fcntl",
                "ioctl", // Needed for terminal, but restricted
                "poll",
                "ppoll",
                "epoll_create1",
                "epoll_ctl",
                "epoll_wait",
                "epoll_pwait",
                "eventfd2",
                "pipe2",
                // Scheduler
                "sched_yield",
                "sched_getaffinity",
            ],

            Self::WithNetwork => {
                let mut syscalls = Self::MinimalCompute.allowed_syscalls();
                syscalls.extend([
                    "socket",
                    "bind",
                    "listen",
                    "accept",
                    "accept4",
                    "connect",
                    "sendto",
                    "recvfrom",
                    "sendmsg",
                    "recvmsg",
                    "shutdown",
                    "setsockopt",
                    "getsockopt",
                    "getsockname",
                    "getpeername",
                ]);
                syscalls
            }

            Self::WithFilesystem => {
                let mut syscalls = Self::MinimalCompute.allowed_syscalls();
                syscalls.extend([
                    "open",
                    "openat",
                    "close",
                    "stat",
                    "fstat",
                    "lstat",
                    "fstatat",
                    "access",
                    "faccessat",
                    "readlink",
                    "readlinkat",
                    "getcwd",
                    "chdir",
                    "fchdir",
                    "rename",
                    "renameat",
                    "unlink",
                    "unlinkat",
                    "mkdir",
                    "mkdirat",
                    "rmdir",
                    "lseek",
                    "getdents64",
                ]);
                syscalls
            }

            Self::Full => {
                let mut syscalls = Self::WithNetwork.allowed_syscalls();
                let fs_syscalls = Self::WithFilesystem.allowed_syscalls();
                for s in fs_syscalls {
                    if !syscalls.contains(&s) {
                        syscalls.push(s);
                    }
                }
                syscalls
            }

            Self::Custom(rules) => rules.iter().map(|r| r.syscall).collect(),
        }
    }

    /// Check if this policy allows a syscall.
    pub fn allows(&self, syscall: &str) -> bool {
        match self {
            Self::Permissive => true,
            _ => self.allowed_syscalls().contains(&syscall),
        }
    }
}

/// A custom seccomp rule.
#[derive(Clone, Debug)]
pub struct SeccompRule {
    /// Syscall name.
    pub syscall: &'static str,
    /// Optional argument constraints.
    pub args: Option<Vec<ArgConstraint>>,
}

impl SeccompRule {
    /// Create a rule that allows a syscall unconditionally.
    pub fn allow(syscall: &'static str) -> Self {
        Self {
            syscall,
            args: None,
        }
    }

    /// Create a rule with argument constraints.
    pub fn with_args(syscall: &'static str, args: Vec<ArgConstraint>) -> Self {
        Self {
            syscall,
            args: Some(args),
        }
    }
}

/// Constraint on a syscall argument.
#[derive(Clone, Debug)]
pub struct ArgConstraint {
    /// Argument index (0-5).
    pub arg: u8,
    /// Comparison operator.
    pub op: ArgOp,
    /// Value to compare against.
    pub value: u64,
}

/// Comparison operator for argument constraints.
#[derive(Clone, Copy, Debug)]
pub enum ArgOp {
    /// Argument equals value.
    Eq,
    /// Argument not equals value.
    Ne,
    /// Argument less than value.
    Lt,
    /// Argument less than or equal to value.
    Le,
    /// Argument greater than value.
    Gt,
    /// Argument greater than or equal to value.
    Ge,
    /// Argument masked with value equals value (flags check).
    MaskedEq(u64),
}

/// Resource limits via cgroups v2.
#[derive(Clone, Debug)]
pub struct CgroupLimits {
    /// Maximum memory in bytes (memory.max).
    pub memory_max: Option<u64>,

    /// Memory high watermark in bytes (memory.high).
    pub memory_high: Option<u64>,

    /// CPU quota as a fraction (e.g., 0.5 = 50% of one CPU).
    pub cpu_quota: Option<f32>,

    /// CPU period in microseconds (default 100000 = 100ms).
    pub cpu_period: u32,

    /// Maximum number of PIDs.
    pub pids_max: Option<u32>,

    /// I/O weight (1-10000, default 100).
    pub io_weight: Option<u16>,
}

impl Default for CgroupLimits {
    fn default() -> Self {
        Self {
            memory_max: Some(512 * 1024 * 1024),  // 512 MB
            memory_high: Some(256 * 1024 * 1024), // 256 MB soft limit
            cpu_quota: Some(1.0),                 // 100% of one CPU
            cpu_period: 100_000,                  // 100ms
            pids_max: Some(64),                   // Max 64 processes/threads
            io_weight: Some(100),                 // Default weight
        }
    }
}

impl CgroupLimits {
    /// Create limits with no restrictions.
    pub fn unlimited() -> Self {
        Self {
            memory_max: None,
            memory_high: None,
            cpu_quota: None,
            cpu_period: 100_000,
            pids_max: None,
            io_weight: None,
        }
    }

    /// Set memory limit.
    pub fn with_memory(mut self, max_bytes: u64) -> Self {
        self.memory_max = Some(max_bytes);
        self.memory_high = Some(max_bytes * 3 / 4); // 75% soft limit
        self
    }

    /// Set CPU quota.
    pub fn with_cpu(mut self, quota: f32) -> Self {
        self.cpu_quota = Some(quota);
        self
    }

    /// Set PID limit.
    pub fn with_pids(mut self, max: u32) -> Self {
        self.pids_max = Some(max);
        self
    }

    /// Calculate cpu.max value for cgroup v2.
    pub fn cpu_max(&self) -> Option<String> {
        self.cpu_quota.map(|quota| {
            let quota_us = (quota * self.cpu_period as f32) as u32;
            format!("{} {}", quota_us, self.cpu_period)
        })
    }
}

/// A path allowed in the sandbox.
#[derive(Clone, Debug)]
pub struct AllowedPath {
    /// Host path.
    pub host_path: String,
    /// Path inside the sandbox (defaults to host_path).
    pub sandbox_path: Option<String>,
    /// Whether the path is writable.
    pub writable: bool,
}

impl AllowedPath {
    /// Allow read-only access to a path.
    pub fn read_only(path: impl Into<String>) -> Self {
        Self {
            host_path: path.into(),
            sandbox_path: None,
            writable: false,
        }
    }

    /// Allow read-write access to a path.
    pub fn read_write(path: impl Into<String>) -> Self {
        Self {
            host_path: path.into(),
            sandbox_path: None,
            writable: true,
        }
    }

    /// Map to a different path inside the sandbox.
    pub fn map_to(mut self, sandbox_path: impl Into<String>) -> Self {
        self.sandbox_path = Some(sandbox_path.into());
        self
    }

    /// Get the path as seen inside the sandbox.
    pub fn target_path(&self) -> &str {
        self.sandbox_path.as_deref().unwrap_or(&self.host_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_default() {
        let sandbox = ElementSandbox::default();
        assert!(sandbox.mount_namespace);
        assert!(sandbox.network_namespace);
        assert!(sandbox.pid_namespace);
        assert!(sandbox.uid_gid.is_none());
        assert!(matches!(sandbox.seccomp, SeccompPolicy::MinimalCompute));
    }

    #[test]
    fn test_sandbox_strict() {
        let sandbox = ElementSandbox::strict();
        assert_eq!(sandbox.uid_gid, Some((65534, 65534)));
        assert!(sandbox.cgroup_limits.is_some());
    }

    #[test]
    fn test_sandbox_permissive() {
        let sandbox = ElementSandbox::permissive();
        assert!(!sandbox.mount_namespace);
        assert!(!sandbox.network_namespace);
        assert!(!sandbox.pid_namespace);
        assert!(matches!(sandbox.seccomp, SeccompPolicy::Permissive));
    }

    #[test]
    fn test_sandbox_builder() {
        let sandbox = ElementSandbox::default()
            .drop_privileges(1000, 1000)
            .with_limits(CgroupLimits::default().with_memory(1024 * 1024 * 1024))
            .allow_path(AllowedPath::read_only("/usr/lib"))
            .with_env("RUST_LOG", "debug");

        assert_eq!(sandbox.uid_gid, Some((1000, 1000)));
        assert!(sandbox.cgroup_limits.is_some());
        assert_eq!(sandbox.allowed_paths.len(), 1);
        assert_eq!(sandbox.environment.len(), 1);
    }

    #[test]
    fn test_seccomp_policy_syscalls() {
        let minimal = SeccompPolicy::MinimalCompute;
        let syscalls = minimal.allowed_syscalls();

        assert!(syscalls.contains(&"read"));
        assert!(syscalls.contains(&"write"));
        assert!(syscalls.contains(&"mmap"));
        assert!(!syscalls.contains(&"socket"));
        assert!(!syscalls.contains(&"open"));

        let with_net = SeccompPolicy::WithNetwork;
        let syscalls = with_net.allowed_syscalls();
        assert!(syscalls.contains(&"socket"));
        assert!(syscalls.contains(&"connect"));

        let with_fs = SeccompPolicy::WithFilesystem;
        let syscalls = with_fs.allowed_syscalls();
        assert!(syscalls.contains(&"open"));
        assert!(syscalls.contains(&"stat"));
    }

    #[test]
    fn test_seccomp_allows() {
        let policy = SeccompPolicy::MinimalCompute;
        assert!(policy.allows("read"));
        assert!(policy.allows("write"));
        assert!(!policy.allows("socket"));

        let permissive = SeccompPolicy::Permissive;
        assert!(permissive.allows("anything"));
    }

    #[test]
    fn test_cgroup_limits_default() {
        let limits = CgroupLimits::default();
        assert_eq!(limits.memory_max, Some(512 * 1024 * 1024));
        assert_eq!(limits.cpu_quota, Some(1.0));
        assert_eq!(limits.pids_max, Some(64));
    }

    #[test]
    fn test_cgroup_limits_cpu_max() {
        let limits = CgroupLimits::default().with_cpu(0.5);
        let cpu_max = limits.cpu_max().unwrap();
        assert_eq!(cpu_max, "50000 100000");
    }

    #[test]
    fn test_allowed_path() {
        let path = AllowedPath::read_only("/usr/lib").map_to("/lib");
        assert_eq!(path.host_path, "/usr/lib");
        assert_eq!(path.target_path(), "/lib");
        assert!(!path.writable);

        let rw_path = AllowedPath::read_write("/tmp");
        assert!(rw_path.writable);
        assert_eq!(rw_path.target_path(), "/tmp");
    }
}
