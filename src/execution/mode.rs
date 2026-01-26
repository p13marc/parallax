//! Execution modes for pipeline elements.

use super::sandbox::ElementSandbox;
use std::collections::HashMap;

/// Unique identifier for an element group.
///
/// Elements in the same group run in the same process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupId(pub u32);

impl GroupId {
    /// The supervisor group (group 0) - elements that run in the main process.
    pub const SUPERVISOR: GroupId = GroupId(0);

    /// Create a new group ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the numeric ID.
    pub const fn id(&self) -> u32 {
        self.0
    }
}

impl Default for GroupId {
    fn default() -> Self {
        Self::SUPERVISOR
    }
}

impl From<u32> for GroupId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for GroupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Group({})", self.0)
    }
}

/// Execution mode for pipeline elements.
///
/// Determines how elements are executed: all in-process, each in its own
/// sandboxed process, or grouped to balance isolation and overhead.
#[derive(Clone, Debug)]
pub enum ExecutionMode {
    /// All elements run as Tokio tasks in a single process.
    ///
    /// This is the default mode with the lowest overhead. No process
    /// isolation is provided - a crash in one element crashes the whole
    /// pipeline.
    ///
    /// Best for:
    /// - Development and testing
    /// - Trusted pipelines with no untrusted plugins
    /// - Maximum performance when isolation isn't needed
    InProcess,

    /// Each element runs in a separate sandboxed process.
    ///
    /// Provides maximum isolation at the cost of higher overhead.
    /// Data is shared via memfd arenas, so there's no copying penalty,
    /// but there is IPC latency for control messages.
    ///
    /// Best for:
    /// - Running untrusted plugins
    /// - Maximum fault isolation (one element crash doesn't affect others)
    /// - Security-sensitive deployments
    Isolated {
        /// Sandbox configuration for all elements.
        sandbox: ElementSandbox,
    },

    /// Elements are grouped to balance isolation and overhead.
    ///
    /// Certain elements (matching `isolated_patterns`) run in isolated
    /// processes, while others are grouped together. This allows isolating
    /// untrusted or crash-prone elements while keeping trusted elements
    /// together for lower overhead.
    ///
    /// Best for:
    /// - Production deployments with some untrusted components
    /// - Balancing security and performance
    Grouped {
        /// Glob patterns for elements that should be isolated.
        ///
        /// Elements matching any pattern run in their own sandboxed process.
        /// Examples: `["*decoder*", "untrusted_*"]`
        isolated_patterns: Vec<String>,

        /// Sandbox configuration for isolated elements.
        sandbox: ElementSandbox,

        /// Explicit group assignments for elements.
        ///
        /// Maps element names to group IDs. Elements not in this map
        /// and not matching `isolated_patterns` are auto-grouped.
        ///
        /// If `None`, all non-isolated elements run in the supervisor process.
        groups: Option<HashMap<String, GroupId>>,
    },
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::InProcess
    }
}

impl ExecutionMode {
    /// Create an in-process execution mode.
    pub fn in_process() -> Self {
        Self::InProcess
    }

    /// Create an isolated execution mode with default sandbox.
    pub fn isolated() -> Self {
        Self::Isolated {
            sandbox: ElementSandbox::default(),
        }
    }

    /// Create an isolated execution mode with custom sandbox.
    pub fn isolated_with_sandbox(sandbox: ElementSandbox) -> Self {
        Self::Isolated { sandbox }
    }

    /// Create a grouped execution mode.
    pub fn grouped(isolated_patterns: Vec<String>) -> Self {
        Self::Grouped {
            isolated_patterns,
            sandbox: ElementSandbox::default(),
            groups: None,
        }
    }

    /// Check if an element should be isolated based on its name.
    pub fn should_isolate(&self, element_name: &str) -> bool {
        match self {
            Self::InProcess => false,
            Self::Isolated { .. } => true,
            Self::Grouped {
                isolated_patterns, ..
            } => isolated_patterns
                .iter()
                .any(|pattern| matches_pattern(pattern, element_name)),
        }
    }

    /// Get the group for an element.
    ///
    /// Returns `None` if the element should be isolated.
    pub fn get_group(&self, element_name: &str) -> Option<GroupId> {
        match self {
            Self::InProcess => Some(GroupId::SUPERVISOR),
            Self::Isolated { .. } => None, // Each element is its own "group"
            Self::Grouped {
                isolated_patterns,
                groups,
                ..
            } => {
                // Check if should be isolated
                if isolated_patterns
                    .iter()
                    .any(|p| matches_pattern(p, element_name))
                {
                    return None;
                }

                // Check explicit group assignment
                if let Some(groups) = groups {
                    if let Some(&group) = groups.get(element_name) {
                        return Some(group);
                    }
                }

                // Default to supervisor group
                Some(GroupId::SUPERVISOR)
            }
        }
    }

    /// Check if this mode provides process isolation.
    pub fn is_isolated(&self) -> bool {
        !matches!(self, Self::InProcess)
    }

    /// Get the sandbox configuration (if any).
    pub fn sandbox(&self) -> Option<&ElementSandbox> {
        match self {
            Self::InProcess => None,
            Self::Isolated { sandbox } | Self::Grouped { sandbox, .. } => Some(sandbox),
        }
    }
}

/// Simple glob-like pattern matching.
///
/// Supports:
/// - `*` matches any sequence of characters
/// - `?` matches any single character
/// - Other characters match literally
fn matches_pattern(pattern: &str, text: &str) -> bool {
    // Dynamic programming approach for glob matching
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let plen = p.len();
    let tlen = t.len();

    // dp[i][j] = true if pattern[0..i] matches text[0..j]
    let mut dp = vec![vec![false; tlen + 1]; plen + 1];

    // Empty pattern matches empty text
    dp[0][0] = true;

    // Handle patterns starting with '*' (they can match empty string)
    for i in 1..=plen {
        if p[i - 1] == '*' {
            dp[i][0] = dp[i - 1][0];
        } else {
            break;
        }
    }

    for i in 1..=plen {
        for j in 1..=tlen {
            if p[i - 1] == '*' {
                // '*' matches zero characters (dp[i-1][j]) or one+ characters (dp[i][j-1])
                dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            } else if p[i - 1] == '?' || p[i - 1] == t[j - 1] {
                // '?' matches any single char, or exact match
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }

    dp[plen][tlen]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_id() {
        let g1 = GroupId::new(1);
        let g2 = GroupId::from(2);

        assert_eq!(g1.id(), 1);
        assert_eq!(g2.id(), 2);
        assert_eq!(GroupId::SUPERVISOR.id(), 0);
    }

    #[test]
    fn test_execution_mode_default() {
        let mode = ExecutionMode::default();
        assert!(matches!(mode, ExecutionMode::InProcess));
        assert!(!mode.is_isolated());
    }

    #[test]
    fn test_execution_mode_isolated() {
        let mode = ExecutionMode::isolated();
        assert!(mode.is_isolated());
        assert!(mode.should_isolate("any_element"));
        assert!(mode.sandbox().is_some());
    }

    #[test]
    fn test_execution_mode_grouped() {
        let mode = ExecutionMode::grouped(vec!["*decoder*".into(), "untrusted_*".into()]);

        assert!(mode.is_isolated());
        assert!(mode.should_isolate("h264_decoder"));
        assert!(mode.should_isolate("untrusted_plugin"));
        assert!(!mode.should_isolate("passthrough"));
        assert!(!mode.should_isolate("filesrc"));
    }

    #[test]
    fn test_get_group() {
        let mut groups = HashMap::new();
        groups.insert("sink1".into(), GroupId::new(1));
        groups.insert("sink2".into(), GroupId::new(1));

        let mode = ExecutionMode::Grouped {
            isolated_patterns: vec!["*decoder*".into()],
            sandbox: ElementSandbox::default(),
            groups: Some(groups),
        };

        // Isolated elements return None
        assert_eq!(mode.get_group("h264_decoder"), None);

        // Explicitly grouped elements
        assert_eq!(mode.get_group("sink1"), Some(GroupId::new(1)));
        assert_eq!(mode.get_group("sink2"), Some(GroupId::new(1)));

        // Other elements default to supervisor
        assert_eq!(mode.get_group("filesrc"), Some(GroupId::SUPERVISOR));
    }

    #[test]
    fn test_matches_pattern_exact() {
        assert!(matches_pattern("hello", "hello"));
        assert!(!matches_pattern("hello", "world"));
        assert!(!matches_pattern("hello", "hell"));
        assert!(!matches_pattern("hello", "helloo"));
    }

    #[test]
    fn test_matches_pattern_star() {
        assert!(matches_pattern("*", "anything"));
        assert!(matches_pattern("*", ""));
        assert!(matches_pattern("hello*", "hello"));
        assert!(matches_pattern("hello*", "helloworld"));
        assert!(!matches_pattern("hello*", "world"));
        assert!(matches_pattern("*world", "helloworld"));
        assert!(matches_pattern("*world", "world"));
        assert!(!matches_pattern("*world", "worldx"));
        assert!(matches_pattern("*decoder*", "h264_decoder"));
        assert!(matches_pattern("*decoder*", "decoder"));
        assert!(matches_pattern("*decoder*", "my_decoder_element"));
    }

    #[test]
    fn test_matches_pattern_question() {
        assert!(matches_pattern("h?llo", "hello"));
        assert!(matches_pattern("h?llo", "hallo"));
        assert!(!matches_pattern("h?llo", "hllo"));
        assert!(!matches_pattern("h?llo", "heello"));
    }

    #[test]
    fn test_matches_pattern_combined() {
        assert!(matches_pattern("h*o", "hello"));
        assert!(matches_pattern("h*o", "ho"));
        assert!(matches_pattern("h?ll*", "hello"));
        assert!(matches_pattern("h?ll*", "hallo world"));
    }
}
