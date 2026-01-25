//! Element runtime context.
//!
//! The context provides elements with access to runtime information and
//! services during pipeline execution.

use crate::memory::MemoryPool;
use std::sync::Arc;

/// Runtime context for an element.
///
/// The context is passed to elements during initialization and provides
/// access to shared resources like memory pools and configuration.
#[derive(Clone)]
pub struct ElementContext {
    /// Name of this element instance.
    name: String,
    /// Optional memory pool for buffer allocation.
    pool: Option<Arc<MemoryPool>>,
}

impl ElementContext {
    /// Create a new element context.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pool: None,
        }
    }

    /// Create a context with a memory pool.
    pub fn with_pool(name: impl Into<String>, pool: Arc<MemoryPool>) -> Self {
        Self {
            name: name.into(),
            pool: Some(pool),
        }
    }

    /// Get the element's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the memory pool, if one is configured.
    pub fn pool(&self) -> Option<&Arc<MemoryPool>> {
        self.pool.as_ref()
    }

    /// Set the memory pool.
    pub fn set_pool(&mut self, pool: Arc<MemoryPool>) {
        self.pool = Some(pool);
    }
}

impl std::fmt::Debug for ElementContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElementContext")
            .field("name", &self.name)
            .field("has_pool", &self.pool.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HeapSegment;

    #[test]
    fn test_context_creation() {
        let ctx = ElementContext::new("test-element");
        assert_eq!(ctx.name(), "test-element");
        assert!(ctx.pool().is_none());
    }

    #[test]
    fn test_context_with_pool() {
        let segment = HeapSegment::new(1024).unwrap();
        let pool = Arc::new(MemoryPool::new(segment, 256).unwrap());

        let ctx = ElementContext::with_pool("test-element", pool.clone());
        assert!(ctx.pool().is_some());
        assert_eq!(ctx.pool().unwrap().capacity(), pool.capacity());
    }
}
