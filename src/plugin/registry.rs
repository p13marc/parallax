//! Plugin registry for managing loaded plugins and element factories.

use super::loader::{Plugin, PluginError, PluginLoader};
use crate::element::ElementDyn;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Registry for loaded plugins and their elements.
///
/// The registry provides a central place to:
/// - Load and unload plugins
/// - Query available elements
/// - Create element instances
pub struct PluginRegistry {
    /// Loaded plugins indexed by name.
    plugins: RwLock<HashMap<String, Arc<Plugin>>>,
    /// Element name -> plugin name mapping for quick lookup.
    element_index: RwLock<HashMap<String, String>>,
    /// Plugin loader instance.
    loader: PluginLoader,
}

impl PluginRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            element_index: RwLock::new(HashMap::new()),
            loader: PluginLoader::new(),
        }
    }

    /// Create a registry with custom search paths.
    pub fn with_search_paths(
        paths: impl IntoIterator<Item = impl Into<std::path::PathBuf>>,
    ) -> Self {
        let mut loader = PluginLoader::new();
        for path in paths {
            loader.add_search_path(path);
        }
        Self {
            plugins: RwLock::new(HashMap::new()),
            element_index: RwLock::new(HashMap::new()),
            loader,
        }
    }

    /// Add a search path for plugins.
    pub fn add_search_path(&mut self, path: impl Into<std::path::PathBuf>) {
        self.loader.add_search_path(path);
    }

    /// Load a plugin from a specific path.
    ///
    /// # Safety
    ///
    /// Loading plugins executes code from shared libraries.
    /// The plugin must be trusted and properly implement the plugin ABI.
    pub unsafe fn load_plugin(&self, path: impl AsRef<Path>) -> Result<(), PluginError> {
        // SAFETY: Caller guarantees the plugin is trusted.
        let plugin = unsafe { self.loader.load_from_path(path)? };
        self.register_plugin(plugin);
        Ok(())
    }

    /// Load a plugin by name.
    ///
    /// # Safety
    ///
    /// Loading plugins executes code from shared libraries.
    /// The plugin must be trusted and properly implement the plugin ABI.
    pub unsafe fn load_plugin_by_name(&self, name: &str) -> Result<(), PluginError> {
        // SAFETY: Caller guarantees the plugin is trusted.
        let plugin = unsafe { self.loader.load_by_name(name)? };
        self.register_plugin(plugin);
        Ok(())
    }

    /// Register an already-loaded plugin.
    fn register_plugin(&self, plugin: Plugin) {
        let plugin_name = plugin.name().to_string();
        let element_names: Vec<String> = plugin
            .element_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let plugin = Arc::new(plugin);

        // Register plugin
        {
            let mut plugins = self.plugins.write().unwrap();
            plugins.insert(plugin_name.clone(), plugin);
        }

        // Index elements
        {
            let mut index = self.element_index.write().unwrap();
            for elem_name in element_names {
                index.insert(elem_name, plugin_name.clone());
            }
        }
    }

    /// Unload a plugin by name.
    ///
    /// Returns true if the plugin was found and unloaded.
    pub fn unload_plugin(&self, name: &str) -> bool {
        let plugin = {
            let mut plugins = self.plugins.write().unwrap();
            plugins.remove(name)
        };

        if let Some(plugin) = plugin {
            // Remove element index entries
            let mut index = self.element_index.write().unwrap();
            let elements_to_remove: Vec<String> = plugin
                .element_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            for elem_name in elements_to_remove {
                index.remove(&elem_name);
            }
            true
        } else {
            false
        }
    }

    /// Check if a plugin is loaded.
    pub fn has_plugin(&self, name: &str) -> bool {
        let plugins = self.plugins.read().unwrap();
        plugins.contains_key(name)
    }

    /// Get information about a loaded plugin.
    pub fn plugin_info(&self, name: &str) -> Option<super::PluginInfo> {
        let plugins = self.plugins.read().unwrap();
        plugins.get(name).map(|p| p.info().clone())
    }

    /// List all loaded plugins.
    pub fn list_plugins(&self) -> Vec<String> {
        let plugins = self.plugins.read().unwrap();
        plugins.keys().cloned().collect()
    }

    /// List all available elements from all plugins.
    pub fn list_elements(&self) -> Vec<String> {
        let index = self.element_index.read().unwrap();
        index.keys().cloned().collect()
    }

    /// Check if an element is available.
    pub fn has_element(&self, name: &str) -> bool {
        let index = self.element_index.read().unwrap();
        index.contains_key(name)
    }

    /// Get the plugin that provides an element.
    pub fn element_plugin(&self, element_name: &str) -> Option<String> {
        let index = self.element_index.read().unwrap();
        index.get(element_name).cloned()
    }

    /// Create an element by name.
    ///
    /// This looks up which plugin provides the element and creates an instance.
    pub fn create_element(&self, name: &str) -> Result<Box<dyn ElementDyn>, PluginError> {
        // Find which plugin has this element
        let plugin_name = {
            let index = self.element_index.read().unwrap();
            index.get(name).cloned()
        };

        let plugin_name =
            plugin_name.ok_or_else(|| PluginError::ElementNotFound(name.to_string()))?;

        // Get the plugin and create the element
        let plugins = self.plugins.read().unwrap();
        let plugin = plugins
            .get(&plugin_name)
            .ok_or_else(|| PluginError::ElementNotFound(name.to_string()))?;

        plugin.create_element(name)
    }

    /// Scan a directory and load all plugins found.
    ///
    /// Returns the number of successfully loaded plugins.
    ///
    /// # Safety
    ///
    /// Loading plugins executes code from shared libraries.
    /// All plugins in the directory must be trusted.
    pub unsafe fn load_all_from_dir(&self, dir: impl AsRef<Path>) -> usize {
        // SAFETY: Caller guarantees all plugins are trusted.
        let results = unsafe { self.loader.load_all_from_dir(dir) };
        let mut count = 0;
        for result in results {
            if let Ok(plugin) = result {
                self.register_plugin(plugin);
                count += 1;
            }
        }
        count
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PluginRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let plugins = self.plugins.read().unwrap();
        let elements = self.element_index.read().unwrap();
        f.debug_struct("PluginRegistry")
            .field("plugins", &plugins.len())
            .field("elements", &elements.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = PluginRegistry::new();
        assert!(registry.list_plugins().is_empty());
        assert!(registry.list_elements().is_empty());
    }

    #[test]
    fn test_registry_with_search_paths() {
        let registry = PluginRegistry::with_search_paths(["/custom/path"]);
        assert!(registry.list_plugins().is_empty());
    }

    #[test]
    fn test_has_element_not_found() {
        let registry = PluginRegistry::new();
        assert!(!registry.has_element("nonexistent"));
    }

    #[test]
    fn test_create_element_not_found() {
        let registry = PluginRegistry::new();
        let result = registry.create_element("nonexistent");
        assert!(matches!(result, Err(PluginError::ElementNotFound(_))));
    }

    #[test]
    fn test_unload_nonexistent_plugin() {
        let registry = PluginRegistry::new();
        assert!(!registry.unload_plugin("nonexistent"));
    }
}
