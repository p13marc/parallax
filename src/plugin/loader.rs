//! Dynamic plugin loading using libloading.

use super::descriptor::{PARALLAX_ABI_VERSION, PluginDescriptor, PluginInfo};
use crate::element::DynAsyncElement;
use libloading::{Library, Symbol};
use std::ffi::OsStr;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur when loading plugins.
#[derive(Debug, Error)]
pub enum PluginError {
    /// Failed to load the shared library.
    #[error("failed to load library: {0}")]
    LoadFailed(String),

    /// The plugin doesn't have the required entry point.
    #[error("missing plugin entry point: parallax_plugin_descriptor")]
    MissingEntryPoint,

    /// The plugin returned a null descriptor.
    #[error("plugin returned null descriptor")]
    NullDescriptor,

    /// ABI version mismatch.
    #[error("ABI version mismatch: expected {expected}, got {actual}")]
    AbiMismatch {
        /// Expected ABI version.
        expected: u32,
        /// Actual ABI version found.
        actual: u32,
    },

    /// Plugin descriptor validation failed.
    #[error("invalid plugin descriptor: {0}")]
    InvalidDescriptor(&'static str),

    /// Element not found in plugin.
    #[error("element '{0}' not found in plugin")]
    ElementNotFound(String),

    /// Failed to create element.
    #[error("failed to create element '{0}'")]
    CreateFailed(String),
}

/// Type of the plugin entry point function.
type PluginEntryPoint = unsafe extern "C" fn() -> *const PluginDescriptor;

/// A loaded plugin.
///
/// The plugin holds a reference to the shared library to keep it loaded.
/// When the Plugin is dropped, the library is unloaded.
pub struct Plugin {
    /// The loaded library (kept alive).
    _library: Arc<Library>,
    /// Pointer to the plugin descriptor (valid as long as library is loaded).
    descriptor: *const PluginDescriptor,
    /// Cached plugin info.
    info: PluginInfo,
}

// SAFETY: Plugin only accesses static data from the loaded library
// through validated pointers. The library is kept alive by Arc<Library>.
unsafe impl Send for Plugin {}
unsafe impl Sync for Plugin {}

impl Plugin {
    /// Get information about the plugin.
    pub fn info(&self) -> &PluginInfo {
        &self.info
    }

    /// Get the plugin name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Get the plugin version.
    pub fn version(&self) -> &str {
        &self.info.version
    }

    /// List element names provided by this plugin.
    pub fn element_names(&self) -> Vec<&str> {
        self.info.elements.iter().map(|e| e.name.as_str()).collect()
    }

    /// Check if the plugin provides an element with the given name.
    pub fn has_element(&self, name: &str) -> bool {
        self.info.elements.iter().any(|e| e.name == name)
    }

    /// Create an instance of an element by name.
    pub fn create_element(&self, name: &str) -> Result<Box<DynAsyncElement<'static>>, PluginError> {
        // SAFETY: The descriptor was validated at load time and the library is kept alive.
        let desc = unsafe { &*self.descriptor };
        let elements = unsafe { desc.elements() };

        for elem_desc in elements {
            let elem_name = unsafe { elem_desc.name_str() };
            if elem_name == name {
                // SAFETY: The element descriptor was validated at load time.
                let element = unsafe { elem_desc.create_element() };
                return element.ok_or_else(|| PluginError::CreateFailed(name.to_string()));
            }
        }

        Err(PluginError::ElementNotFound(name.to_string()))
    }
}

impl std::fmt::Debug for Plugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plugin")
            .field("name", &self.info.name)
            .field("version", &self.info.version)
            .field("elements", &self.info.elements.len())
            .finish()
    }
}

/// Plugin loader for dynamically loading plugins from shared libraries.
pub struct PluginLoader {
    /// Search paths for plugins.
    search_paths: Vec<std::path::PathBuf>,
}

impl PluginLoader {
    /// Create a new plugin loader with default search paths.
    pub fn new() -> Self {
        Self {
            search_paths: vec![
                // Current directory
                std::path::PathBuf::from("."),
                // Standard plugin directory
                std::path::PathBuf::from("/usr/lib/parallax/plugins"),
                std::path::PathBuf::from("/usr/local/lib/parallax/plugins"),
            ],
        }
    }

    /// Add a search path for plugins.
    pub fn add_search_path(&mut self, path: impl Into<std::path::PathBuf>) {
        self.search_paths.push(path.into());
    }

    /// Load a plugin from a specific path.
    ///
    /// # Safety
    ///
    /// Loading plugins is inherently unsafe because we're executing
    /// arbitrary code from shared libraries. The plugin must:
    /// - Export a valid `parallax_plugin_descriptor` function
    /// - Return a valid, static plugin descriptor
    /// - Properly implement the element creation functions
    pub unsafe fn load_from_path(&self, path: impl AsRef<Path>) -> Result<Plugin, PluginError> {
        let path = path.as_ref();

        // SAFETY: Loading a dynamic library. Caller ensures the library is trusted.
        let library =
            unsafe { Library::new(path).map_err(|e| PluginError::LoadFailed(e.to_string()))? };

        // SAFETY: Getting a symbol from the library. Library was just loaded successfully.
        let entry_point: Symbol<PluginEntryPoint> = unsafe {
            library
                .get(b"parallax_plugin_descriptor\0")
                .map_err(|_| PluginError::MissingEntryPoint)?
        };

        // SAFETY: Calling the entry point function. Caller guarantees plugin is valid.
        let descriptor = unsafe { entry_point() };
        if descriptor.is_null() {
            return Err(PluginError::NullDescriptor);
        }

        // SAFETY: Dereferencing the descriptor pointer. Entry point returned non-null.
        let desc = unsafe { &*descriptor };
        if desc.abi_version != PARALLAX_ABI_VERSION {
            return Err(PluginError::AbiMismatch {
                expected: PARALLAX_ABI_VERSION,
                actual: desc.abi_version,
            });
        }

        // SAFETY: Validating the descriptor. Caller guarantees plugin is properly formed.
        unsafe {
            desc.validate()
                .map_err(|e| PluginError::InvalidDescriptor(e))?;
        }

        // SAFETY: Creating PluginInfo from validated descriptor.
        let info = unsafe { PluginInfo::from_descriptor(desc) };

        Ok(Plugin {
            _library: Arc::new(library),
            descriptor,
            info,
        })
    }

    /// Load a plugin by name, searching in all search paths.
    ///
    /// The name should be without the "lib" prefix and ".so" suffix.
    /// For example, "myfilter" will search for "libmyfilter.so".
    ///
    /// # Safety
    ///
    /// See `load_from_path` for safety requirements.
    pub unsafe fn load_by_name(&self, name: &str) -> Result<Plugin, PluginError> {
        let lib_name = format!("lib{}.so", name);

        for search_path in &self.search_paths {
            let path = search_path.join(&lib_name);
            if path.exists() {
                // SAFETY: Caller guarantees plugin is trusted.
                return unsafe { self.load_from_path(&path) };
            }
        }

        Err(PluginError::LoadFailed(format!(
            "plugin '{}' not found in search paths",
            name
        )))
    }

    /// Scan a directory for plugins and load all of them.
    ///
    /// # Safety
    ///
    /// See `load_from_path` for safety requirements.
    pub unsafe fn load_all_from_dir(
        &self,
        dir: impl AsRef<Path>,
    ) -> Vec<Result<Plugin, PluginError>> {
        let dir = dir.as_ref();
        let mut plugins = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension() == Some(OsStr::new("so")) {
                    // SAFETY: Caller guarantees all plugins in directory are trusted.
                    plugins.push(unsafe { self.load_from_path(&path) });
                }
            }
        }

        plugins
    }
}

impl Default for PluginLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_loader_creation() {
        let loader = PluginLoader::new();
        assert!(!loader.search_paths.is_empty());
    }

    #[test]
    fn test_plugin_loader_add_search_path() {
        let mut loader = PluginLoader::new();
        let initial_count = loader.search_paths.len();
        loader.add_search_path("/custom/path");
        assert_eq!(loader.search_paths.len(), initial_count + 1);
    }

    #[test]
    fn test_load_nonexistent_plugin() {
        let loader = PluginLoader::new();
        let result = unsafe { loader.load_by_name("nonexistent_plugin_xyz") };
        assert!(matches!(result, Err(PluginError::LoadFailed(_))));
    }
}
