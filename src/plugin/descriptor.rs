//! Plugin and element descriptors for the C-compatible ABI.
//!
//! These types define the minimal ABI surface that plugins must implement.
//! The design prioritizes safety and simplicity over flexibility.

use crate::element::ElementDyn;
use std::ffi::{CStr, c_char, c_int, c_void};

/// Current ABI version. Plugins must match this version to be loaded.
pub const PARALLAX_ABI_VERSION: u32 = 1;

/// Function pointer type for creating element instances.
///
/// # Safety
///
/// The returned pointer must be a valid `Box<dyn ElementDyn>` that was
/// created using `Box::into_raw(Box::new(...) as Box<dyn ElementDyn>)`.
pub type CreateElementFn = unsafe extern "C" fn() -> *mut c_void;

/// Function pointer type for destroying element instances.
///
/// # Safety
///
/// The pointer must have been created by the corresponding `CreateElementFn`.
pub type DestroyElementFn = unsafe extern "C" fn(*mut c_void);

/// Describes a single element type provided by a plugin.
///
/// This struct is `#[repr(C)]` for C ABI compatibility.
#[repr(C)]
pub struct ElementDescriptor {
    /// Null-terminated element name (e.g., "myfilter").
    pub name: *const c_char,
    /// Null-terminated description.
    pub description: *const c_char,
    /// Element type: 0 = Source, 1 = Transform, 2 = Sink.
    pub element_type: c_int,
    /// Function to create an instance of this element.
    pub create: CreateElementFn,
    /// Function to destroy an instance (or null to use default).
    pub destroy: Option<DestroyElementFn>,
}

// SAFETY: ElementDescriptor contains only raw pointers to static data
// and function pointers, which are inherently Send + Sync.
unsafe impl Send for ElementDescriptor {}
unsafe impl Sync for ElementDescriptor {}

impl ElementDescriptor {
    /// Get the element name as a Rust string.
    ///
    /// # Safety
    ///
    /// The `name` pointer must be valid and null-terminated.
    pub unsafe fn name_str(&self) -> &str {
        // SAFETY: Caller guarantees `name` is valid and null-terminated.
        unsafe { CStr::from_ptr(self.name).to_str().unwrap_or("unknown") }
    }

    /// Get the description as a Rust string.
    ///
    /// # Safety
    ///
    /// The `description` pointer must be valid and null-terminated.
    pub unsafe fn description_str(&self) -> &str {
        // SAFETY: Caller guarantees `description` is valid and null-terminated.
        unsafe { CStr::from_ptr(self.description).to_str().unwrap_or("") }
    }

    /// Create an element instance.
    ///
    /// # Safety
    ///
    /// The `create` function must return a valid pointer that was created
    /// using the `element_to_raw` helper function or equivalent.
    pub unsafe fn create_element(&self) -> Option<Box<dyn ElementDyn>> {
        // SAFETY: Caller guarantees the create function returns a valid pointer.
        let ptr = unsafe { (self.create)() };
        if ptr.is_null() {
            None
        } else {
            // SAFETY: The pointer was created by element_to_raw or equivalent,
            // which uses Box::into_raw on a fat pointer.
            unsafe { Some(element_from_raw(ptr)) }
        }
    }
}

/// Convert an ElementDyn box to a raw pointer for C ABI.
///
/// This is used by plugins to return elements from their create functions.
pub fn element_to_raw(element: Box<dyn ElementDyn>) -> *mut c_void {
    // Convert the fat pointer to a raw pointer
    // We store both the data pointer and vtable by boxing the trait object
    let boxed: Box<Box<dyn ElementDyn>> = Box::new(element);
    Box::into_raw(boxed) as *mut c_void
}

/// Convert a raw pointer back to an ElementDyn box.
///
/// # Safety
///
/// The pointer must have been created by `element_to_raw`.
pub unsafe fn element_from_raw(ptr: *mut c_void) -> Box<dyn ElementDyn> {
    // SAFETY: Caller guarantees ptr was created by element_to_raw.
    let boxed: Box<Box<dyn ElementDyn>> = unsafe { Box::from_raw(ptr as *mut Box<dyn ElementDyn>) };
    *boxed
}

/// Plugin descriptor returned by `parallax_plugin_descriptor()`.
///
/// This struct is `#[repr(C)]` for C ABI compatibility.
#[repr(C)]
pub struct PluginDescriptor {
    /// ABI version - must match `PARALLAX_ABI_VERSION`.
    pub abi_version: u32,
    /// Null-terminated plugin name.
    pub name: *const c_char,
    /// Null-terminated plugin version string.
    pub version: *const c_char,
    /// Null-terminated author/maintainer.
    pub author: *const c_char,
    /// Null-terminated description.
    pub description: *const c_char,
    /// Number of elements in the `elements` array.
    pub num_elements: u32,
    /// Array of element descriptors.
    pub elements: *const ElementDescriptor,
}

// SAFETY: PluginDescriptor contains only raw pointers to static data,
// which are inherently Send + Sync.
unsafe impl Send for PluginDescriptor {}
unsafe impl Sync for PluginDescriptor {}

impl PluginDescriptor {
    /// Get plugin name as a Rust string.
    ///
    /// # Safety
    ///
    /// The `name` pointer must be valid and null-terminated.
    pub unsafe fn name_str(&self) -> &str {
        // SAFETY: Caller guarantees `name` is valid and null-terminated.
        unsafe { CStr::from_ptr(self.name).to_str().unwrap_or("unknown") }
    }

    /// Get version as a Rust string.
    ///
    /// # Safety
    ///
    /// The `version` pointer must be valid and null-terminated.
    pub unsafe fn version_str(&self) -> &str {
        // SAFETY: Caller guarantees `version` is valid and null-terminated.
        unsafe { CStr::from_ptr(self.version).to_str().unwrap_or("0.0.0") }
    }

    /// Get author as a Rust string.
    ///
    /// # Safety
    ///
    /// The `author` pointer must be valid and null-terminated.
    pub unsafe fn author_str(&self) -> &str {
        // SAFETY: Caller guarantees `author` is valid and null-terminated.
        unsafe { CStr::from_ptr(self.author).to_str().unwrap_or("") }
    }

    /// Get description as a Rust string.
    ///
    /// # Safety
    ///
    /// The `description` pointer must be valid and null-terminated.
    pub unsafe fn description_str(&self) -> &str {
        // SAFETY: Caller guarantees `description` is valid and null-terminated.
        unsafe { CStr::from_ptr(self.description).to_str().unwrap_or("") }
    }

    /// Get the slice of element descriptors.
    ///
    /// # Safety
    ///
    /// The `elements` pointer must be valid and point to `num_elements` items.
    pub unsafe fn elements(&self) -> &[ElementDescriptor] {
        if self.elements.is_null() || self.num_elements == 0 {
            &[]
        } else {
            // SAFETY: Caller guarantees `elements` points to valid array.
            unsafe { std::slice::from_raw_parts(self.elements, self.num_elements as usize) }
        }
    }

    /// Validate that this descriptor is safe to use.
    ///
    /// # Safety
    ///
    /// All pointer fields must be valid.
    pub unsafe fn validate(&self) -> Result<(), &'static str> {
        if self.abi_version != PARALLAX_ABI_VERSION {
            return Err("ABI version mismatch");
        }
        if self.name.is_null() {
            return Err("Plugin name is null");
        }
        if self.version.is_null() {
            return Err("Plugin version is null");
        }
        // Validate each element descriptor
        // SAFETY: We're in an unsafe fn, caller guarantees validity.
        for elem in unsafe { self.elements() } {
            if elem.name.is_null() {
                return Err("Element name is null");
            }
        }
        Ok(())
    }
}

/// Safe Rust representation of plugin information.
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Plugin name.
    pub name: String,
    /// Plugin version.
    pub version: String,
    /// Plugin author.
    pub author: String,
    /// Plugin description.
    pub description: String,
    /// Elements provided by this plugin.
    pub elements: Vec<ElementInfo>,
}

/// Safe Rust representation of element information.
#[derive(Debug, Clone)]
pub struct ElementInfo {
    /// Element name.
    pub name: String,
    /// Element description.
    pub description: String,
    /// Element type (Source, Transform, Sink).
    pub element_type: crate::element::ElementType,
}

impl PluginInfo {
    /// Create PluginInfo from a raw descriptor.
    ///
    /// # Safety
    ///
    /// The descriptor must be valid and all its pointers must be valid.
    pub unsafe fn from_descriptor(desc: &PluginDescriptor) -> Self {
        // SAFETY: Caller guarantees descriptor is valid.
        let elements = unsafe {
            desc.elements()
                .iter()
                .map(|e| ElementInfo {
                    name: e.name_str().to_string(),
                    description: e.description_str().to_string(),
                    element_type: match e.element_type {
                        0 => crate::element::ElementType::Source,
                        2 => crate::element::ElementType::Sink,
                        _ => crate::element::ElementType::Transform,
                    },
                })
                .collect()
        };

        // SAFETY: Caller guarantees descriptor is valid.
        unsafe {
            Self {
                name: desc.name_str().to_string(),
                version: desc.version_str().to_string(),
                author: desc.author_str().to_string(),
                description: desc.description_str().to_string(),
                elements,
            }
        }
    }
}

/// Helper macro for defining plugin descriptors in Rust.
///
/// # Example
///
/// ```ignore
/// use parallax::define_plugin;
///
/// define_plugin! {
///     name: "my_plugin",
///     version: "1.0.0",
///     author: "Me",
///     description: "My awesome plugin",
///     elements: [
///         {
///             name: "myfilter",
///             description: "A filter element",
///             element_type: 1, // 0=Source, 1=Transform, 2=Sink
///             create: || Box::new(MyFilter::new()),
///         }
///     ]
/// }
/// ```
#[macro_export]
macro_rules! define_plugin {
    (
        name: $name:literal,
        version: $version:literal,
        author: $author:literal,
        description: $desc:literal,
        elements: [
            $(
                {
                    name: $elem_name:literal,
                    description: $elem_desc:literal,
                    element_type: $elem_type:expr,
                    create: $create:expr $(,)?
                }
            ),* $(,)?
        ]
    ) => {
        // Element name constants
        $(
            paste::paste! {
                static [<ELEM_NAME_ $elem_name:upper>]: &[u8] = concat!($elem_name, "\0").as_bytes();
                static [<ELEM_DESC_ $elem_name:upper>]: &[u8] = concat!($elem_desc, "\0").as_bytes();

                #[unsafe(no_mangle)]
                extern "C" fn [<create_ $elem_name>]() -> *mut std::ffi::c_void {
                    let creator: fn() -> Box<dyn $crate::element::ElementDyn> = $create;
                    let element = creator();
                    $crate::plugin::element_to_raw(element)
                }
            }
        )*

        // Plugin metadata
        static PLUGIN_NAME: &[u8] = concat!($name, "\0").as_bytes();
        static PLUGIN_VERSION: &[u8] = concat!($version, "\0").as_bytes();
        static PLUGIN_AUTHOR: &[u8] = concat!($author, "\0").as_bytes();
        static PLUGIN_DESC: &[u8] = concat!($desc, "\0").as_bytes();

        // Element descriptors array
        paste::paste! {
            static ELEMENT_DESCRIPTORS: &[$crate::plugin::ElementDescriptor] = &[
                $(
                    $crate::plugin::ElementDescriptor {
                        name: [<ELEM_NAME_ $elem_name:upper>].as_ptr() as *const std::ffi::c_char,
                        description: [<ELEM_DESC_ $elem_name:upper>].as_ptr() as *const std::ffi::c_char,
                        element_type: $elem_type,
                        create: [<create_ $elem_name>],
                        destroy: None,
                    },
                )*
            ];
        }

        // Plugin descriptor
        static PLUGIN_DESCRIPTOR: $crate::plugin::PluginDescriptor = $crate::plugin::PluginDescriptor {
            abi_version: $crate::plugin::PARALLAX_ABI_VERSION,
            name: PLUGIN_NAME.as_ptr() as *const std::ffi::c_char,
            version: PLUGIN_VERSION.as_ptr() as *const std::ffi::c_char,
            author: PLUGIN_AUTHOR.as_ptr() as *const std::ffi::c_char,
            description: PLUGIN_DESC.as_ptr() as *const std::ffi::c_char,
            num_elements: ELEMENT_DESCRIPTORS.len() as u32,
            elements: ELEMENT_DESCRIPTORS.as_ptr(),
        };

        /// Plugin entry point.
        #[unsafe(no_mangle)]
        pub extern "C" fn parallax_plugin_descriptor() -> *const $crate::plugin::PluginDescriptor {
            &PLUGIN_DESCRIPTOR
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abi_version() {
        assert_eq!(PARALLAX_ABI_VERSION, 1);
    }

    #[test]
    fn test_element_descriptor_size() {
        // Ensure the struct has a predictable size for C ABI
        let size = std::mem::size_of::<ElementDescriptor>();
        assert!(size > 0);
    }

    #[test]
    fn test_plugin_descriptor_size() {
        let size = std::mem::size_of::<PluginDescriptor>();
        assert!(size > 0);
    }
}
