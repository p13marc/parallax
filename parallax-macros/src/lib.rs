//! Procedural macros for the Parallax streaming pipeline engine.
//!
//! This crate provides the `#[pipeline_plugin]` attribute macro for easily
//! creating Parallax plugins from Rust elements.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemStruct, LitStr, parse_macro_input};

/// Attribute macro for marking an element as a plugin element.
///
/// This macro generates the necessary boilerplate to expose an element
/// through the plugin system.
///
/// # Example
///
/// ```ignore
/// use parallax::prelude::*;
/// use parallax_macros::pipeline_element;
///
/// #[pipeline_element(name = "myfilter", description = "A custom filter")]
/// pub struct MyFilter {
///     // fields
/// }
///
/// impl Element for MyFilter {
///     fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
///         Ok(Some(buffer))
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn pipeline_element(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let args = parse_macro_input!(attr as ElementArgs);

    let struct_name = &input.ident;
    let element_name = args
        .name
        .unwrap_or_else(|| struct_name.to_string().to_lowercase());
    let element_desc = args
        .description
        .unwrap_or_else(|| format!("{} element", struct_name));
    let element_type = args.element_type.unwrap_or_else(|| "Transform".to_string());

    let create_fn_name = format_ident!("__parallax_create_{}", element_name.replace('-', "_"));
    let name_const = format_ident!(
        "__PARALLAX_ELEM_NAME_{}",
        element_name.to_uppercase().replace('-', "_")
    );
    let desc_const = format_ident!(
        "__PARALLAX_ELEM_DESC_{}",
        element_name.to_uppercase().replace('-', "_")
    );

    let element_type_value = match element_type.as_str() {
        "Source" => 0i32,
        "Sink" => 2i32,
        _ => 1i32, // Transform
    };

    let expanded = quote! {
        #input

        #[doc(hidden)]
        static #name_const: &[u8] = concat!(#element_name, "\0").as_bytes();

        #[doc(hidden)]
        static #desc_const: &[u8] = concat!(#element_desc, "\0").as_bytes();

        #[doc(hidden)]
        #[no_mangle]
        pub unsafe extern "C" fn #create_fn_name() -> *mut std::ffi::c_void {
            let element: Box<dyn parallax::element::ElementDyn> =
                Box::new(parallax::element::ElementAdapter::new(#struct_name::default()));
            parallax::plugin::element_to_raw(element)
        }

        impl #struct_name {
            /// Get the element descriptor for this element.
            #[doc(hidden)]
            pub const fn __parallax_element_descriptor() -> parallax::plugin::ElementDescriptor {
                parallax::plugin::ElementDescriptor {
                    name: #name_const.as_ptr() as *const std::ffi::c_char,
                    description: #desc_const.as_ptr() as *const std::ffi::c_char,
                    element_type: #element_type_value,
                    create: #create_fn_name,
                    destroy: None,
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Arguments for the pipeline_element attribute.
struct ElementArgs {
    name: Option<String>,
    description: Option<String>,
    element_type: Option<String>,
}

impl syn::parse::Parse for ElementArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut description = None;
        let mut element_type = None;

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            let _: syn::Token![=] = input.parse()?;

            match ident.to_string().as_str() {
                "name" => {
                    let lit: LitStr = input.parse()?;
                    name = Some(lit.value());
                }
                "description" => {
                    let lit: LitStr = input.parse()?;
                    description = Some(lit.value());
                }
                "element_type" => {
                    let lit: LitStr = input.parse()?;
                    element_type = Some(lit.value());
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown attribute: {}", other),
                    ));
                }
            }

            if input.peek(syn::Token![,]) {
                let _: syn::Token![,] = input.parse()?;
            }
        }

        Ok(ElementArgs {
            name,
            description,
            element_type,
        })
    }
}

/// Derive macro for creating a complete plugin from multiple elements.
///
/// # Example
///
/// ```ignore
/// use parallax_macros::plugin;
///
/// plugin! {
///     name: "my_plugin",
///     version: "1.0.0",
///     author: "Me",
///     description: "My awesome plugin",
///     elements: [MyFilter, MySource]
/// }
/// ```
#[proc_macro]
pub fn plugin(input: TokenStream) -> TokenStream {
    let plugin_def = parse_macro_input!(input as PluginDef);

    let name = &plugin_def.name;
    let version = &plugin_def.version;
    let author = &plugin_def.author;
    let description = &plugin_def.description;
    let elements = &plugin_def.elements;

    let num_elements = elements.len() as u32;

    let element_descriptors = elements.iter().map(|elem| {
        quote! {
            #elem::__parallax_element_descriptor()
        }
    });

    let expanded = quote! {
        #[doc(hidden)]
        static __PARALLAX_PLUGIN_NAME: &[u8] = concat!(#name, "\0").as_bytes();
        #[doc(hidden)]
        static __PARALLAX_PLUGIN_VERSION: &[u8] = concat!(#version, "\0").as_bytes();
        #[doc(hidden)]
        static __PARALLAX_PLUGIN_AUTHOR: &[u8] = concat!(#author, "\0").as_bytes();
        #[doc(hidden)]
        static __PARALLAX_PLUGIN_DESC: &[u8] = concat!(#description, "\0").as_bytes();

        #[doc(hidden)]
        static __PARALLAX_ELEMENT_DESCRIPTORS: &[parallax::plugin::ElementDescriptor] = &[
            #(#element_descriptors),*
        ];

        #[doc(hidden)]
        static __PARALLAX_PLUGIN_DESCRIPTOR: parallax::plugin::PluginDescriptor =
            parallax::plugin::PluginDescriptor {
                abi_version: parallax::plugin::PARALLAX_ABI_VERSION,
                name: __PARALLAX_PLUGIN_NAME.as_ptr() as *const std::ffi::c_char,
                version: __PARALLAX_PLUGIN_VERSION.as_ptr() as *const std::ffi::c_char,
                author: __PARALLAX_PLUGIN_AUTHOR.as_ptr() as *const std::ffi::c_char,
                description: __PARALLAX_PLUGIN_DESC.as_ptr() as *const std::ffi::c_char,
                num_elements: #num_elements,
                elements: __PARALLAX_ELEMENT_DESCRIPTORS.as_ptr(),
            };

        /// Plugin entry point.
        #[no_mangle]
        pub extern "C" fn parallax_plugin_descriptor() -> *const parallax::plugin::PluginDescriptor {
            &__PARALLAX_PLUGIN_DESCRIPTOR
        }
    };

    TokenStream::from(expanded)
}

/// Definition for a plugin.
struct PluginDef {
    name: String,
    version: String,
    author: String,
    description: String,
    elements: Vec<syn::Ident>,
}

impl syn::parse::Parse for PluginDef {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut version = None;
        let mut author = None;
        let mut description = None;
        let mut elements = Vec::new();

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            let _: syn::Token![:] = input.parse()?;

            match ident.to_string().as_str() {
                "name" => {
                    let lit: LitStr = input.parse()?;
                    name = Some(lit.value());
                }
                "version" => {
                    let lit: LitStr = input.parse()?;
                    version = Some(lit.value());
                }
                "author" => {
                    let lit: LitStr = input.parse()?;
                    author = Some(lit.value());
                }
                "description" => {
                    let lit: LitStr = input.parse()?;
                    description = Some(lit.value());
                }
                "elements" => {
                    let content;
                    syn::bracketed!(content in input);
                    while !content.is_empty() {
                        let elem: syn::Ident = content.parse()?;
                        elements.push(elem);
                        if content.peek(syn::Token![,]) {
                            let _: syn::Token![,] = content.parse()?;
                        }
                    }
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown field: {}", other),
                    ));
                }
            }

            if input.peek(syn::Token![,]) {
                let _: syn::Token![,] = input.parse()?;
            }
        }

        Ok(PluginDef {
            name: name.ok_or_else(|| syn::Error::new(input.span(), "missing 'name' field"))?,
            version: version
                .ok_or_else(|| syn::Error::new(input.span(), "missing 'version' field"))?,
            author: author
                .ok_or_else(|| syn::Error::new(input.span(), "missing 'author' field"))?,
            description: description
                .ok_or_else(|| syn::Error::new(input.span(), "missing 'description' field"))?,
            elements,
        })
    }
}
