//! Utility elements.
//!
//! - [`PassThrough`]: Pass buffers unchanged
//! - [`Identity`]: Pass-through with callbacks for debugging

mod identity;
mod passthrough;

pub use identity::{Identity, IdentityStats};
pub use passthrough::PassThrough;
