//! Codec-adjacent utilities that depend on no codec library.
//!
//! Everything here is pure byte arithmetic over encoded bitstreams, so it is
//! compiled unconditionally — unlike [`crate::elements::codec`], which exists
//! only when a codec feature is enabled. A consumer that receives an H.264
//! packet over the network must be able to ask whether it is a keyframe without
//! linking an encoder.

pub mod annexb;
