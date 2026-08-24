//! Consume the zenkey `@media` plane: a tier key in, decoded video out (#228).
//!
//! This is parallax's side of the multi-machine story. Something else — a
//! camera daemon, `zensight-sensor-parallax` — publishes encoded video on an
//! `@media` tier key; this subscribes to **one exact tier**, decodes it, and
//! displays it. The whole pipeline is a string:
//!
//! ```text
//! zenohsrc key=<tier> profile=media ! h264dec ! autovideosink
//! ```
//!
//! Two things make that string work, and both are the point of the example:
//!
//! - **`Pipeline::parse_async`.** `zenohsrc` builds by awaiting real I/O — it
//!   opens a session and declares a subscriber — so it cannot be built from
//!   the synchronous `parse`. `parallax-launch` uses the same entry point,
//!   which is why `parallax-launch "zenohsrc key=… ! h264dec ! autovideosink"`
//!   is a working command line rather than an aspiration.
//! - **Caps come from the key, not from the wire.** The codec is a literal
//!   chunk of an `@media` tier key, so `prepare()` — which is synchronous and
//!   runs before a single sample arrives — knows the format is H.264. Caps
//!   derived from received data would be `Video(H264)` on one run and `Any` on
//!   the next depending on network timing, and the same pipeline string would
//!   pass under `ConverterPolicy::Deny` once and fail the next time.
//!
//! The key must name one exact tier. A wildcard is refused at construction,
//! not warned about: `…/video/h264/*` delivers every tier of the stream
//! interleaved on one subscriber, unseparable except by re-parsing the key per
//! sample — the opposite of demand-driven simulcast. Choosing the tier *is*
//! choosing the quality.
//!
//! Run:
//! ```text
//! cargo run --example 63_media_consume --features zenoh,h264,display -- \
//!     zensight/v1/h-3fa9c2d41b7e/@media/parallax/cam0/video/h264/high
//! ```
//!
//! With no argument it prints the pipeline it would run and exits, so the
//! example is useful without a publisher on the network.

use parallax::error::Result;
use parallax::pipeline::Pipeline;

/// Where the frames go once decoded. `autovideosink` needs the `display`
/// feature; without it the example still proves the interesting half — that
/// the string parses, negotiates and runs — into a sink that counts.
#[cfg(feature = "display")]
const SINK: &str = "autovideosink";
#[cfg(not(feature = "display"))]
const SINK: &str = "nullsink";

#[tokio::main]
async fn main() -> Result<()> {
    let key = std::env::args().nth(1);

    let Some(key) = key else {
        eprintln!(
            "usage: 63_media_consume <@media tier key>\n\n\
             example:\n  \
             63_media_consume zensight/v1/h-abc/@media/parallax/cam0/video/h264/high\n\n\
             the pipeline it would build:\n  \
             zenohsrc key=<key> profile=media ! h264dec ! {SINK}\n\n\
             the same thing from the command line:\n  \
             parallax-launch \"zenohsrc key=<key> profile=media ! h264dec ! {SINK}\""
        );
        return Ok(());
    };

    let description = format!("zenohsrc key={key} profile=media ! h264dec ! {SINK}");
    println!("running: {description}");

    // `parse_async`, not `parse`: `zenohsrc`'s constructor awaits real I/O.
    let mut pipeline = Pipeline::parse_async(&description).await?;

    // Ctrl-C ends it gracefully. `stopper()` outlives the handle that
    // `wait()` consumes, which is why it exists (#237) — no source wrapper
    // needed.
    let handle = pipeline.start()?;
    let stopper = handle.stopper();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            println!("\nstopping…");
            stopper.stop();
        }
    });

    let ended = handle.ended();
    handle.wait().await?;
    println!("ended: {:?}", ended.await);
    Ok(())
}
