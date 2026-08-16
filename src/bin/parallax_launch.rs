//! `parallax-launch` — build and run a pipeline from a gst-launch-style
//! description string.
//!
//! ```text
//! parallax-launch videotestsrc num-buffers=100 ! videoconvert ! nullsink
//! parallax-launch -v filesrc location=in.bin ! passthrough ! filesink location=out.bin
//! parallax-launch --list-elements
//! ```
//!
//! Exit codes: 0 = ran to EOS, 1 = parse/startup/runtime error, 2 = usage,
//! 130 = interrupted (ctrl-C).

use std::process::ExitCode;

use clap::Parser;
use futures::StreamExt;
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::{ElementFactory, Executor, Pipeline, PipelineHandle};

#[derive(Parser)]
#[command(
    name = "parallax-launch",
    version,
    about = "Build and run a parallax pipeline from a description string",
    long_about = "Build and run a parallax pipeline from a gst-launch-style description.\n\
                  The grammar is a strictly linear chain: `elem prop=val ! elem ...` —\n\
                  no caps filters, no branching, no bins. Fan-out and container\n\
                  demuxing need the programmatic API."
)]
struct Args {
    /// Print every bus message (default prints only warnings and errors)
    #[arg(short, long, conflicts_with = "quiet")]
    verbose: bool,

    /// Print only errors, to stderr
    #[arg(short, long)]
    quiet: bool,

    /// List available elements (and unavailable feature-gated ones) and exit
    #[arg(long)]
    list_elements: bool,

    /// Print the parsed pipeline as Graphviz DOT and exit without running
    #[arg(long)]
    dot: bool,

    /// The pipeline description (quoting optional: trailing words are joined)
    #[arg(trailing_var_arg = true)]
    pipeline: Vec<String>,
}

fn list_elements() {
    let factory = ElementFactory::new();
    println!("Available elements:");
    for name in factory.list_elements() {
        println!("  {name}");
    }
    let unavailable = factory.unavailable_elements();
    if !unavailable.is_empty() {
        println!("\nUnavailable in this build (rebuild with the named cargo feature):");
        for (name, feature) in unavailable {
            println!("  {name}  (requires feature \"{feature}\")");
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args = Args::parse();

    if args.list_elements {
        list_elements();
        return ExitCode::SUCCESS;
    }

    let description = args.pipeline.join(" ");
    if description.trim().is_empty() {
        eprintln!("error: no pipeline description given (try --help)");
        return ExitCode::from(2);
    }

    let mut pipeline = match Pipeline::parse(&description) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    if args.dot {
        println!("{}", pipeline.to_dot());
        return ExitCode::SUCCESS;
    }

    // Take the bus before start so the printer sees everything.
    let bus = pipeline.take_bus();

    let executor = Executor::new();
    let handle: PipelineHandle = match executor.start(&mut pipeline) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: failed to start pipeline: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Bus printer. -v prints everything; default prints warnings/errors;
    // -q prints errors only. Errors always go to stderr.
    let verbose = args.verbose;
    let quiet = args.quiet;
    let printer = bus.map(|bus| {
        tokio::spawn(async move {
            let mut stream = bus.into_stream();
            while let Some(msg) = stream.next().await {
                match &msg.kind {
                    MessageKind::Error { .. } => eprintln!("{msg}"),
                    MessageKind::Warning { .. } if !quiet => println!("{msg}"),
                    _ if verbose => println!("{msg}"),
                    _ => {}
                }
            }
        })
    });

    let mut interrupted = false;
    let code = loop {
        tokio::select! {
            reason = handle.ended() => {
                use parallax::pipeline::EndReason;
                break match reason {
                    _ if interrupted => ExitCode::from(130),
                    EndReason::Eos => ExitCode::SUCCESS,
                    EndReason::Error(e) => {
                        eprintln!("pipeline error: {e}");
                        ExitCode::FAILURE
                    }
                    EndReason::Aborted => ExitCode::from(130),
                };
            }
            _ = tokio::signal::ctrl_c() => {
                if interrupted {
                    // Second ctrl-C: give up on graceful teardown.
                    std::process::exit(130);
                }
                interrupted = true;
                eprintln!("interrupted — stopping (ctrl-C again to force)");
                handle.stop();
            }
        }
    };

    if let Some(p) = printer {
        p.abort();
    }
    code
}
