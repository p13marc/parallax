//! # File I/O
//!
//! Read from a file and write to another file using FileSrc and FileSink.
//!
//! ```text
//! [FileSrc] → [FileSink]
//! ```
//!
//! Run: `cargo run --example 07_file_io`

use parallax::elements::{FileSink, FileSrc};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::pipeline::Pipeline;
use std::io::Write;
use tempfile::tempdir;

#[tokio::main]
async fn main() -> Result<()> {
    let dir = tempdir()?;
    let input_path = dir.path().join("input.txt");
    let output_path = dir.path().join("output.txt");

    // Create input file
    {
        let mut f = std::fs::File::create(&input_path)?;
        writeln!(f, "Line 1: Hello from Parallax!")?;
        writeln!(f, "Line 2: File I/O is working.")?;
        writeln!(f, "Line 3: End of file.")?;
    }
    println!("Created input file: {:?}", input_path);

    // Build pipeline
    let arena = SharedArena::new(4096, 8)?;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source_with_arena("filesrc", FileSrc::new(&input_path), arena);
    let sink = pipeline.add_sink("filesink", FileSink::new(&output_path));
    pipeline.link(src, sink)?;

    // Run
    pipeline.run().await?;

    // Verify output
    let output = std::fs::read_to_string(&output_path)?;
    println!("\nOutput file contents:");
    print!("{}", output);

    Ok(())
}
