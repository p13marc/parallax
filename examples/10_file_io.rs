//! File source and sink for reading/writing files.
//!
//! Run with: cargo run --example 10_file_io

use parallax::element::{DynAsyncElement, SinkAdapter, SourceAdapter};
use parallax::elements::{FileSink, FileSrc};
use parallax::error::Result;
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
        writeln!(f, "Hello from file!")?;
        writeln!(f, "Line 2")?;
        writeln!(f, "Line 3")?;
    }

    println!(
        "Copying {} -> {}",
        input_path.display(),
        output_path.display()
    );

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "filesrc",
        DynAsyncElement::new_box(SourceAdapter::new(FileSrc::new(&input_path))),
    );
    let sink = pipeline.add_node(
        "filesink",
        DynAsyncElement::new_box(SinkAdapter::new(FileSink::new(&output_path))),
    );

    pipeline.link(src, sink)?;
    pipeline.run().await?;

    // Verify output
    let content = std::fs::read_to_string(&output_path)?;
    println!();
    println!("Output file content:");
    print!("{}", content);

    Ok(())
}
