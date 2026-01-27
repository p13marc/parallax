//! File processing pipeline example.
//!
//! Demonstrates reading a file, processing it through a transform chain,
//! and writing the output to another file. This pattern is similar to
//! what a transcoding pipeline would do.
//!
//! Run with: cargo run --example 22_file_processing

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{DynAsyncElement, Element, ElementAdapter, SinkAdapter, SourceAdapter};
use parallax::elements::io::{FileSink, FileSrc};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use tempfile::NamedTempFile;

/// A transform that converts text to uppercase (simulating a codec).
struct UppercaseTransform;

impl Element for UppercaseTransform {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input = buffer.as_bytes();

        // Create new buffer with uppercase content
        let segment = Arc::new(HeapSegment::new(input.len())?);
        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            for (i, &byte) in input.iter().enumerate() {
                let upper = if byte.is_ascii_lowercase() {
                    byte.to_ascii_uppercase()
                } else {
                    byte
                };
                *ptr.add(i) = upper;
            }
        }

        Ok(Some(Buffer::new(
            MemoryHandle::from_segment(segment),
            buffer.metadata().clone(),
        )))
    }
}

/// A transform that adds line numbers (simulating a filter).
struct LineNumberTransform {
    line_number: u64,
}

impl Element for LineNumberTransform {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input = buffer.as_bytes();

        // Create output with line numbers prepended
        let input_str = String::from_utf8_lossy(input);
        let mut output = String::new();

        for line in input_str.lines() {
            self.line_number += 1;
            output.push_str(&format!("{:04}: {}\n", self.line_number, line));
        }

        // Handle partial lines (no trailing newline)
        if !input.is_empty() && input.last() != Some(&b'\n') && output.ends_with('\n') {
            output.pop();
        }

        // Create new buffer with transformed content
        let output_bytes = output.as_bytes();
        let segment = Arc::new(HeapSegment::new(output_bytes.len())?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_bytes.as_ptr(),
                segment.as_mut_ptr().unwrap(),
                output_bytes.len(),
            );
        }

        Ok(Some(Buffer::new(
            MemoryHandle::from_segment(segment),
            buffer.metadata().clone(),
        )))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("File Processing Pipeline Example");
    println!("================================");
    println!();

    // Create a temporary input file with some text
    let input_file = NamedTempFile::new()?;
    let input_path = input_file.path().to_path_buf();

    std::fs::write(
        &input_path,
        r#"Hello, Parallax!
This is a demonstration of file processing.
The pipeline will:
1. Read this file
2. Convert to uppercase
3. Add line numbers
4. Write to output

Each step is a separate element in the pipeline.
Zero-copy buffers are passed between elements.
"#,
    )?;

    // Create output file
    let output_file = NamedTempFile::new()?;
    let output_path = output_file.path().to_path_buf();

    println!("Input file: {}", input_path.display());
    println!("Output file: {}", output_path.display());
    println!();

    // Build the pipeline: file_src -> uppercase -> line_numbers -> file_sink
    let mut pipeline = Pipeline::new();

    // Source: read from file
    let file_src = FileSrc::new(&input_path).with_chunk_size(1024);
    let src = pipeline.add_node(
        "file_src",
        DynAsyncElement::new_box(SourceAdapter::new(file_src)),
    );

    // Transform 1: uppercase
    let uppercase = pipeline.add_node(
        "uppercase",
        DynAsyncElement::new_box(ElementAdapter::new(UppercaseTransform)),
    );

    // Transform 2: add line numbers
    let line_numbers = pipeline.add_node(
        "line_numbers",
        DynAsyncElement::new_box(ElementAdapter::new(LineNumberTransform { line_number: 0 })),
    );

    // Sink: write to file
    let file_sink = FileSink::new(&output_path);
    let sink = pipeline.add_node(
        "file_sink",
        DynAsyncElement::new_box(SinkAdapter::new(file_sink)),
    );

    // Link the pipeline
    pipeline.link(src, uppercase)?;
    pipeline.link(uppercase, line_numbers)?;
    pipeline.link(line_numbers, sink)?;

    println!("Pipeline: file_src -> uppercase -> line_numbers -> file_sink");
    println!();
    println!("Running pipeline...");

    pipeline.run().await?;

    println!("Pipeline complete!");
    println!();

    // Show the output
    println!("Output file contents:");
    println!("---------------------");
    let output = std::fs::read_to_string(&output_path)?;
    print!("{}", output);

    // Demonstrate the isolation capability
    println!();
    println!("---------------------");
    println!();
    println!("This same pipeline could run with process isolation:");
    println!("  pipeline.run_isolating(vec![\"uppercase\", \"line_numbers\"]).await?");
    println!();
    println!("This would run transforms in sandboxed processes while");
    println!("still using zero-copy shared memory for data transfer.");

    Ok(())
}
