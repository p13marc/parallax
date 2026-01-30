//! Element Retrieval and Downcasting Example
//!
//! Demonstrates how to retrieve and modify elements after pipeline creation,
//! similar to GStreamer's `gst_bin_get_by_name()`.
//!
//! Run with:
//!   cargo run --example 49_element_retrieval
//!
//! This example shows:
//! 1. Retrieving elements by auto-generated names (type_index format)
//! 2. Using custom names via the `name=` property
//! 3. Downcasting to concrete element types
//! 4. Modifying element properties after pipeline creation

use parallax::elements::PassThrough;
use parallax::elements::io::{FileSink, FileSrc};
use parallax::elements::testing::NullSink;
use parallax::error::Result;
use parallax::pipeline::Pipeline;

fn main() -> Result<()> {
    println!("Element Retrieval and Downcasting Example");
    println!("==========================================");
    println!();

    // ========================================================================
    // Example 1: Auto-generated element names
    // ========================================================================
    println!("1. Auto-generated element names");
    println!("--------------------------------");

    // When parsing a pipeline, elements get auto-generated names:
    // format: {element_type}_{index}
    let pipeline = Pipeline::parse("nullsource ! passthrough ! nullsink")?;

    println!("   Pipeline: nullsource ! passthrough ! nullsink");
    println!("   Auto-generated names:");
    println!("     - nullsource_0");
    println!("     - passthrough_1");
    println!("     - nullsink_2");

    // Retrieve the passthrough element by its auto-generated name
    if pipeline
        .get_element::<PassThrough>("passthrough_1")
        .is_some()
    {
        println!("   Successfully retrieved PassThrough element!");
    }

    println!();

    // ========================================================================
    // Example 2: Custom element names using name= property
    // ========================================================================
    println!("2. Custom element names using name= property");
    println!("---------------------------------------------");

    // Use the `name=` property to give elements predictable names
    let pipeline = Pipeline::parse(
        "filesrc name=source location=input.bin ! passthrough name=filter ! filesink name=output location=output.bin",
    )?;

    println!(
        "   Pipeline: filesrc name=source ... ! passthrough name=filter ! filesink name=output ..."
    );
    println!("   Custom names: source, filter, output");

    // Retrieve elements by their custom names
    if let Some(source) = pipeline.get_element::<FileSrc>("source") {
        println!(
            "   Retrieved FileSrc 'source': reading from {:?}",
            source.path()
        );
    }

    if let Some(output) = pipeline.get_element::<FileSink>("output") {
        println!(
            "   Retrieved FileSink 'output': writing to {:?}",
            output.path()
        );
    }

    println!();

    // ========================================================================
    // Example 3: Mutable access and property modification
    // ========================================================================
    println!("3. Mutable access and property modification");
    println!("--------------------------------------------");

    let mut pipeline =
        Pipeline::parse("filesrc name=src location=original.bin ! passthrough ! nullsink")?;

    println!("   Initial path: original.bin");

    // Get mutable reference and modify the element
    if let Some(filesrc) = pipeline.get_element_mut::<FileSrc>("src") {
        // Modify the file path
        *filesrc = FileSrc::new("modified.bin");
        println!("   Modified path: modified.bin");
    }

    // Verify the change
    if let Some(filesrc) = pipeline.get_element::<FileSrc>("src") {
        println!("   Verified path: {:?}", filesrc.path());
    }

    println!();

    // ========================================================================
    // Example 4: Type-safe downcasting (wrong type returns None)
    // ========================================================================
    println!("4. Type-safe downcasting");
    println!("-------------------------");

    let pipeline = Pipeline::parse("filesrc name=src location=test.bin ! nullsink name=sink")?;

    // Correct type works
    let correct = pipeline.get_element::<FileSrc>("src");
    println!(
        "   get_element::<FileSrc>(\"src\") = {}",
        if correct.is_some() {
            "Some(...)"
        } else {
            "None"
        }
    );

    // Wrong type returns None (no runtime panic)
    let wrong = pipeline.get_element::<NullSink>("src");
    println!(
        "   get_element::<NullSink>(\"src\") = {}",
        if wrong.is_some() { "Some(...)" } else { "None" }
    );

    // Non-existent element returns None
    let missing = pipeline.get_element::<FileSrc>("nonexistent");
    println!(
        "   get_element::<FileSrc>(\"nonexistent\") = {}",
        if missing.is_some() {
            "Some(...)"
        } else {
            "None"
        }
    );

    println!();

    // ========================================================================
    // Example 5: Using with programmatic pipeline construction
    // ========================================================================
    println!("5. Programmatic pipeline construction");
    println!("--------------------------------------");

    let mut pipeline = Pipeline::new();

    // Add elements with explicit names (returns NodeId for linking)
    let src = pipeline.add_source("my_source", FileSrc::new("data.bin"));
    let filter = pipeline.add_filter("my_filter", PassThrough::new());
    let sink = pipeline.add_sink("my_sink", NullSink::new());

    // Link them using NodeIds
    pipeline.link(src, filter)?;
    pipeline.link(filter, sink)?;

    // Retrieve by name
    if pipeline.get_element::<FileSrc>("my_source").is_some() {
        println!("   Retrieved 'my_source' from programmatic pipeline");
    }

    println!();
    println!("Example complete!");

    Ok(())
}
