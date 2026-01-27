//! # Typed Pipeline
//!
//! Type-safe pipeline API with compile-time checked operations.
//! Uses typed sources, transforms, and sinks instead of raw buffers.
//!
//! Run: `cargo run --example 09_typed`

use parallax::error::Result;
use parallax::typed::{CollectSink, PipelineWithSource, from_iter, map};

fn main() -> Result<()> {
    println!("=== Typed Pipeline Examples ===\n");

    // Example 1: Simple transformation chain
    println!("--- Map and Collect ---");
    let source = from_iter(1..=5);
    let sink = PipelineWithSource::new(source)
        .then(map(|x: i32| x * 2)) // Double each value
        .sink(CollectSink::new())
        .run()?;
    let results = sink.into_inner();
    println!("1..5 doubled: {:?}", results);

    // Example 2: Chained transforms
    println!("\n--- Chained Transforms ---");
    let source = from_iter(1..=5);
    let sink = PipelineWithSource::new(source)
        .then(map(|x: i32| x * x)) // Square
        .then(map(|x: i32| x + 1)) // Add 1
        .sink(CollectSink::new())
        .run()?;
    let results = sink.into_inner();
    println!("(n² + 1) for n=1..5: {:?}", results);
    println!("Sum: {}", results.iter().sum::<i32>());

    // Example 3: String processing
    println!("\n--- String Processing ---");
    let source = from_iter(vec!["hello", "typed", "pipeline"]);
    let sink = PipelineWithSource::new(source)
        .then(map(|s: &str| s.to_uppercase()))
        .sink(CollectSink::new())
        .run()?;
    let results = sink.into_inner();
    println!("Uppercased: {:?}", results);

    Ok(())
}
