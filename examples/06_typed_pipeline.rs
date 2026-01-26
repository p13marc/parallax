//! Type-safe pipelines with compile-time checking.
//!
//! Run with: cargo run --example 06_typed_pipeline

use parallax::error::Result;
use parallax::typed::{collect, filter, from_iter, map, pipeline, take};

fn main() -> Result<()> {
    println!("Typed pipeline: [1..10] -> filter(even) -> map(*2) -> take(3)");
    println!();

    let source = from_iter(1..=10);
    let sink = pipeline(source)
        .then(filter(|x: &i32| x % 2 == 0)) // Keep even: 2,4,6,8,10
        .then(map(|x: i32| x * 2)) // Double: 4,8,12,16,20
        .then(take(3)) // First 3: 4,8,12
        .sink(collect())
        .run()?;

    let result: Vec<i32> = sink.into_inner();
    println!("Result: {:?}", result);

    Ok(())
}
