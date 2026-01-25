//! Typed pipeline example with compile-time safety.
//!
//! This example demonstrates building pipelines using the typed API,
//! which provides compile-time type checking for element connections.
//!
//! Run with: cargo run --example typed_pipeline

use parallax::error::Result;
use parallax::typed::{collect, filter, filter_map, from_iter, inspect, map, pipeline, skip, take};

fn main() -> Result<()> {
    println!("=== Typed Pipeline Example ===\n");

    // Example 1: Simple map operation
    println!("1. Map operation: multiply by 2");
    run_map_example()?;

    // Example 2: Filter operation
    println!("\n2. Filter operation: keep even numbers");
    run_filter_example()?;

    // Example 3: Chained operations
    println!("\n3. Chained operations: filter -> map -> take");
    run_chained_example()?;

    // Example 4: FilterMap operation
    println!("\n4. FilterMap: parse strings to numbers");
    run_filter_map_example()?;

    // Example 5: Complex pipeline
    println!("\n5. Complex pipeline with multiple stages");
    run_complex_example()?;

    // Example 6: Using >> operator
    println!("\n6. Using >> operator for pipeline building");
    run_operator_example()?;

    println!("\n=== All typed examples completed ===");
    Ok(())
}

fn run_map_example() -> Result<()> {
    // Create a pipeline that multiplies each number by 2
    let source = from_iter(vec![1i32, 2, 3, 4, 5]);

    let sink = pipeline(source)
        .then(map(|x: i32| x * 2))
        .sink(collect::<i32>());

    let results = sink.run()?.into_inner();
    println!("   Input:  [1, 2, 3, 4, 5]");
    println!("   Output: {:?}", results);
    Ok(())
}

fn run_filter_example() -> Result<()> {
    // Create a pipeline that keeps only even numbers
    let source = from_iter(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    let sink = pipeline(source)
        .then(filter(|x: &i32| x % 2 == 0))
        .sink(collect::<i32>());

    let results = sink.run()?.into_inner();
    println!("   Input:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
    println!("   Output: {:?}", results);
    Ok(())
}

fn run_chained_example() -> Result<()> {
    // Create a pipeline with multiple chained operations
    let source = from_iter(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    let sink = pipeline(source)
        .then(filter(|x: &i32| x % 2 == 0)) // Keep even: [2, 4, 6, 8, 10]
        .then(map(|x: i32| x * 10)) // Multiply: [20, 40, 60, 80, 100]
        .then(take(3)) // Take first 3: [20, 40, 60]
        .sink(collect::<i32>());

    let results = sink.run()?.into_inner();
    println!("   Input:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
    println!("   Steps:  filter(even) -> map(*10) -> take(3)");
    println!("   Output: {:?}", results);
    Ok(())
}

fn run_filter_map_example() -> Result<()> {
    // Create a pipeline that parses strings to numbers, filtering failures
    let source = from_iter(vec![
        "1".to_string(),
        "two".to_string(),
        "3".to_string(),
        "four".to_string(),
        "5".to_string(),
    ]);

    let sink = pipeline(source)
        .then(filter_map(|s: String| s.parse::<i32>().ok()))
        .sink(collect::<i32>());

    let results = sink.run()?.into_inner();
    println!("   Input:  [\"1\", \"two\", \"3\", \"four\", \"5\"]");
    println!("   Output: {:?}", results);
    Ok(())
}

fn run_complex_example() -> Result<()> {
    // A more complex example with custom types
    #[derive(Debug, Clone)]
    struct SensorReading {
        sensor_id: u32,
        value: f64,
        timestamp: u64,
    }

    // Generate simulated sensor data
    let readings: Vec<SensorReading> = (0..20)
        .map(|i| SensorReading {
            sensor_id: i % 3,
            value: (i as f64) * 1.5 + ((i % 5) as f64),
            timestamp: 1000 + i as u64,
        })
        .collect();

    let source = from_iter(readings);

    let sink = pipeline(source)
        // Only sensor 1
        .then(filter(|r: &SensorReading| r.sensor_id == 1))
        // Only readings above threshold
        .then(filter(|r: &SensorReading| r.value > 10.0))
        // Add debug output
        .then(inspect(|r: &SensorReading| {
            println!(
                "      Processing: sensor={}, value={:.1}, ts={}",
                r.sensor_id, r.value, r.timestamp
            );
        }))
        // Skip first result
        .then(skip(1))
        // Take up to 3
        .then(take(3))
        // Format as string
        .then(map(|r: SensorReading| {
            format!("Sensor {} @ {}: {:.2}", r.sensor_id, r.timestamp, r.value)
        }))
        .sink(collect::<String>());

    let results = sink.run()?.into_inner();
    println!("   Final results:");
    for result in results {
        println!("      {}", result);
    }
    Ok(())
}

fn run_operator_example() -> Result<()> {
    // Using the >> operator for a cleaner syntax
    let source = from_iter(vec![1i32, 2, 3, 4, 5, 6]);

    // pipeline >> transform >> transform >> ...
    let pipe = pipeline(source) >> filter(|x: &i32| *x % 2 == 0) >> map(|x: i32| x * 100);

    let results = pipe.sink(collect()).run()?.into_inner();
    println!("   Input:  [1, 2, 3, 4, 5, 6]");
    println!("   Chain:  filter(even) >> map(*100)");
    println!("   Output: {:?}", results);
    Ok(())
}
