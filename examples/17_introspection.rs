//! Example: Pipeline Introspection and Caps Negotiation
//!
//! This example demonstrates how to:
//! - List all elements in a pipeline
//! - List all links with their negotiated formats
//! - Run caps negotiation explicitly
//! - Query negotiated formats and memory types
//! - Get a human-readable pipeline description
//!
//! Run with: cargo run --example 17_introspection

use parallax::Result;
use parallax::pipeline::Pipeline;

fn main() -> Result<()> {
    // Create a pipeline with multiple elements
    let mut pipeline =
        Pipeline::parse("nullsource count=5 ! passthrough ! passthrough ! nullsink")?;

    println!("=== Pipeline Structure ===\n");

    // List all elements using nodes() iterator
    println!("Elements ({} total):", pipeline.node_count());
    for (id, node) in pipeline.nodes() {
        println!(
            "  [{}] {} ({:?})",
            id.index(),
            node.name(),
            node.element_type()
        );
        println!("       input caps:  {:?}", node.input_caps());
        println!("       output caps: {:?}", node.output_caps());
    }

    println!("\n=== Links Before Negotiation ===\n");

    // List all links - before negotiation, formats are None
    println!("Links ({} total):", pipeline.edge_count());
    for link in pipeline.links() {
        println!(
            "  {} ({}) -> {} ({})",
            link.source_name, link.source_pad, link.sink_name, link.sink_pad
        );
        println!(
            "       format: {:?}, memory: {:?}",
            link.negotiated_format, link.negotiated_memory
        );
    }

    println!("\n=== Running Negotiation ===\n");

    // Check negotiation status
    println!("Is negotiated: {}", pipeline.is_negotiated());

    // Run caps negotiation explicitly
    pipeline.negotiate()?;

    println!("Negotiation complete!");
    println!("Is negotiated: {}", pipeline.is_negotiated());

    println!("\n=== Links After Negotiation ===\n");

    // Now links have negotiated formats
    for link in pipeline.links() {
        println!("  {} -> {}", link.source_name, link.sink_name);
        if let Some(format) = &link.negotiated_format {
            println!("       format: {:?}", format);
        }
        if let Some(memory) = link.negotiated_memory {
            println!("       memory: {:?}", memory);
        }
    }

    // Query specific link format
    println!("\n=== Querying Specific Link ===\n");

    let links: Vec<_> = pipeline.links().collect();
    if let Some(first_link) = links.first() {
        let link_id = first_link.id;
        println!("First link ID: {}", link_id.index());
        println!("  format: {:?}", pipeline.link_format(link_id));
        println!("  memory: {:?}", pipeline.link_memory_type(link_id));
    }

    // Check for pending converters (none in this case since all caps are compatible)
    let converters = pipeline.pending_converters();
    if converters.is_empty() {
        println!("\nNo format converters needed.");
    } else {
        println!("\nConverters to insert:");
        for conv in converters {
            println!(
                "  Link {}: {} (cost: {})",
                conv.link_id, conv.reason, conv.cost
            );
        }
    }

    println!("\n=== Human-Readable Description ===\n");

    // Get a complete human-readable description
    println!("{}", pipeline.describe());

    println!("=== DOT Export ===\n");

    // Can also export to DOT format for visualization
    println!("DOT format (use with Graphviz):");
    println!("{}", pipeline.to_dot());

    Ok(())
}
