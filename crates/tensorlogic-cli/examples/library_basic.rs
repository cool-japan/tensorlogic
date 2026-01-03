//! Basic example of using tensorlogic-cli as a library
//!
//! This example demonstrates:
//! - Parsing logical expressions
//! - Creating a compiler context
//! - Compiling expressions to einsum graphs
//! - Analyzing graph complexity
//!
//! Run with: cargo run --example library_basic

use tensorlogic_cli::{analysis, parser, CompilationContext};
use tensorlogic_compiler::compile_to_einsum_with_context;

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic Library Mode - Basic Example ===\n");

    // 1. Parse a logical expression
    println!("1. Parsing expression...");
    let expression = "AND(knows(Alice, Bob), knows(Bob, Charlie))";
    println!("   Expression: {}", expression);

    let expr = parser::parse_expression(expression)?;
    println!("   Parsed successfully: {:?}\n", expr);

    // 2. Set up compilation context
    println!("2. Setting up compiler context...");
    let mut context = CompilationContext::new();

    // Add domain for people
    context.add_domain("Person", 100);
    println!("   Added domain 'Person' with 100 entities");
    println!("   Using default compilation strategy\n");

    // 3. Compile to einsum graph
    println!("3. Compiling to einsum graph...");
    let graph = compile_to_einsum_with_context(&expr, &mut context)?;
    println!("   Compilation successful!");
    println!(
        "   Graph has {} tensors and {} nodes\n",
        graph.tensors.len(),
        graph.nodes.len()
    );

    // 4. Analyze graph complexity
    println!("4. Analyzing graph complexity...");
    let metrics = analysis::GraphMetrics::analyze(&graph);

    println!("   Tensor count: {}", metrics.tensor_count);
    println!("   Node count: {}", metrics.node_count);
    println!("   Graph depth: {}", metrics.depth);
    println!("   Average fanout: {:.2}", metrics.avg_fanout);
    println!(
        "   Estimated FLOPs: {}",
        analysis::format_number(metrics.estimated_flops)
    );
    println!(
        "   Estimated memory: {}",
        analysis::format_bytes(metrics.estimated_memory)
    );

    println!("\n=== Example completed successfully! ===");

    Ok(())
}
