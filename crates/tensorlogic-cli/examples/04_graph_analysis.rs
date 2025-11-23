//! Example: Graph Analysis
//!
//! Demonstrates how to analyze compiled graphs for complexity and performance metrics.
//!
//! Run with:
//! ```bash
//! cargo run --example 04_graph_analysis
//! ```

use tensorlogic_cli::{analysis, parser, CompilationContext};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig};

fn analyze_expression(expr_str: &str, description: &str) -> anyhow::Result<()> {
    println!("=== {} ===", description);
    println!("Expression: {}\n", expr_str);

    // Parse and compile
    let expr = parser::parse_expression(expr_str)?;
    let config = CompilationConfig::soft_differentiable();
    let mut ctx = CompilationContext::with_config(config);
    ctx.add_domain("Person", 100);
    ctx.add_domain("City", 50);
    ctx.add_domain("D", 100);

    let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;

    // Analyze the graph
    let metrics = analysis::GraphMetrics::analyze(&graph);

    println!("Graph Structure:");
    println!("  Tensors: {}", metrics.tensor_count);
    println!("  Nodes: {}", metrics.node_count);
    println!("  Inputs: {}", metrics.input_count);
    println!("  Outputs: {}", metrics.output_count);
    println!();

    println!("Complexity Metrics:");
    println!("  Depth (longest path): {}", metrics.depth);
    println!("  Average fanout: {:.2}", metrics.avg_fanout);
    println!();

    println!("Operation Breakdown:");
    for (op_type, count) in &metrics.op_breakdown {
        println!("  {}: {}", op_type, count);
    }
    println!();

    println!("Performance Estimates:");
    println!("  Estimated FLOPs: {}", metrics.estimated_flops);
    println!(
        "  Estimated memory: {} bytes ({:.2} KB)",
        metrics.estimated_memory,
        metrics.estimated_memory as f64 / 1024.0
    );
    println!();

    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic CLI: Graph Analysis Examples ===\n");

    // Example 1: Simple predicate
    analyze_expression("knows(x, y)", "Simple Predicate")?;

    // Example 2: Binary operation
    analyze_expression("knows(x, y) AND likes(y, z)", "Binary Operation (AND)")?;

    // Example 3: Complex nested expression
    analyze_expression(
        "(knows(x, y) AND likes(y, z)) OR (friend(x, z) AND NOT enemy(x, z))",
        "Complex Nested Expression",
    )?;

    // Example 4: Quantified expression
    analyze_expression(
        "EXISTS x IN Person. knows(x, alice)",
        "Existential Quantifier",
    )?;

    // Example 5: Nested quantifiers
    analyze_expression(
        "EXISTS x IN Person. FORALL y IN Person. knows(x, y)",
        "Nested Quantifiers",
    )?;

    // Example 6: Multi-domain expression
    analyze_expression(
        "EXISTS p IN Person. EXISTS c IN City. lives_in(p, c) AND likes(p, c)",
        "Multi-Domain Expression",
    )?;

    println!("=== Analysis Examples Complete ===");
    println!();
    println!("Tips:");
    println!("- Higher depth means more sequential operations");
    println!("- Higher fanout means more parallel opportunities");
    println!("- FLOP estimates help predict computational cost");
    println!("- Memory estimates help predict memory requirements");

    Ok(())
}
