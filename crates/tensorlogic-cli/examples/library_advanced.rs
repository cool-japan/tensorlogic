//! Advanced example showing optimization
//!
//! This example demonstrates:
//! - Parsing complex nested expressions
//! - Graph optimization
//! - Analyzing optimization impact
//!
//! Run with: cargo run --example library_advanced

use tensorlogic_cli::{analysis, optimize, parser, CompilationContext};
use tensorlogic_compiler::compile_to_einsum_with_context;

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic Library Mode - Advanced Example ===\n");

    // 1. Parse a complex expression
    println!("1. Parsing complex expression...");
    let expression = "FORALL x IN Person. (EXISTS y IN Person. (knows(x, y) AND smart(y)))";
    println!("   {}\n", expression);

    let expr = parser::parse_expression(expression)?;

    // 2. Set up compilation context
    let mut context = CompilationContext::new();
    context.add_domain("Person", 100);

    // 3. Compile and analyze initial graph
    println!("2. Compiling initial graph...");
    let graph = compile_to_einsum_with_context(&expr, &mut context)?;

    let metrics_before = analysis::GraphMetrics::analyze(&graph);
    println!("   Initial metrics:");
    println!("     Tensors: {}", metrics_before.tensor_count);
    println!("     Nodes: {}", metrics_before.node_count);
    println!("     Depth: {}", metrics_before.depth);
    println!(
        "     Est. FLOPs: {}",
        analysis::format_number(metrics_before.estimated_flops)
    );
    println!(
        "     Est. Memory: {}\n",
        analysis::format_bytes(metrics_before.estimated_memory)
    );

    // 4. Optimize the graph
    println!("3. Optimizing graph...");

    let opt_config = optimize::OptimizationConfig {
        level: optimize::OptimizationLevel::Aggressive,
        verbose: true,
        show_stats: true,
        enable_cse: true,
        enable_dce: true,
        enable_identity: true,
    };

    let (optimized, _opt_stats) = optimize::optimize_einsum_graph(graph.clone(), &opt_config)?;

    let metrics_after = analysis::GraphMetrics::analyze(&optimized);
    println!("\n   Optimized metrics:");
    println!(
        "     Tensors: {} ({}%)",
        metrics_after.tensor_count,
        ((metrics_after.tensor_count as f64 / metrics_before.tensor_count as f64) * 100.0) as i32
    );
    println!(
        "     Nodes: {} ({}%)",
        metrics_after.node_count,
        ((metrics_after.node_count as f64 / metrics_before.node_count as f64) * 100.0) as i32
    );
    println!(
        "     Est. speedup: {:.2}x\n",
        metrics_before.estimated_flops as f64 / metrics_after.estimated_flops.max(1) as f64
    );

    // 5. Compare different optimization levels
    println!("4. Comparing optimization levels...");

    let levels = vec![
        ("None", optimize::OptimizationLevel::None),
        ("Basic", optimize::OptimizationLevel::Basic),
        ("Standard", optimize::OptimizationLevel::Standard),
        ("Aggressive", optimize::OptimizationLevel::Aggressive),
    ];

    for (name, level) in levels {
        let config = optimize::OptimizationConfig {
            level,
            verbose: false,
            show_stats: false,
            enable_cse: true,
            enable_dce: true,
            enable_identity: true,
        };

        let (opt_graph, _) = optimize::optimize_einsum_graph(graph.clone(), &config)?;
        let metrics = analysis::GraphMetrics::analyze(&opt_graph);

        println!(
            "   {}: {} tensors, {} nodes, {} FLOPs",
            name,
            metrics.tensor_count,
            metrics.node_count,
            analysis::format_number(metrics.estimated_flops)
        );
    }

    println!("\n=== Advanced example completed successfully! ===");

    Ok(())
}
