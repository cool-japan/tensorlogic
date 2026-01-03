//! Example: Graph Optimization
//!
//! Demonstrates how to optimize compiled graphs using various optimization passes.
//!
//! Run with:
//! ```bash
//! cargo run --example 05_optimization
//! ```

use tensorlogic_cli::{analysis, optimize, parser, CompilationContext};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig};

fn main() -> anyhow::Result<()> {
    println!("=== TensorLogic CLI: Graph Optimization Examples ===\n");

    // Create a complex expression with potential optimizations
    let expr_str = "(p(x) AND p(x)) OR (q(y) AND NOT NOT q(y))";
    println!("Expression: {}", expr_str);
    println!("This expression has redundant operations:");
    println!("  - p(x) AND p(x) can be simplified to p(x)");
    println!("  - NOT NOT q(y) can be simplified to q(y)");
    println!();

    // Parse and compile
    let expr = parser::parse_expression(expr_str)?;
    let config = CompilationConfig::soft_differentiable();
    let mut ctx = CompilationContext::with_config(config);
    ctx.add_domain("D", 100);

    let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;

    // Analyze original graph
    let orig_metrics = analysis::GraphMetrics::analyze(&graph);
    println!("Original Graph:");
    println!("  Tensors: {}", orig_metrics.tensor_count);
    println!("  Nodes: {}", orig_metrics.node_count);
    println!("  Estimated FLOPs: {}", orig_metrics.estimated_flops);
    println!();

    // Optimization Level: None (baseline)
    println!("=== Optimization Level: None ===");
    let opt_config = optimize::OptimizationConfig {
        level: optimize::OptimizationLevel::None,
        enable_dce: false,
        enable_cse: false,
        enable_identity: false,
        show_stats: true,
        verbose: false,
    };
    let (opt_graph, stats) = optimize::optimize_einsum_graph(graph.clone(), &opt_config)?;
    let opt_metrics = analysis::GraphMetrics::analyze(&opt_graph);
    println!("Optimized Graph:");
    println!("  Tensors: {}", opt_metrics.tensor_count);
    println!("  Nodes: {}", opt_metrics.node_count);
    println!("  Estimated FLOPs: {}", opt_metrics.estimated_flops);
    println!(
        "  Identity simplifications: {}",
        stats.identity_simplifications
    );
    println!("  Estimated speedup: {:.2}x", stats.estimated_speedup);
    println!();

    // Optimization Level: Basic
    println!("=== Optimization Level: Basic ===");
    let opt_config = optimize::OptimizationConfig {
        level: optimize::OptimizationLevel::Basic,
        enable_dce: true,
        enable_cse: false,
        enable_identity: true,
        show_stats: true,
        verbose: false,
    };
    let (opt_graph, stats) = optimize::optimize_einsum_graph(graph.clone(), &opt_config)?;
    let opt_metrics = analysis::GraphMetrics::analyze(&opt_graph);
    println!("Optimized Graph:");
    println!("  Tensors: {}", opt_metrics.tensor_count);
    println!("  Nodes: {}", opt_metrics.node_count);
    println!("  Estimated FLOPs: {}", opt_metrics.estimated_flops);
    println!(
        "  Identity simplifications: {}",
        stats.identity_simplifications
    );
    println!("  Estimated speedup: {:.2}x", stats.estimated_speedup);
    println!();

    // Optimization Level: Standard
    println!("=== Optimization Level: Standard ===");
    let opt_config = optimize::OptimizationConfig {
        level: optimize::OptimizationLevel::Standard,
        enable_dce: true,
        enable_cse: true,
        enable_identity: true,
        show_stats: true,
        verbose: false,
    };
    let (opt_graph, stats) = optimize::optimize_einsum_graph(graph.clone(), &opt_config)?;
    let opt_metrics = analysis::GraphMetrics::analyze(&opt_graph);
    println!("Optimized Graph:");
    println!("  Tensors: {}", opt_metrics.tensor_count);
    println!("  Nodes: {}", opt_metrics.node_count);
    println!("  Estimated FLOPs: {}", opt_metrics.estimated_flops);
    println!(
        "  Identity simplifications: {}",
        stats.identity_simplifications
    );
    println!("  Estimated speedup: {:.2}x", stats.estimated_speedup);
    println!();

    // Optimization Level: Aggressive
    println!("=== Optimization Level: Aggressive ===");
    let opt_config = optimize::OptimizationConfig {
        level: optimize::OptimizationLevel::Aggressive,
        enable_dce: true,
        enable_cse: true,
        enable_identity: true,
        show_stats: true,
        verbose: false,
    };
    let (opt_graph, stats) = optimize::optimize_einsum_graph(graph.clone(), &opt_config)?;
    let opt_metrics = analysis::GraphMetrics::analyze(&opt_graph);
    println!("Optimized Graph:");
    println!("  Tensors: {}", opt_metrics.tensor_count);
    println!("  Nodes: {}", opt_metrics.node_count);
    println!("  Estimated FLOPs: {}", opt_metrics.estimated_flops);
    println!(
        "  Identity simplifications: {}",
        stats.identity_simplifications
    );
    println!("  Estimated speedup: {:.2}x", stats.estimated_speedup);
    println!();

    // Compare results
    println!("=== Optimization Summary ===");
    println!("Original -> Aggressive:");
    println!(
        "  Tensor reduction: {} -> {} ({:.1}%)",
        orig_metrics.tensor_count,
        opt_metrics.tensor_count,
        (1.0 - opt_metrics.tensor_count as f64 / orig_metrics.tensor_count as f64) * 100.0
    );
    println!(
        "  Node reduction: {} -> {} ({:.1}%)",
        orig_metrics.node_count,
        opt_metrics.node_count,
        (1.0 - opt_metrics.node_count as f64 / orig_metrics.node_count as f64) * 100.0
    );
    println!(
        "  FLOP reduction: {} -> {} ({:.1}%)",
        orig_metrics.estimated_flops,
        opt_metrics.estimated_flops,
        (1.0 - opt_metrics.estimated_flops as f64 / orig_metrics.estimated_flops as f64) * 100.0
    );
    println!();

    println!("=== Optimization Examples Complete ===");
    println!();
    println!("Optimization Passes:");
    println!("  - DCE (Dead Code Elimination): Removes unused tensors and nodes");
    println!("  - CSE (Common Subexpression Elimination): Reuses duplicate computations");
    println!("  - Identity Elimination: Removes identity operations like NOT NOT");

    Ok(())
}
