//! # Compilation Profiling and Optimization Example
//!
//! This example demonstrates the tensorlogic-compiler's profiling and advanced
//! optimization capabilities including:
//! - Compilation profiling (time, memory, cache statistics)
//! - Dataflow analysis
//! - Contraction optimization
//! - Loop fusion
//! - Post-compilation validation and optimization
//!
//! Run with:
//! ```sh
//! cargo run --example 21_profiling_and_optimization
//! ```

use anyhow::Result;
use tensorlogic_compiler::{
    compile_to_einsum_with_context,
    passes::{
        analyze_dataflow, analyze_graph_dataflow, fuse_loops, optimize_contractions,
        post_compilation_passes, PostCompilationOptions,
    },
    profiling::CompilationProfiler,
    CompilerContext,
};
use tensorlogic_ir::{TLExpr, Term};

fn main() -> Result<()> {
    println!("=== Compilation Profiling and Optimization Example ===\n");

    example_1_basic_profiling()?;
    example_2_dataflow_analysis()?;
    example_3_contraction_optimization()?;
    example_4_loop_fusion()?;
    example_5_integrated_post_compilation()?;

    println!("\n=== Summary ===");
    println!("✓ All profiling and optimization examples completed successfully");
    println!("\nKey features demonstrated:");
    println!("  • Compilation time tracking and profiling");
    println!("  • Memory usage monitoring");
    println!("  • Cache statistics");
    println!("  • Dataflow analysis (live variables, reaching definitions)");
    println!("  • Tensor contraction optimization");
    println!("  • Loop fusion for performance");
    println!("  • Integrated post-compilation pipeline");

    Ok(())
}

fn example_1_basic_profiling() -> Result<()> {
    println!("## Example 1: Basic Compilation Profiling");
    println!("Profiling the compilation of a transitive rule\n");

    // Create a profiler
    let mut profiler = CompilationProfiler::new();
    profiler.start();

    // Profile compilation phases
    profiler.start_phase("context_setup");
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);
    profiler.end_phase("context_setup");

    // Build a complex expression: ∀x,y,z. knows(x,y) ∧ knows(y,z) → knows(x,z)
    profiler.start_phase("expression_building");
    let knows_xy = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let knows_yz = TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")]);
    let knows_xz = TLExpr::pred("knows", vec![Term::var("x"), Term::var("z")]);
    let premise = TLExpr::and(knows_xy, knows_yz);
    let rule = TLExpr::imply(premise, knows_xz);
    profiler.end_phase("expression_building");

    // Profile compilation
    profiler.start_phase("compilation");
    let _graph = compile_to_einsum_with_context(&rule, &mut ctx)?;
    profiler.end_phase("compilation");

    // Generate and display profiling report
    let report = profiler.generate_report();
    println!("{}", report);

    // Also show JSON format
    println!("JSON Report (excerpt):");
    let json = profiler.generate_json_report();
    let lines: Vec<&str> = json.lines().take(10).collect();
    println!("{}", lines.join("\n"));
    println!("  ...\n");

    Ok(())
}

fn example_2_dataflow_analysis() -> Result<()> {
    println!("\n## Example 2: Dataflow Analysis");
    println!("Analyzing live variables and reaching definitions\n");

    // Create an expression with let bindings
    let expr = TLExpr::Let {
        var: "x".to_string(),
        value: Box::new(TLExpr::Constant(5.0)),
        body: Box::new(TLExpr::Let {
            var: "y".to_string(),
            value: Box::new(TLExpr::add(
                TLExpr::pred("P", vec![Term::var("x")]),
                TLExpr::Constant(10.0),
            )),
            body: Box::new(TLExpr::pred("Q", vec![Term::var("y")])),
        }),
    };

    // Perform dataflow analysis
    let analysis = analyze_dataflow(&expr);

    println!("Dataflow Analysis Results:");
    println!(
        "  Live variables detected: {}",
        analysis.live_variables.len()
    );
    println!("  Reaching definitions: {}", analysis.reaching_defs.len());
    println!(
        "  Available expressions: {}",
        analysis.available_exprs.len()
    );

    // Show reaching definitions
    println!("\nReaching Definitions:");
    for (var, defs) in &analysis.reaching_defs {
        println!("  Variable '{}': {:?}", var, defs);
    }

    // Compile and analyze graph dataflow
    let mut ctx = CompilerContext::new();
    ctx.add_domain("D", 10);
    let simple_expr = TLExpr::pred("P", vec![Term::var("x")]);
    let graph = compile_to_einsum_with_context(&simple_expr, &mut ctx)?;

    let graph_analysis = analyze_graph_dataflow(&graph);
    println!("\nGraph Dataflow:");
    println!(
        "  Tensor dependencies tracked: {}",
        graph_analysis.dependencies.len()
    );
    println!(
        "  Live tensors at nodes: {}",
        graph_analysis.live_tensors.len()
    );
    println!();

    Ok(())
}

fn example_3_contraction_optimization() -> Result<()> {
    println!("\n## Example 3: Tensor Contraction Optimization");
    println!("Optimizing einsum contraction order\n");

    // Create a complex expression with multiple operations
    let mut ctx = CompilerContext::new();
    ctx.add_domain("D", 20);

    let p = TLExpr::pred("P", vec![Term::var("i"), Term::var("j")]);
    let q = TLExpr::pred("Q", vec![Term::var("j"), Term::var("k")]);
    let r = TLExpr::pred("R", vec![Term::var("k"), Term::var("l")]);

    // Chain of operations: P(i,j) ∧ Q(j,k) ∧ R(k,l)
    let expr = TLExpr::and(TLExpr::and(p, q), r);
    let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;

    println!("Original graph:");
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Tensors: {}", graph.tensors.len());

    // Apply contraction optimization
    let (optimized_graph, stats) = optimize_contractions(&graph);

    println!("\nContraction Optimization Results:");
    println!("  Contractions reordered: {}", stats.contractions_reordered);
    println!("  FLOPs reduction: {:.1}%", stats.flops_reduction_percent);
    println!("  Memory reduction: {:.1}%", stats.memory_reduction_percent);
    println!("  Intermediates saved: {}", stats.intermediates_saved);
    println!("  Total optimizations: {}", stats.total_optimizations());

    println!("\nOptimized graph:");
    println!("  Nodes: {}", optimized_graph.nodes.len());
    println!("  Tensors: {}", optimized_graph.tensors.len());
    println!();

    Ok(())
}

fn example_4_loop_fusion() -> Result<()> {
    println!("\n## Example 4: Loop Fusion Optimization");
    println!("Fusing loops over the same axes\n");

    // Create expressions with multiple reductions
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 50);

    // ∃x. P(x) ∧ ∃y. Q(y) - two separate reductions
    let exists_p = TLExpr::exists("x", "Person", TLExpr::pred("P", vec![Term::var("x")]));
    let exists_q = TLExpr::exists("y", "Person", TLExpr::pred("Q", vec![Term::var("y")]));
    let expr = TLExpr::and(exists_p, exists_q);

    let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;

    println!("Original graph:");
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Reduction operations: {}", count_reductions(&graph));

    // Apply loop fusion
    let (fused_graph, stats) = fuse_loops(&graph);

    println!("\nLoop Fusion Results:");
    println!("  Loops fused: {}", stats.loops_fused);
    println!("  Reductions merged: {}", stats.reductions_merged);
    println!(
        "  Intermediates eliminated: {}",
        stats.intermediates_eliminated
    );
    println!("  Total optimizations: {}", stats.total_optimizations());

    println!("\nFused graph:");
    println!("  Nodes: {}", fused_graph.nodes.len());
    println!("  Reduction operations: {}", count_reductions(&fused_graph));
    println!();

    Ok(())
}

fn example_5_integrated_post_compilation() -> Result<()> {
    println!("\n## Example 5: Integrated Post-Compilation Pipeline");
    println!("Running all optimizations through the post-compilation pipeline\n");

    // Create a complex expression
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Entity", 100);

    let p = TLExpr::pred("related", vec![Term::var("x"), Term::var("y")]);
    let q = TLExpr::pred("related", vec![Term::var("y"), Term::var("z")]);
    let r = TLExpr::pred("related", vec![Term::var("x"), Term::var("z")]);

    // Transitive closure rule
    let rule = TLExpr::imply(TLExpr::and(p, q), r);
    let mut graph = compile_to_einsum_with_context(&rule, &mut ctx)?;

    println!("Before post-compilation:");
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Tensors: {}", graph.tensors.len());

    // Configure post-compilation options
    let options = PostCompilationOptions {
        validate_graph_structure: true,
        validate_axes: true,
        validate_shapes: true,
        apply_optimizations: true,
        enable_contraction_opt: true,
        enable_loop_fusion: true,
        strict_mode: false,
    };

    // Run post-compilation passes
    let result = post_compilation_passes(&mut graph, &ctx, options)?;

    println!("\nPost-Compilation Results:");
    println!("  Valid: {}", result.is_valid);
    println!(
        "  Checks performed: {}",
        result.validation_report.checks_performed
    );
    println!("  Optimizations applied: {}", result.optimizations_applied);

    println!("\nMessages:");
    for msg in &result.messages {
        println!("  {}", msg);
    }

    println!("\nAfter post-compilation:");
    println!("  Nodes: {}", graph.nodes.len());
    println!("  Tensors: {}", graph.tensors.len());
    println!();

    Ok(())
}

/// Helper function to count reduction operations in a graph
fn count_reductions(graph: &tensorlogic_ir::EinsumGraph) -> usize {
    graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, tensorlogic_ir::OpType::Reduce { .. }))
        .count()
}
