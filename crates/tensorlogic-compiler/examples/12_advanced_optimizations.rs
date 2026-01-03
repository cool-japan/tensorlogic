//! Example demonstrating advanced optimization passes.
//!
//! This example showcases the new optimization passes added to tensorlogic-compiler:
//! - Strength reduction (replace expensive ops with cheaper equivalents)
//! - Distributivity (factor common subexpressions)
//! - Complexity analysis (estimate expression costs)
//! - Quantifier optimization (loop-invariant code motion)
//! - Memory estimation (tensor memory footprint analysis)

use tensorlogic_compiler::optimize::{
    analyze_complexity, estimate_batch_memory, estimate_memory, optimize_distributivity,
    optimize_quantifiers, reduce_strength, CostWeights, OptimizationPipeline, PipelineConfig,
};
use tensorlogic_compiler::CompilerContext;
use tensorlogic_ir::{TLExpr, Term};

fn main() {
    println!("=== Advanced Optimization Passes Demo ===\n");

    // ============================================================
    // 1. Strength Reduction
    // ============================================================
    println!("1. Strength Reduction");
    println!("   Replaces expensive operations with cheaper equivalents\n");

    // Example 1a: Power optimizations
    println!("   1a. Power Optimizations:");
    let x = TLExpr::pred("x", vec![Term::var("i")]);

    // x^2 → x * x
    let expr_pow2 = TLExpr::pow(x.clone(), TLExpr::Constant(2.0));
    let (optimized, stats) = reduce_strength(&expr_pow2);
    println!("       x^2 → {:?}", optimized);
    println!("       Power reductions: {}\n", stats.power_reductions);

    // x^0.5 → sqrt(x)
    let expr_sqrt = TLExpr::pow(x.clone(), TLExpr::Constant(0.5));
    let (optimized, stats) = reduce_strength(&expr_sqrt);
    println!("       x^0.5 → {:?}", optimized);
    println!("       Power reductions: {}\n", stats.power_reductions);

    // Example 1b: Exp/Log simplifications
    println!("   1b. Exp/Log Simplifications:");

    // exp(log(x)) → x
    let expr_exp_log = TLExpr::exp(TLExpr::log(x.clone()));
    let (optimized, stats) = reduce_strength(&expr_exp_log);
    println!("       exp(log(x)) → {:?}", optimized);
    println!(
        "       Special function optimizations: {}\n",
        stats.special_function_optimizations
    );

    // log(exp(x)) → x
    let expr_log_exp = TLExpr::log(TLExpr::exp(x.clone()));
    let (optimized, stats) = reduce_strength(&expr_log_exp);
    println!("       log(exp(x)) → {:?}", optimized);
    println!(
        "       Special function optimizations: {}\n",
        stats.special_function_optimizations
    );

    // Example 1c: Total optimizations
    println!("   1c. Total Optimizations:");
    let expr_combined = TLExpr::add(
        TLExpr::pow(x.clone(), TLExpr::Constant(2.0)),
        TLExpr::exp(TLExpr::log(x.clone())),
    );
    let (optimized, stats) = reduce_strength(&expr_combined);
    println!("       x^2 + exp(log(x)) → {:?}", optimized);
    println!(
        "       Total optimizations: {}\n",
        stats.total_optimizations()
    );

    // ============================================================
    // 2. Distributivity Optimization
    // ============================================================
    println!("2. Distributivity Optimization");
    println!("   Factors common subexpressions to reduce computation\n");

    // Example 2a: Arithmetic factoring
    println!("   2a. Arithmetic Factoring:");
    let a = TLExpr::pred("a", vec![Term::var("i")]);
    let b = TLExpr::pred("b", vec![Term::var("i")]);
    let c = TLExpr::pred("c", vec![Term::var("i")]);

    // a*b + a*c → a*(b+c)
    let expr_distrib = TLExpr::add(
        TLExpr::mul(a.clone(), b.clone()),
        TLExpr::mul(a.clone(), c.clone()),
    );
    let (optimized, stats) = optimize_distributivity(&expr_distrib);
    println!("       a*b + a*c → {:?}", optimized);
    println!(
        "       Expressions factored: {}\n",
        stats.expressions_factored
    );

    // Example 2b: Logical factoring
    println!("   2b. Logical Factoring:");

    // (a OR b) AND (a OR c) → a OR (b AND c)
    let expr_logic = TLExpr::and(
        TLExpr::or(a.clone(), b.clone()),
        TLExpr::or(a.clone(), c.clone()),
    );
    let (optimized, stats) = optimize_distributivity(&expr_logic);
    println!("       (a OR b) AND (a OR c) → {:?}", optimized);
    println!(
        "       Total optimizations: {}\n",
        stats.total_optimizations()
    );

    // ============================================================
    // 3. Expression Complexity Analysis
    // ============================================================
    println!("3. Expression Complexity Analysis");
    println!("   Estimates computational cost of expressions\n");

    // Simple expression
    let simple = TLExpr::add(x.clone(), TLExpr::Constant(1.0));
    let complexity_simple = analyze_complexity(&simple);
    println!("   3a. Simple expression: x + 1");
    println!("       Max depth: {}", complexity_simple.max_depth);
    println!(
        "       Total operations: {}",
        complexity_simple.total_operations()
    );
    println!("       Additions: {}", complexity_simple.additions);
    println!("       Total cost: {}\n", complexity_simple.total_cost());

    // Complex expression with nested operations
    let complex = TLExpr::add(
        TLExpr::mul(TLExpr::exp(x.clone()), TLExpr::log(x.clone())),
        TLExpr::div(TLExpr::sin(x.clone()), TLExpr::cos(x.clone())),
    );
    let complexity_complex = analyze_complexity(&complex);
    println!("   3b. Complex expression: exp(x)*log(x) + sin(x)/cos(x)");
    println!("       Max depth: {}", complexity_complex.max_depth);
    println!(
        "       Total operations: {}",
        complexity_complex.total_operations()
    );
    println!(
        "       Transcendental ops: {}",
        complexity_complex.exponentials + complexity_complex.logarithms
    );
    println!("       Total cost: {}\n", complexity_complex.total_cost());

    // GPU-optimized cost weights
    let gpu_weights = CostWeights::gpu_optimized();
    let gpu_cost = complexity_complex.total_cost_with_weights(&gpu_weights);
    println!("   3c. GPU-optimized cost: {}", gpu_cost);
    println!("       (GPU favors parallel operations)\n");

    // Complexity level and potential optimizations
    println!(
        "   3d. Complexity level: {}",
        complexity_complex.complexity_level()
    );
    println!(
        "       CSE potential: {}",
        complexity_complex.cse_potential()
    );
    println!(
        "       Strength reduction potential: {}\n",
        complexity_complex.strength_reduction_potential()
    );

    // ============================================================
    // 4. Quantifier Optimization
    // ============================================================
    println!("4. Quantifier Optimization");
    println!("   Loop-invariant code motion for quantified expressions\n");

    // Example 4a: Hoisting from EXISTS
    println!("   4a. Hoisting Constants from EXISTS:");

    // ∃x. (a + p(x)) → a + ∃x. p(x)
    let p_x = TLExpr::pred("p", vec![Term::var("x")]);
    let expr_exists = TLExpr::Exists {
        var: "x".to_string(),
        domain: "D".to_string(),
        body: Box::new(TLExpr::add(a.clone(), p_x.clone())),
    };
    let (optimized, stats) = optimize_quantifiers(&expr_exists);
    println!("       ∃x. (a + p(x)) → {:?}", optimized);
    println!("       Invariants hoisted: {}\n", stats.invariants_hoisted);

    // Example 4b: Hoisting from FORALL
    println!("   4b. Hoisting Constants from FORALL:");

    // ∀x. (a * p(x)) → a * ∀x. p(x)
    let expr_forall = TLExpr::ForAll {
        var: "x".to_string(),
        domain: "D".to_string(),
        body: Box::new(TLExpr::mul(a.clone(), p_x.clone())),
    };
    let (optimized, stats) = optimize_quantifiers(&expr_forall);
    println!("       ∀x. (a * p(x)) → {:?}", optimized);
    println!("       Invariants hoisted: {}\n", stats.invariants_hoisted);

    // ============================================================
    // 5. Memory Estimation
    // ============================================================
    println!("5. Memory Estimation");
    println!("   Estimates tensor memory footprint based on domain sizes\n");

    // Create a context with domain information
    let mut ctx = CompilerContext::new();

    // Define domains with sizes
    ctx.add_domain("batch", 64);
    ctx.add_domain("features", 1024);
    ctx.add_domain("hidden", 2048);

    // Bind variables to domains
    let _ = ctx.bind_var("b", "batch");
    let _ = ctx.bind_var("f", "features");
    let _ = ctx.bind_var("h", "hidden");

    // Simple expression
    let simple_expr = TLExpr::pred("tensor", vec![Term::var("b"), Term::var("f")]);
    let mem_simple = estimate_memory(&simple_expr, &ctx);
    println!("   5a. Simple tensor [batch x features]:");
    println!("       Total memory: {} bytes", mem_simple.total_bytes);
    println!(
        "       Peak memory: {} bytes ({:.2} KB)\n",
        mem_simple.peak_bytes,
        mem_simple.peak_bytes as f64 / 1024.0
    );

    // Complex expression with multiple tensors
    let complex_expr = TLExpr::add(
        TLExpr::mul(
            TLExpr::pred("input", vec![Term::var("b"), Term::var("f")]),
            TLExpr::pred("weight", vec![Term::var("f"), Term::var("h")]),
        ),
        TLExpr::pred("bias", vec![Term::var("h")]),
    );
    let mem_complex = estimate_memory(&complex_expr, &ctx);
    println!("   5b. Matrix multiply + bias:");
    println!("       Total memory: {} bytes", mem_complex.total_bytes);
    println!(
        "       Peak memory: {} bytes ({:.2} MB)",
        mem_complex.peak_bytes,
        mem_complex.peak_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "       Intermediate tensors: {}\n",
        mem_complex.intermediate_count
    );

    // Batch memory estimation
    println!("   5c. Batch Memory Comparison:");
    let mem_batch_32 = estimate_batch_memory(&complex_expr, &ctx, 32);
    let mem_batch_128 = estimate_batch_memory(&complex_expr, &ctx, 128);
    println!(
        "       Batch 32:  {:.2} MB",
        mem_batch_32.peak_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "       Batch 128: {:.2} MB\n",
        mem_batch_128.peak_bytes as f64 / (1024.0 * 1024.0)
    );

    // ============================================================
    // 6. Integrated Pipeline
    // ============================================================
    println!("6. Integrated Pipeline");
    println!("   Using all optimizations together\n");

    // Create a complex expression that benefits from multiple passes
    // Original: exp(log(x^2)) + (a*b + a*c) / 2.0
    let complex_combined = TLExpr::add(
        TLExpr::exp(TLExpr::log(TLExpr::pow(x.clone(), TLExpr::Constant(2.0)))),
        TLExpr::div(
            TLExpr::add(
                TLExpr::mul(a.clone(), b.clone()),
                TLExpr::mul(a.clone(), c.clone()),
            ),
            TLExpr::Constant(2.0),
        ),
    );

    println!("   Original: exp(log(x^2)) + (a*b + a*c) / 2.0");
    let complexity_before = analyze_complexity(&complex_combined);
    println!(
        "   Complexity before: {:.1} (depth: {}, ops: {})",
        complexity_before.total_cost(),
        complexity_before.max_depth,
        complexity_before.total_operations()
    );

    // Apply full pipeline
    let config = PipelineConfig::aggressive()
        .with_strength_reduction(true)
        .with_distributivity(true)
        .with_quantifier_opt(true);

    let pipeline = OptimizationPipeline::with_config(config);
    let (optimized, stats) = pipeline.optimize(&complex_combined);

    let complexity_after = analyze_complexity(&optimized);
    println!("   Optimized: {:?}", optimized);
    println!(
        "   Complexity after: {:.1} (depth: {}, ops: {})",
        complexity_after.total_cost(),
        complexity_after.max_depth,
        complexity_after.total_operations()
    );
    let reduction = if complexity_before.total_cost() > 0.0 {
        (1.0 - complexity_after.total_cost() / complexity_before.total_cost()) * 100.0
    } else {
        0.0
    };
    println!("   Reduction: {:.1}%\n", reduction);

    // Print pipeline statistics
    println!("   Pipeline Statistics:");
    println!("   {}", stats);

    // ============================================================
    // 7. Real-World Example: Neural Network Layer
    // ============================================================
    println!("\n7. Real-World Example: Neural Network Layer");
    println!("   Optimizing a transformer attention computation\n");

    // Q * K^T / sqrt(d_k) + mask
    let q = TLExpr::pred("Q", vec![Term::var("b"), Term::var("h"), Term::var("s")]);
    let k = TLExpr::pred("K", vec![Term::var("b"), Term::var("h"), Term::var("s")]);
    let mask = TLExpr::pred("mask", vec![Term::var("s"), Term::var("s")]);
    let d_k = TLExpr::Constant(64.0);

    // Original with potential optimizations
    let attention = TLExpr::add(
        TLExpr::div(
            TLExpr::mul(q.clone(), k.clone()),
            TLExpr::pow(d_k, TLExpr::Constant(0.5)), // sqrt via pow
        ),
        TLExpr::mul(mask, TLExpr::Constant(1.0)), // identity mul
    );

    println!("   Original: Q*K / sqrt(64) + mask*1");

    let (optimized, stats) = reduce_strength(&attention);
    let complexity_attn = analyze_complexity(&optimized);

    println!("   After strength reduction: {:?}", optimized);
    println!("   Power reductions: {}", stats.power_reductions);
    println!("   Total cost: {}", complexity_attn.total_cost());

    // ============================================================
    // Summary
    // ============================================================
    println!("\n=== Summary ===");
    println!("Advanced optimization passes provide:");
    println!("  1. Strength Reduction - Replace expensive ops with cheaper equivalents");
    println!("  2. Distributivity     - Factor common subexpressions");
    println!("  3. Complexity Analysis - Estimate computational costs");
    println!("  4. Quantifier Opt     - Loop-invariant code motion");
    println!("  5. Memory Estimation  - Tensor memory footprint analysis");
    println!("\nKey Benefits:");
    println!("  - Reduced computational complexity");
    println!("  - Better memory utilization");
    println!("  - Informed optimization decisions");
    println!("  - Configurable for different hardware targets (CPU/GPU/SIMD)");
}
