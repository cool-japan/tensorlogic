//! Example demonstrating the unified optimization pipeline.
//!
//! This example showcases the multi-pass optimization pipeline that combines:
//! - Negation optimization (De Morgan's laws, double negation elimination)
//! - Constant folding (compile-time evaluation)
//! - Algebraic simplification (mathematical identities)
//!
//! The pipeline applies these passes iteratively until a fixed point is reached.

use tensorlogic_compiler::optimize::{OptimizationPipeline, PipelineConfig};
use tensorlogic_ir::{TLExpr, Term};

fn main() {
    println!("=== Tensorlogic Optimization Pipeline Demo ===\n");

    // 1. Default Pipeline - All Passes Enabled
    println!("1. Default Pipeline (All Passes)");
    println!("   Original: NOT(AND(x + 0, 2.0 * 3.0))");

    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let expr1 = TLExpr::negate(TLExpr::and(
        TLExpr::add(x.clone(), TLExpr::Constant(0.0)),
        TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
    ));

    let pipeline = OptimizationPipeline::new();
    let (optimized1, stats1) = pipeline.optimize(&expr1);

    println!("   Optimized: {:?}", optimized1);
    println!("{}", stats1);
    println!();

    // 2. Custom Configuration - Aggressive Optimization
    println!("2. Aggressive Optimization (More Iterations)");
    println!("   Original: NOT(NOT(x + (0 * y) + (2.0 + 3.0)))");

    let y = TLExpr::pred("y", vec![Term::var("i")]);
    let expr2 = TLExpr::negate(TLExpr::negate(TLExpr::add(
        TLExpr::add(x.clone(), TLExpr::mul(TLExpr::Constant(0.0), y.clone())),
        TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
    )));

    let config = PipelineConfig::aggressive();
    let pipeline = OptimizationPipeline::with_config(config);
    let (optimized2, stats2) = pipeline.optimize(&expr2);

    println!("   Optimized: {:?}", optimized2);
    println!("   Iterations: {}", stats2.total_iterations);
    println!("   Total optimizations: {}", stats2.total_optimizations());
    println!("   Reached fixed point: {}", stats2.reached_fixed_point);
    println!();

    // 3. Constant Folding Only
    println!("3. Constant Folding Only");
    println!("   Original: sqrt(16.0) + exp(0.0) * 2.0");

    let expr3 = TLExpr::add(
        TLExpr::sqrt(TLExpr::Constant(16.0)),
        TLExpr::mul(TLExpr::exp(TLExpr::Constant(0.0)), TLExpr::Constant(2.0)),
    );

    let config = PipelineConfig::constant_folding_only();
    let pipeline = OptimizationPipeline::with_config(config);
    let (optimized3, stats3) = pipeline.optimize(&expr3);

    println!("   Optimized: {:?}", optimized3);
    println!(
        "   Binary ops folded: {}",
        stats3.constant_folding.binary_ops_folded
    );
    println!(
        "   Unary ops folded: {}",
        stats3.constant_folding.unary_ops_folded
    );
    println!();

    // 4. Algebraic Simplification Only
    println!("4. Algebraic Simplification Only");
    println!("   Original: (x + 0) * 1 / 1");

    let expr4 = TLExpr::div(
        TLExpr::mul(
            TLExpr::add(x.clone(), TLExpr::Constant(0.0)),
            TLExpr::Constant(1.0),
        ),
        TLExpr::Constant(1.0),
    );

    let config = PipelineConfig::algebraic_only();
    let pipeline = OptimizationPipeline::with_config(config);
    let (optimized4, stats4) = pipeline.optimize(&expr4);

    println!("   Optimized: {:?}", optimized4);
    println!(
        "   Identities eliminated: {}",
        stats4.algebraic.identities_eliminated
    );
    println!();

    // 5. Complex Real-World Example: Softmax with Temperature
    println!("5. Real-World Example: Softmax with Temperature Scaling");
    println!("   Original: exp((x - max) / 1.0) + 0.0  (temperature = 1.0)");

    let x_pred = TLExpr::pred("x", vec![Term::var("i")]);
    let max_pred = TLExpr::pred("max", vec![]);
    let temp = TLExpr::Constant(1.0);

    let expr5 = TLExpr::add(
        TLExpr::exp(TLExpr::div(TLExpr::sub(x_pred, max_pred), temp)),
        TLExpr::Constant(0.0),
    );

    let pipeline = OptimizationPipeline::new();
    let (optimized5, stats5) = pipeline.optimize(&expr5);

    println!("   Optimized: {:?}", optimized5);
    println!(
        "   Identities eliminated: {} (div by 1, add 0)",
        stats5.algebraic.identities_eliminated
    );
    println!();

    // 6. Logical Expression with De Morgan's Laws
    println!("6. De Morgan's Laws Application");
    println!("   Original: NOT(AND(NOT(p), NOT(q)))");

    let p = TLExpr::pred("p", vec![Term::var("i")]);
    let q = TLExpr::pred("q", vec![Term::var("i")]);
    let expr6 = TLExpr::negate(TLExpr::and(
        TLExpr::negate(p.clone()),
        TLExpr::negate(q.clone()),
    ));

    let pipeline = OptimizationPipeline::new();
    let (optimized6, stats6) = pipeline.optimize(&expr6);

    println!("   Optimized: {:?}", optimized6);
    println!(
        "   De Morgan's applied: {}",
        stats6.negation.demorgans_applied
    );
    println!(
        "   Double negations eliminated: {}",
        stats6.negation.double_negations_eliminated
    );
    println!();

    // 7. Fixed Point Detection
    println!("7. Fixed Point Detection");
    println!("   Original: x (already optimal)");

    let expr7 = TLExpr::pred("x", vec![Term::var("i")]);

    let config = PipelineConfig::default().with_max_iterations(10);
    let pipeline = OptimizationPipeline::with_config(config);
    let (optimized7, stats7) = pipeline.optimize(&expr7);

    println!("   Optimized: {:?}", optimized7);
    println!("   Iterations: {} (stopped early)", stats7.total_iterations);
    println!("   Reached fixed point: {}", stats7.reached_fixed_point);
    println!();

    // 8. Per-Iteration Analysis
    println!("8. Per-Iteration Analysis");
    println!("   Original: NOT(NOT(x + 0)) * 1 + (2.0 * 3.0)");

    let expr8 = TLExpr::add(
        TLExpr::mul(
            TLExpr::negate(TLExpr::negate(TLExpr::add(
                x.clone(),
                TLExpr::Constant(0.0),
            ))),
            TLExpr::Constant(1.0),
        ),
        TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
    );

    let pipeline = OptimizationPipeline::new();
    let (optimized8, stats8) = pipeline.optimize(&expr8);

    println!("   Optimized: {:?}", optimized8);
    println!("   Total iterations: {}", stats8.total_iterations);
    println!("\n   Per-iteration breakdown:");
    for (i, iter_stats) in stats8.iterations.iter().enumerate() {
        println!(
            "   Iteration {}: {} optimizations",
            i + 1,
            iter_stats.total_optimizations()
        );
        if iter_stats.negation.double_negations_eliminated > 0 {
            println!(
                "     - Double negations: {}",
                iter_stats.negation.double_negations_eliminated
            );
        }
        if iter_stats.constant_folding.binary_ops_folded > 0 {
            println!(
                "     - Constants folded: {}",
                iter_stats.constant_folding.binary_ops_folded
            );
        }
        if iter_stats.algebraic.identities_eliminated > 0 {
            println!(
                "     - Identities eliminated: {}",
                iter_stats.algebraic.identities_eliminated
            );
        }
    }
    println!();

    // 9. Most Productive Iteration
    println!("9. Most Productive Iteration");
    let a = TLExpr::pred("a", vec![Term::var("i")]);
    let b = TLExpr::pred("b", vec![Term::var("i")]);

    let expr9 = TLExpr::negate(TLExpr::negate(TLExpr::add(
        TLExpr::mul(TLExpr::add(a, TLExpr::Constant(0.0)), TLExpr::Constant(1.0)),
        TLExpr::mul(TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)), b),
    )));

    let pipeline = OptimizationPipeline::new();
    let (_optimized9, stats9) = pipeline.optimize(&expr9);

    if let Some((iter_idx, iter_stats)) = stats9.most_productive_iteration() {
        println!(
            "   Most productive: Iteration {} with {} optimizations",
            iter_idx + 1,
            iter_stats.total_optimizations()
        );
    }
    println!();

    // 10. Custom Configuration Builder
    println!("10. Custom Configuration with Builder Pattern");
    let expr10 = TLExpr::add(
        TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
        TLExpr::mul(x.clone(), TLExpr::Constant(1.0)),
    );

    let config = PipelineConfig::default()
        .with_negation_opt(false)
        .with_constant_folding(true)
        .with_algebraic_simplification(true)
        .with_max_iterations(5);

    let pipeline = OptimizationPipeline::with_config(config);
    let (optimized10, stats10) = pipeline.optimize(&expr10);

    println!("   Configuration: negation=off, folding=on, algebraic=on");
    println!("   Optimized: {:?}", optimized10);
    println!("   Total optimizations: {}", stats10.total_optimizations());
    println!();

    // 11. Pythagorean Identity Optimization
    println!("11. Trigonometric Expression: sin²(x) + cos²(x) + 0");
    let x_trig = TLExpr::pred("x", vec![Term::var("i")]);
    let sin_x = TLExpr::sin(x_trig.clone());
    let cos_x = TLExpr::cos(x_trig);
    let two = TLExpr::Constant(2.0);

    let expr11 = TLExpr::add(
        TLExpr::add(TLExpr::pow(sin_x, two.clone()), TLExpr::pow(cos_x, two)),
        TLExpr::Constant(0.0),
    );

    let pipeline = OptimizationPipeline::new();
    let (optimized11, stats11) = pipeline.optimize(&expr11);

    println!("   Optimized: {:?}", optimized11);
    println!(
        "   Identities eliminated: {} (removed + 0)",
        stats11.algebraic.identities_eliminated
    );
    println!();

    // Summary
    println!("=== Summary ===");
    println!("The optimization pipeline provides:");
    println!("  1. Unified interface for all optimization passes");
    println!("  2. Iterative optimization until fixed point");
    println!("  3. Comprehensive statistics and per-iteration tracking");
    println!("  4. Flexible configuration (enable/disable passes, max iterations)");
    println!("  5. Builder pattern for custom configurations");
    println!("\nKey Benefits:");
    println!("  - Reduces expression complexity before compilation");
    println!("  - Improves runtime performance by eliminating redundant operations");
    println!("  - Provides insight into optimization effectiveness");
    println!("  - Configurable for different use cases (debug vs production)");
}
