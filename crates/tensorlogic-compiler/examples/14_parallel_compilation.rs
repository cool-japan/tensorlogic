//! Demonstrates parallel compilation capabilities for complex expressions.
//!
//! This example shows how to use the parallel compiler to potentially speed up
//! compilation of large, complex logical expressions by leveraging multiple CPU cores.
//!
//! Run with:
//! ```sh
//! cargo run --example 14_parallel_compilation --features parallel
//! ```

#[cfg(feature = "parallel")]
use tensorlogic_compiler::{
    parallel::{ParallelCompiler, ParallelConfig},
    CompilerContext,
};

#[cfg(feature = "parallel")]
use tensorlogic_ir::{TLExpr, Term};

#[cfg(feature = "parallel")]
fn create_complex_expression() -> TLExpr {
    // Build a complex knowledge graph inference rule
    // (knows(x,y) ∧ knows(y,z)) → knows(x,z)  (transitivity)
    let knows_xy = TLExpr::Pred {
        name: "knows".to_string(),
        args: vec![Term::Var("x".to_string()), Term::Var("y".to_string())],
    };

    let knows_yz = TLExpr::Pred {
        name: "knows".to_string(),
        args: vec![Term::Var("y".to_string()), Term::Var("z".to_string())],
    };

    let knows_xz = TLExpr::Pred {
        name: "knows".to_string(),
        args: vec![Term::Var("x".to_string()), Term::Var("z".to_string())],
    };

    // Add more complex nested structure
    let premise = TLExpr::And(Box::new(knows_xy), Box::new(knows_yz));

    // Create a larger expression with multiple implications
    let implication1 = TLExpr::Imply(Box::new(premise.clone()), Box::new(knows_xz.clone()));

    // Add another layer: trust transitivity
    let trusts_xy = TLExpr::Pred {
        name: "trusts".to_string(),
        args: vec![Term::Var("x".to_string()), Term::Var("y".to_string())],
    };

    let trusts_yz = TLExpr::Pred {
        name: "trusts".to_string(),
        args: vec![Term::Var("y".to_string()), Term::Var("z".to_string())],
    };

    let trusts_xz = TLExpr::Pred {
        name: "trusts".to_string(),
        args: vec![Term::Var("x".to_string()), Term::Var("z".to_string())],
    };

    let trust_premise = TLExpr::And(Box::new(trusts_xy), Box::new(trusts_yz));
    let implication2 = TLExpr::Imply(Box::new(trust_premise), Box::new(trusts_xz));

    // Combine multiple rules
    TLExpr::And(Box::new(implication1), Box::new(implication2))
}

#[cfg(feature = "parallel")]
fn main() -> anyhow::Result<()> {
    println!("=== Parallel Compilation Example ===\n");

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 1000); // Large domain for parallel benefits

    let expr = create_complex_expression();

    // Example 1: Default parallel compilation
    println!("1. Default Parallel Compilation");
    println!("   Complexity threshold: 10 operations");
    println!("   Max threads: all available cores\n");

    let compiler = ParallelCompiler::new();

    let start = std::time::Instant::now();
    let graph = compiler.compile(&expr, &mut ctx)?;
    let elapsed = start.elapsed();

    let stats = compiler.stats();
    println!("   Compilation completed in {:?}", elapsed);
    println!("   Graph nodes: {}", graph.nodes.len());
    println!("   Tensors: {}", graph.tensors.len());
    println!("\n   Statistics:");
    println!("     Parallel tasks: {}", stats.parallel_tasks);
    println!("     Sequential tasks: {}", stats.sequential_tasks);
    println!("     Total tasks: {}", stats.total_tasks());
    println!(
        "     Parallelization ratio: {:.1}%",
        stats.parallelization_ratio() * 100.0
    );
    println!("     Threads used: {}", stats.threads_used);

    // Example 2: Custom configuration
    println!("\n2. Custom Parallel Configuration");
    println!("   Complexity threshold: 5 operations");
    println!("   Max threads: 4\n");

    let config = ParallelConfig::new()
        .with_min_complexity(5)
        .with_max_threads(4)
        .with_parallel_optimization(true);

    let compiler2 = ParallelCompiler::with_config(config);
    compiler2.reset_stats();

    let mut ctx2 = CompilerContext::new();
    ctx2.add_domain("Person", 1000);

    let start2 = std::time::Instant::now();
    let graph2 = compiler2.compile(&expr, &mut ctx2)?;
    let elapsed2 = start2.elapsed();

    let stats2 = compiler2.stats();
    println!("   Compilation completed in {:?}", elapsed2);
    println!("   Graph nodes: {}", graph2.nodes.len());
    println!("\n   Statistics:");
    println!("     Parallel tasks: {}", stats2.parallel_tasks);
    println!("     Sequential tasks: {}", stats2.sequential_tasks);
    println!(
        "     Parallelization ratio: {:.1}%",
        stats2.parallelization_ratio() * 100.0
    );

    // Example 3: With optimization
    println!("\n3. Parallel Compilation with Optimization");

    let compiler3 = ParallelCompiler::new();
    compiler3.reset_stats();

    let mut ctx3 = CompilerContext::new();
    ctx3.add_domain("Person", 1000);

    // Create expression with double negations for optimization
    let expr_with_opt = TLExpr::Not(Box::new(TLExpr::Not(Box::new(expr.clone()))));

    let opt_config = tensorlogic_compiler::optimize::PipelineConfig::default();
    let start3 = std::time::Instant::now();
    let (graph3, opt_stats) =
        compiler3.compile_with_optimization(&expr_with_opt, &mut ctx3, opt_config)?;
    let elapsed3 = start3.elapsed();

    println!("   Compilation + optimization completed in {:?}", elapsed3);
    println!("   Graph nodes: {}", graph3.nodes.len());
    println!("\n   Optimization statistics:");
    println!(
        "     Double negations eliminated: {}",
        opt_stats.negation.double_negations_eliminated
    );
    println!(
        "     Total optimizations: {}",
        opt_stats.total_optimizations()
    );

    println!("\n=== Parallel Compilation Complete ===");

    Ok(())
}

#[cfg(not(feature = "parallel"))]
fn main() {
    eprintln!("This example requires the 'parallel' feature.");
    eprintln!("Run with: cargo run --example 14_parallel_compilation --features parallel");
    std::process::exit(1);
}
