//! Demonstrates incremental compilation for efficient recompilation when expressions change.
//!
//! This example shows how the IncrementalCompiler tracks dependencies and recompiles
//! only the parts that have changed, which is crucial for interactive environments
//! like REPLs, notebooks, and IDEs.

use tensorlogic_compiler::{incremental::IncrementalCompiler, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

fn main() {
    println!("=== Incremental Compilation Demo ===\n");

    // Create a compilation context
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);
    ctx.add_domain("City", 50);

    // Create incremental compiler
    let mut compiler = IncrementalCompiler::new(ctx);

    println!("1. Initial Compilation");
    println!("   Compiling: knows(x, y)");
    let expr1 = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let _graph1 = compiler.compile(&expr1).unwrap();

    let stats = compiler.stats();
    println!("   Cache misses: {}", stats.cache_misses);
    println!("   Cache hits: {}", stats.cache_hits);
    println!();

    println!("2. Recompiling Same Expression (should hit cache)");
    println!("   Compiling: knows(x, y) again");
    let _graph2 = compiler.compile(&expr1).unwrap();

    let stats = compiler.stats();
    println!("   Cache misses: {}", stats.cache_misses);
    println!("   Cache hits: {}", stats.cache_hits);
    println!("   Hit rate: {:.1}%", stats.hit_rate() * 100.0);
    println!();

    println!("3. Compiling Different Expression");
    println!("   Compiling: likes(x, z)");
    let expr2 = TLExpr::pred("likes", vec![Term::var("x"), Term::var("z")]);
    let _graph3 = compiler.compile(&expr2).unwrap();

    let stats = compiler.stats();
    println!("   Cache misses: {}", stats.cache_misses);
    println!("   Cache hits: {}", stats.cache_hits);
    println!();

    println!("4. Compiling Complex Expression");
    println!("   Compiling: knows(x, y) ∧ likes(x, z)");
    let expr3 = TLExpr::and(
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
        TLExpr::pred("likes", vec![Term::var("x"), Term::var("z")]),
    );
    let _graph4 = compiler.compile(&expr3).unwrap();

    let stats = compiler.stats();
    println!("   Total compilations: {}", stats.total_compilations());
    println!("   Cache hits: {}", stats.cache_hits);
    println!("   Cache misses: {}", stats.cache_misses);
    println!();

    println!("5. Domain Change Invalidation");
    println!("   Changing Person domain from 100 to 200...");
    compiler.context_mut().add_domain("Person", 200);

    println!("   Recompiling: knows(x, y)");
    let _graph5 = compiler.compile(&expr1).unwrap();

    let stats = compiler.stats();
    println!("   Cache invalidations: {}", stats.invalidations);
    println!(
        "   Cache misses: {} (increased due to invalidation)",
        stats.cache_misses
    );
    println!();

    println!("6. Performance Summary");
    let stats = compiler.stats();
    println!("   Total compilations: {}", stats.total_compilations());
    println!("   Cache hits: {}", stats.cache_hits);
    println!("   Cache misses: {}", stats.cache_misses);
    println!("   Hit rate: {:.1}%", stats.hit_rate() * 100.0);
    println!("   Invalidations: {}", stats.invalidations);
    println!();

    println!("7. Dependency Analysis");
    println!("   Analyzing expression: ∃x ∈ Person. knows(x, y) ∧ lives_in(x, c)");

    use tensorlogic_compiler::incremental::ExpressionDependencies;

    let complex_expr = TLExpr::exists(
        "x",
        "Person",
        TLExpr::and(
            TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
            TLExpr::pred("lives_in", vec![Term::var("x"), Term::var("c")]),
        ),
    );

    let deps = ExpressionDependencies::analyze(&complex_expr, compiler.context());
    println!("   Predicates used: {:?}", deps.predicates);
    println!("   Variables referenced: {:?}", deps.variables);
    println!("   Domains used: {:?}", deps.domains);
    println!();

    println!("=== Demo Complete ===");
    println!("\nKey Benefits of Incremental Compilation:");
    println!("• Caches compiled expressions to avoid redundant work");
    println!("• Tracks dependencies to invalidate only affected expressions");
    println!("• Detects changes in domains and configurations");
    println!("• Provides statistics for performance monitoring");
    println!("• Ideal for interactive environments (REPLs, notebooks, IDEs)");
}
