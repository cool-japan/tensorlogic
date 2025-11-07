//! Example: Compilation caching with metadata for production deployments
//!
//! This example demonstrates how to use the compilation cache to improve
//! performance when compiling multiple expressions, combined with metadata
//! for debugging and provenance tracking.

use tensorlogic_compiler::passes::{propagate_metadata, MetadataBuilder};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationCache, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

fn main() {
    println!("=== Compilation Caching with Metadata Example ===\n");

    // Create a compilation cache
    let cache = CompilationCache::new(100);
    println!(
        "1. Created compilation cache (max size: {})",
        cache.max_size()
    );

    // Set up compiler context
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);
    ctx.add_domain("City", 50);

    // Example 1: First compilation (cache miss)
    println!("\n2. First compilation (cache miss):");
    let expr1 = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

    let graph1 = cache
        .get_or_compile(&expr1, &mut ctx, |e, c| {
            println!("   Compiling: {:?}", e);
            compile_to_einsum_with_context(e, c)
        })
        .unwrap();

    let stats = cache.stats();
    println!(
        "   Cache stats: {} hits, {} misses",
        stats.hits, stats.misses
    );
    println!(
        "   Graph has {} tensors, {} nodes",
        graph1.tensors.len(),
        graph1.nodes.len()
    );

    // Example 2: Second compilation (cache hit)
    println!("\n3. Second compilation of same expression (cache hit):");
    let graph2 = cache
        .get_or_compile(&expr1, &mut ctx, |e, c| {
            println!("   Compiling: {:?}", e);
            compile_to_einsum_with_context(e, c)
        })
        .unwrap();

    let stats = cache.stats();
    println!(
        "   Cache stats: {} hits, {} misses",
        stats.hits, stats.misses
    );
    println!("   Hit rate: {:.1}%", stats.hit_rate() * 100.0);
    assert_eq!(graph1, graph2); // Same graph

    // Example 3: Multiple different expressions
    println!("\n4. Compiling multiple different expressions:");
    let expressions = vec![
        TLExpr::pred("lives_in", vec![Term::var("x"), Term::var("c")]),
        TLExpr::And(
            Box::new(TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")])),
            Box::new(TLExpr::pred("likes", vec![Term::var("y"), Term::var("z")])),
        ),
        TLExpr::exists(
            "y",
            "Person",
            TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
        ),
    ];

    for (i, expr) in expressions.iter().enumerate() {
        let _ = cache.get_or_compile(expr, &mut ctx, compile_to_einsum_with_context);
        println!(
            "   Compiled expression {}: {}",
            i + 1,
            format!("{:?}", expr).chars().take(50).collect::<String>()
        );
    }

    let stats = cache.stats();
    println!("\n   Final cache stats:");
    println!("     Total lookups: {}", stats.total_lookups());
    println!("     Hits: {}", stats.hits);
    println!("     Misses: {}", stats.misses);
    println!("     Hit rate: {:.1}%", stats.hit_rate() * 100.0);
    println!(
        "     Current size: {}/{}",
        stats.current_size,
        cache.max_size()
    );

    // Example 4: Re-compiling expressions (more cache hits)
    println!("\n5. Re-compiling some expressions (demonstrating cache hits):");
    let _ = cache.get_or_compile(&expressions[0], &mut ctx, |e, c| {
        compile_to_einsum_with_context(e, c)
    });
    let _ = cache.get_or_compile(&expressions[2], &mut ctx, |e, c| {
        compile_to_einsum_with_context(e, c)
    });

    let stats = cache.stats();
    println!("   Cache hits: {} (improved from previous)", stats.hits);
    println!("   Hit rate: {:.1}%", stats.hit_rate() * 100.0);

    // Example 5: Combining caching with metadata
    println!("\n6. Combining caching with metadata tracking:");
    let expr_with_meta = TLExpr::pred("friends", vec![Term::var("a"), Term::var("b")]);

    let mut graph_with_meta = cache
        .get_or_compile(&expr_with_meta, &mut ctx, |e, c| {
            compile_to_einsum_with_context(e, c)
        })
        .unwrap();

    // Add metadata after compilation (even from cache)
    let mut metadata_builder = MetadataBuilder::new()
        .with_source_file("social_rules.tl")
        .with_rule_id("friendship_symmetric");

    propagate_metadata(&mut graph_with_meta, &ctx, &mut metadata_builder);

    println!("   Graph with metadata:");
    println!("     Tensors: {}", graph_with_meta.tensors.len());
    println!(
        "     Metadata entries: {}",
        graph_with_meta.tensor_metadata.len()
    );
    for (idx, meta) in &graph_with_meta.tensor_metadata {
        println!("       Tensor {}: {:?}", idx, meta.name);
    }

    // Example 6: Cache performance demonstration
    println!("\n7. Performance demonstration (repeated compilations):");
    let test_expr = TLExpr::pred("test_pred", vec![Term::var("x")]);

    // First compilation
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = cache.get_or_compile(&test_expr, &mut ctx, |e, c| {
            compile_to_einsum_with_context(e, c)
        });
    }
    let duration = start.elapsed();

    println!("   10 compilations (with caching): {:?}", duration);
    println!("   Average per compilation: {:?}", duration / 10);

    let stats = cache.stats();
    println!("   Final hit rate: {:.1}%", stats.hit_rate() * 100.0);

    // Example 7: Cache management
    println!("\n8. Cache management:");
    println!("   Current cache size: {}", cache.len());
    println!("   Cache is empty: {}", cache.is_empty());

    // Clear cache
    cache.clear();
    println!("   After clearing:");
    println!("   Current cache size: {}", cache.len());
    println!("   Cache is empty: {}", cache.is_empty());

    println!("\n=== Complete! ===");
    println!("\nKey benefits demonstrated:");
    println!("  • Compilation caching reduces redundant work");
    println!("  • Cache hits improve performance significantly");
    println!("  • Thread-safe caching for concurrent workflows");
    println!("  • Metadata can be added to cached results");
    println!("  • Automatic eviction when cache is full");
    println!("  • Detailed statistics for monitoring performance");
}
