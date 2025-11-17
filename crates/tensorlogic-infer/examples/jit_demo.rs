//! Minimal JIT compilation demo
//!
//! Demonstrates JIT compilation and caching
//!
//! Run with: cargo run --example jit_demo

use tensorlogic_infer::*;
use tensorlogic_ir::EinsumGraph;

fn main() {
    println!("=== JIT Compilation Demo ===\n");

    // Create a simple graph
    let graph = EinsumGraph::new();

    // Setup JIT compiler
    let config = JitConfig {
        initial_optimization: OptimizationLevel::Basic,
        hot_path_optimization: OptimizationLevel::Aggressive,
        hot_path_threshold: 5,
        enable_adaptive_optimization: true,
        ..Default::default()
    };

    let mut jit = JitCompiler::new(config);

    println!("Simulating 10 executions...");

    // Execute multiple times to trigger hot path optimization
    for i in 0..10 {
        match jit.compile_or_retrieve(&graph, &[]) {
            Ok(_) => {
                jit.record_execution(&graph, &[], std::time::Duration::from_millis(5));

                if i == 0 {
                    println!("  Execution {}: Compiled from scratch", i + 1);
                } else if i < 3 {
                    println!("  Execution {}: Retrieved from cache", i + 1);
                }
            }
            Err(e) => eprintln!("  Execution {} failed: {}", i + 1, e),
        }
    }

    // Trigger adaptive optimization
    match jit.optimize_hot_paths() {
        Ok(count) => println!("\nOptimized {} hot paths", count),
        Err(e) => eprintln!("\nOptimization failed: {}", e),
    }

    // Show cache statistics
    let stats = jit.cache_stats();
    println!("\nJIT Cache Statistics:");
    println!("  Total entries: {}", stats.total_entries);
    println!("  Hot entries: {}", stats.hot_entries);
    println!("  Specialized entries: {}", stats.specialized_entries);
    println!("  Total executions: {}", stats.total_executions);

    println!("\n=== Demo Complete ===");
}
