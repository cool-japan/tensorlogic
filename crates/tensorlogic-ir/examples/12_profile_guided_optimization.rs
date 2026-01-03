//! Profile-Guided Optimization Example
//!
//! Demonstrates how to use runtime profiling data to guide optimization decisions.
//! PGO helps identify performance bottlenecks and apply targeted optimizations.

use std::time::Duration;
use tensorlogic_ir::{
    EinsumGraph, EinsumNode, ExecutionProfile, NodeStats, OptimizationHint, ProfileGuidedOptimizer,
};

fn main() {
    println!("=== Profile-Guided Optimization in TensorLogic ===\n");

    // Example 1: Basic profiling
    example_basic_profiling();

    // Example 2: Node statistics
    example_node_statistics();

    // Example 3: Hot node identification
    example_hot_nodes();

    // Example 4: Memory-intensive operations
    example_memory_intensive();

    // Example 5: Profile merging
    example_profile_merging();

    // Example 6: Optimization hints generation
    example_optimization_hints();

    // Example 7: Profile serialization
    example_profile_serialization();

    // Example 8: Complete PGO workflow
    example_complete_workflow();
}

fn example_basic_profiling() {
    println!("--- Example 1: Basic Profiling ---");

    let mut profile = ExecutionProfile::new();

    // Simulate execution of node 0
    profile.record_node(0, Duration::from_millis(10), 1024);
    profile.record_node(0, Duration::from_millis(12), 1024);
    profile.record_node(0, Duration::from_millis(11), 1024);

    println!("Recorded 3 executions of node 0");

    if let Some(stats) = profile.node_stats.get(&0) {
        println!("  Execution count: {}", stats.execution_count);
        println!("  Total time: {:?}", stats.total_time);
        println!("  Average time: {:?}", stats.avg_time());
        println!("  Min time: {:?}", stats.min_time);
        println!("  Max time: {:?}", stats.max_time);
        println!("  Peak memory: {} bytes", stats.peak_memory);
    }

    println!();
}

fn example_node_statistics() {
    println!("--- Example 2: Node Statistics ---");

    let mut stats = NodeStats::new();

    // Record multiple executions
    stats.record_execution(Duration::from_millis(50), 2048);
    stats.record_execution(Duration::from_millis(75), 3072);
    stats.record_execution(Duration::from_millis(60), 2560);

    println!("Node statistics:");
    println!("  Executions: {}", stats.execution_count);
    println!("  Average time: {:?}", stats.avg_time());
    println!("  Time variance (max-min): {:?}", stats.time_variance());
    println!("  Peak memory: {} bytes", stats.peak_memory);
    println!("  Is hot (threshold=10): {}", stats.is_hot(10));
    println!("  Is hot (threshold=2): {}", stats.is_hot(2));
    println!("  Performance score: {:.2}\n", stats.performance_score());
}

fn example_hot_nodes() {
    println!("--- Example 3: Hot Node Identification ---");

    let mut profile = ExecutionProfile::new();

    // Node 0: frequently executed, fast
    for _ in 0..100 {
        profile.record_node(0, Duration::from_millis(5), 512);
    }

    // Node 1: rarely executed, slow
    for _ in 0..3 {
        profile.record_node(1, Duration::from_millis(500), 10240);
    }

    // Node 2: moderate frequency, moderate speed
    for _ in 0..20 {
        profile.record_node(2, Duration::from_millis(50), 2048);
    }

    println!("Execution summary:");
    println!("  Node 0: 100 executions @ 5ms");
    println!("  Node 1: 3 executions @ 500ms");
    println!("  Node 2: 20 executions @ 50ms");

    // Get top 3 hot nodes
    let hot_nodes = profile.get_hot_nodes(3);

    println!("\nTop 3 hot nodes (by performance score):");
    for (i, (node_id, score)) in hot_nodes.iter().enumerate() {
        println!("  {}. Node {} (score: {:.2})", i + 1, node_id, score);
    }

    println!();
}

fn example_memory_intensive() {
    println!("--- Example 4: Memory-Intensive Operations ---");

    let mut profile = ExecutionProfile::new();

    // Small memory usage
    profile.record_node(0, Duration::from_millis(10), 1024);

    // Medium memory usage
    profile.record_node(1, Duration::from_millis(20), 50 * 1024 * 1024); // 50 MB

    // Large memory usage
    profile.record_node(2, Duration::from_millis(30), 200 * 1024 * 1024); // 200 MB

    let threshold = 100 * 1024 * 1024; // 100 MB
    let memory_nodes = profile.get_memory_intensive_nodes(threshold);

    println!("Memory-intensive nodes (>= 100 MB):");
    for node_id in &memory_nodes {
        if let Some(stats) = profile.node_stats.get(node_id) {
            let mb = stats.peak_memory as f64 / (1024.0 * 1024.0);
            println!("  Node {}: {:.2} MB", node_id, mb);
        }
    }

    println!();
}

fn example_profile_merging() {
    println!("--- Example 5: Profile Merging ---");

    // Profile from run 1
    let mut profile1 = ExecutionProfile::new();
    profile1.record_node(0, Duration::from_millis(100), 1024);
    profile1.record_node(1, Duration::from_millis(200), 2048);
    profile1.total_executions = 1;

    // Profile from run 2
    let mut profile2 = ExecutionProfile::new();
    profile2.record_node(0, Duration::from_millis(110), 1024);
    profile2.record_node(1, Duration::from_millis(210), 2048);
    profile2.record_node(2, Duration::from_millis(50), 512); // New node
    profile2.total_executions = 1;

    println!(
        "Profile 1: {} nodes, {} total executions",
        profile1.node_stats.len(),
        profile1.total_executions
    );
    println!(
        "Profile 2: {} nodes, {} total executions",
        profile2.node_stats.len(),
        profile2.total_executions
    );

    // Merge profiles
    profile1.merge(&profile2);

    println!("\nMerged profile:");
    println!("  Unique nodes: {}", profile1.node_stats.len());
    println!("  Total executions: {}", profile1.total_executions);

    for (node_id, stats) in &profile1.node_stats {
        println!(
            "  Node {}: {} executions, avg {:?}",
            node_id,
            stats.execution_count,
            stats.avg_time()
        );
    }

    println!();
}

fn example_optimization_hints() {
    println!("--- Example 6: Optimization Hints Generation ---");

    let mut profile = ExecutionProfile::new();

    // Create hot nodes (for fusion)
    for _ in 0..50 {
        profile.record_node(0, Duration::from_millis(10), 1024);
        profile.record_node(1, Duration::from_millis(10), 1024);
    }

    // Memory-intensive node
    profile.record_node(2, Duration::from_millis(100), 200 * 1024 * 1024);

    // Frequently accessed tensor
    for _ in 0..100 {
        profile.record_tensor_access(0, 4096);
    }

    let optimizer = ProfileGuidedOptimizer::new(profile)
        .with_hot_threshold(10)
        .with_memory_threshold(100 * 1024 * 1024);

    let graph = EinsumGraph::new(); // Empty graph for demo
    let hints = optimizer.analyze(&graph);

    println!("Generated {} optimization hints:", hints.len());
    for (i, hint) in hints.iter().enumerate() {
        println!("  {}. {:?}", i + 1, hint);
    }

    println!();
}

fn example_profile_serialization() {
    println!("--- Example 7: Profile Serialization ---");

    let mut profile = ExecutionProfile::new();

    // Add some data
    profile.record_node(0, Duration::from_millis(50), 1024);
    profile.record_node(1, Duration::from_millis(75), 2048);
    profile.record_tensor_access(0, 4096);

    // Export to JSON
    match profile.to_json() {
        Ok(json) => {
            println!("Exported profile to JSON:");
            println!("{}", &json[..json.len().min(300)]); // Show first 300 chars
            if json.len() > 300 {
                println!("... (truncated)");
            }

            // Import back
            match ExecutionProfile::from_json(&json) {
                Ok(restored) => {
                    println!("\n✓ Successfully restored profile from JSON");
                    println!("  Node stats: {}", restored.node_stats.len());
                    println!("  Tensor stats: {}", restored.tensor_stats.len());
                }
                Err(e) => println!("\n✗ Failed to restore: {}", e),
            }
        }
        Err(e) => println!("✗ Failed to export: {}", e),
    }

    println!();
}

fn example_complete_workflow() {
    println!("--- Example 8: Complete PGO Workflow ---");

    // Step 1: Create a simple computation graph
    let mut graph = EinsumGraph::new();
    let a = graph.add_tensor("A");
    let b = graph.add_tensor("B");
    let c = graph.add_tensor("C");

    graph
        .add_node(EinsumNode::einsum("ij,jk->ik", vec![a, b], vec![c]))
        .unwrap();
    graph.add_output(c).unwrap();

    println!("Step 1: Created computation graph");
    println!("  Tensors: {}", graph.tensor_count());
    println!("  Nodes: {}", graph.node_count());

    // Step 2: Simulate execution and collect profile
    let mut profile = ExecutionProfile::new();

    println!("\nStep 2: Simulating execution...");
    for run in 0..10 {
        // Simulate node execution with varying performance
        let time = Duration::from_millis(50 + (run % 3) * 10);
        let memory = 1024 * (1 + run % 2);

        profile.record_node(0, time, memory);

        // Simulate tensor accesses
        profile.record_tensor_access(a, 1024);
        profile.record_tensor_access(b, 2048);
        profile.record_tensor_access(c, 1024);

        profile.total_executions += 1;
    }

    println!("  Completed {} executions", profile.total_executions);

    // Step 3: Analyze profile
    println!("\nStep 3: Analyzing profile...");
    let hot_nodes = profile.get_hot_nodes(5);

    for (node_id, score) in &hot_nodes {
        let stats = &profile.node_stats[node_id];
        println!(
            "  Node {}: {} execs, avg {:?}, score: {:.2}",
            node_id,
            stats.execution_count,
            stats.avg_time(),
            score
        );
    }

    // Step 4: Generate and apply optimization hints
    println!("\nStep 4: Generating optimization hints...");
    let optimizer = ProfileGuidedOptimizer::new(profile);
    let hints = optimizer.analyze(&graph);

    println!("  Generated {} hints:", hints.len());
    for hint in &hints {
        match hint {
            OptimizationHint::FuseNodes(nodes) => {
                println!("    - Fuse nodes: {:?}", nodes);
            }
            OptimizationHint::CacheTensor(tid) => {
                println!("    - Cache tensor: {}", tid);
            }
            OptimizationHint::PreAllocate { tensor, size } => {
                println!("    - Pre-allocate tensor {} ({} bytes)", tensor, size);
            }
            _ => println!("    - {:?}", hint),
        }
    }

    // Step 5: Apply hints (simplified - would modify graph in real implementation)
    println!("\nStep 5: Applying optimizations...");
    let mut optimized_graph = graph.clone();

    match optimizer.apply_hints(&mut optimized_graph, &hints) {
        Ok(applied) => println!("  Applied {} optimization(s)", applied),
        Err(e) => println!("  Error applying hints: {}", e),
    }

    println!("\n✓ Complete PGO workflow finished");
}
