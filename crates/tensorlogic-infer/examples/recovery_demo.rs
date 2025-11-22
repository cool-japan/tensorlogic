//! Comprehensive example demonstrating error recovery and fault tolerance.
//!
//! This example shows:
//! - Retry with exponential backoff
//! - Checkpointing and restart
//! - Graceful degradation
//! - Partial results on failure
//! - Recovery statistics and monitoring

use std::collections::HashMap;
use tensorlogic_infer::{
    recovery::{
        Checkpoint, CheckpointManager, DegradationPolicy, FallbackStrategy, RecoveryConfig,
        RecoveryResult, RecoveryStats, RecoveryStrategy, RetryPolicy,
    },
    DummyExecutor, DummyTensor,
};
use tensorlogic_ir::{EinsumGraph, EinsumNode, NodeId, OpType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TensorLogic Error Recovery Demo ===\n");

    // Create a sample computation graph
    let graph = create_sample_graph();
    println!(
        "Created computation graph with {} nodes\n",
        graph.nodes.len()
    );

    // Demo 1: Retry with exponential backoff
    println!("--- Demo 1: Retry with Exponential Backoff ---");
    demo_retry_backoff()?;

    // Demo 2: Checkpointing
    println!("\n--- Demo 2: Checkpointing and Restart ---");
    demo_checkpointing()?;

    // Demo 3: Graceful degradation
    println!("\n--- Demo 3: Graceful Degradation ---");
    demo_degradation()?;

    // Demo 4: Partial results
    println!("\n--- Demo 4: Partial Results on Failure ---");
    demo_partial_results()?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn create_sample_graph() -> EinsumGraph {
    let mut graph = EinsumGraph::new();

    // Input node
    let input_node = EinsumNode {
        id: NodeId(0),
        op: OpType::Input {
            name: "x".to_string(),
        },
        inputs: vec![],
        output_shape: vec![64, 128],
        metadata: HashMap::new(),
    };
    graph.nodes.push(input_node);

    // Einsum operation
    let einsum_node = EinsumNode {
        id: NodeId(1),
        op: OpType::Einsum {
            spec: "ij,jk->ik".to_string(),
        },
        inputs: vec![0],
        output_shape: vec![64, 256],
        metadata: HashMap::new(),
    };
    graph.nodes.push(einsum_node);

    graph
}

fn demo_retry_backoff() -> Result<(), Box<dyn std::error::Error>> {
    println!("Configuring retry with exponential backoff...");

    // Create retry policy: 3 retries, 100ms initial delay
    let retry_policy = RetryPolicy::exponential(3, 100);
    println!("Max retries: {}", retry_policy.max_retries());
    println!("Initial delay: {}ms", retry_policy.initial_delay_ms());

    // Configure recovery
    let config = RecoveryConfig::default()
        .with_strategy(RecoveryStrategy::RetryWithBackoff)
        .with_retry_policy(retry_policy);

    println!("\nRecovery strategy: {:?}", config.strategy());
    println!("Checkpointing enabled: {}", config.checkpointing_enabled());

    // Simulate recovery result
    let stats = RecoveryStats {
        retries: 2,
        checkpoints_saved: 0,
        checkpoints_loaded: 0,
        total_recovery_time_ms: 250.0,
        failed_nodes: vec![],
    };

    println!("\nRecovery completed:");
    println!("  Retries: {}", stats.retries);
    println!(
        "  Total recovery time: {:.2}ms",
        stats.total_recovery_time_ms
    );
    println!("  Success after {} retries", stats.retries);

    Ok(())
}

fn demo_checkpointing() -> Result<(), Box<dyn std::error::Error>> {
    println!("Setting up checkpointing...");

    // Create checkpoint manager
    let mut manager = CheckpointManager::new("/tmp/tensorlogic_checkpoints".to_string());
    println!("Checkpoint directory: {}", manager.checkpoint_dir());

    // Create sample checkpoint
    let mut state = HashMap::new();
    state.insert("tensor_0".to_string(), DummyTensor::new(vec![64, 128]));
    state.insert("tensor_1".to_string(), DummyTensor::new(vec![64, 256]));

    let checkpoint = Checkpoint::new("checkpoint_1".to_string(), state.clone(), 0);

    println!("\nCreating checkpoint: {}", checkpoint.id());
    println!("Graph node: {}", checkpoint.graph_node());
    println!("Tensors: {}", checkpoint.state().len());

    // Save checkpoint
    manager.save(checkpoint.clone())?;
    println!("Checkpoint saved successfully");

    // List checkpoints
    let checkpoints = manager.list_checkpoints()?;
    println!("\nAvailable checkpoints: {}", checkpoints.len());
    for cp in checkpoints {
        println!("  - {} (age: {:.2}s)", cp.id(), cp.age_seconds());
    }

    // Load latest checkpoint
    if let Some(loaded) = manager.load_latest()? {
        println!("\nLoaded latest checkpoint: {}", loaded.id());
        println!("Restored {} tensors", loaded.state().len());
    }

    // Cleanup old checkpoints
    let removed = manager.cleanup_old(3600)?; // Remove checkpoints older than 1 hour
    println!("\nCleaned up {} old checkpoints", removed);

    Ok(())
}

fn demo_degradation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Configuring graceful degradation...");

    // Create degradation policy
    let policy = DegradationPolicy::default()
        .with_skip_failed_nodes(true)
        .with_use_cached_results(true)
        .with_reduce_precision(false);

    println!("Skip failed nodes: {}", policy.skip_failed_nodes());
    println!("Use cached results: {}", policy.use_cached_results());
    println!("Reduce precision: {}", policy.reduce_precision());

    // Configure recovery with degradation
    let config = RecoveryConfig::default()
        .with_strategy(RecoveryStrategy::GracefulDegradation)
        .with_degradation_policy(policy)
        .with_fallback_strategy(FallbackStrategy::UseCache);

    println!("\nFallback strategy: {:?}", config.fallback_strategy());

    // Simulate degraded execution
    println!("\nExecuting with graceful degradation...");
    println!("  Some operations may be skipped on failure");
    println!("  Cached results will be used when available");
    println!("  Execution continues despite errors");

    let stats = RecoveryStats {
        retries: 0,
        checkpoints_saved: 0,
        checkpoints_loaded: 0,
        total_recovery_time_ms: 50.0,
        failed_nodes: vec![NodeId(2), NodeId(5)],
    };

    println!("\nDegradation results:");
    println!("  Failed nodes: {:?}", stats.failed_nodes);
    println!("  Execution completed with partial results");
    println!("  Recovery time: {:.2}ms", stats.total_recovery_time_ms);

    Ok(())
}

fn demo_partial_results() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating partial results on failure...");

    // Simulate partial success scenario
    let mut successful_outputs = vec![];
    successful_outputs.push(DummyTensor::new(vec![64, 128]));
    successful_outputs.push(DummyTensor::new(vec![64, 256]));

    let failed_nodes = vec![NodeId(3), NodeId(4)];

    let stats = RecoveryStats {
        retries: 1,
        checkpoints_saved: 2,
        checkpoints_loaded: 1,
        total_recovery_time_ms: 150.0,
        failed_nodes: failed_nodes.clone(),
    };

    let result = RecoveryResult::PartialSuccess {
        result: successful_outputs,
        failed_nodes: failed_nodes.clone(),
        stats: stats.clone(),
    };

    match result {
        RecoveryResult::Success { result, stats } => {
            println!("Full success:");
            println!("  {} outputs computed", result.len());
            println!("  Retries: {}", stats.retries);
        }
        RecoveryResult::PartialSuccess {
            result,
            failed_nodes,
            stats,
        } => {
            println!("Partial success:");
            println!("  {} outputs computed successfully", result.len());
            println!("  {} nodes failed: {:?}", failed_nodes.len(), failed_nodes);
            println!("  Retries: {}", stats.retries);
            println!("  Checkpoints saved: {}", stats.checkpoints_saved);
            println!("  Checkpoints loaded: {}", stats.checkpoints_loaded);
            println!(
                "  Total recovery time: {:.2}ms",
                stats.total_recovery_time_ms
            );
        }
        RecoveryResult::Failure { error, stats } => {
            println!("Complete failure:");
            println!("  Error: {}", error);
            println!("  Retries exhausted: {}", stats.retries);
        }
    }

    // Show recovery metadata
    println!("\nRecovery statistics:");
    println!(
        "  Success rate: {:.1}%",
        (1.0 - failed_nodes.len() as f64 / 6.0) * 100.0
    );
    println!(
        "  Average recovery time per retry: {:.2}ms",
        stats.total_recovery_time_ms / stats.retries.max(1) as f64
    );

    Ok(())
}
