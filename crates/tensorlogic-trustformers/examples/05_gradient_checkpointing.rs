//! Example demonstrating gradient checkpointing for memory-efficient training.
//!
//! Gradient checkpointing trades compute for memory by recomputing activations
//! during the backward pass instead of storing them. This allows training much
//! larger models or using larger batch sizes.
//!
//! Run with:
//! ```bash
//! cargo run --example 05_gradient_checkpointing
//! ```

use tensorlogic_trustformers::{CheckpointConfig, EncoderStackConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gradient Checkpointing Example ===\n");

    // Create a large encoder stack (12 layers, 768-dim, like BERT-base)
    let config = EncoderStackConfig::new(12, 768, 12, 3072, 512)?;

    println!("Model Configuration:");
    println!("  Layers: 12");
    println!("  d_model: 768");
    println!("  n_heads: 12");
    println!("  d_ff: 3072");
    println!("  max_seq_len: 512\n");

    // Calculate memory usage without checkpointing
    let stats = tensorlogic_trustformers::utils::encoder_stack_stats(&config);
    println!("Model Statistics (no checkpointing):");
    println!("{}", stats.summary());
    println!();

    // Demonstrate different checkpointing strategies
    demonstrate_uniform_checkpointing(&config);
    demonstrate_selective_checkpointing(&config);
    demonstrate_dynamic_checkpointing(&config);
    demonstrate_comparison(&config);

    Ok(())
}

/// Demonstrate uniform checkpointing
fn demonstrate_uniform_checkpointing(config: &EncoderStackConfig) {
    println!("=== 1. Uniform Checkpointing ===");
    println!("Checkpoint every N layers uniformly\n");

    for interval in [2, 3, 4] {
        let checkpoint_config = CheckpointConfig::uniform(interval);
        let num_layers = config.num_layers;

        println!("Interval: {} layers", interval);
        println!("  Strategy: {}", checkpoint_config.summary());
        println!(
            "  Memory savings: {:.1}%",
            checkpoint_config.memory_savings(num_layers) * 100.0
        );
        println!(
            "  Compute overhead: {:.2}x",
            checkpoint_config.compute_overhead(num_layers)
        );

        // Show which layers are checkpointed
        let checkpointed: Vec<usize> = (0..num_layers)
            .filter(|&i| checkpoint_config.should_checkpoint(i))
            .collect();
        println!("  Checkpointed layers: {:?}", checkpointed);
        println!();
    }
}

/// Demonstrate selective checkpointing
fn demonstrate_selective_checkpointing(config: &EncoderStackConfig) {
    println!("=== 2. Selective Checkpointing ===");
    println!("Checkpoint specific layers only\n");

    // Checkpoint at strategic points (e.g., every 3-4 layers)
    let layers = vec![0, 3, 6, 9];
    let checkpoint_config = CheckpointConfig::selective(layers.clone());
    let num_layers = config.num_layers;

    println!("Strategy: {}", checkpoint_config.summary());
    println!(
        "Memory savings: {:.1}%",
        checkpoint_config.memory_savings(num_layers) * 100.0
    );
    println!(
        "Compute overhead: {:.2}x",
        checkpoint_config.compute_overhead(num_layers)
    );
    println!("Checkpointed layers: {:?}", layers);
    println!();

    // Custom configuration: checkpoint only attention, not FFN
    let custom_config = CheckpointConfig::selective(layers)
        .with_checkpoint_attention(true)
        .with_checkpoint_ffn(false);

    println!("Custom: Checkpoint attention only");
    println!(
        "  Attention checkpointing: {}",
        custom_config.checkpoint_attention
    );
    println!("  FFN checkpointing: {}", custom_config.checkpoint_ffn);
    println!();
}

/// Demonstrate dynamic checkpointing
fn demonstrate_dynamic_checkpointing(config: &EncoderStackConfig) {
    println!("=== 3. Dynamic Checkpointing ===");
    println!("Automatically balance memory vs. compute\n");

    let num_layers = config.num_layers;

    for memory_fraction in [0.2, 0.3, 0.5] {
        let checkpoint_config = CheckpointConfig::dynamic(num_layers, memory_fraction).unwrap();

        println!(
            "Target memory: {:.0}% of full storage",
            memory_fraction * 100.0
        );
        println!("  Strategy: {}", checkpoint_config.summary());
        println!(
            "  Actual memory savings: {:.1}%",
            checkpoint_config.memory_savings(num_layers) * 100.0
        );
        println!(
            "  Compute overhead: {:.2}x",
            checkpoint_config.compute_overhead(num_layers)
        );

        // Show which layers are checkpointed
        let checkpointed: Vec<usize> = (0..num_layers)
            .filter(|&i| checkpoint_config.should_checkpoint(i))
            .collect();
        println!("  Checkpointed layers: {:?}", checkpointed);
        println!();
    }
}

/// Compare different strategies side by side
fn demonstrate_comparison(config: &EncoderStackConfig) {
    println!("=== 4. Strategy Comparison ===\n");

    let num_layers = config.num_layers;

    let strategies = vec![
        ("No checkpointing", CheckpointConfig::none()),
        ("Uniform (every 2 layers)", CheckpointConfig::uniform(2)),
        ("Uniform (every 3 layers)", CheckpointConfig::uniform(3)),
        (
            "Selective (4 checkpoints)",
            CheckpointConfig::selective(vec![0, 4, 8, 11]),
        ),
        (
            "Dynamic (30% memory)",
            CheckpointConfig::dynamic(num_layers, 0.3).unwrap(),
        ),
    ];

    println!(
        "{:<30} {:>15} {:>15}",
        "Strategy", "Memory Saved", "Compute Cost"
    );
    println!("{}", "-".repeat(62));

    for (name, config) in strategies {
        let memory_saved = config.memory_savings(num_layers) * 100.0;
        let compute_cost = config.compute_overhead(num_layers);

        println!(
            "{:<30} {:>14.1}% {:>14.2}x",
            name, memory_saved, compute_cost
        );
    }

    println!();
    println!("Recommendations:");
    println!("  - For 8GB GPU: Use uniform checkpointing every 2-3 layers");
    println!("  - For 16GB GPU: Use selective checkpointing or uniform every 4 layers");
    println!("  - For 24GB+ GPU: Dynamic checkpointing with 40-50% memory target");
    println!("  - For very large models: Uniform every 1-2 layers + selective attention only");
    println!();

    // Practical example for different model sizes
    println!("=== Practical Examples ===\n");

    let examples = vec![
        ("BERT-base (12 layers, 768d)", 12, 768),
        ("BERT-large (24 layers, 1024d)", 24, 1024),
        ("GPT-2 (12 layers, 768d)", 12, 768),
        ("GPT-2 XL (48 layers, 1600d)", 48, 1600),
    ];

    for (model_name, n_layers, d_model) in examples {
        println!("{}:", model_name);

        // Estimate memory usage (very rough)
        let params_millions =
            n_layers * (12 * d_model * d_model + 4 * d_model * d_model) / 1_000_000;
        let activation_memory_gb = (n_layers * d_model * d_model * 4) as f64 / 1_000_000_000.0;

        println!("  Parameters: ~{}M", params_millions);
        println!(
            "  Activation memory (no checkpoint): ~{:.1} GB",
            activation_memory_gb
        );

        // With checkpointing every 3 layers
        let checkpoint_config = CheckpointConfig::uniform(3);
        let savings = checkpoint_config.memory_savings(n_layers);
        let saved_gb = activation_memory_gb * savings;

        println!(
            "  With uniform(3): ~{:.1} GB (saves {:.1} GB)",
            activation_memory_gb - saved_gb,
            saved_gb
        );
        println!();
    }
}
