//! Mixture-of-Experts (MoE) example
//!
//! This example demonstrates how to create and configure MoE layers for
//! sparse transformer models using the tensorlogic-trustformers crate.
//!
//! Run with:
//! ```bash
//! cargo run --example 08_mixture_of_experts
//! ```

use tensorlogic_ir::EinsumGraph;
use tensorlogic_trustformers::moe::{MoeConfig, MoeLayer, MoePreset, RouterType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Mixture-of-Experts (MoE) Example ===\n");

    // Example 1: Create a custom MoE configuration
    println!("1. Creating custom MoE configuration:");
    println!("   - Model dimension: 512");
    println!("   - FFN dimension: 2048");
    println!("   - Number of experts: 8");
    println!("   - Experts per token: 2 (Top-2 routing)");

    let moe_config = MoeConfig::new(
        512,  // d_model
        2048, // d_ff
        8,    // num_experts
        2,    // experts_per_tok
    )?;

    let moe = MoeLayer::new(moe_config)?;

    println!("\n   Configuration created successfully!");
    println!("   Number of experts: {}", moe.config().num_experts);
    println!("   Experts per token: {}", moe.config().experts_per_tok);
    println!(
        "   Sparsity factor: {:.2}% (only {:.0}% of experts active)",
        moe.config().sparsity_factor() * 100.0,
        moe.config().sparsity_factor() * 100.0
    );
    println!(
        "   Total parameters: {:.2}M",
        moe.count_parameters() as f64 / 1_000_000.0
    );

    // Example 2: MoE Statistics
    println!("\n2. MoE layer statistics:");

    let stats = moe.stats();
    println!(
        "   Total parameters: {:.2}M",
        stats.total_params as f64 / 1_000_000.0
    );
    println!(
        "   Parameters per expert: {:.2}M",
        stats.params_per_expert as f64 / 1_000_000.0
    );
    println!(
        "   Active parameters per forward pass: {:.2}M",
        stats.active_params as f64 / 1_000_000.0
    );
    println!("   Sparsity: {:.2}", stats.sparsity);
    println!(
        "   Theoretical speedup vs dense: {:.2}x",
        stats.theoretical_speedup
    );

    // Example 3: Using MoE presets
    println!("\n3. MoE presets from research papers:");

    let presets = vec![
        (MoePreset::Mixtral8x7B, "Mixtral 8x7B (Mistral AI)"),
        (MoePreset::Switch, "Switch Transformer (Google)"),
        (MoePreset::GShard, "GShard (Google)"),
        (MoePreset::ExpertChoice, "Expert Choice"),
    ];

    for (preset, description) in &presets {
        let config = preset.config(512, 2048)?;
        let moe_preset = MoeLayer::new(config)?;
        let preset_stats = moe_preset.stats();

        println!("\n   {}", description);
        println!("     Name: {}", preset.name());
        println!("     Description: {}", preset.description());
        println!("     Experts: {}", moe_preset.config().num_experts);
        println!(
            "     Experts per token: {}",
            moe_preset.config().experts_per_tok
        );
        println!(
            "     Total params: {:.2}M",
            preset_stats.total_params as f64 / 1_000_000.0
        );
        println!(
            "     Active params: {:.2}M",
            preset_stats.active_params as f64 / 1_000_000.0
        );
        println!("     Speedup: {:.2}x", preset_stats.theoretical_speedup);
    }

    // Example 4: Building MoE computation graph
    println!("\n4. Building MoE computation graph:");

    let small_moe_config = MoeConfig::new(384, 1536, 4, 2)?;
    let small_moe = MoeLayer::new(small_moe_config)?;

    let mut graph = EinsumGraph::new();
    graph.add_tensor("input"); // Tensor 0: [batch, seq, d_model]
    graph.add_tensor("router_weights"); // Tensor 1: [d_model, num_experts]

    let outputs = small_moe.build_moe_graph(&mut graph)?;

    println!("   Graph built successfully!");
    println!("   Number of nodes: {}", graph.nodes.len());
    println!("   Number of tensors: {}", graph.tensors.len());
    println!(
        "   Output tensors: {} (moe_output + routing_weights)",
        outputs.len()
    );

    // Example 5: Router types
    println!("\n5. Different router/gating strategies:");

    let router_types = vec![
        (
            RouterType::TopK,
            "Top-K: Select K experts with highest scores",
        ),
        (
            RouterType::Softmax,
            "Softmax: Weighted combination of all experts",
        ),
        (
            RouterType::Switch,
            "Switch: Top-1 routing (single expert per token)",
        ),
        (
            RouterType::ExpertChoice,
            "Expert Choice: Experts select tokens",
        ),
    ];

    for (router_type, description) in router_types {
        let config = MoeConfig::new(512, 2048, 8, 2)?.with_router_type(router_type);

        println!("\n   {:?}:", router_type);
        println!("     {}", description);
        println!(
            "     Load balance coefficient: {}",
            config.load_balance_coef
        );
    }

    // Example 6: FLOPs and efficiency analysis
    println!("\n6. Computational efficiency analysis:");

    let batch_size = 32;
    let seq_len = 128;

    let configs = vec![
        ("Dense FFN (baseline)", MoeConfig::new(512, 2048, 1, 1)?),
        ("MoE 4 experts, Top-1", MoeConfig::new(512, 2048, 4, 1)?),
        ("MoE 8 experts, Top-2", MoeConfig::new(512, 2048, 8, 2)?),
        ("MoE 16 experts, Top-2", MoeConfig::new(512, 2048, 16, 2)?),
    ];

    for (name, config) in configs {
        let moe_layer = MoeLayer::new(config)?;
        let flops = moe_layer.count_flops(batch_size, seq_len);
        let params = moe_layer.count_parameters();
        let stats = moe_layer.stats();

        println!("\n   {}:", name);
        println!("     Parameters: {:.2}M", params as f64 / 1_000_000.0);
        println!(
            "     Active params: {:.2}M",
            stats.active_params as f64 / 1_000_000.0
        );
        println!("     FLOPs: {:.2}M", flops as f64 / 1_000_000.0);
        println!(
            "     Param efficiency: {:.2}x capacity at {:.0}% compute",
            moe_layer.config().num_experts as f64,
            stats.sparsity * 100.0
        );
    }

    // Example 7: Advanced configuration
    println!("\n7. Advanced MoE configuration:");

    let advanced_moe = MoeConfig::new(512, 2048, 8, 2)?
        .with_router_type(RouterType::TopK)
        .with_load_balance_coef(0.02) // Stronger load balancing
        .with_router_dropout(0.1) // Router dropout
        .with_expert_dropout(0.1) // Expert dropout
        .with_activation("swiglu"); // SwiGLU activation

    println!("   Advanced MoE configuration:");
    println!("     Router type: {:?}", advanced_moe.router_type);
    println!(
        "     Load balance coefficient: {}",
        advanced_moe.load_balance_coef
    );
    println!("     Router dropout: {}", advanced_moe.router_dropout);
    println!("     Expert dropout: {}", advanced_moe.expert_dropout);
    println!("     Activation: {}", advanced_moe.activation);

    // Example 8: Memory usage analysis
    println!("\n8. Memory usage analysis:");

    let memory_configs = vec![
        ("Small (4 experts)", MoeConfig::new(512, 2048, 4, 2)?),
        ("Medium (8 experts)", MoeConfig::new(512, 2048, 8, 2)?),
        ("Large (16 experts)", MoeConfig::new(512, 2048, 16, 2)?),
        ("Very Large (32 experts)", MoeConfig::new(512, 2048, 32, 2)?),
    ];

    for (name, config) in memory_configs {
        let moe_layer = MoeLayer::new(config)?;
        let memory_bytes = moe_layer.expert_memory_usage();
        let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);

        println!("\n   {}:", name);
        println!("     Expert memory: {:.2} MB", memory_mb);
        println!(
            "     Per expert: {:.2} MB",
            memory_mb / moe_layer.config().num_experts as f64
        );
    }

    // Example 9: Comparison with dense models
    println!("\n9. MoE vs Dense model comparison:");

    println!("\n   Scenario: Same compute budget");
    let dense_config = MoeConfig::new(512, 2048, 1, 1)?;
    let moe_config = MoeConfig::new(512, 2048, 8, 2)?;

    let dense_moe = MoeLayer::new(dense_config)?;
    let sparse_moe = MoeLayer::new(moe_config)?;

    let dense_stats = dense_moe.stats();
    let sparse_stats = sparse_moe.stats();

    println!("     Dense model:");
    println!(
        "       Parameters: {:.2}M",
        dense_stats.total_params as f64 / 1_000_000.0
    );
    println!(
        "       Active params: {:.2}M",
        dense_stats.active_params as f64 / 1_000_000.0
    );

    println!("\n     MoE model (8 experts, Top-2):");
    println!(
        "       Parameters: {:.2}M ({:.1}x more)",
        sparse_stats.total_params as f64 / 1_000_000.0,
        sparse_stats.total_params as f64 / dense_stats.total_params as f64
    );
    println!(
        "       Active params: {:.2}M ({:.1}x more)",
        sparse_stats.active_params as f64 / 1_000_000.0,
        sparse_stats.active_params as f64 / dense_stats.active_params as f64
    );
    println!(
        "       Capacity increase: {:.1}x at same compute",
        sparse_stats.total_params as f64 / dense_stats.total_params as f64
    );

    println!("\n=== Example completed successfully! ===");

    Ok(())
}
