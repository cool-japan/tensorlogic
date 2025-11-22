//! Example 09: Modern LLM Optimizations
//!
//! This example demonstrates the new alpha.2 features for efficient LLM training and inference:
//! - Grouped-Query Attention (GQA): Reduces KV cache memory
//! - Sliding Window Attention: Efficient long-context handling
//! - LoRA: Parameter-efficient fine-tuning
//!
//! These optimizations are used in modern models like LLaMA 2, Mistral, and others.

use tensorlogic_ir::EinsumGraph;
use tensorlogic_trustformers::{
    gqa::{GQAConfig, GQAPreset, GQAStats, GroupedQueryAttention},
    lora::{LoRAAttention, LoRAConfig, LoRALinear, LoRAPreset, LoRAStats},
    sliding_window::{
        SlidingWindowAttention, SlidingWindowConfig, SlidingWindowPreset, SlidingWindowStats,
    },
};

fn main() {
    println!("=== Modern LLM Optimizations Demo ===\n");

    demo_grouped_query_attention();
    demo_sliding_window_attention();
    demo_lora();
}

fn demo_grouped_query_attention() {
    println!("--- Grouped-Query Attention (GQA) ---\n");

    // GQA reduces KV cache memory by using fewer KV heads than query heads

    // 1. Compare different attention types
    println!("Comparing attention configurations:\n");

    // Standard MHA (LLaMA 2 7B)
    let mha_config = GQAPreset::Llama2_7B.config().unwrap();
    let mha_stats = GQAStats::from_config(&mha_config);
    println!("LLaMA 2 7B (MHA):");
    println!(
        "  n_heads: {}, n_kv_heads: {}",
        mha_config.n_heads, mha_config.n_kv_heads
    );
    println!("  KV cache reduction: {:.1}x", mha_stats.kv_cache_reduction);
    println!();

    // GQA (LLaMA 2 70B)
    let gqa_config = GQAPreset::Llama2_70B.config().unwrap();
    let gqa_stats = GQAStats::from_config(&gqa_config);
    println!("LLaMA 2 70B (GQA):");
    println!(
        "  n_heads: {}, n_kv_heads: {}",
        gqa_config.n_heads, gqa_config.n_kv_heads
    );
    println!("  KV cache reduction: {:.1}x", gqa_stats.kv_cache_reduction);
    println!();

    // GQA (Mistral 7B)
    let mistral_config = GQAPreset::Mistral7B.config().unwrap();
    let mistral_stats = GQAStats::from_config(&mistral_config);
    println!("Mistral 7B (GQA):");
    println!(
        "  n_heads: {}, n_kv_heads: {}",
        mistral_config.n_heads, mistral_config.n_kv_heads
    );
    println!(
        "  KV cache reduction: {:.1}x",
        mistral_stats.kv_cache_reduction
    );
    println!();

    // MQA (Falcon 40B)
    let mqa_config = GQAPreset::Falcon40B.config().unwrap();
    let mqa_stats = GQAStats::from_config(&mqa_config);
    println!("Falcon 40B (MQA):");
    println!(
        "  n_heads: {}, n_kv_heads: {}",
        mqa_config.n_heads, mqa_config.n_kv_heads
    );
    println!("  KV cache reduction: {:.1}x", mqa_stats.kv_cache_reduction);
    println!();

    // 2. Build a GQA graph
    let config = GQAConfig::new(512, 8, 2).unwrap().with_causal(true);
    let gqa = GroupedQueryAttention::new(config.clone()).unwrap();

    let mut graph = EinsumGraph::new();
    graph.add_tensor("Q");
    graph.add_tensor("K");
    graph.add_tensor("V");

    let outputs = gqa.build_gqa_graph(&mut graph).unwrap();

    println!("Custom GQA graph built:");
    println!(
        "  d_model: {}, n_heads: {}, n_kv_heads: {}",
        config.d_model, config.n_heads, config.n_kv_heads
    );
    println!("  Group size: {}", config.group_size());
    println!("  Output tensor index: {}", outputs[0]);
    println!("  Nodes in graph: {}", graph.nodes.len());

    // 3. Memory savings calculation
    let savings = gqa.memory_savings(1, 4096);
    println!("  KV cache memory savings: {:.1}%", savings * 100.0);

    println!("\n");
}

fn demo_sliding_window_attention() {
    println!("--- Sliding Window Attention ---\n");

    // Sliding Window Attention reduces O(n^2) to O(n*w) complexity

    // 1. Show preset configurations
    println!("Preset configurations:\n");

    let mistral_config = SlidingWindowPreset::Mistral7B.config().unwrap();
    println!("Mistral 7B:");
    println!("  window_size: {}", mistral_config.window_size);
    println!("  causal: {}", mistral_config.causal);

    let longformer_config = SlidingWindowPreset::LongformerBase.config().unwrap();
    println!("\nLongformer Base:");
    println!("  window_size: {}", longformer_config.window_size);

    let bigbird_config = SlidingWindowPreset::BigBirdBase.config().unwrap();
    println!("\nBigBird Base:");
    println!("  window_size: {}", bigbird_config.window_size);

    // 2. Calculate complexity reduction for different sequence lengths
    println!("\nComplexity reduction for different sequence lengths:\n");

    let config = SlidingWindowConfig::new(512, 8, 256).unwrap();

    for seq_len in [512, 2048, 8192, 32768] {
        let stats = SlidingWindowStats::from_config(&config, seq_len);
        let reduction_pct = (1.0 - stats.complexity_reduction) * 100.0;
        println!("  seq_len={}: {:.1}% reduction", seq_len, reduction_pct);
    }

    // 3. Build a sliding window attention graph
    println!("\nBuilding sliding window attention graph:");

    let config = SlidingWindowConfig::new(512, 8, 256)
        .unwrap()
        .with_causal(true);
    let swa = SlidingWindowAttention::new(config.clone()).unwrap();

    let mut graph = EinsumGraph::new();
    graph.add_tensor("Q");
    graph.add_tensor("K");
    graph.add_tensor("V");

    let outputs = swa.build_swa_graph(&mut graph).unwrap();

    println!(
        "  d_model: {}, n_heads: {}, window_size: {}",
        config.d_model, config.n_heads, config.window_size
    );
    println!("  Output tensor index: {}", outputs[0]);
    println!("  Nodes in graph: {}", graph.nodes.len());

    println!("\n");
}

fn demo_lora() {
    println!("--- LoRA (Low-Rank Adaptation) ---\n");

    // LoRA enables parameter-efficient fine-tuning

    // 1. Show compression ratios for different ranks
    println!("Compression ratios for d=512 linear layer:\n");

    for (name, preset) in [
        ("Minimal (r=4)", LoRAPreset::Minimal),
        ("Standard (r=8)", LoRAPreset::Standard),
        ("Extended (r=16)", LoRAPreset::Extended),
    ] {
        let config = preset.config();
        let ratio = config.compression_ratio(512, 512);
        let trainable = config.trainable_params(512, 512);
        println!(
            "  {}: {:.1}x compression ({} trainable params)",
            name, ratio, trainable
        );
    }

    // 2. Calculate stats for a full model
    println!("\nLoRA stats for a 6-layer, d=512 model:\n");

    let config = LoRAConfig::new(8, 16.0);
    let stats = LoRAStats::for_model(&config, 512, 6);

    println!("  Trainable parameters: {}", stats.trainable_params);
    println!("  Frozen parameters: {}", stats.frozen_params);
    println!(
        "  Total parameters: {}",
        stats.trainable_params + stats.frozen_params
    );
    println!("  Compression ratio: {:.1}x", stats.compression_ratio);
    println!("  Memory savings: {:.1}%", stats.memory_savings * 100.0);

    // 3. Build a LoRA linear layer graph
    println!("\nBuilding LoRA linear layer graph:");

    let config = LoRAConfig::new(8, 16.0);
    let lora = LoRALinear::new(512, 512, config.clone())
        .unwrap()
        .with_name("demo");

    let mut graph = EinsumGraph::new();
    graph.add_tensor("x");

    let outputs = lora.build_lora_graph(&mut graph).unwrap();

    println!(
        "  rank: {}, alpha: {}, scaling: {}",
        config.rank,
        config.alpha,
        config.scaling()
    );
    println!("  Trainable params: {}", lora.trainable_params());
    println!("  Output tensor index: {}", outputs[0]);
    println!("  Nodes in graph: {}", graph.nodes.len());

    // 4. Build a LoRA attention layer graph
    println!("\nBuilding LoRA attention layer graph:");

    let config = LoRAConfig::new(8, 16.0).with_projections(true, true);
    let lora_attn = LoRAAttention::new(512, 8, config).unwrap();

    let mut graph = EinsumGraph::new();
    graph.add_tensor("Q");
    graph.add_tensor("K");
    graph.add_tensor("V");

    let outputs = lora_attn.build_lora_attention_graph(&mut graph).unwrap();

    println!("  LoRA applied to: Q and V projections");
    println!("  Trainable params: {}", lora_attn.trainable_params());
    println!("  Output tensor index: {}", outputs[0]);
    println!("  Nodes in graph: {}", graph.nodes.len());

    println!("\n");
    println!("=== Demo Complete ===");
}
