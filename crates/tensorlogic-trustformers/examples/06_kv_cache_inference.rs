//! Example demonstrating KV-cache for efficient autoregressive inference.
//!
//! During autoregressive generation (e.g., text generation), transformers repeatedly
//! compute attention over the same prefix tokens. KV-caching stores the key and value
//! projections from previous steps, avoiding redundant computation.
//!
//! This example shows:
//! - Creating and configuring KV-cache
//! - Simulating autoregressive generation with caching
//! - Measuring cache statistics and memory usage
//! - Comparing cached vs. uncached computation costs
//!
//! Run with:
//! ```bash
//! cargo run --example 06_kv_cache_inference
//! ```

use tensorlogic_trustformers::{DecoderStackConfig, KVCache, KVCacheConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== KV-Cache for Efficient Inference ===\n");

    // Configuration for a GPT-2 small-like model
    let model_config = DecoderStackConfig::new(12, 768, 12, 3072, 1024)?;

    println!("Model Configuration:");
    println!("  Layers: {}", model_config.num_layers);
    println!("  d_model: 768");
    println!("  n_heads: 12");
    println!("  d_ff: 3072");
    println!("  max_seq_len: 1024\n");

    // Demonstrate different cache configurations
    demonstrate_basic_caching();
    demonstrate_memory_usage();
    demonstrate_generation_simulation();
    demonstrate_performance_comparison();

    Ok(())
}

/// Demonstrate basic KV-cache usage
fn demonstrate_basic_caching() {
    println!("=== 1. Basic KV-Cache Usage ===\n");

    // Create cache for 12-layer model with 12 heads and 64-dim per head
    let mut cache = KVCache::new(12, 12, 64);

    println!("Initial state:");
    println!("  Cached layers: {}", cache.num_cached_layers());
    println!("  Generation step: {}", cache.step());
    println!("  Memory usage: {:.2} MB", cache.current_memory_usage_mb());
    println!();

    // Simulate adding keys/values for first token
    println!("Step 1: Processing first token...");
    let keys = vec![0.1f32; 12 * 64]; // batch=1, heads=12, tokens=1, dim=64
    let values = vec![0.2f32; 12 * 64];

    for layer_idx in 0..12 {
        cache
            .update_layer(layer_idx, keys.clone(), values.clone())
            .unwrap();
    }
    cache.next_step();

    println!("  Cached layers: {}", cache.num_cached_layers());
    println!("  Sequence length: {}", cache.get_seq_len(0).unwrap());
    println!("  Memory usage: {:.2} MB", cache.current_memory_usage_mb());
    println!();

    // Add second token
    println!("Step 2: Processing second token...");
    cache.update_layer(0, keys.clone(), values.clone()).unwrap();
    cache.next_step();

    println!("  Sequence length: {}", cache.get_seq_len(0).unwrap());
    println!(
        "  Memory usage: {:.2} MB\n",
        cache.current_memory_usage_mb()
    );
}

/// Demonstrate memory usage with different configurations
fn demonstrate_memory_usage() {
    println!("=== 2. Memory Usage Analysis ===\n");

    let configurations = vec![
        ("GPT-2 Small", 12, 12, 64, 1024),
        ("GPT-2 Medium", 24, 16, 64, 1024),
        ("GPT-2 Large", 36, 20, 64, 1024),
        ("GPT-2 XL", 48, 25, 64, 1024),
    ];

    println!(
        "{:<20} {:>10} {:>15} {:>15}",
        "Model", "Layers", "Max Memory", "Per Token"
    );
    println!("{}", "-".repeat(62));

    for (name, layers, heads, head_dim, max_seq) in configurations {
        let config = KVCacheConfig::new(layers, heads, head_dim).with_max_seq_len(max_seq);

        let total_memory_mb = config.memory_usage_mb();
        let per_token_mb = total_memory_mb / max_seq as f64;

        println!(
            "{:<20} {:>10} {:>13.1} MB {:>13.3} MB",
            name, layers, total_memory_mb, per_token_mb
        );
    }

    println!("\nNote: Memory usage scales linearly with sequence length.");
    println!("KV-cache is most beneficial for long sequences!\n");
}

/// Simulate autoregressive generation with caching
fn demonstrate_generation_simulation() {
    println!("=== 3. Autoregressive Generation Simulation ===\n");

    let num_layers = 12;
    let num_heads = 12;
    let head_dim = 64;

    let mut cache = KVCache::new(num_layers, num_heads, head_dim);

    println!("Generating 10 tokens with KV-cache:\n");

    for step in 0..10 {
        // Simulate computing K,V for new token
        let keys = vec![0.1f32; num_heads * head_dim];
        let values = vec![0.2f32; num_heads * head_dim];

        // Update cache for all layers
        for layer_idx in 0..num_layers {
            cache
                .update_layer(layer_idx, keys.clone(), values.clone())
                .unwrap();
        }

        cache.next_step();

        // Get statistics
        let stats = cache.stats();

        println!(
            "Step {:2}: seq_len={:2}, memory={:5.2} MB, cached_layers={}",
            step + 1,
            stats.total_seq_len,
            stats.memory_usage_mb,
            stats.num_layers
        );
    }

    println!("\nFinal cache statistics:");
    println!("{}", cache.stats().summary());
    println!();
}

/// Compare performance with and without caching
fn demonstrate_performance_comparison() {
    println!("=== 4. Performance Comparison ===\n");

    let num_layers = 12;
    let num_heads = 12;
    let head_dim = 64;
    let d_model = num_heads * head_dim; // 768

    println!("Analyzing computational cost for different sequence lengths:\n");
    println!(
        "{:<12} {:>15} {:>15} {:>15}",
        "Seq Length", "Without Cache", "With Cache", "Speedup"
    );
    println!("{}", "-".repeat(60));

    for seq_len in [10, 50, 100, 200, 500, 1000] {
        // Without cache: recompute attention for all tokens at each step
        // Cost: O(seq_len * seq_len * d_model) per step
        // For seq_len steps: O(seq_len^3 * d_model)
        let uncached_ops =
            seq_len as u64 * seq_len as u64 * seq_len as u64 * d_model as u64 * num_layers as u64;

        // With cache: only compute attention for new token
        // Cost: O(seq_len * d_model) per step
        // For seq_len steps: O(seq_len^2 * d_model)
        let cached_ops = seq_len as u64 * seq_len as u64 * d_model as u64 * num_layers as u64;

        let speedup = uncached_ops as f64 / cached_ops as f64;

        println!(
            "{:<12} {:>13.1}M {:>13.1}M {:>14.1}x",
            seq_len,
            uncached_ops as f64 / 1_000_000.0,
            cached_ops as f64 / 1_000_000.0,
            speedup
        );
    }

    println!("\nKey Insights:");
    println!("  - Speedup increases linearly with sequence length");
    println!("  - For 1000 tokens: ~1000x faster with KV-cache");
    println!("  - Memory cost: ~2-10 MB for typical models");
    println!("  - Trade-off: O(N) memory for O(N^2) time savings\n");

    // Demonstrate cache configuration options
    println!("=== Cache Configuration Options ===\n");

    let configs = vec![
        ("Default", KVCacheConfig::new(12, 12, 64)),
        (
            "Large Context",
            KVCacheConfig::new(12, 12, 64).with_max_seq_len(4096),
        ),
        (
            "Large Batch",
            KVCacheConfig::new(12, 12, 64).with_max_batch_size(64),
        ),
        (
            "Disabled",
            KVCacheConfig::new(12, 12, 64).with_enabled(false),
        ),
    ];

    for (name, config) in configs {
        println!("{}:", name);
        println!("  Max sequence: {}", config.max_seq_len);
        println!("  Max batch: {}", config.max_batch_size);
        println!("  Memory: {:.1} MB", config.memory_usage_mb());
        println!("  Enabled: {}", config.enabled);
        println!();
    }

    // Real-world scenarios
    println!("=== Real-World Scenarios ===\n");

    let scenarios = vec![
        ("Chatbot Response", 50, "Fast interactive chat"),
        ("Code Completion", 100, "IDE autocomplete"),
        ("Document Generation", 500, "Article writing"),
        ("Long-form Content", 2000, "Book chapter generation"),
    ];

    println!(
        "{:<25} {:>12} {:>20}",
        "Use Case", "Tokens", "Speedup (approx)"
    );
    println!("{}", "-".repeat(60));

    for (name, tokens, description) in scenarios {
        // Approximate speedup for autoregressive generation
        let speedup = tokens as f64;

        println!(
            "{:<25} {:>12} {:>19.0}x - {}",
            name, tokens, speedup, description
        );
    }

    println!("\nRecommendations:");
    println!("  - Always enable KV-cache for autoregressive generation");
    println!("  - Adjust max_seq_len based on your use case");
    println!("  - Monitor memory usage for large batch sizes");
    println!("  - Clear cache between independent generations");
}
