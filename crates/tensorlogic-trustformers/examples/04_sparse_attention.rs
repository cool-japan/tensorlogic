//! Sparse Attention Example
//!
//! This example demonstrates efficient sparse attention patterns for long sequences,
//! reducing the O(n²) complexity of standard attention.
//!
//! Run with: `cargo run --example 04_sparse_attention`

use tensorlogic_ir::EinsumGraph;
use tensorlogic_trustformers::{
    AttentionConfig, LocalAttention, Result, SparseAttention, SparseAttentionConfig,
    SparsePatternType,
};

fn main() -> Result<()> {
    println!("=== Sparse Attention Example ===\n");

    // Example 1: Strided Sparse Attention
    println!("1. Strided sparse attention (attend every k-th position)...");

    let base_config = AttentionConfig::new(512, 8)?;
    let strided_config = SparseAttentionConfig::strided(base_config.clone(), 4)?;
    let strided_attn = SparseAttention::new(strided_config)?;

    println!("   ✓ Created strided sparse attention with stride=4");
    println!("   ✓ Complexity: O(n²/4) instead of O(n²)");
    println!("   ✓ Pattern: Attend to positions [0, 4, 8, 12, ...]");
    println!("   ✓ Use case: Long-range dependencies with fixed stride\n");

    let sparsity = strided_attn.sparsity_factor(1024);
    println!(
        "   Memory savings at seq_len=1024: {:.1}%\n",
        (1.0 - sparsity) * 100.0
    );

    // Example 2: Local Windowed Attention
    println!("2. Local windowed attention (attend within a window)...");

    let local_config = SparseAttentionConfig::local(base_config.clone(), 128)?;
    let _local_attn = SparseAttention::new(local_config)?;

    println!("   ✓ Created local attention with window_size=128");
    println!("   ✓ Complexity: O(n × w) where w=128");
    println!("   ✓ Pattern: Each position attends to ±64 neighbors");
    println!("   ✓ Use case: Local context modeling (e.g., character-level)\n");

    // Dedicated LocalAttention for efficiency
    let efficient_local = LocalAttention::new(base_config.clone(), 64)?;
    println!("   ✓ Dedicated LocalAttention implementation");
    println!(
        "   Memory savings at seq_len=2048: {:.1}%\n",
        efficient_local.memory_savings(2048) * 100.0
    );

    // Example 3: Block-Sparse Attention
    println!("3. Block-sparse attention (block diagonal pattern)...");

    let block_config = SparseAttentionConfig::block_sparse(base_config.clone(), 64)?;
    let block_attn = SparseAttention::new(block_config)?;

    println!("   ✓ Created block-sparse attention with block_size=64");
    println!("   ✓ Complexity: O(n²/block_size)");
    println!("   ✓ Pattern: Attend within 64×64 blocks along diagonal");
    println!("   ✓ Use case: Document modeling with paragraph boundaries\n");

    let block_sparsity = block_attn.sparsity_factor(2048);
    println!(
        "   Sparsity at seq_len=2048: {:.3} (only {:.1}% of full attention)\n",
        block_sparsity,
        block_sparsity * 100.0
    );

    // Example 4: Global-Local Attention
    println!("4. Global-local attention (some tokens attend globally)...");

    // Global positions: every 8th position
    let global_positions: Vec<usize> = (0..256).step_by(8).collect();
    let global_local_config =
        SparseAttentionConfig::global_local(base_config.clone(), 64, global_positions)?;
    let global_local_attn = SparseAttention::new(global_local_config)?;

    println!("   ✓ Created global-local attention");
    println!("   ✓ Local window: 64");
    println!("   ✓ Global tokens: every 8th position");
    println!("   ✓ Pattern: Local + global tokens for long-range");
    println!("   ✓ Use case: Longformer-style attention\n");

    // Build graph
    let mut graph = EinsumGraph::new();
    graph.add_tensor("Q");
    graph.add_tensor("K");
    graph.add_tensor("V");

    let outputs = global_local_attn.build_sparse_attention_graph(&mut graph)?;
    println!("   ✓ Built graph with {} outputs\n", outputs.len());

    // Example 5: Pattern Comparison
    println!("5. Sparse pattern comparison...");
    println!("\n   Sequence length: 2048 tokens");
    println!("   ╔════════════════╦══════════════╦═══════════════╦════════════════╗");
    println!("   ║ Pattern        ║ Complexity   ║ Memory Usage  ║ Sparsity       ║");
    println!("   ╠════════════════╬══════════════╬═══════════════╬════════════════╣");
    println!("   ║ Full Attention ║ O(n²)        ║ 4.2M elements ║ 1.000 (100%)   ║");
    println!("   ╠════════════════╬══════════════╬═══════════════╬════════════════╣");

    let global_pos: Vec<usize> = (0..256).step_by(8).collect();
    let patterns = vec![
        (
            "Strided (s=4)",
            SparseAttentionConfig::strided(base_config.clone(), 4)?,
        ),
        (
            "Local (w=128)",
            SparseAttentionConfig::local(base_config.clone(), 128)?,
        ),
        (
            "Block (b=64)",
            SparseAttentionConfig::block_sparse(base_config.clone(), 64)?,
        ),
        (
            "Global-Local",
            SparseAttentionConfig::global_local(base_config.clone(), 64, global_pos)?,
        ),
    ];

    for (name, config) in patterns {
        let attn = SparseAttention::new(config)?;
        let sparsity = attn.sparsity_factor(2048);
        let memory = 2048.0 * 2048.0 * sparsity / 1_000_000.0;
        let complexity = match attn.config.pattern {
            SparsePatternType::Strided { stride } => format!("O(n²/{})", stride),
            SparsePatternType::Local { window_size } => format!("O(n×{})", window_size),
            SparsePatternType::BlockSparse { block_size } => {
                format!("O(n²/{})", block_size)
            }
            SparsePatternType::GlobalLocal { .. } => "O(n×w + g²)".to_string(),
            SparsePatternType::Random { num_random } => format!("O(n×{})", num_random),
        };

        println!(
            "   ║ {:<14} ║ {:<12} ║ {:>6.1}M        ║ {:>5.3} ({:>5.1}%) ║",
            name,
            complexity,
            memory,
            sparsity,
            sparsity * 100.0
        );
    }
    println!("   ╚════════════════╩══════════════╩═══════════════╩════════════════╝\n");

    // Example 6: Memory Savings Analysis
    println!("6. Memory savings for different sequence lengths...");
    println!("\n   Using Local Attention (window=128):");

    let local_64 = LocalAttention::new(base_config.clone(), 64)?;

    let seq_lengths = vec![512, 1024, 2048, 4096, 8192];
    for seq_len in seq_lengths {
        let savings = local_64.memory_savings(seq_len);
        println!(
            "      seq_len={:>5}: {:.1}% memory savings",
            seq_len,
            savings * 100.0
        );
    }
    println!();

    // Example 7: Pattern Descriptions
    println!("7. Sparse pattern descriptions...");

    let global_p: Vec<usize> = (0..256).step_by(16).collect();
    let configs = vec![
        (
            "Strided (stride=8)",
            SparseAttentionConfig::strided(base_config.clone(), 8)?,
        ),
        (
            "Local (window=256)",
            SparseAttentionConfig::local(base_config.clone(), 256)?,
        ),
        (
            "Block Sparse (block=128)",
            SparseAttentionConfig::block_sparse(base_config.clone(), 128)?,
        ),
        (
            "Global-Local (window=128)",
            SparseAttentionConfig::global_local(base_config.clone(), 128, global_p)?,
        ),
    ];

    for (desc, config) in configs {
        match config.pattern {
            SparsePatternType::Strided { stride } => {
                println!("   • {}: Attend every {}th position", desc, stride);
            }
            SparsePatternType::Local { window_size } => {
                println!(
                    "   • {}: Attend within window of size {}",
                    desc, window_size
                );
            }
            SparsePatternType::BlockSparse { block_size } => {
                println!(
                    "   • {}: Divide into {}×{} blocks",
                    desc, block_size, block_size
                );
            }
            SparsePatternType::GlobalLocal {
                window_size,
                ref global_positions,
            } => {
                println!(
                    "   • {}: Local window {} + {} global tokens",
                    desc,
                    window_size,
                    global_positions.len()
                );
            }
            SparsePatternType::Random { num_random } => {
                println!("   • {}: Randomly sample {} positions", desc, num_random);
            }
        }
    }
    println!();

    // Example 8: Choosing the Right Pattern
    println!("8. Choosing the right sparse pattern...");
    println!("\n   ┌─────────────────────┬────────────────────────────────────────┐");
    println!("   │ Use Case            │ Recommended Pattern                    │");
    println!("   ├─────────────────────┼────────────────────────────────────────┤");
    println!("   │ Character-level LM  │ Local (small window, ~64-128)          │");
    println!("   ├─────────────────────┼────────────────────────────────────────┤");
    println!("   │ Long documents      │ Global-Local (Longformer-style)        │");
    println!("   ├─────────────────────┼────────────────────────────────────────┤");
    println!("   │ Image patches       │ Block-Sparse (patch boundaries)        │");
    println!("   ├─────────────────────┼────────────────────────────────────────┤");
    println!("   │ Time series         │ Local or Strided                       │");
    println!("   ├─────────────────────┼────────────────────────────────────────┤");
    println!("   │ Protein sequences   │ Global-Local (distant interactions)    │");
    println!("   └─────────────────────┴────────────────────────────────────────┘\n");

    // Example 9: Combining with Other Features
    println!("9. Combining sparse attention with other features...");
    println!("\n   ✓ Can combine with:");
    println!("      - Position encodings (relative, RoPE, ALiBi)");
    println!("      - Layer normalization (LayerNorm, RMSNorm)");
    println!("      - Rule-based patterns for interpretability");
    println!("      - Multi-head attention (each head can have different pattern)");
    println!("\n   ✓ Use in:");
    println!("      - Encoder stacks (for long documents)");
    println!("      - Decoder stacks (for long-context generation)");
    println!("      - Cross-attention (encoder-decoder)");

    println!("\n=== Sparse attention example completed successfully! ===");
    Ok(())
}
