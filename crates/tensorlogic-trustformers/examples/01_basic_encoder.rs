//! Basic Transformer Encoder Example
//!
//! This example demonstrates how to create and use a basic transformer encoder
//! using the tensorlogic-trustformers crate.
//!
//! Run with: `cargo run --example 01_basic_encoder`

use tensorlogic_ir::EinsumGraph;
use tensorlogic_trustformers::{
    AttentionConfig, EncoderLayer, EncoderLayerConfig, EncoderStack, EncoderStackConfig,
    FeedForwardConfig, Result,
};

fn main() -> Result<()> {
    println!("=== TensorLogic Transformer Encoder Example ===\n");

    // Example 1: Single Encoder Layer
    println!("1. Creating a single encoder layer...");
    let config = EncoderLayerConfig::new(512, 8, 2048)?;
    let encoder_layer = EncoderLayer::new(config)?;
    println!("   ✓ Created encoder layer with d_model=512, n_heads=8, d_ff=2048");

    // Build einsum graph for the encoder layer
    let mut graph = EinsumGraph::new();
    graph.add_tensor("input"); // [batch, seq_len, d_model]

    let outputs = encoder_layer.build_encoder_layer_graph(&mut graph)?;
    println!(
        "   ✓ Built einsum graph with {} output tensors",
        outputs.len()
    );
    println!(
        "   ✓ Graph has {} nodes and {} tensors\n",
        graph.nodes.len(),
        graph.tensors.len()
    );

    // Example 2: Multi-Layer Encoder Stack (BERT-style)
    println!("2. Creating a 6-layer encoder stack (BERT-style)...");
    let stack_config = EncoderStackConfig::new(
        6,    // num_layers
        512,  // d_model
        8,    // n_heads
        2048, // d_ff
        512,  // max_seq_len
    )?
    .with_dropout(0.1);

    let encoder_stack = EncoderStack::new(stack_config)?;
    println!("   ✓ Created 6-layer encoder stack");

    // Build einsum graph for the stack
    let mut stack_graph = EinsumGraph::new();
    stack_graph.add_tensor("input_embeddings");

    let stack_outputs = encoder_stack.build_encoder_stack_graph(&mut stack_graph)?;
    println!(
        "   ✓ Built encoder stack graph with {} output tensors",
        stack_outputs.len()
    );
    println!(
        "   ✓ Stack graph has {} nodes and {} tensors\n",
        stack_graph.nodes.len(),
        stack_graph.tensors.len()
    );

    // Example 3: Custom Configuration
    println!("3. Creating encoder with custom configuration...");
    let custom_attn = AttentionConfig::new(768, 12)?
        .with_dropout(0.1)
        .with_causal(false);

    let custom_ffn = FeedForwardConfig::new(768, 3072)
        .with_activation("gelu")
        .with_dropout(0.1);

    println!(
        "   ✓ Attention: d_model={}, n_heads={}, d_k={}",
        custom_attn.d_model, custom_attn.n_heads, custom_attn.d_k
    );
    println!(
        "   ✓ FFN: d_model={}, d_ff={}, activation={}",
        custom_ffn.d_model, custom_ffn.d_ff, custom_ffn.activation
    );
    println!(
        "   ✓ Configuration validated: {}",
        custom_attn.validate().is_ok() && custom_ffn.validate().is_ok()
    );

    // Example 4: Using Model Presets
    println!("\n4. Using model presets...");
    use tensorlogic_trustformers::utils::presets;

    let bert_base = presets::bert_base();
    println!(
        "   ✓ BERT-base: {} layers, d_model={}, n_heads={}",
        bert_base.num_layers,
        bert_base.layer_config.attention.d_model,
        bert_base.layer_config.attention.n_heads
    );

    let gpt2_small = presets::gpt2_small();
    println!(
        "   ✓ GPT-2 small: {} layers, d_model={}, n_heads={}",
        gpt2_small.num_layers,
        gpt2_small.layer_config.attention.d_model,
        gpt2_small.layer_config.attention.n_heads
    );

    // Example 5: Model Statistics
    println!("\n5. Analyzing model statistics...");
    use tensorlogic_trustformers::encoder_stack_stats;

    let stats = encoder_stack_stats(&bert_base);
    println!("{}", stats.summary());

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
