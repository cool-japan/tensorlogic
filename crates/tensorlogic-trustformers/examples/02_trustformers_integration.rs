//! TrustformeRS Integration Example
//!
//! This example demonstrates how to integrate TensorLogic transformer components
//! with TrustformeRS, including bidirectional conversion and weight loading.
//!
//! Run with: `cargo run --example 02_trustformers_integration`

use tensorlogic_ir::{EinsumGraph, TLExpr};
use tensorlogic_trustformers::{
    EncoderLayer, EncoderLayerConfig, EncoderStack, EncoderStackConfig, IntegrationConfig,
    ModelConfig, Result, TensorLogicModel, TrustformersConverter, TrustformersWeightLoader,
};

fn main() -> Result<()> {
    println!("=== TrustformeRS Integration Example ===\n");

    // Example 1: Wrap TensorLogic Model for TrustformeRS
    println!("1. Wrapping TensorLogic components as TrustformeRS models...");

    let layer_config = EncoderLayerConfig::new(512, 8, 2048)?;
    let encoder_layer = EncoderLayer::new(layer_config.clone())?;

    // Wrap as TensorLogicModel
    let tl_model = TensorLogicModel::from_encoder_layer(encoder_layer, layer_config)?;
    println!("   ✓ Wrapped encoder layer as TensorLogicModel");

    // Get model configuration
    if let ModelConfig::EncoderLayer {
        d_model,
        n_heads,
        d_ff,
        ..
    } = tl_model.config()
    {
        println!(
            "   ✓ Model config: d_model={}, n_heads={}, d_ff={}",
            d_model, n_heads, d_ff
        );
    }

    // Build einsum graph
    let mut graph = EinsumGraph::new();
    graph.add_tensor("input");
    let outputs = tl_model.build_graph(&mut graph)?;
    println!("   ✓ Built graph with {} outputs\n", outputs.len());

    // Example 2: Convert TensorLogic to TLExpr
    println!("2. Converting TensorLogic models to TLExpr...");

    let tlexpr = tl_model.to_tlexpr()?;
    match &tlexpr {
        TLExpr::And(..) => {
            println!("   ✓ Converted to TLExpr (And of attention and FFN)");
        }
        _ => {
            println!("   ✓ Converted to TLExpr: {:?}", tlexpr);
        }
    }

    // For encoder stacks, the representation uses ForAll
    let stack_config = EncoderStackConfig::new(6, 512, 8, 2048, 512)?;
    let encoder_stack = EncoderStack::new(stack_config.clone())?;
    let stack_model = TensorLogicModel::from_encoder_stack(encoder_stack, stack_config)?;

    let stack_expr = stack_model.to_tlexpr()?;
    if let TLExpr::ForAll { var, domain, .. } = &stack_expr {
        println!(
            "   ✓ Stack converted to ForAll: var='{}', domain='{}'",
            var, domain
        );
    }
    println!();

    // Example 3: Convert TrustformeRS Architectures to TLExpr
    println!("3. Converting TrustformeRS architectures to TensorLogic IR...");

    let converter = TrustformersConverter::new();

    // Convert BERT-style encoder
    let _bert_expr = converter.convert_bert_encoder(
        12,   // n_layers
        768,  // d_model
        12,   // n_heads
        3072, // d_ff
    )?;
    println!("   ✓ Converted BERT-base architecture to TLExpr");

    // Convert GPT-style decoder
    let _gpt_expr = converter.convert_gpt_decoder(
        12,   // n_layers
        768,  // d_model
        12,   // n_heads
        3072, // d_ff
    )?;
    println!("   ✓ Converted GPT-2 architecture to TLExpr (with causal masking)");

    // Convert full encoder-decoder transformer
    let _t5_expr = converter.convert_transformer(
        12,   // encoder_layers
        12,   // decoder_layers
        768,  // d_model
        12,   // n_heads
        3072, // d_ff
    )?;
    println!("   ✓ Converted T5-style encoder-decoder to TLExpr\n");

    // Example 4: Custom Integration Configuration
    println!("4. Using custom integration configuration...");

    let custom_config = IntegrationConfig::new()
        .with_shape_validation(true)
        .with_dropout_preservation(false)
        .with_pre_norm(true)
        .with_numerical_tolerance(1e-5);

    let _custom_converter = TrustformersConverter::with_config(custom_config);
    println!("   ✓ Created converter with custom config");
    println!("   ✓ Shape validation: enabled");
    println!("   ✓ Dropout preservation: disabled");
    println!("   ✓ Pre-normalization: enabled");
    println!("   ✓ Numerical tolerance: 1e-5\n");

    // Example 5: Weight Loading and Name Mapping
    println!("5. Weight loading and checkpoint format support...");

    let loader = TrustformersWeightLoader::new();

    // Map TrustformeRS layer names to TensorLogic tensor names
    let examples = vec![
        "encoder.layer.0.attention.query.weight",
        "encoder.layer.0.attention.key.weight",
        "encoder.layer.0.attention.value.weight",
        "encoder.layer.5.feed_forward.weight",
        "decoder.layer.3.attention.query.weight",
    ];

    println!("   Layer name mappings:");
    for name in examples {
        let mapped = loader.map_layer_name(name)?;
        println!("      {} → {}", name, mapped);
    }

    println!("\n   ✓ Weight loader supports:");
    println!("      - SafeTensors format");
    println!("      - PyTorch .bin files");
    println!("      - TensorFlow SavedModel");
    println!("      - Automatic name mapping\n");

    // Example 6: Typical Workflow
    println!("6. Complete integration workflow example...");
    println!("   Step 1: Create TensorLogic encoder");
    let config = EncoderStackConfig::new(6, 512, 8, 2048, 512)?;
    let encoder = EncoderStack::new(config.clone())?;
    println!("      ✓ Created 6-layer encoder");

    println!("   Step 2: Wrap as TrustformeRS-compatible model");
    let model = TensorLogicModel::from_encoder_stack(encoder, config)?;
    println!("      ✓ Wrapped model");

    println!("   Step 3: Convert to TLExpr for compilation");
    let _expr = model.to_tlexpr()?;
    println!("      ✓ Converted to TLExpr");

    println!("   Step 4: Build execution graph");
    let mut exec_graph = EinsumGraph::new();
    exec_graph.add_tensor("embeddings");
    let _outputs = model.build_graph(&mut exec_graph)?;
    println!(
        "      ✓ Built einsum graph ({} nodes)",
        exec_graph.nodes.len()
    );

    println!("   Step 5: Ready for compilation and execution on any backend!");
    println!("      ✓ Graph can be compiled with tensorlogic-compiler");
    println!("      ✓ Execute on tensorlogic-scirs-backend or other backends");
    println!("      ✓ Optimize using graph optimization passes\n");

    println!("=== Integration example completed successfully! ===");
    Ok(())
}
