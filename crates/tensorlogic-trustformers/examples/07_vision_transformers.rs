//! Vision Transformer (ViT) example
//!
//! This example demonstrates how to create and configure Vision Transformers
//! for image classification tasks using the tensorlogic-trustformers crate.
//!
//! Run with:
//! ```bash
//! cargo run --example 07_vision_transformers
//! ```

use tensorlogic_ir::EinsumGraph;
use tensorlogic_trustformers::vision::{
    PatchEmbeddingConfig, ViTPreset, VisionTransformer, VisionTransformerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vision Transformer (ViT) Example ===\n");

    // Example 1: Create a custom ViT configuration
    println!("1. Creating custom ViT configuration:");
    println!("   - Image size: 224x224");
    println!("   - Patch size: 16x16");
    println!("   - Input channels: 3 (RGB)");
    println!("   - Model dimension: 768");
    println!("   - Attention heads: 12");
    println!("   - FFN dimension: 3072");
    println!("   - Encoder layers: 12");
    println!("   - Number of classes: 1000 (ImageNet)");

    let vit_config = VisionTransformerConfig::new(
        224,  // image_size
        16,   // patch_size
        3,    // in_channels (RGB)
        768,  // d_model
        12,   // n_heads
        3072, // d_ff
        12,   // n_layers
        1000, // num_classes
    )?;

    let vit = VisionTransformer::new(vit_config)?;

    println!("\n   Configuration created successfully!");
    println!(
        "   Number of patches: {}",
        vit.config().patch_embed.num_patches()
    );
    println!(
        "   Patch dimension: {}",
        vit.config().patch_embed.patch_dim()
    );
    println!(
        "   Sequence length: {} (patches + class token)",
        vit.config().seq_length()
    );
    println!(
        "   Total parameters: ~{:.2}M",
        vit.count_parameters() as f64 / 1_000_000.0
    );

    // Example 2: Using ViT presets
    println!("\n2. Using ViT presets:");

    let presets = vec![
        (ViTPreset::Tiny16, "ViT-Tiny/16 (5.7M params)"),
        (ViTPreset::Small16, "ViT-Small/16 (22M params)"),
        (ViTPreset::Base16, "ViT-Base/16 (86M params)"),
        (ViTPreset::Large16, "ViT-Large/16 (307M params)"),
        (ViTPreset::Huge14, "ViT-Huge/14 (632M params)"),
    ];

    for (preset, description) in &presets {
        let config = preset.config(1000)?;
        let vit_preset = VisionTransformer::new(config)?;

        println!("\n   {}", description);
        println!(
            "     - Patches: {}",
            vit_preset.config().patch_embed.num_patches()
        );
        println!(
            "     - Model dim: {}",
            vit_preset.config().patch_embed.d_model
        );
        println!(
            "     - Parameters: {:.2}M",
            vit_preset.count_parameters() as f64 / 1_000_000.0
        );
    }

    // Example 3: Building ViT graph
    println!("\n3. Building Vision Transformer computation graph:");

    let small_vit_config = VisionTransformerConfig::new(
        224,  // image_size
        16,   // patch_size
        3,    // in_channels
        384,  // d_model (smaller for demonstration)
        6,    // n_heads
        1536, // d_ff
        6,    // n_layers (fewer for demonstration)
        10,   // num_classes (e.g., CIFAR-10)
    )?;

    let small_vit = VisionTransformer::new(small_vit_config)?;

    let mut graph = EinsumGraph::new();
    // Add required input tensors
    graph.add_tensor("patches"); // Tensor 0: [batch, num_patches, patch_dim]
    graph.add_tensor("W_patch_embed"); // Tensor 1: [patch_dim, d_model]
    graph.add_tensor("pos_embed"); // Tensor 2: [seq_len, d_model]

    let outputs = small_vit.build_vit_graph(&mut graph)?;

    println!("   Graph built successfully!");
    println!("   Number of nodes: {}", graph.nodes.len());
    println!("   Number of tensors: {}", graph.tensors.len());
    println!("   Output tensors: {}", outputs.len());

    // Example 4: Configuration options
    println!("\n4. Advanced configuration options:");

    let advanced_vit = VisionTransformerConfig::new(224, 16, 3, 768, 12, 3072, 12, 1000)?
        .with_class_token(true) // Use class token for classification
        .with_classifier_dropout(0.1) // Dropout in classification head
        .with_pre_norm(true) // Use pre-normalization
        .with_dropout(0.1); // Dropout in encoder layers

    println!("   Advanced ViT configuration:");
    println!("     - Class token: {}", advanced_vit.use_class_token);
    println!(
        "     - Classifier dropout: {}",
        advanced_vit.classifier_dropout
    );
    println!(
        "     - Pre-normalization: {}",
        advanced_vit.encoder.layer_config.pre_norm
    );
    println!("     - Sequence length: {}", advanced_vit.seq_length());

    // Example 5: Different image sizes
    println!("\n5. ViT with different image resolutions:");

    let resolutions = vec![
        (224, 16, "Standard resolution"),
        (384, 16, "Higher resolution"),
        (512, 32, "Very high resolution"),
    ];

    for (image_size, patch_size, desc) in resolutions {
        let patch_config = PatchEmbeddingConfig::new(image_size, patch_size, 3, 768)?;

        println!("\n   {}:", desc);
        println!("     - Image size: {}x{}", image_size, image_size);
        println!("     - Patch size: {}x{}", patch_size, patch_size);
        println!("     - Number of patches: {}", patch_config.num_patches());
        println!("     - Patches per side: {}", image_size / patch_size);
    }

    // Example 6: Parameter breakdown
    println!("\n6. Parameter breakdown for ViT-Base/16:");

    let base_vit_config = ViTPreset::Base16.config(1000)?;
    let base_vit = VisionTransformer::new(base_vit_config)?;

    let d_model = base_vit.config().patch_embed.d_model;
    let patch_dim = base_vit.config().patch_embed.patch_dim();
    let num_patches = base_vit.config().patch_embed.num_patches();
    let num_classes = base_vit.config().num_classes;

    println!("   - Patch embedding: {} params", patch_dim * d_model);
    println!("   - Class token: {} params", d_model);
    println!(
        "   - Position embeddings: {} params",
        (num_patches + 1) * d_model
    );

    let layer_params = tensorlogic_trustformers::utils::count_encoder_layer_params(
        &base_vit.config().encoder.layer_config,
    );
    let num_layers = base_vit.config().encoder.num_layers;

    println!(
        "   - Encoder ({} layers): {:.2}M params",
        num_layers,
        (layer_params * num_layers) as f64 / 1_000_000.0
    );
    println!(
        "   - Classification head: {} params",
        d_model * num_classes + num_classes
    );
    println!("   ---");
    println!(
        "   Total: {:.2}M parameters",
        base_vit.count_parameters() as f64 / 1_000_000.0
    );

    println!("\n=== Example completed successfully! ===");

    Ok(())
}
