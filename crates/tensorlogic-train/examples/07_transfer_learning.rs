//! Example: Transfer Learning
//!
//! This example demonstrates transfer learning techniques:
//! - Layer freezing and unfreezing
//! - Progressive unfreezing
//! - Discriminative fine-tuning
//! - Feature extraction mode
//!
//! Run with: cargo run --example 07_transfer_learning

use tensorlogic_train::{
    DiscriminativeFineTuning, FeatureExtractorMode, LayerFreezingConfig, ProgressiveUnfreezing,
    TransferLearningManager,
};

fn main() {
    println!("=== Transfer Learning Examples ===\n");

    // Example 1: Basic Layer Freezing
    println!("1. Basic Layer Freezing");
    println!("   Freeze specific layers to preserve pretrained features\n");

    let mut config = LayerFreezingConfig::new();

    // Freeze encoder layers
    config.freeze_layers(&["encoder.layer1", "encoder.layer2", "encoder.layer3"]);

    println!("   Frozen layers:");
    for layer in config.frozen_layers() {
        println!("     - {}", layer);
    }

    println!(
        "   encoder.layer1 frozen? {}",
        config.is_frozen("encoder.layer1")
    );
    println!(
        "   classifier.fc frozen? {}",
        config.is_frozen("classifier.fc")
    );
    println!("   Number of frozen layers: {}\n", config.num_frozen());

    // Unfreeze for fine-tuning
    config.unfreeze_layers(&["encoder.layer3"]);
    println!("   After unfreezing layer3:");
    println!(
        "   encoder.layer3 frozen? {}\n",
        config.is_frozen("encoder.layer3")
    );

    // Example 2: Progressive Unfreezing
    println!("2. Progressive Unfreezing");
    println!("   Gradually unfreeze layers from top to bottom\n");

    let layer_order = vec![
        "encoder.layer1".to_string(),
        "encoder.layer2".to_string(),
        "encoder.layer3".to_string(),
        "classifier.fc".to_string(),
    ];

    let mut unfreezing = ProgressiveUnfreezing::new(layer_order, 5).unwrap();

    println!("   Training schedule (unfreeze every 5 epochs):");
    println!("   Stage 0 (epochs 0-4): All frozen");
    println!("   Stage 1 (epochs 5-9): Unfreeze classifier.fc");
    println!("   Stage 2 (epochs 10-14): Unfreeze encoder.layer3 + classifier.fc");
    println!("   Stage 3 (epochs 15-19): Unfreeze encoder.layer2-3 + classifier.fc");
    println!("   Stage 4 (epochs 20+): All layers trainable\n");

    // Simulate epoch progression
    for epoch in [0, 5, 10, 15, 20] {
        unfreezing.update_stage(epoch);
        let trainable = unfreezing.get_trainable_layers();
        let frozen = unfreezing.get_frozen_layers();

        println!("   Epoch {}:", epoch);
        println!("     Trainable layers: {:?}", trainable);
        println!("     Frozen layers: {} layers\n", frozen.len());
    }

    // Example 3: Discriminative Fine-Tuning
    println!("3. Discriminative Fine-Tuning");
    println!("   Use different learning rates for different layers\n");

    let mut finetuning = DiscriminativeFineTuning::new(1e-3, 0.5).unwrap();

    let layers = vec![
        "encoder.layer1".to_string(),
        "encoder.layer2".to_string(),
        "encoder.layer3".to_string(),
        "classifier.fc".to_string(),
    ];

    finetuning.compute_layer_lrs(&layers);

    println!("   Layer-specific learning rates (base_lr = 1e-3, decay = 0.5):");
    for layer in &layers {
        let lr = finetuning.get_layer_lr(layer);
        println!("     {}: {:.6}", layer, lr);
    }

    println!("\n   Rationale:");
    println!("   - Earlier layers (closer to input) use smaller LR");
    println!("   - Later layers (task-specific) use larger LR");
    println!("   - Prevents catastrophic forgetting of low-level features\n");

    // Example 4: Feature Extraction Mode
    println!("4. Feature Extraction Mode");
    println!("   Freeze entire feature extractor, train only the head\n");

    let mode = FeatureExtractorMode::new("encoder".to_string(), "classifier".to_string());

    let all_layers = vec![
        "encoder.layer1".to_string(),
        "encoder.layer2".to_string(),
        "encoder.layer3".to_string(),
        "classifier.fc".to_string(),
        "classifier.output".to_string(),
    ];

    let _extraction_config = mode.get_freezing_config(&all_layers);

    println!("   Feature extractor (frozen):");
    for layer in &all_layers {
        if mode.is_feature_extractor(layer) {
            println!("     - {}", layer);
        }
    }

    println!("\n   Classification head (trainable):");
    for layer in &all_layers {
        if mode.is_head(layer) {
            println!("     - {}", layer);
        }
    }

    println!("\n   Use case: Quick adaptation to new task with limited data\n");

    // Example 5: Transfer Learning Manager (Unified Interface)
    println!("5. Transfer Learning Manager");
    println!("   Unified management of all transfer learning strategies\n");

    // Strategy 1: Feature extraction
    let mode = FeatureExtractorMode::new("encoder".to_string(), "classifier".to_string());
    let mut manager = TransferLearningManager::new().with_feature_extraction(mode, &all_layers);

    println!("   Phase 1: Feature Extraction (epochs 0-10)");
    for epoch in 0..3 {
        manager.on_epoch_begin(epoch);
        println!(
            "     Epoch {}: encoder trainable? {}",
            epoch,
            manager.should_update_layer("encoder.layer1")
        );
    }

    // Strategy 2: Full fine-tuning with discriminative LR
    let mut finetuning = DiscriminativeFineTuning::new(1e-4, 0.5).unwrap();
    finetuning.compute_layer_lrs(&layers);

    let manager = TransferLearningManager::new().with_discriminative_finetuning(finetuning);

    println!("\n   Phase 2: Fine-Tuning with Discriminative LR (epochs 10+)");
    println!("     Layer LRs:");
    for layer in &layers {
        let lr = manager.get_layer_lr(layer, 1e-4);
        println!("       {}: {:.6}", layer, lr);
    }

    // Strategy 3: Progressive unfreezing
    let unfreezing = ProgressiveUnfreezing::new(layers.clone(), 5).unwrap();
    let mut manager = TransferLearningManager::new().with_progressive_unfreezing(unfreezing);

    println!("\n   Phase 3: Progressive Unfreezing");
    for epoch in [0, 5, 10, 15] {
        manager.on_epoch_begin(epoch);
        println!(
            "     Epoch {}: layer3 trainable? {}",
            epoch,
            manager.should_update_layer("encoder.layer3")
        );
    }

    println!("\n=== Training Workflow Example ===\n");
    println!("Typical 3-phase transfer learning workflow:");
    println!("```rust");
    println!("// Phase 1: Feature Extraction (10 epochs)");
    println!("let mode = FeatureExtractorMode::new(\"encoder\", \"classifier\");");
    println!("let manager = TransferLearningManager::new()");
    println!("    .with_feature_extraction(mode, &all_layers);");
    println!();
    println!("for epoch in 0..10 {{");
    println!("    manager.on_epoch_begin(epoch);");
    println!("    // Train only classifier layers");
    println!("}}");
    println!();
    println!("// Phase 2: Discriminative Fine-Tuning (20 epochs)");
    println!("let mut finetuning = DiscriminativeFineTuning::new(1e-4, 0.5)?;");
    println!("finetuning.compute_layer_lrs(&layers);");
    println!("let manager = manager.with_discriminative_finetuning(finetuning);");
    println!();
    println!("for epoch in 10..30 {{");
    println!("    manager.on_epoch_begin(epoch);");
    println!("    // Train all layers with different LRs");
    println!("    for layer in &layers {{");
    println!("        let lr = manager.get_layer_lr(layer, base_lr);");
    println!("        // Apply layer-specific learning rate");
    println!("    }}");
    println!("}}");
    println!();
    println!("// Phase 3: Progressive Unfreezing (optional)");
    println!("let unfreezing = ProgressiveUnfreezing::new(layers, 5)?;");
    println!("let mut manager = manager.with_progressive_unfreezing(unfreezing);");
    println!();
    println!("for epoch in 30..50 {{");
    println!("    manager.on_epoch_begin(epoch);");
    println!("    // Gradually unfreeze more layers");
    println!("}}");
    println!("```");

    println!("\n=== Best Practices ===");
    println!("1. Start with feature extraction for fast initial adaptation");
    println!("2. Use discriminative LR: lower for early layers, higher for late layers");
    println!("3. Progressive unfreezing helps prevent catastrophic forgetting");
    println!("4. Monitor validation performance to decide when to unfreeze layers");
    println!("5. Use smaller learning rates (1e-4 to 1e-5) for fine-tuning pretrained models");
}
