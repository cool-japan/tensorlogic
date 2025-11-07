//! Training with constraint-based loss.
//!
//! This example demonstrates:
//! - Using constraint violation loss
//! - Rule satisfaction monitoring
//! - Multi-objective training concepts
//! - AdamW optimizer with weight decay

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use tensorlogic_train::{
    AdamWOptimizer, BatchConfig, CallbackList, ConstraintViolationLoss, EpochCallback,
    OptimizerConfig, Trainer, TrainerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Logical Loss Training ===\n");

    // Generate synthetic data for a constrained classification problem
    // We want to learn a classifier that respects certain logical rules
    let n_samples = 120;
    let n_classes = 3;
    let n_features = 4;

    let mut train_data = Array2::zeros((n_samples, n_features));
    let mut train_targets = Array2::zeros((n_samples, n_classes));

    // Generate data with built-in logical structure
    for i in 0..n_samples {
        let class = i / 40;
        for j in 0..n_features {
            train_data[[i, j]] =
                (class as f64 * 1.5 + (i % 40) as f64 * 0.1 + j as f64 * 0.2) / (n_features as f64);
        }
        train_targets[[i, class]] = 1.0;
    }

    let val_data = Array2::from_shape_fn((30, n_features), |(i, j)| {
        let class = i / 10;
        (class as f64 * 1.5 + (i % 10) as f64 * 0.12 + j as f64 * 0.22) / (n_features as f64)
    });

    let mut val_targets = Array2::zeros((30, n_classes));
    for i in 0..30 {
        val_targets[[i, i / 10]] = 1.0;
    }

    println!("Dataset:");
    println!("  Train: {} samples", n_samples);
    println!("  Val: 30 samples");
    println!("  Features: {}", n_features);
    println!("  Classes: {}\n", n_classes);

    // For this example, we use a simple constraint violation loss
    // In a real scenario, you would combine multiple losses using LogicalLoss
    let loss = Box::new(ConstraintViolationLoss::default());

    // Create optimizer with weight decay for regularization
    let optimizer = Box::new(AdamWOptimizer::new(OptimizerConfig {
        learning_rate: 0.001,
        weight_decay: 0.01, // L2 regularization
        ..Default::default()
    }));

    println!("Optimizer: AdamW (lr=0.001, weight_decay=0.01)");
    println!("Loss: ConstraintViolation\n");

    // Configure trainer
    let config = TrainerConfig {
        num_epochs: 40,
        batch_config: BatchConfig {
            batch_size: 24,
            shuffle: true,
            ..Default::default()
        },
        validate_every_epoch: false, // Validate every 2 epochs
        ..Default::default()
    };

    let mut trainer = Trainer::new(config, loss, optimizer);

    // Add callbacks
    let mut callbacks = CallbackList::new();
    callbacks.add(Box::new(EpochCallback::new(true)));
    trainer = trainer.with_callbacks(callbacks);

    // Initialize parameters
    let mut parameters = HashMap::new();
    parameters.insert(
        "weights".to_string(),
        Array2::from_shape_fn((n_features, n_classes), |(i, j)| {
            // Small random initialization
            ((i + j) as f64 * 0.01) % 0.1 - 0.05
        }),
    );
    parameters.insert("bias".to_string(), Array2::zeros((1, n_classes)));

    println!("Starting constraint-based training...\n");
    println!("Training objective:");
    println!("  Minimize constraint violations while learning patterns\n");

    let history = trainer.train(
        &train_data.view(),
        &train_targets.view(),
        Some(&val_data.view()),
        Some(&val_targets.view()),
        &mut parameters,
    )?;

    // Results
    println!("\n=== Training Results ===\n");

    let initial_loss = history.train_loss.first().unwrap_or(&0.0);
    let final_loss = history.train_loss.last().unwrap_or(&0.0);

    println!("Loss progression:");
    println!("  Initial: {:.6}", initial_loss);
    println!("  Final: {:.6}", final_loss);
    println!(
        "  Reduction: {:.2}%",
        (1.0 - final_loss / initial_loss) * 100.0
    );

    if let Some((epoch, val_loss)) = history.best_val_loss() {
        println!("\nBest validation:");
        println!("  Epoch: {}", epoch);
        println!("  Loss: {:.6}", val_loss);
    }

    println!("\nFinal parameters:");
    let weights = parameters.get("weights").unwrap();
    println!("  Weights shape: {:?}", weights.shape());
    println!(
        "  Weight magnitude: {:.6}",
        weights.iter().map(|x| x.abs()).sum::<f64>() / (weights.len() as f64)
    );

    println!("\n✅ Constraint-based training completed!");
    println!("\n💡 Note: For complex logical training:");
    println!("   - Use LogicalLoss.compute_total() with rule and constraint arrays");
    println!("   - Define domain-specific logical rules");
    println!("   - Implement custom training loops for multi-objective optimization");
    println!("   - Monitor rule satisfaction and constraint violations separately");

    Ok(())
}
