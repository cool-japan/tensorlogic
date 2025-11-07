//! Basic training example with SGD optimizer and MSE loss.
//!
//! This example demonstrates:
//! - Creating a simple training loop
//! - Using MSE loss for regression
//! - SGD optimizer with momentum
//! - Epoch-level callbacks
//! - Training history tracking

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use tensorlogic_train::{
    CallbackList, EpochCallback, MseLoss, OptimizerConfig, SgdOptimizer, Trainer, TrainerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Training Example ===\n");

    // Generate synthetic regression data
    // y = 2*x1 + 3*x2 + noise
    let train_data =
        Array2::from_shape_fn((100, 2), |(i, j)| (i as f64 * 0.1 + j as f64 * 0.2) / 10.0);

    let train_targets = Array2::from_shape_fn((100, 1), |(i, _)| {
        let x1 = (i as f64 * 0.1) / 10.0;
        let x2 = (i as f64 * 0.2 + 0.2) / 10.0;
        2.0 * x1 + 3.0 * x2 + 0.01 * (i as f64).sin()
    });

    // Create validation data
    let val_data =
        Array2::from_shape_fn((20, 2), |(i, j)| (i as f64 * 0.15 + j as f64 * 0.25) / 10.0);

    let val_targets = Array2::from_shape_fn((20, 1), |(i, _)| {
        let x1 = (i as f64 * 0.15) / 10.0;
        let x2 = (i as f64 * 0.25 + 0.25) / 10.0;
        2.0 * x1 + 3.0 * x2 + 0.01 * (i as f64).sin()
    });

    println!("Dataset shapes:");
    println!(
        "  Train: {:?}, {:?}",
        train_data.shape(),
        train_targets.shape()
    );
    println!(
        "  Val:   {:?}, {:?}\n",
        val_data.shape(),
        val_targets.shape()
    );

    // Create loss function
    let loss = Box::new(MseLoss);

    // Create optimizer
    let optimizer_config = OptimizerConfig {
        learning_rate: 0.01,
        momentum: 0.9,
        ..Default::default()
    };
    let optimizer = Box::new(SgdOptimizer::new(optimizer_config));

    println!("Optimizer: SGD with momentum=0.9, lr=0.01");
    println!("Loss: MSE\n");

    // Create trainer
    use tensorlogic_train::BatchConfig;
    let config = TrainerConfig {
        num_epochs: 20,
        batch_config: BatchConfig {
            batch_size: 16,
            shuffle: true,
            ..Default::default()
        },
        validate_every_epoch: true,
        ..Default::default()
    };

    let mut trainer = Trainer::new(config, loss, optimizer);

    // Add callbacks
    let mut callbacks = CallbackList::new();
    callbacks.add(Box::new(EpochCallback::new(true)));
    trainer = trainer.with_callbacks(callbacks);

    // Initialize model parameters (simple linear model: y = Wx + b)
    let mut parameters = HashMap::new();
    parameters.insert(
        "weights".to_string(),
        Array2::from_elem((2, 1), 0.1), // Small random initialization
    );
    parameters.insert("bias".to_string(), Array2::zeros((1, 1)));

    println!("Initial parameters:");
    println!("  weights: {:?}", parameters.get("weights").unwrap());
    println!("  bias: {:?}\n", parameters.get("bias").unwrap());

    // Train the model
    println!("Starting training...\n");
    let history = trainer.train(
        &train_data.view(),
        &train_targets.view(),
        Some(&val_data.view()),
        Some(&val_targets.view()),
        &mut parameters,
    )?;

    // Print final results
    println!("\n=== Training Complete ===\n");
    println!("Final parameters:");
    println!("  weights: {:?}", parameters.get("weights").unwrap());
    println!("  bias: {:?}\n", parameters.get("bias").unwrap());

    println!("Training history:");
    println!("  Epochs: {}", history.train_loss.len());
    println!(
        "  Initial train loss: {:.6}",
        history.train_loss.first().unwrap_or(&0.0)
    );
    println!(
        "  Final train loss: {:.6}",
        history.train_loss.last().unwrap_or(&0.0)
    );

    if let Some((epoch, loss)) = history.best_val_loss() {
        println!("  Best validation loss: {:.6} at epoch {}", loss, epoch);
    }

    println!("\n✅ Training completed successfully!");

    Ok(())
}
