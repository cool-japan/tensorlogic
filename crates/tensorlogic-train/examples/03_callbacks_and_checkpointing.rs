//! Advanced training with callbacks and checkpointing.
//!
//! This example demonstrates:
//! - Early stopping to prevent overfitting
//! - Model checkpointing (save best models)
//! - Learning rate scheduling (CosineAnnealing)
//! - Learning rate finder for hyperparameter tuning
//! - Gradient monitoring
//! - Training state persistence

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use tensorlogic_train::{
    AdamOptimizer, BatchConfig, CallbackList, CheckpointCallback, CosineAnnealingLrScheduler,
    EarlyStoppingCallback, EpochCallback, GradientMonitor, MseLoss, OptimizerConfig,
    ReduceLrOnPlateauCallback, Trainer, TrainerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Training with Callbacks ===\n");

    // Generate synthetic data
    let train_data =
        Array2::from_shape_fn((200, 5), |(i, j)| (i as f64 * 0.05 + j as f64 * 0.1) / 5.0);

    let train_targets = Array2::from_shape_fn((200, 1), |(i, _)| {
        let sum: f64 = (0..5).map(|j| train_data[[i, j]]).sum();
        sum * 2.0 + 0.5 + (i as f64 * 0.01).sin()
    });

    let val_data =
        Array2::from_shape_fn((40, 5), |(i, j)| (i as f64 * 0.06 + j as f64 * 0.11) / 5.0);

    let val_targets = Array2::from_shape_fn((40, 1), |(i, _)| {
        let sum: f64 = (0..5).map(|j| val_data[[i, j]]).sum();
        sum * 2.0 + 0.5 + (i as f64 * 0.01).sin()
    });

    println!("Dataset: 200 train, 40 val samples with 5 features\n");

    // Create loss and optimizer
    let loss = Box::new(MseLoss);
    let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig {
        learning_rate: 0.001,
        ..Default::default()
    }));

    // Create learning rate scheduler
    let scheduler = Box::new(CosineAnnealingLrScheduler::new(
        0.001,  // initial_lr
        0.0001, // min_lr
        50,     // t_max (total epochs)
    ));

    println!("Configuration:");
    println!("  Optimizer: Adam (lr=0.001)");
    println!("  Scheduler: CosineAnnealing (min_lr=0.0001, t_max=50)");
    println!("  Loss: MSE\n");

    // Create trainer
    let config = TrainerConfig {
        num_epochs: 50,
        batch_config: BatchConfig {
            batch_size: 32,
            shuffle: true,
            ..Default::default()
        },
        validate_every_epoch: false, // Will validate every 2 epochs via callback
        ..Default::default()
    };

    let mut trainer = Trainer::new(config, loss, optimizer);
    trainer = trainer.with_scheduler(scheduler);

    // Configure callbacks
    let mut callbacks = CallbackList::new();

    // 1. Epoch logging
    callbacks.add(Box::new(EpochCallback::new(true)));

    // 2. Early stopping (stop if no improvement for 10 epochs)
    callbacks.add(Box::new(EarlyStoppingCallback::new(
        10,    // patience
        0.001, // min_delta
    )));
    println!("✓ Early stopping: patience=10, min_delta=0.001");

    // 3. Model checkpointing (save best model)
    let checkpoint_dir = std::env::temp_dir().join("tensorlogic_checkpoints");
    std::fs::create_dir_all(&checkpoint_dir)?;
    callbacks.add(Box::new(CheckpointCallback::new(
        checkpoint_dir.clone(),
        2,    // save every 2 epochs
        true, // save_best_only
    )));
    println!("✓ Checkpointing: {:?} (best only)", checkpoint_dir);

    // 4. Reduce LR on plateau
    callbacks.add(Box::new(ReduceLrOnPlateauCallback::new(
        0.5,    // factor (reduce by 50%)
        5,      // patience
        0.01,   // min_delta
        0.0001, // min_lr
    )));
    println!("✓ ReduceLROnPlateau: factor=0.5, patience=5, min_lr=0.0001");

    // 5. Gradient monitoring
    callbacks.add(Box::new(GradientMonitor::new(
        10,    // log every 10 batches
        1e-7,  // vanishing threshold
        100.0, // exploding threshold
    )));
    println!("✓ Gradient monitor: log_freq=10, thresholds=[1e-7, 100.0]");

    // 6. Learning rate finder (optional - run before main training)
    // Uncomment to find optimal learning rate:
    /*
    println!("\n--- Running LR Finder ---");
    callbacks.add(Box::new(LearningRateFinder::new(
        1e-7,  // start_lr
        1.0,   // end_lr
        100,   // num_steps
    ).with_exponential_scaling()));
    println!("✓ LR Finder: range [1e-7, 1.0] over 100 steps");
    */

    trainer = trainer.with_callbacks(callbacks);

    // Initialize parameters
    let mut parameters = HashMap::new();
    parameters.insert("weights".to_string(), Array2::zeros((5, 1)));
    parameters.insert("bias".to_string(), Array2::zeros((1, 1)));

    println!("\nStarting training with advanced callbacks...\n");

    // Train
    let history = trainer.train(
        &train_data.view(),
        &train_targets.view(),
        Some(&val_data.view()),
        Some(&val_targets.view()),
        &mut parameters,
    )?;

    // Results
    println!("\n=== Training Results ===\n");
    println!("Epochs completed: {}", history.train_loss.len());

    if let Some((epoch, loss)) = history.best_val_loss() {
        println!("Best validation loss: {:.6} at epoch {}", loss, epoch);
    }

    println!(
        "Final train loss: {:.6}",
        history.train_loss.last().unwrap_or(&0.0)
    );
    println!(
        "Improvement: {:.2}%",
        (1.0 - history.train_loss.last().unwrap_or(&1.0)
            / history.train_loss.first().unwrap_or(&1.0))
            * 100.0
    );

    // Check if early stopped
    if history.train_loss.len() < 50 {
        println!(
            "\n⚠️  Training stopped early at epoch {} (early stopping triggered)",
            history.train_loss.len()
        );
    }

    // List saved checkpoints
    println!("\nSaved checkpoints:");
    if let Ok(entries) = std::fs::read_dir(&checkpoint_dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "json") {
                println!("  - {:?}", entry.file_name());
            }
        }
    }

    println!("\n✅ Advanced training completed!");
    println!("\n💡 Tip: Inspect checkpoints at {:?}", checkpoint_dir);

    Ok(())
}
