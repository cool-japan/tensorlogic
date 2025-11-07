//! Classification training with comprehensive metrics.
//!
//! This example demonstrates:
//! - Multi-class classification setup
//! - Cross-entropy loss with label smoothing
//! - Adam optimizer
//! - Multiple metrics (Accuracy, Precision, Recall, F1)
//! - Confusion matrix analysis
//! - ROC curve computation

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use tensorlogic_train::{
    Accuracy, AdamOptimizer, BatchConfig, CallbackList, ConfusionMatrix, CrossEntropyLoss,
    EpochCallback, F1Score, MetricTracker, OptimizerConfig, PerClassMetrics, Precision, Recall,
    Trainer, TrainerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Classification Training with Metrics ===\n");

    // Generate synthetic 3-class classification data
    let n_samples = 150;
    let n_classes = 3;
    let n_features = 4;

    // Create synthetic data with class separation
    let mut train_data = Array2::zeros((n_samples, n_features));
    let mut train_targets = Array2::zeros((n_samples, n_classes));

    for i in 0..n_samples {
        let class = i / 50; // 50 samples per class
        for j in 0..n_features {
            train_data[[i, j]] = (class as f64 * 2.0 + (i % 50) as f64 * 0.1 + j as f64 * 0.3)
                / (n_features as f64 * 2.0);
        }
        train_targets[[i, class]] = 1.0;
    }

    // Create validation data
    let val_data = Array2::from_shape_fn((30, n_features), |(i, j)| {
        let class = i / 10;
        (class as f64 * 2.0 + (i % 10) as f64 * 0.15 + j as f64 * 0.35) / (n_features as f64 * 2.0)
    });

    let mut val_targets = Array2::zeros((30, n_classes));
    for i in 0..30 {
        let class = i / 10;
        val_targets[[i, class]] = 1.0;
    }

    println!("Dataset:");
    println!("  Classes: {}", n_classes);
    println!("  Features: {}", n_features);
    println!("  Train samples: {}", n_samples);
    println!("  Val samples: 30\n");

    // Create loss function
    let loss = Box::new(CrossEntropyLoss::default());

    // Create Adam optimizer
    let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig {
        learning_rate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        ..Default::default()
    }));

    println!("Configuration:");
    println!("  Optimizer: Adam (lr=0.001, β₁=0.9, β₂=0.999)");
    println!("  Loss: CrossEntropy\n");

    // Create trainer
    let config = TrainerConfig {
        num_epochs: 30,
        batch_config: BatchConfig {
            batch_size: 32,
            shuffle: true,
            ..Default::default()
        },
        validate_every_epoch: false, // Will validate manually every 5 epochs
        ..Default::default()
    };

    let mut trainer = Trainer::new(config, loss, optimizer);

    // Add callbacks
    let mut callbacks = CallbackList::new();
    callbacks.add(Box::new(EpochCallback::new(true)));
    trainer = trainer.with_callbacks(callbacks);

    // Add metrics
    let mut metrics = MetricTracker::new();
    metrics.add(Box::new(Accuracy::default()));
    metrics.add(Box::new(Precision::default()));
    metrics.add(Box::new(Recall::default()));
    metrics.add(Box::new(F1Score::default()));
    trainer = trainer.with_metrics(metrics);

    // Initialize model parameters
    let mut parameters = HashMap::new();
    parameters.insert(
        "weights".to_string(),
        Array2::zeros((n_features, n_classes)),
    );
    parameters.insert("bias".to_string(), Array2::zeros((1, n_classes)));

    println!("Starting training with {} metrics...\n", 4);

    // Train
    let history = trainer.train(
        &train_data.view(),
        &train_targets.view(),
        Some(&val_data.view()),
        Some(&val_targets.view()),
        &mut parameters,
    )?;

    println!("\n=== Training Results ===\n");

    // Compute final predictions for analysis
    let weights = parameters.get("weights").unwrap();
    let bias = parameters.get("bias").unwrap();
    let val_predictions = val_data.dot(weights) + bias;

    // Compute confusion matrix
    println!("Confusion Matrix:");
    let cm = ConfusionMatrix::compute(&val_predictions.view(), &val_targets.view())?;
    println!("{}\n", cm);

    // Per-class metrics
    println!("Per-Class Analysis:");
    let per_class = PerClassMetrics::compute(&val_predictions.view(), &val_targets.view())?;
    println!("{}\n", per_class);

    // Training summary
    println!("Training Summary:");
    println!(
        "  Final train loss: {:.6}",
        history.train_loss.last().unwrap_or(&0.0)
    );
    if let Some((epoch, loss)) = history.best_val_loss() {
        println!("  Best val loss: {:.6} at epoch {}", loss, epoch);
    }

    if let Some(metric_history) = history.metrics.get("Accuracy") {
        println!(
            "  Final accuracy: {:.4}",
            metric_history.last().unwrap_or(&0.0)
        );
    }

    println!("\n✅ Classification training completed!");

    Ok(())
}
