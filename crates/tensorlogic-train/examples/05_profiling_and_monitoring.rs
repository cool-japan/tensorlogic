//! Advanced monitoring with profiling and histogram tracking.
//!
//! This example demonstrates:
//! - Performance profiling during training
//! - Weight histogram tracking
//! - Gradient monitoring
//! - Comprehensive training diagnostics
//! - Performance optimization insights

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use tensorlogic_train::{
    AdamOptimizer, BatchConfig, CallbackList, EpochCallback, GradientMonitor, HistogramCallback,
    MseLoss, OptimizerConfig, ProfilingCallback, Trainer, TrainerConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Training with Advanced Monitoring ===\n");

    // Generate a larger dataset for meaningful profiling
    let n_train = 1000;
    let n_val = 200;
    let n_features = 20;

    println!("Generating dataset...");
    let train_data = Array2::from_shape_fn((n_train, n_features), |(i, j)| {
        ((i * j) as f64 * 0.001 + (i + j) as f64 * 0.01).sin()
    });

    let train_targets = Array2::from_shape_fn((n_train, 1), |(i, _)| {
        let sum: f64 = (0..n_features)
            .map(|j| train_data[[i, j]] * (j as f64 + 1.0))
            .sum();
        sum / (n_features as f64)
    });

    let val_data = Array2::from_shape_fn((n_val, n_features), |(i, j)| {
        ((i * j) as f64 * 0.001 + (i + j) as f64 * 0.01 + 0.1).sin()
    });

    let val_targets = Array2::from_shape_fn((n_val, 1), |(i, _)| {
        let sum: f64 = (0..n_features)
            .map(|j| val_data[[i, j]] * (j as f64 + 1.0))
            .sum();
        sum / (n_features as f64)
    });

    println!("Dataset:");
    println!("  Train: {} samples, {} features", n_train, n_features);
    println!("  Val: {} samples\n", n_val);

    // Create loss and optimizer
    let loss = Box::new(MseLoss);
    let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig {
        learning_rate: 0.001,
        ..Default::default()
    }));

    // Create trainer with larger dataset
    let config = TrainerConfig {
        num_epochs: 20,
        batch_config: BatchConfig {
            batch_size: 64, // Larger batch size for profiling
            shuffle: true,
            ..Default::default()
        },
        validate_every_epoch: true,
        ..Default::default()
    };

    let mut trainer = Trainer::new(config, loss, optimizer);

    // ========================================
    // ADVANCED MONITORING SETUP
    // ========================================

    let mut callbacks = CallbackList::new();

    // 1. Basic epoch logging
    callbacks.add(Box::new(EpochCallback::new(true)));
    println!("✓ Epoch logging enabled");

    // 2. PROFILING CALLBACK - Track performance
    let profiling = ProfilingCallback::new(
        true, // verbose - print detailed stats
        5,    // log every 5 epochs
    );
    callbacks.add(Box::new(profiling));
    println!("✓ Performance profiling enabled");
    println!("  - Tracking: batch times, epoch times, throughput");
    println!("  - Logging: every 5 epochs");

    // 3. HISTOGRAM CALLBACK - Track weight distributions
    let histogram = HistogramCallback::new(
        10,    // log every 10 epochs
        20,    // 20 bins for histogram
        false, // verbose=false (don't print ASCII histograms in this example)
    );
    callbacks.add(Box::new(histogram));
    println!("✓ Histogram tracking enabled");
    println!("  - Tracking: weight distributions");
    println!("  - Logging: every 10 epochs");
    println!("  - Bins: 20");

    // 4. GRADIENT MONITOR - Detect training issues
    let gradient_monitor = GradientMonitor::new(
        50,   // log every 50 batches
        1e-8, // vanishing gradient threshold
        10.0, // exploding gradient threshold
    );
    callbacks.add(Box::new(gradient_monitor));
    println!("✓ Gradient monitoring enabled");
    println!("  - Vanishing threshold: 1e-8");
    println!("  - Exploding threshold: 10.0");
    println!("  - Logging: every 50 batches\n");

    trainer = trainer.with_callbacks(callbacks);

    // Initialize model parameters
    let mut parameters = HashMap::new();
    parameters.insert(
        "weights".to_string(),
        Array2::from_shape_fn((n_features, 1), |(i, _)| {
            // Xavier/Glorot initialization
            let fan_in = n_features as f64;
            let limit = (6.0_f64 / fan_in).sqrt();
            (i as f64 * 0.01) % (2.0 * limit) - limit
        }),
    );
    parameters.insert("bias".to_string(), Array2::zeros((1, 1)));

    println!("Model:");
    println!("  Input: {}", n_features);
    println!("  Output: 1");
    println!("  Parameters: {}", n_features + 1);
    println!("  Initialization: Xavier/Glorot\n");

    println!("Starting training with comprehensive monitoring...\n");
    println!("{}", "=".repeat(70));

    // Train with monitoring
    let history = trainer.train(
        &train_data.view(),
        &train_targets.view(),
        Some(&val_data.view()),
        Some(&val_targets.view()),
        &mut parameters,
    )?;

    println!("{}", "=".repeat(70));
    println!("\n=== Training Complete ===\n");

    // ========================================
    // PERFORMANCE ANALYSIS
    // ========================================

    println!("Training Summary:");
    println!("  Epochs completed: {}", history.train_loss.len());
    println!(
        "  Initial loss: {:.6}",
        history.train_loss.first().unwrap_or(&0.0)
    );
    println!(
        "  Final loss: {:.6}",
        history.train_loss.last().unwrap_or(&0.0)
    );

    if let Some((epoch, loss)) = history.best_val_loss() {
        println!("  Best val loss: {:.6} at epoch {}", loss, epoch);
    }

    let improvement = if history.train_loss.len() >= 2 {
        let initial = history.train_loss.first().unwrap();
        let final_loss = history.train_loss.last().unwrap();
        (1.0 - final_loss / initial) * 100.0
    } else {
        0.0
    };
    println!("  Loss reduction: {:.2}%", improvement);

    // ========================================
    // MONITORING INSIGHTS
    // ========================================

    println!("\n=== Monitoring Insights ===\n");

    println!("📊 Performance Profiling:");
    println!(
        "  Total batches: {}",
        history.train_loss.len() * (n_train / 64)
    );
    println!("  Avg samples/epoch: {}", n_train);
    println!("  Batch size: 64");
    println!(
        "  Estimated throughput: ~{:.0} samples/sec (if profiled)",
        n_train as f64 / 2.0
    ); // Placeholder

    println!("\n🔍 Gradient Health:");
    println!("  Status: Monitored throughout training");
    println!("  Vanishing detections: Check logs above");
    println!("  Exploding detections: Check logs above");
    println!("  Recommendation: Review gradient norms if issues detected");

    println!("\n📈 Weight Distributions:");
    println!("  Tracked every 10 epochs");
    println!("  Use histogram data to detect:");
    println!("    - Weight saturation (all near 0 or max)");
    println!("    - Imbalanced initialization");
    println!("    - Dead neurons (weights not updating)");

    // ========================================
    // FINAL MODEL ANALYSIS
    // ========================================

    println!("\n=== Final Model State ===\n");

    let weights = parameters.get("weights").unwrap();
    let bias = parameters.get("bias").unwrap();

    println!("Weights:");
    println!("  Shape: {:?}", weights.shape());
    println!(
        "  Mean: {:.6}",
        weights.iter().sum::<f64>() / weights.len() as f64
    );
    println!("  Std: {:.6}", {
        let mean = weights.iter().sum::<f64>() / weights.len() as f64;
        let variance =
            weights.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / weights.len() as f64;
        variance.sqrt()
    });
    println!(
        "  Min: {:.6}",
        weights.iter().copied().fold(f64::INFINITY, f64::min)
    );
    println!(
        "  Max: {:.6}",
        weights.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );

    println!("\nBias:");
    println!("  Value: {:.6}", bias[[0, 0]]);

    // ========================================
    // RECOMMENDATIONS
    // ========================================

    println!("\n=== Optimization Recommendations ===\n");

    let final_loss = history.train_loss.last().unwrap_or(&1.0);

    if *final_loss > 0.1 {
        println!("⚠️  High final loss detected:");
        println!("  - Try increasing learning rate");
        println!("  - Try more epochs");
        println!("  - Consider different optimizer (e.g., AdamW)");
    } else if *final_loss < 0.001 {
        println!("✅ Excellent convergence!");
        println!("  - Model is well-optimized");
        println!("  - Monitor for overfitting if val loss increases");
    }

    if history.train_loss.len() < 10 {
        println!("\n💡 Training stopped early:");
        println!("  - May need more epochs");
        println!("  - Check early stopping criteria");
    }

    println!("\n=== Next Steps ===\n");
    println!("1. Review profiling output for performance bottlenecks");
    println!("2. Check histogram data for weight distribution issues");
    println!("3. Analyze gradient norms for training stability");
    println!("4. Adjust hyperparameters based on monitoring data");
    println!("5. Consider using LearningRateFinder for optimal LR");

    println!("\n✅ Training with advanced monitoring completed!");
    println!("\n💡 Pro tips:");
    println!("  - Use verbose=true on ProfilingCallback for detailed timing");
    println!("  - Histogram tracking helps debug initialization issues");
    println!("  - Gradient monitoring prevents silent training failures");
    println!("  - Combine with TensorBoard/W&B for visualization (future)");

    Ok(())
}
