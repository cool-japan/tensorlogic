//! Gradient Centralization for Improved Training
//!
//! This example demonstrates how Gradient Centralization (GC) improves training
//! by normalizing gradients before applying optimizer updates.
//!
//! GC has been shown to:
//! - Improve generalization and test accuracy
//! - Accelerate convergence
//! - Stabilize gradient flow
//! - Work seamlessly with any optimizer
//!
//! Reference: Yong et al., "Gradient Centralization: A New Optimization Technique
//! for Deep Neural Networks", ECCV 2020
//!
//! Run with: cargo run --example 22_gradient_centralization

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use tensorlogic_train::*;

/// Simulate a simple training step and return loss.
fn compute_loss(params: &HashMap<String, Array2<f64>>, x: &Array2<f64>, y: &Array2<f64>) -> f64 {
    // Simple linear model: y_pred = W * x
    let w = &params["W"];
    let y_pred = w.dot(x);

    // MSE loss
    let diff = &y_pred - y;
    diff.iter().map(|&d| d * d).sum::<f64>() / diff.len() as f64
}

/// Compute gradients (simplified - using finite differences for demonstration).
fn compute_gradients(
    params: &HashMap<String, Array2<f64>>,
    x: &Array2<f64>,
    y: &Array2<f64>,
) -> HashMap<String, Array2<f64>> {
    let mut grads = HashMap::new();

    // Simulate gradients (in practice, would use autodiff)
    // For demonstration, use simple pattern that benefits from centralization
    let w = &params["W"];
    let y_pred = w.dot(x);
    let diff = &y_pred - y;

    // Gradient: 2 * (y_pred - y) * x^T / n
    let n = diff.len() as f64;
    let grad = (2.0 / n) * diff.dot(&x.t());

    grads.insert("W".to_string(), grad);
    grads
}

/// Train a model with or without gradient centralization.
fn train_model(use_gc: bool, gc_strategy: GcStrategy, n_epochs: usize) -> Vec<f64> {
    // Initialize parameters
    let mut params = HashMap::new();
    params.insert("W".to_string(), Array2::from_elem((3, 5), 0.5));

    // Create training data (simple linear relationship with noise)
    let x = Array2::from_shape_fn((5, 10), |(i, j)| (i as f64 * 0.1 + j as f64 * 0.2) / 5.0);

    let y_true = Array2::from_shape_fn((3, 10), |(i, j)| (i as f64 * 0.3 + j as f64 * 0.15) / 3.0);

    // Create optimizer
    let config = OptimizerConfig {
        learning_rate: 0.1,
        ..Default::default()
    };
    let adam = AdamOptimizer::new(config);

    // Wrap with GC if requested
    let mut optimizer: Box<dyn Optimizer> = if use_gc {
        let gc_config = GcConfig::new(gc_strategy);
        Box::new(GradientCentralization::new(Box::new(adam), gc_config))
    } else {
        Box::new(adam)
    };

    // Training loop
    let mut loss_history = Vec::new();

    for _epoch in 0..n_epochs {
        let grads = compute_gradients(&params, &x, &y_true);
        optimizer.step(&mut params, &grads).unwrap();

        let loss = compute_loss(&params, &x, &y_true);
        loss_history.push(loss);
    }

    loss_history
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        Gradient Centralization for Improved Training       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. Baseline: Training without GC
    // ============================================================================

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Baseline: Training WITHOUT Gradient Centralization      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let n_epochs = 50;
    let baseline_losses = train_model(false, GcStrategy::LayerWise, n_epochs);

    println!("Training progress (baseline):");
    for (epoch, &loss) in baseline_losses.iter().enumerate() {
        if epoch % 10 == 0 || epoch == n_epochs - 1 {
            println!("  Epoch {:3}: Loss = {:.6}", epoch, loss);
        }
    }
    println!();

    // ============================================================================
    // 2. Layer-wise Gradient Centralization
    // ============================================================================

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Training WITH Layer-wise Gradient Centralization       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let layerwise_losses = train_model(true, GcStrategy::LayerWise, n_epochs);

    println!("Training progress (layer-wise GC):");
    for (epoch, &loss) in layerwise_losses.iter().enumerate() {
        if epoch % 10 == 0 || epoch == n_epochs - 1 {
            println!("  Epoch {:3}: Loss = {:.6}", epoch, loss);
        }
    }
    println!();

    // ============================================================================
    // 3. Per-Row Gradient Centralization
    // ============================================================================

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Training WITH Per-Row Gradient Centralization          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let perrow_losses = train_model(true, GcStrategy::PerRow, n_epochs);

    println!("Training progress (per-row GC):");
    for (epoch, &loss) in perrow_losses.iter().enumerate() {
        if epoch % 10 == 0 || epoch == n_epochs - 1 {
            println!("  Epoch {:3}: Loss = {:.6}", epoch, loss);
        }
    }
    println!();

    // ============================================================================
    // 4. Per-Column Gradient Centralization
    // ============================================================================

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Training WITH Per-Column Gradient Centralization       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let percol_losses = train_model(true, GcStrategy::PerColumn, n_epochs);

    println!("Training progress (per-column GC):");
    for (epoch, &loss) in percol_losses.iter().enumerate() {
        if epoch % 10 == 0 || epoch == n_epochs - 1 {
            println!("  Epoch {:3}: Loss = {:.6}", epoch, loss);
        }
    }
    println!();

    // ============================================================================
    // 5. Global Gradient Centralization
    // ============================================================================

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Training WITH Global Gradient Centralization           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let global_losses = train_model(true, GcStrategy::Global, n_epochs);

    println!("Training progress (global GC):");
    for (epoch, &loss) in global_losses.iter().enumerate() {
        if epoch % 10 == 0 || epoch == n_epochs - 1 {
            println!("  Epoch {:3}: Loss = {:.6}", epoch, loss);
        }
    }
    println!();

    // ============================================================================
    // 6. Comparison
    // ============================================================================

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║     Comparison: Final Loss After {} Epochs                 ║",
        n_epochs
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let baseline_final = baseline_losses.last().unwrap();
    let layerwise_final = layerwise_losses.last().unwrap();
    let perrow_final = perrow_losses.last().unwrap();
    let percol_final = percol_losses.last().unwrap();
    let global_final = global_losses.last().unwrap();

    println!("Final Loss:");
    println!("  • Baseline (no GC):     {:.6}", baseline_final);
    println!("  • Layer-wise GC:        {:.6}", layerwise_final);
    println!("  • Per-row GC:           {:.6}", perrow_final);
    println!("  • Per-column GC:        {:.6}", percol_final);
    println!("  • Global GC:            {:.6}", global_final);
    println!();

    // Find best strategy
    let strategies = [
        ("Baseline (no GC)", *baseline_final),
        ("Layer-wise GC", *layerwise_final),
        ("Per-row GC", *perrow_final),
        ("Per-column GC", *percol_final),
        ("Global GC", *global_final),
    ];

    let best = strategies
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("Best strategy: {} (loss = {:.6})", best.0, best.1);

    // Compute improvements
    if baseline_final > layerwise_final {
        let improvement = (baseline_final - layerwise_final) / baseline_final * 100.0;
        println!(
            "\nLayer-wise GC improved convergence by {:.2}% over baseline!",
            improvement
        );
    }

    // ============================================================================
    // 7. Statistics and Monitoring
    // ============================================================================

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     Gradient Centralization Statistics                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Demonstrate statistics tracking
    let mut params = HashMap::new();
    params.insert("W".to_string(), Array2::from_elem((3, 5), 0.5));

    let config = OptimizerConfig {
        learning_rate: 0.1,
        ..Default::default()
    };
    let adam = AdamOptimizer::new(config);

    let gc_config = GcConfig::new(GcStrategy::LayerWise);
    let mut gc_optimizer = GradientCentralization::new(Box::new(adam), gc_config);

    // Run a few steps and track statistics
    let x = Array2::from_elem((5, 10), 0.5);
    let y = Array2::from_elem((3, 10), 0.3);

    for i in 0..5 {
        let grads = compute_gradients(&params, &x, &y);
        gc_optimizer.step(&mut params, &grads).unwrap();

        let stats = gc_optimizer.stats();
        println!("Step {}: ", i);
        println!("  • Parameters centralized: {}", stats.num_centralized);
        println!("  • Parameters skipped:     {}", stats.num_skipped);
        println!(
            "  • Avg grad norm (before): {:.6}",
            stats.avg_grad_norm_before
        );
        println!(
            "  • Avg grad norm (after):  {:.6}",
            stats.avg_grad_norm_after
        );
        println!("  • Total operations:       {}", stats.total_operations);
    }

    // ============================================================================
    // 8. Dynamic Configuration
    // ============================================================================

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     Dynamic Configuration                                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Gradient Centralization can be toggled dynamically:");
    println!();

    let config = OptimizerConfig {
        learning_rate: 0.1,
        ..Default::default()
    };
    let adam = AdamOptimizer::new(config);
    let gc_config = GcConfig::new(GcStrategy::LayerWise);
    let mut gc_optimizer = GradientCentralization::new(Box::new(adam), gc_config);

    println!(
        "Initial state: GC enabled = {}",
        gc_optimizer.config().enabled
    );

    gc_optimizer.config_mut().disable();
    println!(
        "After disable: GC enabled = {}",
        gc_optimizer.config().enabled
    );

    gc_optimizer.config_mut().enable();
    println!(
        "After enable:  GC enabled = {}",
        gc_optimizer.config().enabled
    );

    println!("\n✅ Gradient Centralization demonstration complete!");
    println!("\nKey takeaways:");
    println!("  1. GC normalizes gradients by subtracting their mean");
    println!("  2. Improves training stability and convergence");
    println!("  3. Works as a drop-in wrapper for any optimizer");
    println!("  4. Multiple strategies available (layer-wise, per-row, per-column, global)");
    println!("  5. Can be enabled/disabled dynamically during training");
    println!("  6. Minimal computational overhead");
}
