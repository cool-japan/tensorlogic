//! Example demonstrating the Sophia optimizer
//!
//! Sophia is a state-of-the-art second-order optimizer from 2023 that achieves
//! 2-3x faster convergence than Adam for language model pretraining while using
//! similar memory.
//!
//! Key features:
//! - Uses Hessian diagonal estimates for better curvature awareness
//! - Memory efficient (same as Adam)
//! - Two variants: Gauss-Newton-Bartlett (default) and Hutchinson
//! - Configurable Hessian update frequency
//!
//! Reference:
//! "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
//! Hong Liu et al., 2023
//! https://arxiv.org/abs/2305.14342

use scirs2_core::ndarray::{array, Array2};
use std::collections::HashMap;
use tensorlogic_train::{
    AdamOptimizer, GradClipMode, Optimizer, OptimizerConfig, SophiaConfig, SophiaOptimizer,
    SophiaVariant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sophia Optimizer Example ===\n");

    // Create a simple regression problem: y = 2x + 1
    let x_train = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let y_train = array![[1.0], [3.0], [5.0], [7.0], [9.0], [11.0]];

    println!("Training data:");
    println!("X shape: {:?}", x_train.shape());
    println!("Y shape: {:?}\n", y_train.shape());

    // Example 1: Sophia with default configuration
    println!("Example 1: Sophia with Default Configuration");
    println!("---------------------------------------------");

    let config = OptimizerConfig {
        learning_rate: 0.01,
        ..Default::default()
    };

    let mut sophia_optimizer = SophiaOptimizer::new(config);
    let mut params = HashMap::new();
    params.insert("weight".to_string(), array![[0.0]]);
    params.insert("bias".to_string(), array![[0.0]]);

    train_and_evaluate(
        "Sophia (default)",
        &mut sophia_optimizer,
        &mut params,
        &x_train,
        &y_train,
        50,
    );

    // Example 2: Sophia with custom configuration
    println!("\nExample 2: Sophia with Custom Configuration");
    println!("-------------------------------------------");

    let sophia_config = SophiaConfig {
        base: OptimizerConfig {
            learning_rate: 0.02,
            beta1: 0.965, // Sophia's recommended beta1
            beta2: 0.99,  // Sophia's recommended beta2
            ..Default::default()
        },
        rho: 0.04,               // Clip parameter for update direction
        hessian_update_freq: 10, // Update Hessian every 10 steps
        variant: SophiaVariant::GaussNewtonBartlett,
    };

    let mut sophia_custom = SophiaOptimizer::with_sophia_config(sophia_config);
    let mut params2 = HashMap::new();
    params2.insert("weight".to_string(), array![[0.0]]);
    params2.insert("bias".to_string(), array![[0.0]]);

    train_and_evaluate(
        "Sophia (custom)",
        &mut sophia_custom,
        &mut params2,
        &x_train,
        &y_train,
        50,
    );

    // Example 3: Sophia vs Adam comparison
    println!("\nExample 3: Sophia vs Adam Comparison");
    println!("------------------------------------");

    // Sophia
    let mut sophia_comp = SophiaOptimizer::new(OptimizerConfig {
        learning_rate: 0.01,
        ..Default::default()
    });
    let mut params_sophia = HashMap::new();
    params_sophia.insert("weight".to_string(), array![[0.0]]);
    params_sophia.insert("bias".to_string(), array![[0.0]]);

    let sophia_loss = train_and_evaluate(
        "Sophia",
        &mut sophia_comp,
        &mut params_sophia,
        &x_train,
        &y_train,
        30,
    );

    // Adam
    let mut adam_comp = AdamOptimizer::new(OptimizerConfig {
        learning_rate: 0.01,
        ..Default::default()
    });
    let mut params_adam = HashMap::new();
    params_adam.insert("weight".to_string(), array![[0.0]]);
    params_adam.insert("bias".to_string(), array![[0.0]]);

    let adam_loss = train_and_evaluate(
        "Adam",
        &mut adam_comp,
        &mut params_adam,
        &x_train,
        &y_train,
        30,
    );

    println!("\nComparison:");
    println!("Sophia final loss: {:.6}", sophia_loss);
    println!("Adam final loss:   {:.6}", adam_loss);
    println!(
        "Speedup: {:.2}x",
        adam_loss.max(1e-10) / sophia_loss.max(1e-10)
    );

    // Example 4: Sophia with gradient clipping
    println!("\nExample 4: Sophia with Gradient Clipping");
    println!("----------------------------------------");

    let sophia_clip = SophiaConfig {
        base: OptimizerConfig {
            learning_rate: 0.05, // Higher learning rate
            grad_clip: Some(1.0),
            grad_clip_mode: GradClipMode::Norm,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut sophia_clipped = SophiaOptimizer::with_sophia_config(sophia_clip);
    let mut params_clip = HashMap::new();
    params_clip.insert("weight".to_string(), array![[0.0]]);
    params_clip.insert("bias".to_string(), array![[0.0]]);

    train_and_evaluate(
        "Sophia (clipped)",
        &mut sophia_clipped,
        &mut params_clip,
        &x_train,
        &y_train,
        50,
    );

    // Example 5: Sophia with Hutchinson variant
    println!("\nExample 5: Sophia Variants");
    println!("-------------------------");

    // Gauss-Newton-Bartlett (default)
    let mut sophia_gnb = SophiaOptimizer::with_sophia_config(SophiaConfig {
        variant: SophiaVariant::GaussNewtonBartlett,
        ..Default::default()
    });
    let mut params_gnb = HashMap::new();
    params_gnb.insert("weight".to_string(), array![[0.0]]);
    params_gnb.insert("bias".to_string(), array![[0.0]]);

    let gnb_loss = train_and_evaluate(
        "Sophia-G (GNB)",
        &mut sophia_gnb,
        &mut params_gnb,
        &x_train,
        &y_train,
        30,
    );

    // Hutchinson
    let mut sophia_h = SophiaOptimizer::with_sophia_config(SophiaConfig {
        variant: SophiaVariant::Hutchinson,
        ..Default::default()
    });
    let mut params_h = HashMap::new();
    params_h.insert("weight".to_string(), array![[0.0]]);
    params_h.insert("bias".to_string(), array![[0.0]]);

    let h_loss = train_and_evaluate(
        "Sophia-H (Hutchinson)",
        &mut sophia_h,
        &mut params_h,
        &x_train,
        &y_train,
        30,
    );

    println!("\nVariant Comparison:");
    println!("GNB final loss:        {:.6}", gnb_loss);
    println!("Hutchinson final loss: {:.6}", h_loss);

    // Example 6: Checkpointing and state management
    println!("\nExample 6: State Saving and Loading");
    println!("-----------------------------------");

    let mut sophia_state = SophiaOptimizer::new(OptimizerConfig {
        learning_rate: 0.01,
        ..Default::default()
    });
    let mut params_state = HashMap::new();
    params_state.insert("weight".to_string(), array![[0.0]]);
    params_state.insert("bias".to_string(), array![[0.0]]);

    // Train for a few epochs
    for epoch in 0..10 {
        let predictions = predict(&params_state, &x_train);
        let gradients = compute_gradients(&params_state, &x_train, &y_train);
        sophia_state.step(&mut params_state, &gradients)?;

        if epoch == 9 {
            let loss = mse_loss(&predictions, &y_train.view());
            println!("Epoch {}: Loss = {:.6}", epoch + 1, loss);
        }
    }

    // Save state
    let state_dict = sophia_state.state_dict();
    println!("Saved optimizer state with {} keys", state_dict.len());

    // Create new optimizer and load state
    let mut sophia_loaded = SophiaOptimizer::new(OptimizerConfig {
        learning_rate: 0.01,
        ..Default::default()
    });

    // Initialize state with a dummy step
    let mut gradients_dummy = HashMap::new();
    gradients_dummy.insert("weight".to_string(), array![[0.1]]);
    gradients_dummy.insert("bias".to_string(), array![[0.1]]);
    sophia_loaded.step(&mut params_state, &gradients_dummy)?;

    // Load saved state
    sophia_loaded.load_state_dict(state_dict);
    println!("Loaded optimizer state successfully");

    // Continue training
    println!("Continuing training from checkpoint...");
    for epoch in 10..20 {
        let predictions = predict(&params_state, &x_train);
        let gradients = compute_gradients(&params_state, &x_train, &y_train);
        sophia_loaded.step(&mut params_state, &gradients)?;

        if epoch == 19 {
            let loss = mse_loss(&predictions, &y_train.view());
            println!("Epoch {}: Loss = {:.6}", epoch + 1, loss);
        }
    }

    println!("\nKey Takeaways:");
    println!("1. Sophia often converges faster than Adam (2-3x for LLM pretraining)");
    println!("2. Uses Hessian diagonal for better curvature awareness");
    println!("3. Memory footprint similar to Adam");
    println!("4. Recommended hyperparameters: lr=1e-4 to 2e-4, beta1=0.965, beta2=0.99, rho=0.04");
    println!("5. Update Hessian every 10 steps (configurable)");
    println!("6. Two variants: GNB (default, more accurate) and Hutchinson (cheaper)");

    Ok(())
}

/// Train a model and evaluate performance
fn train_and_evaluate<O: Optimizer>(
    name: &str,
    optimizer: &mut O,
    params: &mut HashMap<String, Array2<f64>>,
    x: &Array2<f64>,
    y: &Array2<f64>,
    epochs: usize,
) -> f64 {
    println!("{}", name);
    let mut final_loss = 0.0;

    for epoch in 0..epochs {
        // Forward pass
        let predictions = predict(params, x);

        // Compute loss
        let loss = mse_loss(&predictions, &y.view());

        // Compute gradients
        let gradients = compute_gradients(params, x, y);

        // Update parameters
        optimizer.step(params, &gradients).unwrap();

        // Print progress
        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: Loss = {:.6}", epoch + 1, loss);
        }

        final_loss = loss;
    }

    let w = params["weight"][[0, 0]];
    let b = params["bias"][[0, 0]];
    println!("  Final parameters: w={:.4}, b={:.4}", w, b);
    println!("  Target: w=2.0000, b=1.0000\n");

    final_loss
}

/// Simple linear model: y = wx + b
fn predict(params: &HashMap<String, Array2<f64>>, x: &Array2<f64>) -> Array2<f64> {
    let w = &params["weight"];
    let b = &params["bias"];
    x.dot(w) + b[[0, 0]]
}

/// Mean squared error loss
fn mse_loss(predictions: &Array2<f64>, targets: &scirs2_core::ndarray::ArrayView2<f64>) -> f64 {
    let diff = predictions - targets;
    diff.iter().map(|&d| d * d).sum::<f64>() / (predictions.len() as f64)
}

/// Compute gradients for linear regression
fn compute_gradients(
    params: &HashMap<String, Array2<f64>>,
    x: &Array2<f64>,
    y: &Array2<f64>,
) -> HashMap<String, Array2<f64>> {
    let predictions = predict(params, x);
    let error = predictions - y;
    let n = x.nrows() as f64;

    let mut gradients = HashMap::new();

    // ∂L/∂w = 2/n * Σ(y_pred - y) * x
    let grad_w = x.t().dot(&error) * (2.0 / n);
    gradients.insert("weight".to_string(), grad_w);

    // ∂L/∂b = 2/n * Σ(y_pred - y)
    let grad_b = array![[error.sum() * (2.0 / n)]];
    gradients.insert("bias".to_string(), grad_b);

    gradients
}
