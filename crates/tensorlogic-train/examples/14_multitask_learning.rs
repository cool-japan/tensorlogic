//! # Multi-Task Learning Example
//!
//! This example demonstrates multi-task learning (MTL), where a single model is trained
//! to perform multiple related tasks simultaneously. Benefits include:
//! - Improved generalization through shared representations
//! - Better sample efficiency (shared learning)
//! - Implicit regularization
//! - Single model deployment for multiple tasks
//!
//! We cover:
//! 1. Fixed task weighting
//! 2. Dynamic Task Prioritization (DTP)
//! 3. GradNorm for gradient balancing
//! 4. PCGrad for resolving gradient conflicts
//!
//! References:
//! - "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al., 2018)
//! - "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (Chen et al., 2018)
//! - "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)

use scirs2_core::array;
use scirs2_core::ndarray::Array;
use std::collections::HashMap;
use tensorlogic_train::{
    CrossEntropyLoss, Loss, MseLoss, MultiTaskLoss, PCGrad, TaskWeightingStrategy, TrainError,
};

fn main() -> Result<(), TrainError> {
    println!("=== Multi-Task Learning Example ===\n");

    // ============================================================================
    // Scenario: Joint training for image classification and segmentation
    // Task 1: Classification (5 classes)
    // Task 2: Segmentation (20 output channels)
    // ============================================================================

    let batch_size = 16;
    let task1_classes = 5; // Classification
    let task2_channels = 20; // Segmentation

    println!("Multi-task scenario:");
    println!("  Task 1: Image classification ({} classes)", task1_classes);
    println!(
        "  Task 2: Semantic segmentation ({} channels)",
        task2_channels
    );
    println!("  Batch size: {}\n", batch_size);

    // Generate sample predictions and targets
    // Predictions are concatenated: [task1_logits | task2_logits]
    let total_outputs = task1_classes + task2_channels;
    let mut predictions = Array::zeros((batch_size, total_outputs));
    let mut targets = Array::zeros((batch_size, total_outputs));

    // Task 1: Classification predictions and targets
    for i in 0..batch_size {
        for j in 0..task1_classes {
            predictions[[i, j]] = if j == i % task1_classes {
                2.0 + (i as f64 * 0.1)
            } else {
                -0.5 + (j as f64 * 0.05)
            };
        }
        let target_class = i % task1_classes;
        targets[[i, target_class]] = 1.0;
    }

    // Task 2: Segmentation predictions and targets
    for i in 0..batch_size {
        for j in task1_classes..total_outputs {
            let idx = j - task1_classes;
            predictions[[i, j]] = 0.5 + (idx as f64 * 0.1) + (i as f64 * 0.05);
            targets[[i, j]] = 0.6 + (idx as f64 * 0.08);
        }
    }

    let task_splits = vec![0, task1_classes, total_outputs];

    // ============================================================================
    // 1. Fixed Task Weighting
    // ============================================================================
    println!("--- 1. Fixed Task Weighting ---");

    let losses1: Vec<Box<dyn Loss>> =
        vec![Box::new(CrossEntropyLoss::default()), Box::new(MseLoss)];

    let weights = vec![0.6, 0.4]; // 60% classification, 40% segmentation

    let mut fixed_loss = MultiTaskLoss::new_fixed(losses1, weights.clone())?;

    println!("Task weights: {:?}", weights);
    println!("  → Classification: 60% (primary task)");
    println!("  → Segmentation:   40% (auxiliary task)");

    let loss1 =
        fixed_loss.compute_multi_task(&predictions.view(), &targets.view(), &task_splits)?;

    println!("Combined loss: {:.4}", loss1);
    println!("Weights remain constant throughout training\n");

    // ============================================================================
    // 2. Dynamic Task Prioritization (DTP)
    // ============================================================================
    println!("--- 2. Dynamic Task Prioritization (DTP) ---");

    let losses2: Vec<Box<dyn Loss>> =
        vec![Box::new(CrossEntropyLoss::default()), Box::new(MseLoss)];

    let mut dtp_loss = MultiTaskLoss::new_dynamic(
        losses2,
        TaskWeightingStrategy::DynamicTaskPrioritization,
        0.01, // Learning rate for weight updates
    )?;

    println!("Strategy: Weight tasks based on their current loss");
    println!("  → Tasks with higher loss get higher weight");
    println!("  → Automatically focuses on harder tasks\n");

    // Simulate multiple training steps
    println!("Simulating 5 training steps:");
    println!(
        "{:<8} {:<15} {:<20} {:<20}",
        "Step", "Loss", "Task1 Weight", "Task2 Weight"
    );
    println!("{}", "-".repeat(63));

    for step in 0..5 {
        let loss =
            dtp_loss.compute_multi_task(&predictions.view(), &targets.view(), &task_splits)?;

        let weights = dtp_loss.get_weights();
        println!(
            "{:<8} {:<15.4} {:<20.4} {:<20.4}",
            step + 1,
            loss,
            weights[0],
            weights[1]
        );

        // Simulate improvement (loss decreases)
        for i in 0..predictions.nrows() {
            for j in 0..predictions.ncols() {
                predictions[[i, j]] *= 0.95; // Gradual improvement
            }
        }
    }

    println!("\n  → Weights automatically adjust based on task difficulty");
    println!("  → No manual tuning required\n");

    // ============================================================================
    // 3. GradNorm Strategy
    // ============================================================================
    println!("--- 3. GradNorm Strategy ---");

    let losses3: Vec<Box<dyn Loss>> =
        vec![Box::new(CrossEntropyLoss::default()), Box::new(MseLoss)];

    let alpha = 1.5; // Balance parameter
    let mut gradnorm_loss =
        MultiTaskLoss::new_dynamic(losses3, TaskWeightingStrategy::GradNorm { alpha }, 0.01)?;

    println!("Strategy: Balance gradient magnitudes across tasks");
    println!("Alpha: {} (controls balancing strength)", alpha);
    println!("  → Prevents one task from dominating gradients");
    println!("  → Aims for equal training progress across tasks\n");

    // Reset predictions
    let mut predictions_reset = predictions.clone();

    println!("Simulating 5 training steps:");
    println!(
        "{:<8} {:<15} {:<20} {:<20}",
        "Step", "Loss", "Task1 Weight", "Task2 Weight"
    );
    println!("{}", "-".repeat(63));

    for step in 0..5 {
        let loss = gradnorm_loss.compute_multi_task(
            &predictions_reset.view(),
            &targets.view(),
            &task_splits,
        )?;

        let weights = gradnorm_loss.get_weights();
        println!(
            "{:<8} {:<15.4} {:<20.4} {:<20.4}",
            step + 1,
            loss,
            weights[0],
            weights[1]
        );

        // Simulate improvement
        for i in 0..predictions_reset.nrows() {
            for j in 0..predictions_reset.ncols() {
                predictions_reset[[i, j]] *= 0.93;
            }
        }
    }

    println!("\n  → GradNorm dynamically balances training rates");
    println!("  → Particularly effective for tasks with different scales\n");

    // ============================================================================
    // 4. PCGrad - Gradient Conflict Resolution
    // ============================================================================
    println!("--- 4. PCGrad - Gradient Conflict Resolution ---");

    println!("When task gradients conflict (point in opposite directions),");
    println!("PCGrad projects conflicting gradients onto normal planes.\n");

    // Simulate gradients for two tasks
    let mut task1_grads = HashMap::new();
    let mut task2_grads = HashMap::new();

    // Layer 1: Aligned gradients
    task1_grads.insert("layer1".to_string(), array![[1.0, 2.0], [3.0, 4.0]]);
    task2_grads.insert("layer1".to_string(), array![[1.2, 1.8], [2.8, 3.9]]);

    // Layer 2: Conflicting gradients
    task1_grads.insert("layer2".to_string(), array![[1.0, 0.0], [0.0, 1.0]]);
    task2_grads.insert("layer2".to_string(), array![[-0.8, 0.0], [0.0, -0.9]]);

    let task_gradients = vec![task1_grads, task2_grads];

    println!("Task gradients before PCGrad:");
    println!("  Layer 1 (aligned):");
    println!("    Task 1: [[1.0, 2.0], [3.0, 4.0]]");
    println!("    Task 2: [[1.2, 1.8], [2.8, 3.9]]");
    println!("    → Positive dot product (aligned)");
    println!();
    println!("  Layer 2 (conflicting):");
    println!("    Task 1: [[1.0, 0.0], [0.0, 1.0]]");
    println!("    Task 2: [[-0.8, 0.0], [0.0, -0.9]]");
    println!("    → Negative dot product (conflicting)\n");

    let combined_grads = PCGrad::apply(&task_gradients)?;

    println!("After PCGrad:");
    if let Some(layer1) = combined_grads.get("layer1") {
        println!("  Layer 1 (aligned gradients preserved):");
        println!(
            "    Combined: [[{:.2}, {:.2}], [{:.2}, {:.2}]]",
            layer1[[0, 0]],
            layer1[[0, 1]],
            layer1[[1, 0]],
            layer1[[1, 1]]
        );
    }
    if let Some(layer2) = combined_grads.get("layer2") {
        println!("  Layer 2 (conflicts resolved):");
        println!(
            "    Combined: [[{:.2}, {:.2}], [{:.2}, {:.2}]]",
            layer2[[0, 0]],
            layer2[[0, 1]],
            layer2[[1, 0]],
            layer2[[1, 1]]
        );
        println!("    → Conflicting components reduced");
    }
    println!();
    println!("  → Aligned gradients averaged normally");
    println!("  → Conflicting gradients projected to resolve conflicts\n");

    // ============================================================================
    // 5. Choosing the Right Strategy
    // ============================================================================
    println!("=== Choosing the Right Strategy ===\n");

    println!("Fixed Weighting:");
    println!("  When to use:");
    println!("    ✓ You know the relative importance of tasks");
    println!("    ✓ Tasks are well-balanced");
    println!("    ✓ Simplest approach, good baseline");
    println!("  Pros: Simple, stable, interpretable");
    println!("  Cons: Requires manual tuning, doesn't adapt");
    println!();

    println!("Dynamic Task Prioritization (DTP):");
    println!("  When to use:");
    println!("    ✓ Tasks have different difficulty levels");
    println!("    ✓ Want automatic adaptation");
    println!("    ✓ Tasks should be weighted by current performance");
    println!("  Pros: Automatic, focuses on harder tasks");
    println!("  Cons: May oscillate, less interpretable");
    println!();

    println!("GradNorm:");
    println!("  When to use:");
    println!("    ✓ Tasks have very different loss scales");
    println!("    ✓ Want equal training progress across tasks");
    println!("    ✓ Tasks should converge at similar rates");
    println!("  Pros: Balanced training, prevents dominance");
    println!("  Cons: Adds hyperparameter (alpha), computational overhead");
    println!();

    println!("PCGrad:");
    println!("  When to use:");
    println!("    ✓ Tasks may have conflicting gradients");
    println!("    ✓ Training is unstable");
    println!("    ✓ Want to preserve positive transfer, remove negative");
    println!("  Pros: Resolves conflicts, improves stability");
    println!("  Cons: Higher computational cost, works on gradient level");
    println!();

    println!("Uncertainty Weighting:");
    println!("  When to use:");
    println!("    ✓ Tasks have different inherent uncertainties");
    println!("    ✓ Want principled weighting based on noise");
    println!("  Pros: Theoretically motivated");
    println!("  Cons: Requires learnable uncertainty parameters");
    println!();

    // ============================================================================
    // 6. Best Practices
    // ============================================================================
    println!("=== Best Practices ===\n");

    println!("1. Architecture Design:");
    println!("   - Share early layers (features)");
    println!("   - Task-specific heads for final predictions");
    println!("   - Consider auxiliary tasks for regularization");
    println!();

    println!("2. Task Selection:");
    println!("   - Choose related tasks (shared knowledge)");
    println!("   - Balance task complexity");
    println!("   - Consider task correlations");
    println!();

    println!("3. Weight Initialization:");
    println!("   - Start with equal weights (1/n_tasks)");
    println!("   - Gradually introduce dynamic weighting");
    println!("   - Monitor individual task performance");
    println!();

    println!("4. Monitoring:");
    println!("   - Track per-task losses separately");
    println!("   - Monitor task weight evolution");
    println!("   - Watch for task dominance or collapse");
    println!();

    println!("5. Debugging:");
    println!("   - If one task dominates: increase its weight ceiling");
    println!("   - If training is unstable: try PCGrad");
    println!("   - If tasks converge at different rates: try GradNorm");
    println!();

    println!("6. Common Pitfalls:");
    println!("   ✗ Using unrelated tasks (negative transfer)");
    println!("   ✗ Ignoring task imbalance");
    println!("   ✗ Not monitoring individual task performance");
    println!("   ✗ Over-complicating with too many tasks");
    println!();

    // ============================================================================
    // 7. Recommended Combinations
    // ============================================================================
    println!("=== Recommended Combinations ===\n");

    println!("For stable, balanced training:");
    println!("  → GradNorm + PCGrad");
    println!("    (Balance gradients, then resolve conflicts)");
    println!();

    println!("For quick experimentation:");
    println!("  → Fixed weights → DTP");
    println!("    (Start simple, add dynamics if needed)");
    println!();

    println!("For production deployment:");
    println!("  → Tune fixed weights with validation set");
    println!("    (Stability and predictability matter)");
    println!();

    println!("For research/maximum performance:");
    println!("  → GradNorm or Uncertainty Weighting + PCGrad");
    println!("    (Sophisticated balancing and conflict resolution)");

    Ok(())
}
