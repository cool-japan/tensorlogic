//! # Label Smoothing and Mixup Example
//!
//! This example demonstrates two powerful regularization techniques:
//! 1. Label Smoothing - prevents overconfident predictions by smoothing target distribution
//! 2. Mixup - data augmentation that mixes training examples and their labels
//!
//! These techniques improve generalization and calibration of neural networks.
//!
//! References:
//! - "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., 2016)
//! - "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)

use scirs2_core::array;
use scirs2_core::ndarray::{Array, Array2};
use tensorlogic_train::{
    CrossEntropyLoss, LabelSmoothingLoss, Loss, MixupLoss, MseLoss, TrainError,
};

fn generate_sample_data(num_samples: usize, num_classes: usize) -> (Array2<f64>, Array2<f64>) {
    let mut predictions = Array::zeros((num_samples, num_classes));
    let mut targets = Array::zeros((num_samples, num_classes));

    for i in 0..num_samples {
        // Generate logits with some pattern
        for j in 0..num_classes {
            predictions[[i, j]] = if j == i % num_classes {
                3.0 + (i as f64 * 0.1) // True class gets higher logit
            } else {
                -1.0 + (j as f64 * 0.05)
            };
        }

        // One-hot encode targets
        let target_class = i % num_classes;
        targets[[i, target_class]] = 1.0;
    }

    (predictions, targets)
}

fn main() -> Result<(), TrainError> {
    println!("=== Label Smoothing and Mixup Example ===\n");

    let num_samples = 100;
    let num_classes = 10;

    // Generate sample data
    let (predictions, hard_targets) = generate_sample_data(num_samples, num_classes);

    println!(
        "Dataset: {} samples, {} classes\n",
        num_samples, num_classes
    );

    // ============================================================================
    // 1. Standard Cross-Entropy (Baseline)
    // ============================================================================
    println!("--- 1. Standard Cross-Entropy Loss (Baseline) ---");

    let ce_loss = CrossEntropyLoss::default();
    let baseline_loss = ce_loss.compute(&predictions.view(), &hard_targets.view())?;

    println!("Cross-entropy loss: {:.4}", baseline_loss);
    println!("  → Uses hard 0/1 labels");
    println!("  → Can lead to overconfident predictions");
    println!("  → No regularization effect\n");

    // ============================================================================
    // 2. Label Smoothing with Different Epsilon Values
    // ============================================================================
    println!("--- 2. Label Smoothing ---");

    let epsilon_values = vec![0.0, 0.05, 0.1, 0.2, 0.3];

    println!("Comparing different smoothing strengths:\n");
    println!("{:<10} {:<15} {:<30}", "Epsilon", "Loss", "Description");
    println!("{}", "-".repeat(55));

    for &epsilon in &epsilon_values {
        let ls_loss = LabelSmoothingLoss::new(epsilon, num_classes)?;
        let loss_value = ls_loss.compute(&predictions.view(), &hard_targets.view())?;

        let description = match epsilon {
            e if (e - 0.0).abs() < 1e-6 => "No smoothing (same as CE)",
            e if e <= 0.1 => "Light smoothing (recommended)",
            e if e <= 0.2 => "Medium smoothing",
            _ => "Heavy smoothing",
        };

        println!("{:<10.2} {:<15.4} {:<30}", epsilon, loss_value, description);

        // Show what smoothed labels look like for epsilon = 0.1
        if (epsilon - 0.1).abs() < 1e-6 {
            let sample_target = array![[0.0, 1.0, 0.0, 0.0, 0.0]];
            let smoothed = ls_loss.smooth_labels(&sample_target.view());

            println!("\n  Example label smoothing (ε=0.1, 5 classes):");
            println!("  Original: [0.0, 1.0, 0.0, 0.0, 0.0]");
            print!("  Smoothed: [");
            for (i, &val) in smoothed.iter().enumerate() {
                print!("{:.3}", val);
                if i < smoothed.len() - 1 {
                    print!(", ");
                }
            }
            println!("]");
            println!("  → True class: {} (was 1.0)", 1.0 - epsilon);
            println!("  → Other classes: {} each (was 0.0)\n", epsilon / 4.0);
        }
    }

    // ============================================================================
    // 3. Understanding Label Smoothing Mathematics
    // ============================================================================
    println!("\n--- 3. Label Smoothing Mathematics ---");

    let epsilon = 0.1;
    println!("For ε = {} and K = {} classes:", epsilon, num_classes);
    println!();
    println!("Smoothed labels:");
    println!("  y_smooth(k) = (1 - ε)           if k is true class");
    println!("              = ε / (K - 1)        otherwise");
    println!();
    println!("In our case:");
    println!("  True class:   1 - {} = {}", epsilon, 1.0 - epsilon);
    println!(
        "  Other classes: {} / ({} - 1) = {:.4}",
        epsilon,
        num_classes,
        epsilon / (num_classes - 1) as f64
    );
    println!();
    println!("Effect:");
    println!("  ✓ Prevents model from being overconfident");
    println!("  ✓ Improves calibration (predicted probabilities match true probabilities)");
    println!("  ✓ Acts as regularization");
    println!("  ✓ Often improves test accuracy\n");

    // ============================================================================
    // 4. Mixup Data Augmentation
    // ============================================================================
    println!("--- 4. Mixup Data Augmentation ---");

    let alpha = 1.0; // Beta distribution parameter
    let _mixup_loss = MixupLoss::new(alpha, Box::new(MseLoss))?;

    println!("Alpha parameter: {} (controls mixing strength)", alpha);
    println!();

    // Demonstrate mixing two examples
    let sample1 = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
    let sample2 = array![[5.0, 4.0, 3.0, 2.0, 1.0]];

    let lambda_values = vec![0.0, 0.25, 0.5, 0.75, 1.0];

    println!("Mixing two samples with different λ (lambda):\n");
    println!("Sample 1: [1.0, 2.0, 3.0, 4.0, 5.0]");
    println!("Sample 2: [5.0, 4.0, 3.0, 2.0, 1.0]\n");

    println!("{:<8} {:<40}", "Lambda", "Mixed Sample");
    println!("{}", "-".repeat(48));

    for &lambda in &lambda_values {
        let mixed = MixupLoss::mix_data(&sample1.view(), &sample2.view(), lambda)?;

        print!("{:<8.2} [", lambda);
        for (i, &val) in mixed.iter().enumerate() {
            print!("{:.1}", val);
            if i < mixed.len() - 1 {
                print!(", ");
            }
        }
        println!("]");

        let description = match lambda {
            l if (l - 0.0).abs() < 1e-6 => " ← Pure sample 2",
            l if (l - 1.0).abs() < 1e-6 => " ← Pure sample 1",
            l if (l - 0.5).abs() < 1e-6 => " ← Equal mix",
            _ => "",
        };
        if !description.is_empty() {
            println!("         {}", description);
        }
    }

    println!("\nMixup formula: x_mixed = λ·x₁ + (1-λ)·x₂");
    println!("               y_mixed = λ·y₁ + (1-λ)·y₂");
    println!();
    println!("Where λ ~ Beta(α, α), typically α = 1.0\n");

    // ============================================================================
    // 5. Combining Label Smoothing with Other Techniques
    // ============================================================================
    println!("--- 5. Combining Label Smoothing with Mixup ---");

    println!("Best practice: Use both techniques together!");
    println!();
    println!("Training pipeline:");
    println!("  1. Apply Mixup during data loading:");
    println!("     - Sample λ ~ Beta(α, α)");
    println!("     - Mix input pairs: x_mixed = λ·x₁ + (1-λ)·x₂");
    println!("     - Mix target pairs: y_mixed = λ·y₁ + (1-λ)·y₂");
    println!();
    println!("  2. Apply Label Smoothing to mixed targets:");
    println!("     - Smooth the already-mixed labels");
    println!("     - Use in loss computation");
    println!();
    println!("  3. Compute loss with smoothed, mixed targets:");
    println!("     - Standard cross-entropy or other loss");
    println!("     - Both techniques contribute to regularization\n");

    // Demonstrate combined effect
    let epsilon = 0.1;
    let ls_loss = LabelSmoothingLoss::new(epsilon, num_classes)?;

    // First mix two targets
    let target1 = array![[1.0, 0.0, 0.0, 0.0, 0.0]];
    let target2 = array![[0.0, 0.0, 1.0, 0.0, 0.0]];
    let lambda = 0.5;

    let mixed_target = MixupLoss::mix_data(&target1.view(), &target2.view(), lambda)?;
    println!("Example: Combining Mixup and Label Smoothing");
    println!("Target 1 (class 0):  [1.0, 0.0, 0.0, 0.0, 0.0]");
    println!("Target 2 (class 2):  [0.0, 0.0, 1.0, 0.0, 0.0]");
    println!(
        "Mixed (λ=0.5):       [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
        mixed_target[[0, 0]],
        mixed_target[[0, 1]],
        mixed_target[[0, 2]],
        mixed_target[[0, 3]],
        mixed_target[[0, 4]]
    );

    let smoothed = ls_loss.smooth_labels(&mixed_target.view());
    println!(
        "After smoothing:     [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        smoothed[[0, 0]],
        smoothed[[0, 1]],
        smoothed[[0, 2]],
        smoothed[[0, 3]],
        smoothed[[0, 4]]
    );
    println!("  → Combines soft mixing with smoothing regularization\n");

    // ============================================================================
    // 6. Recommended Hyperparameters
    // ============================================================================
    println!("=== Recommended Hyperparameters ===");
    println!();
    println!("Label Smoothing (ε):");
    println!("  Image classification: 0.1 (standard choice)");
    println!("  Small datasets:       0.05-0.1 (lighter)");
    println!("  Large datasets:       0.1-0.2 (can be stronger)");
    println!("  Language models:      0.1 (widely used)");
    println!();
    println!("Mixup (α):");
    println!("  Standard choice:      1.0 (uniform λ distribution)");
    println!("  Conservative:         0.2-0.4 (less mixing)");
    println!("  Aggressive:           2.0 (more extreme mixing)");
    println!();
    println!("When to Use:");
    println!("  ✓ Classification tasks");
    println!("  ✓ When model is overfitting");
    println!("  ✓ When calibration is important");
    println!("  ✓ With large models / small datasets");
    println!();
    println!("When NOT to Use:");
    println!("  ✗ Regression tasks (for label smoothing)");
    println!("  ✗ When you need hard predictions");
    println!("  ✗ With very small models");
    println!("  ✗ When training time is very limited");
    println!();
    println!("Expected Benefits:");
    println!("  • Improved test accuracy: +0.5-2%");
    println!("  • Better calibration");
    println!("  • More robust to label noise");
    println!("  • Reduced overfitting");

    Ok(())
}
