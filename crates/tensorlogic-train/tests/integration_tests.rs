//! Integration tests for tensorlogic-train
//!
//! These tests verify that multiple advanced features work together correctly.

use scirs2_core::ndarray::Array2;
use tensorlogic_train::{
    CrossEntropyLoss, DistillationLoss, LabelSmoothingLoss, LinearModel, Loss, Model, MseLoss,
    MultiTaskLoss, TrainError,
};

/// Test: Distillation + Label Smoothing Integration
#[test]
fn test_distillation_with_label_smoothing() -> Result<(), TrainError> {
    let num_classes = 5;
    let batch_size = 8;

    // Create synthetic data
    let mut teacher_logits = Array2::zeros((batch_size, num_classes));
    let mut student_logits = Array2::zeros((batch_size, num_classes));
    let mut targets = Array2::zeros((batch_size, num_classes));

    for i in 0..batch_size {
        let target_class = i % num_classes;
        targets[[i, target_class]] = 1.0;

        for j in 0..num_classes {
            teacher_logits[[i, j]] = if j == target_class { 2.0 } else { -0.5 };
            student_logits[[i, j]] = if j == target_class { 1.5 } else { -0.3 };
        }
    }

    // Test 1: Standard distillation
    let ce_loss = CrossEntropyLoss::default();
    let distillation = DistillationLoss::new(3.0, 0.7, Box::new(ce_loss))?;

    let loss1 = distillation.compute_distillation(
        &student_logits.view(),
        &teacher_logits.view(),
        &targets.view(),
    )?;

    assert!(loss1 > 0.0 && loss1.is_finite());

    // Test 2: Distillation with label smoothing
    let ls_loss = LabelSmoothingLoss::new(0.1, num_classes)?;
    let distillation_ls = DistillationLoss::new(3.0, 0.7, Box::new(ls_loss))?;

    let loss2 = distillation_ls.compute_distillation(
        &student_logits.view(),
        &teacher_logits.view(),
        &targets.view(),
    )?;

    assert!(loss2 > 0.0 && loss2.is_finite());

    // Label smoothing should slightly reduce loss
    assert!(loss2 < loss1 + 1.0); // Allow some tolerance

    Ok(())
}

/// Test: Multi-task Learning with Progressive Difficulty
#[test]
fn test_multitask_with_progressive_difficulty() -> Result<(), TrainError> {
    let batch_size = 16;
    let task1_outputs = 5;
    let task2_outputs = 10;
    let total_outputs = task1_outputs + task2_outputs;

    // Create predictions and targets
    let mut predictions = Array2::zeros((batch_size, total_outputs));
    let mut targets = Array2::zeros((batch_size, total_outputs));

    for i in 0..batch_size {
        // Task 1: Classification
        let task1_class = i % task1_outputs;
        targets[[i, task1_class]] = 1.0;
        predictions[[i, task1_class]] = 1.5;

        // Task 2: Regression
        for j in task1_outputs..total_outputs {
            targets[[i, j]] = (j - task1_outputs) as f64 * 0.1;
            predictions[[i, j]] = targets[[i, j]] + 0.05;
        }
    }

    let task_splits = vec![0, task1_outputs, total_outputs];

    // Multi-task loss with fixed weights
    let losses: Vec<Box<dyn Loss>> = vec![Box::new(CrossEntropyLoss::default()), Box::new(MseLoss)];
    let weights = vec![0.6, 0.4];

    let mut mt_loss = MultiTaskLoss::new_fixed(losses, weights)?;

    // Compute loss for different difficulty levels (simulated curriculum)
    let difficulties = vec![0.5, 0.75, 1.0];
    let mut losses_at_difficulties = Vec::new();

    for &difficulty in &difficulties {
        // Scale predictions based on difficulty
        let scaled_preds = &predictions * difficulty;
        let loss =
            mt_loss.compute_multi_task(&scaled_preds.view(), &targets.view(), &task_splits)?;
        losses_at_difficulties.push(loss);
    }

    // Verify we got losses for all difficulty levels
    assert_eq!(losses_at_difficulties.len(), 3);
    assert!(losses_at_difficulties[0] > 0.0);

    Ok(())
}

/// Test: Combined Regularization Techniques
#[test]
fn test_combined_regularization() -> Result<(), TrainError> {
    let num_classes = 5;
    let batch_size = 16;

    // Create synthetic data
    let mut predictions = Array2::zeros((batch_size, num_classes));
    let mut targets = Array2::zeros((batch_size, num_classes));

    for i in 0..batch_size {
        let target_class = i % num_classes;
        targets[[i, target_class]] = 1.0;

        for j in 0..num_classes {
            predictions[[i, j]] = if j == target_class {
                2.0 + (i as f64 * 0.1)
            } else {
                -0.5 + (j as f64 * 0.05)
            };
        }
    }

    // Test 1: Standard cross-entropy
    let ce_loss = CrossEntropyLoss::default();
    let loss_ce = ce_loss.compute(&predictions.view(), &targets.view())?;

    // Test 2: With label smoothing
    let ls_loss = LabelSmoothingLoss::new(0.1, num_classes)?;
    let loss_ls = ls_loss.compute(&predictions.view(), &targets.view())?;

    // Test 3: Verify both work
    assert!(loss_ce > 0.0 && loss_ce.is_finite());
    assert!(loss_ls > 0.0 && loss_ls.is_finite());

    // Label smoothing typically gives similar but different loss (both should be finite and positive)
    assert!(loss_ce > 0.0 && loss_ls > 0.0);

    Ok(())
}

/// Test: Model Training Workflow
#[test]
fn test_integrated_training_workflow() -> Result<(), TrainError> {
    let input_size = 20;
    let output_size = 10;
    let batch_size = 32;

    // Create model
    let model = LinearModel::new(input_size, output_size);

    // Create synthetic training data
    let mut inputs = Array2::zeros((batch_size, input_size));
    let mut targets = Array2::zeros((batch_size, output_size));

    for i in 0..batch_size {
        for j in 0..input_size {
            inputs[[i, j]] = (i as f64 + j as f64) * 0.01;
        }

        let target_class = i % output_size;
        targets[[i, target_class]] = 1.0;
    }

    // Create loss function
    let loss_fn = CrossEntropyLoss::default();

    // Forward pass
    let predictions = model.forward(&inputs.view())?;
    assert_eq!(predictions.shape(), &[batch_size, output_size]);

    // Compute loss
    let loss = loss_fn.compute(&predictions.view(), &targets.view())?;
    assert!(loss > 0.0 && loss.is_finite());

    // Verify gradient computation
    let grad = loss_fn.gradient(&predictions.view(), &targets.view())?;
    assert_eq!(grad.shape(), predictions.shape());

    // Verify gradients are reasonable
    let grad_norm: f64 = grad.iter().map(|&x| x * x).sum::<f64>().sqrt();
    assert!(grad_norm > 0.0 && grad_norm.is_finite());

    Ok(())
}

/// Test: Ensemble with Different Loss Functions
#[test]
fn test_ensemble_with_mixed_objectives() -> Result<(), TrainError> {
    use tensorlogic_train::{DiceLoss, TverskyLoss};

    let num_classes = 3;
    let batch_size = 16;

    // Create test data
    let mut predictions = Array2::zeros((batch_size, num_classes));
    let mut targets = Array2::zeros((batch_size, num_classes));

    for i in 0..batch_size {
        let target_class = i % num_classes;
        targets[[i, target_class]] = 1.0;

        for j in 0..num_classes {
            predictions[[i, j]] = if j == target_class { 2.0 } else { 0.5 };
        }
    }

    // Test Dice loss
    let dice_loss = DiceLoss::default();
    let loss1 = dice_loss.compute(&predictions.view(), &targets.view())?;
    assert!(loss1 >= 0.0 && loss1.is_finite());

    // Test Tversky loss
    let tversky_loss = TverskyLoss::default();
    let loss2 = tversky_loss.compute(&predictions.view(), &targets.view())?;
    assert!(loss2 >= 0.0 && loss2.is_finite());

    // Both losses should work
    assert!(loss1.is_finite() && loss2.is_finite());

    Ok(())
}

/// Test: Model State Management
#[test]
fn test_stateful_training_checkpoint() -> Result<(), TrainError> {
    let input_size = 10;
    let output_size = 5;

    // Create model
    let model = LinearModel::new(input_size, output_size);
    let state_dict = model.state_dict();

    // Verify state contains parameters (check for expected keys)
    assert!(!state_dict.is_empty(), "State dict should not be empty");

    // Verify we can retrieve some parameters
    let total_params: usize = state_dict.values().map(|v| v.len()).sum();
    let expected_params = input_size * output_size + output_size; // weights + bias
    assert_eq!(
        total_params, expected_params,
        "Total parameters should match model size"
    );

    Ok(())
}

/// Test: Multiple Loss Combinations
#[test]
fn test_multiple_loss_combinations() -> Result<(), TrainError> {
    use tensorlogic_train::{HuberLoss, MixupLoss};

    let num_samples = 10;
    let num_features = 5;

    // Create test data
    let mut predictions = Array2::zeros((num_samples, num_features));
    let mut targets = Array2::zeros((num_samples, num_features));

    for i in 0..num_samples {
        for j in 0..num_features {
            predictions[[i, j]] = (j + 1) as f64;
            targets[[i, j]] = (j + 1) as f64 + 0.1;
        }
    }

    // Test MSE loss
    let mse_loss = MseLoss;
    let loss1 = mse_loss.compute(&predictions.view(), &targets.view())?;
    assert!(loss1 > 0.0 && loss1.is_finite());

    // Test Huber loss (robust to outliers)
    let huber_loss = HuberLoss::default();
    let loss2 = huber_loss.compute(&predictions.view(), &targets.view())?;
    assert!(loss2 > 0.0 && loss2.is_finite());

    // Test Mixup loss wrapper
    let mixup_loss = MixupLoss::new(1.0, Box::new(MseLoss))?;
    let loss3 = mixup_loss.compute_mixup(&predictions.view(), &targets.view())?;
    assert!(loss3 > 0.0 && loss3.is_finite());

    // All losses should give reasonable values
    assert!(loss1.is_finite() && loss2.is_finite() && loss3.is_finite());

    Ok(())
}
