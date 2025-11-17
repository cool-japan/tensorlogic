//! # Knowledge Distillation Example
//!
//! This example demonstrates knowledge distillation, where a smaller "student" model
//! learns from a larger "teacher" model's outputs. This is useful for:
//! - Model compression (deploy smaller models with similar performance)
//! - Transfer learning (distill knowledge from pretrained models)
//! - Ensemble distillation (compress an ensemble into a single model)
//!
//! Based on "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

use scirs2_core::array;
use scirs2_core::ndarray::{Array, Array2};
use tensorlogic_train::{
    AttentionTransferLoss, CrossEntropyLoss, DistillationLoss, FeatureDistillationLoss,
    LinearModel, Model, TrainError,
};

/// Simulated teacher model with more parameters
struct TeacherModel {
    num_features: usize,
    num_classes: usize,
}

impl TeacherModel {
    fn new(num_features: usize, num_classes: usize) -> Self {
        Self {
            num_features,
            num_classes,
        }
    }

    /// Simulate teacher predictions (would be actual forward pass in practice)
    fn predict(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let batch_size = inputs.nrows();
        let mut logits = Array::zeros((batch_size, self.num_classes));

        // Simulate teacher predictions with some pattern
        for i in 0..batch_size {
            for j in 0..self.num_classes {
                // Teacher produces "soft" predictions
                let val = if j == 0 {
                    2.5 + inputs[[i, 0]] * 0.5
                } else if j == 1 {
                    1.8 + inputs[[i, 1]] * 0.5
                } else {
                    0.5 + inputs[[i, j % self.num_features]] * 0.2
                };
                logits[[i, j]] = val;
            }
        }

        logits
    }

    /// Simulate intermediate feature extraction
    fn extract_features(&self, _inputs: &Array2<f64>) -> Vec<Array2<f64>> {
        // Simulate 3 layers of features
        vec![
            array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            array![[0.5, 1.5], [2.5, 3.5]],
            array![[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        ]
    }
}

/// Simulated student model with fewer parameters
struct StudentModel {
    model: LinearModel,
}

impl StudentModel {
    fn new(num_features: usize, num_classes: usize) -> Self {
        Self {
            model: LinearModel::new(num_features, num_classes),
        }
    }

    fn predict(&self, inputs: &Array2<f64>) -> Result<Array2<f64>, TrainError> {
        self.model.forward(&inputs.view())
    }

    /// Simulate intermediate feature extraction for student
    fn extract_features(&self, _inputs: &Array2<f64>) -> Vec<Array2<f64>> {
        // Student has similar structure but different values
        vec![
            array![[0.9, 1.9, 2.9], [3.9, 4.9, 5.9]],
            array![[0.4, 1.4], [2.4, 3.4]],
            array![[0.05, 0.15, 0.25, 0.35], [0.45, 0.55, 0.65, 0.75]],
        ]
    }
}

fn main() -> Result<(), TrainError> {
    println!("=== Knowledge Distillation Example ===\n");

    // Configuration
    let num_features = 10;
    let num_classes = 5;
    let batch_size = 32;

    // Create models
    let teacher = TeacherModel::new(num_features, num_classes);
    let student = StudentModel::new(num_features, num_classes);

    println!(
        "Teacher model: {} features -> {} classes",
        num_features, num_classes
    );
    println!(
        "Student model: {} features -> {} classes",
        num_features, num_classes
    );
    println!("Batch size: {}\n", batch_size);

    // Generate sample data
    let mut inputs = Array::zeros((batch_size, num_features));
    let mut targets = Array::zeros((batch_size, num_classes));

    for i in 0..batch_size {
        for j in 0..num_features {
            inputs[[i, j]] = (i as f64 * 0.1 + j as f64 * 0.05) % 1.0;
        }
        // One-hot encode targets
        let target_class = i % num_classes;
        targets[[i, target_class]] = 1.0;
    }

    // ============================================================================
    // 1. Standard Distillation Loss
    // ============================================================================
    println!("--- 1. Standard Knowledge Distillation ---");

    let temperature = 3.0;
    let alpha = 0.7; // Weight for soft targets
    let distillation_loss =
        DistillationLoss::new(temperature, alpha, Box::new(CrossEntropyLoss::default()))?;

    println!("Temperature: {} (softer predictions)", temperature);
    println!("Alpha: {} (70% soft, 30% hard targets)", alpha);

    // Get teacher and student predictions
    let teacher_logits = teacher.predict(&inputs);
    let student_logits = student.predict(&inputs)?;

    // Compute distillation loss
    let loss_value = distillation_loss.compute_distillation(
        &student_logits.view(),
        &teacher_logits.view(),
        &targets.view(),
    )?;

    println!("Distillation loss: {:.4}", loss_value);
    println!(
        "  → Combines soft targets from teacher ({:.0}%) with hard targets ({:.0}%)",
        alpha * 100.0,
        (1.0 - alpha) * 100.0
    );
    println!("  → Temperature scaling prevents overconfident predictions\n");

    // ============================================================================
    // 2. Feature-based Distillation
    // ============================================================================
    println!("--- 2. Feature-based Distillation ---");

    // Layer weights for intermediate layers
    let layer_weights = vec![0.5, 0.3, 0.2];
    let feature_loss = FeatureDistillationLoss::new(layer_weights.clone(), 2.0)?;

    println!("Layer weights: {:?}", layer_weights);
    println!("Distance metric: L2 norm");

    // Extract intermediate features
    let teacher_features = teacher.extract_features(&inputs);
    let student_features = student.extract_features(&inputs);

    let teacher_views: Vec<_> = teacher_features.iter().map(|f| f.view()).collect();
    let student_views: Vec<_> = student_features.iter().map(|f| f.view()).collect();

    // Compute feature matching loss
    let feature_loss_value = feature_loss.compute_feature_loss(&student_views, &teacher_views)?;

    println!("Feature distillation loss: {:.4}", feature_loss_value);
    println!("  → Matches intermediate representations between teacher and student");
    println!("  → Layer 1 weight: {} (early features)", layer_weights[0]);
    println!("  → Layer 2 weight: {} (mid features)", layer_weights[1]);
    println!("  → Layer 3 weight: {} (late features)\n", layer_weights[2]);

    // ============================================================================
    // 3. Attention Transfer
    // ============================================================================
    println!("--- 3. Attention Transfer ---");

    let beta = 2.0; // Power for attention map normalization
    let attention_loss = AttentionTransferLoss::new(beta);

    println!("Beta parameter: {} (attention normalization power)", beta);

    // Simulate attention maps (in practice, these come from attention mechanisms)
    let teacher_attention = array![
        [0.3, 0.5, 0.2, 0.0],
        [0.4, 0.4, 0.1, 0.1],
        [0.2, 0.3, 0.3, 0.2]
    ];

    let student_attention = array![
        [0.35, 0.45, 0.15, 0.05],
        [0.35, 0.45, 0.1, 0.1],
        [0.25, 0.25, 0.3, 0.2]
    ];

    let attention_loss_value = attention_loss
        .compute_attention_loss(&student_attention.view(), &teacher_attention.view())?;

    println!("Attention transfer loss: {:.4}", attention_loss_value);
    println!("  → Transfers attention patterns from teacher to student");
    println!("  → Helps student focus on same important regions\n");

    // ============================================================================
    // 4. Combined Distillation Strategy
    // ============================================================================
    println!("--- 4. Combined Distillation Strategy ---");

    let total_loss = loss_value * 1.0 + feature_loss_value * 0.5 + attention_loss_value * 0.3;

    println!("Combined loss components:");
    println!(
        "  Standard distillation:  {:.4} × 1.0 = {:.4}",
        loss_value,
        loss_value * 1.0
    );
    println!(
        "  Feature distillation:   {:.4} × 0.5 = {:.4}",
        feature_loss_value,
        feature_loss_value * 0.5
    );
    println!(
        "  Attention transfer:     {:.4} × 0.3 = {:.4}",
        attention_loss_value,
        attention_loss_value * 0.3
    );
    println!("  ─────────────────────────────────────");
    println!("  Total combined loss:          {:.4}\n", total_loss);

    // ============================================================================
    // 5. Best Practices and Tips
    // ============================================================================
    println!("=== Best Practices ===");
    println!("1. Temperature Selection:");
    println!("   - Low (1-2): Harder targets, closer to one-hot");
    println!("   - Medium (3-5): Balanced soft targets (recommended)");
    println!("   - High (>5): Very soft targets, more regularization");
    println!();
    println!("2. Alpha (soft/hard weight) Selection:");
    println!("   - High (0.7-0.9): Trust teacher more (when teacher is strong)");
    println!("   - Medium (0.5): Balanced (default choice)");
    println!("   - Low (0.1-0.3): Trust labels more (when teacher is uncertain)");
    println!();
    println!("3. Feature Distillation:");
    println!("   - Match intermediate layers for better transfer");
    println!("   - Weight later layers higher if task-specific knowledge is important");
    println!("   - Weight earlier layers higher for general feature learning");
    println!();
    println!("4. Training Schedule:");
    println!("   - Start with higher temperature (softer targets)");
    println!("   - Gradually decrease temperature during training");
    println!("   - Adjust alpha based on student performance");
    println!();
    println!("5. When to Use:");
    println!("   ✓ Model compression (large → small model)");
    println!("   ✓ Ensemble distillation (many models → one model)");
    println!("   ✓ Cross-architecture transfer");
    println!("   ✓ Domain adaptation with pretrained teacher");
    println!("   ✗ When teacher and student are similar size");
    println!("   ✗ When teacher is poorly trained");

    Ok(())
}
