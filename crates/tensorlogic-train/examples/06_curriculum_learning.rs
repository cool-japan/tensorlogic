//! Example: Curriculum Learning
//!
//! This example demonstrates how to use curriculum learning strategies
//! to progressively train models from easy to difficult samples.
//!
//! Run with: cargo run --example 06_curriculum_learning

use scirs2_core::ndarray::array;
use tensorlogic_train::{
    CompetenceCurriculum, CurriculumManager, CurriculumStrategy, ExponentialCurriculum,
    LinearCurriculum, SelfPacedCurriculum,
};

fn main() {
    println!("=== Curriculum Learning Examples ===\n");

    // Sample dataset with varying difficulty
    let data = array![
        [1.0, 1.0],   // Easy sample
        [2.0, 2.0],   // Easy sample
        [5.0, 5.0],   // Medium sample
        [8.0, 8.0],   // Hard sample
        [10.0, 10.0]  // Very hard sample
    ];

    let labels = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];

    // Example 1: Linear Curriculum
    println!("1. Linear Curriculum (gradual difficulty increase)");
    println!("   Starts with 20% easiest samples, linearly grows to 100%\n");

    let linear_curriculum = LinearCurriculum::new(0.2).unwrap();

    // Simulate model predictions (entropy-based difficulty)
    let predictions = array![
        [0.9, 0.1],   // Low entropy = easy
        [0.85, 0.15], // Low entropy = easy
        [0.6, 0.4],   // Medium entropy
        [0.5, 0.5],   // High entropy = hard
        [0.45, 0.55]  // High entropy = hard
    ];

    let difficulties = linear_curriculum
        .compute_difficulty(&data, &labels, Some(&predictions))
        .unwrap();

    println!("   Difficulty scores: {:?}", difficulties);

    // Epoch 0: Select easiest 20%
    let selected = linear_curriculum
        .select_samples(0, 10, &difficulties.view())
        .unwrap();
    println!("   Epoch 0 (20%): Selected samples {:?}", selected);

    // Epoch 5: Select ~60%
    let selected = linear_curriculum
        .select_samples(5, 10, &difficulties.view())
        .unwrap();
    println!("   Epoch 5 (60%): Selected samples {:?}", selected);

    // Epoch 9: Select all samples
    let selected = linear_curriculum
        .select_samples(9, 10, &difficulties.view())
        .unwrap();
    println!("   Epoch 9 (100%): Selected samples {:?}\n", selected);

    // Example 2: Exponential Curriculum
    println!("2. Exponential Curriculum (rapid growth)");
    println!("   Fast ramp-up using exponential schedule\n");

    let exp_curriculum = ExponentialCurriculum::new(0.1, 2.0).unwrap();

    let selected_exp_0 = exp_curriculum
        .select_samples(0, 10, &difficulties.view())
        .unwrap();
    println!("   Epoch 0: Selected {} samples", selected_exp_0.len());

    let selected_exp_5 = exp_curriculum
        .select_samples(5, 10, &difficulties.view())
        .unwrap();
    println!("   Epoch 5: Selected {} samples", selected_exp_5.len());

    let selected_exp_9 = exp_curriculum
        .select_samples(9, 10, &difficulties.view())
        .unwrap();
    println!("   Epoch 9: Selected {} samples\n", selected_exp_9.len());

    // Example 3: Self-Paced Learning
    println!("3. Self-Paced Learning (model-driven pace)");
    println!("   Model decides which samples it's ready to learn from\n");

    let self_paced = SelfPacedCurriculum::new(1.0, 0.5).unwrap();

    // Compute difficulty based on loss (not just entropy)
    let sp_difficulties = self_paced
        .compute_difficulty(&data, &labels, Some(&predictions))
        .unwrap();

    let selected_sp = self_paced
        .select_samples(0, 10, &sp_difficulties.view())
        .unwrap();
    println!("   Selected samples (difficulty < 0.5): {:?}", selected_sp);
    println!("   Number selected: {}\n", selected_sp.len());

    // Example 4: Competence-Based Curriculum
    println!("4. Competence-Based Curriculum (adaptive difficulty)");
    println!("   Adapts to model's improving competence level\n");

    let competence_curriculum = CompetenceCurriculum::new(0.3, 0.1).unwrap();

    // Normalize difficulties to [0, 1]
    let comp_difficulties = competence_curriculum
        .compute_difficulty(&data, &labels, Some(&predictions))
        .unwrap();

    println!("   Normalized difficulties: {:?}", comp_difficulties);

    // Epoch 0: Competence = 0.3
    let selected_comp_0 = competence_curriculum
        .select_samples(0, 10, &comp_difficulties.view())
        .unwrap();
    println!(
        "   Epoch 0 (competence=0.3): Selected {} samples",
        selected_comp_0.len()
    );

    // Epoch 5: Competence = 0.8
    let selected_comp_5 = competence_curriculum
        .select_samples(5, 10, &comp_difficulties.view())
        .unwrap();
    println!(
        "   Epoch 5 (competence=0.8): Selected {} samples\n",
        selected_comp_5.len()
    );

    // Example 5: Curriculum Manager (Stateful Management)
    println!("5. Curriculum Manager (state management)");
    println!("   Manages curriculum state across training\n");

    let mut manager = CurriculumManager::new(LinearCurriculum::default());

    // Cache difficulty scores
    manager
        .compute_difficulty("train", &data, &labels, Some(&predictions))
        .unwrap();

    // Simulate training loop
    for epoch in 0..5 {
        manager.set_epoch(epoch);

        let selected_indices = manager.get_selected_samples("train", 10).unwrap();

        println!(
            "   Epoch {}: Training with {} samples (indices: {:?})",
            epoch,
            selected_indices.len(),
            selected_indices
        );
    }

    println!("\n=== Training Workflow Example ===\n");
    println!("Typical usage in training loop:");
    println!("```rust");
    println!("let mut manager = CurriculumManager::new(LinearCurriculum::new(0.2)?);");
    println!("manager.compute_difficulty(\"train\", &data, &labels, Some(&predictions))?;");
    println!();
    println!("for epoch in 0..num_epochs {{");
    println!("    manager.set_epoch(epoch);");
    println!("    let indices = manager.get_selected_samples(\"train\", num_epochs)?;");
    println!("    ");
    println!("    // Train only on selected samples");
    println!("    let batch_data = data.select(Axis(0), &indices);");
    println!("    let batch_labels = labels.select(Axis(0), &indices);");
    println!("    ");
    println!("    // ... training step ...");
    println!("}}");
    println!("```");

    println!("\n=== Key Takeaways ===");
    println!("1. LinearCurriculum: Best for gradual, predictable difficulty progression");
    println!("2. ExponentialCurriculum: Fast ramp-up, good for quick convergence");
    println!("3. SelfPacedCurriculum: Let the model decide when it's ready");
    println!("4. CompetenceCurriculum: Adaptive to model's current capability");
    println!("5. Use CurriculumManager for stateful management across epochs");
}
