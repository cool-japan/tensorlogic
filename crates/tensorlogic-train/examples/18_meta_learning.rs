//! Example: Meta-Learning (Learning to Learn)
//!
//! This example demonstrates meta-learning algorithms (MAML and Reptile) that
//! learn model initializations enabling rapid adaptation to new tasks.
//!
//! Run with: cargo run --example 18_meta_learning

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use tensorlogic_train::{
    MAMLConfig, MetaLearner, MetaStats, MetaTask, Reptile, ReptileConfig, MAML,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Meta-Learning Examples ===\n");

    // Example 1: Understanding Meta-Learning
    println!("1. What is Meta-Learning?");
    println!("   Meta-learning (or 'learning to learn') optimizes for rapid adaptation.\n");
    println!("   Traditional ML: Train on Task A → Model for Task A");
    println!("   Meta-Learning:  Train on Tasks A,B,C,... → Initialization that");
    println!("                   quickly adapts to Task D with few examples\n");

    // Example 2: Creating Meta-Learning Tasks
    println!("2. Meta-Learning Task Structure:");
    println!("   Each task has support set (for adaptation) and query set (for evaluation)\n");

    // Create a simple task: learn to classify 2D points
    let support_x = Array2::from_shape_vec(
        (5, 2),
        vec![1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 5.0, 5.0, 5.1, 4.9],
    )?;
    let support_y = Array2::from_shape_vec((5, 1), vec![0.0, 0.0, 0.0, 1.0, 1.0])?;
    let query_x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 0.95, 1.05, 5.0, 5.0, 4.9, 5.1])?;
    let query_y = Array2::from_shape_vec((4, 1), vec![0.0, 0.0, 1.0, 1.0])?;

    let task = MetaTask::new(support_x, support_y, query_x, query_y)?;

    println!("   ✓ Task created");
    println!("     Support set: {} examples", task.support_size());
    println!("     Query set: {} examples", task.query_size());

    // Example 3: MAML Configuration
    println!("\n3. MAML (Model-Agnostic Meta-Learning):");
    println!("   Learns initialization θ* for rapid adaptation via gradient descent\n");

    let maml_config = MAMLConfig {
        inner_steps: 5,    // Adaptation steps per task
        inner_lr: 0.01,    // Learning rate for adaptation
        outer_lr: 0.001,   // Learning rate for meta-update
        first_order: true, // Use first-order approximation (faster)
    };

    println!("   Configuration:");
    println!("     Inner steps (adaptation): {}", maml_config.inner_steps);
    println!("     Inner LR: {}", maml_config.inner_lr);
    println!("     Outer LR: {}", maml_config.outer_lr);
    println!(
        "     First-order: {} ({})",
        maml_config.first_order,
        if maml_config.first_order {
            "faster"
        } else {
            "more accurate"
        }
    );

    let maml = MAML::new(maml_config);
    println!("\n   ✓ MAML meta-learner created");

    // Example 4: Reptile Configuration
    println!("\n4. Reptile Algorithm:");
    println!("   Simpler first-order alternative to MAML\n");

    let reptile_config = ReptileConfig {
        inner_steps: 10, // More steps for adaptation
        inner_lr: 0.01,  // Learning rate for adaptation
        outer_lr: 0.1,   // Meta-update step size
    };

    println!("   Configuration:");
    println!("     Inner steps: {}", reptile_config.inner_steps);
    println!("     Inner LR: {}", reptile_config.inner_lr);
    println!("     Outer LR: {}", reptile_config.outer_lr);

    let reptile = Reptile::new(reptile_config);
    println!("\n   ✓ Reptile meta-learner created");

    // Example 5: Meta-Training Simulation
    println!("\n5. Meta-Training Process:");
    println!("   (Simplified simulation with dummy parameters)\n");

    // Initialize meta-parameters
    let mut meta_params = HashMap::new();
    meta_params.insert("weights".to_string(), Array1::zeros(2));

    println!(
        "   Initial meta-parameters: {:?}",
        meta_params.get("weights")
    );

    // Create a batch of tasks
    let tasks = vec![
        create_dummy_task(2)?,
        create_dummy_task(2)?,
        create_dummy_task(2)?,
    ];

    println!("   Task batch size: {}", tasks.len());

    // Perform MAML meta-step
    println!("\n   Performing MAML meta-step...");
    let (updated_params_maml, loss_maml) = maml.meta_step(&tasks, &meta_params)?;
    println!("     ✓ Meta-loss: {:.4}", loss_maml);
    println!(
        "     ✓ Updated parameters: {:?}",
        updated_params_maml.get("weights")
    );

    // Perform Reptile meta-step
    println!("\n   Performing Reptile meta-step...");
    let (updated_params_reptile, loss_reptile) = reptile.meta_step(&tasks, &meta_params)?;
    println!("     ✓ Meta-loss: {:.4}", loss_reptile);
    println!(
        "     ✓ Updated parameters: {:?}",
        updated_params_reptile.get("weights")
    );

    // Example 6: Task Adaptation
    println!("\n6. Task Adaptation (Inner Loop):");
    println!("   Given meta-parameters, adapt to a specific task\n");

    let new_task = create_dummy_task(2)?;

    println!("   MAML adaptation:");
    let adapted_maml = maml.adapt(&new_task, &meta_params)?;
    println!(
        "     ✓ Adapted parameters: {:?}",
        adapted_maml.get("weights")
    );

    println!("\n   Reptile adaptation:");
    let adapted_reptile = reptile.adapt(&new_task, &meta_params)?;
    println!(
        "     ✓ Adapted parameters: {:?}",
        adapted_reptile.get("weights")
    );

    // Example 7: Meta-Statistics Tracking
    println!("\n7. Meta-Training Statistics:");
    println!("   Track progress over meta-training iterations\n");

    let mut stats = MetaStats::new();

    // Simulate meta-training
    println!("   Simulating 20 meta-training iterations...");
    for i in 1..=20 {
        // Simulate decreasing loss
        let loss = 1.0 - (i as f64 * 0.04);
        stats.record_meta_step(loss);

        if i % 5 == 0 {
            println!("     Iteration {}: loss = {:.3}", i, loss);
        }
    }

    println!("\n   Statistics:");
    println!("     Total iterations: {}", stats.iterations);
    println!("     Avg loss (last 5): {:.3}", stats.avg_meta_loss(5));
    println!("     Avg loss (last 10): {:.3}", stats.avg_meta_loss(10));
    println!("     Is improving: {}", stats.is_improving(5));

    // Example 8: Comparison of MAML vs Reptile
    println!("\n8. MAML vs Reptile Comparison:\n");

    println!("   MAML:");
    println!("     ✓ More accurate gradients (includes second-order terms)");
    println!("     ✓ Better theoretical guarantees");
    println!("     ✗ Slower (requires backprop through adaptation)");
    println!("     ✗ More memory intensive");
    println!("     Use when: Accuracy is critical, computational budget allows\n");

    println!("   Reptile:");
    println!("     ✓ Simpler algorithm (first-order only)");
    println!("     ✓ Faster training");
    println!("     ✓ Less memory usage");
    println!("     ✗ Approximates MAML gradients");
    println!("     Use when: Speed matters, many meta-training tasks\n");

    println!("   First-order MAML:");
    println!("     ✓ Middle ground between full MAML and Reptile");
    println!("     ✓ Faster than full MAML");
    println!("     ✓ Often performs similarly to full MAML in practice");

    // Example 9: Practical Workflow
    println!("\n9. Practical Meta-Learning Workflow:\n");

    println!("   Step 1: Collect diverse tasks");
    println!("           - Each task should have support and query examples");
    println!("           - Tasks should be related but distinct\n");

    println!("   Step 2: Choose algorithm");
    println!("           - MAML: When accuracy is critical");
    println!("           - Reptile: When training many tasks quickly");
    println!("           - First-order MAML: Good default choice\n");

    println!("   Step 3: Configure hyperparameters");
    println!("           - Inner steps: 1-10 (how much to adapt)");
    println!("           - Inner LR: 0.001-0.1 (adaptation learning rate)");
    println!("           - Outer LR: 0.0001-0.01 (meta-learning rate)\n");

    println!("   Step 4: Meta-training loop");
    println!("           for meta_iteration in 1..max_iterations {{");
    println!("               // Sample batch of tasks");
    println!("               tasks = sample_task_batch()");
    println!();
    println!("               // Meta-update");
    println!("               (params, loss) = meta_learner.meta_step(tasks, params)");
    println!();
    println!("               // Track progress");
    println!("               stats.record_meta_step(loss)");
    println!("           }}\n");

    println!("   Step 5: Evaluation on new tasks");
    println!("           - Given meta-learned parameters");
    println!("           - Adapt to new task with few examples");
    println!("           - Evaluate on query set");

    // Example 10: Use Cases
    println!("\n10. Real-World Applications:\n");

    println!("    Personalization:");
    println!("      • Learn user preferences from few interactions");
    println!("      • Adapt recommender systems to new users");
    println!("      • Personalized content generation\n");

    println!("    Few-Shot Classification:");
    println!("      • New product categories with few examples");
    println!("      • Rare disease diagnosis");
    println!("      • Wildlife species identification\n");

    println!("    Robotics:");
    println!("      • Adapt to new objects quickly");
    println!("      • Transfer skills to new environments");
    println!("      • Learn from demonstrations\n");

    println!("    Drug Discovery:");
    println!("      • Predict properties of new molecules");
    println!("      • Optimize with limited experimental data");
    println!("      • Transfer knowledge across similar compounds\n");

    println!("    Neural Architecture Search:");
    println!("      • Learn good initializations for network weights");
    println!("      • Quickly evaluate candidate architectures");
    println!("      • Warm-start hyperparameter optimization");

    println!("\n=== Summary ===");
    println!("Meta-learning enables:");
    println!("  • Rapid adaptation to new tasks (few gradient steps)");
    println!("  • Learning from diverse task distributions");
    println!("  • Transfer learning with minimal fine-tuning");
    println!("  • Efficient use of limited data per task");
    println!();
    println!("Key algorithms:");
    println!("  • MAML: Accurate but computationally expensive");
    println!("  • Reptile: Fast and simple first-order approximation");
    println!("  • First-order MAML: Practical middle ground");

    Ok(())
}

/// Helper function to create dummy tasks for examples
fn create_dummy_task(feature_dim: usize) -> Result<MetaTask, Box<dyn std::error::Error>> {
    let support_x = Array2::zeros((5, feature_dim));
    let support_y = Array2::zeros((5, 1));
    let query_x = Array2::zeros((15, feature_dim));
    let query_y = Array2::zeros((15, 1));

    Ok(MetaTask::new(support_x, support_y, query_x, query_y)?)
}
