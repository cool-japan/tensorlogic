//! Example: Ensemble Learning
//!
//! This example demonstrates model ensembling strategies:
//! - Voting ensemble (hard and soft)
//! - Averaging ensemble
//! - Stacking ensemble
//! - Bagging utilities
//!
//! Run with: cargo run --example 10_ensemble_learning

use scirs2_core::ndarray::array;
use tensorlogic_train::{
    AveragingEnsemble, BaggingHelper, Ensemble, LinearModel, StackingEnsemble, VotingEnsemble,
    VotingMode,
};

fn main() {
    println!("=== Ensemble Learning Examples ===\n");

    // Example 1: Voting Ensemble (Hard Voting)
    println!("1. Voting Ensemble - Hard Voting");
    println!("   Combines predictions by majority vote\n");

    // Create three simple models
    let model1 = LinearModel::new(2, 3); // 2 inputs, 3 classes
    let model2 = LinearModel::new(2, 3);
    let model3 = LinearModel::new(2, 3);

    let voting_hard = VotingEnsemble::new(vec![model1, model2, model3], VotingMode::Hard).unwrap();

    println!("   Configuration:");
    println!("     Number of models: {}", voting_hard.num_models());
    println!("     Voting mode: Hard (majority vote)");

    let input = array![[1.0, 2.0], [3.0, 4.0]];
    let predictions = voting_hard.predict(&input).unwrap();

    println!("\n   Input shape: {:?}", input.shape());
    println!("   Output shape: {:?}", predictions.shape());
    println!("   Predictions: {:?}\n", predictions);

    println!("   How it works:");
    println!("     1. Each model makes a prediction");
    println!("     2. Take argmax for each model → predicted class");
    println!("     3. Return class with most votes");
    println!("     4. Result: One-hot encoded prediction\n");

    // Example 2: Voting Ensemble (Soft Voting)
    println!("2. Voting Ensemble - Soft Voting");
    println!("   Combines predictions by averaging probabilities\n");

    let model1 = LinearModel::new(2, 3);
    let model2 = LinearModel::new(2, 3);
    let model3 = LinearModel::new(2, 3);

    let voting_soft = VotingEnsemble::new(vec![model1, model2, model3], VotingMode::Soft).unwrap();

    println!("   Configuration:");
    println!("     Number of models: {}", voting_soft.num_models());
    println!("     Voting mode: Soft (average probabilities)");

    let predictions = voting_soft.predict(&input).unwrap();

    println!("\n   Predictions: {:?}\n", predictions);

    println!("   How it works:");
    println!("     1. Each model outputs probability distribution");
    println!("     2. Average all probability vectors");
    println!("     3. Result: Averaged probability distribution");
    println!("     4. More robust than hard voting\n");

    // Example 3: Weighted Voting
    println!("3. Weighted Voting");
    println!("   Assign different weights to models based on performance\n");

    let model1 = LinearModel::new(2, 3);
    let model2 = LinearModel::new(2, 3);
    let model3 = LinearModel::new(2, 3);

    // Model 1 is best (0.5), model 2 is okay (0.3), model 3 is weak (0.2)
    let weights = vec![0.5, 0.3, 0.2];

    let weighted_voting = VotingEnsemble::new(vec![model1, model2, model3], VotingMode::Soft)
        .unwrap()
        .with_weights(weights)
        .unwrap();

    println!("   Model weights:");
    println!("     Model 1: 0.5 (best performer)");
    println!("     Model 2: 0.3 (good performer)");
    println!("     Model 3: 0.2 (weak performer)");

    let predictions = weighted_voting.predict(&input).unwrap();
    println!("\n   Weighted predictions: {:?}\n", predictions);

    println!("   Benefits:");
    println!("     • Leverages model diversity");
    println!("     • Weights reflect validation performance");
    println!("     • Better than simple averaging\n");

    // Example 4: Averaging Ensemble
    println!("4. Averaging Ensemble");
    println!("   Average predictions from multiple regression models\n");

    let model1 = LinearModel::new(2, 1); // Regression: 1 output
    let model2 = LinearModel::new(2, 1);
    let model3 = LinearModel::new(2, 1);

    let averaging = AveragingEnsemble::new(vec![model1, model2, model3]).unwrap();

    println!("   Configuration:");
    println!("     Number of models: {}", averaging.num_models());
    println!("     Task: Regression");

    let input_reg = array![[1.0, 2.0], [3.0, 4.0]];
    let predictions = averaging.predict(&input_reg).unwrap();

    println!("\n   Input shape: {:?}", input_reg.shape());
    println!("   Output shape: {:?} (regression)", predictions.shape());
    println!("   Predictions: {:?}\n", predictions);

    println!("   Use case:");
    println!("     • Regression tasks");
    println!("     • Reduces variance");
    println!("     • Smooths out individual model errors\n");

    // Example 5: Stacking Ensemble
    println!("5. Stacking Ensemble");
    println!("   Use base models' predictions as features for meta-learner\n");

    // Base models: 2 inputs → 3 outputs each
    let base1 = LinearModel::new(2, 3);
    let base2 = LinearModel::new(2, 3);
    let base3 = LinearModel::new(2, 3);

    // Meta-model: 9 inputs (3 models × 3 outputs) → 3 outputs
    let meta_model = LinearModel::new(9, 3);

    let stacking = StackingEnsemble::new(vec![base1, base2, base3], meta_model).unwrap();

    println!("   Architecture:");
    println!("     Level 1 (Base): 3 models");
    println!("       • Each: 2 inputs → 3 outputs");
    println!("     Level 2 (Meta): 1 model");
    println!("       • Input: 9 features (3×3 concatenated predictions)");
    println!("       • Output: 3 classes");

    let predictions = stacking.predict(&input).unwrap();

    println!("\n   Input shape: {:?}", input.shape());
    println!("   Meta-features: [batch, 9] (concatenated base predictions)");
    println!("   Output shape: {:?}", predictions.shape());
    println!("\n   Advantages:");
    println!("     • Learns how to best combine base models");
    println!("     • Often outperforms voting/averaging");
    println!("     • Can correct systematic errors\n");

    println!("   Training procedure:");
    println!("     1. Split data: train (80%) + holdout (20%)");
    println!("     2. Train base models on train set");
    println!("     3. Generate predictions on holdout set");
    println!("     4. Train meta-model on (holdout_predictions, holdout_labels)");
    println!("     5. For inference: base_models → meta_model\n");

    // Example 6: Bagging (Bootstrap Aggregating)
    println!("6. Bagging Utilities");
    println!("   Generate bootstrap samples for training diverse models\n");

    let bagging = BaggingHelper::new(10, 42).unwrap();

    println!("   Configuration:");
    println!("     Number of estimators: {}", bagging.n_estimators);
    println!("     Random seed: {}", bagging.random_seed);

    let n_samples = 20;

    println!("\n   Bootstrap samples:");
    for i in 0..3 {
        let bootstrap_indices = bagging.generate_bootstrap_indices(n_samples, i);
        let oob_indices = bagging.get_oob_indices(n_samples, &bootstrap_indices);

        println!("   Estimator {}:", i);
        println!("     Bootstrap size: {} samples", bootstrap_indices.len());
        println!("     Out-of-bag size: {} samples", oob_indices.len());
        println!("     OOB indices: {:?}", oob_indices);
    }

    println!("\n   How bagging works:");
    println!("     1. Create bootstrap sample (random sampling with replacement)");
    println!("     2. Train model on bootstrap sample");
    println!("     3. Some samples left out (out-of-bag)");
    println!("     4. Use OOB samples for validation (no separate val set needed)");
    println!("     5. Repeat for N estimators");
    println!("     6. Combine predictions (voting/averaging)\n");

    println!("=== Practical Workflow ===\n");
    println!("Strategy 1: Voting Ensemble");
    println!("```rust");
    println!("// Train diverse models");
    println!("let model1 = train_with_config(config1)?;");
    println!("let model2 = train_with_config(config2)?;");
    println!("let model3 = train_with_config(config3)?;");
    println!();
    println!("// Create ensemble");
    println!("let ensemble = VotingEnsemble::new(");
    println!("    vec![model1, model2, model3],");
    println!("    VotingMode::Soft");
    println!(")?;");
    println!();
    println!("// Make predictions");
    println!("let predictions = ensemble.predict(&test_data)?;");
    println!("```\n");

    println!("Strategy 2: Stacking Ensemble");
    println!("```rust");
    println!("// Step 1: Train base models on train set");
    println!("let base1 = train_model(&train_data)?;");
    println!("let base2 = train_model(&train_data)?;");
    println!("let base3 = train_model(&train_data)?;");
    println!();
    println!("// Step 2: Generate meta-features on validation set");
    println!("let meta_features = generate_meta_features(");
    println!("    &[base1, base2, base3],");
    println!("    &val_data");
    println!(")?;");
    println!();
    println!("// Step 3: Train meta-model");
    println!("let meta_model = train_meta_model(&meta_features, &val_labels)?;");
    println!();
    println!("// Step 4: Create stacking ensemble");
    println!("let stacking = StackingEnsemble::new(");
    println!("    vec![base1, base2, base3],");
    println!("    meta_model");
    println!(")?;");
    println!("```\n");

    println!("Strategy 3: Bagging");
    println!("```rust");
    println!("let bagging = BaggingHelper::new(10, 42)?;");
    println!("let mut models = Vec::new();");
    println!();
    println!("for i in 0..10 {{");
    println!("    // Generate bootstrap sample");
    println!("    let indices = bagging.generate_bootstrap_indices(n_samples, i);");
    println!("    let boot_data = data.select(Axis(0), &indices);");
    println!("    ");
    println!("    // Train model on bootstrap");
    println!("    let model = train_model(&boot_data)?;");
    println!("    models.push(model);");
    println!("}}");
    println!();
    println!("// Combine with voting");
    println!("let ensemble = VotingEnsemble::new(models, VotingMode::Soft)?;");
    println!("```");

    println!("\n=== When to Use Each Strategy ===\n");
    println!("Voting (Hard):");
    println!("  • Simple and interpretable");
    println!("  • Good when models have similar accuracy");
    println!("  • Classification only");
    println!("  • Fast inference\n");

    println!("Voting (Soft):");
    println!("  • Better than hard voting");
    println!("  • Leverages prediction confidence");
    println!("  • Requires probabilistic outputs");
    println!("  • Recommended for most classification tasks\n");

    println!("Averaging:");
    println!("  • Regression tasks");
    println!("  • Reduces variance");
    println!("  • Simple and effective");
    println!("  • No hyperparameters\n");

    println!("Stacking:");
    println!("  • Best potential performance");
    println!("  • Learns optimal combination");
    println!("  • Requires additional validation set");
    println!("  • More complex to implement\n");

    println!("Bagging:");
    println!("  • Reduces variance");
    println!("  • Works with high-variance models (decision trees)");
    println!("  • Provides OOB validation");
    println!("  • Training can be parallelized\n");

    println!("=== Key Insights ===");
    println!("1. Ensembles work best with diverse models");
    println!("2. Diversity trumps individual accuracy");
    println!("3. 5-10 models usually sufficient");
    println!("4. Soft voting > hard voting (when applicable)");
    println!("5. Stacking > voting (but more complex)");
    println!("6. Always validate ensemble vs. best single model");
}
