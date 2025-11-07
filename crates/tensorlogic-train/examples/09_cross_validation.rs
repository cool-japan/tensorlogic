//! Example: Cross-Validation
//!
//! This example demonstrates various cross-validation strategies:
//! - K-Fold cross-validation
//! - Stratified K-Fold
//! - Time Series Split
//! - Leave-One-Out
//!
//! Run with: cargo run --example 09_cross_validation

use std::collections::HashMap;
use tensorlogic_train::{
    CrossValidationResults, CrossValidationSplit, KFold, LeaveOneOut, StratifiedKFold,
    TimeSeriesSplit,
};

fn main() {
    println!("=== Cross-Validation Examples ===\n");

    let n_samples = 20;

    // Example 1: K-Fold Cross-Validation
    println!("1. K-Fold Cross-Validation");
    println!("   Split data into K equally-sized folds\n");

    let kfold = KFold::new(5).unwrap();
    println!("   Configuration: {} folds", kfold.num_splits());
    println!("   Dataset size: {} samples\n", n_samples);

    for fold in 0..3 {
        let (train_idx, val_idx) = kfold.get_split(fold, n_samples).unwrap();

        println!("   Fold {}:", fold);
        println!("     Training: {} samples", train_idx.len());
        println!("     Validation: {} samples", val_idx.len());
        println!("     Val indices: {:?}\n", val_idx);
    }

    // With shuffling
    let kfold_shuffled = KFold::new(5).unwrap().with_shuffle(42);
    println!("   With shuffling (seed=42):");
    let (_train, val) = kfold_shuffled.get_split(0, n_samples).unwrap();
    println!("     Fold 0 validation indices: {:?}\n", val);

    // Example 2: Stratified K-Fold
    println!("2. Stratified K-Fold Cross-Validation");
    println!("   Maintains class distribution in each fold\n");

    // Imbalanced dataset: 12 class-0, 6 class-1, 2 class-2
    let labels = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 12 samples of class 0
        1, 1, 1, 1, 1, 1, // 6 samples of class 1
        2, 2, // 2 samples of class 2
    ];

    println!("   Dataset:");
    println!("     Class 0: 12 samples (60%)");
    println!("     Class 1: 6 samples (30%)");
    println!("     Class 2: 2 samples (10%)");
    println!("     Total: 20 samples\n");

    let stratified = StratifiedKFold::new(5).unwrap();

    println!("   Stratified splits (5 folds):");
    for fold in 0..3 {
        let (_train_idx, val_idx) = stratified.get_stratified_split(fold, &labels).unwrap();

        // Count classes in validation set
        let mut class_counts = HashMap::new();
        for &idx in &val_idx {
            *class_counts.entry(labels[idx]).or_insert(0) += 1;
        }

        println!("   Fold {}:", fold);
        println!("     Validation: {} samples", val_idx.len());
        println!(
            "       Class 0: {} samples",
            class_counts.get(&0).unwrap_or(&0)
        );
        println!(
            "       Class 1: {} samples",
            class_counts.get(&1).unwrap_or(&0)
        );
        println!(
            "       Class 2: {} samples\n",
            class_counts.get(&2).unwrap_or(&0)
        );
    }

    println!("   ✓ Class proportions maintained in each fold\n");

    // Example 3: Time Series Split
    println!("3. Time Series Split");
    println!("   Respects temporal order (no data leakage from future)\n");

    let ts_split = TimeSeriesSplit::new(5).unwrap();

    println!("   Temporal dataset: 30 time steps");
    println!("   Configuration: 5 splits\n");

    for fold in 0..5 {
        let (train_idx, val_idx) = ts_split.get_split(fold, 30).unwrap();

        if !train_idx.is_empty() && !val_idx.is_empty() {
            let train_range = format!(
                "{}-{}",
                train_idx.first().unwrap(),
                train_idx.last().unwrap()
            );
            let val_range = format!("{}-{}", val_idx.first().unwrap(), val_idx.last().unwrap());

            println!("   Fold {}:", fold);
            println!(
                "     Train: steps {} ({} samples)",
                train_range,
                train_idx.len()
            );
            println!("     Val: steps {} ({} samples)", val_range, val_idx.len());
            println!("     ✓ Train always before validation\n");
        }
    }

    // With sliding window
    let ts_split_window = TimeSeriesSplit::new(5).unwrap().with_max_train_size(10);

    println!("   With sliding window (max_train_size=10):");
    for fold in 2..4 {
        let (train_idx, _val_idx) = ts_split_window.get_split(fold, 30).unwrap();

        println!("   Fold {}: Train size = {} (≤10)", fold, train_idx.len());
    }

    // Example 4: Leave-One-Out Cross-Validation
    println!("\n4. Leave-One-Out Cross-Validation (LOO)");
    println!("   Use each sample once as validation\n");

    let loo = LeaveOneOut::new();
    let small_n = 8;

    println!("   Dataset size: {} samples", small_n);
    println!("   Number of folds: {} (= n_samples)\n", small_n);

    for fold in 0..4 {
        let (train_idx, val_idx) = loo.get_split(fold, small_n).unwrap();

        println!("   Fold {}:", fold);
        println!("     Train: {} samples", train_idx.len());
        println!("     Val: 1 sample (index {})", val_idx[0]);
    }

    println!("\n   Use case: Very small datasets (n < 50)");
    println!("   Warning: Computationally expensive for large n\n");

    // Example 5: Cross-Validation Results Aggregation
    println!("5. Cross-Validation Results");
    println!("   Aggregate and analyze results across folds\n");

    let mut cv_results = CrossValidationResults::new();

    // Simulate training across folds
    let fold_scores = [0.85, 0.87, 0.83, 0.86, 0.84];

    println!("   Simulating 5-fold CV:");
    for (fold, &score) in fold_scores.iter().enumerate() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), score);
        metrics.insert("loss".to_string(), 1.0 - score);
        metrics.insert("f1_score".to_string(), score - 0.02);

        cv_results.add_fold(score, metrics);

        println!("     Fold {}: accuracy = {:.3}", fold, score);
    }

    println!("\n   Aggregated Results:");
    println!("     Mean accuracy: {:.4}", cv_results.mean_score());
    println!("     Std accuracy: {:.4}", cv_results.std_score());
    println!(
        "     Mean F1: {:.4}",
        cv_results.mean_metric("f1_score").unwrap()
    );
    println!(
        "     95% CI: [{:.4}, {:.4}]",
        cv_results.mean_score() - 1.96 * cv_results.std_score(),
        cv_results.mean_score() + 1.96 * cv_results.std_score()
    );

    println!("\n=== Practical Workflow ===\n");
    println!("```rust");
    println!("// 1. Choose CV strategy");
    println!("let cv_strategy = KFold::new(5)?.with_shuffle(42);");
    println!("// or");
    println!("let cv_strategy = StratifiedKFold::new(5)?;");
    println!("// or");
    println!("let cv_strategy = TimeSeriesSplit::new(5)?;");
    println!();
    println!("// 2. Initialize results tracker");
    println!("let mut cv_results = CrossValidationResults::new();");
    println!();
    println!("// 3. Run cross-validation loop");
    println!("for fold in 0..cv_strategy.num_splits() {{");
    println!("    // Get train/val split");
    println!("    let (train_idx, val_idx) = cv_strategy.get_split(fold, n_samples)?;");
    println!("    ");
    println!("    // Extract data");
    println!("    let train_data = data.select(Axis(0), &train_idx);");
    println!("    let val_data = data.select(Axis(0), &val_idx);");
    println!("    ");
    println!("    // Train model");
    println!("    let model = train_model(&train_data)?;");
    println!("    ");
    println!("    // Evaluate");
    println!("    let score = evaluate(&model, &val_data);");
    println!("    let metrics = compute_metrics(&model, &val_data);");
    println!("    ");
    println!("    // Record results");
    println!("    cv_results.add_fold(score, metrics);");
    println!("}}");
    println!();
    println!("// 4. Analyze results");
    println!("println!(\"Mean: {{:.4}} ± {{:.4}}\",");
    println!("    cv_results.mean_score(),");
    println!("    cv_results.std_score()");
    println!(");");
    println!("```");

    println!("\n=== Strategy Selection Guide ===\n");
    println!("K-Fold:");
    println!("  • Use for: General-purpose CV");
    println!("  • Pros: Simple, efficient, widely used");
    println!("  • Cons: May not preserve class distribution");
    println!("  • Recommended: 5 or 10 folds\n");

    println!("Stratified K-Fold:");
    println!("  • Use for: Imbalanced classification");
    println!("  • Pros: Maintains class proportions");
    println!("  • Cons: Slightly more complex");
    println!("  • Recommended: Always use for classification\n");

    println!("Time Series Split:");
    println!("  • Use for: Temporal data");
    println!("  • Pros: No data leakage from future");
    println!("  • Cons: Unequal fold sizes");
    println!("  • Recommended: Financial, forecasting tasks\n");

    println!("Leave-One-Out:");
    println!("  • Use for: Very small datasets (n < 50)");
    println!("  • Pros: Maximum training data per fold");
    println!("  • Cons: Computationally expensive");
    println!("  • Recommended: Only when n is very small\n");

    println!("=== Best Practices ===");
    println!("1. Use stratified K-fold for classification (maintains class balance)");
    println!("2. Use time series split for temporal data (prevents data leakage)");
    println!("3. Use 5-10 folds for K-fold (good bias-variance tradeoff)");
    println!("4. Always shuffle data (except time series) to reduce sampling bias");
    println!("5. Report mean ± std dev for transparency");
    println!("6. Use same CV strategy when comparing models");
}
