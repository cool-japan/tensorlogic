//! # Advanced Integration Example
//!
//! This example demonstrates how to combine multiple advanced training features
//! in a realistic end-to-end workflow:
//!
//! 1. Hyperparameter optimization with random search
//! 2. Cross-validation for robust evaluation
//! 3. Curriculum learning for progressive training (conceptual demonstration)
//! 4. Ensemble learning by training multiple models
//!
//! This represents a production-grade training pipeline.

use scirs2_core::ndarray::{s, Array1, Array2};
use std::collections::HashMap;
use tensorlogic_train::*;

/// Helper to create classification data with known difficulty levels
fn create_data_with_difficulty(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let mut data = Array2::zeros((n_samples, n_features));
    let mut targets = Array2::zeros((n_samples, n_classes));
    let mut difficulties = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        let base_value = class as f64 * 2.0;

        // Create features with varying difficulty
        for j in 0..n_features {
            let noise = (i as f64 * 0.01).sin() * 0.5;
            data[[i, j]] = (base_value + j as f64 * 0.2 + noise) / (n_features as f64);
        }

        targets[[i, class]] = 1.0;

        // Difficulty: samples near class boundaries are harder
        let boundary_dist =
            ((i % (n_samples / n_classes)) as f64 - (n_samples / n_classes) as f64 / 2.0).abs();
        difficulties[i] = 1.0 - boundary_dist / (n_samples / n_classes) as f64;
    }

    (data, targets, difficulties)
}

/// Evaluate model accuracy
fn compute_accuracy(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let mut correct = 0;
    for i in 0..predictions.nrows() {
        let pred_class = predictions
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let true_class = targets
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if pred_class == true_class {
            correct += 1;
        }
    }

    correct as f64 / predictions.nrows() as f64
}

/// Simple forward pass for evaluation
fn forward(data: &Array2<f64>, weights: &Array2<f64>, bias: &Array2<f64>) -> Array2<f64> {
    data.dot(weights) + bias
}

fn main() -> TrainResult<()> {
    println!("=== Advanced Integration Example ===\n");
    println!("Demonstrating: Hyperparameter Optimization + Cross-Validation + Ensemble Learning\n");

    // Generate dataset
    const N_SAMPLES: usize = 200;
    const N_FEATURES: usize = 8;
    const N_CLASSES: usize = 3;

    let (data, targets, _difficulties) =
        create_data_with_difficulty(N_SAMPLES, N_FEATURES, N_CLASSES);

    // Split train/val
    let val_split = (N_SAMPLES as f64 * 0.8) as usize;
    let train_data = data.slice(s![..val_split, ..]).to_owned();
    let train_targets = targets.slice(s![..val_split, ..]).to_owned();
    let val_data = data.slice(s![val_split.., ..]).to_owned();
    let val_targets = targets.slice(s![val_split.., ..]).to_owned();

    println!(
        "Dataset: {} train, {} val samples\n",
        train_data.nrows(),
        val_data.nrows()
    );

    // ============================================================================
    // PHASE 1: Hyperparameter Optimization
    // ============================================================================
    println!("--- PHASE 1: Hyperparameter Optimization ---");
    println!("Using random search to find optimal learning rate and batch size...\n");

    let mut param_space = HashMap::new();
    param_space.insert(
        "learning_rate".to_string(),
        HyperparamSpace::LogUniform {
            min: 1e-4,
            max: 1e-1,
        },
    );
    param_space.insert(
        "batch_size".to_string(),
        HyperparamSpace::Discrete(vec![
            HyperparamValue::Int(16),
            HyperparamValue::Int(32),
            HyperparamValue::Int(64),
        ]),
    );

    let mut random_search = RandomSearch::new(param_space, 6, 42);
    let configs = random_search.generate_configs();

    let mut best_config = configs[0].clone();
    let mut best_score = f64::NEG_INFINITY;

    for config in configs.iter() {
        let lr = config.get("learning_rate").unwrap().as_float().unwrap();
        let batch_size = config.get("batch_size").unwrap().as_int().unwrap() as usize;

        // Quick evaluation
        let trainer_config = TrainerConfig {
            num_epochs: 10,
            batch_config: BatchConfig {
                batch_size,
                shuffle: true,
                ..Default::default()
            },
            validate_every_epoch: false,
            ..Default::default()
        };

        let loss = Box::new(CrossEntropyLoss::default());
        let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig {
            learning_rate: lr,
            ..Default::default()
        }));

        let mut trainer = Trainer::new(trainer_config, loss, optimizer);

        let mut params = HashMap::new();
        params.insert(
            "weights".to_string(),
            Array2::from_elem((N_FEATURES, N_CLASSES), 0.01),
        );
        params.insert("bias".to_string(), Array2::zeros((1, N_CLASSES)));

        let history = trainer.train(
            &train_data.view(),
            &train_targets.view(),
            Some(&val_data.view()),
            Some(&val_targets.view()),
            &mut params,
        )?;

        let val_loss = history.val_loss.last().copied().unwrap_or(f64::INFINITY);
        let score = -val_loss;

        println!(
            "  lr={:.6}, batch={}: val_loss={:.4}",
            lr, batch_size, val_loss
        );

        let result = HyperparamResult {
            config: config.clone(),
            score,
            metrics: HashMap::new(),
        };
        random_search.add_result(result);

        if score > best_score {
            best_score = score;
            best_config = config.clone();
        }
    }

    let best_lr = best_config
        .get("learning_rate")
        .unwrap()
        .as_float()
        .unwrap();
    let best_batch = best_config.get("batch_size").unwrap().as_int().unwrap() as usize;

    println!("\n✓ Best config: lr={:.6}, batch={}\n", best_lr, best_batch);

    // ============================================================================
    // PHASE 2: Cross-Validation
    // ============================================================================
    println!("--- PHASE 2: Cross-Validation ---");
    println!("Performing 5-fold cross-validation with best hyperparameters...\n");

    let k_fold = KFold::new(5)?;
    let mut fold_scores = Vec::new();

    for fold in 0..k_fold.num_splits() {
        let (train_idx, val_idx) = k_fold.get_split(fold, train_data.nrows())?;

        let fold_train_data = train_data.select(scirs2_core::ndarray::Axis(0), &train_idx);
        let fold_train_targets = train_targets.select(scirs2_core::ndarray::Axis(0), &train_idx);
        let fold_val_data = train_data.select(scirs2_core::ndarray::Axis(0), &val_idx);
        let fold_val_targets = train_targets.select(scirs2_core::ndarray::Axis(0), &val_idx);

        let trainer_config = TrainerConfig {
            num_epochs: 20,
            batch_config: BatchConfig {
                batch_size: best_batch,
                shuffle: true,
                ..Default::default()
            },
            validate_every_epoch: false,
            ..Default::default()
        };

        let loss = Box::new(CrossEntropyLoss::default());
        let optimizer = Box::new(AdamWOptimizer::new(OptimizerConfig {
            learning_rate: best_lr,
            ..Default::default()
        }));

        let mut trainer = Trainer::new(trainer_config, loss, optimizer);

        let mut params = HashMap::new();
        params.insert(
            "weights".to_string(),
            Array2::from_elem((N_FEATURES, N_CLASSES), 0.01),
        );
        params.insert("bias".to_string(), Array2::zeros((1, N_CLASSES)));

        trainer.train(
            &fold_train_data.view(),
            &fold_train_targets.view(),
            Some(&fold_val_data.view()),
            Some(&fold_val_targets.view()),
            &mut params,
        )?;

        // Evaluate
        let predictions = forward(
            &fold_val_data,
            params.get("weights").unwrap(),
            params.get("bias").unwrap(),
        );
        let acc = compute_accuracy(&predictions, &fold_val_targets);
        fold_scores.push(acc);

        println!("  Fold {}: accuracy = {:.4}", fold + 1, acc);
    }

    let mut cv_results = CrossValidationResults::new();
    for score in fold_scores {
        cv_results.add_fold(score, HashMap::new());
    }
    println!(
        "\n✓ CV Results: {:.4} ± {:.4}\n",
        cv_results.mean_score(),
        cv_results.std_score()
    );

    // ============================================================================
    // PHASE 3: Ensemble Learning
    // ============================================================================
    println!("--- PHASE 3: Ensemble Learning ---");
    println!("Training ensemble of 5 models with bagging...\n");

    let n_models = 5;
    let mut ensemble_params = Vec::new();
    let bagging = BaggingHelper::new(n_models, 42)?;

    for i in 0..n_models {
        println!("  Training model {} / {}...", i + 1, n_models);

        // Generate bootstrap sample
        let bootstrap_idx = bagging.generate_bootstrap_indices(train_data.nrows(), i);
        let boot_data = train_data.select(scirs2_core::ndarray::Axis(0), &bootstrap_idx);
        let boot_targets = train_targets.select(scirs2_core::ndarray::Axis(0), &bootstrap_idx);

        let trainer_config = TrainerConfig {
            num_epochs: 25,
            batch_config: BatchConfig {
                batch_size: best_batch,
                shuffle: true,
                ..Default::default()
            },
            validate_every_epoch: false,
            ..Default::default()
        };

        let loss = Box::new(CrossEntropyLoss::default());
        let optimizer = Box::new(AdamWOptimizer::new(OptimizerConfig {
            learning_rate: best_lr,
            ..Default::default()
        }));

        let mut trainer = Trainer::new(trainer_config, loss, optimizer);

        let mut params = HashMap::new();
        params.insert(
            "weights".to_string(),
            Array2::from_elem((N_FEATURES, N_CLASSES), 0.01),
        );
        params.insert("bias".to_string(), Array2::zeros((1, N_CLASSES)));

        trainer.train(
            &boot_data.view(),
            &boot_targets.view(),
            Some(&val_data.view()),
            Some(&val_targets.view()),
            &mut params,
        )?;

        ensemble_params.push(params);
    }

    // Ensemble prediction (soft voting)
    let mut ensemble_predictions = Array2::zeros(val_data.dim());
    for params in &ensemble_params {
        let pred = forward(
            &val_data,
            params.get("weights").unwrap(),
            params.get("bias").unwrap(),
        );
        ensemble_predictions = &ensemble_predictions + &pred;
    }
    ensemble_predictions = &ensemble_predictions / (n_models as f64);

    let ensemble_acc = compute_accuracy(&ensemble_predictions, &val_targets);

    // Individual model accuracy (for comparison)
    let single_pred = forward(
        &val_data,
        ensemble_params[0].get("weights").unwrap(),
        ensemble_params[0].get("bias").unwrap(),
    );
    let single_acc = compute_accuracy(&single_pred, &val_targets);

    println!("\n✓ Ensemble Results:");
    println!("  Number of models: {}", n_models);
    println!("  Single model accuracy: {:.4}", single_acc);
    println!("  Ensemble accuracy: {:.4}", ensemble_acc);
    println!(
        "  Improvement: {:.2}%\n",
        (ensemble_acc - single_acc) * 100.0
    );

    // ============================================================================
    // FINAL SUMMARY
    // ============================================================================
    println!("=== FINAL SUMMARY ===\n");
    println!("Complete pipeline results:");
    println!(
        "  1. Hyperparameter optimization: lr={:.6}, batch={}",
        best_lr, best_batch
    );
    println!(
        "  2. Cross-validation:            {:.4} ± {:.4}",
        cv_results.mean_score(),
        cv_results.std_score()
    );
    println!(
        "  3. Ensemble accuracy:           {:.4} (+{:.2}% vs single model)",
        ensemble_acc,
        (ensemble_acc - single_acc) * 100.0
    );
    println!("\n✓ Advanced integration example completed successfully!");

    Ok(())
}
