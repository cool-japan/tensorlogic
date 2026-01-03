//! Bayesian Optimization for Hyperparameter Tuning
//!
//! This example demonstrates how to use Bayesian Optimization to efficiently
//! search for optimal hyperparameters using Gaussian Processes and acquisition functions.
//!
//! Run with: cargo run --example 21_bayesian_optimization

use std::collections::HashMap;
use tensorlogic_train::*;

/// Simulate training a model and return a validation metric.
///
/// This is a toy objective function that simulates the performance of a model
/// with given hyperparameters. In practice, this would involve actual training.
fn objective_function(lr: f64, batch_size: i64, dropout: f64) -> f64 {
    // Simulate an objective function with a known optimum
    // Best params are approximately: lr=0.01, batch_size=64, dropout=0.2
    let lr_score = -((lr - 0.01).powi(2)) / 0.001;
    let batch_score = -((batch_size as f64 - 64.0).powi(2)) / 500.0;
    let dropout_score = -((dropout - 0.2).powi(2)) / 0.05;

    // Combined score (normalized to [0, 1])
    let score = (lr_score + batch_score + dropout_score + 10.0) / 12.0;

    // Add some noise to make it more realistic
    score + (lr * batch_size as f64 * dropout).sin() * 0.05
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      Bayesian Optimization for Hyperparameter Tuning        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ============================================================================
    // 1. Define hyperparameter search space
    // ============================================================================

    println!("📋 Defining hyperparameter search space...\n");

    let mut param_space = HashMap::new();

    // Learning rate: log-uniform from 1e-4 to 1e-1
    param_space.insert(
        "lr".to_string(),
        HyperparamSpace::log_uniform(1e-4, 1e-1).unwrap(),
    );

    // Batch size: integer from 16 to 128
    param_space.insert(
        "batch_size".to_string(),
        HyperparamSpace::int_range(16, 128).unwrap(),
    );

    // Dropout: continuous from 0.0 to 0.5
    param_space.insert(
        "dropout".to_string(),
        HyperparamSpace::continuous(0.0, 0.5).unwrap(),
    );

    println!("Search space:");
    println!("  • lr: log-uniform [1e-4, 1e-1]");
    println!("  • batch_size: int [16, 128]");
    println!("  • dropout: continuous [0.0, 0.5]\n");

    // ============================================================================
    // 2. Example 1: Expected Improvement (EI)
    // ============================================================================

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Example 1: Expected Improvement (EI)                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut bayes_opt_ei = BayesianOptimization::new(
        param_space.clone(),
        15, // n_iterations
        5,  // n_initial_points
        42, // seed
    )
    .with_acquisition(AcquisitionFunction::ExpectedImprovement { xi: 0.01 })
    .with_kernel(GpKernel::Rbf {
        sigma: 1.0,
        length_scale: 1.0,
    });

    println!("Configuration:");
    println!("  • Acquisition: Expected Improvement (xi=0.01)");
    println!("  • Kernel: RBF (σ²=1.0, l=1.0)");
    println!("  • Budget: {} evaluations\n", bayes_opt_ei.total_budget());

    run_optimization(&mut bayes_opt_ei, "EI");

    // ============================================================================
    // 3. Example 2: Upper Confidence Bound (UCB)
    // ============================================================================

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     Example 2: Upper Confidence Bound (UCB)                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut bayes_opt_ucb = BayesianOptimization::new(
        param_space.clone(),
        15,
        5,
        123, // Different seed for variation
    )
    .with_acquisition(AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 })
    .with_kernel(GpKernel::Rbf {
        sigma: 1.0,
        length_scale: 0.5, // Shorter length scale
    });

    println!("Configuration:");
    println!("  • Acquisition: Upper Confidence Bound (κ=2.0)");
    println!("  • Kernel: RBF (σ²=1.0, l=0.5)");
    println!("  • Budget: {} evaluations\n", bayes_opt_ucb.total_budget());

    run_optimization(&mut bayes_opt_ucb, "UCB");

    // ============================================================================
    // 4. Example 3: Matérn Kernel
    // ============================================================================

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     Example 3: Matérn 3/2 Kernel                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut bayes_opt_matern = BayesianOptimization::new(param_space.clone(), 15, 5, 456)
        .with_acquisition(AcquisitionFunction::ExpectedImprovement { xi: 0.01 })
        .with_kernel(GpKernel::Matern32 {
            sigma: 1.0,
            length_scale: 1.0,
        });

    println!("Configuration:");
    println!("  • Acquisition: Expected Improvement (xi=0.01)");
    println!("  • Kernel: Matérn 3/2 (σ²=1.0, l=1.0)");
    println!(
        "  • Budget: {} evaluations\n",
        bayes_opt_matern.total_budget()
    );

    run_optimization(&mut bayes_opt_matern, "Matérn");

    // ============================================================================
    // 5. Comparison with Grid Search and Random Search
    // ============================================================================

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     Comparison: Bayesian vs Grid vs Random                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Grid Search (20 evaluations)
    let mut grid_search = GridSearch::new(param_space.clone(), 3);
    let grid_configs = grid_search.generate_configs();
    let grid_configs = grid_configs.into_iter().take(20).collect::<Vec<_>>();

    let mut grid_best_score: f64 = 0.0;
    for config in grid_configs {
        let lr = config.get("lr").unwrap().as_float().unwrap();
        let batch_size = config.get("batch_size").unwrap().as_int().unwrap();
        let dropout = config.get("dropout").unwrap().as_float().unwrap();

        let score = objective_function(lr, batch_size, dropout);
        grid_search.add_result(HyperparamResult::new(config, score));
        grid_best_score = grid_best_score.max(score);
    }

    // Random Search (20 evaluations)
    let mut random_search = RandomSearch::new(param_space.clone(), 20, 789);
    let random_configs = random_search.generate_configs();

    let mut random_best_score: f64 = 0.0;
    for config in random_configs {
        let lr = config.get("lr").unwrap().as_float().unwrap();
        let batch_size = config.get("batch_size").unwrap().as_int().unwrap();
        let dropout = config.get("dropout").unwrap().as_float().unwrap();

        let score = objective_function(lr, batch_size, dropout);
        random_search.add_result(HyperparamResult::new(config, score));
        random_best_score = random_best_score.max(score);
    }

    // Get best score from Bayesian Optimization (EI variant)
    let bayes_best_score = bayes_opt_ei.best_result().map(|r| r.score).unwrap_or(0.0);

    println!("Results after 20 evaluations:");
    println!("  • Grid Search:      {:.6}", grid_best_score);
    println!("  • Random Search:    {:.6}", random_best_score);
    println!("  • Bayesian Opt (EI): {:.6}", bayes_best_score);
    println!();

    let improvement_vs_grid = (bayes_best_score - grid_best_score) / grid_best_score * 100.0;
    let improvement_vs_random = (bayes_best_score - random_best_score) / random_best_score * 100.0;

    println!("Improvement:");
    println!("  • vs Grid:   {:>6.2}%", improvement_vs_grid);
    println!("  • vs Random: {:>6.2}%", improvement_vs_random);

    // ============================================================================
    // 6. Best Hyperparameters
    // ============================================================================

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     Best Hyperparameters Found                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if let Some(best) = bayes_opt_ei.best_result() {
        let lr = best.config.get("lr").unwrap().as_float().unwrap();
        let batch_size = best.config.get("batch_size").unwrap().as_int().unwrap();
        let dropout = best.config.get("dropout").unwrap().as_float().unwrap();

        println!("Bayesian Optimization (EI) found:");
        println!("  • Learning rate: {:.6}", lr);
        println!("  • Batch size:    {}", batch_size);
        println!("  • Dropout:       {:.4}", dropout);
        println!("  • Score:         {:.6}", best.score);
        println!();
        println!("Known optimum (for reference):");
        println!("  • Learning rate: 0.010000");
        println!("  • Batch size:    64");
        println!("  • Dropout:       0.2000");
    }

    println!("\n✅ Bayesian Optimization demonstration complete!");
    println!("\nKey takeaways:");
    println!("  1. Bayesian Optimization intelligently explores the search space");
    println!("  2. Uses Gaussian Processes to model the objective function");
    println!("  3. Acquisition functions balance exploration vs exploitation");
    println!("  4. More efficient than grid/random search for expensive objectives");
    println!("  5. Different kernels and acquisition functions suit different problems");
}

/// Run Bayesian Optimization and display results.
fn run_optimization(bayes_opt: &mut BayesianOptimization, _name: &str) {
    println!("Running optimization...\n");

    let mut iteration = 0;
    let mut best_score_so_far: f64 = 0.0;

    while !bayes_opt.is_complete() {
        // Get next suggested configuration
        let config = bayes_opt.suggest().unwrap();

        // Extract hyperparameters
        let lr = config.get("lr").unwrap().as_float().unwrap();
        let batch_size = config.get("batch_size").unwrap().as_int().unwrap();
        let dropout = config.get("dropout").unwrap().as_float().unwrap();

        // Evaluate the objective function
        let score = objective_function(lr, batch_size, dropout);

        // Track improvement
        if score > best_score_so_far {
            best_score_so_far = score;
            println!(
                "  ✨ Iter {:2}: score={:.6} (NEW BEST!) lr={:.6}, bs={:3}, dp={:.4}",
                iteration, score, lr, batch_size, dropout
            );
        } else if iteration % 5 == 0 || iteration < 5 {
            println!(
                "     Iter {:2}: score={:.6}            lr={:.6}, bs={:3}, dp={:.4}",
                iteration, score, lr, batch_size, dropout
            );
        }

        // Add result to Bayesian Optimization
        bayes_opt.add_result(HyperparamResult::new(config, score));

        iteration += 1;
    }

    println!();
    println!("Optimization complete!");
    println!("  • Total evaluations: {}", bayes_opt.current_iteration());
    println!("  • Best score:        {:.6}", best_score_so_far);

    // Show top 3 results
    println!("\nTop 3 configurations:");
    for (i, result) in bayes_opt.sorted_results().iter().take(3).enumerate() {
        let lr = result.config.get("lr").unwrap().as_float().unwrap();
        let batch_size = result.config.get("batch_size").unwrap().as_int().unwrap();
        let dropout = result.config.get("dropout").unwrap().as_float().unwrap();

        println!(
            "  {}. Score={:.6}: lr={:.6}, bs={:3}, dp={:.4}",
            i + 1,
            result.score,
            lr,
            batch_size,
            dropout
        );
    }
}
