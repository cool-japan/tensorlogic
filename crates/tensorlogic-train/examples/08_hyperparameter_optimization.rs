//! Example: Hyperparameter Optimization
//!
//! This example demonstrates automated hyperparameter search:
//! - Grid search (exhaustive)
//! - Random search (stochastic)
//! - Parameter space definition
//! - Result analysis
//!
//! Run with: cargo run --example 08_hyperparameter_optimization

use std::collections::HashMap;
use tensorlogic_train::{
    GridSearch, HyperparamConfig, HyperparamResult, HyperparamSpace, HyperparamValue, RandomSearch,
};

fn main() {
    println!("=== Hyperparameter Optimization Examples ===\n");

    // Example 1: Parameter Space Definition
    println!("1. Defining Parameter Spaces");
    println!("   Different types of hyperparameter distributions\n");

    // Discrete choices
    let _optimizer_space = HyperparamSpace::discrete(vec![
        HyperparamValue::String("sgd".to_string()),
        HyperparamValue::String("adam".to_string()),
        HyperparamValue::String("adamw".to_string()),
    ])
    .unwrap();

    println!("   Optimizer (discrete): {{sgd, adam, adamw}}");

    // Continuous range
    let _lr_space = HyperparamSpace::continuous(1e-4, 1e-1).unwrap();
    println!("   Learning rate (continuous): [1e-4, 1e-1]");

    // Log-uniform (better for learning rate)
    let _lr_log_space = HyperparamSpace::log_uniform(1e-5, 1e-2).unwrap();
    println!("   Learning rate (log-uniform): [1e-5, 1e-2]");

    // Integer range
    let _batch_size_space = HyperparamSpace::int_range(16, 128).unwrap();
    println!("   Batch size (integer): [16, 128]\n");

    // Example 2: Grid Search
    println!("2. Grid Search (Exhaustive Search)");
    println!("   Systematically explores all parameter combinations\n");

    let mut param_space = HashMap::new();

    // Small search space for demonstration
    param_space.insert(
        "learning_rate".to_string(),
        HyperparamSpace::discrete(vec![
            HyperparamValue::Float(1e-3),
            HyperparamValue::Float(1e-2),
        ])
        .unwrap(),
    );

    param_space.insert(
        "batch_size".to_string(),
        HyperparamSpace::discrete(vec![HyperparamValue::Int(32), HyperparamValue::Int(64)])
            .unwrap(),
    );

    param_space.insert(
        "optimizer".to_string(),
        HyperparamSpace::discrete(vec![
            HyperparamValue::String("adam".to_string()),
            HyperparamValue::String("adamw".to_string()),
        ])
        .unwrap(),
    );

    let mut grid_search = GridSearch::new(param_space, 3);

    println!("   Parameter space:");
    println!("     learning_rate: {{1e-3, 1e-2}}");
    println!("     batch_size: {{32, 64}}");
    println!("     optimizer: {{adam, adamw}}");
    println!(
        "   Total configurations: {} (2 × 2 × 2)\n",
        grid_search.total_configs()
    );

    // Generate all configurations
    let configs = grid_search.generate_configs();

    println!("   Generated configurations:");
    for (i, config) in configs.iter().enumerate().take(4) {
        println!("   Config {}: ", i + 1);
        for (name, value) in config {
            print!("     {}: ", name);
            match value {
                HyperparamValue::Float(v) => println!("{}", v),
                HyperparamValue::Int(v) => println!("{}", v),
                HyperparamValue::String(v) => println!("{}", v),
                _ => println!("{:?}", value),
            }
        }
    }

    // Simulate training with each configuration
    println!("\n   Simulating training...");
    for (i, config) in configs.iter().enumerate() {
        // Simulate model training (would be actual training in practice)
        let score = simulate_training(config);

        let result = HyperparamResult::new(config.clone(), score)
            .with_metric("accuracy".to_string(), score)
            .with_metric("loss".to_string(), 1.0 - score);

        grid_search.add_result(result);

        println!("     Config {}: Score = {:.4}", i + 1, score);
    }

    // Get best result
    if let Some(best) = grid_search.best_result() {
        println!("\n   Best configuration:");
        println!("     Score: {:.4}", best.score);
        for (name, value) in &best.config {
            print!("     {}: ", name);
            match value {
                HyperparamValue::Float(v) => println!("{}", v),
                HyperparamValue::Int(v) => println!("{}", v),
                HyperparamValue::String(v) => println!("{}", v),
                _ => println!("{:?}", value),
            }
        }
    }

    // Example 3: Random Search
    println!("\n3. Random Search (Stochastic Sampling)");
    println!("   Randomly samples from parameter space\n");

    let mut param_space_random = HashMap::new();

    param_space_random.insert(
        "learning_rate".to_string(),
        HyperparamSpace::log_uniform(1e-5, 1e-2).unwrap(),
    );

    param_space_random.insert(
        "dropout".to_string(),
        HyperparamSpace::continuous(0.0, 0.5).unwrap(),
    );

    param_space_random.insert(
        "hidden_size".to_string(),
        HyperparamSpace::int_range(64, 512).unwrap(),
    );

    let mut random_search = RandomSearch::new(param_space_random, 10, 42);

    println!("   Parameter space:");
    println!("     learning_rate: log-uniform[1e-5, 1e-2]");
    println!("     dropout: continuous[0.0, 0.5]");
    println!("     hidden_size: int[64, 512]");
    println!("   Number of trials: 10\n");

    // Generate random configurations
    let random_configs = random_search.generate_configs();

    println!("   Sampled configurations:");
    for (i, config) in random_configs.iter().take(5).enumerate() {
        println!("   Trial {}: ", i + 1);
        for (name, value) in config {
            match name.as_str() {
                "learning_rate" => println!("     learning_rate: {:.6}", value.as_float().unwrap()),
                "dropout" => println!("     dropout: {:.3}", value.as_float().unwrap()),
                "hidden_size" => println!("     hidden_size: {}", value.as_int().unwrap()),
                _ => {}
            }
        }
    }

    // Simulate training
    println!("\n   Running trials...");
    for (i, config) in random_configs.iter().enumerate() {
        let score = simulate_training(config);
        let result = HyperparamResult::new(config.clone(), score);
        random_search.add_result(result);

        println!("     Trial {}: Score = {:.4}", i + 1, score);
    }

    // Get best result
    if let Some(best) = random_search.best_result() {
        println!("\n   Best configuration:");
        println!("     Score: {:.4}", best.score);
        println!(
            "     learning_rate: {:.6}",
            best.config
                .get("learning_rate")
                .unwrap()
                .as_float()
                .unwrap()
        );
        println!(
            "     dropout: {:.3}",
            best.config.get("dropout").unwrap().as_float().unwrap()
        );
        println!(
            "     hidden_size: {}",
            best.config.get("hidden_size").unwrap().as_int().unwrap()
        );
    }

    // Example 4: Result Analysis
    println!("\n4. Result Analysis");
    println!("   Analyzing and comparing search results\n");

    let sorted = random_search.sorted_results();

    println!("   Top 5 configurations:");
    for (i, result) in sorted.iter().take(5).enumerate() {
        println!("   Rank {}: Score = {:.4}", i + 1, result.score);
        println!(
            "     lr: {:.6}, dropout: {:.3}, hidden: {}",
            result
                .config
                .get("learning_rate")
                .unwrap()
                .as_float()
                .unwrap(),
            result.config.get("dropout").unwrap().as_float().unwrap(),
            result.config.get("hidden_size").unwrap().as_int().unwrap()
        );
    }

    println!("\n=== Practical Workflow ===\n");
    println!("```rust");
    println!("// 1. Define parameter space");
    println!("let mut param_space = HashMap::new();");
    println!("param_space.insert(\"lr\", HyperparamSpace::log_uniform(1e-5, 1e-2)?);");
    println!("param_space.insert(\"batch_size\", HyperparamSpace::int_range(16, 128)?);");
    println!();
    println!("// 2. Choose search strategy");
    println!("let mut search = RandomSearch::new(param_space, 50, 42);");
    println!("// or");
    println!("let mut search = GridSearch::new(param_space, 5);");
    println!();
    println!("// 3. Generate configurations");
    println!("let configs = search.generate_configs();");
    println!();
    println!("// 4. Train and evaluate each configuration");
    println!("for config in configs {{");
    println!("    // Extract hyperparameters");
    println!("    let lr = config.get(\"lr\").unwrap().as_float().unwrap();");
    println!("    let batch_size = config.get(\"batch_size\").unwrap().as_int().unwrap();");
    println!("    ");
    println!("    // Train model with these hyperparameters");
    println!("    let score = train_and_evaluate(lr, batch_size)?;");
    println!("    ");
    println!("    // Record result");
    println!("    let result = HyperparamResult::new(config, score);");
    println!("    search.add_result(result);");
    println!("}}");
    println!();
    println!("// 5. Get best configuration");
    println!("let best = search.best_result().unwrap();");
    println!("println!(\"Best score: {{}}\", best.score);");
    println!("```");

    println!("\n=== Strategy Comparison ===");
    println!("Grid Search:");
    println!("  ✓ Exhaustive (guaranteed to find best in grid)");
    println!("  ✓ Good for small search spaces (< 100 configs)");
    println!("  ✗ Exponentially expensive with more parameters");
    println!();
    println!("Random Search:");
    println!("  ✓ Scalable to high-dimensional spaces");
    println!("  ✓ Better than grid for many parameters");
    println!("  ✓ Can run indefinitely (anytime algorithm)");
    println!("  ✗ No guarantees on finding optimal");
    println!();
    println!("Rule of Thumb:");
    println!("  - Use grid search for ≤3 hyperparameters");
    println!("  - Use random search for ≥4 hyperparameters");
    println!("  - Use log-uniform for learning rates");
    println!("  - Run random search 2-3× longer than grid search would take");
}

/// Simulate model training (mock function for demo)
fn simulate_training(config: &HyperparamConfig) -> f64 {
    // In practice, this would train an actual model
    // For demo, return a score based on config
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for (key, value) in config {
        key.hash(&mut hasher);
        match value {
            HyperparamValue::Float(v) => v.to_bits().hash(&mut hasher),
            HyperparamValue::Int(v) => v.hash(&mut hasher),
            HyperparamValue::String(v) => v.hash(&mut hasher),
            _ => {}
        }
    }

    // Generate pseudo-random score
    let hash = hasher.finish();
    0.7 + (hash % 30) as f64 / 100.0 // Score between 0.70 and 0.99
}
