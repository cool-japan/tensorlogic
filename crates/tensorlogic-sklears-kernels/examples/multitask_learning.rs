//! Example demonstrating multi-task kernel learning.
//!
//! This example shows:
//! 1. ICM (Intrinsic Coregionalization Model) for related tasks
//! 2. LMC (Linear Model of Coregionalization) with multiple latent processes
//! 3. Task similarity configurations
//! 4. Multi-output regression scenario
//!
//! Run with: cargo run --example multitask_learning

use tensorlogic_sklears_kernels::{
    ICMKernel, LMCKernel, LinearKernel, MultiTaskKernelBuilder, RbfKernel, RbfKernelConfig,
    TaskInput,
};

fn main() {
    println!("=== Multi-Task Kernel Learning Demo ===\n");

    // 1. Basic ICM with correlated tasks
    demo_icm_correlated();

    // 2. ICM with independent tasks
    demo_icm_independent();

    // 3. LMC with multiple latent processes
    demo_lmc_multiprocess();

    // 4. Multi-output regression scenario
    demo_multioutput_regression();

    println!("\n=== Demo Complete ===");
}

/// Demonstrate ICM with correlated tasks
fn demo_icm_correlated() {
    println!("1. ICM Kernel - Correlated Tasks");
    println!("{}", "-".repeat(50));

    // Three related tasks with specified correlations
    let task_covariance = vec![
        vec![1.0, 0.8, 0.5], // Task 0 correlates highly with Task 1
        vec![0.8, 1.0, 0.6], // Task 1 is the "bridge" task
        vec![0.5, 0.6, 1.0], // Task 2 less correlated with Task 0
    ];

    let base_kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
    let icm = ICMKernel::new(Box::new(base_kernel), task_covariance).unwrap();

    println!("Tasks: 3 (correlated)");
    println!("Base kernel: RBF (γ=0.5)");
    println!("Task correlations:");
    println!("  Task 0-1: 0.8 (high)");
    println!("  Task 0-2: 0.5 (medium)");
    println!("  Task 1-2: 0.6 (medium)\n");

    // Create sample points
    let samples = vec![
        TaskInput::new(vec![0.0, 0.0], 0), // Task 0
        TaskInput::new(vec![0.0, 0.0], 1), // Task 1, same features
        TaskInput::new(vec![0.0, 0.0], 2), // Task 2, same features
        TaskInput::new(vec![1.0, 0.0], 0), // Task 0, different features
    ];

    // Compute kernel matrix
    let matrix = icm.compute_task_matrix(&samples).unwrap();

    println!("Kernel matrix (same features, different tasks):");
    println!("         Task0   Task1   Task2   Task0'");
    for (i, row) in matrix.iter().enumerate() {
        let label = match i {
            0 => "Task0  ",
            1 => "Task1  ",
            2 => "Task2  ",
            3 => "Task0' ",
            _ => "       ",
        };
        print!("  {}", label);
        for val in row {
            print!("{:7.3} ", val);
        }
        println!();
    }

    println!("\nObservations:");
    println!("  - Same task, same features: K = 1.0 (diagonal)");
    println!("  - Same features, different tasks: K = task correlation");
    println!("  - Different features: K = correlation × RBF similarity");

    println!();
}

/// Demonstrate ICM with independent tasks
fn demo_icm_independent() {
    println!("2. ICM Kernel - Independent Tasks");
    println!("{}", "-".repeat(50));

    let base_kernel = LinearKernel::new();
    let icm = ICMKernel::independent(Box::new(base_kernel), 3).unwrap();

    println!("Tasks: 3 (independent)");
    println!("Base kernel: Linear\n");

    let x = TaskInput::new(vec![1.0, 2.0], 0);
    let y = TaskInput::new(vec![1.0, 2.0], 1);
    let z = TaskInput::new(vec![1.0, 2.0], 0);

    let k_same = icm.compute_tasks(&x, &z).unwrap();
    let k_diff = icm.compute_tasks(&x, &y).unwrap();

    println!("Same task (0-0):      K = {:.3}", k_same);
    println!("Different task (0-1): K = {:.3}", k_diff);
    println!("\nWith independent tasks, cross-task similarity is zero.");

    println!();
}

/// Demonstrate LMC with multiple latent processes
fn demo_lmc_multiprocess() {
    println!("3. LMC Kernel - Multiple Latent Processes");
    println!("{}", "-".repeat(50));

    let mut lmc = LMCKernel::new(2);

    // Process 1: Long-range RBF with high cross-task correlation
    let rbf_long = RbfKernel::new(RbfKernelConfig::new(0.1)).unwrap();
    let cov1 = vec![vec![1.0, 0.9], vec![0.9, 1.0]];
    lmc.add_component(Box::new(rbf_long), cov1).unwrap();

    // Process 2: Short-range RBF with lower cross-task correlation
    let rbf_short = RbfKernel::new(RbfKernelConfig::new(2.0)).unwrap();
    let cov2 = vec![vec![1.0, 0.3], vec![0.3, 1.0]];
    lmc.add_component(Box::new(rbf_short), cov2).unwrap();

    println!("Tasks: 2");
    println!("Latent processes: 2");
    println!("  Process 1: RBF(γ=0.1) - long range, high correlation (0.9)");
    println!("  Process 2: RBF(γ=2.0) - short range, low correlation (0.3)\n");

    // Compare near and far points
    let near = [TaskInput::new(vec![0.0], 0), TaskInput::new(vec![0.1], 1)];
    let far = [TaskInput::new(vec![0.0], 0), TaskInput::new(vec![2.0], 1)];

    let k_near = lmc.compute_tasks(&near[0], &near[1]).unwrap();
    let k_far = lmc.compute_tasks(&far[0], &far[1]).unwrap();

    println!("Cross-task kernel (near, d=0.1): K = {:.4}", k_near);
    println!("Cross-task kernel (far, d=2.0):  K = {:.4}", k_far);
    println!("\nLMC captures both long-range and short-range correlations.");

    println!();
}

/// Demonstrate multi-output regression scenario
fn demo_multioutput_regression() {
    println!("4. Multi-Output Regression Scenario");
    println!("{}", "-".repeat(50));

    // Scenario: Predict related physical properties
    // Task 0: Temperature
    // Task 1: Pressure (correlated with temperature)
    // Task 2: Humidity (less correlated)

    println!("Scenario: Predicting correlated physical measurements");
    println!("  Task 0: Temperature");
    println!("  Task 1: Pressure (highly correlated with temperature)");
    println!("  Task 2: Humidity (moderately correlated)\n");

    // Build ICM using the builder
    let base_kernel = RbfKernel::new(RbfKernelConfig::new(1.0)).unwrap();
    let task_covariance = vec![
        vec![1.0, 0.85, 0.4],
        vec![0.85, 1.0, 0.3],
        vec![0.4, 0.3, 1.0],
    ];

    let icm = MultiTaskKernelBuilder::new(3)
        .add_component(Box::new(base_kernel), task_covariance)
        .build_icm()
        .unwrap();

    // Training data (mixed tasks)
    let training_data = vec![
        TaskInput::new(vec![0.0, 0.0], 0), // Temperature at location (0, 0)
        TaskInput::new(vec![1.0, 0.0], 0), // Temperature at location (1, 0)
        TaskInput::new(vec![0.0, 0.0], 1), // Pressure at location (0, 0)
        TaskInput::new(vec![0.5, 0.5], 2), // Humidity at location (0.5, 0.5)
    ];

    // Query point: Pressure at (1, 0)
    let query = TaskInput::new(vec![1.0, 0.0], 1);

    println!("Training points:");
    for (i, input) in training_data.iter().enumerate() {
        let task_name = match input.task {
            0 => "Temperature",
            1 => "Pressure",
            2 => "Humidity",
            _ => "Unknown",
        };
        println!(
            "  {:2}. {} at ({:.1}, {:.1})",
            i + 1,
            task_name,
            input.features[0],
            input.features[1]
        );
    }

    println!("\nQuery: Pressure at (1.0, 0.0)");
    println!("\nKernel values to training points:");

    for train in &training_data {
        let k = icm.compute_tasks(&query, train).unwrap();
        let task_name = match train.task {
            0 => "Temperature",
            1 => "Pressure",
            2 => "Humidity",
            _ => "Unknown",
        };
        println!("  K(query, {}) = {:.4}", task_name, k);
    }

    println!("\nAnalysis:");
    println!("  - High similarity to Pressure at (0, 0) due to same task type");
    println!("  - Good similarity to Temperature at (1, 0) due to:");
    println!("    * Same location (RBF contribution)");
    println!("    * High task correlation (0.85)");
    println!("  - Lower similarity to Temperature at (0, 0) due to distance");
    println!("  - Lowest similarity to Humidity (different location + low correlation)");
}
