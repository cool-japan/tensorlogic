//! Example demonstrating online kernel updates for streaming ML scenarios.
//!
//! This example shows:
//! 1. OnlineKernelMatrix - Incremental kernel matrix updates
//! 2. WindowedKernelMatrix - Sliding window for bounded memory
//! 3. ForgetfulKernelMatrix - Exponential forgetting for concept drift
//! 4. AdaptiveKernelMatrix - Automatic bandwidth adjustment
//!
//! Run with: cargo run --example online_kernel_updates

use tensorlogic_sklears_kernels::{
    ForgetfulConfig, ForgetfulKernelMatrix, LinearKernel, OnlineKernelMatrix, RbfKernel,
    RbfKernelConfig, WindowedKernelMatrix,
};

fn main() {
    println!("=== Online Kernel Updates Demo ===\n");

    // 1. Basic Online Kernel Matrix
    demo_online_kernel_matrix();

    // 2. Windowed Kernel Matrix
    demo_windowed_kernel_matrix();

    // 3. Forgetful Kernel Matrix
    demo_forgetful_kernel_matrix();

    // 4. Streaming Classification Scenario
    demo_streaming_classification();

    println!("\n=== Demo Complete ===");
}

/// Demonstrate basic online kernel matrix updates
fn demo_online_kernel_matrix() {
    println!("1. Online Kernel Matrix - Incremental Updates");
    println!("{}", "-".repeat(50));

    let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
    let mut online = OnlineKernelMatrix::new(Box::new(kernel));

    // Simulate streaming data
    let samples = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 0.0],
        vec![0.0, 2.0],
    ];

    println!("Adding samples incrementally:");
    for (i, sample) in samples.into_iter().enumerate() {
        online.add_sample(sample).unwrap();
        let stats = online.stats();
        println!(
            "  Sample {}: {} samples, {} kernel computations",
            i + 1,
            online.len(),
            stats.kernel_computations
        );
    }

    // Show matrix properties
    let matrix = online.get_matrix();
    println!("\nKernel matrix (5x5):");
    for row in matrix {
        print!("  [");
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.3}", val);
        }
        println!("]");
    }

    // Query against all samples
    let query = vec![1.5, 0.5];
    let similarities = online.compute_with_all(&query).unwrap();
    println!("\nQuery [1.5, 0.5] similarities:");
    for (i, sim) in similarities.iter().enumerate() {
        println!("  Sample {}: {:.4}", i, sim);
    }

    println!();
}

/// Demonstrate windowed kernel matrix for bounded memory
fn demo_windowed_kernel_matrix() {
    println!("2. Windowed Kernel Matrix - Sliding Window");
    println!("{}", "-".repeat(50));

    let kernel = LinearKernel::new();
    let mut windowed = WindowedKernelMatrix::new(Box::new(kernel), 3);

    println!("Window size: 3");
    println!("Adding samples to window:\n");

    // Simulate time series data
    for i in 1..=6 {
        let sample = vec![i as f64];
        let evicted = windowed.add_sample(sample).unwrap();

        print!("  Add [{}]: ", i);
        if let Some(ev) = evicted {
            print!("evicted {:?}, ", ev);
        }

        let current: Vec<_> = windowed.get_samples().iter().map(|s| s[0] as i32).collect();
        println!("window = {:?}", current);
    }

    // Show final matrix
    println!("\nFinal kernel matrix (3x3):");
    let matrix = windowed.get_matrix();
    for row in matrix {
        print!("  [");
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.1}", val);
        }
        println!("]");
    }

    println!(
        "\nStats: {} added, {} removed",
        windowed.stats().samples_added,
        windowed.stats().samples_removed
    );

    println!();
}

/// Demonstrate forgetful kernel matrix for concept drift
fn demo_forgetful_kernel_matrix() {
    println!("3. Forgetful Kernel Matrix - Exponential Decay");
    println!("{}", "-".repeat(50));

    let kernel = LinearKernel::new();
    let config = ForgetfulConfig {
        lambda: 0.8,
        removal_threshold: Some(0.3),
        max_samples: None,
    };
    let mut forgetful = ForgetfulKernelMatrix::new(Box::new(kernel), config);

    println!("Forgetting factor λ = 0.8");
    println!("Removal threshold = 0.3\n");

    // Add samples and show weight decay
    for i in 1..=5 {
        forgetful.add_sample(vec![i as f64]).unwrap();

        let weights = forgetful.get_weights();
        let samples: Vec<_> = forgetful
            .get_samples()
            .iter()
            .map(|s| s[0] as i32)
            .collect();

        println!("After adding [{}]:", i);
        println!("  Samples: {:?}", samples);
        print!("  Weights: [");
        for (j, w) in weights.iter().enumerate() {
            if j > 0 {
                print!(", ");
            }
            print!("{:.3}", w);
        }
        println!("]");
        println!("  Effective size: {:.3}", forgetful.effective_size());
        println!();
    }

    // Show weighted vs unweighted matrix
    println!("Raw kernel matrix:");
    let raw = forgetful.get_matrix();
    for row in raw.iter().take(3) {
        print!("  [");
        for (i, val) in row.iter().take(3).enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.1}", val);
        }
        println!(", ...]");
    }

    println!("\nWeighted kernel matrix (samples weighted by decay):");
    let weighted = forgetful.get_weighted_matrix();
    for row in weighted.iter().take(3) {
        print!("  [");
        for (i, val) in row.iter().take(3).enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.3}", val);
        }
        println!(", ...]");
    }

    println!();
}

/// Demonstrate a streaming classification scenario
fn demo_streaming_classification() {
    println!("4. Streaming Classification Scenario");
    println!("{}", "-".repeat(50));

    // Scenario: Online anomaly detection with RBF kernel
    let kernel = RbfKernel::new(RbfKernelConfig::new(1.0)).unwrap();
    let config = ForgetfulConfig {
        lambda: 0.95,
        removal_threshold: Some(0.1),
        max_samples: Some(10),
    };
    let mut detector = ForgetfulKernelMatrix::new(Box::new(kernel), config);

    println!("Anomaly detection with forgetful kernel (λ=0.95, max=10 samples)\n");

    // Normal data cluster around [5, 5]
    let normal_samples = vec![
        vec![4.8, 5.1],
        vec![5.2, 4.9],
        vec![5.0, 5.0],
        vec![4.9, 5.2],
        vec![5.1, 4.8],
    ];

    println!("Training on normal samples around [5, 5]:");
    for sample in normal_samples {
        detector.add_sample(sample.clone()).unwrap();
        println!("  Added {:?}", sample);
    }

    // Test points
    let test_points = vec![
        (vec![5.0, 5.0], "normal"),
        (vec![5.5, 4.5], "normal"),
        (vec![10.0, 10.0], "anomaly"),
        (vec![0.0, 0.0], "anomaly"),
    ];

    println!("\nTesting points (average similarity to training data):");
    for (query, expected) in test_points {
        let similarities = detector.compute_weighted(&query).unwrap();
        let avg_sim: f64 = similarities.iter().sum::<f64>() / similarities.len() as f64;

        let detected = if avg_sim < 0.3 { "anomaly" } else { "normal" };
        let correct = if detected == expected { "✓" } else { "✗" };

        println!(
            "  {:?}: avg_sim={:.4} -> {} (expected: {}) {}",
            query, avg_sim, detected, expected, correct
        );
    }

    println!("\nFinal detector state:");
    println!("  Samples: {}", detector.len());
    println!("  Effective size: {:.2}", detector.effective_size());
}
