//! Low-Rank Approximation with Nyström Method
//!
//! This example demonstrates how to use the Nyström method for
//! efficient kernel matrix approximation on large datasets.
//!
//! Run with: cargo run --example low_rank_approximation

use tensorlogic_sklears_kernels::{
    Kernel, LinearKernel, NystromApproximation, NystromConfig, RbfKernel, RbfKernelConfig,
    SamplingMethod,
};

fn main() -> anyhow::Result<()> {
    println!("=== Low-Rank Kernel Matrix Approximation ===\n");

    // Generate sample dataset
    println!("1. Generating sample dataset...");
    let n = 1000; // Large dataset
    let dim = 50;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..dim).map(|j| ((i * dim + j) as f64).sin()).collect())
        .collect();

    println!("   Dataset: {} samples × {} features", n, dim);
    println!(
        "   Full kernel matrix size: {} × {} = {} elements",
        n,
        n,
        n * n
    );
    println!(
        "   Memory (full): ~{:.2} MB",
        (n * n * 8) as f64 / 1_000_000.0
    );
    println!();

    // Set up Nyström approximation
    println!("2. Setting up Nyström approximation...");
    let num_landmarks = 100; // Use 100 landmarks instead of full 1000×1000
    let kernel = RbfKernel::new(RbfKernelConfig::new(0.5))?;

    println!("   Number of landmarks: {}", num_landmarks);
    println!(
        "   Compression ratio: {:.1}x",
        n as f64 / num_landmarks as f64
    );
    println!();

    // Compare different sampling methods
    println!("3. Comparing sampling methods...");
    println!();

    let sampling_methods = vec![
        ("Uniform", SamplingMethod::Uniform),
        ("First", SamplingMethod::First),
        ("K-means++", SamplingMethod::KMeansPlusPlus),
    ];

    for (name, method) in sampling_methods {
        println!("   === {} Sampling ===", name);

        let config = NystromConfig::new(num_landmarks)?
            .with_sampling(method)
            .with_regularization(1e-6)?;

        // Fit approximation
        let start = std::time::Instant::now();
        let nystrom = NystromApproximation::fit(&data, &kernel, config)?;
        let fit_time = start.elapsed();

        println!("     Fit time: {:.2?}", fit_time);
        println!(
            "     Compression ratio: {:.2}x",
            nystrom.compression_ratio()
        );

        // Test approximation quality
        let test_idx1 = 10;
        let test_idx2 = 20;
        let exact = kernel.compute(&data[test_idx1], &data[test_idx2])?;
        let approx = nystrom.approximate(test_idx1, test_idx2)?;
        let error = (exact - approx).abs();

        println!(
            "     Sample K[{},{}]: exact={:.6}, approx={:.6}, error={:.6}",
            test_idx1, test_idx2, exact, approx, error
        );
        println!();
    }

    // Demonstrate memory savings
    println!("4. Memory savings analysis...");
    let full_memory = (n * n * 8) as f64 / 1_000_000.0;
    let approx_memory = (n * num_landmarks * 8 * 2) as f64 / 1_000_000.0; // C and W_inv
    let savings = ((full_memory - approx_memory) / full_memory) * 100.0;

    println!("   Full matrix memory: {:.2} MB", full_memory);
    println!("   Approximation memory: {:.2} MB", approx_memory);
    println!("   Memory savings: {:.1}%", savings);
    println!();

    // Demonstrate effect of number of landmarks
    println!("5. Effect of landmark count on quality...");
    println!("   ┌────────────┬──────────────┬───────────────┐");
    println!("   │ Landmarks  │ Sample Error │ Compression   │");
    println!("   ├────────────┼──────────────┼───────────────┤");

    for &num_landmarks_test in &[50, 100, 200, 500] {
        let config = NystromConfig::new(num_landmarks_test)?
            .with_sampling(SamplingMethod::KMeansPlusPlus)
            .with_regularization(1e-6)?;

        let nystrom = NystromApproximation::fit(&data, &kernel, config)?;
        let compression = nystrom.compression_ratio();

        // For demonstration, we use compression ratio as a proxy for quality
        // In production, you would compare against exact kernel matrix
        println!(
            "   │   {:4}     │     N/A      │   {:.2}x        │",
            num_landmarks_test, compression
        );
    }

    println!("   └────────────┴──────────────┴───────────────┘");
    println!("   → More landmarks = better accuracy, less compression");
    println!();

    // Practical use case: SVM with kernel approximation
    println!("6. Practical Use Case: SVM Training");
    println!("   For large-scale SVM, Nyström approximation enables:");
    println!("   • Training on datasets with millions of samples");
    println!("   • Reduced memory footprint (10-100x)");
    println!("   • Faster kernel matrix operations");
    println!("   • Comparable accuracy with full kernel");
    println!();

    // Compare computational complexity
    println!("7. Computational Complexity");
    println!("   Full kernel matrix:");
    println!("     Computation: O(n²d) where n={}, d={}", n, dim);
    println!("     Storage: O(n²) = {} elements", n * n);
    println!();
    println!("   Nyström approximation (m={} landmarks):", num_landmarks);
    println!(
        "     Computation: O(nmd) = ~{:.1}x faster",
        (n * n) / (n * num_landmarks)
    );
    println!(
        "     Storage: O(nm) = ~{:.1}x less memory",
        (n * n) / (n * num_landmarks)
    );
    println!();

    // Demonstrate with linear kernel for comparison
    println!("8. Comparison: RBF vs Linear Kernel");
    let linear_kernel = LinearKernel::new();

    let config_rbf = NystromConfig::new(100)?
        .with_sampling(SamplingMethod::KMeansPlusPlus)
        .with_regularization(1e-6)?;

    let config_linear = NystromConfig::new(100)?
        .with_sampling(SamplingMethod::KMeansPlusPlus)
        .with_regularization(1e-6)?;

    let nystrom_rbf = NystromApproximation::fit(&data, &kernel, config_rbf)?;
    let nystrom_linear = NystromApproximation::fit(&data, &linear_kernel, config_linear)?;

    println!(
        "   RBF kernel compression ratio: {:.2}x",
        nystrom_rbf.compression_ratio()
    );
    println!(
        "   Linear kernel compression ratio: {:.2}x",
        nystrom_linear.compression_ratio()
    );
    println!("   → Both provide significant memory reduction");
    println!();

    // Best practices
    println!("9. Best Practices");
    println!("   ✓ Use K-means++ sampling for best approximation quality");
    println!("   ✓ Set num_landmarks = sqrt(n) as a starting point");
    println!("   ✓ Add regularization (1e-6) for numerical stability");
    println!("   ✓ Validate quality by checking sample approximations");
    println!("   ✓ Trade off accuracy vs compression based on application");
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}
