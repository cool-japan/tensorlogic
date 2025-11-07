//! Comprehensive ML workflow example using tensorlogic-sklears-kernels.
//!
//! This example demonstrates a complete machine learning workflow:
//! 1. Data preparation and normalization
//! 2. Kernel selection using kernel-target alignment
//! 3. Bandwidth selection using median heuristic
//! 4. Kernel matrix computation and validation
//! 5. Kernel transformation (normalization, centering)
//! 6. Performance optimization with caching and low-rank approximation

use tensorlogic_sklears_kernels::{
    kernel_transform::{center_kernel_matrix, normalize_kernel_matrix},
    kernel_utils::{
        compute_gram_matrix, kernel_target_alignment, median_heuristic_bandwidth, normalize_rows,
    },
    CachedKernel, ChiSquaredKernel, Kernel, LaplacianKernel, LinearKernel, NystromApproximation,
    NystromConfig, RbfKernel, RbfKernelConfig, SamplingMethod, WeightedSumKernel,
};

fn main() {
    println!("=== Comprehensive Kernel ML Workflow ===\n");

    // ===== Step 1: Data Preparation =====
    println!("Step 1: Data Preparation");

    // Simulated binary classification dataset (2 clusters)
    let training_data = vec![
        // Class +1 (around origin)
        vec![0.1, 0.2],
        vec![0.2, 0.1],
        vec![0.3, 0.3],
        vec![0.15, 0.25],
        vec![0.25, 0.15],
        // Class -1 (around (1,1))
        vec![0.9, 0.8],
        vec![0.8, 0.9],
        vec![0.85, 0.85],
        vec![0.95, 0.75],
        vec![0.75, 0.95],
    ];

    let labels = vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];

    println!("  - Training samples: {}", training_data.len());
    println!("  - Feature dimension: {}", training_data[0].len());
    println!("  - Class distribution: {} positive, {} negative\n", 5, 5);

    // Normalize data (optional preprocessing)
    let normalized_data = normalize_rows(&training_data).expect("Failed to normalize data");
    println!("  ✓ Data normalized to unit rows\n");

    // ===== Step 2: Kernel Selection =====
    println!("Step 2: Kernel Selection using Kernel-Target Alignment");

    // Test different kernel types
    let linear_kernel = LinearKernel::new();
    let rbf_kernel =
        RbfKernel::new(RbfKernelConfig::new(0.5)).expect("Failed to create RBF kernel");
    let laplacian_kernel = LaplacianKernel::new(0.5).expect("Failed to create Laplacian kernel");

    // Compute kernel matrices
    let k_linear = compute_gram_matrix(&normalized_data, &linear_kernel)
        .expect("Failed to compute linear kernel");
    let k_rbf =
        compute_gram_matrix(&normalized_data, &rbf_kernel).expect("Failed to compute RBF kernel");
    let k_laplacian = compute_gram_matrix(&normalized_data, &laplacian_kernel)
        .expect("Failed to compute Laplacian kernel");

    // Compute kernel-target alignment for each
    let kta_linear =
        kernel_target_alignment(&k_linear, &labels).expect("Failed to compute KTA for linear");
    let kta_rbf = kernel_target_alignment(&k_rbf, &labels).expect("Failed to compute KTA for RBF");
    let kta_laplacian = kernel_target_alignment(&k_laplacian, &labels)
        .expect("Failed to compute KTA for Laplacian");

    println!("  Kernel-Target Alignment (KTA):");
    println!("    - Linear:     {:.4}", kta_linear);
    println!("    - RBF:        {:.4}", kta_rbf);
    println!("    - Laplacian:  {:.4}", kta_laplacian);

    // Select best kernel
    let best_kta = kta_linear.max(kta_rbf).max(kta_laplacian);
    let best_kernel_name = if (best_kta - kta_linear).abs() < 1e-10 {
        "Linear"
    } else if (best_kta - kta_rbf).abs() < 1e-10 {
        "RBF"
    } else {
        "Laplacian"
    };

    println!(
        "  ✓ Best kernel: {} (KTA: {:.4})\n",
        best_kernel_name, best_kta
    );

    // ===== Step 3: Bandwidth Selection (for RBF) =====
    println!("Step 3: Bandwidth Selection using Median Heuristic");

    let optimal_gamma = median_heuristic_bandwidth(&normalized_data, &linear_kernel, Some(20))
        .expect("Failed to compute optimal bandwidth");

    println!("  - Median heuristic gamma: {:.4}", optimal_gamma);

    // Create optimized RBF kernel
    let optimized_rbf = RbfKernel::new(RbfKernelConfig::new(optimal_gamma))
        .expect("Failed to create optimized RBF");

    let k_optimized = compute_gram_matrix(&normalized_data, &optimized_rbf)
        .expect("Failed to compute optimized kernel");

    let kta_optimized = kernel_target_alignment(&k_optimized, &labels)
        .expect("Failed to compute KTA for optimized");

    println!("  ✓ Optimized RBF KTA: {:.4}\n", kta_optimized);

    // ===== Step 4: Kernel Matrix Transformation =====
    println!("Step 4: Kernel Matrix Transformation");

    // Normalize kernel matrix
    let k_normalized =
        normalize_kernel_matrix(&k_optimized).expect("Failed to normalize kernel matrix");

    println!("  - Normalized kernel matrix:");
    println!("      Diagonal[0]: {:.4}", k_normalized[0][0]);
    println!("      Diagonal[1]: {:.4}", k_normalized[1][1]);

    // Center kernel matrix (for kernel PCA)
    let k_centered = center_kernel_matrix(&k_normalized).expect("Failed to center kernel matrix");

    // Verify centering
    let row_sum: f64 = k_centered[0].iter().sum();
    println!("  - Centered kernel matrix:");
    println!("      Row sum (should be ~0): {:.4e}\n", row_sum);

    // ===== Step 5: Performance Optimization with Caching =====
    println!("Step 5: Performance Optimization");

    // Wrap kernel with caching
    let cached_kernel = CachedKernel::new(Box::new(optimized_rbf));

    println!("  - Using cached kernel for repeated computations...");

    // Simulate multiple kernel evaluations
    for i in 0..5 {
        let _ = cached_kernel
            .compute(&normalized_data[0], &normalized_data[i])
            .expect("Failed to compute kernel");
    }

    let stats = cached_kernel.stats();
    println!("    Cache hits: {}", stats.hits);
    println!("    Cache misses: {}", stats.misses);
    println!("    Hit rate: {:.2}%\n", stats.hit_rate() * 100.0);

    // ===== Step 6: Low-Rank Approximation =====
    println!("Step 6: Low-Rank Approximation (Nyström method)");

    let rbf_for_nystrom = RbfKernel::new(RbfKernelConfig::new(optimal_gamma))
        .expect("Failed to create RBF for Nyström");

    // Use 5 landmarks for 10 samples (50% compression)
    let config = NystromConfig::new(5)
        .expect("Failed to create Nyström config")
        .with_sampling(SamplingMethod::KMeansPlusPlus)
        .with_regularization(1e-6)
        .expect("Failed to set regularization");

    let nystrom = NystromApproximation::fit(&normalized_data, &rbf_for_nystrom, config)
        .expect("Failed to fit Nyström approximation");

    println!("  - Number of landmarks: 5");
    println!("  - Sampling method: K-means++");
    println!("  - Compression ratio: {:.2}x", nystrom.compression_ratio());

    // Compute approximation error
    let k_exact = compute_gram_matrix(&normalized_data, &rbf_for_nystrom)
        .expect("Failed to compute exact kernel");

    let error = nystrom
        .approximation_error(&k_exact)
        .expect("Failed to compute error");

    println!("  - Approximation error: {:.4e}\n", error);

    // ===== Step 7: Composite Kernels =====
    println!("Step 7: Composite Kernels");

    // Combine multiple kernels
    let linear_boxed = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
    let rbf_boxed = Box::new(
        RbfKernel::new(RbfKernelConfig::new(optimal_gamma)).expect("Failed to create RBF"),
    ) as Box<dyn Kernel>;

    let weights = vec![0.3, 0.7]; // 30% linear, 70% RBF
    let composite = WeightedSumKernel::new(vec![linear_boxed, rbf_boxed], weights)
        .expect("Failed to create composite kernel");

    let k_composite =
        compute_gram_matrix(&normalized_data, &composite).expect("Failed to compute composite");

    let kta_composite = kernel_target_alignment(&k_composite, &labels)
        .expect("Failed to compute KTA for composite");

    println!("  - Weighted sum kernel: 0.3*Linear + 0.7*RBF");
    println!("  - Composite KTA: {:.4}\n", kta_composite);

    // ===== Step 8: Histogram Kernels for Special Data =====
    println!("Step 8: Specialized Kernels");

    // Example with histogram data (normalized feature distributions)
    let histogram_data = vec![
        vec![0.3, 0.4, 0.3],   // Sample 1 histogram
        vec![0.35, 0.35, 0.3], // Sample 2 histogram
        vec![0.2, 0.5, 0.3],   // Sample 3 histogram
    ];

    let chi_squared = ChiSquaredKernel::new(1.0).expect("Failed to create chi-squared kernel");

    let k_histogram = compute_gram_matrix(&histogram_data, &chi_squared)
        .expect("Failed to compute histogram kernel");

    println!("  - Chi-squared kernel for histogram data:");
    println!("      K[0,1] (similar): {:.4}", k_histogram[0][1]);
    println!("      K[0,2] (different): {:.4}\n", k_histogram[0][2]);

    // ===== Summary =====
    println!("=== Workflow Complete ===");
    println!("\nKey Results:");
    println!(
        "  ✓ Optimal kernel: {} (KTA: {:.4})",
        best_kernel_name, best_kta
    );
    println!("  ✓ Optimal bandwidth (gamma): {:.4}", optimal_gamma);
    println!("  ✓ Cache hit rate: {:.2}%", stats.hit_rate() * 100.0);
    println!(
        "  ✓ Nyström compression: {:.2}x with error {:.4e}",
        nystrom.compression_ratio(),
        error
    );
    println!("  ✓ Composite kernel KTA: {:.4}", kta_composite);

    println!("\nThis workflow demonstrates:");
    println!("  1. Data normalization and preprocessing");
    println!("  2. Kernel selection via kernel-target alignment");
    println!("  3. Automatic bandwidth selection");
    println!("  4. Kernel matrix transformations");
    println!("  5. Performance optimization with caching");
    println!("  6. Memory-efficient low-rank approximations");
    println!("  7. Flexible kernel composition");
    println!("  8. Specialized kernels for specific data types");
}
