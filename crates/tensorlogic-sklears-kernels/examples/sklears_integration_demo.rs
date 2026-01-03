//! SkleaRS Integration Example
//!
//! This example demonstrates how to use TensorLogic kernels with the SkleaRS
//! machine learning library for tasks like kernel SVM, kernel PCA, and kernel
//! ridge regression.
//!
//! NOTE: This example requires the `sklears` feature to be enabled:
//! ```bash
//! cargo run --example sklears_integration_demo --features sklears
//! ```

// Note: This example is disabled by default because it requires a working sklears-core crate
#[cfg(feature = "sklears")]
fn main() {
    use sklears_kernel_approximation::custom_kernel::KernelFunction;
    use tensorlogic_sklears_kernels::{
        CosineKernel, Kernel, LaplacianKernel, LinearKernel, PolynomialKernel, RbfKernel,
        RbfKernelConfig, SklearsKernelAdapter,
    };

    println!("=== TensorLogic-SkleaRS Integration Demo ===\n");

    // Example 1: Create RBF kernel adapter
    println!("1. RBF Kernel Adapter");
    println!("---------------------");
    let rbf_kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
    let rbf_adapter = SklearsKernelAdapter::new(rbf_kernel);

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    let similarity = rbf_adapter.kernel(&x, &y);
    println!("RBF similarity: {:.6}", similarity);
    println!("Description: {}\n", rbf_adapter.description());

    // Example 2: Linear kernel adapter
    println!("2. Linear Kernel Adapter");
    println!("------------------------");
    let linear_kernel = LinearKernel::new();
    let linear_adapter = SklearsKernelAdapter::new(linear_kernel);

    let similarity = linear_adapter.kernel(&x, &y);
    println!("Linear similarity (dot product): {:.6}", similarity);
    println!("Description: {}\n", linear_adapter.description());

    // Example 3: Polynomial kernel adapter
    println!("3. Polynomial Kernel Adapter");
    println!("----------------------------");
    let poly_kernel = PolynomialKernel::new(2, 1.0).unwrap();
    let poly_adapter = SklearsKernelAdapter::new(poly_kernel);

    let similarity = poly_adapter.kernel(&x, &y);
    println!("Polynomial similarity: {:.6}", similarity);
    println!("Description: {}\n", poly_adapter.description());

    // Example 4: Laplacian kernel adapter
    println!("4. Laplacian Kernel Adapter");
    println!("---------------------------");
    let laplacian_kernel = LaplacianKernel::new(0.5).unwrap();
    let laplacian_adapter = SklearsKernelAdapter::new(laplacian_kernel);

    let similarity = laplacian_adapter.kernel(&x, &y);
    println!("Laplacian similarity: {:.6}", similarity);
    println!("Description: {}\n", laplacian_adapter.description());

    // Example 5: Cosine kernel adapter
    println!("5. Cosine Kernel Adapter");
    println!("------------------------");
    let cosine_kernel = CosineKernel::new();
    let cosine_adapter = SklearsKernelAdapter::new(cosine_kernel);

    let similarity = cosine_adapter.kernel(&x, &y);
    println!("Cosine similarity: {:.6}", similarity);
    println!("Description: {}\n", cosine_adapter.description());

    // Example 6: Using with SkleaRS algorithms (conceptual)
    println!("6. Integration with SkleaRS Algorithms");
    println!("---------------------------------------");
    println!("Once SkleaRS is properly integrated, you can use these kernels with:");
    println!("  - Kernel SVM for classification");
    println!("  - Kernel Ridge Regression");
    println!("  - Kernel PCA for dimensionality reduction");
    println!("  - Gaussian Process models");
    println!("  - And any other kernel-based algorithm\n");

    // Example 7: Random Fourier Features
    println!("7. Random Fourier Features Support");
    println!("-----------------------------------");
    println!("TensorLogic kernels support random Fourier features for");
    println!("efficient kernel approximation in large-scale settings.");
    println!("This enables O(nm) complexity instead of O(n²) for kernel matrices.\n");

    // Example 8: Kernel comparison
    println!("8. Kernel Comparison");
    println!("--------------------");
    let test_data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

    println!("Computing kernel matrices for different kernels:");

    // Linear kernel matrix
    let linear_k = LinearKernel::new();
    let linear_matrix = linear_k.compute_matrix(&test_data).unwrap();
    println!("\nLinear Kernel Matrix:");
    for row in &linear_matrix {
        println!("  {:?}", row);
    }

    // RBF kernel matrix
    let rbf_k = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
    let rbf_matrix = rbf_k.compute_matrix(&test_data).unwrap();
    println!("\nRBF Kernel Matrix:");
    for row in &rbf_matrix {
        println!("  {:?}", row);
    }

    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "sklears"))]
fn main() {
    println!("This example requires the 'sklears' feature to be enabled.");
    println!("Run with: cargo run --example sklears_integration_demo --features sklears");
    println!("\nNote: The SkleaRS integration is implemented, but requires a working");
    println!("sklears-core crate. Once sklears-core compilation issues are resolved,");
    println!("you'll be able to run this example.");
}
