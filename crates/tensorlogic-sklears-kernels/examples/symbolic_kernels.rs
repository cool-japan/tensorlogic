//! # Symbolic Kernel Composition Example
//!
//! This example demonstrates how to use symbolic kernel composition to:
//! - Build complex kernels using algebraic expressions
//! - Combine multiple kernels declaratively
//! - Create kernel pipelines for experimentation
//! - Use the builder pattern for readable kernel construction
//!
//! Run with: cargo run --example symbolic_kernels

use std::sync::Arc;
use tensorlogic_sklears_kernels::{
    CosineKernel, Kernel, KernelBuilder, KernelExpr, LinearKernel, PolynomialKernel, RbfKernel,
    RbfKernelConfig, SymbolicKernel,
};

fn main() {
    println!("=== Symbolic Kernel Composition Example ===\n");

    // Example 1: Simple scaled kernel
    println!("1. Scaled Kernel");
    println!("{}", "-".repeat(50));
    scaled_kernel_example();
    println!();

    // Example 2: Sum of kernels
    println!("2. Sum of Kernels");
    println!("{}", "-".repeat(50));
    sum_kernel_example();
    println!();

    // Example 3: Product of kernels
    println!("3. Product of Kernels");
    println!("{}", "-".repeat(50));
    product_kernel_example();
    println!();

    // Example 4: Complex composition
    println!("4. Complex Composition");
    println!("{}", "-".repeat(50));
    complex_composition_example();
    println!();

    // Example 5: Kernel builder
    println!("5. Kernel Builder Pattern");
    println!("{}", "-".repeat(50));
    builder_pattern_example();
    println!();

    // Example 6: Power kernels
    println!("6. Power Kernels");
    println!("{}", "-".repeat(50));
    power_kernel_example();
    println!();

    // Example 7: Hybrid kernel for ML
    println!("7. Hybrid Kernel for ML");
    println!("{}", "-".repeat(50));
    hybrid_kernel_example();
    println!();
}

/// Example 1: Simple scaled kernel
fn scaled_kernel_example() {
    // Create a linear kernel scaled by 0.5
    let linear = Arc::new(LinearKernel::new());
    let expr = KernelExpr::base(linear).scale(0.5).unwrap();
    let kernel = SymbolicKernel::new(expr);

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    let result = kernel.compute(&x, &y).unwrap();
    println!(
        "Linear kernel: x·y = {}",
        LinearKernel::new().compute(&x, &y).unwrap()
    );
    println!("Scaled (0.5 * linear): {}", result);
    println!("Verified: 0.5 * 32 = 16");
}

/// Example 2: Sum of kernels
fn sum_kernel_example() {
    // Sum of linear and RBF kernels
    let linear = Arc::new(LinearKernel::new());
    let rbf = Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap());

    let expr = KernelExpr::base(linear).add(KernelExpr::base(rbf));
    let kernel = SymbolicKernel::new(expr);

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0]; // Same vector

    let result = kernel.compute(&x, &y).unwrap();
    println!(
        "Linear(x,x): {}",
        LinearKernel::new().compute(&x, &x).unwrap()
    );
    println!(
        "RBF(x,x): {}",
        RbfKernel::new(RbfKernelConfig::new(0.5))
            .unwrap()
            .compute(&x, &x)
            .unwrap()
    );
    println!("Sum (linear + rbf): {}", result);
}

/// Example 3: Product of kernels
fn product_kernel_example() {
    // Product of two kernels
    let linear = Arc::new(LinearKernel::new());
    let cosine = Arc::new(CosineKernel::new());

    let expr = KernelExpr::base(linear).multiply(KernelExpr::base(cosine));
    let kernel = SymbolicKernel::new(expr);

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    let result = kernel.compute(&x, &y).unwrap();
    let linear_val = LinearKernel::new().compute(&x, &y).unwrap();
    let cosine_val = CosineKernel::new().compute(&x, &y).unwrap();

    println!("Linear(x,y): {:.4}", linear_val);
    println!("Cosine(x,y): {:.4}", cosine_val);
    println!("Product (linear * cosine): {:.4}", result);
    println!(
        "Verified: {:.4} * {:.4} = {:.4}",
        linear_val,
        cosine_val,
        linear_val * cosine_val
    );
}

/// Example 4: Complex composition
fn complex_composition_example() {
    // Build: 0.7 * linear + 0.3 * rbf^2
    let linear = Arc::new(LinearKernel::new());
    let rbf = Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap());

    let linear_scaled = KernelExpr::base(linear).scale(0.7).unwrap();
    let rbf_squared = KernelExpr::base(rbf).power(2).unwrap();
    let rbf_scaled = rbf_squared.scale(0.3).unwrap();

    let expr = linear_scaled.add(rbf_scaled);
    let kernel = SymbolicKernel::new(expr);

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.5, 2.5, 3.5];

    let result = kernel.compute(&x, &y).unwrap();
    println!("Complex kernel: 0.7*linear + 0.3*rbf^2");
    println!("Result: {:.4}", result);

    // Print expression structure
    println!("\nExpression: {:?}", kernel.expression());
}

/// Example 5: Kernel builder pattern
fn builder_pattern_example() {
    // Use builder for more readable code
    let kernel = KernelBuilder::new()
        .add_scaled(Arc::new(LinearKernel::new()), 0.5)
        .add_scaled(
            Arc::new(RbfKernel::new(RbfKernelConfig::new(1.0)).unwrap()),
            0.3,
        )
        .add_scaled(Arc::new(CosineKernel::new()), 0.2)
        .build();

    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0];

    let result = kernel.compute(&x, &y).unwrap();
    println!("Builder: 0.5*linear + 0.3*rbf + 0.2*cosine");
    println!("Result: {:.4}", result);

    // Compare with individual kernels
    let linear_val = LinearKernel::new().compute(&x, &y).unwrap();
    let rbf_val = RbfKernel::new(RbfKernelConfig::new(1.0))
        .unwrap()
        .compute(&x, &y)
        .unwrap();
    let cosine_val = CosineKernel::new().compute(&x, &y).unwrap();

    let expected = 0.5 * linear_val + 0.3 * rbf_val + 0.2 * cosine_val;
    println!("Expected: {:.4}", expected);
    println!("Match: {}", (result - expected).abs() < 1e-6);
}

/// Example 6: Power kernels
fn power_kernel_example() {
    // Compare linear vs linear^2 vs linear^3
    let x = vec![2.0];
    let y = vec![3.0];

    let linear = Arc::new(LinearKernel::new());

    let k1 = SymbolicKernel::new(KernelExpr::base(linear.clone()));
    let k2 = SymbolicKernel::new(KernelExpr::base(linear.clone()).power(2).unwrap());
    let k3 = SymbolicKernel::new(KernelExpr::base(linear).power(3).unwrap());

    let r1 = k1.compute(&x, &y).unwrap();
    let r2 = k2.compute(&x, &y).unwrap();
    let r3 = k3.compute(&x, &y).unwrap();

    println!("Base value: {}", r1);
    println!("Power 2: {} (verified: {}^2 = {})", r2, r1, r1.powi(2));
    println!("Power 3: {} (verified: {}^3 = {})", r3, r1, r1.powi(3));
}

/// Example 7: Hybrid kernel for ML
fn hybrid_kernel_example() {
    println!("Building a hybrid kernel for classification:");
    println!("- Linear component for interpretability");
    println!("- RBF component for non-linear patterns");
    println!("- Polynomial component for feature interactions");

    let kernel = KernelBuilder::new()
        .add_scaled(Arc::new(LinearKernel::new()), 0.4)
        .add_scaled(
            Arc::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()),
            0.4,
        )
        .add_scaled(Arc::new(PolynomialKernel::new(2, 1.0).unwrap()), 0.2)
        .build();

    // Test on sample data
    let samples = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
    ];

    println!("\nKernel matrix (3x3):");
    let matrix = kernel.compute_matrix(&samples).unwrap();

    for row in &matrix {
        print!("  [{:.3}", row[0]);
        for val in &row[1..] {
            print!(", {:.3}", val);
        }
        println!("]");
    }

    // Check properties
    println!("\nKernel properties:");
    println!("  Is PSD: {}", kernel.is_psd());
    println!("  Diagonal (self-similarity): {:.3}", matrix[0][0]);
    println!("  Symmetric: {}", is_symmetric(&matrix));
}

/// Helper function to check matrix symmetry
#[allow(clippy::needless_range_loop)]
fn is_symmetric(matrix: &[Vec<f64>]) -> bool {
    for i in 0..matrix.len() {
        for j in 0..matrix.len() {
            if (matrix[i][j] - matrix[j][i]).abs() > 1e-10 {
                return false;
            }
        }
    }
    true
}
