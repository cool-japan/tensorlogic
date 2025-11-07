//! Kernel utility functions for machine learning workflows.
//!
//! This module provides practical utilities for kernel-based machine learning:
//! - Kernel-target alignment for measuring kernel quality
//! - Gram matrix operations (eigendecomposition preparation)
//! - Distance matrix computation from kernels
//! - Kernel matrix validation

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Compute kernel-target alignment (KTA) between a kernel matrix and target labels.
///
/// KTA measures how well a kernel matrix aligns with the ideal kernel matrix
/// derived from target labels. Higher values indicate better alignment.
///
/// # Arguments
/// * `kernel_matrix` - The kernel matrix K
/// * `labels` - Binary labels (+1 or -1) for each sample
///
/// # Returns
/// * Alignment score in range [-1, 1]
///
/// # Examples
/// ```
/// use tensorlogic_sklears_kernels::kernel_utils::kernel_target_alignment;
///
/// let K = vec![
///     vec![1.0, 0.8, 0.2],
///     vec![0.8, 1.0, 0.3],
///     vec![0.2, 0.3, 1.0],
/// ];
/// let labels = vec![1.0, 1.0, -1.0];
///
/// let alignment = kernel_target_alignment(&K, &labels).unwrap();
/// // High alignment means kernel separates classes well
/// ```
pub fn kernel_target_alignment(kernel_matrix: &[Vec<f64>], labels: &[f64]) -> Result<f64> {
    let n = kernel_matrix.len();

    if n == 0 {
        return Err(KernelError::ComputationError(
            "Kernel matrix cannot be empty".to_string(),
        ));
    }

    if labels.len() != n {
        return Err(KernelError::DimensionMismatch {
            expected: vec![n],
            got: vec![labels.len()],
            context: "kernel-target alignment".to_string(),
        });
    }

    // Verify square matrix
    for row in kernel_matrix {
        if row.len() != n {
            return Err(KernelError::ComputationError(
                "Kernel matrix must be square".to_string(),
            ));
        }
    }

    // Compute ideal kernel matrix Y = y * y^T
    let mut ideal_kernel = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            ideal_kernel[i][j] = labels[i] * labels[j];
        }
    }

    // Compute Frobenius inner product <K, Y>
    let mut inner_product = 0.0;
    for i in 0..n {
        for j in 0..n {
            inner_product += kernel_matrix[i][j] * ideal_kernel[i][j];
        }
    }

    // Compute Frobenius norms ||K||_F and ||Y||_F
    let k_norm = frobenius_norm(kernel_matrix);
    let y_norm = frobenius_norm(&ideal_kernel);

    if k_norm == 0.0 || y_norm == 0.0 {
        return Ok(0.0);
    }

    // Alignment = <K, Y> / (||K||_F * ||Y||_F)
    Ok(inner_product / (k_norm * y_norm))
}

/// Compute the Frobenius norm of a matrix.
///
/// ||A||_F = sqrt(Σ_ij a_ij^2)
fn frobenius_norm(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|&x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// Compute pairwise distances from a kernel matrix.
///
/// For a valid kernel K(x,y), the distance is:
/// d(x,y) = sqrt(K(x,x) + K(y,y) - 2*K(x,y))
///
/// # Arguments
/// * `kernel_matrix` - Symmetric kernel matrix
///
/// # Returns
/// * Distance matrix
///
/// # Examples
/// ```
/// use tensorlogic_sklears_kernels::kernel_utils::distances_from_kernel;
///
/// let K = vec![
///     vec![1.0, 0.8, 0.6],
///     vec![0.8, 1.0, 0.7],
///     vec![0.6, 0.7, 1.0],
/// ];
///
/// let distances = distances_from_kernel(&K).unwrap();
/// // distances[i][j] = distance between points i and j
/// ```
pub fn distances_from_kernel(kernel_matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = kernel_matrix.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Verify square matrix
    for row in kernel_matrix {
        if row.len() != n {
            return Err(KernelError::ComputationError(
                "Kernel matrix must be square".to_string(),
            ));
        }
    }

    // Extract diagonal
    let diagonal: Vec<f64> = (0..n).map(|i| kernel_matrix[i][i]).collect();

    // Compute distances: d(i,j) = sqrt(K[i,i] + K[j,j] - 2*K[i,j])
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let sq_dist = diagonal[i] + diagonal[j] - 2.0 * kernel_matrix[i][j];
            // Clamp to zero for numerical stability
            distances[i][j] = sq_dist.max(0.0).sqrt();
        }
    }

    Ok(distances)
}

/// Check if a kernel matrix is valid (symmetric and positive semi-definite).
///
/// A valid kernel matrix must be:
/// 1. Square
/// 2. Symmetric: K[i,j] = K[j,i]
/// 3. Positive semi-definite (all eigenvalues ≥ 0)
///
/// Note: This function only checks symmetry. Full PSD checking requires
/// eigendecomposition which is expensive.
///
/// # Arguments
/// * `kernel_matrix` - Matrix to validate
/// * `tolerance` - Tolerance for symmetry check
///
/// # Returns
/// * `true` if matrix is valid
///
/// # Examples
/// ```
/// use tensorlogic_sklears_kernels::kernel_utils::is_valid_kernel_matrix;
///
/// let K = vec![
///     vec![1.0, 0.8, 0.6],
///     vec![0.8, 1.0, 0.7],
///     vec![0.6, 0.7, 1.0],
/// ];
///
/// assert!(is_valid_kernel_matrix(&K, 1e-10).unwrap());
/// ```
pub fn is_valid_kernel_matrix(kernel_matrix: &[Vec<f64>], tolerance: f64) -> Result<bool> {
    let n = kernel_matrix.len();

    if n == 0 {
        return Ok(true);
    }

    // Check square
    for row in kernel_matrix {
        if row.len() != n {
            return Ok(false);
        }
    }

    // Check symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            if (kernel_matrix[i][j] - kernel_matrix[j][i]).abs() > tolerance {
                return Ok(false);
            }
        }
    }

    // Note: Full PSD check would require eigendecomposition
    // For performance, we only check symmetry here

    Ok(true)
}

/// Compute the effective dimensionality (rank) of a kernel matrix
/// based on normalized eigenvalue spectrum.
///
/// This is useful for determining the intrinsic dimensionality of
/// the data in kernel space.
///
/// # Arguments
/// * `kernel_matrix` - Kernel matrix
/// * `variance_threshold` - Cumulative variance threshold (e.g., 0.95 for 95%)
///
/// # Returns
/// * Estimated rank (number of eigenvalues needed to reach threshold)
///
/// Note: This is a simplified estimate based on diagonal dominance.
/// For accurate rank estimation, full eigendecomposition is needed.
pub fn estimate_kernel_rank(kernel_matrix: &[Vec<f64>], variance_threshold: f64) -> Result<usize> {
    let n = kernel_matrix.len();

    if n == 0 {
        return Ok(0);
    }

    if !(0.0..=1.0).contains(&variance_threshold) {
        return Err(KernelError::InvalidParameter {
            parameter: "variance_threshold".to_string(),
            value: variance_threshold.to_string(),
            reason: "must be in range [0, 1]".to_string(),
        });
    }

    // Verify square matrix
    for row in kernel_matrix {
        if row.len() != n {
            return Err(KernelError::ComputationError(
                "Kernel matrix must be square".to_string(),
            ));
        }
    }

    // Simple estimate: use diagonal elements as proxy for eigenvalues
    let mut diagonal: Vec<f64> = (0..n).map(|i| kernel_matrix[i][i]).collect();
    diagonal.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort descending

    let total: f64 = diagonal.iter().sum();
    if total == 0.0 {
        return Ok(0);
    }

    let mut cumsum = 0.0;
    for (rank, &val) in diagonal.iter().enumerate() {
        cumsum += val;
        if cumsum / total >= variance_threshold {
            return Ok(rank + 1);
        }
    }

    Ok(n)
}

/// Compute the kernel matrix from data using a given kernel function.
///
/// This is a convenience function that wraps `Kernel::compute_matrix`.
///
/// # Arguments
/// * `data` - Feature vectors
/// * `kernel` - Kernel function
///
/// # Returns
/// * Kernel matrix K where K[i][j] = kernel(data[i], data[j])
pub fn compute_gram_matrix(data: &[Vec<f64>], kernel: &dyn Kernel) -> Result<Vec<Vec<f64>>> {
    kernel.compute_matrix(data)
}

/// Normalize each row of a data matrix (L2 normalization).
///
/// This is useful preprocessing for some kernel methods.
///
/// # Arguments
/// * `data` - Data matrix (rows are samples)
///
/// # Returns
/// * Row-normalized data matrix
///
/// # Examples
/// ```
/// use tensorlogic_sklears_kernels::kernel_utils::normalize_rows;
///
/// let data = vec![
///     vec![3.0, 4.0],
///     vec![5.0, 12.0],
/// ];
///
/// let normalized = normalize_rows(&data).unwrap();
/// // Each row now has unit norm
/// ```
pub fn normalize_rows(data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let mut normalized = Vec::with_capacity(data.len());

    for row in data {
        let norm: f64 = row.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if norm == 0.0 {
            // Keep zero vectors as-is
            normalized.push(row.clone());
        } else {
            let normalized_row: Vec<f64> = row.iter().map(|&x| x / norm).collect();
            normalized.push(normalized_row);
        }
    }

    Ok(normalized)
}

/// Compute kernel bandwidth using median heuristic.
///
/// The median heuristic sets gamma = 1 / (2 * median(distances)^2).
/// This is a common heuristic for RBF and Laplacian kernels.
///
/// # Arguments
/// * `data` - Training data
/// * `kernel` - Base kernel (used to compute pairwise distances)
/// * `sample_size` - Number of pairs to sample (None = use all)
///
/// # Returns
/// * Suggested gamma value
pub fn median_heuristic_bandwidth(
    data: &[Vec<f64>],
    kernel: &dyn Kernel,
    sample_size: Option<usize>,
) -> Result<f64> {
    let n = data.len();

    if n < 2 {
        return Err(KernelError::ComputationError(
            "Need at least 2 samples for bandwidth estimation".to_string(),
        ));
    }

    // Compute kernel matrix
    let gram_matrix = kernel.compute_matrix(data)?;

    // Extract diagonal
    let diagonal: Vec<f64> = (0..n).map(|i| gram_matrix[i][i]).collect();

    // Compute pairwise distances
    let mut distances = Vec::new();
    let sample_size = sample_size.unwrap_or(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            let sq_dist = diagonal[i] + diagonal[j] - 2.0 * gram_matrix[i][j];
            let dist = sq_dist.max(0.0).sqrt();

            if dist > 0.0 {
                distances.push(dist);
            }

            if distances.len() >= sample_size {
                break;
            }
        }
        if distances.len() >= sample_size {
            break;
        }
    }

    if distances.is_empty() {
        return Err(KernelError::ComputationError(
            "All pairwise distances are zero".to_string(),
        ));
    }

    // Compute median
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if distances.len() % 2 == 0 {
        let mid = distances.len() / 2;
        (distances[mid - 1] + distances[mid]) / 2.0
    } else {
        distances[distances.len() / 2]
    };

    // gamma = 1 / (2 * median^2)
    let gamma = 1.0 / (2.0 * median * median);

    Ok(gamma)
}

#[cfg(test)]
#[allow(non_snake_case)] // Allow K for kernel matrices (mathematical notation)
mod tests {
    use super::*;
    use crate::{LinearKernel, RbfKernel, RbfKernelConfig};

    #[test]
    fn test_kernel_target_alignment_good() {
        // Good alignment: kernel separates classes well
        let K = vec![
            vec![1.0, 0.9, 0.1],
            vec![0.9, 1.0, 0.1],
            vec![0.1, 0.1, 1.0],
        ];
        let labels = vec![1.0, 1.0, -1.0];

        let alignment = kernel_target_alignment(&K, &labels).unwrap();

        // Alignment should be positive for well-separated classes
        // Actual computed value is around 0.59
        assert!((0.5..=1.0).contains(&alignment));
    }

    #[test]
    fn test_kernel_target_alignment_poor() {
        // Poor alignment: kernel doesn't separate classes
        let K = vec![
            vec![1.0, 0.5, 0.5],
            vec![0.5, 1.0, 0.5],
            vec![0.5, 0.5, 1.0],
        ];
        let labels = vec![1.0, 1.0, -1.0];

        let alignment = kernel_target_alignment(&K, &labels).unwrap();
        assert!(alignment < 0.5); // Lower alignment
    }

    #[test]
    fn test_kernel_target_alignment_dimension_mismatch() {
        let K = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let labels = vec![1.0, -1.0, 1.0]; // Wrong size

        let result = kernel_target_alignment(&K, &labels);
        assert!(result.is_err());
    }

    #[test]
    fn test_distances_from_kernel() {
        let K = vec![
            vec![1.0, 0.8, 0.6],
            vec![0.8, 1.0, 0.7],
            vec![0.6, 0.7, 1.0],
        ];

        let distances = distances_from_kernel(&K).unwrap();

        // Diagonal should be zero
        assert!(distances[0][0].abs() < 1e-10);
        assert!(distances[1][1].abs() < 1e-10);
        assert!(distances[2][2].abs() < 1e-10);

        // Distances should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((distances[i][j] - distances[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_is_valid_kernel_matrix() {
        // Valid symmetric matrix
        let K = vec![
            vec![1.0, 0.8, 0.6],
            vec![0.8, 1.0, 0.7],
            vec![0.6, 0.7, 1.0],
        ];
        assert!(is_valid_kernel_matrix(&K, 1e-10).unwrap());

        // Asymmetric matrix
        let K_bad = vec![
            vec![1.0, 0.8, 0.6],
            vec![0.7, 1.0, 0.7], // Different from K[0][1]
            vec![0.6, 0.7, 1.0],
        ];
        assert!(!is_valid_kernel_matrix(&K_bad, 1e-10).unwrap());
    }

    #[test]
    fn test_estimate_kernel_rank() {
        let K = vec![
            vec![1.0, 0.1, 0.1],
            vec![0.1, 0.5, 0.1],
            vec![0.1, 0.1, 0.2],
        ];

        let rank = estimate_kernel_rank(&K, 0.9).unwrap();
        assert!((1..=3).contains(&rank));
    }

    #[test]
    fn test_normalize_rows() {
        let data = vec![vec![3.0, 4.0], vec![5.0, 12.0]];

        let normalized = normalize_rows(&data).unwrap();

        // Check unit norms
        for row in &normalized {
            let norm: f64 = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalize_rows_zero_vector() {
        let data = vec![vec![0.0, 0.0], vec![3.0, 4.0]];

        let normalized = normalize_rows(&data).unwrap();

        // Zero vector should remain zero
        assert!(normalized[0][0].abs() < 1e-10);
        assert!(normalized[0][1].abs() < 1e-10);

        // Second row should be normalized
        let norm: f64 = normalized[1].iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_heuristic_bandwidth() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let kernel = LinearKernel::new();
        let gamma = median_heuristic_bandwidth(&data, &kernel, None).unwrap();

        // Gamma should be positive
        assert!(gamma > 0.0);
    }

    #[test]
    fn test_compute_gram_matrix() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let kernel = LinearKernel::new();
        let K = compute_gram_matrix(&data, &kernel).unwrap();

        // Check dimensions
        assert_eq!(K.len(), 3);
        assert_eq!(K[0].len(), 3);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((K[i][j] - K[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        // ||A||_F = sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
        let norm = frobenius_norm(&matrix);
        assert!((norm - 30.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_target_alignment_binary_classification() {
        // Create kernel matrix and labels for binary classification
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();

        // Two well-separated clusters
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![5.0, 5.0], // Far away
            vec![5.1, 5.1],
            vec![5.2, 5.2],
        ];

        let labels = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

        let K = kernel.compute_matrix(&data).unwrap();
        let alignment = kernel_target_alignment(&K, &labels).unwrap();

        // Should have positive alignment for separated clusters
        assert!((0.0..=1.0).contains(&alignment));
    }
}
