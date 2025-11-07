//! Sparse kernel matrix support for large-scale problems.
//!
//! Provides efficient storage and operations for sparse kernel matrices.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{KernelError, Result};
use crate::types::Kernel;

/// Sparse kernel matrix using Compressed Sparse Row (CSR) format
///
/// Stores only non-zero entries for efficient memory usage.
///
/// # Example
///
/// ```rust
/// use tensorlogic_sklears_kernels::SparseKernelMatrix;
///
/// let mut matrix = SparseKernelMatrix::new(3);
/// matrix.set(0, 1, 0.8);
/// matrix.set(1, 2, 0.6);
///
/// assert_eq!(matrix.get(0, 1), Some(0.8));
/// assert_eq!(matrix.get(0, 2), None);
/// assert_eq!(matrix.nnz(), 2);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseKernelMatrix {
    /// Number of rows/columns (square matrix)
    size: usize,
    /// Row pointers for CSR format
    row_ptr: Vec<usize>,
    /// Column indices
    col_idx: Vec<usize>,
    /// Non-zero values
    values: Vec<f64>,
    /// Temporary map for construction (not serialized)
    #[serde(skip)]
    temp_map: HashMap<(usize, usize), f64>,
}

impl SparseKernelMatrix {
    /// Create a new sparse kernel matrix
    pub fn new(size: usize) -> Self {
        Self {
            size,
            row_ptr: vec![0; size + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
            temp_map: HashMap::new(),
        }
    }

    /// Set a value in the matrix
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.size || col >= self.size {
            return;
        }

        if value.abs() < 1e-10 {
            // Remove near-zero values
            self.temp_map.remove(&(row, col));
        } else {
            self.temp_map.insert((row, col), value);
        }
    }

    /// Get a value from the matrix
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.size || col >= self.size {
            return None;
        }

        // Check temp map first
        if let Some(&value) = self.temp_map.get(&(row, col)) {
            return Some(value);
        }

        // Search in CSR format
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        for i in start..end {
            if self.col_idx[i] == col {
                return Some(self.values[i]);
            }
        }

        None
    }

    /// Finalize the matrix (convert temp map to CSR format)
    pub fn finalize(&mut self) {
        if self.temp_map.is_empty() {
            return;
        }

        // Clear existing CSR data
        self.col_idx.clear();
        self.values.clear();
        self.row_ptr = vec![0; self.size + 1];

        // Sort entries by row, then column
        let mut entries: Vec<_> = self.temp_map.iter().collect();
        entries.sort_by_key(|&((row, col), _)| (*row, *col));

        // Build CSR format
        let mut current_row = 0;
        for (&(row, col), &value) in &entries {
            // Update row pointers
            while current_row < row {
                current_row += 1;
                self.row_ptr[current_row] = self.col_idx.len();
            }

            self.col_idx.push(col);
            self.values.push(value);
        }

        // Finalize row pointers
        while current_row < self.size {
            current_row += 1;
            self.row_ptr[current_row] = self.col_idx.len();
        }

        // Clear temp map
        self.temp_map.clear();
    }

    /// Get number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.values.len() + self.temp_map.len()
    }

    /// Get matrix size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get density (fraction of non-zero entries)
    pub fn density(&self) -> f64 {
        let total = self.size * self.size;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    /// Convert to dense matrix
    #[allow(clippy::needless_range_loop)]
    pub fn to_dense(&mut self) -> Vec<Vec<f64>> {
        self.finalize();

        let mut dense = vec![vec![0.0; self.size]; self.size];

        for row in 0..self.size {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                let col = self.col_idx[i];
                let value = self.values[i];
                dense[row][col] = value;
            }
        }

        dense
    }

    /// Compute sparse kernel matrix from data with threshold
    pub fn from_kernel_with_threshold(
        data: &[Vec<f64>],
        kernel: &dyn Kernel,
        threshold: f64,
    ) -> Result<Self> {
        let n = data.len();
        let mut matrix = Self::new(n);

        for i in 0..n {
            for j in 0..n {
                let value = kernel.compute(&data[i], &data[j])?;
                if value.abs() >= threshold {
                    matrix.set(i, j, value);
                }
            }
        }

        matrix.finalize();
        Ok(matrix)
    }

    /// Get row as sparse vector
    pub fn row(&mut self, row_idx: usize) -> Option<Vec<(usize, f64)>> {
        if row_idx >= self.size {
            return None;
        }

        self.finalize();

        let start = self.row_ptr[row_idx];
        let end = self.row_ptr[row_idx + 1];

        let mut row_data = Vec::new();
        for i in start..end {
            row_data.push((self.col_idx[i], self.values[i]));
        }

        Some(row_data)
    }
}

/// Sparse kernel matrix builder with configuration
pub struct SparseKernelMatrixBuilder {
    /// Sparsity threshold (values below this are treated as zero)
    threshold: f64,
    /// Maximum entries per row (for controlled sparsity)
    max_entries_per_row: Option<usize>,
}

impl SparseKernelMatrixBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            threshold: 1e-10,
            max_entries_per_row: None,
        }
    }

    /// Set sparsity threshold
    pub fn with_threshold(mut self, threshold: f64) -> Result<Self> {
        if threshold < 0.0 {
            return Err(KernelError::InvalidParameter {
                parameter: "threshold".to_string(),
                value: threshold.to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        self.threshold = threshold;
        Ok(self)
    }

    /// Set maximum entries per row
    pub fn with_max_entries_per_row(mut self, max_entries: usize) -> Result<Self> {
        if max_entries == 0 {
            return Err(KernelError::InvalidParameter {
                parameter: "max_entries_per_row".to_string(),
                value: max_entries.to_string(),
                reason: "must be positive".to_string(),
            });
        }
        self.max_entries_per_row = Some(max_entries);
        Ok(self)
    }

    /// Build sparse kernel matrix from data
    pub fn build(&self, data: &[Vec<f64>], kernel: &dyn Kernel) -> Result<SparseKernelMatrix> {
        let n = data.len();
        let mut matrix = SparseKernelMatrix::new(n);

        for i in 0..n {
            let mut row_entries = Vec::new();

            // Compute all values for this row
            for j in 0..n {
                let value = kernel.compute(&data[i], &data[j])?;
                if value.abs() >= self.threshold {
                    row_entries.push((j, value));
                }
            }

            // If max_entries_per_row is set, keep only top-k entries
            if let Some(max_entries) = self.max_entries_per_row {
                if row_entries.len() > max_entries {
                    // Sort by absolute value (descending)
                    row_entries.sort_by(|(_, a), (_, b)| b.abs().partial_cmp(&a.abs()).unwrap());
                    row_entries.truncate(max_entries);
                }
            }

            // Add entries to matrix
            for (j, value) in row_entries {
                matrix.set(i, j, value);
            }
        }

        matrix.finalize();
        Ok(matrix)
    }
}

impl Default for SparseKernelMatrixBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_kernel::LinearKernel;

    #[test]
    fn test_sparse_matrix_creation() {
        let matrix = SparseKernelMatrix::new(3);
        assert_eq!(matrix.size(), 3);
        assert_eq!(matrix.nnz(), 0);
    }

    #[test]
    fn test_sparse_matrix_set_get() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 1, 0.8);
        matrix.set(1, 2, 0.6);

        assert_eq!(matrix.get(0, 1), Some(0.8));
        assert_eq!(matrix.get(1, 2), Some(0.6));
        assert_eq!(matrix.get(0, 2), None);
    }

    #[test]
    fn test_sparse_matrix_finalize() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 1, 0.8);
        matrix.set(1, 2, 0.6);
        matrix.set(2, 0, 0.4);

        matrix.finalize();

        assert_eq!(matrix.get(0, 1), Some(0.8));
        assert_eq!(matrix.get(1, 2), Some(0.6));
        assert_eq!(matrix.get(2, 0), Some(0.4));
    }

    #[test]
    fn test_sparse_matrix_nnz() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 1, 0.8);
        matrix.set(1, 2, 0.6);

        assert_eq!(matrix.nnz(), 2);
    }

    #[test]
    fn test_sparse_matrix_density() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 1, 0.8);
        matrix.set(1, 2, 0.6);

        let density = matrix.density();
        assert!((density - 2.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_matrix_to_dense() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 1, 0.8);
        matrix.set(1, 2, 0.6);

        let dense = matrix.to_dense();
        assert_eq!(dense.len(), 3);
        assert!((dense[0][1] - 0.8).abs() < 1e-10);
        assert!((dense[1][2] - 0.6).abs() < 1e-10);
        assert!(dense[0][0].abs() < 1e-10);
    }

    #[test]
    fn test_sparse_matrix_from_kernel() {
        let kernel = LinearKernel::new();
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];

        let mut matrix =
            SparseKernelMatrix::from_kernel_with_threshold(&data, &kernel, 0.1).unwrap();

        assert!(matrix.nnz() > 0);
        let dense = matrix.to_dense();
        assert_eq!(dense.len(), 3);
    }

    #[test]
    fn test_sparse_matrix_row() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 1, 0.8);
        matrix.set(0, 2, 0.6);

        let row = matrix.row(0).unwrap();
        assert_eq!(row.len(), 2);
        assert!(row.contains(&(1, 0.8)));
        assert!(row.contains(&(2, 0.6)));
    }

    #[test]
    fn test_sparse_matrix_builder() {
        let builder = SparseKernelMatrixBuilder::new();
        let kernel = LinearKernel::new();
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let matrix = builder.build(&data, &kernel).unwrap();
        assert!(matrix.nnz() > 0);
    }

    #[test]
    fn test_sparse_matrix_builder_with_threshold() {
        let builder = SparseKernelMatrixBuilder::new()
            .with_threshold(0.5)
            .unwrap();
        let kernel = LinearKernel::new();
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let matrix = builder.build(&data, &kernel).unwrap();
        assert!(matrix.nnz() > 0);
    }

    #[test]
    fn test_sparse_matrix_builder_invalid_threshold() {
        let result = SparseKernelMatrixBuilder::new().with_threshold(-0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_matrix_builder_max_entries() {
        let builder = SparseKernelMatrixBuilder::new()
            .with_max_entries_per_row(2)
            .unwrap();
        let kernel = LinearKernel::new();
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];

        let matrix = builder.build(&data, &kernel).unwrap();
        // Each row should have at most 2 entries
        for i in 0..matrix.size() {
            let mut temp_matrix = matrix.clone();
            let row = temp_matrix.row(i).unwrap();
            assert!(row.len() <= 2);
        }
    }

    #[test]
    fn test_sparse_matrix_builder_invalid_max_entries() {
        let result = SparseKernelMatrixBuilder::new().with_max_entries_per_row(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_matrix_zero_threshold() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 1, 1e-11); // Very small value (below 1e-10 threshold)
        matrix.finalize();

        // Should be treated as zero and filtered out
        assert_eq!(matrix.nnz(), 0);
    }
}
