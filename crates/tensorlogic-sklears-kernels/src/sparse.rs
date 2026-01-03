//! Sparse kernel matrix support for large-scale problems.
//!
//! Provides efficient storage and operations for sparse kernel matrices using
//! Compressed Sparse Row (CSR) format for memory-efficient representation.
//!
//! # Features
//!
//! - **Efficient Storage**: CSR format for sparse matrices with configurable thresholds
//! - **Matrix Operations**: SpMV, transpose, addition, scaling, Frobenius norm
//! - **Parallel Construction**: Multi-threaded matrix building with rayon
//! - **Iterator Support**: Efficient iteration over non-zero entries
//! - **Flexible Builders**: Configurable threshold and max entries per row
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_sklears_kernels::{SparseKernelMatrix, SparseKernelMatrixBuilder};
//! use tensorlogic_sklears_kernels::tensor_kernels::LinearKernel;
//!
//! // Build a sparse kernel matrix with parallel computation
//! let builder = SparseKernelMatrixBuilder::new()
//!     .with_threshold(0.1).unwrap()
//!     .with_max_entries_per_row(100).unwrap();
//!
//! let kernel = LinearKernel::new();
//! let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
//! let matrix = builder.build_parallel(&data, &kernel).unwrap();
//!
//! // Sparse matrix-vector multiplication
//! let mut matrix = SparseKernelMatrix::new(3);
//! matrix.set(0, 0, 2.0);
//! matrix.set(1, 1, 3.0);
//! let x = vec![1.0, 2.0, 0.0];
//! let y = matrix.spmv(&x).unwrap();
//!
//! // Iterate over non-zero entries
//! for (row, col, value) in matrix.iter_nonzeros() {
//!     println!("({}, {}) = {}", row, col, value);
//! }
//! ```

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

/// Advanced sparse matrix operations
impl SparseKernelMatrix {
    /// Sparse matrix-vector multiplication: y = A * x
    pub fn spmv(&mut self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.size {
            return Err(KernelError::InvalidParameter {
                parameter: "x".to_string(),
                value: x.len().to_string(),
                reason: format!("vector length must match matrix size {}", self.size),
            });
        }

        self.finalize();

        let mut y = vec![0.0; self.size];

        for (row, y_elem) in y.iter_mut().enumerate() {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            let mut sum = 0.0;
            for i in start..end {
                let col = self.col_idx[i];
                let value = self.values[i];
                sum += value * x[col];
            }
            *y_elem = sum;
        }

        Ok(y)
    }

    /// Sparse matrix transpose
    pub fn transpose(&self) -> Result<Self> {
        let mut transposed = Self::new(self.size);

        for row in 0..self.size {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                let col = self.col_idx[i];
                let value = self.values[i];
                transposed.set(col, row, value);
            }
        }

        transposed.finalize();
        Ok(transposed)
    }

    /// Add two sparse matrices element-wise
    pub fn add(&mut self, other: &Self) -> Result<Self> {
        if self.size != other.size {
            return Err(KernelError::InvalidParameter {
                parameter: "other".to_string(),
                value: other.size.to_string(),
                reason: format!("matrix sizes must match: {} vs {}", self.size, other.size),
            });
        }

        self.finalize();

        // Clone and finalize other to ensure all values are in CSR format
        let mut other_finalized = other.clone();
        other_finalized.finalize();

        let mut result = Self::new(self.size);

        // Add values from self
        for row in 0..self.size {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                let col = self.col_idx[i];
                let value = self.values[i];
                result.set(row, col, value);
            }
        }

        // Add values from other
        for row in 0..other_finalized.size {
            let start = other_finalized.row_ptr[row];
            let end = other_finalized.row_ptr[row + 1];

            for i in start..end {
                let col = other_finalized.col_idx[i];
                let value = other_finalized.values[i];
                let existing = result.get(row, col).unwrap_or(0.0);
                result.set(row, col, existing + value);
            }
        }

        result.finalize();
        Ok(result)
    }

    /// Frobenius norm of the sparse matrix
    pub fn frobenius_norm(&self) -> f64 {
        let mut sum_squares = 0.0;

        for row in 0..self.size {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                let value = self.values[i];
                sum_squares += value * value;
            }
        }

        sum_squares.sqrt()
    }

    /// Iterator over non-zero entries (row, col, value)
    pub fn iter_nonzeros(&mut self) -> SparseMatrixIterator<'_> {
        self.finalize();
        SparseMatrixIterator {
            matrix: self,
            current_row: 0,
            current_idx: 0,
        }
    }

    /// Scale the matrix by a scalar
    pub fn scale(&mut self, scalar: f64) {
        for value in &mut self.values {
            *value *= scalar;
        }

        for value in self.temp_map.values_mut() {
            *value *= scalar;
        }
    }
}

/// Iterator for sparse matrix non-zero entries
pub struct SparseMatrixIterator<'a> {
    matrix: &'a SparseKernelMatrix,
    current_row: usize,
    current_idx: usize,
}

impl<'a> Iterator for SparseMatrixIterator<'a> {
    type Item = (usize, usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_row < self.matrix.size {
            let row_end = self.matrix.row_ptr[self.current_row + 1];

            if self.current_idx < row_end {
                let col = self.matrix.col_idx[self.current_idx];
                let value = self.matrix.values[self.current_idx];
                self.current_idx += 1;
                return Some((self.current_row, col, value));
            }

            self.current_row += 1;
            self.current_idx = self
                .matrix
                .row_ptr
                .get(self.current_row)
                .copied()
                .unwrap_or(0);
        }

        None
    }
}

/// Parallel sparse kernel matrix builder
impl SparseKernelMatrixBuilder {
    /// Build sparse kernel matrix with parallel computation
    pub fn build_parallel(
        &self,
        data: &[Vec<f64>],
        kernel: &dyn Kernel,
    ) -> Result<SparseKernelMatrix> {
        use rayon::prelude::*;

        let n = data.len();
        let mut matrix = SparseKernelMatrix::new(n);

        // Compute rows in parallel
        let row_data: Vec<Vec<(usize, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row_entries = Vec::new();

                for j in 0..n {
                    match kernel.compute(&data[i], &data[j]) {
                        Ok(value) => {
                            if value.abs() >= self.threshold {
                                row_entries.push((j, value));
                            }
                        }
                        Err(_) => continue,
                    }
                }

                // If max_entries_per_row is set, keep only top-k entries
                if let Some(max_entries) = self.max_entries_per_row {
                    if row_entries.len() > max_entries {
                        row_entries
                            .sort_by(|(_, a), (_, b)| b.abs().partial_cmp(&a.abs()).unwrap());
                        row_entries.truncate(max_entries);
                    }
                }

                row_entries
            })
            .collect();

        // Sequentially insert into matrix
        for (i, row_entries) in row_data.into_iter().enumerate() {
            for (j, value) in row_entries {
                matrix.set(i, j, value);
            }
        }

        matrix.finalize();
        Ok(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_kernels::LinearKernel;

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

    #[test]
    fn test_sparse_matrix_spmv() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 0, 2.0);
        matrix.set(0, 2, 1.0);
        matrix.set(1, 1, 3.0);
        matrix.set(2, 0, 1.0);
        matrix.set(2, 2, 2.0);

        let x = vec![1.0, 2.0, 3.0];
        let y = matrix.spmv(&x).unwrap();

        assert_eq!(y.len(), 3);
        assert!((y[0] - 5.0).abs() < 1e-10); // 2*1 + 1*3
        assert!((y[1] - 6.0).abs() < 1e-10); // 3*2
        assert!((y[2] - 7.0).abs() < 1e-10); // 1*1 + 2*3
    }

    #[test]
    fn test_sparse_matrix_spmv_invalid_size() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 0, 1.0);

        let x = vec![1.0, 2.0]; // Wrong size
        let result = matrix.spmv(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_matrix_transpose() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 1, 0.8);
        matrix.set(1, 2, 0.6);
        matrix.set(2, 0, 0.4);
        matrix.finalize();

        let transposed = matrix.transpose().unwrap();

        assert_eq!(transposed.get(1, 0), Some(0.8));
        assert_eq!(transposed.get(2, 1), Some(0.6));
        assert_eq!(transposed.get(0, 2), Some(0.4));
    }

    #[test]
    fn test_sparse_matrix_add() {
        let mut matrix1 = SparseKernelMatrix::new(3);
        matrix1.set(0, 0, 1.0);
        matrix1.set(0, 1, 2.0);
        matrix1.set(1, 1, 3.0);

        let mut matrix2 = SparseKernelMatrix::new(3);
        matrix2.set(0, 1, 1.0);
        matrix2.set(1, 2, 4.0);
        matrix2.set(2, 2, 5.0);

        let result = matrix1.add(&matrix2).unwrap();

        assert_eq!(result.get(0, 0), Some(1.0));
        assert_eq!(result.get(0, 1), Some(3.0)); // 2.0 + 1.0
        assert_eq!(result.get(1, 1), Some(3.0));
        assert_eq!(result.get(1, 2), Some(4.0));
        assert_eq!(result.get(2, 2), Some(5.0));
    }

    #[test]
    fn test_sparse_matrix_add_invalid_size() {
        let mut matrix1 = SparseKernelMatrix::new(3);
        matrix1.set(0, 0, 1.0);

        let matrix2 = SparseKernelMatrix::new(2);
        let result = matrix1.add(&matrix2);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_matrix_frobenius_norm() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 0, 3.0);
        matrix.set(1, 1, 4.0);
        matrix.finalize();

        let norm = matrix.frobenius_norm();
        assert!((norm - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_sparse_matrix_iterator() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 1, 0.8);
        matrix.set(1, 2, 0.6);
        matrix.set(2, 0, 0.4);

        let entries: Vec<_> = matrix.iter_nonzeros().collect();

        assert_eq!(entries.len(), 3);
        assert!(entries.contains(&(0, 1, 0.8)));
        assert!(entries.contains(&(1, 2, 0.6)));
        assert!(entries.contains(&(2, 0, 0.4)));
    }

    #[test]
    fn test_sparse_matrix_scale() {
        let mut matrix = SparseKernelMatrix::new(3);
        matrix.set(0, 0, 2.0);
        matrix.set(1, 1, 4.0);
        matrix.finalize();

        matrix.scale(0.5);

        assert_eq!(matrix.get(0, 0), Some(1.0));
        assert_eq!(matrix.get(1, 1), Some(2.0));
    }

    #[test]
    fn test_sparse_matrix_builder_parallel() {
        let builder = SparseKernelMatrixBuilder::new();
        let kernel = LinearKernel::new();
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];

        let matrix = builder.build_parallel(&data, &kernel).unwrap();
        assert!(matrix.nnz() > 0);

        // Compare with sequential build
        let matrix_seq = builder.build(&data, &kernel).unwrap();
        assert_eq!(matrix.nnz(), matrix_seq.nnz());
    }

    #[test]
    fn test_sparse_matrix_parallel_with_threshold() {
        let builder = SparseKernelMatrixBuilder::new()
            .with_threshold(0.5)
            .unwrap();
        let kernel = LinearKernel::new();
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];

        let matrix = builder.build_parallel(&data, &kernel).unwrap();
        assert!(matrix.nnz() > 0);
    }
}
