//! Sparse tensor support for memory-efficient logical operations.
//!
//! This module provides sparse tensor representations optimized for logical
//! operations where tensors often contain many zeros (e.g., adjacency matrices,
//! predicate truth tables).
//!
//! # Supported Formats
//!
//! - **COO (Coordinate)**: Best for incremental construction and random access
//! - **CSR (Compressed Sparse Row)**: Best for row-wise operations and matrix-vector products
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_scirs_backend::sparse_tensor::{SparseTensor, SparseCOO, SparseFormat};
//!
//! // Create a sparse tensor from coordinates
//! let coo = SparseCOO::from_triplets(
//!     vec![3, 3],  // shape: 3x3
//!     vec![(0, 0, 1.0), (0, 2, 1.0), (1, 1, 1.0), (2, 2, 1.0)],
//! );
//!
//! let sparse = SparseTensor::from_coo(coo);
//!
//! // Check sparsity
//! println!("Sparsity: {:.1}%", sparse.sparsity() * 100.0);
//! println!("Memory savings: {:.1}%", sparse.memory_savings() * 100.0);
//! ```

use scirs2_core::ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;
use thiserror::Error;

/// Errors for sparse tensor operations.
#[derive(Debug, Error)]
pub enum SparseError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("Index out of bounds: index {index:?} for shape {shape:?}")]
    IndexOutOfBounds { index: Vec<usize>, shape: Vec<usize> },

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Empty tensor")]
    EmptyTensor,
}

/// Result type for sparse tensor operations.
pub type SparseResult<T> = Result<T, SparseError>;

/// Sparse storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Coordinate (triplet) format - good for construction
    COO,
    /// Compressed Sparse Row - good for row operations
    CSR,
    /// Compressed Sparse Column - good for column operations
    CSC,
}

impl Default for SparseFormat {
    fn default() -> Self {
        SparseFormat::COO
    }
}

/// Coordinate (COO) sparse matrix representation.
///
/// Stores non-zero values as (indices, value) pairs.
#[derive(Debug, Clone)]
pub struct SparseCOO {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Indices of non-zero elements (each inner vec is a coordinate)
    pub indices: Vec<Vec<usize>>,
    /// Values of non-zero elements
    pub values: Vec<f64>,
    /// Whether the indices are sorted
    sorted: bool,
}

impl SparseCOO {
    /// Create a new empty COO sparse tensor with the given shape.
    pub fn new(shape: Vec<usize>) -> Self {
        SparseCOO {
            shape,
            indices: Vec::new(),
            values: Vec::new(),
            sorted: true,
        }
    }

    /// Create a COO tensor from triplets (2D tensors only for simplicity).
    ///
    /// Each triplet is (row, col, value).
    pub fn from_triplets(shape: Vec<usize>, triplets: Vec<(usize, usize, f64)>) -> Self {
        let mut indices = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());

        for (row, col, val) in triplets {
            if val.abs() > 1e-15 {
                // Skip near-zero values
                indices.push(vec![row, col]);
                values.push(val);
            }
        }

        SparseCOO {
            shape,
            indices,
            values,
            sorted: false,
        }
    }

    /// Create from dense array, extracting non-zero values.
    pub fn from_dense(dense: &ArrayD<f64>) -> Self {
        let shape = dense.shape().to_vec();
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (idx, &val) in dense.indexed_iter() {
            if val.abs() > 1e-15 {
                indices.push(idx.as_array_view().to_vec());
                values.push(val);
            }
        }

        SparseCOO {
            shape,
            indices,
            values,
            sorted: false,
        }
    }

    /// Add a non-zero value at the given indices.
    pub fn add(&mut self, indices: Vec<usize>, value: f64) -> SparseResult<()> {
        // Validate indices
        if indices.len() != self.shape.len() {
            return Err(SparseError::IndexOutOfBounds {
                index: indices,
                shape: self.shape.clone(),
            });
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(SparseError::IndexOutOfBounds {
                    index: indices,
                    shape: self.shape.clone(),
                });
            }
        }

        if value.abs() > 1e-15 {
            self.indices.push(indices);
            self.values.push(value);
            self.sorted = false;
        }

        Ok(())
    }

    /// Get value at the given indices.
    pub fn get(&self, target: &[usize]) -> f64 {
        for (i, indices) in self.indices.iter().enumerate() {
            if indices == target {
                return self.values[i];
            }
        }
        0.0
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Sparsity ratio (fraction of zero elements).
    pub fn sparsity(&self) -> f64 {
        let total = self.numel();
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f64 / total as f64)
    }

    /// Convert to dense array.
    pub fn to_dense(&self) -> ArrayD<f64> {
        let mut dense = ArrayD::zeros(IxDyn(&self.shape));

        for (indices, &value) in self.indices.iter().zip(self.values.iter()) {
            dense[IxDyn(indices)] = value;
        }

        dense
    }

    /// Sort indices for efficient access (row-major order).
    pub fn sort(&mut self) {
        if self.sorted {
            return;
        }

        // Create index permutation
        let mut perm: Vec<usize> = (0..self.nnz()).collect();
        perm.sort_by(|&a, &b| self.indices[a].cmp(&self.indices[b]));

        // Apply permutation
        let new_indices: Vec<Vec<usize>> = perm.iter().map(|&i| self.indices[i].clone()).collect();
        let new_values: Vec<f64> = perm.iter().map(|&i| self.values[i]).collect();

        self.indices = new_indices;
        self.values = new_values;
        self.sorted = true;
    }

    /// Remove duplicate entries by summing values at the same location.
    pub fn sum_duplicates(&mut self) {
        if self.indices.is_empty() {
            return;
        }

        self.sort();

        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        let mut current_idx = self.indices[0].clone();
        let mut current_val = self.values[0];

        for i in 1..self.nnz() {
            if self.indices[i] == current_idx {
                current_val += self.values[i];
            } else {
                if current_val.abs() > 1e-15 {
                    new_indices.push(current_idx);
                    new_values.push(current_val);
                }
                current_idx = self.indices[i].clone();
                current_val = self.values[i];
            }
        }

        // Don't forget the last element
        if current_val.abs() > 1e-15 {
            new_indices.push(current_idx);
            new_values.push(current_val);
        }

        self.indices = new_indices;
        self.values = new_values;
    }
}

/// Compressed Sparse Row (CSR) matrix representation.
///
/// Efficient for row-wise operations and matrix-vector multiplication.
#[derive(Debug, Clone)]
pub struct SparseCSR {
    /// Shape of the matrix (must be 2D)
    pub shape: (usize, usize),
    /// Row pointers (length = nrows + 1)
    pub row_ptr: Vec<usize>,
    /// Column indices
    pub col_idx: Vec<usize>,
    /// Non-zero values
    pub values: Vec<f64>,
}

impl SparseCSR {
    /// Create a new empty CSR matrix.
    pub fn new(shape: (usize, usize)) -> Self {
        SparseCSR {
            shape,
            row_ptr: vec![0; shape.0 + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create from COO format (2D only).
    pub fn from_coo(coo: &SparseCOO) -> SparseResult<Self> {
        if coo.shape.len() != 2 {
            return Err(SparseError::InvalidFormat(
                "CSR only supports 2D tensors".to_string(),
            ));
        }

        let nrows = coo.shape[0];
        let ncols = coo.shape[1];

        // Count elements per row
        let mut row_counts = vec![0usize; nrows];
        for indices in &coo.indices {
            row_counts[indices[0]] += 1;
        }

        // Build row pointers
        let mut row_ptr = vec![0usize; nrows + 1];
        for i in 0..nrows {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }

        // Allocate arrays
        let nnz = coo.nnz();
        let mut col_idx = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];

        // Fill in values
        let mut row_offsets = row_ptr.clone();
        for (indices, &value) in coo.indices.iter().zip(coo.values.iter()) {
            let row = indices[0];
            let col = indices[1];
            let pos = row_offsets[row];
            col_idx[pos] = col;
            values[pos] = value;
            row_offsets[row] += 1;
        }

        // Sort columns within each row
        for row in 0..nrows {
            let start = row_ptr[row];
            let end = row_ptr[row + 1];
            if end > start {
                // Simple insertion sort for small segments
                for i in (start + 1)..end {
                    let mut j = i;
                    while j > start && col_idx[j - 1] > col_idx[j] {
                        col_idx.swap(j - 1, j);
                        values.swap(j - 1, j);
                        j -= 1;
                    }
                }
            }
        }

        Ok(SparseCSR {
            shape: (nrows, ncols),
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Create from dense matrix.
    pub fn from_dense(dense: &ArrayD<f64>) -> SparseResult<Self> {
        if dense.ndim() != 2 {
            return Err(SparseError::InvalidFormat(
                "CSR only supports 2D tensors".to_string(),
            ));
        }

        let coo = SparseCOO::from_dense(dense);
        Self::from_coo(&coo)
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get row as a sparse vector (column indices and values).
    pub fn get_row(&self, row: usize) -> Option<(&[usize], &[f64])> {
        if row >= self.shape.0 {
            return None;
        }

        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        Some((&self.col_idx[start..end], &self.values[start..end]))
    }

    /// Get value at (row, col).
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.shape.0 || col >= self.shape.1 {
            return 0.0;
        }

        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        // Binary search in sorted column indices
        match self.col_idx[start..end].binary_search(&col) {
            Ok(pos) => self.values[start + pos],
            Err(_) => 0.0,
        }
    }

    /// Convert to dense array.
    pub fn to_dense(&self) -> ArrayD<f64> {
        let mut dense = ArrayD::zeros(IxDyn(&[self.shape.0, self.shape.1]));

        for row in 0..self.shape.0 {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for (col_pos, &col) in self.col_idx[start..end].iter().enumerate() {
                dense[[row, col]] = self.values[start + col_pos];
            }
        }

        dense
    }

    /// Convert to COO format.
    pub fn to_coo(&self) -> SparseCOO {
        let mut indices = Vec::with_capacity(self.nnz());
        let mut values = Vec::with_capacity(self.nnz());

        for row in 0..self.shape.0 {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for (i, &col) in self.col_idx[start..end].iter().enumerate() {
                indices.push(vec![row, col]);
                values.push(self.values[start + i]);
            }
        }

        SparseCOO {
            shape: vec![self.shape.0, self.shape.1],
            indices,
            values,
            sorted: true,
        }
    }

    /// Sparse matrix-vector multiplication: y = A * x
    pub fn matvec(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
        if x.len() != self.shape.1 {
            return Err(SparseError::ShapeMismatch {
                expected: vec![self.shape.1],
                got: vec![x.len()],
            });
        }

        let mut y = vec![0.0; self.shape.0];

        for row in 0..self.shape.0 {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for (i, &col) in self.col_idx[start..end].iter().enumerate() {
                y[row] += self.values[start + i] * x[col];
            }
        }

        Ok(y)
    }
}

/// Unified sparse tensor type with multiple format support.
#[derive(Debug, Clone)]
pub struct SparseTensor {
    /// The underlying sparse format
    format: SparseFormat,
    /// COO representation (always available)
    coo: SparseCOO,
    /// CSR representation (computed on demand)
    csr: Option<SparseCSR>,
}

impl SparseTensor {
    /// Create from COO representation.
    pub fn from_coo(coo: SparseCOO) -> Self {
        SparseTensor {
            format: SparseFormat::COO,
            coo,
            csr: None,
        }
    }

    /// Create from CSR representation.
    pub fn from_csr(csr: SparseCSR) -> Self {
        let coo = csr.to_coo();
        SparseTensor {
            format: SparseFormat::CSR,
            coo,
            csr: Some(csr),
        }
    }

    /// Create from dense tensor.
    pub fn from_dense(dense: &ArrayD<f64>) -> Self {
        let coo = SparseCOO::from_dense(dense);
        SparseTensor {
            format: SparseFormat::COO,
            coo,
            csr: None,
        }
    }

    /// Create an empty sparse tensor.
    pub fn zeros(shape: Vec<usize>) -> Self {
        SparseTensor {
            format: SparseFormat::COO,
            coo: SparseCOO::new(shape),
            csr: None,
        }
    }

    /// Create a sparse identity matrix.
    pub fn eye(n: usize) -> Self {
        let triplets: Vec<_> = (0..n).map(|i| (i, i, 1.0)).collect();
        let coo = SparseCOO::from_triplets(vec![n, n], triplets);
        SparseTensor::from_coo(coo)
    }

    /// Get the shape.
    pub fn shape(&self) -> &[usize] {
        &self.coo.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.coo.shape.len()
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.coo.nnz()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.coo.numel()
    }

    /// Sparsity ratio (0.0 = all non-zero, 1.0 = all zero).
    pub fn sparsity(&self) -> f64 {
        self.coo.sparsity()
    }

    /// Memory savings compared to dense storage.
    pub fn memory_savings(&self) -> f64 {
        let dense_bytes = self.numel() * 8; // f64 = 8 bytes
        let sparse_bytes = self.nnz() * (8 + self.ndim() * 8); // value + indices
        if dense_bytes == 0 {
            return 0.0;
        }
        1.0 - (sparse_bytes as f64 / dense_bytes as f64)
    }

    /// Convert to dense tensor.
    pub fn to_dense(&self) -> ArrayD<f64> {
        self.coo.to_dense()
    }

    /// Get the COO representation.
    pub fn as_coo(&self) -> &SparseCOO {
        &self.coo
    }

    /// Get CSR representation (computes if not available).
    pub fn as_csr(&mut self) -> SparseResult<&SparseCSR> {
        if self.csr.is_none() {
            self.csr = Some(SparseCSR::from_coo(&self.coo)?);
        }
        Ok(self.csr.as_ref().unwrap())
    }

    /// Get value at the given indices.
    pub fn get(&self, indices: &[usize]) -> f64 {
        self.coo.get(indices)
    }

    /// Element-wise operation on sparse tensor.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let new_values: Vec<f64> = self.coo.values.iter().map(|&v| f(v)).collect();

        // Filter out zeros that may have been created
        let mut new_indices = Vec::new();
        let mut filtered_values = Vec::new();

        for (indices, &value) in self.coo.indices.iter().zip(new_values.iter()) {
            if value.abs() > 1e-15 {
                new_indices.push(indices.clone());
                filtered_values.push(value);
            }
        }

        let coo = SparseCOO {
            shape: self.coo.shape.clone(),
            indices: new_indices,
            values: filtered_values,
            sorted: self.coo.sorted,
        };

        SparseTensor::from_coo(coo)
    }

    /// Scale all values by a constant.
    pub fn scale(&self, scalar: f64) -> Self {
        if scalar.abs() < 1e-15 {
            return SparseTensor::zeros(self.coo.shape.clone());
        }

        let new_values: Vec<f64> = self.coo.values.iter().map(|&v| v * scalar).collect();

        let coo = SparseCOO {
            shape: self.coo.shape.clone(),
            indices: self.coo.indices.clone(),
            values: new_values,
            sorted: self.coo.sorted,
        };

        SparseTensor::from_coo(coo)
    }

    /// Add two sparse tensors.
    pub fn add(&self, other: &SparseTensor) -> SparseResult<SparseTensor> {
        if self.coo.shape != other.coo.shape {
            return Err(SparseError::ShapeMismatch {
                expected: self.coo.shape.clone(),
                got: other.coo.shape.clone(),
            });
        }

        // Collect all values into a hashmap
        let mut values_map: HashMap<Vec<usize>, f64> = HashMap::new();

        for (indices, &value) in self.coo.indices.iter().zip(self.coo.values.iter()) {
            *values_map.entry(indices.clone()).or_insert(0.0) += value;
        }

        for (indices, &value) in other.coo.indices.iter().zip(other.coo.values.iter()) {
            *values_map.entry(indices.clone()).or_insert(0.0) += value;
        }

        // Filter out zeros
        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        for (indices, value) in values_map {
            if value.abs() > 1e-15 {
                new_indices.push(indices);
                new_values.push(value);
            }
        }

        let mut coo = SparseCOO {
            shape: self.coo.shape.clone(),
            indices: new_indices,
            values: new_values,
            sorted: false,
        };

        coo.sort();

        Ok(SparseTensor::from_coo(coo))
    }

    /// Element-wise multiplication (Hadamard product).
    pub fn hadamard(&self, other: &SparseTensor) -> SparseResult<SparseTensor> {
        if self.coo.shape != other.coo.shape {
            return Err(SparseError::ShapeMismatch {
                expected: self.coo.shape.clone(),
                got: other.coo.shape.clone(),
            });
        }

        // Build index map for the other tensor
        let other_map: HashMap<&Vec<usize>, f64> = other
            .coo
            .indices
            .iter()
            .zip(other.coo.values.iter())
            .map(|(idx, &val)| (idx, val))
            .collect();

        // Multiply matching elements
        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        for (indices, &value) in self.coo.indices.iter().zip(self.coo.values.iter()) {
            if let Some(&other_value) = other_map.get(indices) {
                let product = value * other_value;
                if product.abs() > 1e-15 {
                    new_indices.push(indices.clone());
                    new_values.push(product);
                }
            }
        }

        let coo = SparseCOO {
            shape: self.coo.shape.clone(),
            indices: new_indices,
            values: new_values,
            sorted: self.coo.sorted,
        };

        Ok(SparseTensor::from_coo(coo))
    }

    /// Element-wise maximum (useful for OR operations).
    pub fn maximum(&self, other: &SparseTensor) -> SparseResult<SparseTensor> {
        if self.coo.shape != other.coo.shape {
            return Err(SparseError::ShapeMismatch {
                expected: self.coo.shape.clone(),
                got: other.coo.shape.clone(),
            });
        }

        let mut values_map: HashMap<Vec<usize>, f64> = HashMap::new();

        for (indices, &value) in self.coo.indices.iter().zip(self.coo.values.iter()) {
            values_map
                .entry(indices.clone())
                .and_modify(|v| *v = v.max(value))
                .or_insert(value);
        }

        for (indices, &value) in other.coo.indices.iter().zip(other.coo.values.iter()) {
            values_map
                .entry(indices.clone())
                .and_modify(|v| *v = v.max(value))
                .or_insert(value);
        }

        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        for (indices, value) in values_map {
            if value.abs() > 1e-15 {
                new_indices.push(indices);
                new_values.push(value);
            }
        }

        let mut coo = SparseCOO {
            shape: self.coo.shape.clone(),
            indices: new_indices,
            values: new_values,
            sorted: false,
        };

        coo.sort();

        Ok(SparseTensor::from_coo(coo))
    }

    /// Sum reduction over specified axes.
    pub fn sum(&self, axes: &[usize]) -> SparseResult<SparseTensor> {
        if axes.is_empty() {
            return Ok(self.clone());
        }

        for &axis in axes {
            if axis >= self.ndim() {
                return Err(SparseError::IndexOutOfBounds {
                    index: vec![axis],
                    shape: self.coo.shape.clone(),
                });
            }
        }

        // Compute new shape
        let kept_dims: Vec<usize> = (0..self.ndim()).filter(|d| !axes.contains(d)).collect();
        let new_shape: Vec<usize> = kept_dims.iter().map(|&d| self.coo.shape[d]).collect();

        if new_shape.is_empty() {
            // Summing all axes - return scalar
            let total: f64 = self.coo.values.iter().sum();
            let mut coo = SparseCOO::new(vec![]);
            if total.abs() > 1e-15 {
                coo.indices.push(vec![]);
                coo.values.push(total);
            }
            return Ok(SparseTensor::from_coo(coo));
        }

        // Aggregate values
        let mut values_map: HashMap<Vec<usize>, f64> = HashMap::new();

        for (indices, &value) in self.coo.indices.iter().zip(self.coo.values.iter()) {
            let new_indices: Vec<usize> = kept_dims.iter().map(|&d| indices[d]).collect();
            *values_map.entry(new_indices).or_insert(0.0) += value;
        }

        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        for (indices, value) in values_map {
            if value.abs() > 1e-15 {
                new_indices.push(indices);
                new_values.push(value);
            }
        }

        let mut coo = SparseCOO {
            shape: new_shape,
            indices: new_indices,
            values: new_values,
            sorted: false,
        };

        coo.sort();

        Ok(SparseTensor::from_coo(coo))
    }

    /// Max reduction over specified axes.
    pub fn max(&self, axes: &[usize]) -> SparseResult<SparseTensor> {
        if axes.is_empty() {
            return Ok(self.clone());
        }

        for &axis in axes {
            if axis >= self.ndim() {
                return Err(SparseError::IndexOutOfBounds {
                    index: vec![axis],
                    shape: self.coo.shape.clone(),
                });
            }
        }

        let kept_dims: Vec<usize> = (0..self.ndim()).filter(|d| !axes.contains(d)).collect();
        let new_shape: Vec<usize> = kept_dims.iter().map(|&d| self.coo.shape[d]).collect();

        if new_shape.is_empty() {
            // Max over all axes
            let max_val = self
                .coo
                .values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut coo = SparseCOO::new(vec![]);
            if max_val.abs() > 1e-15 && max_val.is_finite() {
                coo.indices.push(vec![]);
                coo.values.push(max_val);
            }
            return Ok(SparseTensor::from_coo(coo));
        }

        let mut values_map: HashMap<Vec<usize>, f64> = HashMap::new();

        for (indices, &value) in self.coo.indices.iter().zip(self.coo.values.iter()) {
            let new_indices: Vec<usize> = kept_dims.iter().map(|&d| indices[d]).collect();
            values_map
                .entry(new_indices)
                .and_modify(|v| *v = v.max(value))
                .or_insert(value);
        }

        let mut new_indices = Vec::new();
        let mut new_values = Vec::new();

        for (indices, value) in values_map {
            if value.abs() > 1e-15 {
                new_indices.push(indices);
                new_values.push(value);
            }
        }

        let mut coo = SparseCOO {
            shape: new_shape,
            indices: new_indices,
            values: new_values,
            sorted: false,
        };

        coo.sort();

        Ok(SparseTensor::from_coo(coo))
    }
}

/// Configuration for sparse tensor operations.
#[derive(Debug, Clone)]
pub struct SparseConfig {
    /// Threshold below which values are considered zero
    pub zero_threshold: f64,
    /// Sparsity threshold above which sparse format is preferred
    pub sparsity_threshold: f64,
    /// Preferred format for new tensors
    pub preferred_format: SparseFormat,
}

impl Default for SparseConfig {
    fn default() -> Self {
        SparseConfig {
            zero_threshold: 1e-15,
            sparsity_threshold: 0.5,
            preferred_format: SparseFormat::COO,
        }
    }
}

impl SparseConfig {
    /// Check if a dense tensor should be stored as sparse.
    pub fn should_use_sparse(&self, dense: &ArrayD<f64>) -> bool {
        let total = dense.len();
        if total == 0 {
            return false;
        }

        let nnz = dense.iter().filter(|&&v| v.abs() > self.zero_threshold).count();
        let sparsity = 1.0 - (nnz as f64 / total as f64);

        sparsity > self.sparsity_threshold
    }
}

/// Statistics about sparse tensor storage.
#[derive(Debug, Clone)]
pub struct SparseStats {
    /// Number of non-zero elements
    pub nnz: usize,
    /// Total number of elements
    pub numel: usize,
    /// Sparsity ratio
    pub sparsity: f64,
    /// Memory used by sparse representation (bytes)
    pub sparse_bytes: usize,
    /// Memory that would be used by dense representation (bytes)
    pub dense_bytes: usize,
    /// Memory savings ratio
    pub savings: f64,
}

impl SparseStats {
    /// Compute statistics for a sparse tensor.
    pub fn from_tensor(tensor: &SparseTensor) -> Self {
        let nnz = tensor.nnz();
        let numel = tensor.numel();
        let sparsity = tensor.sparsity();

        let dense_bytes = numel * 8;
        let sparse_bytes = nnz * (8 + tensor.ndim() * 8);
        let savings = if dense_bytes > 0 {
            1.0 - (sparse_bytes as f64 / dense_bytes as f64)
        } else {
            0.0
        };

        SparseStats {
            nnz,
            numel,
            sparsity,
            sparse_bytes,
            dense_bytes,
            savings,
        }
    }
}

impl std::fmt::Display for SparseStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SparseStats {{ nnz: {}, numel: {}, sparsity: {:.1}%, savings: {:.1}% }}",
            self.nnz,
            self.numel,
            self.sparsity * 100.0,
            self.savings * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_from_triplets() {
        let coo = SparseCOO::from_triplets(vec![3, 3], vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]);

        assert_eq!(coo.shape, vec![3, 3]);
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.get(&[0, 0]), 1.0);
        assert_eq!(coo.get(&[1, 1]), 2.0);
        assert_eq!(coo.get(&[2, 2]), 3.0);
        assert_eq!(coo.get(&[0, 1]), 0.0); // Not present
    }

    #[test]
    fn test_coo_from_dense() {
        let dense = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
            .unwrap();

        let coo = SparseCOO::from_dense(&dense);

        assert_eq!(coo.shape, vec![2, 3]);
        assert_eq!(coo.nnz(), 3);
        assert!((coo.sparsity() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_coo_to_dense() {
        let coo =
            SparseCOO::from_triplets(vec![2, 2], vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)]);

        let dense = coo.to_dense();

        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[0, 1]], 2.0);
        assert_eq!(dense[[1, 0]], 3.0);
        assert_eq!(dense[[1, 1]], 0.0);
    }

    #[test]
    fn test_csr_from_coo() {
        let coo = SparseCOO::from_triplets(
            vec![3, 4],
            vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 3, 4.0)],
        );

        let csr = SparseCSR::from_coo(&coo).unwrap();

        assert_eq!(csr.shape, (3, 4));
        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 2), 2.0);
        assert_eq!(csr.get(1, 1), 3.0);
        assert_eq!(csr.get(2, 3), 4.0);
        assert_eq!(csr.get(0, 1), 0.0);
    }

    #[test]
    fn test_csr_matvec() {
        // Matrix: [[1, 2], [3, 4]]
        let coo = SparseCOO::from_triplets(
            vec![2, 2],
            vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)],
        );
        let csr = SparseCSR::from_coo(&coo).unwrap();

        // Vector: [1, 2]
        let x = vec![1.0, 2.0];
        let y = csr.matvec(&x).unwrap();

        // Expected: [1*1 + 2*2, 3*1 + 4*2] = [5, 11]
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_tensor_add() {
        let a = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![2, 2],
            vec![(0, 0, 1.0), (0, 1, 2.0)],
        ));
        let b = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![2, 2],
            vec![(0, 0, 3.0), (1, 1, 4.0)],
        ));

        let c = a.add(&b).unwrap();

        assert_eq!(c.get(&[0, 0]), 4.0); // 1 + 3
        assert_eq!(c.get(&[0, 1]), 2.0); // 2 + 0
        assert_eq!(c.get(&[1, 1]), 4.0); // 0 + 4
    }

    #[test]
    fn test_sparse_tensor_hadamard() {
        let a = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![2, 2],
            vec![(0, 0, 2.0), (0, 1, 3.0), (1, 0, 4.0)],
        ));
        let b = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![2, 2],
            vec![(0, 0, 5.0), (1, 0, 2.0), (1, 1, 3.0)],
        ));

        let c = a.hadamard(&b).unwrap();

        assert_eq!(c.get(&[0, 0]), 10.0); // 2 * 5
        assert_eq!(c.get(&[0, 1]), 0.0); // 3 * 0
        assert_eq!(c.get(&[1, 0]), 8.0); // 4 * 2
        assert_eq!(c.get(&[1, 1]), 0.0); // 0 * 3
    }

    #[test]
    fn test_sparse_tensor_maximum() {
        let a = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![2, 2],
            vec![(0, 0, 1.0), (0, 1, 5.0)],
        ));
        let b = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![2, 2],
            vec![(0, 0, 3.0), (1, 1, 4.0)],
        ));

        let c = a.maximum(&b).unwrap();

        assert_eq!(c.get(&[0, 0]), 3.0); // max(1, 3)
        assert_eq!(c.get(&[0, 1]), 5.0); // max(5, 0)
        assert_eq!(c.get(&[1, 1]), 4.0); // max(0, 4)
    }

    #[test]
    fn test_sparse_tensor_sum() {
        let sparse = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![2, 3],
            vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 2, 4.0)],
        ));

        // Sum over axis 1
        let summed = sparse.sum(&[1]).unwrap();
        assert_eq!(summed.shape(), &[2]);
        assert_eq!(summed.get(&[0]), 3.0); // 1 + 2
        assert_eq!(summed.get(&[1]), 7.0); // 3 + 4
    }

    #[test]
    fn test_sparse_tensor_scale() {
        let sparse = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![2, 2],
            vec![(0, 0, 2.0), (1, 1, 3.0)],
        ));

        let scaled = sparse.scale(2.0);

        assert_eq!(scaled.get(&[0, 0]), 4.0);
        assert_eq!(scaled.get(&[1, 1]), 6.0);
    }

    #[test]
    fn test_sparse_tensor_map() {
        let sparse = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![2, 2],
            vec![(0, 0, 4.0), (1, 1, 9.0)],
        ));

        let mapped = sparse.map(|x| x.sqrt());

        assert!((mapped.get(&[0, 0]) - 2.0).abs() < 1e-10);
        assert!((mapped.get(&[1, 1]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_tensor_eye() {
        let eye = SparseTensor::eye(3);

        assert_eq!(eye.shape(), &[3, 3]);
        assert_eq!(eye.nnz(), 3);
        assert_eq!(eye.get(&[0, 0]), 1.0);
        assert_eq!(eye.get(&[1, 1]), 1.0);
        assert_eq!(eye.get(&[2, 2]), 1.0);
        assert_eq!(eye.get(&[0, 1]), 0.0);
    }

    #[test]
    fn test_sparse_stats() {
        let sparse = SparseTensor::from_coo(SparseCOO::from_triplets(
            vec![10, 10],
            vec![(0, 0, 1.0), (5, 5, 2.0)],
        ));

        let stats = SparseStats::from_tensor(&sparse);

        assert_eq!(stats.nnz, 2);
        assert_eq!(stats.numel, 100);
        assert!((stats.sparsity - 0.98).abs() < 0.01);
        assert!(stats.savings > 0.5); // Should have good savings for sparse tensor
    }

    #[test]
    fn test_sparse_config() {
        let config = SparseConfig::default();

        // 90% zeros should be sparse
        let sparse_data = Array::from_shape_vec(
            IxDyn(&[10]),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        assert!(config.should_use_sparse(&sparse_data));

        // 50% zeros should not be sparse (at threshold)
        let dense_data =
            Array::from_shape_vec(IxDyn(&[4]), vec![1.0, 0.0, 2.0, 0.0]).unwrap();
        assert!(!config.should_use_sparse(&dense_data));
    }
}
