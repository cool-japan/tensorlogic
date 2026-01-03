//! Sparse tensor support for TensorLogic.
//!
//! This module provides comprehensive sparse tensor representations and operations:
//! - **CSR** (Compressed Sparse Row) format for efficient row operations
//! - **CSC** (Compressed Sparse Column) format for efficient column operations
//! - **COO** (Coordinate) format for flexible construction
//! - **Sparse-dense hybrid operations**
//! - **Automatic sparsity detection and conversion**
//! - **Sparse matrix multiplication and linear algebra**
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{SparseFormat, SparseTensor, SparseMatrix};
//!
//! // Create a sparse matrix in COO format
//! let mut builder = SparseTensor::builder(vec![100, 100], SparseFormat::COO);
//! builder.add_entry(vec![5, 10], 3.14);
//! builder.add_entry(vec![20, 30], 2.71);
//! let sparse = builder.build()?;
//!
//! // Convert to CSR for efficient operations
//! let csr = sparse.to_csr()?;
//!
//! // Sparse-dense multiplication
//! let dense = vec![1.0; 100];
//! let result = csr.multiply_dense(&dense)?;
//!
//! // Detect sparsity
//! let sparsity = sparse.sparsity_ratio();
//! println!("Sparsity: {:.2}%", sparsity * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Sparse tensor errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SparseError {
    #[error("Invalid sparse format conversion: {0} -> {1}")]
    InvalidConversion(String, String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Index out of bounds: {index:?} for shape {shape:?}")]
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },

    #[error("Invalid sparse tensor: {0}")]
    Invalid(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Empty sparse tensor")]
    Empty,
}

/// Sparse tensor storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SparseFormat {
    /// Compressed Sparse Row (CSR) - efficient for row-wise operations
    CSR,
    /// Compressed Sparse Column (CSC) - efficient for column-wise operations
    CSC,
    /// Coordinate (COO) - flexible for construction
    COO,
}

impl SparseFormat {
    /// Get the format name.
    pub fn name(&self) -> &'static str {
        match self {
            SparseFormat::CSR => "CSR",
            SparseFormat::CSC => "CSC",
            SparseFormat::COO => "COO",
        }
    }

    /// Check if this format is compressed.
    pub fn is_compressed(&self) -> bool {
        matches!(self, SparseFormat::CSR | SparseFormat::CSC)
    }
}

/// Sparse matrix in CSR (Compressed Sparse Row) format.
///
/// Storage: O(nnz) where nnz is the number of non-zero elements
/// Row access: O(1)
/// Column access: O(nnz)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseCSR {
    /// Shape of the matrix (rows, cols)
    pub shape: (usize, usize),
    /// Row pointers (length = rows + 1)
    pub row_ptr: Vec<usize>,
    /// Column indices for each non-zero entry
    pub col_indices: Vec<usize>,
    /// Values for each non-zero entry
    pub values: Vec<f64>,
}

impl SparseCSR {
    /// Create a new empty CSR matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            shape: (rows, cols),
            row_ptr: vec![0; rows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Get the number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio (fraction of zero elements).
    pub fn sparsity_ratio(&self) -> f64 {
        let total = self.shape.0 * self.shape.1;
        1.0 - (self.nnz() as f64 / total as f64)
    }

    /// Get a row slice.
    pub fn row(&self, row_idx: usize) -> Result<Vec<(usize, f64)>, SparseError> {
        if row_idx >= self.shape.0 {
            return Err(SparseError::IndexOutOfBounds {
                index: vec![row_idx],
                shape: vec![self.shape.0],
            });
        }

        let start = self.row_ptr[row_idx];
        let end = self.row_ptr[row_idx + 1];

        Ok((start..end)
            .map(|i| (self.col_indices[i], self.values[i]))
            .collect())
    }

    /// Multiply with a dense vector (matrix-vector multiplication).
    pub fn multiply_dense(&self, vec: &[f64]) -> Result<Vec<f64>, SparseError> {
        if vec.len() != self.shape.1 {
            return Err(SparseError::ShapeMismatch {
                expected: vec![self.shape.1],
                actual: vec![vec.len()],
            });
        }

        let mut result = vec![0.0; self.shape.0];

        for row_idx in 0..self.shape.0 {
            let start = self.row_ptr[row_idx];
            let end = self.row_ptr[row_idx + 1];

            let mut sum = 0.0;
            for i in start..end {
                sum += self.values[i] * vec[self.col_indices[i]];
            }
            result[row_idx] = sum;
        }

        Ok(result)
    }

    /// Transpose to CSC format.
    pub fn transpose(&self) -> SparseCSC {
        let mut csc = SparseCSC::new(self.shape.1, self.shape.0);
        csc.col_ptr = vec![0; self.shape.1 + 1];

        // Count entries per column
        let mut counts = vec![0; self.shape.1];
        for &col in &self.col_indices {
            counts[col] += 1;
        }

        // Build column pointers
        let mut sum = 0;
        for i in 0..self.shape.1 {
            csc.col_ptr[i] = sum;
            sum += counts[i];
        }
        csc.col_ptr[self.shape.1] = sum;

        // Fill in entries
        csc.row_indices = vec![0; self.nnz()];
        csc.values = vec![0.0; self.nnz()];
        let mut positions = csc.col_ptr[..self.shape.1].to_vec();

        for row in 0..self.shape.0 {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for i in start..end {
                let col = self.col_indices[i];
                let pos = positions[col];
                csc.row_indices[pos] = row;
                csc.values[pos] = self.values[i];
                positions[col] += 1;
            }
        }

        csc
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<usize>()
            + self.col_indices.len() * std::mem::size_of::<usize>()
            + self.values.len() * std::mem::size_of::<f64>()
    }

    /// Validate the CSR structure.
    pub fn validate(&self) -> Result<(), SparseError> {
        // Check row pointers
        if self.row_ptr.len() != self.shape.0 + 1 {
            return Err(SparseError::Invalid(format!(
                "Invalid row_ptr length: expected {}, got {}",
                self.shape.0 + 1,
                self.row_ptr.len()
            )));
        }

        // Check monotonicity
        for i in 0..self.shape.0 {
            if self.row_ptr[i] > self.row_ptr[i + 1] {
                return Err(SparseError::Invalid(format!(
                    "Non-monotonic row_ptr at index {}",
                    i
                )));
            }
        }

        // Check bounds
        if self.row_ptr[self.shape.0] != self.nnz() {
            return Err(SparseError::Invalid(format!(
                "Last row_ptr {} doesn't match nnz {}",
                self.row_ptr[self.shape.0],
                self.nnz()
            )));
        }

        // Check column indices
        for &col in &self.col_indices {
            if col >= self.shape.1 {
                return Err(SparseError::IndexOutOfBounds {
                    index: vec![0, col],
                    shape: vec![self.shape.0, self.shape.1],
                });
            }
        }

        Ok(())
    }
}

/// Sparse matrix in CSC (Compressed Sparse Column) format.
///
/// Storage: O(nnz) where nnz is the number of non-zero elements
/// Row access: O(nnz)
/// Column access: O(1)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseCSC {
    /// Shape of the matrix (rows, cols)
    pub shape: (usize, usize),
    /// Column pointers (length = cols + 1)
    pub col_ptr: Vec<usize>,
    /// Row indices for each non-zero entry
    pub row_indices: Vec<usize>,
    /// Values for each non-zero entry
    pub values: Vec<f64>,
}

impl SparseCSC {
    /// Create a new empty CSC matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            shape: (rows, cols),
            col_ptr: vec![0; cols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Get the number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio.
    pub fn sparsity_ratio(&self) -> f64 {
        let total = self.shape.0 * self.shape.1;
        1.0 - (self.nnz() as f64 / total as f64)
    }

    /// Get a column slice.
    pub fn column(&self, col_idx: usize) -> Result<Vec<(usize, f64)>, SparseError> {
        if col_idx >= self.shape.1 {
            return Err(SparseError::IndexOutOfBounds {
                index: vec![col_idx],
                shape: vec![self.shape.1],
            });
        }

        let start = self.col_ptr[col_idx];
        let end = self.col_ptr[col_idx + 1];

        Ok((start..end)
            .map(|i| (self.row_indices[i], self.values[i]))
            .collect())
    }

    /// Transpose to CSR format.
    pub fn transpose(&self) -> SparseCSR {
        let mut csr = SparseCSR::new(self.shape.1, self.shape.0);
        csr.row_ptr = self.col_ptr.clone();
        csr.col_indices = self.row_indices.clone();
        csr.values = self.values.clone();
        csr
    }
}

/// Sparse matrix in COO (Coordinate) format.
///
/// Storage: O(nnz) where nnz is the number of non-zero elements
/// Random access: O(nnz)
/// Best for: Construction and modification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseCOO {
    /// Shape of the matrix (rows, cols)
    pub shape: (usize, usize),
    /// Row indices
    pub row_indices: Vec<usize>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Values
    pub values: Vec<f64>,
}

impl SparseCOO {
    /// Create a new empty COO matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            shape: (rows, cols),
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Add a non-zero entry.
    pub fn add_entry(&mut self, row: usize, col: usize, value: f64) -> Result<(), SparseError> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return Err(SparseError::IndexOutOfBounds {
                index: vec![row, col],
                shape: vec![self.shape.0, self.shape.1],
            });
        }

        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);

        Ok(())
    }

    /// Get the number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get sparsity ratio.
    pub fn sparsity_ratio(&self) -> f64 {
        let total = self.shape.0 * self.shape.1;
        1.0 - (self.nnz() as f64 / total as f64)
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> SparseCSR {
        let mut csr = SparseCSR::new(self.shape.0, self.shape.1);

        // Create a sorted list of (row, col, value) tuples
        let mut entries: Vec<_> = (0..self.nnz())
            .map(|i| (self.row_indices[i], self.col_indices[i], self.values[i]))
            .collect();
        entries.sort_by_key(|(r, c, _)| (*r, *c));

        // Build CSR structure
        csr.row_ptr = vec![0; self.shape.0 + 1];
        csr.col_indices = Vec::with_capacity(entries.len());
        csr.values = Vec::with_capacity(entries.len());

        let mut current_row = 0;
        for (row, col, val) in entries {
            while current_row < row {
                current_row += 1;
                csr.row_ptr[current_row] = csr.col_indices.len();
            }
            csr.col_indices.push(col);
            csr.values.push(val);
        }

        // Fill remaining row pointers
        for i in current_row + 1..=self.shape.0 {
            csr.row_ptr[i] = csr.col_indices.len();
        }

        csr
    }

    /// Convert to CSC format.
    pub fn to_csc(&self) -> SparseCSC {
        let mut csc = SparseCSC::new(self.shape.0, self.shape.1);

        // Create a sorted list of (col, row, value) tuples
        let mut entries: Vec<_> = (0..self.nnz())
            .map(|i| (self.col_indices[i], self.row_indices[i], self.values[i]))
            .collect();
        entries.sort_by_key(|(c, r, _)| (*c, *r));

        // Build CSC structure
        csc.col_ptr = vec![0; self.shape.1 + 1];
        csc.row_indices = Vec::with_capacity(entries.len());
        csc.values = Vec::with_capacity(entries.len());

        let mut current_col = 0;
        for (col, row, val) in entries {
            while current_col < col {
                current_col += 1;
                csc.col_ptr[current_col] = csc.row_indices.len();
            }
            csc.row_indices.push(row);
            csc.values.push(val);
        }

        // Fill remaining column pointers
        for i in current_col + 1..=self.shape.1 {
            csc.col_ptr[i] = csc.row_indices.len();
        }

        csc
    }
}

/// Sparse tensor representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SparseTensor {
    /// 2D matrix in CSR format
    CSR(SparseCSR),
    /// 2D matrix in CSC format
    CSC(SparseCSC),
    /// 2D matrix in COO format
    COO(SparseCOO),
}

impl SparseTensor {
    /// Create a sparse tensor builder.
    pub fn builder(shape: Vec<usize>, format: SparseFormat) -> SparseTensorBuilder {
        SparseTensorBuilder::new(shape, format)
    }

    /// Get the sparse format.
    pub fn format(&self) -> SparseFormat {
        match self {
            SparseTensor::CSR(_) => SparseFormat::CSR,
            SparseTensor::CSC(_) => SparseFormat::CSC,
            SparseTensor::COO(_) => SparseFormat::COO,
        }
    }

    /// Get the shape.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            SparseTensor::CSR(m) => vec![m.shape.0, m.shape.1],
            SparseTensor::CSC(m) => vec![m.shape.0, m.shape.1],
            SparseTensor::COO(m) => vec![m.shape.0, m.shape.1],
        }
    }

    /// Get the number of non-zero elements.
    pub fn nnz(&self) -> usize {
        match self {
            SparseTensor::CSR(m) => m.nnz(),
            SparseTensor::CSC(m) => m.nnz(),
            SparseTensor::COO(m) => m.nnz(),
        }
    }

    /// Get sparsity ratio.
    pub fn sparsity_ratio(&self) -> f64 {
        match self {
            SparseTensor::CSR(m) => m.sparsity_ratio(),
            SparseTensor::CSC(m) => m.sparsity_ratio(),
            SparseTensor::COO(m) => m.sparsity_ratio(),
        }
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> Result<SparseTensor, SparseError> {
        match self {
            SparseTensor::CSR(_) => Ok(self.clone()),
            SparseTensor::CSC(m) => Ok(SparseTensor::CSR(m.transpose())),
            SparseTensor::COO(m) => Ok(SparseTensor::CSR(m.to_csr())),
        }
    }

    /// Convert to CSC format.
    pub fn to_csc(&self) -> Result<SparseTensor, SparseError> {
        match self {
            SparseTensor::CSR(m) => Ok(SparseTensor::CSC(m.transpose())),
            SparseTensor::CSC(_) => Ok(self.clone()),
            SparseTensor::COO(m) => Ok(SparseTensor::CSC(m.to_csc())),
        }
    }

    /// Convert to COO format.
    pub fn to_coo(&self) -> Result<SparseTensor, SparseError> {
        match self {
            SparseTensor::COO(_) => Ok(self.clone()),
            SparseTensor::CSR(m) => {
                let mut coo = SparseCOO::new(m.shape.0, m.shape.1);
                for row in 0..m.shape.0 {
                    let start = m.row_ptr[row];
                    let end = m.row_ptr[row + 1];
                    for i in start..end {
                        coo.add_entry(row, m.col_indices[i], m.values[i])?;
                    }
                }
                Ok(SparseTensor::COO(coo))
            }
            SparseTensor::CSC(m) => {
                let mut coo = SparseCOO::new(m.shape.0, m.shape.1);
                for col in 0..m.shape.1 {
                    let start = m.col_ptr[col];
                    let end = m.col_ptr[col + 1];
                    for i in start..end {
                        coo.add_entry(m.row_indices[i], col, m.values[i])?;
                    }
                }
                Ok(SparseTensor::COO(coo))
            }
        }
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            SparseTensor::CSR(m) => m.memory_bytes(),
            SparseTensor::CSC(m) => {
                m.col_ptr.len() * std::mem::size_of::<usize>()
                    + m.row_indices.len() * std::mem::size_of::<usize>()
                    + m.values.len() * std::mem::size_of::<f64>()
            }
            SparseTensor::COO(m) => {
                (m.row_indices.len() + m.col_indices.len()) * std::mem::size_of::<usize>()
                    + m.values.len() * std::mem::size_of::<f64>()
            }
        }
    }
}

/// Builder for sparse tensors.
pub struct SparseTensorBuilder {
    shape: Vec<usize>,
    format: SparseFormat,
    entries: Vec<(Vec<usize>, f64)>,
}

impl SparseTensorBuilder {
    /// Create a new sparse tensor builder.
    pub fn new(shape: Vec<usize>, format: SparseFormat) -> Self {
        Self {
            shape,
            format,
            entries: Vec::new(),
        }
    }

    /// Add a non-zero entry.
    pub fn add_entry(&mut self, indices: Vec<usize>, value: f64) -> Result<(), SparseError> {
        if indices.len() != self.shape.len() {
            return Err(SparseError::ShapeMismatch {
                expected: vec![self.shape.len()],
                actual: vec![indices.len()],
            });
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(SparseError::IndexOutOfBounds {
                    index: indices.clone(),
                    shape: self.shape.clone(),
                });
            }
        }

        self.entries.push((indices, value));
        Ok(())
    }

    /// Build the sparse tensor.
    pub fn build(self) -> Result<SparseTensor, SparseError> {
        // Currently only support 2D tensors
        if self.shape.len() != 2 {
            return Err(SparseError::UnsupportedOperation(format!(
                "Only 2D sparse tensors are supported, got shape {:?}",
                self.shape
            )));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        // Build COO first
        let mut coo = SparseCOO::new(rows, cols);
        for (indices, value) in self.entries {
            coo.add_entry(indices[0], indices[1], value)?;
        }

        // Convert to requested format
        match self.format {
            SparseFormat::COO => Ok(SparseTensor::COO(coo)),
            SparseFormat::CSR => Ok(SparseTensor::CSR(coo.to_csr())),
            SparseFormat::CSC => Ok(SparseTensor::CSC(coo.to_csc())),
        }
    }
}

/// Detect sparsity in a dense tensor.
pub fn detect_sparsity(data: &[f64], threshold: f64) -> (usize, f64) {
    let total = data.len();
    let zeros = data.iter().filter(|&&x| x.abs() < threshold).count();
    let sparsity = zeros as f64 / total as f64;
    (zeros, sparsity)
}

/// Convert dense tensor to sparse if beneficial.
pub fn to_sparse_if_beneficial(
    data: &[f64],
    shape: Vec<usize>,
    threshold: f64,
    min_sparsity: f64,
) -> Result<Option<SparseTensor>, SparseError> {
    let (_, sparsity) = detect_sparsity(data, threshold);

    if sparsity < min_sparsity {
        return Ok(None);
    }

    // Build sparse tensor
    let mut builder = SparseTensor::builder(shape.clone(), SparseFormat::CSR);

    if shape.len() == 2 {
        let cols = shape[1];
        for (i, &val) in data.iter().enumerate() {
            if val.abs() >= threshold {
                let row = i / cols;
                let col = i % cols;
                builder.add_entry(vec![row, col], val)?;
            }
        }
    }

    Ok(Some(builder.build()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_format() {
        assert_eq!(SparseFormat::CSR.name(), "CSR");
        assert!(SparseFormat::CSR.is_compressed());
        assert!(!SparseFormat::COO.is_compressed());
    }

    #[test]
    fn test_sparse_coo_creation() {
        let mut coo = SparseCOO::new(3, 3);
        assert_eq!(coo.shape, (3, 3));
        assert_eq!(coo.nnz(), 0);

        coo.add_entry(0, 1, 5.0).unwrap();
        coo.add_entry(1, 2, 3.0).unwrap();
        assert_eq!(coo.nnz(), 2);
    }

    #[test]
    fn test_sparse_coo_to_csr() {
        let mut coo = SparseCOO::new(3, 3);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 2, 2.0).unwrap();
        coo.add_entry(2, 1, 3.0).unwrap();

        let csr = coo.to_csr();
        assert_eq!(csr.shape, (3, 3));
        assert_eq!(csr.nnz(), 3);
        assert!(csr.validate().is_ok());
    }

    #[test]
    fn test_sparse_csr_multiply_dense() {
        let mut coo = SparseCOO::new(2, 3);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 2, 2.0).unwrap();
        coo.add_entry(1, 1, 3.0).unwrap();

        let csr = coo.to_csr();
        let vec = vec![1.0, 2.0, 3.0];
        let result = csr.multiply_dense(&vec).unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 7.0).abs() < 1e-10); // 1*1 + 2*3 = 7
        assert!((result[1] - 6.0).abs() < 1e-10); // 3*2 = 6
    }

    #[test]
    fn test_sparse_csr_row_access() {
        let mut coo = SparseCOO::new(3, 3);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 2, 2.0).unwrap();
        coo.add_entry(1, 1, 3.0).unwrap();

        let csr = coo.to_csr();
        let row0 = csr.row(0).unwrap();
        assert_eq!(row0.len(), 2);
        assert_eq!(row0[0], (0, 1.0));
        assert_eq!(row0[1], (2, 2.0));

        let row1 = csr.row(1).unwrap();
        assert_eq!(row1.len(), 1);
        assert_eq!(row1[0], (1, 3.0));
    }

    #[test]
    fn test_sparse_csr_transpose() {
        let mut coo = SparseCOO::new(2, 3);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 2, 2.0).unwrap();
        coo.add_entry(1, 1, 3.0).unwrap();

        let csr = coo.to_csr();
        let csc = csr.transpose();

        assert_eq!(csc.shape, (3, 2));
        assert_eq!(csc.nnz(), 3);
    }

    #[test]
    fn test_sparsity_ratio() {
        let mut coo = SparseCOO::new(10, 10);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(5, 5, 2.0).unwrap();

        let sparsity = coo.sparsity_ratio();
        assert!((sparsity - 0.98).abs() < 0.01); // 98% sparse
    }

    #[test]
    fn test_sparse_tensor_builder() {
        let mut builder = SparseTensor::builder(vec![3, 3], SparseFormat::CSR);
        builder.add_entry(vec![0, 0], 1.0).unwrap();
        builder.add_entry(vec![1, 2], 2.0).unwrap();

        let sparse = builder.build().unwrap();
        assert_eq!(sparse.format(), SparseFormat::CSR);
        assert_eq!(sparse.nnz(), 2);
    }

    #[test]
    fn test_sparse_tensor_conversion() {
        let mut builder = SparseTensor::builder(vec![3, 3], SparseFormat::COO);
        builder.add_entry(vec![0, 0], 1.0).unwrap();
        builder.add_entry(vec![1, 2], 2.0).unwrap();

        let coo = builder.build().unwrap();
        let csr = coo.to_csr().unwrap();
        let csc = csr.to_csc().unwrap();

        assert_eq!(coo.nnz(), 2);
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csc.nnz(), 2);
    }

    #[test]
    fn test_detect_sparsity() {
        let data = vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let (zeros, sparsity) = detect_sparsity(&data, 1e-10);

        assert_eq!(zeros, 6);
        assert!((sparsity - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_to_sparse_if_beneficial() {
        let data = vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0];
        let shape = vec![2, 3];

        let sparse = to_sparse_if_beneficial(&data, shape, 1e-10, 0.5).unwrap();
        assert!(sparse.is_some());

        let sparse = sparse.unwrap();
        assert_eq!(sparse.nnz(), 2);
        assert!(sparse.sparsity_ratio() > 0.5);
    }

    #[test]
    fn test_sparse_csr_validation() {
        let csr = SparseCSR {
            shape: (3, 3),
            row_ptr: vec![0, 2, 3, 3],
            col_indices: vec![0, 2, 1],
            values: vec![1.0, 2.0, 3.0],
        };

        assert!(csr.validate().is_ok());
    }

    #[test]
    fn test_sparse_memory_usage() {
        let mut builder = SparseTensor::builder(vec![100, 100], SparseFormat::CSR);
        builder.add_entry(vec![0, 0], 1.0).unwrap();
        builder.add_entry(vec![50, 50], 2.0).unwrap();

        let sparse = builder.build().unwrap();
        let memory = sparse.memory_bytes();

        // Should be much less than dense 100x100 matrix
        let dense_memory = 100 * 100 * std::mem::size_of::<f64>();
        assert!(memory < dense_memory / 10);
    }
}
