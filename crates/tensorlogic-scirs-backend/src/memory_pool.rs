//! Memory pooling for efficient tensor allocation.
//!
//! This module provides a memory pool that reuses tensor allocations
//! to reduce the overhead of creating and destroying arrays during execution.

use crate::Scirs2Tensor;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// Memory pool for reusing tensor allocations
#[derive(Default)]
pub struct TensorPool {
    /// Available tensors grouped by shape
    available: HashMap<Vec<usize>, Vec<Scirs2Tensor>>,
    /// Statistics
    pub(crate) allocations: usize,
    pub(crate) reuses: usize,
}

impl TensorPool {
    /// Create a new empty tensor pool
    pub fn new() -> Self {
        TensorPool {
            available: HashMap::new(),
            allocations: 0,
            reuses: 0,
        }
    }

    /// Get a tensor with the specified shape, either from the pool or newly allocated
    pub fn get(&mut self, shape: &[usize]) -> Scirs2Tensor {
        let shape_key = shape.to_vec();

        // Try to reuse from pool
        if let Some(tensors) = self.available.get_mut(&shape_key) {
            if let Some(tensor) = tensors.pop() {
                self.reuses += 1;
                // Zero out the tensor before reuse
                let mut tensor = tensor;
                tensor.fill(0.0);
                return tensor;
            }
        }

        // Allocate new tensor
        self.allocations += 1;
        ArrayD::zeros(IxDyn(shape))
    }

    /// Get a tensor filled with ones
    pub fn get_ones(&mut self, shape: &[usize]) -> Scirs2Tensor {
        let mut tensor = self.get(shape);
        tensor.fill(1.0);
        tensor
    }

    /// Return a tensor to the pool for reuse
    pub fn return_tensor(&mut self, tensor: Scirs2Tensor) {
        let shape = tensor.shape().to_vec();
        self.available.entry(shape).or_default().push(tensor);
    }

    /// Clear all pooled tensors
    pub fn clear(&mut self) {
        self.available.clear();
        self.allocations = 0;
        self.reuses = 0;
    }

    /// Get the number of available tensors in the pool
    pub fn available_count(&self) -> usize {
        self.available.values().map(|v| v.len()).sum()
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            allocations: self.allocations,
            reuses: self.reuses,
            available: self.available_count(),
            reuse_rate: if self.allocations + self.reuses > 0 {
                self.reuses as f64 / (self.allocations + self.reuses) as f64
            } else {
                0.0
            },
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of new allocations
    pub allocations: usize,
    /// Total number of reuses from pool
    pub reuses: usize,
    /// Number of tensors currently available in pool
    pub available: usize,
    /// Reuse rate (0.0 to 1.0)
    pub reuse_rate: f64,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PoolStats {{ allocations: {}, reuses: {}, available: {}, reuse_rate: {:.2}% }}",
            self.allocations,
            self.reuses,
            self.available,
            self.reuse_rate * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_basic() {
        let mut pool = TensorPool::new();

        // First allocation
        let t1 = pool.get(&[2, 3]);
        assert_eq!(t1.shape(), &[2, 3]);
        assert_eq!(pool.allocations, 1);
        assert_eq!(pool.reuses, 0);

        // Return to pool
        pool.return_tensor(t1);
        assert_eq!(pool.available_count(), 1);

        // Reuse from pool
        let t2 = pool.get(&[2, 3]);
        assert_eq!(t2.shape(), &[2, 3]);
        assert_eq!(pool.allocations, 1);
        assert_eq!(pool.reuses, 1);
    }

    #[test]
    fn test_pool_different_shapes() {
        let mut pool = TensorPool::new();

        let t1 = pool.get(&[2, 3]);
        let t2 = pool.get(&[4, 5]);

        pool.return_tensor(t1);
        pool.return_tensor(t2);

        assert_eq!(pool.available_count(), 2);

        // Get tensor with shape [2, 3] - should reuse
        let t3 = pool.get(&[2, 3]);
        assert_eq!(t3.shape(), &[2, 3]);
        assert_eq!(pool.reuses, 1);

        // Get tensor with shape [4, 5] - should reuse
        let t4 = pool.get(&[4, 5]);
        assert_eq!(t4.shape(), &[4, 5]);
        assert_eq!(pool.reuses, 2);
    }

    #[test]
    fn test_pool_stats() {
        let mut pool = TensorPool::new();

        // Allocate 3 tensors
        let t1 = pool.get(&[2, 2]);
        let t2 = pool.get(&[2, 2]);
        let t3 = pool.get(&[2, 2]);

        pool.return_tensor(t1);
        pool.return_tensor(t2);
        pool.return_tensor(t3);

        // Reuse 2 tensors
        let _t4 = pool.get(&[2, 2]);
        let _t5 = pool.get(&[2, 2]);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 3);
        assert_eq!(stats.reuses, 2);
        assert_eq!(stats.available, 1);
        assert!((stats.reuse_rate - 0.4).abs() < 1e-6); // 2/(3+2) = 0.4
    }

    #[test]
    fn test_get_ones() {
        let mut pool = TensorPool::new();
        let t = pool.get_ones(&[2, 2]);

        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[1, 1]], 1.0);
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = TensorPool::new();

        let t1 = pool.get(&[2, 2]);
        pool.return_tensor(t1);

        assert_eq!(pool.available_count(), 1);

        pool.clear();
        assert_eq!(pool.available_count(), 0);
        assert_eq!(pool.allocations, 0);
        assert_eq!(pool.reuses, 0);
    }

    #[test]
    fn test_pool_zeroing() {
        let mut pool = TensorPool::new();

        // Create tensor and fill with non-zero values
        let mut t1 = pool.get(&[2, 2]);
        t1.fill(5.0);
        pool.return_tensor(t1);

        // Get tensor from pool - should be zeroed
        let t2 = pool.get(&[2, 2]);
        assert_eq!(t2[[0, 0]], 0.0);
        assert_eq!(t2[[1, 1]], 0.0);
    }
}
