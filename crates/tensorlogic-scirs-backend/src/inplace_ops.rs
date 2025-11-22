//! In-place operations for memory optimization.
//!
//! This module provides infrastructure for executing operations in-place,
//! modifying tensors directly without allocating new memory. This significantly
//! reduces memory overhead for large tensor computations.
//!
//! ## Safety
//!
//! In-place operations are only safe when:
//! 1. The tensor is not referenced elsewhere (unique ownership)
//! 2. The operation preserves the tensor shape
//! 3. The operation doesn't depend on the original values after modification
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_scirs_backend::inplace_ops::{InplaceExecutor, can_execute_inplace};
//!
//! let mut executor = InplaceExecutor::new();
//! let mut tensor = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
//!
//! // Execute ReLU in-place if safe
//! if can_execute_inplace("relu", &tensor) {
//!     executor.execute_inplace_unary("relu", &mut tensor)?;
//! }
//! ```

use crate::{Scirs2Tensor, TlBackendError, TlBackendResult};
use scirs2_core::ndarray::Zip;
use std::collections::HashSet;

/// Executor for in-place tensor operations.
///
/// Tracks tensor aliasing and ensures safe in-place execution.
#[derive(Debug, Clone)]
pub struct InplaceExecutor {
    /// Set of tensor IDs that are aliased (not safe for in-place ops)
    aliased_tensors: HashSet<usize>,

    /// Statistics for in-place execution
    pub stats: InplaceStats,
}

/// Statistics for in-place operation execution.
#[derive(Debug, Clone, Default)]
pub struct InplaceStats {
    /// Number of operations executed in-place
    pub inplace_ops: usize,

    /// Number of operations that could not be executed in-place
    pub non_inplace_ops: usize,

    /// Total memory saved (estimated in bytes)
    pub memory_saved_bytes: usize,
}

impl InplaceExecutor {
    /// Create a new in-place executor.
    pub fn new() -> Self {
        Self {
            aliased_tensors: HashSet::new(),
            stats: InplaceStats::default(),
        }
    }

    /// Mark a tensor as aliased (not safe for in-place operations).
    pub fn mark_aliased(&mut self, tensor_id: usize) {
        self.aliased_tensors.insert(tensor_id);
    }

    /// Check if a tensor can be safely modified in-place.
    pub fn can_execute_inplace(&self, tensor_id: usize) -> bool {
        !self.aliased_tensors.contains(&tensor_id)
    }

    /// Execute a unary operation in-place.
    ///
    /// # Safety
    ///
    /// This function modifies the tensor in-place. The caller must ensure:
    /// - The tensor is not aliased elsewhere
    /// - The operation is shape-preserving
    pub fn execute_inplace_unary(
        &mut self,
        op: &str,
        tensor: &mut Scirs2Tensor,
    ) -> TlBackendResult<()> {
        let element_count = tensor.len();
        let bytes_saved = element_count * std::mem::size_of::<f64>();

        match op {
            "relu" => {
                Zip::from(tensor).for_each(|x| {
                    *x = x.max(0.0);
                });
            }
            "sigmoid" => {
                Zip::from(tensor).for_each(|x| {
                    *x = 1.0 / (1.0 + (-*x).exp());
                });
            }
            "oneminus" => {
                Zip::from(tensor).for_each(|x| {
                    *x = 1.0 - *x;
                });
            }
            "tanh" => {
                Zip::from(tensor).for_each(|x| {
                    *x = x.tanh();
                });
            }
            "abs" => {
                Zip::from(tensor).for_each(|x| {
                    *x = x.abs();
                });
            }
            "neg" => {
                Zip::from(tensor).for_each(|x| {
                    *x = -*x;
                });
            }
            "exp" => {
                Zip::from(tensor).for_each(|x| {
                    *x = x.exp();
                });
            }
            "log" => {
                Zip::from(tensor).for_each(|x| {
                    *x = x.ln();
                });
            }
            "sqrt" => {
                Zip::from(tensor).for_each(|x| {
                    *x = x.sqrt();
                });
            }
            "square" => {
                Zip::from(tensor).for_each(|x| {
                    *x = *x * *x;
                });
            }
            "clip" => {
                // Clip to [0, 1] range
                Zip::from(tensor).for_each(|x| {
                    *x = x.clamp(0.0, 1.0);
                });
            }
            _ => {
                self.stats.non_inplace_ops += 1;
                return Err(TlBackendError::unsupported(format!(
                    "Unsupported in-place unary operation: {}",
                    op
                )));
            }
        }

        self.stats.inplace_ops += 1;
        self.stats.memory_saved_bytes += bytes_saved;

        Ok(())
    }

    /// Execute a binary operation in-place (modifies the first tensor).
    ///
    /// # Safety
    ///
    /// This function modifies `lhs` in-place. The caller must ensure:
    /// - The tensor is not aliased elsewhere
    /// - The shapes are compatible for broadcasting
    pub fn execute_inplace_binary(
        &mut self,
        op: &str,
        lhs: &mut Scirs2Tensor,
        rhs: &Scirs2Tensor,
    ) -> TlBackendResult<()> {
        // Check shape compatibility
        if lhs.shape() != rhs.shape() {
            self.stats.non_inplace_ops += 1;
            return Err(TlBackendError::shape_mismatch(
                op,
                vec![lhs.shape().to_vec()],
                vec![rhs.shape().to_vec()],
            ));
        }

        let element_count = lhs.len();
        let bytes_saved = element_count * std::mem::size_of::<f64>();

        match op {
            "add" => {
                Zip::from(lhs).and(rhs).for_each(|x, &y| {
                    *x += y;
                });
            }
            "subtract" | "sub" => {
                Zip::from(lhs).and(rhs).for_each(|x, &y| {
                    *x -= y;
                });
            }
            "multiply" | "mul" => {
                Zip::from(lhs).and(rhs).for_each(|x, &y| {
                    *x *= y;
                });
            }
            "divide" | "div" => {
                Zip::from(lhs).and(rhs).for_each(|x, &y| {
                    *x /= y;
                });
            }
            "min" => {
                Zip::from(lhs).and(rhs).for_each(|x, &y| {
                    *x = x.min(y);
                });
            }
            "max" => {
                Zip::from(lhs).and(rhs).for_each(|x, &y| {
                    *x = x.max(y);
                });
            }
            _ => {
                self.stats.non_inplace_ops += 1;
                return Err(TlBackendError::unsupported(format!(
                    "Unsupported in-place binary operation: {}",
                    op
                )));
            }
        }

        self.stats.inplace_ops += 1;
        self.stats.memory_saved_bytes += bytes_saved;

        Ok(())
    }

    /// Execute a scalar operation in-place (tensor op scalar).
    pub fn execute_inplace_scalar(
        &mut self,
        op: &str,
        tensor: &mut Scirs2Tensor,
        scalar: f64,
    ) -> TlBackendResult<()> {
        let element_count = tensor.len();
        let bytes_saved = element_count * std::mem::size_of::<f64>();

        match op {
            "add" | "add_scalar" => {
                Zip::from(tensor).for_each(|x| {
                    *x += scalar;
                });
            }
            "sub" | "sub_scalar" => {
                Zip::from(tensor).for_each(|x| {
                    *x -= scalar;
                });
            }
            "mul" | "mul_scalar" => {
                Zip::from(tensor).for_each(|x| {
                    *x *= scalar;
                });
            }
            "div" | "div_scalar" => {
                Zip::from(tensor).for_each(|x| {
                    *x /= scalar;
                });
            }
            "pow" => {
                Zip::from(tensor).for_each(|x| {
                    *x = x.powf(scalar);
                });
            }
            "clamp_min" => {
                Zip::from(tensor).for_each(|x| {
                    *x = x.max(scalar);
                });
            }
            "clamp_max" => {
                Zip::from(tensor).for_each(|x| {
                    *x = x.min(scalar);
                });
            }
            _ => {
                self.stats.non_inplace_ops += 1;
                return Err(TlBackendError::unsupported(format!(
                    "Unsupported in-place scalar operation: {}",
                    op
                )));
            }
        }

        self.stats.inplace_ops += 1;
        self.stats.memory_saved_bytes += bytes_saved;

        Ok(())
    }

    /// Get statistics for in-place execution.
    pub fn statistics(&self) -> &InplaceStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = InplaceStats::default();
    }

    /// Clear aliasing information.
    pub fn clear_aliasing(&mut self) {
        self.aliased_tensors.clear();
    }
}

impl Default for InplaceExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl InplaceStats {
    /// Calculate the percentage of operations executed in-place.
    pub fn inplace_percentage(&self) -> f64 {
        let total = self.inplace_ops + self.non_inplace_ops;
        if total == 0 {
            0.0
        } else {
            (self.inplace_ops as f64 / total as f64) * 100.0
        }
    }

    /// Format memory savings in human-readable form.
    pub fn format_memory_saved(&self) -> String {
        let bytes = self.memory_saved_bytes;
        if bytes < 1024 {
            format!("{} bytes", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

/// Check if an operation can be executed in-place.
pub fn can_execute_inplace(op: &str) -> bool {
    matches!(
        op,
        "relu"
            | "sigmoid"
            | "oneminus"
            | "tanh"
            | "abs"
            | "neg"
            | "exp"
            | "log"
            | "sqrt"
            | "square"
            | "clip"
            | "add"
            | "subtract"
            | "sub"
            | "multiply"
            | "mul"
            | "divide"
            | "div"
            | "min"
            | "max"
    )
}

/// Check if an operation is shape-preserving (required for in-place execution).
pub fn is_shape_preserving(op: &str) -> bool {
    // All unary and element-wise binary operations preserve shape
    // Reduce operations do not
    !matches!(op, "sum" | "mean" | "max_reduce" | "min_reduce" | "product")
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    #[test]
    fn test_inplace_executor_new() {
        let executor = InplaceExecutor::new();
        assert_eq!(executor.stats.inplace_ops, 0);
        assert_eq!(executor.stats.non_inplace_ops, 0);
        assert_eq!(executor.stats.memory_saved_bytes, 0);
    }

    #[test]
    fn test_can_execute_inplace() {
        let mut executor = InplaceExecutor::new();
        assert!(executor.can_execute_inplace(0));

        executor.mark_aliased(0);
        assert!(!executor.can_execute_inplace(0));
        assert!(executor.can_execute_inplace(1));
    }

    #[test]
    fn test_inplace_unary_relu() {
        let mut executor = InplaceExecutor::new();
        let mut tensor = ArrayD::from_shape_vec(vec![3], vec![-1.0, 0.0, 1.0]).unwrap();

        executor.execute_inplace_unary("relu", &mut tensor).unwrap();

        assert_eq!(tensor[[0]], 0.0);
        assert_eq!(tensor[[1]], 0.0);
        assert_eq!(tensor[[2]], 1.0);
        assert_eq!(executor.stats.inplace_ops, 1);
    }

    #[test]
    fn test_inplace_unary_sigmoid() {
        let mut executor = InplaceExecutor::new();
        let mut tensor = ArrayD::from_shape_vec(vec![2], vec![0.0, 1.0]).unwrap();

        executor
            .execute_inplace_unary("sigmoid", &mut tensor)
            .unwrap();

        assert!((tensor[[0]] - 0.5).abs() < 1e-6);
        assert!((tensor[[1]] - 0.731).abs() < 0.01);
        assert_eq!(executor.stats.inplace_ops, 1);
    }

    #[test]
    fn test_inplace_unary_oneminus() {
        let mut executor = InplaceExecutor::new();
        let mut tensor = ArrayD::from_shape_vec(vec![3], vec![0.0, 0.5, 1.0]).unwrap();

        executor
            .execute_inplace_unary("oneminus", &mut tensor)
            .unwrap();

        assert_eq!(tensor[[0]], 1.0);
        assert_eq!(tensor[[1]], 0.5);
        assert_eq!(tensor[[2]], 0.0);
    }

    #[test]
    fn test_inplace_binary_add() {
        let mut executor = InplaceExecutor::new();
        let mut lhs = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let rhs = ArrayD::from_shape_vec(vec![3], vec![4.0, 5.0, 6.0]).unwrap();

        executor
            .execute_inplace_binary("add", &mut lhs, &rhs)
            .unwrap();

        assert_eq!(lhs[[0]], 5.0);
        assert_eq!(lhs[[1]], 7.0);
        assert_eq!(lhs[[2]], 9.0);
        assert_eq!(executor.stats.inplace_ops, 1);
    }

    #[test]
    fn test_inplace_binary_multiply() {
        let mut executor = InplaceExecutor::new();
        let mut lhs = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let rhs = ArrayD::from_shape_vec(vec![3], vec![2.0, 3.0, 4.0]).unwrap();

        executor
            .execute_inplace_binary("multiply", &mut lhs, &rhs)
            .unwrap();

        assert_eq!(lhs[[0]], 2.0);
        assert_eq!(lhs[[1]], 6.0);
        assert_eq!(lhs[[2]], 12.0);
    }

    #[test]
    fn test_inplace_binary_shape_mismatch() {
        let mut executor = InplaceExecutor::new();
        let mut lhs = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
        let rhs = ArrayD::from_shape_vec(vec![2], vec![4.0, 5.0]).unwrap();

        let result = executor.execute_inplace_binary("add", &mut lhs, &rhs);
        assert!(result.is_err());
        assert_eq!(executor.stats.non_inplace_ops, 1);
    }

    #[test]
    fn test_inplace_scalar_add() {
        let mut executor = InplaceExecutor::new();
        let mut tensor = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        executor
            .execute_inplace_scalar("add", &mut tensor, 10.0)
            .unwrap();

        assert_eq!(tensor[[0]], 11.0);
        assert_eq!(tensor[[1]], 12.0);
        assert_eq!(tensor[[2]], 13.0);
    }

    #[test]
    fn test_inplace_scalar_multiply() {
        let mut executor = InplaceExecutor::new();
        let mut tensor = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        executor
            .execute_inplace_scalar("mul", &mut tensor, 2.0)
            .unwrap();

        assert_eq!(tensor[[0]], 2.0);
        assert_eq!(tensor[[1]], 4.0);
        assert_eq!(tensor[[2]], 6.0);
    }

    #[test]
    fn test_inplace_stats() {
        let mut executor = InplaceExecutor::new();
        let mut tensor = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        executor.execute_inplace_unary("relu", &mut tensor).unwrap();
        executor
            .execute_inplace_scalar("add", &mut tensor, 1.0)
            .unwrap();

        assert_eq!(executor.stats.inplace_ops, 2);
        assert!(executor.stats.memory_saved_bytes > 0);
        assert_eq!(executor.stats.inplace_percentage(), 100.0);
    }

    #[test]
    fn test_can_execute_inplace_func() {
        assert!(can_execute_inplace("relu"));
        assert!(can_execute_inplace("sigmoid"));
        assert!(can_execute_inplace("add"));
        assert!(!can_execute_inplace("unknown_op"));
    }

    #[test]
    fn test_is_shape_preserving() {
        assert!(is_shape_preserving("relu"));
        assert!(is_shape_preserving("add"));
        assert!(!is_shape_preserving("sum"));
        assert!(!is_shape_preserving("mean"));
    }

    #[test]
    fn test_format_memory_saved() {
        let stats = InplaceStats {
            memory_saved_bytes: 512,
            ..Default::default()
        };
        assert_eq!(stats.format_memory_saved(), "512 bytes");

        let stats = InplaceStats {
            memory_saved_bytes: 2048,
            ..Default::default()
        };
        assert_eq!(stats.format_memory_saved(), "2.00 KB");

        let stats = InplaceStats {
            memory_saved_bytes: 2 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(stats.format_memory_saved(), "2.00 MB");

        let stats = InplaceStats {
            memory_saved_bytes: 3 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert_eq!(stats.format_memory_saved(), "3.00 GB");
    }

    #[test]
    fn test_reset_stats() {
        let mut executor = InplaceExecutor::new();
        let mut tensor = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

        executor.execute_inplace_unary("relu", &mut tensor).unwrap();
        assert_eq!(executor.stats.inplace_ops, 1);

        executor.reset_stats();
        assert_eq!(executor.stats.inplace_ops, 0);
        assert_eq!(executor.stats.memory_saved_bytes, 0);
    }

    #[test]
    fn test_clear_aliasing() {
        let mut executor = InplaceExecutor::new();
        executor.mark_aliased(0);
        executor.mark_aliased(1);

        assert!(!executor.can_execute_inplace(0));
        assert!(!executor.can_execute_inplace(1));

        executor.clear_aliasing();
        assert!(executor.can_execute_inplace(0));
        assert!(executor.can_execute_inplace(1));
    }
}
