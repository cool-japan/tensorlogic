//! # RbfKernel - Trait Implementations
//!
//! This module contains trait implementations for `RbfKernel`.
//!
//! ## Implemented Traits
//!
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::RbfKernel;

impl Kernel for RbfKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "RBF kernel".to_string(),
            });
        }
        let sq_dist = Self::squared_distance(x, y);
        let result = (-self.config.gamma * sq_dist).exp();
        Ok(result)
    }
    fn name(&self) -> &str {
        "RBF"
    }
}
