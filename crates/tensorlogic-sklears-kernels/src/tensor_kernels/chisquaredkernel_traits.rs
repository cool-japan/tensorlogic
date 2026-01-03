//! # ChiSquaredKernel - Trait Implementations
//!
//! This module contains trait implementations for `ChiSquaredKernel`.
//!
//! ## Implemented Traits
//!
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::ChiSquaredKernel;

impl Kernel for ChiSquaredKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "chi-squared kernel".to_string(),
            });
        }
        let chi_sq_dist = Self::chi_squared_distance(x, y);
        let result = (-self.gamma * chi_sq_dist).exp();
        Ok(result)
    }
    fn name(&self) -> &str {
        "ChiSquared"
    }
}
