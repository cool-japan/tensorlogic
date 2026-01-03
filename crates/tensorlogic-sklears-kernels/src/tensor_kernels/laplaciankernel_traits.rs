//! # LaplacianKernel - Trait Implementations
//!
//! This module contains trait implementations for `LaplacianKernel`.
//!
//! ## Implemented Traits
//!
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::LaplacianKernel;

impl Kernel for LaplacianKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Laplacian kernel".to_string(),
            });
        }
        let l1_dist = Self::l1_distance(x, y);
        let result = (-self.gamma * l1_dist).exp();
        Ok(result)
    }
    fn name(&self) -> &str {
        "Laplacian"
    }
}
