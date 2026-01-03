//! # LinearKernel - Trait Implementations
//!
//! This module contains trait implementations for `LinearKernel`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::LinearKernel;

impl Default for LinearKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel for LinearKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "linear kernel".to_string(),
            });
        }
        Ok(Self::dot_product(x, y))
    }
    fn name(&self) -> &str {
        "Linear"
    }
}
