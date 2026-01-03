//! # SigmoidKernel - Trait Implementations
//!
//! This module contains trait implementations for `SigmoidKernel`.
//!
//! ## Implemented Traits
//!
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::SigmoidKernel;

impl Kernel for SigmoidKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "sigmoid kernel".to_string(),
            });
        }
        let dot = Self::dot_product(x, y);
        let result = (self.alpha * dot + self.offset).tanh();
        Ok(result)
    }
    fn name(&self) -> &str {
        "Sigmoid"
    }
    fn is_psd(&self) -> bool {
        false
    }
}
