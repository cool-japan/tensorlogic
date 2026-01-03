//! # PolynomialKernel - Trait Implementations
//!
//! This module contains trait implementations for `PolynomialKernel`.
//!
//! ## Implemented Traits
//!
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::PolynomialKernel;

impl Kernel for PolynomialKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "polynomial kernel".to_string(),
            });
        }
        let dot = Self::dot_product(x, y);
        let result = (dot + self.constant).powi(self.degree as i32);
        Ok(result)
    }
    fn name(&self) -> &str {
        "Polynomial"
    }
}
