//! # RationalQuadraticKernel - Trait Implementations
//!
//! This module contains trait implementations for `RationalQuadraticKernel`.
//!
//! ## Implemented Traits
//!
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::RationalQuadraticKernel;

impl Kernel for RationalQuadraticKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Rational Quadratic kernel".to_string(),
            });
        }
        let sq_dist = Self::squared_distance(x, y);
        let term = 1.0 + sq_dist / (2.0 * self.alpha * self.length_scale * self.length_scale);
        let result = term.powf(-self.alpha);
        Ok(result)
    }
    fn name(&self) -> &str {
        "RationalQuadratic"
    }
}
