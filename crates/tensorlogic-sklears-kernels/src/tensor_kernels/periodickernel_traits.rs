//! # PeriodicKernel - Trait Implementations
//!
//! This module contains trait implementations for `PeriodicKernel`.
//!
//! ## Implemented Traits
//!
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::PeriodicKernel;

impl Kernel for PeriodicKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Periodic kernel".to_string(),
            });
        }
        let dist = Self::euclidean_distance(x, y);
        let sin_term = (std::f64::consts::PI * dist / self.period).sin();
        let result = (-2.0 * sin_term * sin_term / (self.length_scale * self.length_scale)).exp();
        Ok(result)
    }
    fn name(&self) -> &str {
        "Periodic"
    }
}
