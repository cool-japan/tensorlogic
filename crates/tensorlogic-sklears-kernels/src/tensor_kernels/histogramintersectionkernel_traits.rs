//! # HistogramIntersectionKernel - Trait Implementations
//!
//! This module contains trait implementations for `HistogramIntersectionKernel`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::HistogramIntersectionKernel;

impl Default for HistogramIntersectionKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel for HistogramIntersectionKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "histogram intersection kernel".to_string(),
            });
        }
        if x.iter().any(|&v| v < 0.0) || y.iter().any(|&v| v < 0.0) {
            return Err(KernelError::ComputationError(
                "Histogram features must be non-negative".to_string(),
            ));
        }
        let result = Self::intersection(x, y);
        Ok(result)
    }
    fn name(&self) -> &str {
        "HistogramIntersection"
    }
}
