//! # CosineKernel - Trait Implementations
//!
//! This module contains trait implementations for `CosineKernel`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::CosineKernel;

impl Default for CosineKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel for CosineKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "cosine kernel".to_string(),
            });
        }
        let dot = Self::dot_product(x, y);
        let norm_x = Self::norm(x);
        let norm_y = Self::norm(y);
        if norm_x == 0.0 || norm_y == 0.0 {
            return Ok(0.0);
        }
        Ok(dot / (norm_x * norm_y))
    }
    fn name(&self) -> &str {
        "Cosine"
    }
}
