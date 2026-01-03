//! # MaternKernel - Trait Implementations
//!
//! This module contains trait implementations for `MaternKernel`.
//!
//! ## Implemented Traits
//!
//! - `Kernel`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{KernelError, Result};
use crate::types::Kernel;

use super::types::MaternKernel;

impl Kernel for MaternKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "Matérn kernel".to_string(),
            });
        }
        let dist = Self::euclidean_distance(x, y);
        if dist < 1e-10 {
            return Ok(1.0);
        }
        let scaled_dist = (2.0 * self.nu).sqrt() * dist / self.length_scale;
        let result = if (self.nu - 0.5).abs() < 1e-10 {
            (-scaled_dist).exp()
        } else if (self.nu - 1.5).abs() < 1e-10 {
            (1.0 + scaled_dist) * (-scaled_dist).exp()
        } else if (self.nu - 2.5).abs() < 1e-10 {
            (1.0 + scaled_dist + scaled_dist * scaled_dist / 3.0) * (-scaled_dist).exp()
        } else {
            (1.0 + scaled_dist) * (-scaled_dist).exp()
        };
        Ok(result)
    }
    fn name(&self) -> &str {
        "Matérn"
    }
}
