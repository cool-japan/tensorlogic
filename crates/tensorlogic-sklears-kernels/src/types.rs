//! Core types for kernel operations.

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Trait for kernel functions that compute similarity between inputs.
///
/// Kernels map pairs of inputs to scalar similarity values.
pub trait Kernel: Send + Sync {
    /// Compute kernel value between two inputs.
    ///
    /// # Arguments
    /// * `x` - First input (feature representation)
    /// * `y` - Second input (feature representation)
    ///
    /// # Returns
    /// Similarity score (typically in range [0, 1] or [-1, 1])
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64>;

    /// Compute kernel matrix for a set of inputs.
    ///
    /// # Arguments
    /// * `inputs` - Slice of input feature vectors
    ///
    /// # Returns
    /// Kernel matrix K where `K[i,j] = kernel(inputs[i], inputs[j])`
    fn compute_matrix(&self, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = inputs.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = self.compute(&inputs[i], &inputs[j])?;
            }
        }

        Ok(matrix)
    }

    /// Get kernel name for identification.
    fn name(&self) -> &str;

    /// Check if kernel is positive semi-definite.
    fn is_psd(&self) -> bool {
        true // Most kernels are PSD by construction
    }
}

/// Configuration for rule-based similarity kernel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuleSimilarityConfig {
    /// Weight for satisfied rules (both inputs satisfy)
    pub satisfied_weight: f64,
    /// Weight for violated rules (both inputs violate)
    pub violated_weight: f64,
    /// Weight for mixed cases (one satisfies, one violates)
    pub mixed_weight: f64,
    /// Normalization strategy
    pub normalize: bool,
}

impl RuleSimilarityConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            satisfied_weight: 1.0,
            violated_weight: 0.5,
            mixed_weight: 0.0,
            normalize: true,
        }
    }

    /// Set satisfied weight
    pub fn with_satisfied_weight(mut self, weight: f64) -> Self {
        self.satisfied_weight = weight;
        self
    }

    /// Set violated weight
    pub fn with_violated_weight(mut self, weight: f64) -> Self {
        self.violated_weight = weight;
        self
    }

    /// Set mixed weight
    pub fn with_mixed_weight(mut self, weight: f64) -> Self {
        self.mixed_weight = weight;
        self
    }

    /// Set normalization flag
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Default for RuleSimilarityConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for predicate overlap kernel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredicateOverlapConfig {
    /// Weight for exact predicate matches
    pub exact_match_weight: f64,
    /// Weight for predicate co-occurrence
    pub cooccurrence_weight: f64,
    /// Minimum overlap threshold
    pub min_overlap: f64,
}

impl PredicateOverlapConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            exact_match_weight: 1.0,
            cooccurrence_weight: 0.5,
            min_overlap: 0.0,
        }
    }

    /// Set exact match weight
    pub fn with_exact_match_weight(mut self, weight: f64) -> Self {
        self.exact_match_weight = weight;
        self
    }

    /// Set co-occurrence weight
    pub fn with_cooccurrence_weight(mut self, weight: f64) -> Self {
        self.cooccurrence_weight = weight;
        self
    }

    /// Set minimum overlap threshold
    pub fn with_min_overlap(mut self, threshold: f64) -> Self {
        self.min_overlap = threshold;
        self
    }
}

impl Default for PredicateOverlapConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for RBF (Gaussian) kernel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RbfKernelConfig {
    /// Bandwidth parameter (gamma = 1 / (2 * sigma^2))
    pub gamma: f64,
}

impl RbfKernelConfig {
    /// Create configuration with specified gamma
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }

    /// Create from bandwidth (sigma)
    pub fn from_sigma(sigma: f64) -> Self {
        Self {
            gamma: 1.0 / (2.0 * sigma * sigma),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_similarity_config() {
        let config = RuleSimilarityConfig::new()
            .with_satisfied_weight(2.0)
            .with_violated_weight(1.0)
            .with_mixed_weight(0.5);

        assert_eq!(config.satisfied_weight, 2.0);
        assert_eq!(config.violated_weight, 1.0);
        assert_eq!(config.mixed_weight, 0.5);
        assert!(config.normalize);
    }

    #[test]
    fn test_predicate_overlap_config() {
        let config = PredicateOverlapConfig::new()
            .with_exact_match_weight(1.5)
            .with_cooccurrence_weight(0.8);

        assert_eq!(config.exact_match_weight, 1.5);
        assert_eq!(config.cooccurrence_weight, 0.8);
    }

    #[test]
    fn test_rbf_kernel_config() {
        let config = RbfKernelConfig::new(0.5);
        assert_eq!(config.gamma, 0.5);

        let config = RbfKernelConfig::from_sigma(2.0);
        assert!((config.gamma - 0.125).abs() < 1e-10);
    }
}
