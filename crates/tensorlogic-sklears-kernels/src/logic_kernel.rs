//! Logic-derived similarity kernels.
//!
//! These kernels measure similarity based on logical rule satisfaction patterns.
//! Two inputs are similar if they satisfy the same set of logical rules.

use tensorlogic_ir::TLExpr;

use crate::error::{KernelError, Result};
use crate::types::{Kernel, RuleSimilarityConfig};

/// Logic-based similarity kernel.
///
/// Measures similarity based on which logical rules are satisfied by each input.
///
/// ## Formula
///
/// ```text
/// K(x, y) = Σ_r w_r * agreement(x, y, r)
/// ```
///
/// Where:
/// - `r` ranges over logical rules
/// - `w_r` is the weight for rule r
/// - `agreement(x, y, r)` measures if x and y agree on rule r:
///   - Both satisfy: satisfied_weight
///   - Both violate: violated_weight
///   - Disagree: mixed_weight
pub struct RuleSimilarityKernel {
    /// Logical rules for comparison
    rules: Vec<TLExpr>,
    /// Configuration
    config: RuleSimilarityConfig,
}

impl RuleSimilarityKernel {
    /// Create a new rule similarity kernel
    pub fn new(rules: Vec<TLExpr>, config: RuleSimilarityConfig) -> Result<Self> {
        if rules.is_empty() {
            return Err(KernelError::InvalidParameter {
                parameter: "rules".to_string(),
                value: "empty".to_string(),
                reason: "at least one rule required".to_string(),
            });
        }

        Ok(Self { rules, config })
    }

    /// Evaluate if input satisfies a rule (simplified: uses feature index as rule ID)
    fn evaluate_rule(&self, input: &[f64], rule_idx: usize) -> bool {
        // Simplified: Check if the feature value at rule_idx > 0.5
        // Real implementation would compile and execute the TLExpr
        if rule_idx < input.len() {
            input[rule_idx] > 0.5
        } else {
            false
        }
    }

    /// Compute agreement between two inputs on a rule
    fn compute_agreement(&self, x: &[f64], y: &[f64], rule_idx: usize) -> f64 {
        let x_satisfies = self.evaluate_rule(x, rule_idx);
        let y_satisfies = self.evaluate_rule(y, rule_idx);

        match (x_satisfies, y_satisfies) {
            (true, true) => self.config.satisfied_weight,
            (false, false) => self.config.violated_weight,
            _ => self.config.mixed_weight,
        }
    }
}

impl Kernel for RuleSimilarityKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(KernelError::DimensionMismatch {
                expected: vec![x.len()],
                got: vec![y.len()],
                context: "rule similarity kernel".to_string(),
            });
        }

        let mut similarity = 0.0;
        for rule_idx in 0..self.rules.len() {
            similarity += self.compute_agreement(x, y, rule_idx);
        }

        if self.config.normalize {
            similarity /= self.rules.len() as f64;
        }

        Ok(similarity)
    }

    fn name(&self) -> &str {
        "RuleSimilarity"
    }
}

/// Predicate overlap kernel.
///
/// Measures similarity based on how many predicates are true for both inputs.
pub struct PredicateOverlapKernel {
    /// Number of predicates to consider
    n_predicates: usize,
    /// Weight for each predicate
    predicate_weights: Vec<f64>,
}

impl PredicateOverlapKernel {
    /// Create a new predicate overlap kernel
    pub fn new(n_predicates: usize) -> Self {
        Self {
            n_predicates,
            predicate_weights: vec![1.0; n_predicates],
        }
    }

    /// Create with custom predicate weights
    pub fn with_weights(n_predicates: usize, weights: Vec<f64>) -> Result<Self> {
        if weights.len() != n_predicates {
            return Err(KernelError::DimensionMismatch {
                expected: vec![n_predicates],
                got: vec![weights.len()],
                context: "predicate weights".to_string(),
            });
        }

        Ok(Self {
            n_predicates,
            predicate_weights: weights,
        })
    }

    /// Check if predicate is satisfied (threshold at 0.5)
    fn is_predicate_true(&self, input: &[f64], pred_idx: usize) -> bool {
        if pred_idx < input.len() {
            input[pred_idx] > 0.5
        } else {
            false
        }
    }
}

impl Kernel for PredicateOverlapKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() < self.n_predicates || y.len() < self.n_predicates {
            return Err(KernelError::DimensionMismatch {
                expected: vec![self.n_predicates],
                got: vec![x.len().min(y.len())],
                context: "predicate overlap kernel".to_string(),
            });
        }

        let mut overlap = 0.0;
        for pred_idx in 0..self.n_predicates {
            if self.is_predicate_true(x, pred_idx) && self.is_predicate_true(y, pred_idx) {
                overlap += self.predicate_weights[pred_idx];
            }
        }

        // Normalize by total weight
        let total_weight: f64 = self.predicate_weights.iter().sum();
        Ok(overlap / total_weight)
    }

    fn name(&self) -> &str {
        "PredicateOverlap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_rules(n: usize) -> Vec<TLExpr> {
        (0..n)
            .map(|i| TLExpr::pred(format!("rule_{}", i), vec![]))
            .collect()
    }

    #[test]
    fn test_rule_similarity_kernel_creation() {
        let rules = create_dummy_rules(5);
        let config = RuleSimilarityConfig::new();
        let kernel = RuleSimilarityKernel::new(rules, config).unwrap();
        assert_eq!(kernel.name(), "RuleSimilarity");
    }

    #[test]
    fn test_rule_similarity_kernel_empty_rules() {
        let rules = vec![];
        let config = RuleSimilarityConfig::new();
        let result = RuleSimilarityKernel::new(rules, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_rule_similarity_compute() {
        let rules = create_dummy_rules(3);
        let config = RuleSimilarityConfig::new()
            .with_satisfied_weight(1.0)
            .with_violated_weight(0.5)
            .with_mixed_weight(0.0);

        let kernel = RuleSimilarityKernel::new(rules, config).unwrap();

        // Both satisfy all rules
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0, 1.0, 1.0];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 1.0).abs() < 1e-10); // Normalized: 3.0 / 3 = 1.0

        // Both violate all rules
        let x = vec![0.0, 0.0, 0.0];
        let y = vec![0.0, 0.0, 0.0];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 0.5).abs() < 1e-10); // Normalized: 1.5 / 3 = 0.5

        // Completely disagree
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![0.0, 0.0, 0.0];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim.abs() < 1e-10); // Normalized: 0.0 / 3 = 0.0
    }

    #[test]
    fn test_rule_similarity_dimension_mismatch() {
        let rules = create_dummy_rules(3);
        let config = RuleSimilarityConfig::new();
        let kernel = RuleSimilarityKernel::new(rules, config).unwrap();

        let x = vec![1.0, 1.0];
        let y = vec![1.0, 1.0, 1.0];
        let result = kernel.compute(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_predicate_overlap_kernel() {
        let kernel = PredicateOverlapKernel::new(4);
        assert_eq!(kernel.name(), "PredicateOverlap");

        // All predicates match
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = vec![1.0, 1.0, 1.0, 1.0];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);

        // Half predicates match
        let x = vec![1.0, 1.0, 0.0, 0.0];
        let y = vec![1.0, 1.0, 1.0, 1.0];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 0.5).abs() < 1e-10);

        // No predicates match
        let x = vec![0.0, 0.0, 0.0, 0.0];
        let y = vec![1.0, 1.0, 1.0, 1.0];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_predicate_overlap_with_weights() {
        let weights = vec![1.0, 2.0, 1.0, 2.0]; // Total = 6.0
        let kernel = PredicateOverlapKernel::with_weights(4, weights).unwrap();

        // Higher-weighted predicates match
        let x = vec![0.0, 1.0, 0.0, 1.0];
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let sim = kernel.compute(&x, &y).unwrap();
        assert!((sim - 4.0 / 6.0).abs() < 1e-10); // (2.0 + 2.0) / 6.0
    }

    #[test]
    fn test_predicate_overlap_dimension_mismatch() {
        let kernel = PredicateOverlapKernel::new(5);
        let x = vec![1.0, 1.0]; // Only 2 features, need 5
        let y = vec![1.0, 1.0];
        let result = kernel.compute(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_matrix_computation() {
        let kernel = PredicateOverlapKernel::new(3);
        let inputs = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];

        let matrix = kernel.compute_matrix(&inputs).unwrap();
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);

        // Check diagonal (self-similarity)
        for (i, row) in matrix.iter().enumerate().take(3) {
            assert!(row[i] >= 0.0);
        }

        // Check symmetry
        for (i, row) in matrix.iter().enumerate().take(3) {
            for j in 0..3 {
                assert!((row[j] - matrix[j][i]).abs() < 1e-10);
            }
        }
    }
}
