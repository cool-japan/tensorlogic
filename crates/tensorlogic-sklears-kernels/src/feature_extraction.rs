//! Automatic feature extraction from logical expressions.
//!
//! Converts TLExpr into feature vectors for use with kernel methods.

use std::collections::HashMap;

use tensorlogic_ir::TLExpr;

use crate::error::Result;

/// Feature extractor for logical expressions
///
/// Automatically converts TLExpr into numerical feature vectors
/// suitable for kernel computation.
///
/// # Example
///
/// ```rust
/// use tensorlogic_sklears_kernels::{FeatureExtractor, FeatureExtractionConfig};
/// use tensorlogic_ir::TLExpr;
///
/// let config = FeatureExtractionConfig::new()
///     .with_max_depth(3)
///     .with_encode_structure(true);
///
/// let extractor = FeatureExtractor::new(config);
///
/// let expr = TLExpr::and(
///     TLExpr::pred("tall", vec![]),
///     TLExpr::pred("smart", vec![]),
/// );
///
/// let features = extractor.extract(&expr).unwrap();
/// println!("Feature vector: {:?}", features);
/// ```
#[derive(Clone, Debug)]
pub struct FeatureExtractor {
    config: FeatureExtractionConfig,
    /// Vocabulary for predicate names
    vocabulary: HashMap<String, usize>,
}

/// Configuration for feature extraction
#[derive(Clone, Debug)]
pub struct FeatureExtractionConfig {
    /// Maximum tree depth to encode
    pub max_depth: usize,
    /// Whether to encode structural information
    pub encode_structure: bool,
    /// Whether to encode quantifier information
    pub encode_quantifiers: bool,
    /// Feature vector dimension (if fixed)
    pub fixed_dimension: Option<usize>,
}

impl FeatureExtractionConfig {
    /// Create default configuration
    pub fn new() -> Self {
        Self {
            max_depth: 5,
            encode_structure: true,
            encode_quantifiers: true,
            fixed_dimension: None,
        }
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set whether to encode structure
    pub fn with_encode_structure(mut self, encode: bool) -> Self {
        self.encode_structure = encode;
        self
    }

    /// Set whether to encode quantifiers
    pub fn with_encode_quantifiers(mut self, encode: bool) -> Self {
        self.encode_quantifiers = encode;
        self
    }

    /// Set fixed feature dimension
    pub fn with_fixed_dimension(mut self, dim: usize) -> Self {
        self.fixed_dimension = Some(dim);
        self
    }
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(config: FeatureExtractionConfig) -> Self {
        Self {
            config,
            vocabulary: HashMap::new(),
        }
    }

    /// Extract features from a logical expression
    pub fn extract(&self, expr: &TLExpr) -> Result<Vec<f64>> {
        let mut features = Vec::new();

        // Extract predicate frequencies
        let pred_counts = self.count_predicates(expr);

        // Extract structural features
        if self.config.encode_structure {
            features.extend(self.extract_structural_features(expr));
        }

        // Extract predicate features
        features.extend(self.extract_predicate_features(&pred_counts));

        // Extract quantifier features
        if self.config.encode_quantifiers {
            features.extend(self.extract_quantifier_features(expr));
        }

        // Pad or truncate to fixed dimension if specified
        if let Some(dim) = self.config.fixed_dimension {
            features.resize(dim, 0.0);
        }

        Ok(features)
    }

    /// Extract features from multiple expressions
    pub fn extract_batch(&self, exprs: &[TLExpr]) -> Result<Vec<Vec<f64>>> {
        exprs.iter().map(|expr| self.extract(expr)).collect()
    }

    /// Build vocabulary from a set of expressions
    pub fn build_vocabulary(&mut self, exprs: &[TLExpr]) {
        let mut vocab_index = 0;

        for expr in exprs {
            self.collect_predicates(expr, &mut vocab_index);
        }
    }

    /// Collect predicates from expression
    fn collect_predicates(&mut self, expr: &TLExpr, vocab_index: &mut usize) {
        match expr {
            TLExpr::Pred { name, .. } => {
                if !self.vocabulary.contains_key(name) {
                    self.vocabulary.insert(name.clone(), *vocab_index);
                    *vocab_index += 1;
                }
            }
            TLExpr::And(left, right) | TLExpr::Or(left, right) | TLExpr::Imply(left, right) => {
                self.collect_predicates(left, vocab_index);
                self.collect_predicates(right, vocab_index);
            }
            TLExpr::Not(inner) => {
                self.collect_predicates(inner, vocab_index);
            }
            TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
                self.collect_predicates(body, vocab_index);
            }
            _ => {}
        }
    }

    /// Count predicates in expression
    fn count_predicates(&self, expr: &TLExpr) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        self.count_predicates_recursive(expr, &mut counts);
        counts
    }

    #[allow(clippy::only_used_in_recursion)]
    fn count_predicates_recursive(&self, expr: &TLExpr, counts: &mut HashMap<String, usize>) {
        match expr {
            TLExpr::Pred { name, .. } => {
                *counts.entry(name.clone()).or_insert(0) += 1;
            }
            TLExpr::And(left, right) | TLExpr::Or(left, right) | TLExpr::Imply(left, right) => {
                self.count_predicates_recursive(left, counts);
                self.count_predicates_recursive(right, counts);
            }
            TLExpr::Not(inner) => {
                self.count_predicates_recursive(inner, counts);
            }
            TLExpr::Exists { body, .. } | TLExpr::ForAll { body, .. } => {
                self.count_predicates_recursive(body, counts);
            }
            _ => {}
        }
    }

    /// Extract structural features
    fn extract_structural_features(&self, expr: &TLExpr) -> Vec<f64> {
        vec![
            self.compute_depth(expr, 0) as f64,
            self.count_nodes(expr) as f64,
            self.count_operators(expr, "and") as f64,
            self.count_operators(expr, "or") as f64,
            self.count_operators(expr, "not") as f64,
            self.count_operators(expr, "imply") as f64,
        ]
    }

    /// Compute tree depth
    fn compute_depth(&self, expr: &TLExpr, current_depth: usize) -> usize {
        if current_depth >= self.config.max_depth {
            return current_depth;
        }

        match expr {
            TLExpr::Pred { .. } => current_depth,
            TLExpr::And(left, right) | TLExpr::Or(left, right) | TLExpr::Imply(left, right) => {
                let left_depth = self.compute_depth(left, current_depth + 1);
                let right_depth = self.compute_depth(right, current_depth + 1);
                left_depth.max(right_depth)
            }
            TLExpr::Not(inner)
            | TLExpr::Exists { body: inner, .. }
            | TLExpr::ForAll { body: inner, .. } => self.compute_depth(inner, current_depth + 1),
            _ => current_depth,
        }
    }

    /// Count total nodes
    #[allow(clippy::only_used_in_recursion)]
    fn count_nodes(&self, expr: &TLExpr) -> usize {
        match expr {
            TLExpr::Pred { .. } => 1,
            TLExpr::And(left, right) | TLExpr::Or(left, right) | TLExpr::Imply(left, right) => {
                1 + self.count_nodes(left) + self.count_nodes(right)
            }
            TLExpr::Not(inner)
            | TLExpr::Exists { body: inner, .. }
            | TLExpr::ForAll { body: inner, .. } => 1 + self.count_nodes(inner),
            _ => 1,
        }
    }

    /// Count specific operators
    #[allow(clippy::only_used_in_recursion)]
    fn count_operators(&self, expr: &TLExpr, op_type: &str) -> usize {
        let this_count = match (op_type, expr) {
            ("and", TLExpr::And(_, _)) => 1,
            ("or", TLExpr::Or(_, _)) => 1,
            ("not", TLExpr::Not(_)) => 1,
            ("imply", TLExpr::Imply(_, _)) => 1,
            _ => 0,
        };

        let child_count = match expr {
            TLExpr::And(left, right) | TLExpr::Or(left, right) | TLExpr::Imply(left, right) => {
                self.count_operators(left, op_type) + self.count_operators(right, op_type)
            }
            TLExpr::Not(inner)
            | TLExpr::Exists { body: inner, .. }
            | TLExpr::ForAll { body: inner, .. } => self.count_operators(inner, op_type),
            _ => 0,
        };

        this_count + child_count
    }

    /// Extract predicate features
    fn extract_predicate_features(&self, counts: &HashMap<String, usize>) -> Vec<f64> {
        if self.vocabulary.is_empty() {
            // If no vocabulary, return counts as-is
            counts.values().map(|&c| c as f64).collect()
        } else {
            // Use vocabulary for consistent ordering
            let mut features = vec![0.0; self.vocabulary.len()];
            for (pred, &count) in counts {
                if let Some(&idx) = self.vocabulary.get(pred) {
                    features[idx] = count as f64;
                }
            }
            features
        }
    }

    /// Extract quantifier features
    fn extract_quantifier_features(&self, expr: &TLExpr) -> Vec<f64> {
        vec![
            self.count_quantifiers(expr, "exists") as f64,
            self.count_quantifiers(expr, "forall") as f64,
        ]
    }

    /// Count quantifiers
    #[allow(clippy::only_used_in_recursion)]
    fn count_quantifiers(&self, expr: &TLExpr, quant_type: &str) -> usize {
        let this_count = match (quant_type, expr) {
            ("exists", TLExpr::Exists { .. }) => 1,
            ("forall", TLExpr::ForAll { .. }) => 1,
            _ => 0,
        };

        let child_count = match expr {
            TLExpr::And(left, right) | TLExpr::Or(left, right) | TLExpr::Imply(left, right) => {
                self.count_quantifiers(left, quant_type) + self.count_quantifiers(right, quant_type)
            }
            TLExpr::Not(inner)
            | TLExpr::Exists { body: inner, .. }
            | TLExpr::ForAll { body: inner, .. } => self.count_quantifiers(inner, quant_type),
            _ => 0,
        };

        this_count + child_count
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get vocabulary
    pub fn vocabulary(&self) -> &HashMap<String, usize> {
        &self.vocabulary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction_basic() {
        let config = FeatureExtractionConfig::new();
        let extractor = FeatureExtractor::new(config);

        let expr = TLExpr::pred("tall", vec![]);
        let features = extractor.extract(&expr).unwrap();

        assert!(!features.is_empty());
    }

    #[test]
    fn test_feature_extraction_compound() {
        let config = FeatureExtractionConfig::new();
        let extractor = FeatureExtractor::new(config);

        let expr = TLExpr::and(TLExpr::pred("tall", vec![]), TLExpr::pred("smart", vec![]));

        let features = extractor.extract(&expr).unwrap();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_structural_features() {
        let config = FeatureExtractionConfig::new().with_encode_structure(true);
        let extractor = FeatureExtractor::new(config);

        let expr = TLExpr::and(
            TLExpr::pred("a", vec![]),
            TLExpr::or(TLExpr::pred("b", vec![]), TLExpr::pred("c", vec![])),
        );

        let features = extractor.extract(&expr).unwrap();

        // Should have depth > 1
        assert!(features[0] > 1.0);

        // Should have multiple nodes
        assert!(features[1] > 1.0);
    }

    #[test]
    fn test_quantifier_features() {
        let config = FeatureExtractionConfig::new().with_encode_quantifiers(true);
        let extractor = FeatureExtractor::new(config);

        let expr = TLExpr::exists("x", "Person", TLExpr::pred("likes", vec![]));

        let features = extractor.extract(&expr).unwrap();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_vocabulary_building() {
        let config = FeatureExtractionConfig::new();
        let mut extractor = FeatureExtractor::new(config);

        let exprs = vec![
            TLExpr::pred("tall", vec![]),
            TLExpr::pred("smart", vec![]),
            TLExpr::pred("tall", vec![]),
        ];

        extractor.build_vocabulary(&exprs);

        assert_eq!(extractor.vocab_size(), 2); // tall and smart
    }

    #[test]
    fn test_batch_extraction() {
        let config = FeatureExtractionConfig::new();
        let extractor = FeatureExtractor::new(config);

        let exprs = vec![
            TLExpr::pred("a", vec![]),
            TLExpr::pred("b", vec![]),
            TLExpr::and(TLExpr::pred("a", vec![]), TLExpr::pred("b", vec![])),
        ];

        let features = extractor.extract_batch(&exprs).unwrap();
        assert_eq!(features.len(), 3);
    }

    #[test]
    fn test_fixed_dimension() {
        let config = FeatureExtractionConfig::new().with_fixed_dimension(10);
        let extractor = FeatureExtractor::new(config);

        let expr = TLExpr::pred("test", vec![]);
        let features = extractor.extract(&expr).unwrap();

        assert_eq!(features.len(), 10);
    }

    #[test]
    fn test_depth_computation() {
        let config = FeatureExtractionConfig::new();
        let extractor = FeatureExtractor::new(config);

        // Depth 0: single predicate
        let expr1 = TLExpr::pred("a", vec![]);
        assert_eq!(extractor.compute_depth(&expr1, 0), 0);

        // Depth 2: nested structure
        let expr2 = TLExpr::and(
            TLExpr::pred("a", vec![]),
            TLExpr::and(TLExpr::pred("b", vec![]), TLExpr::pred("c", vec![])),
        );
        assert_eq!(extractor.compute_depth(&expr2, 0), 2);
    }

    #[test]
    fn test_operator_counting() {
        let config = FeatureExtractionConfig::new();
        let extractor = FeatureExtractor::new(config);

        let expr = TLExpr::and(
            TLExpr::and(TLExpr::pred("a", vec![]), TLExpr::pred("b", vec![])),
            TLExpr::or(TLExpr::pred("c", vec![]), TLExpr::pred("d", vec![])),
        );

        assert_eq!(extractor.count_operators(&expr, "and"), 2);
        assert_eq!(extractor.count_operators(&expr, "or"), 1);
        assert_eq!(extractor.count_operators(&expr, "not"), 0);
    }
}
