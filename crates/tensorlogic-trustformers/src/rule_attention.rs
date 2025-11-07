//! Rule-based attention patterns for interpretable transformers.
//!
//! This module provides attention mechanisms that can be constrained or guided
//! by logical rules, enabling interpretable and controllable attention patterns.
//!
//! ## Concepts
//!
//! 1. **Structured Attention**: Attention patterns defined by predicates
//! 2. **Rule-Guided Attention**: Soft constraints on attention weights
//! 3. **Hierarchical Attention**: Multi-level attention based on structural rules
//!
//! ## Example Rules
//!
//! ```text
//! // Syntactic attention: attend to syntactic parents
//! ATTEND(token_i, token_j) :- SyntacticParent(token_j, token_i)
//!
//! // Semantic attention: attend to semantically related tokens
//! ATTEND(token_i, token_j) :- SemanticSimilarity(token_i, token_j) > threshold
//!
//! // Coreference attention: attend to coreferent mentions
//! ATTEND(token_i, token_j) :- Coreference(token_i, token_j)
//! ```

use serde::{Deserialize, Serialize};
use tensorlogic_ir::{EinsumGraph, EinsumNode, TLExpr, Term};

use crate::{
    config::AttentionConfig,
    error::{Result, TrustformerError},
};

/// Type of rule-based attention pattern
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RuleAttentionType {
    /// Hard constraint: only attend where rule is satisfied
    Hard,
    /// Soft constraint: bias attention towards rule-satisfying positions
    Soft {
        /// Strength of the soft constraint (0.0 = no effect, 1.0 = strong bias)
        strength: f64,
    },
    /// Gated: combine rule-based and content-based attention
    Gated {
        /// Weight for rule-based attention (0.0 = all content, 1.0 = all rule)
        rule_weight: f64,
    },
}

/// Configuration for rule-based attention
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuleAttentionConfig {
    /// Base attention configuration
    pub base_attention: AttentionConfig,
    /// Type of rule-based attention
    pub rule_type: RuleAttentionType,
    /// Whether to normalize attention after applying rules
    pub normalize_after_rules: bool,
}

impl RuleAttentionConfig {
    /// Create a new hard rule-based attention configuration
    pub fn hard(base_attention: AttentionConfig) -> Self {
        Self {
            base_attention,
            rule_type: RuleAttentionType::Hard,
            normalize_after_rules: true,
        }
    }

    /// Create a new soft rule-based attention configuration
    pub fn soft(base_attention: AttentionConfig, strength: f64) -> Self {
        Self {
            base_attention,
            rule_type: RuleAttentionType::Soft { strength },
            normalize_after_rules: true,
        }
    }

    /// Create a new gated rule-based attention configuration
    pub fn gated(base_attention: AttentionConfig, rule_weight: f64) -> Self {
        Self {
            base_attention,
            rule_type: RuleAttentionType::Gated { rule_weight },
            normalize_after_rules: true,
        }
    }

    /// Set whether to normalize after applying rules
    pub fn with_normalize_after_rules(mut self, normalize: bool) -> Self {
        self.normalize_after_rules = normalize;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.base_attention.validate()?;

        match &self.rule_type {
            RuleAttentionType::Soft { strength } => {
                if !(0.0..=1.0).contains(strength) {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: format!("strength must be in [0,1], got {}", strength),
                    });
                }
            }
            RuleAttentionType::Gated { rule_weight } => {
                if !(0.0..=1.0).contains(rule_weight) {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: format!("rule_weight must be in [0,1], got {}", rule_weight),
                    });
                }
            }
            RuleAttentionType::Hard => {}
        }

        Ok(())
    }
}

/// Rule-based attention component
#[derive(Clone, Debug)]
pub struct RuleBasedAttention {
    /// Configuration
    pub config: RuleAttentionConfig,
    /// Logical rule defining attention pattern
    pub attention_rule: Option<TLExpr>,
}

impl RuleBasedAttention {
    /// Create a new rule-based attention component
    pub fn new(config: RuleAttentionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            attention_rule: None,
        })
    }

    /// Set the attention rule
    pub fn with_rule(mut self, rule: TLExpr) -> Self {
        self.attention_rule = Some(rule);
        self
    }

    /// Build einsum graph for rule-based attention
    ///
    /// Input tensors:
    /// - 0: Q (query) [batch, seq_len, d_model]
    /// - 1: K (key) [batch, seq_len, d_model]
    /// - 2: V (value) [batch, seq_len, d_model]
    /// - 3: rule_mask [batch, seq_q, seq_k] (attention pattern from rules)
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_rule_attention_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Compute content-based attention scores
        let scores_tensor = graph.add_tensor("rule_attn_scores");
        let scores_node = EinsumNode::new("bqd,bkd->bqk", vec![0, 1], vec![scores_tensor]);
        graph.add_node(scores_node)?;

        // Step 2: Scale scores
        let scale_factor = (self.config.base_attention.d_k as f64).sqrt();
        let scale_tensor = graph.add_tensor("rule_attn_scale");
        let scaled_tensor = graph.add_tensor("rule_attn_scaled_scores");
        let scale_node = EinsumNode::elem_binary(
            format!("div_scalar_{}", scale_factor),
            scores_tensor,
            scale_tensor,
            scaled_tensor,
        );
        graph.add_node(scale_node)?;

        // Step 3: Apply rule-based modification
        let modified_scores = match &self.config.rule_type {
            RuleAttentionType::Hard => {
                // Hard constraint: mask out positions not allowed by rule
                let masked_tensor = graph.add_tensor("rule_hard_masked");
                let mask_node = EinsumNode::elem_binary("mul", scaled_tensor, 3, masked_tensor);
                graph.add_node(mask_node)?;
                masked_tensor
            }
            RuleAttentionType::Soft { strength } => {
                // Soft constraint: add bias based on rule satisfaction
                let strength_const = graph.add_tensor("strength_const");
                let scaled_bias = graph.add_tensor("scaled_bias");

                // Scale rule mask by strength
                let scale_bias_node = EinsumNode::elem_binary(
                    format!("mul_scalar_{}", strength),
                    3,
                    strength_const,
                    scaled_bias,
                );
                graph.add_node(scale_bias_node)?;

                // Add bias to scores
                let biased_tensor = graph.add_tensor("rule_soft_biased");
                let add_node =
                    EinsumNode::elem_binary("add", scaled_tensor, scaled_bias, biased_tensor);
                graph.add_node(add_node)?;
                biased_tensor
            }
            RuleAttentionType::Gated { rule_weight } => {
                // Gated: interpolate between content and rule attention
                let content_weight = 1.0 - rule_weight;

                // Content component
                let content_weighted = graph.add_tensor("rule_content_weighted");
                let content_const = graph.add_tensor("content_weight_const");
                let content_node = EinsumNode::elem_binary(
                    format!("mul_scalar_{}", content_weight),
                    scaled_tensor,
                    content_const,
                    content_weighted,
                );
                graph.add_node(content_node)?;

                // Rule component
                let rule_weighted = graph.add_tensor("rule_weighted");
                let rule_const = graph.add_tensor("rule_weight_const");
                let rule_node = EinsumNode::elem_binary(
                    format!("mul_scalar_{}", rule_weight),
                    3,
                    rule_const,
                    rule_weighted,
                );
                graph.add_node(rule_node)?;

                // Combine
                let gated_tensor = graph.add_tensor("rule_gated");
                let gate_node =
                    EinsumNode::elem_binary("add", content_weighted, rule_weighted, gated_tensor);
                graph.add_node(gate_node)?;
                gated_tensor
            }
        };

        // Step 4: Apply softmax
        let softmax_tensor = graph.add_tensor("rule_attention_weights");
        let softmax_node = EinsumNode::elem_unary("softmax_k", modified_scores, softmax_tensor);
        graph.add_node(softmax_node)?;

        // Step 5: Apply attention to values
        let output_tensor = graph.add_tensor("rule_attn_output");
        let output_node =
            EinsumNode::new("bqk,bkv->bqv", vec![softmax_tensor, 2], vec![output_tensor]);
        graph.add_node(output_node)?;

        Ok(vec![output_tensor])
    }

    /// Get the attention rule if set
    pub fn get_rule(&self) -> Option<&TLExpr> {
        self.attention_rule.as_ref()
    }
}

/// Structured attention based on explicit predicates
///
/// This variant directly uses predicates to define attention patterns
/// without computing content-based scores.
#[derive(Clone, Debug)]
pub struct StructuredAttention {
    /// Configuration
    pub config: AttentionConfig,
    /// Predicate defining attention structure
    pub structure_predicate: Option<TLExpr>,
}

impl StructuredAttention {
    /// Create a new structured attention component
    pub fn new(config: AttentionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            structure_predicate: None,
        })
    }

    /// Set the structure predicate
    ///
    /// The predicate should have signature: Attend(source_idx, target_idx) -> bool
    pub fn with_predicate(mut self, predicate: TLExpr) -> Self {
        self.structure_predicate = Some(predicate);
        self
    }

    /// Build einsum graph for structured attention
    ///
    /// Input tensors:
    /// - 0: V (value) [batch, seq_len, d_model]
    /// - 1: structure_matrix [batch, seq_q, seq_k] (from predicate evaluation)
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_structured_attention_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Normalize structure matrix (ensure it sums to 1)
        let normalized_tensor = graph.add_tensor("struct_attn_normalized");
        let norm_node = EinsumNode::elem_unary("softmax_k", 1, normalized_tensor);
        graph.add_node(norm_node)?;

        // Step 2: Apply structured attention to values
        let output_tensor = graph.add_tensor("struct_attn_output");
        let output_node = EinsumNode::new(
            "bqk,bkv->bqv",
            vec![normalized_tensor, 0],
            vec![output_tensor],
        );
        graph.add_node(output_node)?;

        Ok(vec![output_tensor])
    }

    /// Get the structure predicate if set
    pub fn get_predicate(&self) -> Option<&TLExpr> {
        self.structure_predicate.as_ref()
    }
}

/// Helper function to create common rule patterns
pub mod patterns {
    use super::*;

    /// Create a syntactic dependency attention rule
    pub fn syntactic_dependency(head_idx: &str, dep_idx: &str) -> TLExpr {
        TLExpr::Pred {
            name: "SyntacticDep".to_string(),
            args: vec![
                Term::Var(head_idx.to_string()),
                Term::Var(dep_idx.to_string()),
            ],
        }
    }

    /// Create a coreference attention rule
    pub fn coreference(mention1: &str, mention2: &str) -> TLExpr {
        TLExpr::Pred {
            name: "Coref".to_string(),
            args: vec![
                Term::Var(mention1.to_string()),
                Term::Var(mention2.to_string()),
            ],
        }
    }

    /// Create a semantic similarity attention rule
    pub fn semantic_similarity(token1: &str, token2: &str, threshold: f64) -> TLExpr {
        let sim_pred = TLExpr::Pred {
            name: "Similarity".to_string(),
            args: vec![Term::Var(token1.to_string()), Term::Var(token2.to_string())],
        };

        let threshold_term = Term::Const(format!("{}", threshold));

        TLExpr::Pred {
            name: "GreaterThan".to_string(),
            args: vec![Term::Const(format!("{:?}", sim_pred)), threshold_term],
        }
    }

    /// Create a hierarchical attention rule (e.g., sentence -> paragraph)
    pub fn hierarchical(child: &str, parent: &str) -> TLExpr {
        TLExpr::Pred {
            name: "ContainedIn".to_string(),
            args: vec![Term::Var(child.to_string()), Term::Var(parent.to_string())],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_rule_attention_config() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = RuleAttentionConfig::hard(base);
        assert!(matches!(config.rule_type, RuleAttentionType::Hard));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_soft_rule_attention_config() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = RuleAttentionConfig::soft(base, 0.5);
        assert!(matches!(
            config.rule_type,
            RuleAttentionType::Soft { strength: 0.5 }
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gated_rule_attention_config() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = RuleAttentionConfig::gated(base, 0.7);
        assert!(matches!(
            config.rule_type,
            RuleAttentionType::Gated { rule_weight: 0.7 }
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rule_based_attention_creation() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = RuleAttentionConfig::hard(base);
        let attn = RuleBasedAttention::new(config).unwrap();
        assert!(attn.get_rule().is_none());
    }

    #[test]
    fn test_rule_based_attention_with_rule() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = RuleAttentionConfig::hard(base);
        let rule = patterns::syntactic_dependency("i", "j");
        let attn = RuleBasedAttention::new(config).unwrap().with_rule(rule);
        assert!(attn.get_rule().is_some());
    }

    #[test]
    fn test_rule_based_attention_graph_building() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = RuleAttentionConfig::soft(base, 0.5);
        let attn = RuleBasedAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");
        graph.add_tensor("rule_mask");

        let outputs = attn.build_rule_attention_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_structured_attention_creation() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let attn = StructuredAttention::new(config).unwrap();
        assert!(attn.get_predicate().is_none());
    }

    #[test]
    fn test_structured_attention_with_predicate() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let predicate = patterns::coreference("m1", "m2");
        let attn = StructuredAttention::new(config)
            .unwrap()
            .with_predicate(predicate);
        assert!(attn.get_predicate().is_some());
    }

    #[test]
    fn test_structured_attention_graph_building() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let attn = StructuredAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("V");
        graph.add_tensor("structure_matrix");

        let outputs = attn.build_structured_attention_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_pattern_syntactic_dependency() {
        let rule = patterns::syntactic_dependency("head", "dep");
        match rule {
            TLExpr::Pred { name, args } => {
                assert_eq!(name, "SyntacticDep");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Pred"),
        }
    }

    #[test]
    fn test_pattern_coreference() {
        let rule = patterns::coreference("m1", "m2");
        match rule {
            TLExpr::Pred { name, args } => {
                assert_eq!(name, "Coref");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Pred"),
        }
    }

    #[test]
    fn test_invalid_soft_strength() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = RuleAttentionConfig::soft(base, 1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_gated_weight() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = RuleAttentionConfig::gated(base, -0.1);
        assert!(config.validate().is_err());
    }
}
