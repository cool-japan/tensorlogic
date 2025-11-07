//! Rule-based and sparse attention patterns.
//!
//! This module provides various attention masking patterns used in transformer
//! architectures, including causal masks, local attention, strided patterns,
//! and rule-based attention patterns.

use tensorlogic_ir::{EinsumGraph, EinsumNode};

use crate::error::{Result, TrustformerError};

/// Trait for attention mask patterns
pub trait AttentionMask {
    /// Build a mask tensor in the einsum graph
    ///
    /// Returns the tensor ID of the mask with shape [batch, seq_len, seq_len]
    /// where 0.0 = masked (no attention), 1.0 = unmasked (attend)
    fn build_mask(&self, graph: &mut EinsumGraph, seq_len: usize) -> Result<usize>;

    /// Get mask type name for documentation
    fn mask_type(&self) -> &str;
}

/// Causal (autoregressive) attention mask
///
/// Prevents positions from attending to subsequent positions.
/// mask[i, j] = 1 if i >= j, else 0
#[derive(Clone, Debug)]
pub struct CausalMask {
    /// Batch size
    pub batch_size: usize,
}

impl CausalMask {
    /// Create a new causal mask
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }
}

impl AttentionMask for CausalMask {
    fn build_mask(&self, graph: &mut EinsumGraph, seq_len: usize) -> Result<usize> {
        // Create causal mask: lower triangular matrix
        let mask_tensor = graph.add_tensor("causal_mask");
        let mask_node = EinsumNode::elem_unary(
            format!("causal_mask_{}x{}", seq_len, seq_len),
            0, // Placeholder for mask generation
            mask_tensor,
        );
        graph.add_node(mask_node)?;
        Ok(mask_tensor)
    }

    fn mask_type(&self) -> &str {
        "causal"
    }
}

/// Local (windowed) attention mask
///
/// Each position attends only to a fixed window of nearby positions.
#[derive(Clone, Debug)]
pub struct LocalMask {
    /// Batch size
    pub batch_size: usize,
    /// Window size (attends to ±window_size positions)
    pub window_size: usize,
}

impl LocalMask {
    /// Create a new local attention mask
    pub fn new(batch_size: usize, window_size: usize) -> Self {
        Self {
            batch_size,
            window_size,
        }
    }
}

impl AttentionMask for LocalMask {
    fn build_mask(&self, graph: &mut EinsumGraph, seq_len: usize) -> Result<usize> {
        let mask_tensor = graph.add_tensor("local_mask");
        let mask_node = EinsumNode::elem_unary(
            format!("local_mask_w{}_{}x{}", self.window_size, seq_len, seq_len),
            0,
            mask_tensor,
        );
        graph.add_node(mask_node)?;
        Ok(mask_tensor)
    }

    fn mask_type(&self) -> &str {
        "local"
    }
}

/// Strided attention mask
///
/// Attends to every k-th position (used in Sparse Transformers).
#[derive(Clone, Debug)]
pub struct StridedMask {
    /// Batch size
    pub batch_size: usize,
    /// Stride length
    pub stride: usize,
}

impl StridedMask {
    /// Create a new strided attention mask
    pub fn new(batch_size: usize, stride: usize) -> Result<Self> {
        if stride == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "stride must be positive".to_string(),
            });
        }
        Ok(Self { batch_size, stride })
    }
}

impl AttentionMask for StridedMask {
    fn build_mask(&self, graph: &mut EinsumGraph, seq_len: usize) -> Result<usize> {
        let mask_tensor = graph.add_tensor("strided_mask");
        let mask_node = EinsumNode::elem_unary(
            format!("strided_mask_s{}_{}x{}", self.stride, seq_len, seq_len),
            0,
            mask_tensor,
        );
        graph.add_node(mask_node)?;
        Ok(mask_tensor)
    }

    fn mask_type(&self) -> &str {
        "strided"
    }
}

/// Block-sparse attention mask
///
/// Divides attention into fixed-size blocks (used in BigBird, Longformer).
#[derive(Clone, Debug)]
pub struct BlockSparseMask {
    /// Batch size
    pub batch_size: usize,
    /// Block size
    pub block_size: usize,
}

impl BlockSparseMask {
    /// Create a new block-sparse attention mask
    pub fn new(batch_size: usize, block_size: usize) -> Result<Self> {
        if block_size == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "block_size must be positive".to_string(),
            });
        }
        Ok(Self {
            batch_size,
            block_size,
        })
    }
}

impl AttentionMask for BlockSparseMask {
    fn build_mask(&self, graph: &mut EinsumGraph, seq_len: usize) -> Result<usize> {
        let mask_tensor = graph.add_tensor("block_sparse_mask");
        let mask_node = EinsumNode::elem_unary(
            format!(
                "block_sparse_mask_b{}_{}x{}",
                self.block_size, seq_len, seq_len
            ),
            0,
            mask_tensor,
        );
        graph.add_node(mask_node)?;
        Ok(mask_tensor)
    }

    fn mask_type(&self) -> &str {
        "block_sparse"
    }
}

/// Global + Local attention mask
///
/// Combines global tokens (attend to all) with local windows (Longformer-style).
#[derive(Clone, Debug)]
pub struct GlobalLocalMask {
    /// Batch size
    pub batch_size: usize,
    /// Number of global tokens at the start
    pub num_global_tokens: usize,
    /// Window size for local attention
    pub local_window: usize,
}

impl GlobalLocalMask {
    /// Create a new global+local attention mask
    pub fn new(batch_size: usize, num_global_tokens: usize, local_window: usize) -> Self {
        Self {
            batch_size,
            num_global_tokens,
            local_window,
        }
    }
}

impl AttentionMask for GlobalLocalMask {
    fn build_mask(&self, graph: &mut EinsumGraph, seq_len: usize) -> Result<usize> {
        let mask_tensor = graph.add_tensor("global_local_mask");
        let mask_node = EinsumNode::elem_unary(
            format!(
                "global_local_mask_g{}_w{}_{}x{}",
                self.num_global_tokens, self.local_window, seq_len, seq_len
            ),
            0,
            mask_tensor,
        );
        graph.add_node(mask_node)?;
        Ok(mask_tensor)
    }

    fn mask_type(&self) -> &str {
        "global_local"
    }
}

/// Rule-based attention pattern
///
/// Applies attention based on logical rules (hard, soft, or gated).
#[derive(Clone, Debug)]
pub enum RulePattern {
    /// Hard masking: 0 or 1 based on rule satisfaction
    Hard,
    /// Soft masking: continuous values based on rule confidence
    Soft,
    /// Gated masking: learnable combination of rule and data-driven attention
    Gated,
}

/// Rule-based attention mask
#[derive(Clone, Debug)]
pub struct RuleBasedMask {
    /// Batch size
    pub batch_size: usize,
    /// Pattern type
    pub pattern: RulePattern,
    /// Rule specification (opaque for now)
    pub rule_spec: String,
}

impl RuleBasedMask {
    /// Create a new rule-based mask
    pub fn new(batch_size: usize, pattern: RulePattern, rule_spec: String) -> Self {
        Self {
            batch_size,
            pattern,
            rule_spec,
        }
    }

    /// Create a hard rule-based mask
    pub fn hard(batch_size: usize, rule_spec: String) -> Self {
        Self::new(batch_size, RulePattern::Hard, rule_spec)
    }

    /// Create a soft rule-based mask
    pub fn soft(batch_size: usize, rule_spec: String) -> Self {
        Self::new(batch_size, RulePattern::Soft, rule_spec)
    }

    /// Create a gated rule-based mask
    pub fn gated(batch_size: usize, rule_spec: String) -> Self {
        Self::new(batch_size, RulePattern::Gated, rule_spec)
    }
}

impl AttentionMask for RuleBasedMask {
    fn build_mask(&self, graph: &mut EinsumGraph, seq_len: usize) -> Result<usize> {
        let pattern_name = match self.pattern {
            RulePattern::Hard => "hard",
            RulePattern::Soft => "soft",
            RulePattern::Gated => "gated",
        };

        let mask_tensor = graph.add_tensor(format!("rule_mask_{}", pattern_name));
        let mask_node = EinsumNode::elem_unary(
            format!("rule_mask_{}_{}x{}", pattern_name, seq_len, seq_len),
            0,
            mask_tensor,
        );
        graph.add_node(mask_node)?;
        Ok(mask_tensor)
    }

    fn mask_type(&self) -> &str {
        match self.pattern {
            RulePattern::Hard => "rule_hard",
            RulePattern::Soft => "rule_soft",
            RulePattern::Gated => "rule_gated",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_creation() {
        let mask = CausalMask::new(4);
        assert_eq!(mask.batch_size, 4);
        assert_eq!(mask.mask_type(), "causal");
    }

    #[test]
    fn test_causal_mask_build() {
        let mask = CausalMask::new(4);
        let mut graph = EinsumGraph::new();
        let result = mask.build_mask(&mut graph, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_local_mask_creation() {
        let mask = LocalMask::new(4, 3);
        assert_eq!(mask.batch_size, 4);
        assert_eq!(mask.window_size, 3);
        assert_eq!(mask.mask_type(), "local");
    }

    #[test]
    fn test_local_mask_build() {
        let mask = LocalMask::new(4, 5);
        let mut graph = EinsumGraph::new();
        let result = mask.build_mask(&mut graph, 20);
        assert!(result.is_ok());
    }

    #[test]
    fn test_strided_mask_creation() {
        let mask = StridedMask::new(4, 2).unwrap();
        assert_eq!(mask.batch_size, 4);
        assert_eq!(mask.stride, 2);
        assert_eq!(mask.mask_type(), "strided");
    }

    #[test]
    fn test_strided_mask_invalid_stride() {
        let result = StridedMask::new(4, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_strided_mask_build() {
        let mask = StridedMask::new(4, 3).unwrap();
        let mut graph = EinsumGraph::new();
        let result = mask.build_mask(&mut graph, 15);
        assert!(result.is_ok());
    }

    #[test]
    fn test_block_sparse_mask_creation() {
        let mask = BlockSparseMask::new(4, 8).unwrap();
        assert_eq!(mask.batch_size, 4);
        assert_eq!(mask.block_size, 8);
        assert_eq!(mask.mask_type(), "block_sparse");
    }

    #[test]
    fn test_block_sparse_mask_invalid_size() {
        let result = BlockSparseMask::new(4, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_block_sparse_mask_build() {
        let mask = BlockSparseMask::new(4, 16).unwrap();
        let mut graph = EinsumGraph::new();
        let result = mask.build_mask(&mut graph, 64);
        assert!(result.is_ok());
    }

    #[test]
    fn test_global_local_mask_creation() {
        let mask = GlobalLocalMask::new(4, 2, 5);
        assert_eq!(mask.batch_size, 4);
        assert_eq!(mask.num_global_tokens, 2);
        assert_eq!(mask.local_window, 5);
        assert_eq!(mask.mask_type(), "global_local");
    }

    #[test]
    fn test_global_local_mask_build() {
        let mask = GlobalLocalMask::new(4, 3, 7);
        let mut graph = EinsumGraph::new();
        let result = mask.build_mask(&mut graph, 50);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rule_based_mask_hard() {
        let mask = RuleBasedMask::hard(4, "entity_type=person".to_string());
        assert_eq!(mask.batch_size, 4);
        assert!(matches!(mask.pattern, RulePattern::Hard));
        assert_eq!(mask.mask_type(), "rule_hard");
    }

    #[test]
    fn test_rule_based_mask_soft() {
        let mask = RuleBasedMask::soft(4, "similarity>0.5".to_string());
        assert!(matches!(mask.pattern, RulePattern::Soft));
        assert_eq!(mask.mask_type(), "rule_soft");
    }

    #[test]
    fn test_rule_based_mask_gated() {
        let mask = RuleBasedMask::gated(4, "weighted_rule".to_string());
        assert!(matches!(mask.pattern, RulePattern::Gated));
        assert_eq!(mask.mask_type(), "rule_gated");
    }

    #[test]
    fn test_rule_based_mask_build() {
        let mask = RuleBasedMask::hard(4, "test_rule".to_string());
        let mut graph = EinsumGraph::new();
        let result = mask.build_mask(&mut graph, 32);
        assert!(result.is_ok());
    }
}
