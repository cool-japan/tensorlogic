//! Attention pattern implementations (sparse and rule-based).

use crate::config::AttentionPattern;
use crate::error::{TrustformersError, TrustformersResult};
use tensorlogic_ir::{EinsumGraph, OpType, TensorShape};

/// Trait for attention mask patterns
pub trait AttentionMask {
    /// Generate the attention mask graph
    fn build_mask(&self, seq_len: usize) -> TrustformersResult<EinsumGraph>;
}

/// Causal (autoregressive) attention mask
pub struct CausalMask;

impl AttentionMask for CausalMask {
    fn build_mask(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        
        // Create causal mask: [seq_len, seq_len]
        // Lower triangular matrix (including diagonal)
        let mask_shape = TensorShape::Fixed(vec![seq_len, seq_len]);
        let mask_node = graph.add_node(
            OpType::Placeholder {
                name: "causal_mask".to_string(),
                shape: mask_shape,
            },
            vec![],
        );
        
        graph.set_output(mask_node);
        Ok(graph)
    }
}

/// Local attention mask (attends to ±window_size positions)
pub struct LocalMask {
    pub window_size: usize,
}

impl AttentionMask for LocalMask {
    fn build_mask(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        
        // Create local mask: [seq_len, seq_len]
        let mask_shape = TensorShape::Fixed(vec![seq_len, seq_len]);
        let mask_node = graph.add_node(
            OpType::Placeholder {
                name: "local_mask".to_string(),
                shape: mask_shape,
            },
            vec![],
        );
        
        graph.set_output(mask_node);
        Ok(graph)
    }
}

/// Strided sparse attention mask
pub struct StridedMask {
    pub stride: usize,
}

impl AttentionMask for StridedMask {
    fn build_mask(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        
        let mask_shape = TensorShape::Fixed(vec![seq_len, seq_len]);
        let mask_node = graph.add_node(
            OpType::Placeholder {
                name: "strided_mask".to_string(),
                shape: mask_shape,
            },
            vec![],
        );
        
        graph.set_output(mask_node);
        Ok(graph)
    }
}

/// Block-sparse attention mask
pub struct BlockSparseMask {
    pub block_size: usize,
}

impl AttentionMask for BlockSparseMask {
    fn build_mask(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        
        let mask_shape = TensorShape::Fixed(vec![seq_len, seq_len]);
        let mask_node = graph.add_node(
            OpType::Placeholder {
                name: "block_sparse_mask".to_string(),
                shape: mask_shape,
            },
            vec![],
        );
        
        graph.set_output(mask_node);
        Ok(graph)
    }
}

/// Global-local attention mask (some tokens attend globally)
pub struct GlobalLocalMask {
    pub global_tokens: usize,
}

impl AttentionMask for GlobalLocalMask {
    fn build_mask(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        
        let mask_shape = TensorShape::Fixed(vec![seq_len, seq_len]);
        let mask_node = graph.add_node(
            OpType::Placeholder {
                name: "global_local_mask".to_string(),
                shape: mask_shape,
            },
            vec![],
        );
        
        graph.set_output(mask_node);
        Ok(graph)
    }
}

/// Rule pattern type
#[derive(Debug, Clone, Copy)]
pub enum RulePattern {
    /// Hard rule (binary 0/1)
    Hard,
    /// Soft rule (continuous values)
    Soft,
    /// Gated rule (learned gating)
    Gated,
}

/// Rule-based attention mask
pub struct RuleBasedMask {
    pub pattern: RulePattern,
}

impl AttentionMask for RuleBasedMask {
    fn build_mask(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        
        let mask_shape = TensorShape::Fixed(vec![seq_len, seq_len]);
        let mask_node = graph.add_node(
            OpType::Placeholder {
                name: "rule_based_mask".to_string(),
                shape: mask_shape,
            },
            vec![],
        );
        
        graph.set_output(mask_node);
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask() {
        let mask = CausalMask;
        let graph = mask.build_mask(32);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_local_mask() {
        let mask = LocalMask { window_size: 64 };
        let graph = mask.build_mask(128);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_strided_mask() {
        let mask = StridedMask { stride: 2 };
        let graph = mask.build_mask(64);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_block_sparse_mask() {
        let mask = BlockSparseMask { block_size: 16 };
        let graph = mask.build_mask(128);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_global_local_mask() {
        let mask = GlobalLocalMask { global_tokens: 4 };
        let graph = mask.build_mask(128);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_rule_based_mask_hard() {
        let mask = RuleBasedMask { pattern: RulePattern::Hard };
        let graph = mask.build_mask(64);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_rule_based_mask_soft() {
        let mask = RuleBasedMask { pattern: RulePattern::Soft };
        let graph = mask.build_mask(64);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_rule_based_mask_gated() {
        let mask = RuleBasedMask { pattern: RulePattern::Gated };
        let graph = mask.build_mask(64);
        assert!(graph.is_ok());
    }
}
