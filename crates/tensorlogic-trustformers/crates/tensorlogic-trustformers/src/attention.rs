//! Self-attention and multi-head attention implementations.

use crate::config::AttentionConfig;
use crate::error::{TrustformersError, TrustformersResult};
use tensorlogic_ir::{EinsumGraph, OpType, TensorShape};

/// Multi-head attention builder
pub struct MultiHeadAttention {
    config: AttentionConfig,
}

impl MultiHeadAttention {
    /// Create a new MultiHeadAttention with the given configuration
    pub fn new(config: AttentionConfig) -> TrustformersResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build the multi-head attention einsum graph
    ///
    /// Multi-head attention computes:
    /// 1. Q = XW_Q, K = XW_K, V = XW_V (linear projections)
    /// 2. Split into multiple heads
    /// 3. Scaled dot-product attention per head
    /// 4. Concatenate heads and apply output projection
    pub fn build(&self) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_size / num_heads;

        // Input: [batch, seq_len, hidden_size]
        let input_shape = TensorShape::Dynamic;
        let input_node = graph.add_node(
            OpType::Placeholder {
                name: "input".to_string(),
                shape: input_shape.clone(),
            },
            vec![],
        );

        // QKV projection weights: [hidden_size, hidden_size] each
        let wq_shape = TensorShape::Fixed(vec![hidden_size, hidden_size]);
        let wq_node = graph.add_node(
            OpType::Placeholder {
                name: "w_query".to_string(),
                shape: wq_shape,
            },
            vec![],
        );

        let wk_shape = TensorShape::Fixed(vec![hidden_size, hidden_size]);
        let wk_node = graph.add_node(
            OpType::Placeholder {
                name: "w_key".to_string(),
                shape: wk_shape,
            },
            vec![],
        );

        let wv_shape = TensorShape::Fixed(vec![hidden_size, hidden_size]);
        let wv_node = graph.add_node(
            OpType::Placeholder {
                name: "w_value".to_string(),
                shape: wv_shape,
            },
            vec![],
        );

        // Project to Q, K, V: [batch, seq_len, hidden_size]
        let q_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, wq_node],
        );

        let k_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, wk_node],
        );

        let v_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, wv_node],
        );

        // Reshape for multi-head: [batch, seq_len, num_heads, head_dim]
        // Represented symbolically - actual runtime handles the reshape
        let q_heads_node = graph.add_node(
            OpType::ElemUnary {
                op: "reshape_heads".to_string(),
            },
            vec![q_node],
        );

        let k_heads_node = graph.add_node(
            OpType::ElemUnary {
                op: "reshape_heads".to_string(),
            },
            vec![k_node],
        );

        let v_heads_node = graph.add_node(
            OpType::ElemUnary {
                op: "reshape_heads".to_string(),
            },
            vec![v_node],
        );

        // Compute attention scores: Q·K^T / sqrt(d_k)
        // [batch, num_heads, seq_q, seq_k]
        let scores_node = graph.add_node(
            OpType::Einsum {
                spec: "...hqd,...hkd->...hqk".to_string(),
            },
            vec![q_heads_node, k_heads_node],
        );

        // Scale by 1/sqrt(head_dim)
        let scale_value = self.config.scale.unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());
        let scale_shape = TensorShape::Fixed(vec![1]);
        let scale_node = graph.add_node(
            OpType::Placeholder {
                name: "scale".to_string(),
                shape: scale_shape,
            },
            vec![],
        );

        let scaled_scores_node = graph.add_node(
            OpType::ElemBinary {
                op: "mul".to_string(),
            },
            vec![scores_node, scale_node],
        );

        // Apply attention mask if needed (placeholder)
        let mask_shape = TensorShape::Dynamic;
        let mask_node = graph.add_node(
            OpType::Placeholder {
                name: "attention_mask".to_string(),
                shape: mask_shape,
            },
            vec![],
        );

        let masked_scores_node = graph.add_node(
            OpType::ElemBinary {
                op: "add".to_string(),
            },
            vec![scaled_scores_node, mask_node],
        );

        // Softmax over key dimension
        let attention_weights_node = graph.add_node(
            OpType::ElemUnary {
                op: "softmax".to_string(),
            },
            vec![masked_scores_node],
        );

        // Apply attention to values: attention_weights · V
        // [batch, num_heads, seq_q, head_dim]
        let attended_node = graph.add_node(
            OpType::Einsum {
                spec: "...hqk,...hkd->...hqd".to_string(),
            },
            vec![attention_weights_node, v_heads_node],
        );

        // Concatenate heads: [batch, seq_len, hidden_size]
        let concat_node = graph.add_node(
            OpType::ElemUnary {
                op: "concat_heads".to_string(),
            },
            vec![attended_node],
        );

        // Output projection
        let wo_shape = TensorShape::Fixed(vec![hidden_size, hidden_size]);
        let wo_node = graph.add_node(
            OpType::Placeholder {
                name: "w_output".to_string(),
                shape: wo_shape,
            },
            vec![],
        );

        let output_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![concat_node, wo_node],
        );

        graph.set_output(output_node);
        Ok(graph)
    }

    /// Build cross-attention graph (for decoder)
    ///
    /// Q comes from decoder, K and V from encoder
    pub fn build_cross_attention(&self) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_size / num_heads;

        // Query input (from decoder): [batch, tgt_len, hidden_size]
        let q_input_shape = TensorShape::Dynamic;
        let q_input_node = graph.add_node(
            OpType::Placeholder {
                name: "query_input".to_string(),
                shape: q_input_shape,
            },
            vec![],
        );

        // Key/Value input (from encoder): [batch, src_len, hidden_size]
        let kv_input_shape = TensorShape::Dynamic;
        let kv_input_node = graph.add_node(
            OpType::Placeholder {
                name: "kv_input".to_string(),
                shape: kv_input_shape,
            },
            vec![],
        );

        // QKV projection weights
        let wq_shape = TensorShape::Fixed(vec![hidden_size, hidden_size]);
        let wq_node = graph.add_node(
            OpType::Placeholder {
                name: "w_query".to_string(),
                shape: wq_shape,
            },
            vec![],
        );

        let wk_shape = TensorShape::Fixed(vec![hidden_size, hidden_size]);
        let wk_node = graph.add_node(
            OpType::Placeholder {
                name: "w_key".to_string(),
                shape: wk_shape,
            },
            vec![],
        );

        let wv_shape = TensorShape::Fixed(vec![hidden_size, hidden_size]);
        let wv_node = graph.add_node(
            OpType::Placeholder {
                name: "w_value".to_string(),
                shape: wv_shape,
            },
            vec![],
        );

        // Project Q from decoder input
        let q_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![q_input_node, wq_node],
        );

        // Project K, V from encoder input
        let k_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![kv_input_node, wk_node],
        );

        let v_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![kv_input_node, wv_node],
        );

        // Reshape for multi-head
        let q_heads_node = graph.add_node(
            OpType::ElemUnary {
                op: "reshape_heads".to_string(),
            },
            vec![q_node],
        );

        let k_heads_node = graph.add_node(
            OpType::ElemUnary {
                op: "reshape_heads".to_string(),
            },
            vec![k_node],
        );

        let v_heads_node = graph.add_node(
            OpType::ElemUnary {
                op: "reshape_heads".to_string(),
            },
            vec![v_node],
        );

        // Compute attention scores
        let scores_node = graph.add_node(
            OpType::Einsum {
                spec: "...hqd,...hkd->...hqk".to_string(),
            },
            vec![q_heads_node, k_heads_node],
        );

        // Scale
        let scale_shape = TensorShape::Fixed(vec![1]);
        let scale_node = graph.add_node(
            OpType::Placeholder {
                name: "scale".to_string(),
                shape: scale_shape,
            },
            vec![],
        );

        let scaled_scores_node = graph.add_node(
            OpType::ElemBinary {
                op: "mul".to_string(),
            },
            vec![scores_node, scale_node],
        );

        // Softmax
        let attention_weights_node = graph.add_node(
            OpType::ElemUnary {
                op: "softmax".to_string(),
            },
            vec![scaled_scores_node],
        );

        // Apply attention to values
        let attended_node = graph.add_node(
            OpType::Einsum {
                spec: "...hqk,...hkd->...hqd".to_string(),
            },
            vec![attention_weights_node, v_heads_node],
        );

        // Concatenate heads
        let concat_node = graph.add_node(
            OpType::ElemUnary {
                op: "concat_heads".to_string(),
            },
            vec![attended_node],
        );

        // Output projection
        let wo_shape = TensorShape::Fixed(vec![hidden_size, hidden_size]);
        let wo_node = graph.add_node(
            OpType::Placeholder {
                name: "w_output".to_string(),
                shape: wo_shape,
            },
            vec![],
        );

        let output_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![concat_node, wo_node],
        );

        graph.set_output(output_node);
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AttentionPattern;

    #[test]
    fn test_mha_creation() {
        let config = AttentionConfig::default();
        let mha = MultiHeadAttention::new(config);
        assert!(mha.is_ok());
    }

    #[test]
    fn test_mha_build() {
        let config = AttentionConfig {
            hidden_size: 512,
            num_heads: 8,
            attention_dropout: 0.1,
            output_dropout: 0.1,
            pattern: AttentionPattern::Full,
            use_bias: true,
            scale: None,
        };
        let mha = MultiHeadAttention::new(config).unwrap();
        let graph = mha.build();
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        assert!(graph.output().is_some());
    }

    #[test]
    fn test_cross_attention_build() {
        let config = AttentionConfig::default();
        let mha = MultiHeadAttention::new(config).unwrap();
        let graph = mha.build_cross_attention();
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        assert!(graph.output().is_some());
    }

    #[test]
    fn test_different_head_counts() {
        for num_heads in vec![1, 2, 4, 8, 12, 16] {
            let hidden_size = 512;
            assert_eq!(hidden_size % num_heads, 0);
            let config = AttentionConfig {
                hidden_size,
                num_heads,
                ..Default::default()
            };
            let mha = MultiHeadAttention::new(config).unwrap();
            let graph = mha.build();
            assert!(graph.is_ok());
        }
    }

    #[test]
    fn test_custom_scale() {
        let config = AttentionConfig {
            hidden_size: 512,
            num_heads: 8,
            scale: Some(0.125),
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(config).unwrap();
        let graph = mha.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_no_bias() {
        let config = AttentionConfig {
            hidden_size: 512,
            num_heads: 8,
            use_bias: false,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(config).unwrap();
        let graph = mha.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_causal_pattern() {
        let config = AttentionConfig {
            hidden_size: 512,
            num_heads: 8,
            pattern: AttentionPattern::Causal,
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(config).unwrap();
        let graph = mha.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_different_sizes() {
        for (hidden_size, num_heads) in vec![(256, 4), (512, 8), (768, 12), (1024, 16)] {
            let config = AttentionConfig {
                hidden_size,
                num_heads,
                ..Default::default()
            };
            let mha = MultiHeadAttention::new(config).unwrap();
            let graph = mha.build();
            assert!(graph.is_ok());
        }
    }
}
