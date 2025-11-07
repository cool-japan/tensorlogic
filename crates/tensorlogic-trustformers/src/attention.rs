//! Self-attention and multi-head attention as einsum operations.
//!
//! This module implements transformer attention mechanisms using einsum notation,
//! which can be compiled to TensorLogic IR and executed on various backends.
//!
//! ## Self-Attention Formula
//!
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / √d_k) V
//! ```
//!
//! ## Einsum Breakdown
//!
//! 1. Query-Key scores: `einsum("bqd,bkd->bqk", Q, K)`
//! 2. Scaled scores: `scores / sqrt(d_k)`
//! 3. Softmax: `softmax(scores, axis=-1)`
//! 4. Attention-Value: `einsum("bqk,bkv->bqv", attn, V)`
//!
//! Where:
//! - `b` = batch dimension
//! - `q` = query sequence length
//! - `k` = key sequence length
//! - `d` = model dimension
//! - `v` = value dimension (usually same as d)

use tensorlogic_ir::{EinsumGraph, EinsumNode};

use crate::config::AttentionConfig;
use crate::error::Result;

/// Self-attention component
#[derive(Clone, Debug)]
pub struct SelfAttention {
    /// Configuration
    pub config: AttentionConfig,
}

impl SelfAttention {
    /// Create a new self-attention component
    pub fn new(config: AttentionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build einsum graph for self-attention forward pass
    ///
    /// Input tensors:
    /// - 0: Q (query) [batch, seq_len, d_model]
    /// - 1: K (key) [batch, seq_len, d_model]
    /// - 2: V (value) [batch, seq_len, d_model]
    ///
    /// Optional:
    /// - 3: attention_mask [batch, seq_len, seq_len]
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_attention_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Compute attention scores: Q @ K.T
        // einsum("bqd,bkd->bqk", Q, K)
        let scores_tensor = graph.add_tensor("attn_scores");
        let scores_node = EinsumNode::new("bqd,bkd->bqk", vec![0, 1], vec![scores_tensor]);
        graph.add_node(scores_node)?;

        // Step 2: Scale scores by sqrt(d_k)
        let scale_factor = (self.config.d_k as f64).sqrt();
        let scale_tensor = graph.add_tensor("scale_factor");
        let scaled_tensor = graph.add_tensor("scaled_scores");
        let scale_node = EinsumNode::elem_binary(
            format!("div_scalar_{}", scale_factor),
            scores_tensor,
            scale_tensor,
            scaled_tensor,
        );
        graph.add_node(scale_node)?;

        // Step 3: Apply softmax along key dimension
        let softmax_tensor = graph.add_tensor("attention_weights");
        let softmax_node = EinsumNode::elem_unary("softmax_k", scaled_tensor, softmax_tensor);
        graph.add_node(softmax_node)?;

        // Step 4: Apply attention to values: attn @ V
        // einsum("bqk,bkv->bqv", attention_weights, V)
        let output_tensor = graph.add_tensor("attn_output");
        let output_node =
            EinsumNode::new("bqk,bkv->bqv", vec![softmax_tensor, 2], vec![output_tensor]);
        graph.add_node(output_node)?;

        Ok(vec![output_tensor])
    }

    /// Get scaling factor for attention scores
    pub fn get_scale_factor(&self) -> f64 {
        (self.config.d_k as f64).sqrt()
    }
}

/// Multi-head attention component
///
/// Splits the model dimension into multiple heads, applies parallel attention,
/// and concatenates the results.
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    /// Configuration
    pub config: AttentionConfig,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention component
    pub fn new(config: AttentionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build einsum graph for multi-head attention forward pass
    ///
    /// Input tensors:
    /// - 0: Q (query) [batch, seq_len, d_model]
    /// - 1: K (key) [batch, seq_len, d_model]
    /// - 2: V (value) [batch, seq_len, d_model]
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_mha_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        let _h = self.config.n_heads; // Number of heads (used in reshape spec)
        let d_k = self.config.d_k;

        // Step 1: Reshape Q, K, V to split heads
        // [batch, seq, d_model] -> [batch, seq, n_heads, d_k]
        let q_split = graph.add_tensor("q_split");
        let k_split = graph.add_tensor("k_split");
        let v_split = graph.add_tensor("v_split");

        let reshape_spec = format!("bsd->bsh{}", d_k);

        let q_reshape = EinsumNode::new(&reshape_spec, vec![0], vec![q_split]);
        graph.add_node(q_reshape)?;

        let k_reshape = EinsumNode::new(&reshape_spec, vec![1], vec![k_split]);
        graph.add_node(k_reshape)?;

        let v_reshape = EinsumNode::new(&reshape_spec, vec![2], vec![v_split]);
        graph.add_node(v_reshape)?;

        // Step 2: Transpose to [batch, n_heads, seq, d_k]
        let q_transposed = graph.add_tensor("q_transposed");
        let k_transposed = graph.add_tensor("k_transposed");
        let v_transposed = graph.add_tensor("v_transposed");

        let transpose_node_q = EinsumNode::new("bshd->bhsd", vec![q_split], vec![q_transposed]);
        graph.add_node(transpose_node_q)?;

        let transpose_node_k = EinsumNode::new("bshd->bhsd", vec![k_split], vec![k_transposed]);
        graph.add_node(transpose_node_k)?;

        let transpose_node_v = EinsumNode::new("bshd->bhsd", vec![v_split], vec![v_transposed]);
        graph.add_node(transpose_node_v)?;

        // Step 3: Compute attention scores per head
        // einsum("bhqd,bhkd->bhqk", Q, K)
        let scores_tensor = graph.add_tensor("mha_scores");
        let scores_node = EinsumNode::new(
            "bhqd,bhkd->bhqk",
            vec![q_transposed, k_transposed],
            vec![scores_tensor],
        );
        graph.add_node(scores_node)?;

        // Step 4: Scale scores
        let scale_factor = (d_k as f64).sqrt();
        let scale_tensor = graph.add_tensor("mha_scale");
        let scaled_tensor = graph.add_tensor("mha_scaled_scores");
        let scale_node = EinsumNode::elem_binary(
            format!("div_scalar_{}", scale_factor),
            scores_tensor,
            scale_tensor,
            scaled_tensor,
        );
        graph.add_node(scale_node)?;

        // Step 5: Softmax
        let softmax_tensor = graph.add_tensor("mha_attention_weights");
        let softmax_node = EinsumNode::elem_unary("softmax_k", scaled_tensor, softmax_tensor);
        graph.add_node(softmax_node)?;

        // Step 6: Apply attention to values
        // einsum("bhqk,bhkv->bhqv", attn, V)
        let attn_output = graph.add_tensor("mha_attn_output");
        let attn_node = EinsumNode::new(
            "bhqk,bhkv->bhqv",
            vec![softmax_tensor, v_transposed],
            vec![attn_output],
        );
        graph.add_node(attn_node)?;

        // Step 7: Transpose back and concatenate heads
        // [batch, n_heads, seq, d_k] -> [batch, seq, n_heads, d_k]
        let transposed_back = graph.add_tensor("mha_transposed_back");
        let transpose_back_node =
            EinsumNode::new("bhsd->bshd", vec![attn_output], vec![transposed_back]);
        graph.add_node(transpose_back_node)?;

        // Step 8: Reshape to [batch, seq, d_model]
        let output_tensor = graph.add_tensor("mha_output");
        let reshape_back_spec = format!("bsh{}-:bsd", d_k);
        let reshape_back = EinsumNode::new(
            &reshape_back_spec,
            vec![transposed_back],
            vec![output_tensor],
        );
        graph.add_node(reshape_back)?;

        Ok(vec![output_tensor])
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.config.n_heads
    }

    /// Get dimension per head
    pub fn head_dim(&self) -> usize {
        self.config.d_k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_attention_creation() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let attn = SelfAttention::new(config).unwrap();
        assert_eq!(attn.config.d_model, 512);
        assert_eq!(attn.config.n_heads, 8);
        assert_eq!(attn.config.d_k, 64);
    }

    #[test]
    fn test_self_attention_scale_factor() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let attn = SelfAttention::new(config).unwrap();
        let scale = attn.get_scale_factor();
        assert!((scale - 8.0).abs() < 1e-10); // sqrt(64) = 8.0
    }

    #[test]
    fn test_self_attention_graph_building() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let attn = SelfAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        // Add input tensors
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = attn.build_attention_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_multi_head_attention_creation() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let mha = MultiHeadAttention::new(config).unwrap();
        assert_eq!(mha.num_heads(), 8);
        assert_eq!(mha.head_dim(), 64);
    }

    #[test]
    fn test_multi_head_attention_graph_building() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let mha = MultiHeadAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        // Add input tensors
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = mha.build_mha_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_attention_invalid_config() {
        let result = AttentionConfig::new(512, 7); // 512 not divisible by 7
        assert!(result.is_err());
    }
}
