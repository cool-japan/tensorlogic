//! Transformer encoder layers.
//!
//! This module implements transformer encoder layers that combine:
//! - Multi-head self-attention
//! - Feed-forward networks
//! - Layer normalization
//! - Residual connections
//!
//! ## Transformer Encoder Layer
//!
//! Pre-normalization variant:
//! ```text
//! x' = x + MultiHeadAttention(LayerNorm(x))
//! output = x' + FFN(LayerNorm(x'))
//! ```
//!
//! Post-normalization variant:
//! ```text
//! x' = LayerNorm(x + MultiHeadAttention(x))
//! output = LayerNorm(x' + FFN(x'))
//! ```

use tensorlogic_ir::EinsumGraph;

use crate::{
    attention::MultiHeadAttention,
    config::{AttentionConfig, FeedForwardConfig},
    error::Result,
    ffn::FeedForward,
    normalization::{LayerNorm, LayerNormConfig},
};

/// Configuration for transformer encoder layer
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    /// Attention configuration
    pub attention: AttentionConfig,
    /// Feed-forward configuration
    pub feed_forward: FeedForwardConfig,
    /// Layer normalization configuration
    pub layer_norm: LayerNormConfig,
    /// Whether to use pre-layer normalization (vs post)
    pub pre_norm: bool,
}

impl EncoderConfig {
    /// Create a new encoder configuration
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            attention: AttentionConfig::new(d_model, n_heads)?,
            feed_forward: FeedForwardConfig::new(d_model, d_ff),
            layer_norm: LayerNormConfig::new(d_model),
            pre_norm: true,
        })
    }

    /// Set pre-normalization vs post-normalization
    pub fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.pre_norm = pre_norm;
        self
    }

    /// Set causal masking
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.attention = self.attention.with_causal(causal);
        self
    }

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.attention = self.attention.with_dropout(dropout);
        self.feed_forward = self.feed_forward.with_dropout(dropout);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.attention.validate()?;
        self.feed_forward.validate()?;
        self.layer_norm.validate()?;

        // Check dimension consistency
        if self.attention.d_model != self.feed_forward.d_model {
            return Err(crate::error::TrustformerError::InvalidDimension {
                expected: self.attention.d_model,
                got: self.feed_forward.d_model,
                context: "d_model mismatch between attention and FFN".to_string(),
            });
        }

        if self.attention.d_model != self.layer_norm.normalized_shape {
            return Err(crate::error::TrustformerError::InvalidDimension {
                expected: self.attention.d_model,
                got: self.layer_norm.normalized_shape,
                context: "d_model mismatch with layer norm".to_string(),
            });
        }

        Ok(())
    }
}

/// Transformer encoder layer
#[derive(Clone, Debug)]
pub struct Encoder {
    /// Configuration
    pub config: EncoderConfig,
    /// Multi-head attention
    pub attention: MultiHeadAttention,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// First layer normalization
    pub norm1: LayerNorm,
    /// Second layer normalization
    pub norm2: LayerNorm,
}

impl Encoder {
    /// Create a new encoder layer
    pub fn new(config: EncoderConfig) -> Result<Self> {
        config.validate()?;

        let attention = MultiHeadAttention::new(config.attention.clone())?;
        let ffn = FeedForward::new(config.feed_forward.clone())?;
        let norm1 = LayerNorm::new(config.layer_norm.clone())?;
        let norm2 = LayerNorm::new(config.layer_norm.clone())?;

        Ok(Self {
            config,
            attention,
            ffn,
            norm1,
            norm2,
        })
    }

    /// Build einsum graph for encoder layer
    ///
    /// Input tensors:
    /// - 0: x (input) [batch, seq_len, d_model]
    /// - 1-N: weight matrices and parameters for attention, FFN, and layer norms
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_encoder_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        let input_tensor = 0;

        if self.config.pre_norm {
            self.build_pre_norm_encoder(graph, input_tensor)
        } else {
            self.build_post_norm_encoder(graph, input_tensor)
        }
    }

    fn build_pre_norm_encoder(
        &self,
        graph: &mut EinsumGraph,
        input_tensor: usize,
    ) -> Result<Vec<usize>> {
        // Step 1: First layer norm
        let normed1_outputs = self.norm1.build_layernorm_graph(graph)?;
        let normed1 = normed1_outputs[0];

        // Step 2: Multi-head attention (Q, K, V all from normed input)
        // Create copies for Q, K, V
        let q_tensor = graph.add_tensor("encoder_Q");
        let k_tensor = graph.add_tensor("encoder_K");
        let v_tensor = graph.add_tensor("encoder_V");

        let _q_node = tensorlogic_ir::EinsumNode::elem_unary("identity", normed1, q_tensor);
        let _k_node = tensorlogic_ir::EinsumNode::elem_unary("identity", normed1, k_tensor);
        let _v_node = tensorlogic_ir::EinsumNode::elem_unary("identity", normed1, v_tensor);

        let attn_outputs = self.attention.build_mha_graph(graph)?;
        let attn_output = attn_outputs[0];

        // Step 3: Residual connection: x + attention_output
        let residual1 = graph.add_tensor("encoder_residual1");
        let res1_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", input_tensor, attn_output, residual1);
        graph.add_node(res1_node)?;

        // Step 4: Second layer norm
        let _normed2_outputs = self.norm2.build_layernorm_graph(graph)?;

        // Step 5: Feed-forward network
        let ffn_outputs = self.ffn.build_ffn_graph(graph)?;
        let ffn_output = ffn_outputs[0];

        // Step 6: Second residual connection: residual1 + ffn_output
        let output = graph.add_tensor("encoder_output");
        let res2_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", residual1, ffn_output, output);
        graph.add_node(res2_node)?;

        Ok(vec![output])
    }

    fn build_post_norm_encoder(
        &self,
        graph: &mut EinsumGraph,
        input_tensor: usize,
    ) -> Result<Vec<usize>> {
        // Step 1: Multi-head attention
        let attn_outputs = self.attention.build_mha_graph(graph)?;
        let attn_output = attn_outputs[0];

        // Step 2: Residual connection
        let residual1 = graph.add_tensor("encoder_residual1");
        let res1_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", input_tensor, attn_output, residual1);
        graph.add_node(res1_node)?;

        // Step 3: First layer norm
        let normed1_outputs = self.norm1.build_layernorm_graph(graph)?;
        let normed1 = normed1_outputs[0];

        // Step 4: Feed-forward network
        let ffn_outputs = self.ffn.build_ffn_graph(graph)?;
        let ffn_output = ffn_outputs[0];

        // Step 5: Second residual connection
        let residual2 = graph.add_tensor("encoder_residual2");
        let res2_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", normed1, ffn_output, residual2);
        graph.add_node(res2_node)?;

        // Step 6: Second layer norm
        let normed2_outputs = self.norm2.build_layernorm_graph(graph)?;
        let output = normed2_outputs[0];

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_creation() {
        let config = EncoderConfig::new(512, 8, 2048).unwrap();
        assert_eq!(config.attention.d_model, 512);
        assert_eq!(config.attention.n_heads, 8);
        assert_eq!(config.feed_forward.d_ff, 2048);
        assert!(config.pre_norm);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_encoder_config_with_dropout() {
        let config = EncoderConfig::new(512, 8, 2048).unwrap().with_dropout(0.1);
        assert!((config.attention.dropout - 0.1).abs() < 1e-10);
        assert!((config.feed_forward.dropout - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_encoder_config_pre_norm() {
        let config = EncoderConfig::new(512, 8, 2048)
            .unwrap()
            .with_pre_norm(false);
        assert!(!config.pre_norm);
    }

    #[test]
    fn test_encoder_creation() {
        let config = EncoderConfig::new(512, 8, 2048).unwrap();
        let encoder = Encoder::new(config).unwrap();
        assert_eq!(encoder.config.attention.d_model, 512);
    }

    #[test]
    fn test_encoder_graph_building_pre_norm() {
        let config = EncoderConfig::new(512, 8, 2048).unwrap();
        let encoder = Encoder::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");

        let outputs = encoder.build_encoder_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_encoder_graph_building_post_norm() {
        let config = EncoderConfig::new(512, 8, 2048)
            .unwrap()
            .with_pre_norm(false);
        let encoder = Encoder::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");

        let outputs = encoder.build_encoder_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_encoder_config_validation() {
        let config = EncoderConfig::new(512, 8, 2048).unwrap();
        assert!(config.validate().is_ok());

        // Invalid head count
        let result = EncoderConfig::new(512, 7, 2048);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_with_causal() {
        let config = EncoderConfig::new(512, 8, 2048).unwrap().with_causal(true);
        assert!(config.attention.causal);
    }
}
