//! Transformer decoder layers.
//!
//! This module implements transformer decoder layers that combine:
//! - Masked multi-head self-attention
//! - Cross-attention to encoder outputs
//! - Feed-forward networks
//! - Layer normalization
//! - Residual connections
//!
//! ## Transformer Decoder Layer
//!
//! Pre-normalization variant:
//! ```text
//! x' = x + MaskedSelfAttention(LayerNorm(x))
//! x'' = x' + CrossAttention(LayerNorm(x'), encoder_output)
//! output = x'' + FFN(LayerNorm(x''))
//! ```
//!
//! Post-normalization variant:
//! ```text
//! x' = LayerNorm(x + MaskedSelfAttention(x))
//! x'' = LayerNorm(x' + CrossAttention(x', encoder_output))
//! output = LayerNorm(x'' + FFN(x''))
//! ```

use tensorlogic_ir::EinsumGraph;

use crate::{
    attention::MultiHeadAttention,
    config::{AttentionConfig, FeedForwardConfig},
    error::Result,
    ffn::FeedForward,
    normalization::{LayerNorm, LayerNormConfig},
};

/// Configuration for transformer decoder layer
#[derive(Clone, Debug)]
pub struct DecoderConfig {
    /// Self-attention configuration (with causal masking)
    pub self_attention: AttentionConfig,
    /// Cross-attention configuration
    pub cross_attention: AttentionConfig,
    /// Feed-forward configuration
    pub feed_forward: FeedForwardConfig,
    /// Layer normalization configuration
    pub layer_norm: LayerNormConfig,
    /// Whether to use pre-layer normalization
    pub pre_norm: bool,
}

impl DecoderConfig {
    /// Create a new decoder configuration
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            self_attention: AttentionConfig::new(d_model, n_heads)?.with_causal(true),
            cross_attention: AttentionConfig::new(d_model, n_heads)?,
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

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.self_attention = self.self_attention.with_dropout(dropout);
        self.cross_attention = self.cross_attention.with_dropout(dropout);
        self.feed_forward = self.feed_forward.with_dropout(dropout);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.self_attention.validate()?;
        self.cross_attention.validate()?;
        self.feed_forward.validate()?;
        self.layer_norm.validate()?;

        // Verify causal masking is enabled for self-attention
        if !self.self_attention.causal {
            return Err(crate::error::TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "Decoder self-attention must use causal masking".to_string(),
            });
        }

        // Check dimension consistency
        if self.self_attention.d_model != self.cross_attention.d_model
            || self.self_attention.d_model != self.feed_forward.d_model
            || self.self_attention.d_model != self.layer_norm.normalized_shape
        {
            return Err(crate::error::TrustformerError::InvalidDimension {
                expected: self.self_attention.d_model,
                got: 0,
                context: "d_model mismatch between components".to_string(),
            });
        }

        Ok(())
    }
}

/// Transformer decoder layer
#[derive(Clone, Debug)]
pub struct Decoder {
    /// Configuration
    pub config: DecoderConfig,
    /// Masked self-attention
    pub self_attention: MultiHeadAttention,
    /// Cross-attention
    pub cross_attention: MultiHeadAttention,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// First layer normalization (self-attention)
    pub norm1: LayerNorm,
    /// Second layer normalization (cross-attention)
    pub norm2: LayerNorm,
    /// Third layer normalization (FFN)
    pub norm3: LayerNorm,
}

impl Decoder {
    /// Create a new decoder layer
    pub fn new(config: DecoderConfig) -> Result<Self> {
        config.validate()?;

        let self_attention = MultiHeadAttention::new(config.self_attention.clone())?;
        let cross_attention = MultiHeadAttention::new(config.cross_attention.clone())?;
        let ffn = FeedForward::new(config.feed_forward.clone())?;
        let norm1 = LayerNorm::new(config.layer_norm.clone())?;
        let norm2 = LayerNorm::new(config.layer_norm.clone())?;
        let norm3 = LayerNorm::new(config.layer_norm.clone())?;

        Ok(Self {
            config,
            self_attention,
            cross_attention,
            ffn,
            norm1,
            norm2,
            norm3,
        })
    }

    /// Build einsum graph for decoder layer
    ///
    /// Input tensors:
    /// - 0: x (decoder input) [batch, tgt_len, d_model]
    /// - 1: encoder_output [batch, src_len, d_model]
    /// - 2-N: weight matrices and parameters
    ///
    /// Output tensors:
    /// - output: [batch, tgt_len, d_model]
    pub fn build_decoder_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        let decoder_input = 0;
        let encoder_output = 1;

        if self.config.pre_norm {
            self.build_pre_norm_decoder(graph, decoder_input, encoder_output)
        } else {
            self.build_post_norm_decoder(graph, decoder_input, encoder_output)
        }
    }

    fn build_pre_norm_decoder(
        &self,
        graph: &mut EinsumGraph,
        decoder_input: usize,
        _encoder_output: usize,
    ) -> Result<Vec<usize>> {
        // Step 1: First layer norm
        let normed1_outputs = self.norm1.build_layernorm_graph(graph)?;
        let _normed1 = normed1_outputs[0];

        // Step 2: Masked self-attention
        let self_attn_outputs = self.self_attention.build_mha_graph(graph)?;
        let self_attn_output = self_attn_outputs[0];

        // Step 3: First residual
        let residual1 = graph.add_tensor("decoder_residual1");
        let res1_node = tensorlogic_ir::EinsumNode::elem_binary(
            "add",
            decoder_input,
            self_attn_output,
            residual1,
        );
        graph.add_node(res1_node)?;

        // Step 4: Second layer norm
        let normed2_outputs = self.norm2.build_layernorm_graph(graph)?;
        let _normed2 = normed2_outputs[0];

        // Step 5: Cross-attention (Q from decoder, K,V from encoder)
        let cross_attn_outputs = self.cross_attention.build_mha_graph(graph)?;
        let cross_attn_output = cross_attn_outputs[0];

        // Step 6: Second residual
        let residual2 = graph.add_tensor("decoder_residual2");
        let res2_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", residual1, cross_attn_output, residual2);
        graph.add_node(res2_node)?;

        // Step 7: Third layer norm
        let normed3_outputs = self.norm3.build_layernorm_graph(graph)?;
        let _normed3 = normed3_outputs[0];

        // Step 8: Feed-forward network
        let ffn_outputs = self.ffn.build_ffn_graph(graph)?;
        let ffn_output = ffn_outputs[0];

        // Step 9: Third residual
        let output = graph.add_tensor("decoder_output");
        let res3_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", residual2, ffn_output, output);
        graph.add_node(res3_node)?;

        Ok(vec![output])
    }

    fn build_post_norm_decoder(
        &self,
        graph: &mut EinsumGraph,
        decoder_input: usize,
        _encoder_output: usize,
    ) -> Result<Vec<usize>> {
        // Step 1: Masked self-attention
        let self_attn_outputs = self.self_attention.build_mha_graph(graph)?;
        let self_attn_output = self_attn_outputs[0];

        // Step 2: First residual + norm
        let residual1 = graph.add_tensor("decoder_residual1");
        let res1_node = tensorlogic_ir::EinsumNode::elem_binary(
            "add",
            decoder_input,
            self_attn_output,
            residual1,
        );
        graph.add_node(res1_node)?;

        let normed1_outputs = self.norm1.build_layernorm_graph(graph)?;
        let normed1 = normed1_outputs[0];

        // Step 3: Cross-attention
        let cross_attn_outputs = self.cross_attention.build_mha_graph(graph)?;
        let cross_attn_output = cross_attn_outputs[0];

        // Step 4: Second residual + norm
        let residual2 = graph.add_tensor("decoder_residual2");
        let res2_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", normed1, cross_attn_output, residual2);
        graph.add_node(res2_node)?;

        let normed2_outputs = self.norm2.build_layernorm_graph(graph)?;
        let normed2 = normed2_outputs[0];

        // Step 5: Feed-forward network
        let ffn_outputs = self.ffn.build_ffn_graph(graph)?;
        let ffn_output = ffn_outputs[0];

        // Step 6: Third residual + norm
        let residual3 = graph.add_tensor("decoder_residual3");
        let res3_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", normed2, ffn_output, residual3);
        graph.add_node(res3_node)?;

        let normed3_outputs = self.norm3.build_layernorm_graph(graph)?;
        let output = normed3_outputs[0];

        Ok(vec![output])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_config_creation() {
        let config = DecoderConfig::new(512, 8, 2048).unwrap();
        assert_eq!(config.self_attention.d_model, 512);
        assert_eq!(config.cross_attention.d_model, 512);
        assert!(config.self_attention.causal);
        assert!(!config.cross_attention.causal);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_decoder_config_with_dropout() {
        let config = DecoderConfig::new(512, 8, 2048).unwrap().with_dropout(0.1);
        assert!((config.self_attention.dropout - 0.1).abs() < 1e-10);
        assert!((config.cross_attention.dropout - 0.1).abs() < 1e-10);
        assert!((config.feed_forward.dropout - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_decoder_config_pre_norm() {
        let config = DecoderConfig::new(512, 8, 2048)
            .unwrap()
            .with_pre_norm(false);
        assert!(!config.pre_norm);
    }

    #[test]
    fn test_decoder_creation() {
        let config = DecoderConfig::new(512, 8, 2048).unwrap();
        let decoder = Decoder::new(config).unwrap();
        assert_eq!(decoder.config.self_attention.d_model, 512);
    }

    #[test]
    fn test_decoder_graph_building_pre_norm() {
        let config = DecoderConfig::new(512, 8, 2048).unwrap();
        let decoder = Decoder::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("decoder_input");
        graph.add_tensor("encoder_output");

        let outputs = decoder.build_decoder_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_decoder_graph_building_post_norm() {
        let config = DecoderConfig::new(512, 8, 2048)
            .unwrap()
            .with_pre_norm(false);
        let decoder = Decoder::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("decoder_input");
        graph.add_tensor("encoder_output");

        let outputs = decoder.build_decoder_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_decoder_config_validation() {
        let config = DecoderConfig::new(512, 8, 2048).unwrap();
        assert!(config.validate().is_ok());

        // Invalid head count
        let result = DecoderConfig::new(512, 7, 2048);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_requires_causal_masking() {
        let config = DecoderConfig::new(512, 8, 2048).unwrap();
        assert!(config.self_attention.causal);
        assert!(!config.cross_attention.causal);
    }
}
