//! Complete transformer encoder and decoder layers.
//!
//! This module implements full transformer layers that combine:
//! - Multi-head attention
//! - Feed-forward networks
//! - Layer normalization
//! - Residual connections
//!
//! ## Transformer Encoder Layer
//!
//! ```text
//! x' = LayerNorm(x + MultiHeadAttention(x, x, x))
//! output = LayerNorm(x' + FFN(x'))
//! ```
//!
//! ## Transformer Decoder Layer
//!
//! ```text
//! x' = LayerNorm(x + MaskedMultiHeadAttention(x, x, x))
//! x'' = LayerNorm(x' + CrossAttention(x', enc_output, enc_output))
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

/// Configuration for a complete transformer encoder layer
#[derive(Clone, Debug)]
pub struct EncoderLayerConfig {
    /// Attention configuration
    pub attention: AttentionConfig,
    /// Feed-forward configuration
    pub feed_forward: FeedForwardConfig,
    /// Layer normalization configuration
    pub layer_norm: LayerNormConfig,
    /// Whether to use pre-layer normalization (vs post)
    pub pre_norm: bool,
}

impl EncoderLayerConfig {
    /// Create a new encoder layer configuration
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
pub struct EncoderLayer {
    /// Configuration
    pub config: EncoderLayerConfig,
    /// Multi-head attention
    pub attention: MultiHeadAttention,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// First layer normalization
    pub norm1: LayerNorm,
    /// Second layer normalization
    pub norm2: LayerNorm,
}

impl EncoderLayer {
    /// Create a new encoder layer
    pub fn new(config: EncoderLayerConfig) -> Result<Self> {
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
    pub fn build_encoder_layer_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        let input_tensor = 0;

        if self.config.pre_norm {
            // Pre-LN: LN(x) -> Attention -> Add -> LN -> FFN -> Add
            self.build_pre_norm_encoder(graph, input_tensor)
        } else {
            // Post-LN: Attention -> Add -> LN -> FFN -> Add -> LN
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
        // Replace input references with normed1
        let q_tensor = graph.add_tensor("encoder_Q");
        let k_tensor = graph.add_tensor("encoder_K");
        let v_tensor = graph.add_tensor("encoder_V");

        // Create copies for Q, K, V
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
        let _normed1 = normed1_outputs[0];

        // Step 4: Feed-forward network
        let ffn_outputs = self.ffn.build_ffn_graph(graph)?;
        let ffn_output = ffn_outputs[0];

        // Step 5: Second residual connection
        let residual2 = graph.add_tensor("encoder_residual2");
        let res2_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", _normed1, ffn_output, residual2);
        graph.add_node(res2_node)?;

        // Step 6: Second layer norm
        let normed2_outputs = self.norm2.build_layernorm_graph(graph)?;
        let output = normed2_outputs[0];

        Ok(vec![output])
    }
}

/// Configuration for a complete transformer decoder layer
#[derive(Clone, Debug)]
pub struct DecoderLayerConfig {
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

impl DecoderLayerConfig {
    /// Create a new decoder layer configuration
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
pub struct DecoderLayer {
    /// Configuration
    pub config: DecoderLayerConfig,
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

impl DecoderLayer {
    /// Create a new decoder layer
    pub fn new(config: DecoderLayerConfig) -> Result<Self> {
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
    pub fn build_decoder_layer_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
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
        _encoder_output: usize, // Used implicitly in cross-attention
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
        _encoder_output: usize, // Used implicitly in cross-attention
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
        let _normed1 = normed1_outputs[0];

        // Step 3: Cross-attention
        let cross_attn_outputs = self.cross_attention.build_mha_graph(graph)?;
        let cross_attn_output = cross_attn_outputs[0];

        // Step 4: Second residual + norm
        let residual2 = graph.add_tensor("decoder_residual2");
        let res2_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", _normed1, cross_attn_output, residual2);
        graph.add_node(res2_node)?;

        let normed2_outputs = self.norm2.build_layernorm_graph(graph)?;
        let _normed2 = normed2_outputs[0];

        // Step 5: Feed-forward network
        let ffn_outputs = self.ffn.build_ffn_graph(graph)?;
        let ffn_output = ffn_outputs[0];

        // Step 6: Third residual + norm
        let residual3 = graph.add_tensor("decoder_residual3");
        let res3_node =
            tensorlogic_ir::EinsumNode::elem_binary("add", _normed2, ffn_output, residual3);
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
    fn test_encoder_layer_config_creation() {
        let config = EncoderLayerConfig::new(512, 8, 2048).unwrap();
        assert_eq!(config.attention.d_model, 512);
        assert_eq!(config.attention.n_heads, 8);
        assert_eq!(config.feed_forward.d_ff, 2048);
        assert!(config.pre_norm);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_encoder_layer_config_with_dropout() {
        let config = EncoderLayerConfig::new(512, 8, 2048)
            .unwrap()
            .with_dropout(0.1);
        assert!((config.attention.dropout - 0.1).abs() < 1e-10);
        assert!((config.feed_forward.dropout - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_encoder_layer_creation() {
        let config = EncoderLayerConfig::new(512, 8, 2048).unwrap();
        let layer = EncoderLayer::new(config).unwrap();
        assert_eq!(layer.config.attention.d_model, 512);
    }

    #[test]
    fn test_encoder_layer_graph_building() {
        let config = EncoderLayerConfig::new(512, 8, 2048).unwrap();
        let layer = EncoderLayer::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");

        let outputs = layer.build_encoder_layer_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_decoder_layer_config_creation() {
        let config = DecoderLayerConfig::new(512, 8, 2048).unwrap();
        assert_eq!(config.self_attention.d_model, 512);
        assert_eq!(config.cross_attention.d_model, 512);
        assert!(config.self_attention.causal);
        assert!(!config.cross_attention.causal);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_decoder_layer_creation() {
        let config = DecoderLayerConfig::new(512, 8, 2048).unwrap();
        let layer = DecoderLayer::new(config).unwrap();
        assert_eq!(layer.config.self_attention.d_model, 512);
    }

    #[test]
    fn test_decoder_layer_graph_building() {
        let config = DecoderLayerConfig::new(512, 8, 2048).unwrap();
        let layer = DecoderLayer::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("decoder_input");
        graph.add_tensor("encoder_output");

        let outputs = layer.build_decoder_layer_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }
}
