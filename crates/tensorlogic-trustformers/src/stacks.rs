//! Transformer encoder and decoder stacks.
//!
//! This module implements complete transformer stacks by composing multiple
//! encoder or decoder layers.
//!
//! ## Transformer Encoder Stack
//!
//! ```text
//! x = input + position_encoding
//! for layer in encoder_layers:
//!     x = layer(x)
//! output = final_layer_norm(x)
//! ```
//!
//! ## Transformer Decoder Stack
//!
//! ```text
//! x = target + position_encoding
//! for layer in decoder_layers:
//!     x = layer(x, encoder_output)
//! output = final_layer_norm(x)
//! ```

use tensorlogic_ir::EinsumGraph;

use crate::{
    error::Result,
    layers::{DecoderLayer, DecoderLayerConfig, EncoderLayer, EncoderLayerConfig},
    normalization::{LayerNorm, LayerNormConfig},
    position::{LearnedPositionEncoding, PositionEncodingConfig, SinusoidalPositionEncoding},
};

/// Configuration for transformer encoder stack
#[derive(Clone, Debug)]
pub struct EncoderStackConfig {
    /// Number of encoder layers
    pub num_layers: usize,
    /// Configuration for each encoder layer
    pub layer_config: EncoderLayerConfig,
    /// Position encoding configuration
    pub position_encoding: PositionEncodingConfig,
    /// Whether to apply final layer normalization
    pub final_layer_norm: bool,
}

impl EncoderStackConfig {
    /// Create a new encoder stack configuration
    pub fn new(
        num_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            num_layers,
            layer_config: EncoderLayerConfig::new(d_model, n_heads, d_ff)?,
            position_encoding: PositionEncodingConfig::sinusoidal(d_model, max_seq_len),
            final_layer_norm: true,
        })
    }

    /// Set position encoding type to learned
    pub fn with_learned_position_encoding(mut self) -> Self {
        self.position_encoding = PositionEncodingConfig::learned(
            self.position_encoding.d_model,
            self.position_encoding.max_seq_len,
        );
        self
    }

    /// Set whether to use final layer normalization
    pub fn with_final_layer_norm(mut self, final_layer_norm: bool) -> Self {
        self.final_layer_norm = final_layer_norm;
        self
    }

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.layer_config = self.layer_config.with_dropout(dropout);
        self.position_encoding = self.position_encoding.with_dropout(dropout);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.num_layers == 0 {
            return Err(crate::error::TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "num_layers must be positive".to_string(),
            });
        }

        self.layer_config.validate()?;
        self.position_encoding.validate()?;

        Ok(())
    }
}

/// Transformer encoder stack
#[derive(Clone, Debug)]
pub struct EncoderStack {
    /// Configuration
    pub config: EncoderStackConfig,
    /// Encoder layers
    pub layers: Vec<EncoderLayer>,
    /// Position encoding (if sinusoidal)
    pub position_encoding_sin: Option<SinusoidalPositionEncoding>,
    /// Position encoding (if learned)
    pub position_encoding_learned: Option<LearnedPositionEncoding>,
    /// Final layer normalization
    pub final_norm: Option<LayerNorm>,
}

impl EncoderStack {
    /// Create a new encoder stack
    pub fn new(config: EncoderStackConfig) -> Result<Self> {
        config.validate()?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(EncoderLayer::new(config.layer_config.clone())?);
        }

        let position_encoding_sin = match config.position_encoding.encoding_type {
            crate::position::PositionEncodingType::Sinusoidal { .. } => Some(
                SinusoidalPositionEncoding::new(config.position_encoding.clone())?,
            ),
            _ => None,
        };

        let position_encoding_learned = match config.position_encoding.encoding_type {
            crate::position::PositionEncodingType::Learned => Some(LearnedPositionEncoding::new(
                config.position_encoding.clone(),
            )?),
            _ => None,
        };

        let final_norm = if config.final_layer_norm {
            Some(LayerNorm::new(LayerNormConfig::new(
                config.layer_config.attention.d_model,
            ))?)
        } else {
            None
        };

        Ok(Self {
            config,
            layers,
            position_encoding_sin,
            position_encoding_learned,
            final_norm,
        })
    }

    /// Build einsum graph for encoder stack
    ///
    /// Input tensors:
    /// - 0: x (input) [batch, seq_len, d_model]
    /// - 1-N: all parameters for position encoding, layers, and final norm
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_encoder_stack_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Add position encoding
        let mut current_output = if let Some(ref pe_sin) = self.position_encoding_sin {
            pe_sin.build_encoding_graph(graph)?[0]
        } else if let Some(ref pe_learned) = self.position_encoding_learned {
            pe_learned.build_encoding_graph(graph)?[0]
        } else {
            0 // No position encoding
        };

        // Step 2: Apply each encoder layer sequentially
        for (i, layer) in self.layers.iter().enumerate() {
            // Update the input tensor reference for this layer
            let layer_outputs = layer.build_encoder_layer_graph(graph)?;
            current_output = layer_outputs[0];

            // Add a marker for layer boundary
            let layer_marker = graph.add_tensor(format!("encoder_layer_{}_output", i));
            let marker_node =
                tensorlogic_ir::EinsumNode::elem_unary("identity", current_output, layer_marker);
            graph.add_node(marker_node)?;
            current_output = layer_marker;
        }

        // Step 3: Apply final layer normalization if configured
        if let Some(ref final_norm) = self.final_norm {
            let final_outputs = final_norm.build_layernorm_graph(graph)?;
            current_output = final_outputs[0];
        }

        Ok(vec![current_output])
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }
}

/// Configuration for transformer decoder stack
#[derive(Clone, Debug)]
pub struct DecoderStackConfig {
    /// Number of decoder layers
    pub num_layers: usize,
    /// Configuration for each decoder layer
    pub layer_config: DecoderLayerConfig,
    /// Position encoding configuration
    pub position_encoding: PositionEncodingConfig,
    /// Whether to apply final layer normalization
    pub final_layer_norm: bool,
}

impl DecoderStackConfig {
    /// Create a new decoder stack configuration
    pub fn new(
        num_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            num_layers,
            layer_config: DecoderLayerConfig::new(d_model, n_heads, d_ff)?,
            position_encoding: PositionEncodingConfig::sinusoidal(d_model, max_seq_len),
            final_layer_norm: true,
        })
    }

    /// Set position encoding type to learned
    pub fn with_learned_position_encoding(mut self) -> Self {
        self.position_encoding = PositionEncodingConfig::learned(
            self.position_encoding.d_model,
            self.position_encoding.max_seq_len,
        );
        self
    }

    /// Set whether to use final layer normalization
    pub fn with_final_layer_norm(mut self, final_layer_norm: bool) -> Self {
        self.final_layer_norm = final_layer_norm;
        self
    }

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.layer_config = self.layer_config.with_dropout(dropout);
        self.position_encoding = self.position_encoding.with_dropout(dropout);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.num_layers == 0 {
            return Err(crate::error::TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "num_layers must be positive".to_string(),
            });
        }

        self.layer_config.validate()?;
        self.position_encoding.validate()?;

        Ok(())
    }
}

/// Transformer decoder stack
#[derive(Clone, Debug)]
pub struct DecoderStack {
    /// Configuration
    pub config: DecoderStackConfig,
    /// Decoder layers
    pub layers: Vec<DecoderLayer>,
    /// Position encoding (if sinusoidal)
    pub position_encoding_sin: Option<SinusoidalPositionEncoding>,
    /// Position encoding (if learned)
    pub position_encoding_learned: Option<LearnedPositionEncoding>,
    /// Final layer normalization
    pub final_norm: Option<LayerNorm>,
}

impl DecoderStack {
    /// Create a new decoder stack
    pub fn new(config: DecoderStackConfig) -> Result<Self> {
        config.validate()?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(DecoderLayer::new(config.layer_config.clone())?);
        }

        let position_encoding_sin = match config.position_encoding.encoding_type {
            crate::position::PositionEncodingType::Sinusoidal { .. } => Some(
                SinusoidalPositionEncoding::new(config.position_encoding.clone())?,
            ),
            _ => None,
        };

        let position_encoding_learned = match config.position_encoding.encoding_type {
            crate::position::PositionEncodingType::Learned => Some(LearnedPositionEncoding::new(
                config.position_encoding.clone(),
            )?),
            _ => None,
        };

        let final_norm = if config.final_layer_norm {
            Some(LayerNorm::new(LayerNormConfig::new(
                config.layer_config.self_attention.d_model,
            ))?)
        } else {
            None
        };

        Ok(Self {
            config,
            layers,
            position_encoding_sin,
            position_encoding_learned,
            final_norm,
        })
    }

    /// Build einsum graph for decoder stack
    ///
    /// Input tensors:
    /// - 0: x (target input) [batch, tgt_len, d_model]
    /// - 1: encoder_output [batch, src_len, d_model]
    /// - 2-N: all parameters
    ///
    /// Output tensors:
    /// - output: [batch, tgt_len, d_model]
    pub fn build_decoder_stack_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Add position encoding to target
        let mut current_output = if let Some(ref pe_sin) = self.position_encoding_sin {
            pe_sin.build_encoding_graph(graph)?[0]
        } else if let Some(ref pe_learned) = self.position_encoding_learned {
            pe_learned.build_encoding_graph(graph)?[0]
        } else {
            0 // No position encoding
        };

        // Step 2: Apply each decoder layer sequentially
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_outputs = layer.build_decoder_layer_graph(graph)?;
            current_output = layer_outputs[0];

            // Add a marker for layer boundary
            let layer_marker = graph.add_tensor(format!("decoder_layer_{}_output", i));
            let marker_node =
                tensorlogic_ir::EinsumNode::elem_unary("identity", current_output, layer_marker);
            graph.add_node(marker_node)?;
            current_output = layer_marker;
        }

        // Step 3: Apply final layer normalization if configured
        if let Some(ref final_norm) = self.final_norm {
            let final_outputs = final_norm.build_layernorm_graph(graph)?;
            current_output = final_outputs[0];
        }

        Ok(vec![current_output])
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_stack_config_creation() {
        let config = EncoderStackConfig::new(6, 512, 8, 2048, 1024).unwrap();
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.layer_config.attention.d_model, 512);
        assert!(config.final_layer_norm);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_encoder_stack_config_with_learned_pe() {
        let config = EncoderStackConfig::new(6, 512, 8, 2048, 1024)
            .unwrap()
            .with_learned_position_encoding();
        assert!(matches!(
            config.position_encoding.encoding_type,
            crate::position::PositionEncodingType::Learned
        ));
    }

    #[test]
    fn test_encoder_stack_creation() {
        let config = EncoderStackConfig::new(6, 512, 8, 2048, 1024).unwrap();
        let stack = EncoderStack::new(config).unwrap();
        assert_eq!(stack.num_layers(), 6);
        assert!(stack.position_encoding_sin.is_some());
        assert!(stack.final_norm.is_some());
    }

    #[test]
    fn test_encoder_stack_graph_building() {
        let config = EncoderStackConfig::new(2, 512, 8, 2048, 1024).unwrap();
        let stack = EncoderStack::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");

        let outputs = stack.build_encoder_stack_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_decoder_stack_config_creation() {
        let config = DecoderStackConfig::new(6, 512, 8, 2048, 1024).unwrap();
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.layer_config.self_attention.d_model, 512);
        assert!(config.layer_config.self_attention.causal);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_decoder_stack_creation() {
        let config = DecoderStackConfig::new(6, 512, 8, 2048, 1024).unwrap();
        let stack = DecoderStack::new(config).unwrap();
        assert_eq!(stack.num_layers(), 6);
        assert!(stack.position_encoding_sin.is_some());
        assert!(stack.final_norm.is_some());
    }

    #[test]
    fn test_decoder_stack_graph_building() {
        let config = DecoderStackConfig::new(2, 512, 8, 2048, 1024).unwrap();
        let stack = DecoderStack::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("target");
        graph.add_tensor("encoder_output");

        let outputs = stack.build_decoder_stack_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_invalid_zero_layers() {
        let result = EncoderStackConfig::new(0, 512, 8, 2048, 1024);
        // Should fail validation when creating EncoderStack
        if let Ok(config) = result {
            assert!(config.validate().is_err());
        }
    }
}
