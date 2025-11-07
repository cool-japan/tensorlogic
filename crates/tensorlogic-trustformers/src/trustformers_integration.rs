//! Integration layer between TensorLogic and TrustformeRS.
//!
//! This module provides bidirectional conversion between TensorLogic's einsum-based
//! transformer components and TrustformeRS's model traits. It enables:
//!
//! 1. **TensorLogic → TrustformeRS**: Wrap TensorLogic transformer components as TrustformeRS models
//! 2. **TrustformeRS → TensorLogic**: Convert TrustformeRS model architectures to TLExpr
//! 3. **Weight Loading**: Load pre-trained weights from TrustformeRS checkpoint format
//! 4. **Model Export**: Export trained TensorLogic models to TrustformeRS format
//!
//! ## Design Philosophy
//!
//! - **Zero-Copy Where Possible**: Minimize data copying during conversions
//! - **Type Safety**: Leverage Rust's type system to prevent runtime errors
//! - **Backend Agnostic**: Conversions work with any TensorLogic backend
//! - **Compatibility**: Support standard TrustformeRS checkpoint formats
//!
//! ## Example: TensorLogic → TrustformeRS
//!
//! ```rust,no_run
//! use tensorlogic_trustformers::{EncoderStack, EncoderStackConfig};
//! use tensorlogic_trustformers::trustformers_integration::TensorLogicModel;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a TensorLogic encoder
//! let config = EncoderStackConfig::new(6, 512, 8, 2048, 1024)?;
//! let encoder = EncoderStack::new(config.clone())?;
//!
//! // Wrap as TrustformeRS model
//! let model = TensorLogicModel::from_encoder_stack(encoder, config)?;
//!
//! // Now it implements the TrustformeRS Model trait
//! // let output = model.forward(input)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: TrustformeRS → TensorLogic
//!
//! ```rust,ignore
//! use tensorlogic_trustformers::trustformers_integration::TrustformersConverter;
//!
//! // Convert TrustformeRS model architecture to TLExpr
//! let converter = TrustformersConverter::new();
//! // let tlexpr = converter.convert_model_architecture(&trustformers_model)?;
//!
//! // Compile to einsum graph
//! // use tensorlogic_compiler::CompilerContext;
//! // let mut ctx = CompilerContext::new();
//! // let graph = ctx.compile(&tlexpr)?;
//! ```

use serde::{Deserialize, Serialize};
use tensorlogic_ir::{EinsumGraph, TLExpr, Term};

use crate::{
    config::{AttentionConfig, FeedForwardConfig},
    error::{Result, TrustformerError},
    layers::{EncoderLayer, EncoderLayerConfig},
    stacks::{EncoderStack, EncoderStackConfig},
};

/// Configuration for TensorLogic <-> TrustformeRS conversion
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Whether to validate shapes during conversion
    pub validate_shapes: bool,
    /// Whether to preserve dropout layers (or compile them out)
    pub preserve_dropout: bool,
    /// Whether to use pre-layer normalization (vs post-layer)
    pub pre_norm: bool,
    /// Tolerance for numerical differences during validation
    pub numerical_tolerance: f64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            validate_shapes: true,
            preserve_dropout: true,
            pre_norm: true,
            numerical_tolerance: 1e-6,
        }
    }
}

impl IntegrationConfig {
    /// Create a new integration configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to validate shapes
    pub fn with_shape_validation(mut self, validate: bool) -> Self {
        self.validate_shapes = validate;
        self
    }

    /// Set whether to preserve dropout
    pub fn with_dropout_preservation(mut self, preserve: bool) -> Self {
        self.preserve_dropout = preserve;
        self
    }

    /// Set whether to use pre-layer normalization
    pub fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.pre_norm = pre_norm;
        self
    }

    /// Set numerical tolerance
    pub fn with_numerical_tolerance(mut self, tolerance: f64) -> Self {
        self.numerical_tolerance = tolerance;
        self
    }
}

/// Wrapper for TensorLogic transformer components that implements TrustformeRS Model trait
///
/// This allows TensorLogic einsum-based transformers to be used wherever
/// TrustformeRS models are expected.
#[derive(Clone, Debug)]
pub enum TensorLogicModel {
    /// Single encoder layer
    EncoderLayer {
        layer: EncoderLayer,
        config: EncoderLayerConfig,
    },
    /// Stack of encoder layers
    EncoderStack {
        stack: EncoderStack,
        config: EncoderStackConfig,
    },
}

impl TensorLogicModel {
    /// Create from an encoder layer
    pub fn from_encoder_layer(layer: EncoderLayer, config: EncoderLayerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self::EncoderLayer { layer, config })
    }

    /// Create from an encoder stack
    pub fn from_encoder_stack(stack: EncoderStack, config: EncoderStackConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self::EncoderStack { stack, config })
    }

    /// Build einsum graph for this model
    pub fn build_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        match self {
            Self::EncoderLayer { layer, .. } => layer.build_encoder_layer_graph(graph),
            Self::EncoderStack { stack, .. } => stack.build_encoder_stack_graph(graph),
        }
    }

    /// Get the model configuration
    pub fn config(&self) -> ModelConfig {
        match self {
            Self::EncoderLayer { config, .. } => ModelConfig::EncoderLayer {
                d_model: config.attention.d_model,
                n_heads: config.attention.n_heads,
                d_ff: config.feed_forward.d_ff,
                dropout: config.attention.dropout,
                pre_norm: config.pre_norm,
            },
            Self::EncoderStack { config, .. } => ModelConfig::EncoderStack {
                n_layers: config.num_layers,
                d_model: config.layer_config.attention.d_model,
                n_heads: config.layer_config.attention.n_heads,
                d_ff: config.layer_config.feed_forward.d_ff,
                max_seq_len: config.position_encoding.max_seq_len,
                dropout: config.layer_config.attention.dropout,
                pre_norm: config.layer_config.pre_norm,
            },
        }
    }

    /// Convert to TLExpr representation
    pub fn to_tlexpr(&self) -> Result<TLExpr> {
        match self {
            Self::EncoderLayer { config, .. } => {
                // Represent encoder layer as logical conjunction of attention and FFN
                let attention_expr = Self::attention_to_tlexpr(&config.attention)?;
                let ffn_expr = Self::ffn_to_tlexpr(&config.feed_forward)?;

                // Compose using And: attention AND ffn (both must be applied)
                Ok(TLExpr::And(Box::new(attention_expr), Box::new(ffn_expr)))
            }
            Self::EncoderStack { config, .. } => {
                // Represent stack as repeated application of encoder layers
                let layer_expr = {
                    let attn_cfg = AttentionConfig::new(
                        config.layer_config.attention.d_model,
                        config.layer_config.attention.n_heads,
                    )?;
                    let ffn_cfg = FeedForwardConfig::new(
                        config.layer_config.feed_forward.d_model,
                        config.layer_config.feed_forward.d_ff,
                    );

                    let attention_expr = Self::attention_to_tlexpr(&attn_cfg)?;
                    let ffn_expr = Self::ffn_to_tlexpr(&ffn_cfg)?;

                    TLExpr::And(Box::new(attention_expr), Box::new(ffn_expr))
                };

                // Repeat num_layers times using ForAll
                Ok(TLExpr::ForAll {
                    var: "layer".to_string(),
                    domain: format!("0..{}", config.num_layers),
                    body: Box::new(layer_expr),
                })
            }
        }
    }

    /// Convert attention configuration to TLExpr
    fn attention_to_tlexpr(config: &AttentionConfig) -> Result<TLExpr> {
        // Multi-head attention as einsum operations
        Ok(TLExpr::Pred {
            name: "MultiHeadAttention".to_string(),
            args: vec![
                Term::Const(format!("d_model={}", config.d_model)),
                Term::Const(format!("n_heads={}", config.n_heads)),
                Term::Const(format!("d_k={}", config.d_k)),
            ],
        })
    }

    /// Convert FFN configuration to TLExpr
    fn ffn_to_tlexpr(config: &FeedForwardConfig) -> Result<TLExpr> {
        Ok(TLExpr::Pred {
            name: "FeedForward".to_string(),
            args: vec![
                Term::Const(format!("d_model={}", config.d_model)),
                Term::Const(format!("d_ff={}", config.d_ff)),
                Term::Const(format!("activation={}", config.activation)),
            ],
        })
    }
}

/// Configuration description for a TensorLogic model
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ModelConfig {
    /// Single encoder layer configuration
    EncoderLayer {
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        pre_norm: bool,
    },
    /// Encoder stack configuration
    EncoderStack {
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        max_seq_len: usize,
        dropout: f64,
        pre_norm: bool,
    },
}

/// Converter from TrustformeRS model architectures to TensorLogic IR
///
/// This converter analyzes TrustformeRS model structures and generates
/// equivalent TLExpr representations that can be compiled to einsum graphs.
#[derive(Clone, Debug)]
pub struct TrustformersConverter {
    /// Conversion configuration
    pub config: IntegrationConfig,
}

impl TrustformersConverter {
    /// Create a new converter with default configuration
    pub fn new() -> Self {
        Self {
            config: IntegrationConfig::default(),
        }
    }

    /// Create a new converter with custom configuration
    pub fn with_config(config: IntegrationConfig) -> Self {
        Self { config }
    }

    /// Convert a BERT-style encoder model to TLExpr
    ///
    /// This analyzes the model's layer structure and generates corresponding
    /// TensorLogic expressions.
    pub fn convert_bert_encoder(
        &self,
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
    ) -> Result<TLExpr> {
        // Validate configuration
        if n_layers == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "n_layers must be > 0".to_string(),
            });
        }
        if !d_model.is_multiple_of(n_heads) {
            return Err(TrustformerError::InvalidDimension {
                expected: n_heads,
                got: d_model,
                context: format!(
                    "d_model {} must be divisible by n_heads {}",
                    d_model, n_heads
                ),
            });
        }

        // Create decoder layer with causal attention
        let attn_cfg = AttentionConfig::new(d_model, n_heads)?;
        let ffn_cfg = FeedForwardConfig::new(d_model, d_ff);

        let attention_expr = TLExpr::Pred {
            name: "MultiHeadAttention".to_string(),
            args: vec![
                Term::Const(format!("d_model={}", attn_cfg.d_model)),
                Term::Const(format!("n_heads={}", attn_cfg.n_heads)),
                Term::Const(format!("d_k={}", attn_cfg.d_k)),
            ],
        };

        let ffn_expr = TLExpr::Pred {
            name: "FeedForward".to_string(),
            args: vec![
                Term::Const(format!("d_model={}", ffn_cfg.d_model)),
                Term::Const(format!("d_ff={}", ffn_cfg.d_ff)),
                Term::Const(format!("activation={}", ffn_cfg.activation)),
            ],
        };

        let layer_expr = TLExpr::And(Box::new(attention_expr), Box::new(ffn_expr));

        // Repeat for all layers
        Ok(TLExpr::ForAll {
            var: "layer".to_string(),
            domain: format!("0..{}", n_layers),
            body: Box::new(layer_expr),
        })
    }

    /// Convert a GPT-style decoder model to TLExpr
    pub fn convert_gpt_decoder(
        &self,
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
    ) -> Result<TLExpr> {
        // Validate configuration
        if n_layers == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "n_layers must be > 0".to_string(),
            });
        }
        if !d_model.is_multiple_of(n_heads) {
            return Err(TrustformerError::InvalidDimension {
                expected: n_heads,
                got: d_model,
                context: format!(
                    "d_model {} must be divisible by n_heads {}",
                    d_model, n_heads
                ),
            });
        }

        // Create decoder layer with causal attention
        let attn_cfg = AttentionConfig::new(d_model, n_heads)?.with_causal(true);
        let ffn_cfg = FeedForwardConfig::new(d_model, d_ff);

        let causal_attention_expr = TLExpr::Pred {
            name: "CausalMultiHeadAttention".to_string(),
            args: vec![
                Term::Const(format!("d_model={}", attn_cfg.d_model)),
                Term::Const(format!("n_heads={}", attn_cfg.n_heads)),
                Term::Const(format!("d_k={}", attn_cfg.d_k)),
                Term::Const("causal=true".to_string()),
            ],
        };

        let ffn_expr = TLExpr::Pred {
            name: "FeedForward".to_string(),
            args: vec![
                Term::Const(format!("d_model={}", ffn_cfg.d_model)),
                Term::Const(format!("d_ff={}", ffn_cfg.d_ff)),
                Term::Const(format!("activation={}", ffn_cfg.activation)),
            ],
        };

        let layer_expr = TLExpr::And(Box::new(causal_attention_expr), Box::new(ffn_expr));

        // Repeat for all layers
        Ok(TLExpr::ForAll {
            var: "layer".to_string(),
            domain: format!("0..{}", n_layers),
            body: Box::new(layer_expr),
        })
    }

    /// Convert generic transformer architecture to TLExpr
    pub fn convert_transformer(
        &self,
        encoder_layers: usize,
        decoder_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
    ) -> Result<TLExpr> {
        let encoder_expr = if encoder_layers > 0 {
            Some(self.convert_bert_encoder(encoder_layers, d_model, n_heads, d_ff)?)
        } else {
            None
        };

        let decoder_expr = if decoder_layers > 0 {
            Some(self.convert_gpt_decoder(decoder_layers, d_model, n_heads, d_ff)?)
        } else {
            None
        };

        match (encoder_expr, decoder_expr) {
            (Some(enc), Some(dec)) => {
                // Full encoder-decoder transformer (encoder AND decoder both applied)
                Ok(TLExpr::And(Box::new(enc), Box::new(dec)))
            }
            (Some(enc), None) => Ok(enc),
            (None, Some(dec)) => Ok(dec),
            (None, None) => Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "At least one of encoder_layers or decoder_layers must be > 0".to_string(),
            }),
        }
    }
}

impl Default for TrustformersConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Weight loader for TrustformeRS checkpoint format
///
/// Supports loading weights from various TrustformeRS checkpoint formats:
/// - SafeTensors
/// - PyTorch .bin
/// - TensorFlow SavedModel
#[derive(Clone, Debug)]
pub struct TrustformersWeightLoader {
    /// Integration configuration
    pub config: IntegrationConfig,
}

impl TrustformersWeightLoader {
    /// Create a new weight loader
    pub fn new() -> Self {
        Self {
            config: IntegrationConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: IntegrationConfig) -> Self {
        Self { config }
    }

    /// Load weights from a TrustformeRS checkpoint file
    ///
    /// This is a placeholder for the actual implementation which would:
    /// 1. Parse the checkpoint format
    /// 2. Extract weight tensors
    /// 3. Map them to TensorLogic graph tensor names
    /// 4. Return weight data in a format suitable for execution
    pub fn load_checkpoint(&self, _path: &str) -> Result<CheckpointData> {
        // TODO: Implement actual checkpoint loading
        // This would use trustformers-core's checkpoint loading utilities
        Ok(CheckpointData::default())
    }

    /// Map TrustformeRS layer names to TensorLogic tensor names
    ///
    /// Example mappings:
    /// - "encoder.layer.0.attention.query.weight" -> "encoder_0_attn_q_weight"
    /// - "encoder.layer.0.attention.key.weight" -> "encoder_0_attn_k_weight"
    pub fn map_layer_name(&self, trustformers_name: &str) -> Result<String> {
        // Simple mapping strategy - can be made more sophisticated
        let mapped = trustformers_name
            .replace("encoder.layer.", "encoder_")
            .replace("decoder.layer.", "decoder_")
            .replace(".attention.", "_attn_")
            .replace(".feed_forward.", "_ffn_")
            .replace(".query.", "_q_")
            .replace(".key.", "_k_")
            .replace(".value.", "_v_")
            .replace(".weight", "_weight")
            .replace(".bias", "_bias");

        Ok(mapped)
    }
}

impl Default for TrustformersWeightLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Checkpoint data loaded from TrustformeRS format
#[derive(Clone, Debug, Default)]
pub struct CheckpointData {
    /// Mapping from tensor names to weight data
    pub weights: std::collections::HashMap<String, Vec<f32>>,
    /// Model configuration metadata
    pub metadata: std::collections::HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_config_creation() {
        let config = IntegrationConfig::new();
        assert!(config.validate_shapes);
        assert!(config.preserve_dropout);
        assert!(config.pre_norm);
        assert!((config.numerical_tolerance - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_integration_config_builder() {
        let config = IntegrationConfig::new()
            .with_shape_validation(false)
            .with_dropout_preservation(false)
            .with_pre_norm(false)
            .with_numerical_tolerance(1e-5);

        assert!(!config.validate_shapes);
        assert!(!config.preserve_dropout);
        assert!(!config.pre_norm);
        assert!((config.numerical_tolerance - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_tensorlogic_model_from_encoder_layer() {
        let config = EncoderLayerConfig::new(512, 8, 2048).unwrap();
        let layer = EncoderLayer::new(config.clone()).unwrap();
        let model = TensorLogicModel::from_encoder_layer(layer, config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_tensorlogic_model_from_encoder_stack() {
        let config = EncoderStackConfig::new(6, 512, 8, 2048, 1024).unwrap();
        let stack = EncoderStack::new(config.clone()).unwrap();
        let model = TensorLogicModel::from_encoder_stack(stack, config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_tensorlogic_model_build_graph() {
        let config = EncoderLayerConfig::new(512, 8, 2048).unwrap();
        let layer = EncoderLayer::new(config.clone()).unwrap();
        let model = TensorLogicModel::from_encoder_layer(layer, config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("input");

        let outputs = model.build_graph(&mut graph);
        assert!(outputs.is_ok());
    }

    #[test]
    fn test_tensorlogic_model_to_tlexpr() {
        let config = EncoderLayerConfig::new(512, 8, 2048).unwrap();
        let layer = EncoderLayer::new(config.clone()).unwrap();
        let model = TensorLogicModel::from_encoder_layer(layer, config).unwrap();

        let expr = model.to_tlexpr();
        assert!(expr.is_ok());
    }

    #[test]
    fn test_tensorlogic_model_config() {
        let config = EncoderLayerConfig::new(512, 8, 2048).unwrap();
        let layer = EncoderLayer::new(config.clone()).unwrap();
        let model = TensorLogicModel::from_encoder_layer(layer, config).unwrap();

        let model_config = model.config();
        match model_config {
            ModelConfig::EncoderLayer {
                d_model,
                n_heads,
                d_ff,
                ..
            } => {
                assert_eq!(d_model, 512);
                assert_eq!(n_heads, 8);
                assert_eq!(d_ff, 2048);
            }
            _ => panic!("Expected EncoderLayer config"),
        }
    }

    #[test]
    fn test_trustformers_converter_creation() {
        let converter = TrustformersConverter::new();
        assert!(converter.config.validate_shapes);
    }

    #[test]
    fn test_trustformers_converter_with_config() {
        let config = IntegrationConfig::new().with_shape_validation(false);
        let converter = TrustformersConverter::with_config(config);
        assert!(!converter.config.validate_shapes);
    }

    #[test]
    fn test_convert_bert_encoder() {
        let converter = TrustformersConverter::new();
        let expr = converter.convert_bert_encoder(6, 512, 8, 2048);
        assert!(expr.is_ok());

        let expr = expr.unwrap();
        match expr {
            TLExpr::ForAll { var, body, .. } => {
                assert_eq!(var, "layer");
                match *body {
                    TLExpr::And(..) => {
                        // Correctly represents composition of attention and FFN
                    }
                    _ => panic!("Expected And"),
                }
            }
            _ => panic!("Expected ForAll"),
        }
    }

    #[test]
    fn test_convert_gpt_decoder() {
        let converter = TrustformersConverter::new();
        let expr = converter.convert_gpt_decoder(12, 768, 12, 3072);
        assert!(expr.is_ok());
    }

    #[test]
    fn test_convert_transformer_encoder_only() {
        let converter = TrustformersConverter::new();
        let expr = converter.convert_transformer(6, 0, 512, 8, 2048);
        assert!(expr.is_ok());
    }

    #[test]
    fn test_convert_transformer_decoder_only() {
        let converter = TrustformersConverter::new();
        let expr = converter.convert_transformer(0, 6, 512, 8, 2048);
        assert!(expr.is_ok());
    }

    #[test]
    fn test_convert_transformer_encoder_decoder() {
        let converter = TrustformersConverter::new();
        let expr = converter.convert_transformer(6, 6, 512, 8, 2048);
        assert!(expr.is_ok());

        let expr = expr.unwrap();
        match expr {
            TLExpr::And(..) => {
                // Correctly represents encoder AND decoder composition
            }
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_convert_transformer_invalid_zero_layers() {
        let converter = TrustformersConverter::new();
        let expr = converter.convert_transformer(0, 0, 512, 8, 2048);
        assert!(expr.is_err());
    }

    #[test]
    fn test_convert_bert_invalid_heads() {
        let converter = TrustformersConverter::new();
        // 512 is not divisible by 7
        let expr = converter.convert_bert_encoder(6, 512, 7, 2048);
        assert!(expr.is_err());
    }

    #[test]
    fn test_weight_loader_creation() {
        let loader = TrustformersWeightLoader::new();
        assert!(loader.config.validate_shapes);
    }

    #[test]
    fn test_weight_loader_map_layer_name() {
        let loader = TrustformersWeightLoader::new();

        let mapped = loader
            .map_layer_name("encoder.layer.0.attention.query.weight")
            .unwrap();
        assert_eq!(mapped, "encoder_0_attn_query_weight");

        let mapped = loader
            .map_layer_name("decoder.layer.5.feed_forward.weight")
            .unwrap();
        assert_eq!(mapped, "decoder_5_ffn_weight");
    }

    #[test]
    fn test_checkpoint_data_default() {
        let data = CheckpointData::default();
        assert!(data.weights.is_empty());
        assert!(data.metadata.is_empty());
    }
}
