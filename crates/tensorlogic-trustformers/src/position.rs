//! Position encoding implementations for transformer models.
//!
//! This module provides various position encoding strategies that can be
//! compiled to TensorLogic einsum graphs:
//!
//! 1. **Sinusoidal Encoding**: Fixed position encodings using sin/cos functions
//!    (from "Attention Is All You Need")
//! 2. **Learned Encoding**: Trainable position embeddings
//! 3. **Relative Position**: Relative position biases for attention
//!
//! ## Sinusoidal Position Encoding
//!
//! ```text
//! PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
//! PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
//! ```
//!
//! Where:
//! - `pos` = position in sequence
//! - `i` = dimension index
//! - `d_model` = model dimension

use serde::{Deserialize, Serialize};
use tensorlogic_ir::{EinsumGraph, EinsumNode};

use crate::error::{Result, TrustformerError};

/// Configuration for position encodings
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PositionEncodingConfig {
    /// Model dimension
    pub d_model: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Encoding type
    pub encoding_type: PositionEncodingType,
    /// Dropout probability
    pub dropout: f64,
}

/// Type of position encoding
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PositionEncodingType {
    /// Sinusoidal (fixed) position encoding
    Sinusoidal {
        /// Base for frequency computation (default: 10000.0)
        base: f64,
    },
    /// Learned position embedding
    Learned,
    /// Relative position encoding
    Relative {
        /// Number of relative position buckets
        num_buckets: usize,
        /// Maximum relative distance
        max_distance: usize,
    },
    /// Rotary Position Embedding (RoPE) - used in LLaMA, GPT-NeoX
    Rotary {
        /// Base for frequency computation (default: 10000.0)
        base: f64,
        /// Scaling factor for long sequences (default: 1.0)
        scaling_factor: f64,
    },
    /// ALiBi (Attention with Linear Biases) - used in BLOOM
    Alibi {
        /// Number of attention heads
        n_heads: usize,
        /// Maximum sequence length
        max_seq_len: usize,
    },
}

impl PositionEncodingConfig {
    /// Create a new sinusoidal position encoding configuration
    pub fn sinusoidal(d_model: usize, max_seq_len: usize) -> Self {
        Self {
            d_model,
            max_seq_len,
            encoding_type: PositionEncodingType::Sinusoidal { base: 10000.0 },
            dropout: 0.0,
        }
    }

    /// Create a new learned position encoding configuration
    pub fn learned(d_model: usize, max_seq_len: usize) -> Self {
        Self {
            d_model,
            max_seq_len,
            encoding_type: PositionEncodingType::Learned,
            dropout: 0.0,
        }
    }

    /// Create a new relative position encoding configuration
    pub fn relative(d_model: usize, num_buckets: usize, max_distance: usize) -> Self {
        Self {
            d_model,
            max_seq_len: 0, // Not used for relative encoding
            encoding_type: PositionEncodingType::Relative {
                num_buckets,
                max_distance,
            },
            dropout: 0.0,
        }
    }

    /// Create a new rotary position encoding (RoPE) configuration
    pub fn rotary(d_model: usize, max_seq_len: usize) -> Self {
        Self {
            d_model,
            max_seq_len,
            encoding_type: PositionEncodingType::Rotary {
                base: 10000.0,
                scaling_factor: 1.0,
            },
            dropout: 0.0,
        }
    }

    /// Create RoPE with custom base and scaling
    pub fn rotary_scaled(
        d_model: usize,
        max_seq_len: usize,
        base: f64,
        scaling_factor: f64,
    ) -> Self {
        Self {
            d_model,
            max_seq_len,
            encoding_type: PositionEncodingType::Rotary {
                base,
                scaling_factor,
            },
            dropout: 0.0,
        }
    }

    /// Create a new ALiBi position encoding configuration
    pub fn alibi(d_model: usize, n_heads: usize, max_seq_len: usize) -> Self {
        Self {
            d_model,
            max_seq_len,
            encoding_type: PositionEncodingType::Alibi {
                n_heads,
                max_seq_len,
            },
            dropout: 0.0,
        }
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "d_model must be positive".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: format!("dropout must be in [0,1], got {}", self.dropout),
            });
        }

        match &self.encoding_type {
            PositionEncodingType::Sinusoidal { base } => {
                if *base <= 0.0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "base must be positive".to_string(),
                    });
                }
            }
            PositionEncodingType::Relative {
                num_buckets,
                max_distance,
            } => {
                if *num_buckets == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "num_buckets must be positive".to_string(),
                    });
                }
                if *max_distance == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "max_distance must be positive".to_string(),
                    });
                }
            }
            PositionEncodingType::Learned => {
                if self.max_seq_len == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "max_seq_len must be positive for learned encoding".to_string(),
                    });
                }
            }
            PositionEncodingType::Rotary {
                base,
                scaling_factor,
            } => {
                if *base <= 0.0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "RoPE base must be positive".to_string(),
                    });
                }
                if *scaling_factor <= 0.0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "RoPE scaling_factor must be positive".to_string(),
                    });
                }
                if self.max_seq_len == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "max_seq_len must be positive for RoPE".to_string(),
                    });
                }
                if !self.d_model.is_multiple_of(2) {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "d_model must be even for RoPE".to_string(),
                    });
                }
            }
            PositionEncodingType::Alibi {
                n_heads,
                max_seq_len,
            } => {
                if *n_heads == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "n_heads must be positive for ALiBi".to_string(),
                    });
                }
                if *max_seq_len == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "max_seq_len must be positive for ALiBi".to_string(),
                    });
                }
            }
        }

        Ok(())
    }
}

/// Sinusoidal position encoding
#[derive(Clone, Debug)]
pub struct SinusoidalPositionEncoding {
    /// Configuration
    pub config: PositionEncodingConfig,
}

impl SinusoidalPositionEncoding {
    /// Create a new sinusoidal position encoding
    pub fn new(config: PositionEncodingConfig) -> Result<Self> {
        config.validate()?;
        match config.encoding_type {
            PositionEncodingType::Sinusoidal { .. } => Ok(Self { config }),
            _ => Err(TrustformerError::InvalidDimension {
                expected: 0,
                got: 1,
                context: "Expected Sinusoidal encoding type".to_string(),
            }),
        }
    }

    /// Build einsum graph for sinusoidal position encoding
    ///
    /// Input tensors:
    /// - 0: x (input) [batch, seq_len, d_model]
    /// - 1: position_ids [batch, seq_len] (optional, will use 0..seq_len if not provided)
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model] (x + position_encoding)
    pub fn build_encoding_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // The sinusoidal encoding is computed as:
        // PE(pos, 2i) = sin(pos / base^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / base^(2i/d_model))

        // For einsum representation, we add a pre-computed tensor
        let pe_tensor = graph.add_tensor("sinusoidal_pe");

        // Add position encoding to input
        // einsum("bsd,sd->bsd", x, pe) (broadcast addition)
        let output_tensor = graph.add_tensor("x_with_pe");
        let add_node = EinsumNode::elem_binary("add", 0, pe_tensor, output_tensor);
        graph.add_node(add_node)?;

        // Apply dropout if configured
        if self.config.dropout > 0.0 {
            let dropout_tensor = graph.add_tensor("pe_dropout_output");
            let dropout_node = EinsumNode::elem_unary(
                format!("dropout_{}", self.config.dropout),
                output_tensor,
                dropout_tensor,
            );
            graph.add_node(dropout_node)?;
            Ok(vec![dropout_tensor])
        } else {
            Ok(vec![output_tensor])
        }
    }

    /// Get the base frequency for encoding
    pub fn base(&self) -> f64 {
        match self.config.encoding_type {
            PositionEncodingType::Sinusoidal { base } => base,
            _ => 10000.0,
        }
    }
}

/// Learned position encoding
#[derive(Clone, Debug)]
pub struct LearnedPositionEncoding {
    /// Configuration
    pub config: PositionEncodingConfig,
}

impl LearnedPositionEncoding {
    /// Create a new learned position encoding
    pub fn new(config: PositionEncodingConfig) -> Result<Self> {
        config.validate()?;
        match config.encoding_type {
            PositionEncodingType::Learned => Ok(Self { config }),
            _ => Err(TrustformerError::InvalidDimension {
                expected: 0,
                got: 1,
                context: "Expected Learned encoding type".to_string(),
            }),
        }
    }

    /// Build einsum graph for learned position encoding
    ///
    /// Input tensors:
    /// - 0: x (input) [batch, seq_len, d_model]
    /// - 1: position_embeddings [max_seq_len, d_model] (learned parameter)
    /// - 2: position_ids [batch, seq_len] (optional)
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model] (x + position_embedding)
    pub fn build_encoding_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Lookup position embeddings
        // For simplicity, we use direct indexing which maps to gather operation
        let pe_lookup = graph.add_tensor("pe_lookup");
        let lookup_node = EinsumNode::elem_unary("gather_pos_emb", 1, pe_lookup);
        graph.add_node(lookup_node)?;

        // Add position encoding to input
        let output_tensor = graph.add_tensor("x_with_learned_pe");
        let add_node = EinsumNode::elem_binary("add", 0, pe_lookup, output_tensor);
        graph.add_node(add_node)?;

        // Apply dropout if configured
        if self.config.dropout > 0.0 {
            let dropout_tensor = graph.add_tensor("learned_pe_dropout_output");
            let dropout_node = EinsumNode::elem_unary(
                format!("dropout_{}", self.config.dropout),
                output_tensor,
                dropout_tensor,
            );
            graph.add_node(dropout_node)?;
            Ok(vec![dropout_tensor])
        } else {
            Ok(vec![output_tensor])
        }
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }
}

/// Relative position encoding
#[derive(Clone, Debug)]
pub struct RelativePositionEncoding {
    /// Configuration
    pub config: PositionEncodingConfig,
}

impl RelativePositionEncoding {
    /// Create a new relative position encoding
    pub fn new(config: PositionEncodingConfig) -> Result<Self> {
        config.validate()?;
        match config.encoding_type {
            PositionEncodingType::Relative { .. } => Ok(Self { config }),
            _ => Err(TrustformerError::InvalidDimension {
                expected: 0,
                got: 1,
                context: "Expected Relative encoding type".to_string(),
            }),
        }
    }

    /// Build einsum graph for relative position bias
    ///
    /// Input tensors:
    /// - 0: attention_scores [batch, n_heads, seq_len, seq_len]
    /// - 1: relative_position_bias [n_heads, num_buckets]
    /// - 2: relative_position_indices [seq_len, seq_len] (bucket indices)
    ///
    /// Output tensors:
    /// - output: [batch, n_heads, seq_len, seq_len] (scores + bias)
    pub fn build_bias_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Lookup relative position bias based on indices
        let bias_lookup = graph.add_tensor("rel_pos_bias_lookup");
        let lookup_node = EinsumNode::elem_unary("gather_rel_bias", 1, bias_lookup);
        graph.add_node(lookup_node)?;

        // Add bias to attention scores
        // einsum("bhqk,hqk->bhqk", scores, bias) (broadcast addition)
        let output_tensor = graph.add_tensor("scores_with_rel_bias");
        let add_node = EinsumNode::elem_binary("add", 0, bias_lookup, output_tensor);
        graph.add_node(add_node)?;

        Ok(vec![output_tensor])
    }

    /// Get number of relative position buckets
    pub fn num_buckets(&self) -> usize {
        match self.config.encoding_type {
            PositionEncodingType::Relative { num_buckets, .. } => num_buckets,
            _ => 0,
        }
    }

    /// Get maximum relative distance
    pub fn max_distance(&self) -> usize {
        match self.config.encoding_type {
            PositionEncodingType::Relative { max_distance, .. } => max_distance,
            _ => 0,
        }
    }
}

/// Rotary Position Embedding (RoPE)
///
/// Used in models like LLaMA, GPT-NeoX, PaLM. RoPE encodes position by rotating
/// query and key vectors in the complex plane, providing natural relative position
/// information without adding extra parameters.
///
/// Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
/// https://arxiv.org/abs/2104.09864
#[derive(Clone, Debug)]
pub struct RotaryPositionEncoding {
    /// Configuration
    pub config: PositionEncodingConfig,
}

impl RotaryPositionEncoding {
    /// Create a new rotary position encoding
    pub fn new(config: PositionEncodingConfig) -> Result<Self> {
        config.validate()?;
        match config.encoding_type {
            PositionEncodingType::Rotary { .. } => Ok(Self { config }),
            _ => Err(TrustformerError::InvalidDimension {
                expected: 0,
                got: 1,
                context: "Expected Rotary encoding type".to_string(),
            }),
        }
    }

    /// Build einsum graph for RoPE
    ///
    /// RoPE applies rotation to query and key vectors in attention:
    /// - Splits d_model into pairs of dimensions
    /// - Rotates each pair by position-dependent angle
    /// - Preserves relative position information
    ///
    /// Input tensors:
    /// - 0: x (input) [batch, seq_len, d_model]
    /// - 1: cos_cached [max_seq_len, d_model/2] (precomputed cosines)
    /// - 2: sin_cached [max_seq_len, d_model/2] (precomputed sines)
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model] (rotated embeddings)
    pub fn build_encoding_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // RoPE rotation formula:
        // x_rot = [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]

        // Split input into even and odd indices
        let x_even = graph.add_tensor("rope_x_even");
        let x_odd = graph.add_tensor("rope_x_odd");
        let split_node = EinsumNode::elem_unary("split_even_odd", 0, x_even);
        graph.add_node(split_node)?;

        // Apply rotation using cached cos/sin values
        // x_even * cos
        let even_cos = graph.add_tensor("rope_even_cos");
        let even_cos_node = EinsumNode::elem_binary("mul", x_even, 1, even_cos);
        graph.add_node(even_cos_node)?;

        // x_odd * sin
        let odd_sin = graph.add_tensor("rope_odd_sin");
        let odd_sin_node = EinsumNode::elem_binary("mul", x_odd, 2, odd_sin);
        graph.add_node(odd_sin_node)?;

        // First half: x_even * cos - x_odd * sin
        let rotated_0 = graph.add_tensor("rope_rotated_0");
        let sub_node = EinsumNode::elem_binary("sub", even_cos, odd_sin, rotated_0);
        graph.add_node(sub_node)?;

        // x_even * sin
        let even_sin = graph.add_tensor("rope_even_sin");
        let even_sin_node = EinsumNode::elem_binary("mul", x_even, 2, even_sin);
        graph.add_node(even_sin_node)?;

        // x_odd * cos
        let odd_cos = graph.add_tensor("rope_odd_cos");
        let odd_cos_node = EinsumNode::elem_binary("mul", x_odd, 1, odd_cos);
        graph.add_node(odd_cos_node)?;

        // Second half: x_even * sin + x_odd * cos
        let rotated_1 = graph.add_tensor("rope_rotated_1");
        let add_node = EinsumNode::elem_binary("add", even_sin, odd_cos, rotated_1);
        graph.add_node(add_node)?;

        // Concatenate rotated halves
        let output_tensor = graph.add_tensor("rope_output");
        let concat_node = EinsumNode::elem_binary("concat", rotated_0, rotated_1, output_tensor);
        graph.add_node(concat_node)?;

        Ok(vec![output_tensor])
    }

    /// Get the base frequency for RoPE
    pub fn base(&self) -> f64 {
        match self.config.encoding_type {
            PositionEncodingType::Rotary { base, .. } => base,
            _ => 10000.0,
        }
    }

    /// Get the scaling factor for long sequences
    pub fn scaling_factor(&self) -> f64 {
        match self.config.encoding_type {
            PositionEncodingType::Rotary { scaling_factor, .. } => scaling_factor,
            _ => 1.0,
        }
    }
}

/// ALiBi (Attention with Linear Biases)
///
/// Used in models like BLOOM. Instead of adding position embeddings to inputs,
/// ALiBi adds a bias to attention scores that linearly penalizes distance.
/// This allows extrapolation to longer sequences than seen during training.
///
/// Reference: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
/// https://arxiv.org/abs/2108.12409
#[derive(Clone, Debug)]
pub struct AlibiPositionEncoding {
    /// Configuration
    pub config: PositionEncodingConfig,
}

impl AlibiPositionEncoding {
    /// Create a new ALiBi position encoding
    pub fn new(config: PositionEncodingConfig) -> Result<Self> {
        config.validate()?;
        match config.encoding_type {
            PositionEncodingType::Alibi { .. } => Ok(Self { config }),
            _ => Err(TrustformerError::InvalidDimension {
                expected: 0,
                got: 1,
                context: "Expected Alibi encoding type".to_string(),
            }),
        }
    }

    /// Build einsum graph for ALiBi bias
    ///
    /// ALiBi adds linear biases to attention scores based on query-key distance:
    /// bias(i, j) = -m * |i - j|
    /// where m is a head-specific slope
    ///
    /// Input tensors:
    /// - 0: attention_scores [batch, n_heads, seq_len, seq_len]
    /// - 1: alibi_slopes [n_heads] (precomputed slopes, one per head)
    /// - 2: distance_matrix [seq_len, seq_len] (|i - j| for all positions)
    ///
    /// Output tensors:
    /// - output: [batch, n_heads, seq_len, seq_len] (scores with ALiBi bias)
    pub fn build_bias_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Compute -m * |i - j| for each head
        // slopes: [n_heads, 1, 1]
        // distance_matrix: [seq_len, seq_len]
        // bias = -slopes * distance_matrix: [n_heads, seq_len, seq_len]

        let slopes_expanded = graph.add_tensor("alibi_slopes_expanded");
        let expand_node = EinsumNode::elem_unary("expand_dims", 1, slopes_expanded);
        graph.add_node(expand_node)?;

        let bias = graph.add_tensor("alibi_bias");
        let bias_node = EinsumNode::elem_binary("mul", slopes_expanded, 2, bias);
        graph.add_node(bias_node)?;

        let neg_bias = graph.add_tensor("alibi_neg_bias");
        let neg_node = EinsumNode::elem_unary("neg", bias, neg_bias);
        graph.add_node(neg_node)?;

        // Add bias to attention scores
        // scores: [batch, n_heads, seq_len, seq_len]
        // bias: [n_heads, seq_len, seq_len] (broadcasts over batch)
        let output_tensor = graph.add_tensor("scores_with_alibi");
        let add_node = EinsumNode::elem_binary("add", 0, neg_bias, output_tensor);
        graph.add_node(add_node)?;

        Ok(vec![output_tensor])
    }

    /// Get the number of attention heads
    pub fn n_heads(&self) -> usize {
        match self.config.encoding_type {
            PositionEncodingType::Alibi { n_heads, .. } => n_heads,
            _ => 0,
        }
    }

    /// Compute ALiBi slopes for each attention head
    ///
    /// Slopes are computed as: m_i = 2^(-8i/n) for i in 1..n_heads
    /// This gives different rates of distance penalty per head
    pub fn compute_slopes(&self) -> Vec<f64> {
        let n = self.n_heads();
        (1..=n)
            .map(|i| 2_f64.powf(-8.0 * (i as f64) / (n as f64)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_config_creation() {
        let config = PositionEncodingConfig::sinusoidal(512, 2048);
        assert_eq!(config.d_model, 512);
        assert_eq!(config.max_seq_len, 2048);
        assert!(matches!(
            config.encoding_type,
            PositionEncodingType::Sinusoidal { base: 10000.0 }
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_learned_config_creation() {
        let config = PositionEncodingConfig::learned(512, 2048);
        assert_eq!(config.d_model, 512);
        assert_eq!(config.max_seq_len, 2048);
        assert!(matches!(
            config.encoding_type,
            PositionEncodingType::Learned
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_relative_config_creation() {
        let config = PositionEncodingConfig::relative(512, 32, 128);
        assert_eq!(config.d_model, 512);
        assert!(matches!(
            config.encoding_type,
            PositionEncodingType::Relative {
                num_buckets: 32,
                max_distance: 128
            }
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_with_dropout() {
        let config = PositionEncodingConfig::sinusoidal(512, 2048).with_dropout(0.1);
        assert!((config.dropout - 0.1).abs() < 1e-10);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_sinusoidal_encoding_creation() {
        let config = PositionEncodingConfig::sinusoidal(512, 2048);
        let encoding = SinusoidalPositionEncoding::new(config).unwrap();
        assert_eq!(encoding.config.d_model, 512);
        assert_eq!(encoding.base(), 10000.0);
    }

    #[test]
    fn test_learned_encoding_creation() {
        let config = PositionEncodingConfig::learned(512, 2048);
        let encoding = LearnedPositionEncoding::new(config).unwrap();
        assert_eq!(encoding.max_seq_len(), 2048);
    }

    #[test]
    fn test_relative_encoding_creation() {
        let config = PositionEncodingConfig::relative(512, 32, 128);
        let encoding = RelativePositionEncoding::new(config).unwrap();
        assert_eq!(encoding.num_buckets(), 32);
        assert_eq!(encoding.max_distance(), 128);
    }

    #[test]
    fn test_sinusoidal_graph_building() {
        let config = PositionEncodingConfig::sinusoidal(512, 2048);
        let encoding = SinusoidalPositionEncoding::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");

        let outputs = encoding.build_encoding_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_learned_graph_building() {
        let config = PositionEncodingConfig::learned(512, 2048);
        let encoding = LearnedPositionEncoding::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");
        graph.add_tensor("position_embeddings");

        let outputs = encoding.build_encoding_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_relative_bias_graph_building() {
        let config = PositionEncodingConfig::relative(512, 32, 128);
        let encoding = RelativePositionEncoding::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("attention_scores");
        graph.add_tensor("relative_position_bias");
        graph.add_tensor("relative_position_indices");

        let outputs = encoding.build_bias_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_invalid_config_zero_dimension() {
        let mut config = PositionEncodingConfig::sinusoidal(0, 2048);
        assert!(config.validate().is_err());

        config = PositionEncodingConfig::learned(512, 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_dropout() {
        let config = PositionEncodingConfig::sinusoidal(512, 2048).with_dropout(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_wrong_encoding_type() {
        let config = PositionEncodingConfig::learned(512, 2048);
        let result = SinusoidalPositionEncoding::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_rotary_config_creation() {
        let config = PositionEncodingConfig::rotary(512, 2048);
        assert_eq!(config.d_model, 512);
        assert_eq!(config.max_seq_len, 2048);
        assert!(matches!(
            config.encoding_type,
            PositionEncodingType::Rotary {
                base: 10000.0,
                scaling_factor: 1.0
            }
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_rotary_scaled_config() {
        let config = PositionEncodingConfig::rotary_scaled(512, 4096, 10000.0, 2.0);
        assert_eq!(config.max_seq_len, 4096);
        match config.encoding_type {
            PositionEncodingType::Rotary {
                base,
                scaling_factor,
            } => {
                assert!((base - 10000.0).abs() < 1e-10);
                assert!((scaling_factor - 2.0).abs() < 1e-10);
            }
            _ => panic!("Expected Rotary encoding type"),
        }
    }

    #[test]
    fn test_rotary_encoding_creation() {
        let config = PositionEncodingConfig::rotary(512, 2048);
        let encoding = RotaryPositionEncoding::new(config).unwrap();
        assert_eq!(encoding.config.d_model, 512);
        assert_eq!(encoding.base(), 10000.0);
        assert_eq!(encoding.scaling_factor(), 1.0);
    }

    #[test]
    fn test_rotary_graph_building() {
        let config = PositionEncodingConfig::rotary(512, 2048);
        let encoding = RotaryPositionEncoding::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");
        graph.add_tensor("cos_cached");
        graph.add_tensor("sin_cached");

        let outputs = encoding.build_encoding_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_rotary_requires_even_d_model() {
        let config = PositionEncodingConfig::rotary(513, 2048); // Odd d_model
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_alibi_config_creation() {
        let config = PositionEncodingConfig::alibi(512, 8, 2048);
        assert_eq!(config.d_model, 512);
        assert_eq!(config.max_seq_len, 2048);
        assert!(matches!(
            config.encoding_type,
            PositionEncodingType::Alibi {
                n_heads: 8,
                max_seq_len: 2048
            }
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_alibi_encoding_creation() {
        let config = PositionEncodingConfig::alibi(512, 8, 2048);
        let encoding = AlibiPositionEncoding::new(config).unwrap();
        assert_eq!(encoding.n_heads(), 8);
    }

    #[test]
    fn test_alibi_slopes_computation() {
        let config = PositionEncodingConfig::alibi(512, 8, 2048);
        let encoding = AlibiPositionEncoding::new(config).unwrap();
        let slopes = encoding.compute_slopes();

        assert_eq!(slopes.len(), 8);
        // Slopes should be monotonically decreasing
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1]);
        }
        // First slope should be largest
        assert!(slopes[0] < 1.0);
        assert!(slopes[0] > 0.0);
    }

    #[test]
    fn test_alibi_graph_building() {
        let config = PositionEncodingConfig::alibi(512, 8, 2048);
        let encoding = AlibiPositionEncoding::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("attention_scores");
        graph.add_tensor("alibi_slopes");
        graph.add_tensor("distance_matrix");

        let outputs = encoding.build_bias_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_alibi_invalid_zero_heads() {
        let config = PositionEncodingConfig::alibi(512, 0, 2048);
        assert!(config.validate().is_err());
    }
}
