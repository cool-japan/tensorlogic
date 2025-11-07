//! Position encoding implementations for transformers.
//!
//! This module provides various position encoding strategies used in modern transformer models:
//! - Sinusoidal encodings (Vaswani et al., 2017)
//! - Learned embeddings
//! - Relative position encodings (Shaw et al., 2018)
//! - Rotary Position Embeddings (RoPE) - Used in LLaMA, GPT-NeoX
//! - Attention with Linear Biases (ALiBi) - Used in BLOOM

use crate::config::{PositionEncodingConfig, PositionEncodingType};
use crate::error::{TrustformersError, TrustformersResult};
use tensorlogic_ir::{EinsumGraph, NodeId, OpType, TensorShape};

/// Position encoding generator
pub struct PositionEncoder {
    config: PositionEncodingConfig,
}

impl PositionEncoder {
    /// Create a new position encoder with the given configuration
    pub fn new(config: PositionEncodingConfig) -> TrustformersResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Generate position encoding einsum graph
    pub fn encode(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        if seq_len > self.config.max_seq_len {
            return Err(TrustformersError::sequence_too_long(
                seq_len,
                self.config.max_seq_len,
            ));
        }

        match self.config.encoding_type {
            PositionEncodingType::Sinusoidal => self.sinusoidal_encoding(seq_len),
            PositionEncodingType::Learned => self.learned_encoding(seq_len),
            PositionEncodingType::Relative => self.relative_encoding(seq_len),
            PositionEncodingType::RoPE => self.rope_encoding(seq_len),
            PositionEncodingType::ALiBi => self.alibi_encoding(seq_len),
            PositionEncodingType::None => self.no_encoding(seq_len),
        }
    }

    /// Generate sinusoidal position encodings
    ///
    /// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    fn sinusoidal_encoding(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        let hidden_size = self.config.hidden_size;

        // Create position indices: [seq_len]
        let positions_shape = TensorShape::Fixed(vec![seq_len]);
        let positions_node = graph.add_node(
            OpType::Placeholder {
                name: "positions".to_string(),
                shape: positions_shape.clone(),
            },
            vec![],
        );

        // Create dimension indices: [hidden_size]
        let dim_shape = TensorShape::Fixed(vec![hidden_size]);
        let dim_node = graph.add_node(
            OpType::Placeholder {
                name: "dimensions".to_string(),
                shape: dim_shape.clone(),
            },
            vec![],
        );

        // Compute 10000^(2i/d_model) for each dimension
        let freq_node = graph.add_node(
            OpType::ElemUnary {
                op: "exp".to_string(),
            },
            vec![dim_node],
        );

        // Compute pos / freq for each position and dimension: [seq_len, hidden_size]
        let angles_node = graph.add_node(
            OpType::Einsum {
                spec: "i,j->ij".to_string(),
            },
            vec![positions_node, freq_node],
        );

        // Apply sin/cos alternately (simplified as single operation in graph)
        let encoding_node = graph.add_node(
            OpType::ElemUnary {
                op: "sincos".to_string(),
            },
            vec![angles_node],
        );

        graph.set_output(encoding_node);
        Ok(graph)
    }

    /// Generate learned position embeddings
    fn learned_encoding(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();

        // Create embedding table: [max_seq_len, hidden_size]
        let embed_shape = TensorShape::Fixed(vec![self.config.max_seq_len, self.config.hidden_size]);
        let embed_node = graph.add_node(
            OpType::Placeholder {
                name: "position_embeddings".to_string(),
                shape: embed_shape,
            },
            vec![],
        );

        // Select positions [0, seq_len)
        let positions_shape = TensorShape::Fixed(vec![seq_len]);
        let positions_node = graph.add_node(
            OpType::Placeholder {
                name: "position_indices".to_string(),
                shape: positions_shape,
            },
            vec![],
        );

        // Gather embeddings: [seq_len, hidden_size]
        let encoding_node = graph.add_node(
            OpType::Einsum {
                spec: "ij,i->ij".to_string(),
            },
            vec![embed_node, positions_node],
        );

        graph.set_output(encoding_node);
        Ok(graph)
    }

    /// Generate relative position encodings
    ///
    /// Computes pairwise position biases for attention
    fn relative_encoding(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();

        // Create relative position bias table: [2*max_seq_len-1, num_heads]
        let num_buckets = 2 * self.config.max_seq_len - 1;
        let bias_shape = TensorShape::Fixed(vec![num_buckets, self.config.hidden_size]);
        let bias_table_node = graph.add_node(
            OpType::Placeholder {
                name: "relative_bias_table".to_string(),
                shape: bias_shape,
            },
            vec![],
        );

        // Compute relative positions: [seq_len, seq_len]
        let rel_pos_shape = TensorShape::Fixed(vec![seq_len, seq_len]);
        let rel_pos_node = graph.add_node(
            OpType::Placeholder {
                name: "relative_positions".to_string(),
                shape: rel_pos_shape,
            },
            vec![],
        );

        // Gather biases: [seq_len, seq_len, hidden_size]
        let encoding_node = graph.add_node(
            OpType::Einsum {
                spec: "ij,jk->ijk".to_string(),
            },
            vec![rel_pos_node, bias_table_node],
        );

        graph.set_output(encoding_node);
        Ok(graph)
    }

    /// Generate Rotary Position Embeddings (RoPE)
    ///
    /// Used in LLaMA, GPT-NeoX. Applies rotation in complex plane to each pair of dimensions.
    fn rope_encoding(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        let hidden_size = self.config.hidden_size;

        // Create position indices: [seq_len]
        let positions_shape = TensorShape::Fixed(vec![seq_len]);
        let positions_node = graph.add_node(
            OpType::Placeholder {
                name: "positions".to_string(),
                shape: positions_shape,
            },
            vec![],
        );

        // Create frequency bands: [hidden_size // 2]
        let freq_shape = TensorShape::Fixed(vec![hidden_size / 2]);
        let freq_node = graph.add_node(
            OpType::Placeholder {
                name: "rope_frequencies".to_string(),
                shape: freq_shape,
            },
            vec![],
        );

        // Compute angles: [seq_len, hidden_size // 2]
        let angles_node = graph.add_node(
            OpType::Einsum {
                spec: "i,j->ij".to_string(),
            },
            vec![positions_node, freq_node],
        );

        // Compute rotation matrix (cos and sin): [seq_len, hidden_size]
        let rotation_node = graph.add_node(
            OpType::ElemUnary {
                op: "rope_rotation".to_string(),
            },
            vec![angles_node],
        );

        graph.set_output(rotation_node);
        Ok(graph)
    }

    /// Generate Attention with Linear Biases (ALiBi)
    ///
    /// Used in BLOOM. Adds linearly decreasing biases to attention scores based on distance.
    fn alibi_encoding(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();

        // Create distance matrix: [seq_len, seq_len]
        let dist_shape = TensorShape::Fixed(vec![seq_len, seq_len]);
        let dist_node = graph.add_node(
            OpType::Placeholder {
                name: "distance_matrix".to_string(),
                shape: dist_shape,
            },
            vec![],
        );

        // Create slope per head: [num_heads]
        let num_heads = self.config.hidden_size; // Simplified; in practice, this would be separate
        let slopes_shape = TensorShape::Fixed(vec![num_heads]);
        let slopes_node = graph.add_node(
            OpType::Placeholder {
                name: "alibi_slopes".to_string(),
                shape: slopes_shape,
            },
            vec![],
        );

        // Apply slopes to distances: [num_heads, seq_len, seq_len]
        let bias_node = graph.add_node(
            OpType::Einsum {
                spec: "ij,k->kij".to_string(),
            },
            vec![dist_node, slopes_node],
        );

        graph.set_output(bias_node);
        Ok(graph)
    }

    /// No position encoding (returns identity/zero)
    fn no_encoding(&self, seq_len: usize) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();

        // Create zero tensor: [seq_len, hidden_size]
        let zero_shape = TensorShape::Fixed(vec![seq_len, self.config.hidden_size]);
        let zero_node = graph.add_node(
            OpType::Placeholder {
                name: "zero_encoding".to_string(),
                shape: zero_shape,
            },
            vec![],
        );

        graph.set_output(zero_node);
        Ok(graph)
    }

    /// Get the output shape for the given sequence length
    pub fn output_shape(&self, seq_len: usize) -> TensorShape {
        match self.config.encoding_type {
            PositionEncodingType::Relative => {
                TensorShape::Fixed(vec![seq_len, seq_len, self.config.hidden_size])
            }
            PositionEncodingType::ALiBi => {
                TensorShape::Fixed(vec![self.config.hidden_size, seq_len, seq_len])
            }
            _ => TensorShape::Fixed(vec![seq_len, self.config.hidden_size]),
        }
    }
}

/// Helper function to compute RoPE frequencies
pub fn compute_rope_frequencies(hidden_size: usize, base: f64) -> Vec<f64> {
    let half_dim = hidden_size / 2;
    (0..half_dim)
        .map(|i| {
            let exponent = 2.0 * (i as f64) / (hidden_size as f64);
            1.0 / base.powf(exponent)
        })
        .collect()
}

/// Helper function to compute ALiBi slopes
pub fn compute_alibi_slopes(num_heads: usize) -> Vec<f64> {
    let ratio = 2.0_f64.powf(-(8.0 / num_heads as f64));
    (0..num_heads)
        .map(|i| ratio.powi(i as i32 + 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_encoder_creation() {
        let config = PositionEncodingConfig::default();
        let encoder = PositionEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_sinusoidal_encoding() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::Sinusoidal,
            max_seq_len: 512,
            hidden_size: 128,
            dropout: 0.1,
            base: 10000.0,
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let graph = encoder.encode(32);
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        assert!(graph.output().is_some());
    }

    #[test]
    fn test_learned_encoding() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::Learned,
            max_seq_len: 512,
            hidden_size: 128,
            dropout: 0.1,
            base: 10000.0,
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let graph = encoder.encode(32);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_relative_encoding() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::Relative,
            max_seq_len: 512,
            hidden_size: 128,
            dropout: 0.1,
            base: 10000.0,
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let graph = encoder.encode(32);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_rope_encoding() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::RoPE,
            max_seq_len: 512,
            hidden_size: 128,
            dropout: 0.1,
            base: 10000.0,
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let graph = encoder.encode(32);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_alibi_encoding() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::ALiBi,
            max_seq_len: 512,
            hidden_size: 8,
            dropout: 0.1,
            base: 10000.0,
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let graph = encoder.encode(32);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_no_encoding() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::None,
            max_seq_len: 512,
            hidden_size: 128,
            dropout: 0.1,
            base: 10000.0,
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let graph = encoder.encode(32);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_sequence_too_long() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::Sinusoidal,
            max_seq_len: 128,
            hidden_size: 128,
            dropout: 0.1,
            base: 10000.0,
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let result = encoder.encode(256);
        assert!(result.is_err());
    }

    #[test]
    fn test_output_shape_sinusoidal() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::Sinusoidal,
            hidden_size: 128,
            ..Default::default()
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let shape = encoder.output_shape(32);
        match shape {
            TensorShape::Fixed(dims) => {
                assert_eq!(dims, vec![32, 128]);
            }
            _ => panic!("Expected fixed shape"),
        }
    }

    #[test]
    fn test_output_shape_relative() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::Relative,
            hidden_size: 128,
            ..Default::default()
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let shape = encoder.output_shape(32);
        match shape {
            TensorShape::Fixed(dims) => {
                assert_eq!(dims, vec![32, 32, 128]);
            }
            _ => panic!("Expected fixed shape"),
        }
    }

    #[test]
    fn test_output_shape_alibi() {
        let config = PositionEncodingConfig {
            encoding_type: PositionEncodingType::ALiBi,
            hidden_size: 8,
            ..Default::default()
        };
        let encoder = PositionEncoder::new(config).unwrap();
        let shape = encoder.output_shape(32);
        match shape {
            TensorShape::Fixed(dims) => {
                assert_eq!(dims, vec![8, 32, 32]);
            }
            _ => panic!("Expected fixed shape"),
        }
    }

    #[test]
    fn test_compute_rope_frequencies() {
        let freqs = compute_rope_frequencies(128, 10000.0);
        assert_eq!(freqs.len(), 64);
        assert!(freqs[0] > freqs[63]); // Frequencies decrease
    }

    #[test]
    fn test_compute_alibi_slopes() {
        let slopes = compute_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);
        assert!(slopes[0] < slopes[7]); // Slopes increase
    }
}
