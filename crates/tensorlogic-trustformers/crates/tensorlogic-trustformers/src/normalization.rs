//! Layer normalization implementations.

use crate::config::{NormalizationConfig, NormalizationType};
use crate::error::TrustformersResult;
use tensorlogic_ir::{EinsumGraph, OpType, TensorShape};

/// Layer normalization builder
pub struct LayerNorm {
    config: NormalizationConfig,
}

impl LayerNorm {
    /// Create a new LayerNorm with the given configuration
    pub fn new(config: NormalizationConfig) -> TrustformersResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build the normalization einsum graph
    pub fn build(&self) -> TrustformersResult<EinsumGraph> {
        match self.config.norm_type {
            NormalizationType::LayerNorm => self.layer_norm(),
            NormalizationType::RMSNorm => self.rms_norm(),
        }
    }

    /// Standard Layer Normalization
    ///
    /// y = γ * (x - mean) / sqrt(var + ε) + β
    fn layer_norm(&self) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();

        // Input: [batch, seq_len, hidden_size]
        let input_shape = TensorShape::Dynamic;
        let input_node = graph.add_node(
            OpType::Placeholder {
                name: "input".to_string(),
                shape: input_shape,
            },
            vec![],
        );

        // Compute mean: reduce over hidden dimension
        let mean_node = graph.add_node(
            OpType::Reduce {
                op: "mean".to_string(),
                axes: vec![2],
            },
            vec![input_node],
        );

        // Compute x - mean
        let centered_node = graph.add_node(
            OpType::ElemBinary {
                op: "sub".to_string(),
            },
            vec![input_node, mean_node],
        );

        // Compute variance
        let squared_node = graph.add_node(
            OpType::ElemBinary {
                op: "mul".to_string(),
            },
            vec![centered_node, centered_node],
        );

        let var_node = graph.add_node(
            OpType::Reduce {
                op: "mean".to_string(),
                axes: vec![2],
            },
            vec![squared_node],
        );

        // Add epsilon for stability
        let epsilon_shape = TensorShape::Fixed(vec![1]);
        let epsilon_node = graph.add_node(
            OpType::Placeholder {
                name: "epsilon".to_string(),
                shape: epsilon_shape,
            },
            vec![],
        );

        let var_eps_node = graph.add_node(
            OpType::ElemBinary {
                op: "add".to_string(),
            },
            vec![var_node, epsilon_node],
        );

        // Compute 1 / sqrt(var + eps)
        let rstd_node = graph.add_node(
            OpType::ElemUnary {
                op: "rsqrt".to_string(),
            },
            vec![var_eps_node],
        );

        // Normalize: (x - mean) * rstd
        let normalized_node = graph.add_node(
            OpType::ElemBinary {
                op: "mul".to_string(),
            },
            vec![centered_node, rstd_node],
        );

        if self.config.elementwise_affine {
            // Apply scale (gamma)
            let gamma_shape = TensorShape::Fixed(vec![self.config.hidden_size]);
            let gamma_node = graph.add_node(
                OpType::Placeholder {
                    name: "gamma".to_string(),
                    shape: gamma_shape,
                },
                vec![],
            );

            let scaled_node = graph.add_node(
                OpType::ElemBinary {
                    op: "mul".to_string(),
                },
                vec![normalized_node, gamma_node],
            );

            // Apply bias (beta)
            let beta_shape = TensorShape::Fixed(vec![self.config.hidden_size]);
            let beta_node = graph.add_node(
                OpType::Placeholder {
                    name: "beta".to_string(),
                    shape: beta_shape,
                },
                vec![],
            );

            let output_node = graph.add_node(
                OpType::ElemBinary {
                    op: "add".to_string(),
                },
                vec![scaled_node, beta_node],
            );

            graph.set_output(output_node);
        } else {
            graph.set_output(normalized_node);
        }

        Ok(graph)
    }

    /// Root Mean Square Layer Normalization (Used in LLaMA)
    ///
    /// y = x / RMS(x) * γ, where RMS(x) = sqrt(mean(x²) + ε)
    fn rms_norm(&self) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();

        // Input: [batch, seq_len, hidden_size]
        let input_shape = TensorShape::Dynamic;
        let input_node = graph.add_node(
            OpType::Placeholder {
                name: "input".to_string(),
                shape: input_shape,
            },
            vec![],
        );

        // Compute x²
        let squared_node = graph.add_node(
            OpType::ElemBinary {
                op: "mul".to_string(),
            },
            vec![input_node, input_node],
        );

        // Compute mean(x²)
        let mean_sq_node = graph.add_node(
            OpType::Reduce {
                op: "mean".to_string(),
                axes: vec![2],
            },
            vec![squared_node],
        );

        // Add epsilon
        let epsilon_shape = TensorShape::Fixed(vec![1]);
        let epsilon_node = graph.add_node(
            OpType::Placeholder {
                name: "epsilon".to_string(),
                shape: epsilon_shape,
            },
            vec![],
        );

        let mean_sq_eps_node = graph.add_node(
            OpType::ElemBinary {
                op: "add".to_string(),
            },
            vec![mean_sq_node, epsilon_node],
        );

        // Compute 1 / sqrt(mean(x²) + ε)
        let rrms_node = graph.add_node(
            OpType::ElemUnary {
                op: "rsqrt".to_string(),
            },
            vec![mean_sq_eps_node],
        );

        // Normalize: x * rrms
        let normalized_node = graph.add_node(
            OpType::ElemBinary {
                op: "mul".to_string(),
            },
            vec![input_node, rrms_node],
        );

        if self.config.elementwise_affine {
            // Apply scale (gamma)
            let gamma_shape = TensorShape::Fixed(vec![self.config.hidden_size]);
            let gamma_node = graph.add_node(
                OpType::Placeholder {
                    name: "gamma".to_string(),
                    shape: gamma_shape,
                },
                vec![],
            );

            let output_node = graph.add_node(
                OpType::ElemBinary {
                    op: "mul".to_string(),
                },
                vec![normalized_node, gamma_node],
            );

            graph.set_output(output_node);
        } else {
            graph.set_output(normalized_node);
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let config = NormalizationConfig {
            hidden_size: 512,
            norm_type: NormalizationType::LayerNorm,
            epsilon: 1e-5,
            elementwise_affine: true,
        };
        let layer_norm = LayerNorm::new(config);
        assert!(layer_norm.is_ok());
    }

    #[test]
    fn test_layer_norm_build() {
        let config = NormalizationConfig {
            hidden_size: 512,
            norm_type: NormalizationType::LayerNorm,
            epsilon: 1e-5,
            elementwise_affine: true,
        };
        let layer_norm = LayerNorm::new(config).unwrap();
        let graph = layer_norm.build();
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        assert!(graph.output().is_some());
    }

    #[test]
    fn test_layer_norm_no_affine() {
        let config = NormalizationConfig {
            hidden_size: 512,
            norm_type: NormalizationType::LayerNorm,
            epsilon: 1e-5,
            elementwise_affine: false,
        };
        let layer_norm = LayerNorm::new(config).unwrap();
        let graph = layer_norm.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_rms_norm_build() {
        let config = NormalizationConfig {
            hidden_size: 512,
            norm_type: NormalizationType::RMSNorm,
            epsilon: 1e-5,
            elementwise_affine: true,
        };
        let layer_norm = LayerNorm::new(config).unwrap();
        let graph = layer_norm.build();
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        assert!(graph.output().is_some());
    }

    #[test]
    fn test_rms_norm_no_affine() {
        let config = NormalizationConfig {
            hidden_size: 512,
            norm_type: NormalizationType::RMSNorm,
            epsilon: 1e-5,
            elementwise_affine: false,
        };
        let layer_norm = LayerNorm::new(config).unwrap();
        let graph = layer_norm.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_different_hidden_sizes() {
        for hidden_size in vec![128, 256, 512, 768, 1024, 2048] {
            let config = NormalizationConfig {
                hidden_size,
                norm_type: NormalizationType::LayerNorm,
                epsilon: 1e-5,
                elementwise_affine: true,
            };
            let layer_norm = LayerNorm::new(config).unwrap();
            let graph = layer_norm.build();
            assert!(graph.is_ok());
        }
    }
}
