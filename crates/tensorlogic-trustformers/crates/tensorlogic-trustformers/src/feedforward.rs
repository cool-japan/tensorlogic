//! Feed-forward network implementations.

use crate::config::{FeedForwardConfig, FeedForwardType};
use crate::error::TrustformersResult;
use tensorlogic_ir::{EinsumGraph, OpType, TensorShape};

/// Feed-forward network builder
pub struct FeedForward {
    config: FeedForwardConfig,
}

impl FeedForward {
    /// Create a new FeedForward with the given configuration
    pub fn new(config: FeedForwardConfig) -> TrustformersResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build the feed-forward einsum graph
    pub fn build(&self) -> TrustformersResult<EinsumGraph> {
        match self.config.ffn_type {
            FeedForwardType::Standard => self.standard_ffn(),
            FeedForwardType::GLU => self.glu_ffn(),
            FeedForwardType::GeGLU => self.geglu_ffn(),
            FeedForwardType::SwiGLU => self.swiglu_ffn(),
        }
    }

    /// Standard FFN: FFN(x) = activation(xW1 + b1)W2 + b2
    fn standard_ffn(&self) -> TrustformersResult<EinsumGraph> {
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

        // First linear layer: [hidden_size, intermediate_size]
        let w1_shape = TensorShape::Fixed(vec![self.config.hidden_size, self.config.intermediate_size]);
        let w1_node = graph.add_node(
            OpType::Placeholder {
                name: "w1".to_string(),
                shape: w1_shape,
            },
            vec![],
        );

        let linear1_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, w1_node],
        );

        let mut hidden_node = linear1_node;
        if self.config.use_bias {
            let b1_shape = TensorShape::Fixed(vec![self.config.intermediate_size]);
            let b1_node = graph.add_node(
                OpType::Placeholder {
                    name: "b1".to_string(),
                    shape: b1_shape,
                },
                vec![],
            );
            hidden_node = graph.add_node(
                OpType::ElemBinary {
                    op: "add".to_string(),
                },
                vec![linear1_node, b1_node],
            );
        }

        // Activation (GELU)
        let activated_node = graph.add_node(
            OpType::ElemUnary {
                op: "gelu".to_string(),
            },
            vec![hidden_node],
        );

        // Second linear layer: [intermediate_size, hidden_size]
        let w2_shape = TensorShape::Fixed(vec![self.config.intermediate_size, self.config.hidden_size]);
        let w2_node = graph.add_node(
            OpType::Placeholder {
                name: "w2".to_string(),
                shape: w2_shape,
            },
            vec![],
        );

        let linear2_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![activated_node, w2_node],
        );

        let output_node = if self.config.use_bias {
            let b2_shape = TensorShape::Fixed(vec![self.config.hidden_size]);
            let b2_node = graph.add_node(
                OpType::Placeholder {
                    name: "b2".to_string(),
                    shape: b2_shape,
                },
                vec![],
            );
            graph.add_node(
                OpType::ElemBinary {
                    op: "add".to_string(),
                },
                vec![linear2_node, b2_node],
            )
        } else {
            linear2_node
        };

        graph.set_output(output_node);
        Ok(graph)
    }

    /// GLU: FFN(x) = (xW1) ⊙ σ(xW2)
    fn glu_ffn(&self) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();

        let input_shape = TensorShape::Dynamic;
        let input_node = graph.add_node(
            OpType::Placeholder {
                name: "input".to_string(),
                shape: input_shape,
            },
            vec![],
        );

        // Value projection
        let w_v_shape = TensorShape::Fixed(vec![self.config.hidden_size, self.config.intermediate_size]);
        let w_v_node = graph.add_node(
            OpType::Placeholder {
                name: "w_value".to_string(),
                shape: w_v_shape,
            },
            vec![],
        );

        let value_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, w_v_node],
        );

        // Gate projection
        let w_g_shape = TensorShape::Fixed(vec![self.config.hidden_size, self.config.intermediate_size]);
        let w_g_node = graph.add_node(
            OpType::Placeholder {
                name: "w_gate".to_string(),
                shape: w_g_shape,
            },
            vec![],
        );

        let gate_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, w_g_node],
        );

        // Apply sigmoid to gate
        let gate_act_node = graph.add_node(
            OpType::ElemUnary {
                op: "sigmoid".to_string(),
            },
            vec![gate_node],
        );

        // Element-wise multiplication
        let gated_node = graph.add_node(
            OpType::ElemBinary {
                op: "mul".to_string(),
            },
            vec![value_node, gate_act_node],
        );

        // Output projection
        let w_o_shape = TensorShape::Fixed(vec![self.config.intermediate_size, self.config.hidden_size]);
        let w_o_node = graph.add_node(
            OpType::Placeholder {
                name: "w_output".to_string(),
                shape: w_o_shape,
            },
            vec![],
        );

        let output_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![gated_node, w_o_node],
        );

        graph.set_output(output_node);
        Ok(graph)
    }

    /// GeGLU: FFN(x) = GELU(xW1) ⊙ (xW2) (Used in T5)
    fn geglu_ffn(&self) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();

        let input_shape = TensorShape::Dynamic;
        let input_node = graph.add_node(
            OpType::Placeholder {
                name: "input".to_string(),
                shape: input_shape,
            },
            vec![],
        );

        // Value projection
        let w_v_shape = TensorShape::Fixed(vec![self.config.hidden_size, self.config.intermediate_size]);
        let w_v_node = graph.add_node(
            OpType::Placeholder {
                name: "w_value".to_string(),
                shape: w_v_shape,
            },
            vec![],
        );

        let value_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, w_v_node],
        );

        // Gate projection
        let w_g_shape = TensorShape::Fixed(vec![self.config.hidden_size, self.config.intermediate_size]);
        let w_g_node = graph.add_node(
            OpType::Placeholder {
                name: "w_gate".to_string(),
                shape: w_g_shape,
            },
            vec![],
        );

        let gate_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, w_g_node],
        );

        // Apply GELU to gate
        let gate_act_node = graph.add_node(
            OpType::ElemUnary {
                op: "gelu".to_string(),
            },
            vec![gate_node],
        );

        // Element-wise multiplication
        let gated_node = graph.add_node(
            OpType::ElemBinary {
                op: "mul".to_string(),
            },
            vec![value_node, gate_act_node],
        );

        // Output projection
        let w_o_shape = TensorShape::Fixed(vec![self.config.intermediate_size, self.config.hidden_size]);
        let w_o_node = graph.add_node(
            OpType::Placeholder {
                name: "w_output".to_string(),
                shape: w_o_shape,
            },
            vec![],
        );

        let output_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![gated_node, w_o_node],
        );

        graph.set_output(output_node);
        Ok(graph)
    }

    /// SwiGLU: FFN(x) = Swish(xW1) ⊙ (xW2) (Used in LLaMA)
    fn swiglu_ffn(&self) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();

        let input_shape = TensorShape::Dynamic;
        let input_node = graph.add_node(
            OpType::Placeholder {
                name: "input".to_string(),
                shape: input_shape,
            },
            vec![],
        );

        // Value projection
        let w_v_shape = TensorShape::Fixed(vec![self.config.hidden_size, self.config.intermediate_size]);
        let w_v_node = graph.add_node(
            OpType::Placeholder {
                name: "w_value".to_string(),
                shape: w_v_shape,
            },
            vec![],
        );

        let value_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, w_v_node],
        );

        // Gate projection
        let w_g_shape = TensorShape::Fixed(vec![self.config.hidden_size, self.config.intermediate_size]);
        let w_g_node = graph.add_node(
            OpType::Placeholder {
                name: "w_gate".to_string(),
                shape: w_g_shape,
            },
            vec![],
        );

        let gate_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![input_node, w_g_node],
        );

        // Apply Swish (SiLU) to gate
        let gate_act_node = graph.add_node(
            OpType::ElemUnary {
                op: "swish".to_string(),
            },
            vec![gate_node],
        );

        // Element-wise multiplication
        let gated_node = graph.add_node(
            OpType::ElemBinary {
                op: "mul".to_string(),
            },
            vec![value_node, gate_act_node],
        );

        // Output projection
        let w_o_shape = TensorShape::Fixed(vec![self.config.intermediate_size, self.config.hidden_size]);
        let w_o_node = graph.add_node(
            OpType::Placeholder {
                name: "w_output".to_string(),
                shape: w_o_shape,
            },
            vec![],
        );

        let output_node = graph.add_node(
            OpType::Einsum {
                spec: "...i,ij->...j".to_string(),
            },
            vec![gated_node, w_o_node],
        );

        graph.set_output(output_node);
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_creation() {
        let config = FeedForwardConfig::default();
        let ffn = FeedForward::new(config);
        assert!(ffn.is_ok());
    }

    #[test]
    fn test_standard_ffn_build() {
        let config = FeedForwardConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            ffn_type: FeedForwardType::Standard,
            dropout: 0.1,
            use_bias: true,
        };
        let ffn = FeedForward::new(config).unwrap();
        let graph = ffn.build();
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        assert!(graph.output().is_some());
    }

    #[test]
    fn test_glu_ffn_build() {
        let config = FeedForwardConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            ffn_type: FeedForwardType::GLU,
            dropout: 0.1,
            use_bias: false,
        };
        let ffn = FeedForward::new(config).unwrap();
        let graph = ffn.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_geglu_ffn_build() {
        let config = FeedForwardConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            ffn_type: FeedForwardType::GeGLU,
            dropout: 0.1,
            use_bias: false,
        };
        let ffn = FeedForward::new(config).unwrap();
        let graph = ffn.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_swiglu_ffn_build() {
        let config = FeedForwardConfig {
            hidden_size: 512,
            intermediate_size: 2048,
            ffn_type: FeedForwardType::SwiGLU,
            dropout: 0.1,
            use_bias: false,
        };
        let ffn = FeedForward::new(config).unwrap();
        let graph = ffn.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_different_sizes() {
        for (hidden_size, intermediate_size) in vec![(256, 1024), (512, 2048), (768, 3072), (1024, 4096)] {
            let config = FeedForwardConfig {
                hidden_size,
                intermediate_size,
                ffn_type: FeedForwardType::Standard,
                dropout: 0.1,
                use_bias: true,
            };
            let ffn = FeedForward::new(config).unwrap();
            let graph = ffn.build();
            assert!(graph.is_ok());
        }
    }
}
