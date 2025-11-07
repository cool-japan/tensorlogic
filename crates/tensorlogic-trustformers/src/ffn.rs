//! Feed-forward network layers as einsum operations.
//!
//! This module implements the position-wise feed-forward network (FFN)
//! used in transformer architectures.
//!
//! ## FFN Formula
//!
//! ```text
//! FFN(x) = activation(xW1 + b1)W2 + b2
//! ```
//!
//! Where:
//! - W1: [d_model, d_ff] - expansion projection
//! - W2: [d_ff, d_model] - contraction projection
//! - activation: typically GELU or ReLU
//!
//! ## Einsum Notation
//!
//! 1. First linear: `einsum("bsd,df->bsf", x, W1)`
//! 2. Activation: `activation(h1)`
//! 3. Second linear: `einsum("bsf,fd->bsd", h2, W2)`

use tensorlogic_ir::{EinsumGraph, EinsumNode};

use crate::config::FeedForwardConfig;
use crate::error::Result;

/// Feed-forward network component
#[derive(Clone, Debug)]
pub struct FeedForward {
    /// Configuration
    pub config: FeedForwardConfig,
}

impl FeedForward {
    /// Create a new feed-forward network
    pub fn new(config: FeedForwardConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build einsum graph for feed-forward network
    ///
    /// Input tensors:
    /// - 0: x (input) [batch, seq_len, d_model]
    /// - 1: W1 (first weight) [d_model, d_ff]
    /// - 2: b1 (first bias) [d_ff]
    /// - 3: W2 (second weight) [d_ff, d_model]
    /// - 4: b2 (second bias) [d_model]
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_ffn_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: First linear transformation
        // einsum("bsd,df->bsf", x, W1)
        let linear1_tensor = graph.add_tensor("ffn_linear1");
        let linear1_node = EinsumNode::new("bsd,df->bsf", vec![0, 1], vec![linear1_tensor]);
        graph.add_node(linear1_node)?;

        // Step 2: Add bias1
        let bias1_tensor = graph.add_tensor("ffn_bias1");
        let bias1_node = EinsumNode::elem_binary("add", linear1_tensor, 2, bias1_tensor);
        graph.add_node(bias1_node)?;

        // Step 3: Apply activation
        let activation_tensor = graph.add_tensor("ffn_activation");
        let activation_node =
            EinsumNode::elem_unary(&self.config.activation, bias1_tensor, activation_tensor);
        graph.add_node(activation_node)?;

        // Step 4: Second linear transformation
        // einsum("bsf,fd->bsd", h, W2)
        let linear2_tensor = graph.add_tensor("ffn_linear2");
        let linear2_node = EinsumNode::new(
            "bsf,fd->bsd",
            vec![activation_tensor, 3],
            vec![linear2_tensor],
        );
        graph.add_node(linear2_node)?;

        // Step 5: Add bias2
        let output_tensor = graph.add_tensor("ffn_output");
        let bias2_node = EinsumNode::elem_binary("add", linear2_tensor, 4, output_tensor);
        graph.add_node(bias2_node)?;

        Ok(vec![output_tensor])
    }

    /// Get the expansion ratio (d_ff / d_model)
    pub fn expansion_ratio(&self) -> f64 {
        self.config.d_ff as f64 / self.config.d_model as f64
    }

    /// Get activation function name
    pub fn activation(&self) -> &str {
        &self.config.activation
    }
}

/// Gated Linear Unit (GLU) variant for feed-forward networks
///
/// GLU uses two parallel projections with element-wise gating:
/// ```text
/// GLU(x) = (xW1) ⊙ σ(xW2)
/// ```
#[derive(Clone, Debug)]
pub struct GatedFeedForward {
    /// Configuration
    pub config: FeedForwardConfig,
}

impl GatedFeedForward {
    /// Create a new gated feed-forward network
    pub fn new(config: FeedForwardConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build einsum graph for gated feed-forward network
    ///
    /// Input tensors:
    /// - 0: x (input) [batch, seq_len, d_model]
    /// - 1: W_gate [d_model, d_ff]
    /// - 2: W_value [d_model, d_ff]
    /// - 3: W_out [d_ff, d_model]
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_glu_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Gate projection
        // einsum("bsd,df->bsf", x, W_gate)
        let gate_proj = graph.add_tensor("glu_gate_proj");
        let gate_node = EinsumNode::new("bsd,df->bsf", vec![0, 1], vec![gate_proj]);
        graph.add_node(gate_node)?;

        // Step 2: Apply gate activation (sigmoid or other)
        let gate_activated = graph.add_tensor("glu_gate_activated");
        let gate_act_node = EinsumNode::elem_unary("sigmoid", gate_proj, gate_activated);
        graph.add_node(gate_act_node)?;

        // Step 3: Value projection
        // einsum("bsd,df->bsf", x, W_value)
        let value_proj = graph.add_tensor("glu_value_proj");
        let value_node = EinsumNode::new("bsd,df->bsf", vec![0, 2], vec![value_proj]);
        graph.add_node(value_node)?;

        // Step 4: Apply activation to value
        let value_activated = graph.add_tensor("glu_value_activated");
        let value_act_node =
            EinsumNode::elem_unary(&self.config.activation, value_proj, value_activated);
        graph.add_node(value_act_node)?;

        // Step 5: Element-wise multiplication (gating)
        let gated = graph.add_tensor("glu_gated");
        let gate_mul_node = EinsumNode::elem_binary("mul", gate_activated, value_activated, gated);
        graph.add_node(gate_mul_node)?;

        // Step 6: Output projection
        // einsum("bsf,fd->bsd", gated, W_out)
        let output_tensor = graph.add_tensor("glu_output");
        let output_node = EinsumNode::new("bsf,fd->bsd", vec![gated, 3], vec![output_tensor]);
        graph.add_node(output_node)?;

        Ok(vec![output_tensor])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_creation() {
        let config = FeedForwardConfig::new(512, 2048);
        let ffn = FeedForward::new(config).unwrap();
        assert_eq!(ffn.config.d_model, 512);
        assert_eq!(ffn.config.d_ff, 2048);
        assert_eq!(ffn.activation(), "gelu");
    }

    #[test]
    fn test_ffn_expansion_ratio() {
        let config = FeedForwardConfig::new(512, 2048);
        let ffn = FeedForward::new(config).unwrap();
        let ratio = ffn.expansion_ratio();
        assert!((ratio - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ffn_with_custom_activation() {
        let config = FeedForwardConfig::new(512, 2048).with_activation("relu");
        let ffn = FeedForward::new(config).unwrap();
        assert_eq!(ffn.activation(), "relu");
    }

    #[test]
    fn test_ffn_graph_building() {
        let config = FeedForwardConfig::new(512, 2048);
        let ffn = FeedForward::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        // Add input tensors
        graph.add_tensor("x");
        graph.add_tensor("W1");
        graph.add_tensor("b1");
        graph.add_tensor("W2");
        graph.add_tensor("b2");

        let outputs = ffn.build_ffn_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_gated_ffn_creation() {
        let config = FeedForwardConfig::new(512, 2048);
        let glu = GatedFeedForward::new(config).unwrap();
        assert_eq!(glu.config.d_model, 512);
        assert_eq!(glu.config.d_ff, 2048);
    }

    #[test]
    fn test_gated_ffn_graph_building() {
        let config = FeedForwardConfig::new(512, 2048);
        let glu = GatedFeedForward::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        // Add input tensors
        graph.add_tensor("x");
        graph.add_tensor("W_gate");
        graph.add_tensor("W_value");
        graph.add_tensor("W_out");

        let outputs = glu.build_glu_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }
}
