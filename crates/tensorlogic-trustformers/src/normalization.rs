//! Layer normalization for transformer models.
//!
//! This module implements layer normalization, a critical component of
//! transformer architectures for stabilizing training.
//!
//! ## Layer Normalization Formula
//!
//! ```text
//! LN(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
//! ```
//!
//! Where:
//! - `μ` = mean over the feature dimension
//! - `σ²` = variance over the feature dimension
//! - `γ` = learnable scale parameter
//! - `β` = learnable shift parameter
//! - `ε` = small constant for numerical stability (default: 1e-5)
//!
//! ## Einsum Representation
//!
//! Layer norm can be expressed as a series of reductions and element-wise ops:
//! 1. Mean: `reduce_mean(x, axis=-1)` -> `einsum("bsd->bs", x) / d`
//! 2. Variance: `reduce_mean((x - μ)², axis=-1)`
//! 3. Normalize: `(x - μ) / √(σ² + ε)`
//! 4. Affine: `γ ⊙ normalized + β`

use serde::{Deserialize, Serialize};
use tensorlogic_ir::{EinsumGraph, EinsumNode};

use crate::error::{Result, TrustformerError};

/// Configuration for layer normalization
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerNormConfig {
    /// Normalized dimension (typically d_model)
    pub normalized_shape: usize,
    /// Small constant for numerical stability
    pub eps: f64,
    /// Whether to include learnable scale parameter (γ)
    pub elementwise_affine: bool,
}

impl LayerNormConfig {
    /// Create a new layer normalization configuration
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: true,
        }
    }

    /// Set epsilon for numerical stability
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set whether to use elementwise affine transformation
    pub fn with_elementwise_affine(mut self, elementwise_affine: bool) -> Self {
        self.elementwise_affine = elementwise_affine;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.normalized_shape == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "normalized_shape must be positive".to_string(),
            });
        }

        if self.eps <= 0.0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: format!("eps must be positive, got {}", self.eps),
            });
        }

        Ok(())
    }
}

/// Layer normalization component
#[derive(Clone, Debug)]
pub struct LayerNorm {
    /// Configuration
    pub config: LayerNormConfig,
}

impl LayerNorm {
    /// Create a new layer normalization component
    pub fn new(config: LayerNormConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build einsum graph for layer normalization
    ///
    /// Input tensors:
    /// - 0: x (input) [batch, seq_len, d_model]
    /// - 1: gamma (scale) [d_model] (if elementwise_affine)
    /// - 2: beta (shift) [d_model] (if elementwise_affine)
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model] (normalized)
    pub fn build_layernorm_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Compute mean over feature dimension
        // mean = reduce_mean(x, axis=-1, keepdims=True)
        let mean_tensor = graph.add_tensor("ln_mean");
        let mean_node = EinsumNode::reduce("mean", vec![2], 0, mean_tensor); // axis=-1 (d_model dimension)
        graph.add_node(mean_node)?;

        // Step 2: Center the input (x - mean)
        let centered_tensor = graph.add_tensor("ln_centered");
        let center_node = EinsumNode::elem_binary("sub", 0, mean_tensor, centered_tensor);
        graph.add_node(center_node)?;

        // Step 3: Compute variance
        // var = reduce_mean((x - mean)^2, axis=-1, keepdims=True)
        let squared_tensor = graph.add_tensor("ln_squared");
        let square_node =
            EinsumNode::elem_binary("mul", centered_tensor, centered_tensor, squared_tensor);
        graph.add_node(square_node)?;

        let var_tensor = graph.add_tensor("ln_var");
        let var_node = EinsumNode::reduce("mean", vec![2], squared_tensor, var_tensor);
        graph.add_node(var_node)?;

        // Step 4: Add epsilon for numerical stability
        let var_eps_tensor = graph.add_tensor("ln_var_eps");
        let eps_const_tensor = graph.add_tensor("eps_const");
        let eps_node = EinsumNode::elem_binary("add", var_tensor, eps_const_tensor, var_eps_tensor);
        graph.add_node(eps_node)?;

        // Step 5: Compute standard deviation (sqrt(var + eps))
        let std_tensor = graph.add_tensor("ln_std");
        let sqrt_node = EinsumNode::elem_unary("sqrt", var_eps_tensor, std_tensor);
        graph.add_node(sqrt_node)?;

        // Step 6: Normalize (x - mean) / std
        let normalized_tensor = graph.add_tensor("ln_normalized");
        let norm_node =
            EinsumNode::elem_binary("div", centered_tensor, std_tensor, normalized_tensor);
        graph.add_node(norm_node)?;

        // Step 7: Apply affine transformation if configured
        if self.config.elementwise_affine {
            // Scale: gamma * normalized
            let scaled_tensor = graph.add_tensor("ln_scaled");
            let scale_node = EinsumNode::elem_binary("mul", normalized_tensor, 1, scaled_tensor);
            graph.add_node(scale_node)?;

            // Shift: scaled + beta
            let output_tensor = graph.add_tensor("ln_output");
            let shift_node = EinsumNode::elem_binary("add", scaled_tensor, 2, output_tensor);
            graph.add_node(shift_node)?;

            Ok(vec![output_tensor])
        } else {
            Ok(vec![normalized_tensor])
        }
    }

    /// Get epsilon value
    pub fn eps(&self) -> f64 {
        self.config.eps
    }

    /// Check if using elementwise affine
    pub fn has_elementwise_affine(&self) -> bool {
        self.config.elementwise_affine
    }
}

/// RMS (Root Mean Square) normalization
///
/// A simplified variant of layer normalization that only computes RMS:
/// ```text
/// RMSNorm(x) = x / RMS(x) * γ
/// where RMS(x) = √(mean(x²) + ε)
/// ```
#[derive(Clone, Debug)]
pub struct RMSNorm {
    /// Configuration
    pub config: LayerNormConfig,
}

impl RMSNorm {
    /// Create a new RMS normalization component
    pub fn new(config: LayerNormConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build einsum graph for RMS normalization
    ///
    /// Input tensors:
    /// - 0: x (input) [batch, seq_len, d_model]
    /// - 1: gamma (scale) [d_model] (if elementwise_affine)
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model] (normalized)
    pub fn build_rmsnorm_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Compute x^2
        let squared_tensor = graph.add_tensor("rms_squared");
        let square_node = EinsumNode::elem_binary("mul", 0, 0, squared_tensor);
        graph.add_node(square_node)?;

        // Step 2: Compute mean(x^2)
        let mean_sq_tensor = graph.add_tensor("rms_mean_sq");
        let mean_node = EinsumNode::reduce("mean", vec![2], squared_tensor, mean_sq_tensor);
        graph.add_node(mean_node)?;

        // Step 3: Add epsilon
        let mean_sq_eps_tensor = graph.add_tensor("rms_mean_sq_eps");
        let eps_const_tensor = graph.add_tensor("eps_const");
        let eps_node =
            EinsumNode::elem_binary("add", mean_sq_tensor, eps_const_tensor, mean_sq_eps_tensor);
        graph.add_node(eps_node)?;

        // Step 4: Compute RMS = sqrt(mean(x^2) + eps)
        let rms_tensor = graph.add_tensor("rms");
        let sqrt_node = EinsumNode::elem_unary("sqrt", mean_sq_eps_tensor, rms_tensor);
        graph.add_node(sqrt_node)?;

        // Step 5: Normalize x / RMS
        let normalized_tensor = graph.add_tensor("rms_normalized");
        let norm_node = EinsumNode::elem_binary("div", 0, rms_tensor, normalized_tensor);
        graph.add_node(norm_node)?;

        // Step 6: Apply scale if configured
        if self.config.elementwise_affine {
            let output_tensor = graph.add_tensor("rms_output");
            let scale_node = EinsumNode::elem_binary("mul", normalized_tensor, 1, output_tensor);
            graph.add_node(scale_node)?;
            Ok(vec![output_tensor])
        } else {
            Ok(vec![normalized_tensor])
        }
    }

    /// Get epsilon value
    pub fn eps(&self) -> f64 {
        self.config.eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_config_creation() {
        let config = LayerNormConfig::new(512);
        assert_eq!(config.normalized_shape, 512);
        assert!((config.eps - 1e-5).abs() < 1e-10);
        assert!(config.elementwise_affine);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_layernorm_config_with_eps() {
        let config = LayerNormConfig::new(512).with_eps(1e-6);
        assert!((config.eps - 1e-6).abs() < 1e-10);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_layernorm_config_without_affine() {
        let config = LayerNormConfig::new(512).with_elementwise_affine(false);
        assert!(!config.elementwise_affine);
    }

    #[test]
    fn test_layernorm_creation() {
        let config = LayerNormConfig::new(512);
        let ln = LayerNorm::new(config).unwrap();
        assert_eq!(ln.config.normalized_shape, 512);
        assert!(ln.has_elementwise_affine());
    }

    #[test]
    fn test_layernorm_graph_building_with_affine() {
        let config = LayerNormConfig::new(512);
        let ln = LayerNorm::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");
        graph.add_tensor("gamma");
        graph.add_tensor("beta");

        let outputs = ln.build_layernorm_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_layernorm_graph_building_without_affine() {
        let config = LayerNormConfig::new(512).with_elementwise_affine(false);
        let ln = LayerNorm::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");

        let outputs = ln.build_layernorm_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_rmsnorm_creation() {
        let config = LayerNormConfig::new(512);
        let rms = RMSNorm::new(config).unwrap();
        assert_eq!(rms.config.normalized_shape, 512);
    }

    #[test]
    fn test_rmsnorm_graph_building() {
        let config = LayerNormConfig::new(512);
        let rms = RMSNorm::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");
        graph.add_tensor("gamma");

        let outputs = rms.build_rmsnorm_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_invalid_config_zero_shape() {
        let config = LayerNormConfig::new(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_config_negative_eps() {
        let config = LayerNormConfig::new(512).with_eps(-1e-5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_layernorm_eps() {
        let config = LayerNormConfig::new(512).with_eps(1e-6);
        let ln = LayerNorm::new(config).unwrap();
        assert!((ln.eps() - 1e-6).abs() < 1e-10);
    }
}
