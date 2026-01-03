//! Mixture-of-Experts (MoE) layers for sparse transformer models
//!
//! This module implements MoE layers that allow for conditional computation,
//! where different "expert" networks are selected for different inputs.
//!
//! ## Architecture
//!
//! MoE layers consist of:
//! 1. **Router/Gating Network**: Determines which experts to activate
//!    - Computes routing weights: `einsum("bd,de->be", x, W_router)`
//!    - Applies softmax or top-k selection
//! 2. **Expert Networks**: Multiple FFN layers (specialists)
//!    - Each expert is a standard feed-forward network
//!    - Experts are activated sparsely (only top-k per token)
//! 3. **Expert Combination**: Weighted sum of expert outputs
//!    - `output = Σ(gate_i * expert_i(x))`
//!
//! ## Benefits
//!
//! - **Increased Capacity**: More parameters without proportional compute increase
//! - **Specialization**: Experts can specialize in different patterns
//! - **Efficiency**: Sparse activation reduces computation
//!
//! ## Example
//!
//! ```no_run
//! use tensorlogic_trustformers::moe::{MoeConfig, MoeLayer, RouterType};
//! use tensorlogic_ir::EinsumGraph;
//!
//! // Create MoE with 8 experts, top-2 routing
//! let config = MoeConfig::new(512, 2048, 8, 2).unwrap();
//! let moe = MoeLayer::new(config).unwrap();
//!
//! let mut graph = EinsumGraph::new();
//! // Add required tensors...
//! let outputs = moe.build_moe_graph(&mut graph).unwrap();
//! ```

use crate::config::FeedForwardConfig;
use crate::error::{Result, TrustformerError};
use crate::ffn::FeedForward;
use tensorlogic_ir::{EinsumGraph, EinsumNode};

/// Router/Gating strategy for selecting experts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterType {
    /// Top-K routing: Select K experts with highest routing scores
    TopK,
    /// Softmax routing: Use all experts with softmax-weighted combination
    Softmax,
    /// Switch routing: Select single expert (Top-1) per token
    Switch,
    /// Expert Choice: Experts select tokens instead of tokens selecting experts
    ExpertChoice,
}

/// Configuration for Mixture-of-Experts layer
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Model dimension
    pub d_model: usize,
    /// Expert FFN dimension
    pub d_ff: usize,
    /// Number of expert networks
    pub num_experts: usize,
    /// Number of experts to activate per token (top-k)
    pub experts_per_tok: usize,
    /// Router/gating strategy
    pub router_type: RouterType,
    /// Load balancing coefficient
    pub load_balance_coef: f64,
    /// Router dropout
    pub router_dropout: f64,
    /// Expert dropout
    pub expert_dropout: f64,
    /// Activation function for experts
    pub activation: String,
}

impl MoeConfig {
    /// Create new MoE configuration
    pub fn new(
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
        experts_per_tok: usize,
    ) -> Result<Self> {
        if d_model == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "d_model must be > 0".into(),
            });
        }
        if d_ff == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "d_ff must be > 0".into(),
            });
        }
        if num_experts == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "num_experts must be > 0".into(),
            });
        }
        if experts_per_tok == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "experts_per_tok must be > 0".into(),
            });
        }
        if experts_per_tok > num_experts {
            return Err(TrustformerError::InvalidDimension {
                expected: num_experts,
                got: experts_per_tok,
                context: "experts_per_tok must be <= num_experts".into(),
            });
        }

        Ok(Self {
            d_model,
            d_ff,
            num_experts,
            experts_per_tok,
            router_type: RouterType::TopK,
            load_balance_coef: 0.01,
            router_dropout: 0.0,
            expert_dropout: 0.0,
            activation: "gelu".to_string(),
        })
    }

    /// Builder: Set router type
    pub fn with_router_type(mut self, router_type: RouterType) -> Self {
        self.router_type = router_type;
        self
    }

    /// Builder: Set load balancing coefficient
    pub fn with_load_balance_coef(mut self, coef: f64) -> Self {
        self.load_balance_coef = coef;
        self
    }

    /// Builder: Set router dropout
    pub fn with_router_dropout(mut self, dropout: f64) -> Self {
        self.router_dropout = dropout;
        self
    }

    /// Builder: Set expert dropout
    pub fn with_expert_dropout(mut self, dropout: f64) -> Self {
        self.expert_dropout = dropout;
        self
    }

    /// Builder: Set activation function
    pub fn with_activation(mut self, activation: &str) -> Self {
        self.activation = activation.to_string();
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(TrustformerError::CompilationError(
                "d_model must be > 0".into(),
            ));
        }
        if self.num_experts == 0 {
            return Err(TrustformerError::CompilationError(
                "num_experts must be > 0".into(),
            ));
        }
        if self.experts_per_tok == 0 || self.experts_per_tok > self.num_experts {
            return Err(TrustformerError::CompilationError(format!(
                "experts_per_tok ({}) must be in range [1, {}]",
                self.experts_per_tok, self.num_experts
            )));
        }
        if self.load_balance_coef < 0.0 {
            return Err(TrustformerError::CompilationError(
                "load_balance_coef must be >= 0".into(),
            ));
        }
        Ok(())
    }

    /// Get sparsity factor (fraction of experts used per token)
    pub fn sparsity_factor(&self) -> f64 {
        self.experts_per_tok as f64 / self.num_experts as f64
    }
}

/// Mixture-of-Experts layer
pub struct MoeLayer {
    /// Configuration
    pub config: MoeConfig,
    /// Expert networks
    pub experts: Vec<FeedForward>,
}

impl MoeLayer {
    /// Create new MoE layer
    pub fn new(config: MoeConfig) -> Result<Self> {
        config.validate()?;

        // Create expert networks
        let mut experts = Vec::with_capacity(config.num_experts);
        for _ in 0..config.num_experts {
            let expert_config = FeedForwardConfig::new(config.d_model, config.d_ff)
                .with_activation(&config.activation)
                .with_dropout(config.expert_dropout);
            experts.push(FeedForward::new(expert_config)?);
        }

        Ok(Self { config, experts })
    }

    /// Build MoE einsum graph
    ///
    /// # Graph Inputs
    /// - Tensor 0: Input `[batch, seq_len, d_model]`
    /// - Tensor 1: Router weights `[d_model, num_experts]`
    /// - Tensors 2+: Expert weights (W1, b1, W2, b2 for each expert)
    ///
    /// # Graph Outputs
    /// - MoE output `[batch, seq_len, d_model]`
    /// - Routing weights (for load balancing loss)
    ///
    /// Note: This is a simplified representation showing the routing structure.
    /// Full implementation would include top-k selection and expert dispatch.
    pub fn build_moe_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // 1. Compute routing logits: einsum("bsd,de->bse", x, W_router)
        let router_logits = graph.add_tensor("router_logits");
        let router_node = EinsumNode::new("bsd,de->bse", vec![0, 1], vec![router_logits]);
        graph.add_node(router_node)?;

        // 2. Apply routing function (softmax or top-k)
        let routing_weights = graph.add_tensor("routing_weights");
        let routing_op = match self.config.router_type {
            RouterType::Softmax | RouterType::TopK => "softmax_e",
            RouterType::Switch => "argmax_e",
            RouterType::ExpertChoice => "expert_choice",
        };
        let routing_node = EinsumNode::elem_unary(routing_op, router_logits, routing_weights);
        graph.add_node(routing_node)?;

        // 3. For simplified representation, we compute a combined expert output
        // In practice, this would involve:
        // - Dispatching tokens to selected experts
        // - Computing expert outputs in parallel
        // - Combining outputs with routing weights

        // Placeholder: combined output
        let combined_output = graph.add_tensor("moe_output");

        // This would be the weighted combination of expert outputs
        // For now, we represent it as the routing weights (placeholder)
        let combine_node = EinsumNode::elem_unary(
            "combine_experts".to_string(),
            routing_weights,
            combined_output,
        );
        graph.add_node(combine_node)?;

        // Return both the MoE output and routing weights (for aux loss)
        Ok(vec![combined_output, routing_weights])
    }

    /// Get configuration
    pub fn config(&self) -> &MoeConfig {
        &self.config
    }

    /// Count total parameters
    pub fn count_parameters(&self) -> usize {
        let mut total = 0;

        // Router parameters: d_model × num_experts
        total += self.config.d_model * self.config.num_experts;

        // Expert parameters: num_experts × expert_params
        let expert_params = crate::utils::count_ffn_params(&self.experts[0].config);
        total += self.config.num_experts * expert_params;

        total
    }

    /// Calculate theoretical FLOPs for MoE forward pass
    ///
    /// # Arguments
    /// - `batch_size`: Batch size
    /// - `seq_len`: Sequence length
    pub fn count_flops(&self, batch_size: usize, seq_len: usize) -> usize {
        let mut flops = 0;

        // Router computation: batch × seq × d_model × num_experts
        flops += batch_size * seq_len * self.config.d_model * self.config.num_experts;

        // Expert computation (sparse): only experts_per_tok are computed
        let tokens_total = batch_size * seq_len;
        let expert_flops_per_token = 2 * self.config.d_model * self.config.d_ff;
        flops += tokens_total * self.config.experts_per_tok * expert_flops_per_token;

        flops
    }

    /// Calculate memory usage for experts
    pub fn expert_memory_usage(&self) -> usize {
        let expert_params = crate::utils::count_ffn_params(&self.experts[0].config);
        // Assuming 4 bytes per parameter (float32)
        expert_params * self.config.num_experts * 4
    }
}

/// MoE statistics for monitoring and analysis
#[derive(Debug, Clone)]
pub struct MoeStats {
    /// Total parameters
    pub total_params: usize,
    /// Parameters per expert
    pub params_per_expert: usize,
    /// Active parameters per forward pass (considering sparsity)
    pub active_params: usize,
    /// Sparsity factor (fraction of experts used)
    pub sparsity: f64,
    /// Theoretical speedup vs dense (1 / sparsity)
    pub theoretical_speedup: f64,
}

impl MoeLayer {
    /// Get MoE statistics
    pub fn stats(&self) -> MoeStats {
        let total_params = self.count_parameters();
        let params_per_expert = crate::utils::count_ffn_params(&self.experts[0].config);
        let sparsity = self.config.sparsity_factor();

        // Active params: router + (experts_per_tok / num_experts) * total_expert_params
        let router_params = self.config.d_model * self.config.num_experts;
        let total_expert_params = params_per_expert * self.config.num_experts;
        let active_params = router_params + (sparsity * total_expert_params as f64) as usize;

        MoeStats {
            total_params,
            params_per_expert,
            active_params,
            sparsity,
            theoretical_speedup: 1.0 / sparsity,
        }
    }
}

/// Common MoE presets based on research papers
pub enum MoePreset {
    /// Switch Transformer (Google): Top-1 routing
    Switch,
    /// GShard (Google): Top-2 routing
    GShard,
    /// Mixtral 8x7B: 8 experts, top-2
    Mixtral8x7B,
    /// Expert choice routing
    ExpertChoice,
}

impl MoePreset {
    /// Create MoE configuration from preset
    pub fn config(&self, d_model: usize, d_ff: usize) -> Result<MoeConfig> {
        match self {
            MoePreset::Switch => Ok(MoeConfig::new(d_model, d_ff, 128, 1)?
                .with_router_type(RouterType::Switch)
                .with_load_balance_coef(0.01)),
            MoePreset::GShard => Ok(MoeConfig::new(d_model, d_ff, 2048, 2)?
                .with_router_type(RouterType::TopK)
                .with_load_balance_coef(0.01)),
            MoePreset::Mixtral8x7B => Ok(MoeConfig::new(d_model, d_ff, 8, 2)?
                .with_router_type(RouterType::TopK)
                .with_load_balance_coef(0.01)),
            MoePreset::ExpertChoice => Ok(MoeConfig::new(d_model, d_ff, 64, 2)?
                .with_router_type(RouterType::ExpertChoice)
                .with_load_balance_coef(0.01)),
        }
    }

    /// Get preset name
    pub fn name(&self) -> &'static str {
        match self {
            MoePreset::Switch => "Switch Transformer",
            MoePreset::GShard => "GShard",
            MoePreset::Mixtral8x7B => "Mixtral 8x7B",
            MoePreset::ExpertChoice => "Expert Choice",
        }
    }

    /// Get preset description
    pub fn description(&self) -> &'static str {
        match self {
            MoePreset::Switch => "Top-1 routing with 128 experts (Google Switch Transformer)",
            MoePreset::GShard => "Top-2 routing with 2048 experts (Google GShard)",
            MoePreset::Mixtral8x7B => "8 experts, Top-2 routing (Mistral AI Mixtral)",
            MoePreset::ExpertChoice => "Expert Choice routing with 64 experts",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_config_creation() {
        let config = MoeConfig::new(512, 2048, 8, 2).unwrap();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.d_ff, 2048);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.experts_per_tok, 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_moe_config_validation() {
        // Invalid: experts_per_tok > num_experts
        let result = MoeConfig::new(512, 2048, 4, 8);
        assert!(result.is_err());

        // Valid configuration
        let result = MoeConfig::new(512, 2048, 8, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_moe_sparsity_factor() {
        let config = MoeConfig::new(512, 2048, 8, 2).unwrap();
        assert!((config.sparsity_factor() - 0.25).abs() < 1e-10);

        let config = MoeConfig::new(512, 2048, 10, 1).unwrap();
        assert!((config.sparsity_factor() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_moe_layer_creation() {
        let config = MoeConfig::new(512, 2048, 4, 2).unwrap();
        let moe = MoeLayer::new(config).unwrap();

        assert_eq!(moe.experts.len(), 4);
        assert!(moe.config().validate().is_ok());
    }

    #[test]
    fn test_moe_builder_pattern() {
        let config = MoeConfig::new(512, 2048, 8, 2)
            .unwrap()
            .with_router_type(RouterType::Switch)
            .with_load_balance_coef(0.02)
            .with_router_dropout(0.1)
            .with_expert_dropout(0.1)
            .with_activation("relu");

        assert_eq!(config.router_type, RouterType::Switch);
        assert!((config.load_balance_coef - 0.02).abs() < 1e-10);
        assert!((config.router_dropout - 0.1).abs() < 1e-10);
        assert!((config.expert_dropout - 0.1).abs() < 1e-10);
        assert_eq!(config.activation, "relu");
    }

    #[test]
    fn test_router_types() {
        let types = vec![
            RouterType::TopK,
            RouterType::Softmax,
            RouterType::Switch,
            RouterType::ExpertChoice,
        ];

        for router_type in types {
            let config = MoeConfig::new(512, 2048, 8, 2)
                .unwrap()
                .with_router_type(router_type);
            assert_eq!(config.router_type, router_type);
        }
    }

    #[test]
    fn test_moe_graph_building() {
        let config = MoeConfig::new(512, 2048, 4, 2).unwrap();
        let moe = MoeLayer::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("input"); // Tensor 0
        graph.add_tensor("router_weights"); // Tensor 1

        let outputs = moe.build_moe_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 2); // output + routing_weights
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_moe_parameter_count() {
        let config = MoeConfig::new(512, 2048, 8, 2).unwrap();
        let moe = MoeLayer::new(config).unwrap();

        let params = moe.count_parameters();
        assert!(params > 0);

        // Router params + 8 × expert params
        let expected_router = 512 * 8;
        let expert_params = crate::utils::count_ffn_params(&moe.experts[0].config);
        let expected_total = expected_router + 8 * expert_params;
        assert_eq!(params, expected_total);
    }

    #[test]
    fn test_moe_flops_calculation() {
        let config = MoeConfig::new(512, 2048, 8, 2).unwrap();
        let moe = MoeLayer::new(config).unwrap();

        let flops = moe.count_flops(32, 128);
        assert!(flops > 0);
    }

    #[test]
    fn test_moe_stats() {
        let config = MoeConfig::new(512, 2048, 8, 2).unwrap();
        let moe = MoeLayer::new(config).unwrap();

        let stats = moe.stats();
        assert!(stats.total_params > 0);
        assert!(stats.active_params > 0);
        assert!(stats.active_params < stats.total_params);
        assert!((stats.sparsity - 0.25).abs() < 1e-10);
        assert!((stats.theoretical_speedup - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_moe_presets() {
        let presets = vec![
            MoePreset::Switch,
            MoePreset::GShard,
            MoePreset::Mixtral8x7B,
            MoePreset::ExpertChoice,
        ];

        for preset in &presets {
            let config = preset.config(512, 2048).unwrap();
            assert!(config.validate().is_ok());
            assert!(!preset.name().is_empty());
            assert!(!preset.description().is_empty());
        }
    }

    #[test]
    fn test_mixtral_preset_specifics() {
        let config = MoePreset::Mixtral8x7B.config(512, 2048).unwrap();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.experts_per_tok, 2);
        assert_eq!(config.router_type, RouterType::TopK);
    }

    #[test]
    fn test_switch_preset_specifics() {
        let config = MoePreset::Switch.config(512, 2048).unwrap();
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.experts_per_tok, 1);
        assert_eq!(config.router_type, RouterType::Switch);
    }

    #[test]
    fn test_moe_memory_usage() {
        let config = MoeConfig::new(512, 2048, 8, 2).unwrap();
        let moe = MoeLayer::new(config).unwrap();

        let memory = moe.expert_memory_usage();
        assert!(memory > 0);
    }

    #[test]
    fn test_invalid_configurations() {
        // Zero experts
        assert!(MoeConfig::new(512, 2048, 0, 1).is_err());

        // Zero experts_per_tok
        assert!(MoeConfig::new(512, 2048, 8, 0).is_err());

        // Zero d_model
        assert!(MoeConfig::new(0, 2048, 8, 2).is_err());
    }
}
