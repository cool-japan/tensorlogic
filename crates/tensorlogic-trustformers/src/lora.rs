//! # LoRA (Low-Rank Adaptation)
//!
//! Implementation of LoRA for parameter-efficient fine-tuning of transformer models.
//!
//! LoRA freezes the pre-trained model weights and injects trainable rank decomposition
//! matrices into each layer, greatly reducing the number of trainable parameters.
//!
//! ## Architecture
//!
//! For a pre-trained weight matrix W, LoRA adds:
//! ```text
//! W' = W + BA
//! ```
//! where B and A are low-rank matrices with rank r << min(d, k).
//!
//! ## Key Benefits
//!
//! - **Parameter Efficiency**: Reduces trainable parameters by 10,000x
//! - **No Inference Latency**: Merged weights can be used at inference
//! - **Task Switching**: Easy to swap LoRA modules for different tasks

use crate::error::{Result, TrustformerError};
use tensorlogic_ir::{EinsumGraph, EinsumNode};

/// Configuration for LoRA
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    /// Rank of the low-rank decomposition
    pub rank: usize,
    /// Scaling factor (alpha / rank)
    pub alpha: f64,
    /// Dropout probability for LoRA layers
    pub dropout: f64,
    /// Whether to apply LoRA to query projections
    pub apply_to_q: bool,
    /// Whether to apply LoRA to value projections
    pub apply_to_v: bool,
}

impl LoRAConfig {
    /// Create a new LoRA configuration
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            apply_to_q: true,
            apply_to_v: true,
        }
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Apply LoRA to specific projections
    pub fn with_projections(mut self, q: bool, v: bool) -> Self {
        self.apply_to_q = q;
        self.apply_to_v = v;
        self
    }

    /// Get the scaling factor
    pub fn scaling(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(TrustformerError::MissingParameter(
                "rank must be positive".to_string(),
            ));
        }
        if self.alpha <= 0.0 {
            return Err(TrustformerError::CompilationError(
                "alpha must be positive".to_string(),
            ));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(TrustformerError::CompilationError(
                "dropout must be between 0 and 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Calculate number of trainable parameters for a linear layer
    pub fn trainable_params(&self, in_features: usize, out_features: usize) -> usize {
        self.rank * in_features + out_features * self.rank
    }

    /// Calculate compression ratio compared to full fine-tuning
    pub fn compression_ratio(&self, in_features: usize, out_features: usize) -> f64 {
        let full_params = in_features * out_features;
        let lora_params = self.trainable_params(in_features, out_features);
        full_params as f64 / lora_params as f64
    }
}

/// LoRA linear layer implementation
#[derive(Debug, Clone)]
pub struct LoRALinear {
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
    /// LoRA configuration
    config: LoRAConfig,
    /// Layer name for unique tensor naming
    name: String,
}

impl LoRALinear {
    /// Create a new LoRA linear layer
    pub fn new(in_features: usize, out_features: usize, config: LoRAConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            in_features,
            out_features,
            config,
            name: String::new(),
        })
    }

    /// Set the layer name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Get configuration
    pub fn config(&self) -> &LoRAConfig {
        &self.config
    }

    /// Get number of trainable parameters
    pub fn trainable_params(&self) -> usize {
        self.config
            .trainable_params(self.in_features, self.out_features)
    }

    /// Build the LoRA linear layer graph
    ///
    /// Input tensor at index 0
    /// Outputs the combined result of W + BA
    pub fn build_lora_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        let prefix = if self.name.is_empty() {
            "lora".to_string()
        } else {
            format!("lora_{}", self.name)
        };

        // Main weight output
        let main_output = graph.add_tensor(format!("{}_main_out", prefix));
        let main_node = EinsumNode::new(
            "bsd,do->bso",
            vec![0, graph.add_tensor(format!("{}_W", prefix))],
            vec![main_output],
        );
        graph.add_node(main_node)?;

        // LoRA A matrix [rank, in_features]
        let lora_a = graph.add_tensor(format!("{}_A", prefix));

        // LoRA intermediate: x @ A^T
        let lora_int = graph.add_tensor(format!("{}_int", prefix));
        let lora_int_node = EinsumNode::new("bsd,rd->bsr", vec![0, lora_a], vec![lora_int]);
        graph.add_node(lora_int_node)?;

        // LoRA B matrix [out_features, rank]
        let lora_b = graph.add_tensor(format!("{}_B", prefix));

        // LoRA output: (x @ A^T) @ B^T
        let lora_out = graph.add_tensor(format!("{}_out", prefix));
        let lora_out_node = EinsumNode::new("bsr,or->bso", vec![lora_int, lora_b], vec![lora_out]);
        graph.add_node(lora_out_node)?;

        // Scale LoRA output
        let scale_tensor = graph.add_tensor(format!("{}_scale", prefix));
        let scaled_out = graph.add_tensor(format!("{}_scaled", prefix));
        let scale_node = EinsumNode::elem_binary(
            format!("mul_scalar_{}", self.config.scaling()),
            lora_out,
            scale_tensor,
            scaled_out,
        );
        graph.add_node(scale_node)?;

        // Add main + scaled LoRA
        let final_out = graph.add_tensor(format!("{}_final", prefix));
        let add_node =
            EinsumNode::elem_binary("add".to_string(), main_output, scaled_out, final_out);
        graph.add_node(add_node)?;

        Ok(vec![final_out])
    }
}

/// LoRA-enhanced attention layer
#[derive(Debug, Clone)]
pub struct LoRAAttention {
    /// Model dimension
    d_model: usize,
    /// Number of heads
    n_heads: usize,
    /// LoRA configuration
    config: LoRAConfig,
}

impl LoRAAttention {
    /// Create a new LoRA attention layer
    pub fn new(d_model: usize, n_heads: usize, config: LoRAConfig) -> Result<Self> {
        if !d_model.is_multiple_of(n_heads) {
            return Err(TrustformerError::InvalidHeadCount { d_model, n_heads });
        }
        config.validate()?;
        Ok(Self {
            d_model,
            n_heads,
            config,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &LoRAConfig {
        &self.config
    }

    /// Get number of trainable parameters
    pub fn trainable_params(&self) -> usize {
        let mut params = 0;
        let d = self.d_model;

        if self.config.apply_to_q {
            params += self.config.trainable_params(d, d);
        }
        if self.config.apply_to_v {
            params += self.config.trainable_params(d, d);
        }

        params
    }

    /// Build the LoRA attention graph
    pub fn build_lora_attention_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        let d_k = self.d_model / self.n_heads;

        // Create LoRA for Q if enabled
        let q_output = if self.config.apply_to_q {
            let lora_q =
                LoRALinear::new(self.d_model, self.d_model, self.config.clone())?.with_name("q");
            let outputs = lora_q.build_lora_graph(graph)?;
            outputs[0]
        } else {
            0
        };

        // Create LoRA for V if enabled
        let v_idx = if self.config.apply_to_v {
            let lora_v =
                LoRALinear::new(self.d_model, self.d_model, self.config.clone())?.with_name("v");
            let outputs = lora_v.build_lora_graph(graph)?;
            outputs[0]
        } else {
            2
        };

        // Standard attention computation
        let q_split = graph.add_tensor("lora_attn_q_split");
        let k_split = graph.add_tensor("lora_attn_k_split");
        let v_split = graph.add_tensor("lora_attn_v_split");

        let reshape_spec = format!("bsd->bsh{}", d_k);

        let q_reshape = EinsumNode::new(&reshape_spec, vec![q_output], vec![q_split]);
        graph.add_node(q_reshape)?;

        let k_reshape = EinsumNode::new(&reshape_spec, vec![1], vec![k_split]);
        graph.add_node(k_reshape)?;

        let v_reshape = EinsumNode::new(&reshape_spec, vec![v_idx], vec![v_split]);
        graph.add_node(v_reshape)?;

        // Transpose
        let q_t = graph.add_tensor("lora_attn_q_t");
        let k_t = graph.add_tensor("lora_attn_k_t");
        let v_t = graph.add_tensor("lora_attn_v_t");

        graph.add_node(EinsumNode::new("bshd->bhsd", vec![q_split], vec![q_t]))?;
        graph.add_node(EinsumNode::new("bshd->bhsd", vec![k_split], vec![k_t]))?;
        graph.add_node(EinsumNode::new("bshd->bhsd", vec![v_split], vec![v_t]))?;

        // Attention scores
        let scores = graph.add_tensor("lora_attn_scores");
        graph.add_node(EinsumNode::new(
            "bhqd,bhkd->bhqk",
            vec![q_t, k_t],
            vec![scores],
        ))?;

        // Scale
        let scale_factor = (d_k as f64).sqrt();
        let scale_t = graph.add_tensor("lora_attn_scale");
        let scaled = graph.add_tensor("lora_attn_scaled");
        graph.add_node(EinsumNode::elem_binary(
            format!("div_scalar_{}", scale_factor),
            scores,
            scale_t,
            scaled,
        ))?;

        // Softmax
        let weights = graph.add_tensor("lora_attn_weights");
        graph.add_node(EinsumNode::elem_unary("softmax_k", scaled, weights))?;

        // Apply to values
        let attn_out = graph.add_tensor("lora_attn_out");
        graph.add_node(EinsumNode::new(
            "bhqk,bhkv->bhqv",
            vec![weights, v_t],
            vec![attn_out],
        ))?;

        // Transpose back and reshape
        let back = graph.add_tensor("lora_attn_back");
        graph.add_node(EinsumNode::new("bhsd->bshd", vec![attn_out], vec![back]))?;

        let output = graph.add_tensor("lora_attn_output");
        let reshape_back = format!("bsh{}-:bsd", d_k);
        graph.add_node(EinsumNode::new(&reshape_back, vec![back], vec![output]))?;

        Ok(vec![output])
    }
}

/// Statistics for LoRA modules
#[derive(Debug, Clone)]
pub struct LoRAStats {
    /// Total trainable parameters
    pub trainable_params: usize,
    /// Total frozen parameters
    pub frozen_params: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Memory savings percentage
    pub memory_savings: f64,
}

impl LoRAStats {
    /// Create stats for a complete LoRA-enhanced model
    pub fn for_model(config: &LoRAConfig, d_model: usize, n_layers: usize) -> Self {
        let mut trainable = 0;
        let frozen = d_model * d_model * 4 * n_layers; // Q, K, V, O per layer

        for _ in 0..n_layers {
            if config.apply_to_q {
                trainable += config.trainable_params(d_model, d_model);
            }
            if config.apply_to_v {
                trainable += config.trainable_params(d_model, d_model);
            }
        }

        let compression_ratio = if trainable > 0 {
            (frozen + trainable) as f64 / trainable as f64
        } else {
            f64::INFINITY
        };

        let memory_savings = 1.0 - (trainable as f64 / (frozen + trainable) as f64);

        Self {
            trainable_params: trainable,
            frozen_params: frozen,
            compression_ratio,
            memory_savings,
        }
    }

    /// Format as a summary string
    pub fn summary(&self) -> String {
        format!(
            "LoRA Statistics\n  Trainable: {}\n  Frozen: {}\n  \
             Compression: {:.1}x\n  Memory savings: {:.1}%",
            self.trainable_params,
            self.frozen_params,
            self.compression_ratio,
            self.memory_savings * 100.0
        )
    }
}

/// Common LoRA presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoRAPreset {
    /// Small rank (4), minimal parameters
    Minimal,
    /// Medium rank (8), balanced
    Standard,
    /// Larger rank (16), higher capacity
    Extended,
}

impl LoRAPreset {
    /// Get the configuration for this preset
    pub fn config(&self) -> LoRAConfig {
        match self {
            LoRAPreset::Minimal => LoRAConfig::new(4, 8.0),
            LoRAPreset::Standard => LoRAConfig::new(8, 16.0),
            LoRAPreset::Extended => LoRAConfig::new(16, 32.0),
        }
    }

    /// Get the name of this preset
    pub fn name(&self) -> &'static str {
        match self {
            LoRAPreset::Minimal => "Minimal",
            LoRAPreset::Standard => "Standard",
            LoRAPreset::Extended => "Extended",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_creation() {
        let config = LoRAConfig::new(8, 16.0);
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.scaling(), 2.0);
    }

    #[test]
    fn test_lora_config_builder() {
        let config = LoRAConfig::new(8, 16.0)
            .with_dropout(0.1)
            .with_projections(true, true);

        assert!((config.dropout - 0.1).abs() < 1e-10);
        assert!(config.apply_to_q);
        assert!(config.apply_to_v);
    }

    #[test]
    fn test_lora_config_validation() {
        let config = LoRAConfig::new(8, 16.0);
        assert!(config.validate().is_ok());

        // Invalid rank
        let mut bad = config.clone();
        bad.rank = 0;
        assert!(bad.validate().is_err());

        // Invalid alpha
        let mut bad = config.clone();
        bad.alpha = -1.0;
        assert!(bad.validate().is_err());

        // Invalid dropout
        let mut bad = config.clone();
        bad.dropout = 1.5;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_lora_trainable_params() {
        let config = LoRAConfig::new(8, 16.0);
        let params = config.trainable_params(512, 512);
        // A: 8*512 + B: 512*8 = 8192
        assert_eq!(params, 8192);
    }

    #[test]
    fn test_lora_compression_ratio() {
        let config = LoRAConfig::new(8, 16.0);
        let ratio = config.compression_ratio(512, 512);
        // Full: 512*512 = 262144, LoRA: 8192, ratio = 32
        assert!((ratio - 32.0).abs() < 0.1);
    }

    #[test]
    fn test_lora_linear_creation() {
        let config = LoRAConfig::new(8, 16.0);
        let lora = LoRALinear::new(512, 512, config).unwrap();
        assert_eq!(lora.trainable_params(), 8192);
    }

    #[test]
    fn test_lora_linear_graph() {
        let config = LoRAConfig::new(8, 16.0);
        let lora = LoRALinear::new(512, 512, config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");

        let outputs = lora.build_lora_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_lora_attention_creation() {
        let config = LoRAConfig::new(8, 16.0);
        let lora_attn = LoRAAttention::new(512, 8, config).unwrap();

        // Q and V: 2 * 8192 = 16384
        assert_eq!(lora_attn.trainable_params(), 16384);
    }

    #[test]
    fn test_lora_attention_graph() {
        let config = LoRAConfig::new(8, 16.0);
        let lora_attn = LoRAAttention::new(512, 8, config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = lora_attn.build_lora_attention_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_lora_stats() {
        let config = LoRAConfig::new(8, 16.0);
        let stats = LoRAStats::for_model(&config, 512, 6);

        assert!(stats.trainable_params > 0);
        assert!(stats.frozen_params > 0);
        assert!(stats.compression_ratio > 1.0);
        assert!(stats.memory_savings > 0.0);
    }

    #[test]
    fn test_lora_presets() {
        let config = LoRAPreset::Minimal.config();
        assert_eq!(config.rank, 4);

        let config = LoRAPreset::Standard.config();
        assert_eq!(config.rank, 8);

        let config = LoRAPreset::Extended.config();
        assert_eq!(config.rank, 16);
    }

    #[test]
    fn test_lora_preset_names() {
        assert_eq!(LoRAPreset::Minimal.name(), "Minimal");
        assert_eq!(LoRAPreset::Standard.name(), "Standard");
    }

    #[test]
    fn test_lora_invalid_attention() {
        let config = LoRAConfig::new(8, 16.0);
        assert!(LoRAAttention::new(512, 7, config).is_err());
    }
}
