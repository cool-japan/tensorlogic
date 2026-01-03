//! # Grouped-Query Attention (GQA)
//!
//! Implementation of Grouped-Query Attention for efficient attention computation.
//!
//! GQA is a technique used in modern LLMs (LLaMA 2, Mistral, Falcon) that reduces
//! the key-value cache memory by using fewer KV heads than query heads.
//!
//! ## Architecture
//!
//! ```text
//! Multi-Head Attention (MHA):  n_kv_heads = n_heads
//! Grouped-Query Attention:     n_kv_heads < n_heads
//! Multi-Query Attention (MQA): n_kv_heads = 1
//! ```
//!
//! ## Key Benefits
//!
//! - **Reduced KV Cache**: Memory reduced by factor of n_heads / n_kv_heads
//! - **Faster Inference**: Less memory bandwidth for KV cache
//! - **Maintained Quality**: Performance close to MHA

use crate::error::{Result, TrustformerError};
use tensorlogic_ir::{EinsumGraph, EinsumNode};

/// Configuration for Grouped-Query Attention
#[derive(Debug, Clone)]
pub struct GQAConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of query heads
    pub n_heads: usize,
    /// Number of key-value heads (n_heads must be divisible by n_kv_heads)
    pub n_kv_heads: usize,
    /// Dimension per head (d_model / n_heads)
    pub d_k: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Dropout probability
    pub dropout: f64,
}

impl GQAConfig {
    /// Create a new GQA configuration
    pub fn new(d_model: usize, n_heads: usize, n_kv_heads: usize) -> Result<Self> {
        if n_kv_heads == 0 {
            return Err(TrustformerError::MissingParameter(
                "n_kv_heads must be positive".to_string(),
            ));
        }

        if !d_model.is_multiple_of(n_heads) {
            return Err(TrustformerError::InvalidHeadCount { d_model, n_heads });
        }

        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(TrustformerError::CompilationError(format!(
                "n_heads ({}) must be divisible by n_kv_heads ({})",
                n_heads, n_kv_heads
            )));
        }

        let d_k = d_model / n_heads;

        Ok(Self {
            d_model,
            n_heads,
            n_kv_heads,
            d_k,
            causal: false,
            dropout: 0.0,
        })
    }

    /// Create a Multi-Head Attention configuration (n_kv_heads = n_heads)
    pub fn mha(d_model: usize, n_heads: usize) -> Result<Self> {
        Self::new(d_model, n_heads, n_heads)
    }

    /// Create a Multi-Query Attention configuration (n_kv_heads = 1)
    pub fn mqa(d_model: usize, n_heads: usize) -> Result<Self> {
        Self::new(d_model, n_heads, 1)
    }

    /// Enable causal masking
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Get the number of query heads per KV head (group size)
    pub fn group_size(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Calculate KV cache memory reduction factor
    pub fn kv_cache_reduction(&self) -> f64 {
        self.n_heads as f64 / self.n_kv_heads as f64
    }

    /// Check if this is standard MHA
    pub fn is_mha(&self) -> bool {
        self.n_heads == self.n_kv_heads
    }

    /// Check if this is MQA
    pub fn is_mqa(&self) -> bool {
        self.n_kv_heads == 1
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(TrustformerError::MissingParameter(
                "d_model must be positive".to_string(),
            ));
        }
        if self.n_heads == 0 {
            return Err(TrustformerError::MissingParameter(
                "n_heads must be positive".to_string(),
            ));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(TrustformerError::CompilationError(
                "dropout must be between 0 and 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Calculate the number of parameters
    pub fn num_parameters(&self) -> usize {
        // Q projection: d_model * d_model
        // K projection: d_model * (n_kv_heads * d_k)
        // V projection: d_model * (n_kv_heads * d_k)
        // O projection: d_model * d_model
        let q_params = self.d_model * self.d_model;
        let kv_dim = self.n_kv_heads * self.d_k;
        let k_params = self.d_model * kv_dim;
        let v_params = self.d_model * kv_dim;
        let o_params = self.d_model * self.d_model;
        q_params + k_params + v_params + o_params
    }
}

/// Grouped-Query Attention implementation
#[derive(Debug, Clone)]
pub struct GroupedQueryAttention {
    /// Configuration
    config: GQAConfig,
}

impl GroupedQueryAttention {
    /// Create a new Grouped-Query Attention module
    pub fn new(config: GQAConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Get the configuration
    pub fn config(&self) -> &GQAConfig {
        &self.config
    }

    /// Build the GQA graph
    ///
    /// Input tensors expected in graph:
    /// - "Q": Query tensor [batch, seq_q, d_model]
    /// - "K": Key tensor [batch, seq_k, d_model]
    /// - "V": Value tensor [batch, seq_k, d_model]
    pub fn build_gqa_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let d_k = self.config.d_k;
        let group_size = self.config.group_size();

        // Step 1: Reshape Q to multi-head format
        // [batch, seq, d_model] -> [batch, n_heads, seq, d_k]
        let q_split = graph.add_tensor("gqa_q_split");
        let reshape_q_spec = format!("bsd->bs{}{}->bh{}d", n_heads, d_k, d_k);
        let q_reshape = EinsumNode::new(&reshape_q_spec, vec![0], vec![q_split]);
        graph.add_node(q_reshape)?;

        // Transpose Q to [batch, n_heads, seq, d_k]
        let q_transposed = graph.add_tensor("gqa_q_transposed");
        let transpose_q = EinsumNode::new("bshd->bhsd", vec![q_split], vec![q_transposed]);
        graph.add_node(transpose_q)?;

        // Step 2: Reshape K and V to KV head format
        let k_split = graph.add_tensor("gqa_k_split");
        let v_split = graph.add_tensor("gqa_v_split");

        let reshape_kv_spec = format!("bsd->bs{}{}->bh{}d", n_kv_heads, d_k, d_k);
        let k_reshape = EinsumNode::new(&reshape_kv_spec, vec![1], vec![k_split]);
        graph.add_node(k_reshape)?;

        let v_reshape = EinsumNode::new(&reshape_kv_spec, vec![2], vec![v_split]);
        graph.add_node(v_reshape)?;

        // Transpose K and V
        let k_transposed = graph.add_tensor("gqa_k_transposed");
        let v_transposed = graph.add_tensor("gqa_v_transposed");

        let transpose_k = EinsumNode::new("bshd->bhsd", vec![k_split], vec![k_transposed]);
        graph.add_node(transpose_k)?;

        let transpose_v = EinsumNode::new("bshd->bhsd", vec![v_split], vec![v_transposed]);
        graph.add_node(transpose_v)?;

        // Step 3: Repeat KV heads to match query heads if needed
        let (k_expanded, v_expanded) = if group_size > 1 {
            // For GQA, we need to repeat each KV head `group_size` times
            // This is done via einsum broadcasting
            let k_exp = graph.add_tensor("gqa_k_expanded");
            let v_exp = graph.add_tensor("gqa_v_expanded");

            // Repeat operation as einsum
            let repeat_spec = format!("bhsd->b{}hsd", group_size);
            let k_repeat = EinsumNode::new(&repeat_spec, vec![k_transposed], vec![k_exp]);
            graph.add_node(k_repeat)?;

            let v_repeat = EinsumNode::new(&repeat_spec, vec![v_transposed], vec![v_exp]);
            graph.add_node(v_repeat)?;

            (k_exp, v_exp)
        } else {
            (k_transposed, v_transposed)
        };

        // Step 4: Compute attention scores
        let scores = graph.add_tensor("gqa_scores");
        let scores_node = EinsumNode::new(
            "bhqd,bhkd->bhqk",
            vec![q_transposed, k_expanded],
            vec![scores],
        );
        graph.add_node(scores_node)?;

        // Step 5: Scale scores
        let scale_factor = (d_k as f64).sqrt();
        let scale_tensor = graph.add_tensor("gqa_scale");
        let scaled_scores = graph.add_tensor("gqa_scaled_scores");
        let scale_node = EinsumNode::elem_binary(
            format!("div_scalar_{}", scale_factor),
            scores,
            scale_tensor,
            scaled_scores,
        );
        graph.add_node(scale_node)?;

        // Step 6: Softmax
        let attention_weights = graph.add_tensor("gqa_attention_weights");
        let softmax_node = EinsumNode::elem_unary("softmax_k", scaled_scores, attention_weights);
        graph.add_node(softmax_node)?;

        // Step 7: Apply attention to values
        let attn_output = graph.add_tensor("gqa_attn_output");
        let attn_node = EinsumNode::new(
            "bhqk,bhkv->bhqv",
            vec![attention_weights, v_expanded],
            vec![attn_output],
        );
        graph.add_node(attn_node)?;

        // Step 8: Transpose back
        let transposed_back = graph.add_tensor("gqa_transposed_back");
        let transpose_back =
            EinsumNode::new("bhsd->bshd", vec![attn_output], vec![transposed_back]);
        graph.add_node(transpose_back)?;

        // Step 9: Reshape to [batch, seq, d_model]
        let output = graph.add_tensor("gqa_output");
        let reshape_back_spec = format!("bsh{}-:bsd", d_k);
        let reshape_back = EinsumNode::new(&reshape_back_spec, vec![transposed_back], vec![output]);
        graph.add_node(reshape_back)?;

        Ok(vec![output])
    }

    /// Calculate memory usage for KV cache
    pub fn kv_cache_memory(&self, batch_size: usize, seq_len: usize, dtype_bytes: usize) -> usize {
        let per_kv = batch_size * self.config.n_kv_heads * seq_len * self.config.d_k;
        per_kv * 2 * dtype_bytes
    }

    /// Calculate memory savings compared to MHA
    pub fn memory_savings(&self, batch_size: usize, seq_len: usize) -> f64 {
        let gqa_memory = self.kv_cache_memory(batch_size, seq_len, 1);
        let mha_memory = batch_size * self.config.n_heads * seq_len * self.config.d_k * 2;
        1.0 - (gqa_memory as f64 / mha_memory as f64)
    }
}

/// Presets for common GQA configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GQAPreset {
    /// LLaMA 2 7B (32 heads, 32 KV heads - standard MHA)
    Llama2_7B,
    /// LLaMA 2 70B (64 heads, 8 KV heads)
    Llama2_70B,
    /// Mistral 7B (32 heads, 8 KV heads)
    Mistral7B,
    /// Falcon 40B (128 heads, 1 KV head - MQA)
    Falcon40B,
}

impl GQAPreset {
    /// Get the GQA configuration for this preset
    pub fn config(&self) -> Result<GQAConfig> {
        match self {
            GQAPreset::Llama2_7B => GQAConfig::mha(4096, 32),
            GQAPreset::Llama2_70B => GQAConfig::new(8192, 64, 8),
            GQAPreset::Mistral7B => GQAConfig::new(4096, 32, 8),
            GQAPreset::Falcon40B => GQAConfig::mqa(8192, 128),
        }
    }

    /// Get the name of this preset
    pub fn name(&self) -> &'static str {
        match self {
            GQAPreset::Llama2_7B => "LLaMA 2 7B",
            GQAPreset::Llama2_70B => "LLaMA 2 70B",
            GQAPreset::Mistral7B => "Mistral 7B",
            GQAPreset::Falcon40B => "Falcon 40B",
        }
    }
}

/// Statistics for GQA module
#[derive(Debug, Clone)]
pub struct GQAStats {
    /// Configuration
    pub config: GQAConfig,
    /// Total parameters
    pub num_parameters: usize,
    /// KV cache reduction factor
    pub kv_cache_reduction: f64,
    /// Is standard MHA
    pub is_mha: bool,
    /// Is MQA
    pub is_mqa: bool,
    /// Group size (queries per KV head)
    pub group_size: usize,
}

impl GQAStats {
    /// Create stats from a GQA configuration
    pub fn from_config(config: &GQAConfig) -> Self {
        Self {
            config: config.clone(),
            num_parameters: config.num_parameters(),
            kv_cache_reduction: config.kv_cache_reduction(),
            is_mha: config.is_mha(),
            is_mqa: config.is_mqa(),
            group_size: config.group_size(),
        }
    }

    /// Format as a summary string
    pub fn summary(&self) -> String {
        let attention_type = if self.is_mha {
            "Multi-Head Attention (MHA)"
        } else if self.is_mqa {
            "Multi-Query Attention (MQA)"
        } else {
            "Grouped-Query Attention (GQA)"
        };

        format!(
            "{}\n  d_model: {}\n  n_heads: {}\n  n_kv_heads: {}\n  d_k: {}\n  \
             group_size: {}\n  parameters: {}\n  KV cache reduction: {:.1}x",
            attention_type,
            self.config.d_model,
            self.config.n_heads,
            self.config.n_kv_heads,
            self.config.d_k,
            self.group_size,
            self.num_parameters,
            self.kv_cache_reduction
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gqa_config_creation() {
        let config = GQAConfig::new(4096, 32, 8).unwrap();
        assert_eq!(config.d_model, 4096);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 8);
        assert_eq!(config.d_k, 128);
        assert_eq!(config.group_size(), 4);
    }

    #[test]
    fn test_gqa_mha_config() {
        let config = GQAConfig::mha(512, 8).unwrap();
        assert!(config.is_mha());
        assert!(!config.is_mqa());
        assert_eq!(config.group_size(), 1);
        assert_eq!(config.kv_cache_reduction(), 1.0);
    }

    #[test]
    fn test_gqa_mqa_config() {
        let config = GQAConfig::mqa(512, 8).unwrap();
        assert!(!config.is_mha());
        assert!(config.is_mqa());
        assert_eq!(config.group_size(), 8);
        assert_eq!(config.kv_cache_reduction(), 8.0);
    }

    #[test]
    fn test_gqa_config_builder() {
        let config = GQAConfig::new(4096, 32, 8)
            .unwrap()
            .with_causal(true)
            .with_dropout(0.1);

        assert!(config.causal);
        assert!((config.dropout - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_gqa_invalid_configs() {
        // n_kv_heads = 0
        assert!(GQAConfig::new(512, 8, 0).is_err());

        // d_model not divisible by n_heads
        assert!(GQAConfig::new(512, 7, 1).is_err());

        // n_heads not divisible by n_kv_heads
        assert!(GQAConfig::new(512, 8, 3).is_err());
    }

    #[test]
    fn test_gqa_graph_building() {
        let config = GQAConfig::new(512, 8, 2).unwrap();
        let gqa = GroupedQueryAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = gqa.build_gqa_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gqa_mha_graph() {
        let config = GQAConfig::mha(512, 8).unwrap();
        let gqa = GroupedQueryAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = gqa.build_gqa_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_gqa_presets() {
        // LLaMA 2 7B (MHA)
        let config = GQAPreset::Llama2_7B.config().unwrap();
        assert_eq!(config.d_model, 4096);
        assert!(config.is_mha());

        // LLaMA 2 70B (GQA)
        let config = GQAPreset::Llama2_70B.config().unwrap();
        assert_eq!(config.group_size(), 8);

        // Mistral 7B
        let config = GQAPreset::Mistral7B.config().unwrap();
        assert_eq!(config.group_size(), 4);

        // Falcon 40B (MQA)
        let config = GQAPreset::Falcon40B.config().unwrap();
        assert!(config.is_mqa());
    }

    #[test]
    fn test_gqa_memory_calculations() {
        let config = GQAConfig::new(4096, 32, 8).unwrap();
        let gqa = GroupedQueryAttention::new(config).unwrap();

        let savings = gqa.memory_savings(1, 2048);
        // With 8 KV heads vs 32 query heads, we save 75%
        assert!((savings - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_gqa_num_parameters() {
        let config = GQAConfig::mha(4096, 32).unwrap();
        let params = config.num_parameters();
        // 4 * 4096 * 4096 = 67,108,864
        assert_eq!(params, 67_108_864);
    }

    #[test]
    fn test_gqa_stats() {
        let config = GQAConfig::new(4096, 32, 8).unwrap();
        let stats = GQAStats::from_config(&config);

        assert_eq!(stats.group_size, 4);
        assert_eq!(stats.kv_cache_reduction, 4.0);
        assert!(!stats.is_mha);
        assert!(!stats.is_mqa);
    }

    #[test]
    fn test_gqa_preset_names() {
        assert_eq!(GQAPreset::Llama2_7B.name(), "LLaMA 2 7B");
        assert_eq!(GQAPreset::Mistral7B.name(), "Mistral 7B");
    }

    #[test]
    fn test_gqa_validate() {
        let config = GQAConfig::new(512, 8, 2).unwrap();
        assert!(config.validate().is_ok());

        // Invalid dropout
        let mut bad_config = config.clone();
        bad_config.dropout = 1.5;
        assert!(bad_config.validate().is_err());
    }
}
