//! # Sliding Window Attention
//!
//! Implementation of Sliding Window Attention for efficient long-sequence processing.
//!
//! Sliding Window Attention constrains attention to a fixed-size window around each
//! position, reducing complexity from O(n^2) to O(n * w) where w is the window size.
//!
//! ## Used By
//!
//! - Mistral 7B (window size: 4096)
//! - Longformer
//! - BigBird

use crate::error::{Result, TrustformerError};
use tensorlogic_ir::{EinsumGraph, EinsumNode};

/// Configuration for Sliding Window Attention
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Window size (positions attended to)
    pub window_size: usize,
    /// Dimension per head
    pub d_k: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Dropout probability
    pub dropout: f64,
}

impl SlidingWindowConfig {
    /// Create a new Sliding Window Attention configuration
    pub fn new(d_model: usize, n_heads: usize, window_size: usize) -> Result<Self> {
        if !d_model.is_multiple_of(n_heads) {
            return Err(TrustformerError::InvalidHeadCount { d_model, n_heads });
        }

        if window_size == 0 {
            return Err(TrustformerError::MissingParameter(
                "window_size must be positive".to_string(),
            ));
        }

        let d_k = d_model / n_heads;

        Ok(Self {
            d_model,
            n_heads,
            window_size,
            d_k,
            causal: false,
            dropout: 0.0,
        })
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

    /// Calculate complexity reduction compared to full attention
    pub fn complexity_reduction(&self, seq_len: usize) -> f64 {
        if seq_len <= self.window_size {
            1.0
        } else {
            self.window_size as f64 / seq_len as f64
        }
    }

    /// Calculate memory reduction compared to full attention
    pub fn memory_reduction(&self, seq_len: usize) -> f64 {
        if seq_len <= self.window_size {
            1.0
        } else {
            self.window_size as f64 / seq_len as f64
        }
    }
}

/// Sliding Window Attention implementation
#[derive(Debug, Clone)]
pub struct SlidingWindowAttention {
    /// Configuration
    config: SlidingWindowConfig,
}

impl SlidingWindowAttention {
    /// Create a new Sliding Window Attention module
    pub fn new(config: SlidingWindowConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Get the configuration
    pub fn config(&self) -> &SlidingWindowConfig {
        &self.config
    }

    /// Build the sliding window attention graph
    pub fn build_swa_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        let _n_heads = self.config.n_heads;
        let d_k = self.config.d_k;

        // Step 1: Reshape Q, K, V to multi-head format
        let q_split = graph.add_tensor("swa_q_split");
        let k_split = graph.add_tensor("swa_k_split");
        let v_split = graph.add_tensor("swa_v_split");

        let reshape_spec = format!("bsd->bsh{}", d_k);

        let q_reshape = EinsumNode::new(&reshape_spec, vec![0], vec![q_split]);
        graph.add_node(q_reshape)?;

        let k_reshape = EinsumNode::new(&reshape_spec, vec![1], vec![k_split]);
        graph.add_node(k_reshape)?;

        let v_reshape = EinsumNode::new(&reshape_spec, vec![2], vec![v_split]);
        graph.add_node(v_reshape)?;

        // Step 2: Transpose to [batch, n_heads, seq, d_k]
        let q_transposed = graph.add_tensor("swa_q_transposed");
        let k_transposed = graph.add_tensor("swa_k_transposed");
        let v_transposed = graph.add_tensor("swa_v_transposed");

        let transpose_q = EinsumNode::new("bshd->bhsd", vec![q_split], vec![q_transposed]);
        graph.add_node(transpose_q)?;

        let transpose_k = EinsumNode::new("bshd->bhsd", vec![k_split], vec![k_transposed]);
        graph.add_node(transpose_k)?;

        let transpose_v = EinsumNode::new("bshd->bhsd", vec![v_split], vec![v_transposed]);
        graph.add_node(transpose_v)?;

        // Step 3: Compute attention scores
        let scores = graph.add_tensor("swa_scores");
        let scores_node = EinsumNode::new(
            "bhqd,bhkd->bhqk",
            vec![q_transposed, k_transposed],
            vec![scores],
        );
        graph.add_node(scores_node)?;

        // Step 4: Scale scores
        let scale_factor = (d_k as f64).sqrt();
        let scale_tensor = graph.add_tensor("swa_scale");
        let scaled_scores = graph.add_tensor("swa_scaled_scores");
        let scale_node = EinsumNode::elem_binary(
            format!("div_scalar_{}", scale_factor),
            scores,
            scale_tensor,
            scaled_scores,
        );
        graph.add_node(scale_node)?;

        // Step 5: Apply sliding window mask
        let masked_scores = graph.add_tensor("swa_masked_scores");
        let mask_node = EinsumNode::elem_unary(
            format!("sliding_window_mask_{}", self.config.window_size),
            scaled_scores,
            masked_scores,
        );
        graph.add_node(mask_node)?;

        // Step 6: Softmax
        let attention_weights = graph.add_tensor("swa_attention_weights");
        let softmax_node = EinsumNode::elem_unary("softmax_k", masked_scores, attention_weights);
        graph.add_node(softmax_node)?;

        // Step 7: Apply attention to values
        let attn_output = graph.add_tensor("swa_attn_output");
        let attn_node = EinsumNode::new(
            "bhqk,bhkv->bhqv",
            vec![attention_weights, v_transposed],
            vec![attn_output],
        );
        graph.add_node(attn_node)?;

        // Step 8: Transpose back
        let transposed_back = graph.add_tensor("swa_transposed_back");
        let transpose_back =
            EinsumNode::new("bhsd->bshd", vec![attn_output], vec![transposed_back]);
        graph.add_node(transpose_back)?;

        // Step 9: Reshape to [batch, seq, d_model]
        let output = graph.add_tensor("swa_output");
        let reshape_back_spec = format!("bsh{}-:bsd", d_k);
        let reshape_back = EinsumNode::new(&reshape_back_spec, vec![transposed_back], vec![output]);
        graph.add_node(reshape_back)?;

        Ok(vec![output])
    }
}

/// Presets for common Sliding Window Attention configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlidingWindowPreset {
    /// Mistral 7B (window: 4096)
    Mistral7B,
    /// Longformer Base (window: 512)
    LongformerBase,
    /// BigBird Base (window: 256)
    BigBirdBase,
}

impl SlidingWindowPreset {
    /// Get the configuration for this preset
    pub fn config(&self) -> Result<SlidingWindowConfig> {
        match self {
            SlidingWindowPreset::Mistral7B => {
                SlidingWindowConfig::new(4096, 32, 4096)?
                    .with_causal(true)
                    .validate()?;
                Ok(SlidingWindowConfig::new(4096, 32, 4096)?.with_causal(true))
            }
            SlidingWindowPreset::LongformerBase => SlidingWindowConfig::new(768, 12, 512),
            SlidingWindowPreset::BigBirdBase => SlidingWindowConfig::new(768, 12, 256),
        }
    }

    /// Get the name of this preset
    pub fn name(&self) -> &'static str {
        match self {
            SlidingWindowPreset::Mistral7B => "Mistral 7B",
            SlidingWindowPreset::LongformerBase => "Longformer Base",
            SlidingWindowPreset::BigBirdBase => "BigBird Base",
        }
    }
}

/// Statistics for Sliding Window Attention
#[derive(Debug, Clone)]
pub struct SlidingWindowStats {
    /// Configuration
    pub config: SlidingWindowConfig,
    /// Complexity reduction for given sequence length
    pub complexity_reduction: f64,
    /// Memory reduction for given sequence length
    pub memory_reduction: f64,
}

impl SlidingWindowStats {
    /// Create stats from configuration and sequence length
    pub fn from_config(config: &SlidingWindowConfig, seq_len: usize) -> Self {
        Self {
            config: config.clone(),
            complexity_reduction: config.complexity_reduction(seq_len),
            memory_reduction: config.memory_reduction(seq_len),
        }
    }

    /// Format as a summary string
    pub fn summary(&self, seq_len: usize) -> String {
        format!(
            "Sliding Window Attention\n  d_model: {}\n  n_heads: {}\n  window_size: {}\n  \
             causal: {}\n  complexity reduction: {:.1}%\n  memory reduction: {:.1}%\n  \
             seq_len: {}",
            self.config.d_model,
            self.config.n_heads,
            self.config.window_size,
            self.config.causal,
            (1.0 - self.complexity_reduction) * 100.0,
            (1.0 - self.memory_reduction) * 100.0,
            seq_len
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swa_config_creation() {
        let config = SlidingWindowConfig::new(4096, 32, 4096).unwrap();
        assert_eq!(config.d_model, 4096);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.window_size, 4096);
        assert_eq!(config.d_k, 128);
    }

    #[test]
    fn test_swa_config_builder() {
        let config = SlidingWindowConfig::new(4096, 32, 4096)
            .unwrap()
            .with_causal(true)
            .with_dropout(0.1);

        assert!(config.causal);
        assert!((config.dropout - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_swa_invalid_configs() {
        // Invalid d_model
        assert!(SlidingWindowConfig::new(512, 7, 256).is_err());

        // Invalid window size
        assert!(SlidingWindowConfig::new(512, 8, 0).is_err());
    }

    #[test]
    fn test_swa_complexity_reduction() {
        let config = SlidingWindowConfig::new(512, 8, 256).unwrap();

        // Short sequence: no reduction
        assert_eq!(config.complexity_reduction(128), 1.0);

        // Long sequence: significant reduction
        let reduction = config.complexity_reduction(4096);
        assert!((reduction - 0.0625).abs() < 0.001);
    }

    #[test]
    fn test_swa_graph_building() {
        let config = SlidingWindowConfig::new(512, 8, 256).unwrap();
        let swa = SlidingWindowAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = swa.build_swa_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_swa_causal_graph() {
        let config = SlidingWindowConfig::new(512, 8, 256)
            .unwrap()
            .with_causal(true);
        let swa = SlidingWindowAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = swa.build_swa_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_swa_presets() {
        // Mistral 7B
        let config = SlidingWindowPreset::Mistral7B.config().unwrap();
        assert_eq!(config.d_model, 4096);
        assert_eq!(config.window_size, 4096);
        assert!(config.causal);

        // Longformer Base
        let config = SlidingWindowPreset::LongformerBase.config().unwrap();
        assert_eq!(config.d_model, 768);
        assert_eq!(config.window_size, 512);

        // BigBird Base
        let config = SlidingWindowPreset::BigBirdBase.config().unwrap();
        assert_eq!(config.window_size, 256);
    }

    #[test]
    fn test_swa_preset_names() {
        assert_eq!(SlidingWindowPreset::Mistral7B.name(), "Mistral 7B");
        assert_eq!(
            SlidingWindowPreset::LongformerBase.name(),
            "Longformer Base"
        );
    }

    #[test]
    fn test_swa_stats() {
        let config = SlidingWindowConfig::new(4096, 32, 4096).unwrap();
        let stats = SlidingWindowStats::from_config(&config, 32768);

        // 4096/32768 = 0.125
        assert!((stats.complexity_reduction - 0.125).abs() < 0.001);
        assert!((stats.memory_reduction - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_swa_validate() {
        let config = SlidingWindowConfig::new(512, 8, 256).unwrap();
        assert!(config.validate().is_ok());

        // Invalid dropout
        let mut bad = config.clone();
        bad.dropout = -0.1;
        assert!(bad.validate().is_err());
    }
}
