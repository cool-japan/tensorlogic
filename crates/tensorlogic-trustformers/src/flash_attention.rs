//! # Flash Attention
//!
//! Implementation of Flash Attention for memory-efficient attention computation.
//!
//! Flash Attention uses tiling and recomputation to reduce memory usage from O(N²)
//! to O(N) while maintaining exact attention computation.
//!
//! ## Key Benefits
//!
//! - **Memory Efficiency**: O(N) instead of O(N²) memory
//! - **IO Efficiency**: Minimizes HBM access through tiling
//! - **Exact Computation**: No approximation, mathematically equivalent
//!
//! ## Algorithm
//!
//! Flash Attention tiles the computation:
//! 1. Load Q, K, V blocks from HBM to SRAM
//! 2. Compute local attention on-chip
//! 3. Use online softmax for numerical stability
//! 4. Write output block back to HBM

use crate::error::{Result, TrustformerError};
use tensorlogic_ir::{EinsumGraph, EinsumNode};

/// Configuration for Flash Attention
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per head
    pub d_k: usize,
    /// Block size for Q (Br)
    pub block_size_q: usize,
    /// Block size for KV (Bc)
    pub block_size_kv: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Dropout probability
    pub dropout: f64,
    /// Maximum sequence length (for memory estimation)
    pub max_seq_len: usize,
}

impl FlashAttentionConfig {
    /// Create a new Flash Attention configuration
    pub fn new(d_model: usize, n_heads: usize) -> Result<Self> {
        if !d_model.is_multiple_of(n_heads) {
            return Err(TrustformerError::InvalidHeadCount { d_model, n_heads });
        }

        let d_k = d_model / n_heads;

        // Default block sizes optimized for typical GPU SRAM
        let block_size_q = 128;
        let block_size_kv = 128;

        Ok(Self {
            d_model,
            n_heads,
            d_k,
            block_size_q,
            block_size_kv,
            causal: false,
            dropout: 0.0,
            max_seq_len: 4096,
        })
    }

    /// Set block sizes for tiling
    pub fn with_block_sizes(mut self, block_q: usize, block_kv: usize) -> Self {
        self.block_size_q = block_q;
        self.block_size_kv = block_kv;
        self
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

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
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
        if self.block_size_q == 0 || self.block_size_kv == 0 {
            return Err(TrustformerError::MissingParameter(
                "block sizes must be positive".to_string(),
            ));
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(TrustformerError::CompilationError(
                "dropout must be between 0 and 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Calculate SRAM usage per block (in elements)
    pub fn sram_usage_per_block(&self) -> usize {
        // Q block: Br x d
        // K block: Bc x d
        // V block: Bc x d
        // S block: Br x Bc (local attention scores)
        // O block: Br x d (output accumulator)
        let q_block = self.block_size_q * self.d_k;
        let k_block = self.block_size_kv * self.d_k;
        let v_block = self.block_size_kv * self.d_k;
        let s_block = self.block_size_q * self.block_size_kv;
        let o_block = self.block_size_q * self.d_k;

        q_block + k_block + v_block + s_block + o_block
    }

    /// Calculate memory savings compared to standard attention
    pub fn memory_savings(&self, seq_len: usize) -> f64 {
        // Standard attention: O(N² + N*d) for attention matrix + output
        // Flash attention: O(N*d) for output only (tiled computation)
        let standard_memory = seq_len * seq_len + seq_len * self.d_k;
        let flash_memory = seq_len * self.d_k + self.sram_usage_per_block();

        1.0 - (flash_memory as f64 / standard_memory as f64)
    }

    /// Calculate number of passes over KV for given sequence length
    pub fn num_kv_passes(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.block_size_kv)
    }

    /// Calculate number of Q blocks for given sequence length
    pub fn num_q_blocks(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.block_size_q)
    }
}

/// Flash Attention implementation
#[derive(Debug, Clone)]
pub struct FlashAttention {
    /// Configuration
    config: FlashAttentionConfig,
}

impl FlashAttention {
    /// Create a new Flash Attention module
    pub fn new(config: FlashAttentionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Get the configuration
    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }

    /// Build the Flash Attention graph
    ///
    /// This creates a graph representing the Flash Attention algorithm.
    /// The actual tiling is handled by the backend execution.
    pub fn build_flash_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        let _n_heads = self.config.n_heads;
        let d_k = self.config.d_k;

        // Step 1: Reshape Q, K, V to multi-head format
        let q_split = graph.add_tensor("flash_q_split");
        let k_split = graph.add_tensor("flash_k_split");
        let v_split = graph.add_tensor("flash_v_split");

        let reshape_spec = format!("bsd->bsh{}", d_k);

        let q_reshape = EinsumNode::new(&reshape_spec, vec![0], vec![q_split]);
        graph.add_node(q_reshape)?;

        let k_reshape = EinsumNode::new(&reshape_spec, vec![1], vec![k_split]);
        graph.add_node(k_reshape)?;

        let v_reshape = EinsumNode::new(&reshape_spec, vec![2], vec![v_split]);
        graph.add_node(v_reshape)?;

        // Step 2: Transpose to [batch, n_heads, seq, d_k]
        let q_transposed = graph.add_tensor("flash_q_transposed");
        let k_transposed = graph.add_tensor("flash_k_transposed");
        let v_transposed = graph.add_tensor("flash_v_transposed");

        graph.add_node(EinsumNode::new(
            "bshd->bhsd",
            vec![q_split],
            vec![q_transposed],
        ))?;
        graph.add_node(EinsumNode::new(
            "bshd->bhsd",
            vec![k_split],
            vec![k_transposed],
        ))?;
        graph.add_node(EinsumNode::new(
            "bshd->bhsd",
            vec![v_split],
            vec![v_transposed],
        ))?;

        // Step 3: Flash Attention kernel (represented as a special einsum)
        // The einsum specification encodes the flash attention pattern
        // Backend implementations should recognize this pattern and apply tiling
        let flash_output = graph.add_tensor("flash_attn_output");

        // Use a special einsum spec that indicates flash attention
        // Format: flash_bhqd,bhkd,bhkv->bhqv_Br_Bc_causal
        let flash_spec = format!(
            "flash_bhqd,bhkd,bhkv->bhqv_{}_{}",
            self.config.block_size_q, self.config.block_size_kv
        );
        let flash_node = EinsumNode::new(
            &flash_spec,
            vec![q_transposed, k_transposed, v_transposed],
            vec![flash_output],
        );
        graph.add_node(flash_node)?;

        // Step 4: Transpose back
        let transposed_back = graph.add_tensor("flash_transposed_back");
        graph.add_node(EinsumNode::new(
            "bhsd->bshd",
            vec![flash_output],
            vec![transposed_back],
        ))?;

        // Step 5: Reshape to [batch, seq, d_model]
        let output = graph.add_tensor("flash_output");
        let reshape_back_spec = format!("bsh{}-:bsd", d_k);
        graph.add_node(EinsumNode::new(
            &reshape_back_spec,
            vec![transposed_back],
            vec![output],
        ))?;

        Ok(vec![output])
    }
}

/// Flash Attention v2 improvements
#[derive(Debug, Clone)]
pub struct FlashAttentionV2Config {
    /// Base configuration
    pub base: FlashAttentionConfig,
    /// Use sequence parallelism
    pub sequence_parallel: bool,
    /// Use sliding window (for Mistral-style models)
    pub window_size: Option<usize>,
}

impl FlashAttentionV2Config {
    /// Create a new Flash Attention v2 configuration
    pub fn new(d_model: usize, n_heads: usize) -> Result<Self> {
        Ok(Self {
            base: FlashAttentionConfig::new(d_model, n_heads)?,
            sequence_parallel: false,
            window_size: None,
        })
    }

    /// Enable sequence parallelism
    pub fn with_sequence_parallel(mut self, enabled: bool) -> Self {
        self.sequence_parallel = enabled;
        self
    }

    /// Set sliding window size
    pub fn with_window(mut self, window_size: usize) -> Self {
        self.window_size = Some(window_size);
        self
    }

    /// Enable causal masking
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.base = self.base.with_causal(causal);
        self
    }
}

/// Statistics for Flash Attention
#[derive(Debug, Clone)]
pub struct FlashAttentionStats {
    /// Configuration
    pub config: FlashAttentionConfig,
    /// Memory savings for given sequence length
    pub memory_savings: f64,
    /// SRAM usage per block
    pub sram_usage: usize,
    /// Number of KV passes
    pub num_kv_passes: usize,
    /// Number of Q blocks
    pub num_q_blocks: usize,
}

impl FlashAttentionStats {
    /// Create stats from configuration and sequence length
    pub fn from_config(config: &FlashAttentionConfig, seq_len: usize) -> Self {
        Self {
            config: config.clone(),
            memory_savings: config.memory_savings(seq_len),
            sram_usage: config.sram_usage_per_block(),
            num_kv_passes: config.num_kv_passes(seq_len),
            num_q_blocks: config.num_q_blocks(seq_len),
        }
    }

    /// Format as a summary string
    pub fn summary(&self, seq_len: usize) -> String {
        format!(
            "Flash Attention Statistics\n  d_model: {}\n  n_heads: {}\n  \
             block_size_q: {}\n  block_size_kv: {}\n  causal: {}\n  \
             memory savings: {:.1}%\n  SRAM usage: {} elements\n  \
             num_kv_passes: {}\n  num_q_blocks: {}\n  seq_len: {}",
            self.config.d_model,
            self.config.n_heads,
            self.config.block_size_q,
            self.config.block_size_kv,
            self.config.causal,
            self.memory_savings * 100.0,
            self.sram_usage,
            self.num_kv_passes,
            self.num_q_blocks,
            seq_len
        )
    }
}

/// Presets for common Flash Attention configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttentionPreset {
    /// Standard configuration (128x128 blocks)
    Standard,
    /// Large block configuration (256x256 blocks)
    LargeBlocks,
    /// Small block configuration (64x64 blocks)
    SmallBlocks,
    /// Optimized for A100 GPU
    A100Optimized,
    /// Optimized for H100 GPU
    H100Optimized,
}

impl FlashAttentionPreset {
    /// Get the block sizes for this preset
    pub fn block_sizes(&self) -> (usize, usize) {
        match self {
            FlashAttentionPreset::Standard => (128, 128),
            FlashAttentionPreset::LargeBlocks => (256, 256),
            FlashAttentionPreset::SmallBlocks => (64, 64),
            FlashAttentionPreset::A100Optimized => (128, 64),
            FlashAttentionPreset::H100Optimized => (128, 128),
        }
    }

    /// Get the name of this preset
    pub fn name(&self) -> &'static str {
        match self {
            FlashAttentionPreset::Standard => "Standard",
            FlashAttentionPreset::LargeBlocks => "Large Blocks",
            FlashAttentionPreset::SmallBlocks => "Small Blocks",
            FlashAttentionPreset::A100Optimized => "A100 Optimized",
            FlashAttentionPreset::H100Optimized => "H100 Optimized",
        }
    }

    /// Create a configuration with this preset
    pub fn config(&self, d_model: usize, n_heads: usize) -> Result<FlashAttentionConfig> {
        let (block_q, block_kv) = self.block_sizes();
        FlashAttentionConfig::new(d_model, n_heads).map(|c| c.with_block_sizes(block_q, block_kv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_config_creation() {
        let config = FlashAttentionConfig::new(512, 8).unwrap();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.d_k, 64);
        assert_eq!(config.block_size_q, 128);
        assert_eq!(config.block_size_kv, 128);
    }

    #[test]
    fn test_flash_config_builder() {
        let config = FlashAttentionConfig::new(512, 8)
            .unwrap()
            .with_block_sizes(64, 64)
            .with_causal(true)
            .with_dropout(0.1)
            .with_max_seq_len(8192);

        assert_eq!(config.block_size_q, 64);
        assert_eq!(config.block_size_kv, 64);
        assert!(config.causal);
        assert!((config.dropout - 0.1).abs() < 1e-10);
        assert_eq!(config.max_seq_len, 8192);
    }

    #[test]
    fn test_flash_invalid_config() {
        // Invalid d_model
        assert!(FlashAttentionConfig::new(512, 7).is_err());
    }

    #[test]
    fn test_flash_sram_usage() {
        let config = FlashAttentionConfig::new(512, 8).unwrap();
        let sram = config.sram_usage_per_block();

        // Q: 128 * 64 = 8192
        // K: 128 * 64 = 8192
        // V: 128 * 64 = 8192
        // S: 128 * 128 = 16384
        // O: 128 * 64 = 8192
        // Total: 49152
        assert_eq!(sram, 49152);
    }

    #[test]
    fn test_flash_memory_savings() {
        let config = FlashAttentionConfig::new(512, 8).unwrap();
        let savings = config.memory_savings(4096);

        // For long sequences, flash attention saves significant memory
        assert!(savings > 0.9); // Should save >90% for 4096 seq len
    }

    #[test]
    fn test_flash_num_passes() {
        let config = FlashAttentionConfig::new(512, 8).unwrap();

        // 4096 / 128 = 32 passes
        assert_eq!(config.num_kv_passes(4096), 32);
        assert_eq!(config.num_q_blocks(4096), 32);

        // 1000 / 128 = 8 passes (ceiling)
        assert_eq!(config.num_kv_passes(1000), 8);
    }

    #[test]
    fn test_flash_graph_building() {
        let config = FlashAttentionConfig::new(512, 8).unwrap();
        let flash = FlashAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = flash.build_flash_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_flash_causal_graph() {
        let config = FlashAttentionConfig::new(512, 8).unwrap().with_causal(true);
        let flash = FlashAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = flash.build_flash_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_flash_v2_config() {
        let config = FlashAttentionV2Config::new(512, 8)
            .unwrap()
            .with_sequence_parallel(true)
            .with_window(4096)
            .with_causal(true);

        assert!(config.sequence_parallel);
        assert_eq!(config.window_size, Some(4096));
        assert!(config.base.causal);
    }

    #[test]
    fn test_flash_presets() {
        let (q, kv) = FlashAttentionPreset::Standard.block_sizes();
        assert_eq!(q, 128);
        assert_eq!(kv, 128);

        let (q, kv) = FlashAttentionPreset::LargeBlocks.block_sizes();
        assert_eq!(q, 256);
        assert_eq!(kv, 256);

        let (q, kv) = FlashAttentionPreset::A100Optimized.block_sizes();
        assert_eq!(q, 128);
        assert_eq!(kv, 64);
    }

    #[test]
    fn test_flash_preset_config() {
        let config = FlashAttentionPreset::Standard.config(512, 8).unwrap();
        assert_eq!(config.block_size_q, 128);
        assert_eq!(config.block_size_kv, 128);
    }

    #[test]
    fn test_flash_preset_names() {
        assert_eq!(FlashAttentionPreset::Standard.name(), "Standard");
        assert_eq!(FlashAttentionPreset::A100Optimized.name(), "A100 Optimized");
    }

    #[test]
    fn test_flash_stats() {
        let config = FlashAttentionConfig::new(512, 8).unwrap();
        let stats = FlashAttentionStats::from_config(&config, 4096);

        assert!(stats.memory_savings > 0.9);
        assert_eq!(stats.sram_usage, 49152);
        assert_eq!(stats.num_kv_passes, 32);
        assert_eq!(stats.num_q_blocks, 32);
    }

    #[test]
    fn test_flash_validate() {
        let config = FlashAttentionConfig::new(512, 8).unwrap();
        assert!(config.validate().is_ok());

        // Invalid dropout
        let mut bad = config.clone();
        bad.dropout = 1.5;
        assert!(bad.validate().is_err());

        // Invalid block size
        let mut bad = config.clone();
        bad.block_size_q = 0;
        assert!(bad.validate().is_err());
    }
}
