//! Sparse attention patterns for efficient long-sequence processing.
//!
//! This module implements various sparse attention mechanisms that reduce
//! the quadratic complexity of full attention by only attending to a subset
//! of positions.
//!
//! ## Sparse Attention Types
//!
//! 1. **Fixed Pattern**: Pre-defined sparse patterns (strided, local, global)
//! 2. **Random**: Randomly sample attention positions
//! 3. **Learned**: Learn which positions to attend to
//! 4. **Hybrid**: Combine multiple sparse patterns
//!
//! ## Complexity
//!
//! - Full attention: O(n²) memory and compute
//! - Sparse attention: O(n·k) where k << n is the sparsity factor

use serde::{Deserialize, Serialize};
use tensorlogic_ir::{EinsumGraph, EinsumNode};

use crate::{
    config::AttentionConfig,
    error::{Result, TrustformerError},
};

/// Type of sparse attention pattern
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SparsePatternType {
    /// Strided pattern: attend every k-th position
    Strided { stride: usize },
    /// Local window: attend to nearby positions
    Local { window_size: usize },
    /// Global + local: some positions attend globally, others locally
    GlobalLocal {
        window_size: usize,
        global_positions: Vec<usize>,
    },
    /// Block sparse: divide sequence into blocks
    BlockSparse { block_size: usize },
    /// Random: randomly sample k positions per query
    Random { num_random: usize },
}

/// Configuration for sparse attention
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseAttentionConfig {
    /// Base attention configuration
    pub base_attention: AttentionConfig,
    /// Sparse pattern type
    pub pattern: SparsePatternType,
    /// Whether to use exact sparse computation or approximation
    pub exact_sparse: bool,
}

impl SparseAttentionConfig {
    /// Create a new strided sparse attention configuration
    pub fn strided(base_attention: AttentionConfig, stride: usize) -> Result<Self> {
        if stride == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "stride must be positive".to_string(),
            });
        }

        Ok(Self {
            base_attention,
            pattern: SparsePatternType::Strided { stride },
            exact_sparse: true,
        })
    }

    /// Create a new local window sparse attention configuration
    pub fn local(base_attention: AttentionConfig, window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "window_size must be positive".to_string(),
            });
        }

        Ok(Self {
            base_attention,
            pattern: SparsePatternType::Local { window_size },
            exact_sparse: true,
        })
    }

    /// Create a new global-local sparse attention configuration
    pub fn global_local(
        base_attention: AttentionConfig,
        window_size: usize,
        global_positions: Vec<usize>,
    ) -> Result<Self> {
        if window_size == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "window_size must be positive".to_string(),
            });
        }

        Ok(Self {
            base_attention,
            pattern: SparsePatternType::GlobalLocal {
                window_size,
                global_positions,
            },
            exact_sparse: true,
        })
    }

    /// Create a new block sparse attention configuration
    pub fn block_sparse(base_attention: AttentionConfig, block_size: usize) -> Result<Self> {
        if block_size == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "block_size must be positive".to_string(),
            });
        }

        Ok(Self {
            base_attention,
            pattern: SparsePatternType::BlockSparse { block_size },
            exact_sparse: true,
        })
    }

    /// Set whether to use exact sparse computation
    pub fn with_exact_sparse(mut self, exact_sparse: bool) -> Self {
        self.exact_sparse = exact_sparse;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.base_attention.validate()?;

        match &self.pattern {
            SparsePatternType::Strided { stride } => {
                if *stride == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "stride must be positive".to_string(),
                    });
                }
            }
            SparsePatternType::Local { window_size } => {
                if *window_size == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "window_size must be positive".to_string(),
                    });
                }
            }
            SparsePatternType::GlobalLocal {
                window_size,
                global_positions: _,
            } => {
                if *window_size == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "window_size must be positive".to_string(),
                    });
                }
            }
            SparsePatternType::BlockSparse { block_size } => {
                if *block_size == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "block_size must be positive".to_string(),
                    });
                }
            }
            SparsePatternType::Random { num_random } => {
                if *num_random == 0 {
                    return Err(TrustformerError::InvalidDimension {
                        expected: 1,
                        got: 0,
                        context: "num_random must be positive".to_string(),
                    });
                }
            }
        }

        Ok(())
    }
}

/// Sparse attention component
#[derive(Clone, Debug)]
pub struct SparseAttention {
    /// Configuration
    pub config: SparseAttentionConfig,
}

impl SparseAttention {
    /// Create a new sparse attention component
    pub fn new(config: SparseAttentionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build einsum graph for sparse attention
    ///
    /// Input tensors:
    /// - 0: Q (query) [batch, seq_len, d_model]
    /// - 1: K (key) [batch, seq_len, d_model]
    /// - 2: V (value) [batch, seq_len, d_model]
    /// - 3: sparse_mask [batch, seq_q, seq_k] (sparse pattern mask)
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_sparse_attention_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Compute attention scores
        let scores_tensor = graph.add_tensor("sparse_attn_scores");
        let scores_node = EinsumNode::new("bqd,bkd->bqk", vec![0, 1], vec![scores_tensor]);
        graph.add_node(scores_node)?;

        // Step 2: Scale scores
        let scale_factor = (self.config.base_attention.d_k as f64).sqrt();
        let scale_tensor = graph.add_tensor("sparse_scale");
        let scaled_tensor = graph.add_tensor("sparse_scaled_scores");
        let scale_node = EinsumNode::elem_binary(
            format!("div_scalar_{}", scale_factor),
            scores_tensor,
            scale_tensor,
            scaled_tensor,
        );
        graph.add_node(scale_node)?;

        // Step 3: Apply sparse mask
        // Mask out positions not in the sparse pattern
        let masked_tensor = graph.add_tensor("sparse_masked_scores");
        let mask_node = EinsumNode::elem_binary("mul", scaled_tensor, 3, masked_tensor);
        graph.add_node(mask_node)?;

        // Step 4: Apply softmax (only over non-masked positions)
        let softmax_tensor = graph.add_tensor("sparse_attention_weights");
        let softmax_node =
            EinsumNode::elem_unary("sparse_softmax_k", masked_tensor, softmax_tensor);
        graph.add_node(softmax_node)?;

        // Step 5: Apply attention to values
        let output_tensor = graph.add_tensor("sparse_attn_output");
        let output_node =
            EinsumNode::new("bqk,bkv->bqv", vec![softmax_tensor, 2], vec![output_tensor]);
        graph.add_node(output_node)?;

        Ok(vec![output_tensor])
    }

    /// Get the sparsity factor (approximate percentage of attended positions)
    pub fn sparsity_factor(&self, seq_len: usize) -> f64 {
        match &self.config.pattern {
            SparsePatternType::Strided { stride } => 1.0 / (*stride as f64),
            SparsePatternType::Local { window_size } => {
                (*window_size as f64).min(seq_len as f64) / (seq_len as f64)
            }
            SparsePatternType::GlobalLocal {
                window_size,
                global_positions,
            } => {
                let local_fraction = (*window_size as f64) / (seq_len as f64);
                let global_fraction = (global_positions.len() as f64) / (seq_len as f64);
                (local_fraction + global_fraction).min(1.0)
            }
            SparsePatternType::BlockSparse { block_size } => {
                (*block_size as f64) / (seq_len as f64)
            }
            SparsePatternType::Random { num_random } => (*num_random as f64) / (seq_len as f64),
        }
    }

    /// Get pattern description
    pub fn pattern_description(&self) -> String {
        match &self.config.pattern {
            SparsePatternType::Strided { stride } => {
                format!("Strided(stride={})", stride)
            }
            SparsePatternType::Local { window_size } => {
                format!("Local(window={})", window_size)
            }
            SparsePatternType::GlobalLocal {
                window_size,
                global_positions,
            } => {
                format!(
                    "GlobalLocal(window={}, global_tokens={})",
                    window_size,
                    global_positions.len()
                )
            }
            SparsePatternType::BlockSparse { block_size } => {
                format!("BlockSparse(block={})", block_size)
            }
            SparsePatternType::Random { num_random } => {
                format!("Random(k={})", num_random)
            }
        }
    }
}

/// Local attention (windowed attention)
///
/// Each query only attends to keys within a fixed window.
/// This is a special case of sparse attention optimized for efficiency.
#[derive(Clone, Debug)]
pub struct LocalAttention {
    /// Configuration
    pub config: AttentionConfig,
    /// Window size (attend to positions within ±window_size)
    pub window_size: usize,
}

impl LocalAttention {
    /// Create a new local attention component
    pub fn new(config: AttentionConfig, window_size: usize) -> Result<Self> {
        config.validate()?;

        if window_size == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "window_size must be positive".to_string(),
            });
        }

        Ok(Self {
            config,
            window_size,
        })
    }

    /// Build einsum graph for local attention
    ///
    /// Input tensors:
    /// - 0: Q (query) [batch, seq_len, d_model]
    /// - 1: K (key) [batch, seq_len, d_model]
    /// - 2: V (value) [batch, seq_len, d_model]
    ///
    /// Output tensors:
    /// - output: [batch, seq_len, d_model]
    pub fn build_local_attention_graph(&self, graph: &mut EinsumGraph) -> Result<Vec<usize>> {
        // Step 1: Compute attention scores
        let scores_tensor = graph.add_tensor("local_attn_scores");
        let scores_node = EinsumNode::new("bqd,bkd->bqk", vec![0, 1], vec![scores_tensor]);
        graph.add_node(scores_node)?;

        // Step 2: Scale scores
        let scale_factor = (self.config.d_k as f64).sqrt();
        let scale_tensor = graph.add_tensor("local_scale");
        let scaled_tensor = graph.add_tensor("local_scaled_scores");
        let scale_node = EinsumNode::elem_binary(
            format!("div_scalar_{}", scale_factor),
            scores_tensor,
            scale_tensor,
            scaled_tensor,
        );
        graph.add_node(scale_node)?;

        // Step 3: Apply local window mask
        // Mask is generated based on position distance: |i - j| <= window_size
        let window_mask_tensor = graph.add_tensor("local_window_mask");
        let masked_tensor = graph.add_tensor("local_masked_scores");
        let mask_node =
            EinsumNode::elem_binary("mul", scaled_tensor, window_mask_tensor, masked_tensor);
        graph.add_node(mask_node)?;

        // Step 4: Apply softmax
        let softmax_tensor = graph.add_tensor("local_attention_weights");
        let softmax_node = EinsumNode::elem_unary("softmax_k", masked_tensor, softmax_tensor);
        graph.add_node(softmax_node)?;

        // Step 5: Apply attention to values
        let output_tensor = graph.add_tensor("local_attn_output");
        let output_node =
            EinsumNode::new("bqk,bkv->bqv", vec![softmax_tensor, 2], vec![output_tensor]);
        graph.add_node(output_node)?;

        Ok(vec![output_tensor])
    }

    /// Get effective attention span
    pub fn attention_span(&self) -> usize {
        2 * self.window_size + 1
    }

    /// Calculate memory savings compared to full attention
    pub fn memory_savings(&self, seq_len: usize) -> f64 {
        let full_memory = seq_len * seq_len;
        let sparse_memory = seq_len * self.attention_span().min(seq_len);
        1.0 - (sparse_memory as f64 / full_memory as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_sparse_config() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = SparseAttentionConfig::strided(base, 4).unwrap();
        assert!(matches!(
            config.pattern,
            SparsePatternType::Strided { stride: 4 }
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_local_sparse_config() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = SparseAttentionConfig::local(base, 128).unwrap();
        assert!(matches!(
            config.pattern,
            SparsePatternType::Local { window_size: 128 }
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_global_local_sparse_config() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let global_positions = vec![0, 1, 2]; // First 3 tokens attend globally
        let config = SparseAttentionConfig::global_local(base, 64, global_positions).unwrap();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_block_sparse_config() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = SparseAttentionConfig::block_sparse(base, 64).unwrap();
        assert!(matches!(
            config.pattern,
            SparsePatternType::BlockSparse { block_size: 64 }
        ));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_sparse_attention_creation() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = SparseAttentionConfig::strided(base, 2).unwrap();
        let attn = SparseAttention::new(config).unwrap();
        assert_eq!(attn.sparsity_factor(1024), 0.5);
    }

    #[test]
    fn test_sparse_attention_graph_building() {
        let base = AttentionConfig::new(512, 8).unwrap();
        let config = SparseAttentionConfig::local(base, 128).unwrap();
        let attn = SparseAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");
        graph.add_tensor("sparse_mask");

        let outputs = attn.build_sparse_attention_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_local_attention_creation() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let local = LocalAttention::new(config, 64).unwrap();
        assert_eq!(local.window_size, 64);
        assert_eq!(local.attention_span(), 129);
    }

    #[test]
    fn test_local_attention_graph_building() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let local = LocalAttention::new(config, 64).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = local.build_local_attention_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(!graph.nodes.is_empty());
    }

    #[test]
    fn test_sparsity_factors() {
        let base = AttentionConfig::new(512, 8).unwrap();

        // Strided: 1/stride
        let strided = SparseAttentionConfig::strided(base.clone(), 4).unwrap();
        let attn = SparseAttention::new(strided).unwrap();
        assert!((attn.sparsity_factor(1024) - 0.25).abs() < 1e-10);

        // Local: window/seq_len
        let local = SparseAttentionConfig::local(base, 128).unwrap();
        let attn = SparseAttention::new(local).unwrap();
        assert!((attn.sparsity_factor(1024) - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_memory_savings() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let local = LocalAttention::new(config, 64).unwrap();

        // For seq_len=1024, window=64
        // Full: 1024*1024 = 1,048,576
        // Sparse: 1024*129 = 132,096
        // Savings: ~87.4%
        let savings = local.memory_savings(1024);
        assert!(savings > 0.87 && savings < 0.88);
    }

    #[test]
    fn test_pattern_descriptions() {
        let base = AttentionConfig::new(512, 8).unwrap();

        let strided = SparseAttentionConfig::strided(base.clone(), 4).unwrap();
        let attn = SparseAttention::new(strided).unwrap();
        assert_eq!(attn.pattern_description(), "Strided(stride=4)");

        let local = SparseAttentionConfig::local(base, 128).unwrap();
        let attn = SparseAttention::new(local).unwrap();
        assert_eq!(attn.pattern_description(), "Local(window=128)");
    }

    #[test]
    fn test_invalid_configs() {
        let base = AttentionConfig::new(512, 8).unwrap();

        // Zero stride
        let result = SparseAttentionConfig::strided(base.clone(), 0);
        assert!(result.is_err());

        // Zero window
        let result = SparseAttentionConfig::local(base.clone(), 0);
        assert!(result.is_err());

        // Zero block size
        let result = SparseAttentionConfig::block_sparse(base, 0);
        assert!(result.is_err());
    }
}
