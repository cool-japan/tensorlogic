//! # Tensorlogic-Trustformers
//!
//! **Version**: 0.1.0-alpha.2 | **Status**: Production Ready
//!
//! Transform transformer architectures into TensorLogic IR using einsum operations.
//!
//! This crate provides implementations of transformer components (self-attention,
//! multi-head attention, feed-forward networks) as einsum graphs that can be
//! compiled and executed on various TensorLogic backends.
//!
//! ## Features
//!
//! - **Self-Attention**: Scaled dot-product attention as einsum operations
//! - **Multi-Head Attention**: Parallel attention heads with head splitting
//! - **Feed-Forward Networks**: Position-wise FFN with configurable activations
//! - **Gated FFN**: GLU-style gated feed-forward networks
//! - **Einsum-Native**: All operations expressed as einsum for maximum flexibility
//!
//! ## Architecture
//!
//! Transformer components are decomposed into einsum operations:
//!
//! ### Self-Attention
//! ```text
//! scores = einsum("bqd,bkd->bqk", Q, K) / sqrt(d_k)
//! attn = softmax(scores, dim=-1)
//! output = einsum("bqk,bkv->bqv", attn, V)
//! ```
//!
//! ### Multi-Head Attention
//! ```text
//! Q, K, V = [batch, seq, d_model] -> [batch, n_heads, seq, d_k]
//! scores = einsum("bhqd,bhkd->bhqk", Q, K) / sqrt(d_k)
//! attn = softmax(scores, dim=-1)
//! output = einsum("bhqk,bhkv->bhqv", attn, V)
//! output = reshape([batch, seq, d_model])
//! ```
//!
//! ### Feed-Forward Network
//! ```text
//! h1 = einsum("bsd,df->bsf", x, W1) + b1
//! h2 = activation(h1)
//! output = einsum("bsf,fd->bsd", h2, W2) + b2
//! ```
//!
//! ## Example Usage
//!
//! ```rust
//! use tensorlogic_trustformers::{
//!     AttentionConfig, SelfAttention, MultiHeadAttention,
//!     FeedForwardConfig, FeedForward,
//! };
//! use tensorlogic_ir::EinsumGraph;
//!
//! // Configure self-attention
//! let attn_config = AttentionConfig::new(512, 8).unwrap();
//! let self_attn = SelfAttention::new(attn_config.clone()).unwrap();
//!
//! // Build einsum graph
//! let mut graph = EinsumGraph::new();
//! graph.add_tensor("Q");
//! graph.add_tensor("K");
//! graph.add_tensor("V");
//!
//! let outputs = self_attn.build_attention_graph(&mut graph).unwrap();
//!
//! // Configure multi-head attention
//! let mha = MultiHeadAttention::new(attn_config).unwrap();
//! let mut mha_graph = EinsumGraph::new();
//! mha_graph.add_tensor("Q");
//! mha_graph.add_tensor("K");
//! mha_graph.add_tensor("V");
//!
//! let mha_outputs = mha.build_mha_graph(&mut mha_graph).unwrap();
//!
//! // Configure feed-forward network
//! let ffn_config = FeedForwardConfig::new(512, 2048)
//!     .with_activation("gelu");
//! let ffn = FeedForward::new(ffn_config).unwrap();
//!
//! let mut ffn_graph = EinsumGraph::new();
//! ffn_graph.add_tensor("x");
//! ffn_graph.add_tensor("W1");
//! ffn_graph.add_tensor("b1");
//! ffn_graph.add_tensor("W2");
//! ffn_graph.add_tensor("b2");
//!
//! let ffn_outputs = ffn.build_ffn_graph(&mut ffn_graph).unwrap();
//! ```
//!
//! ## Integration with TensorLogic
//!
//! The einsum graphs produced by this crate can be:
//! - Compiled with `tensorlogic-compiler`
//! - Executed on `tensorlogic-scirs-backend` or other backends
//! - Optimized using graph optimization passes
//! - Combined with logical rules for interpretable transformers
//!
//! ## Design Philosophy
//!
//! This crate follows the TensorLogic principle of expressing neural operations
//! as tensor contractions (einsum), enabling:
//!
//! 1. **Backend Independence**: Same graph works on CPU, GPU, TPU
//! 2. **Optimization Opportunities**: Graph-level optimizations like fusion
//! 3. **Interpretability**: Clear mathematical semantics
//! 4. **Composability**: Mix transformer layers with logical rules

pub mod attention;
pub mod checkpointing;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod ffn;
pub mod flash_attention;
pub mod gqa;
pub mod kv_cache;
pub mod layers;
pub mod lora;
pub mod moe;
pub mod normalization;
pub mod patterns;
pub mod position;
pub mod presets;
pub mod rule_attention;
pub mod sliding_window;
pub mod sparse_attention;
pub mod stacks;
pub mod trustformers_integration;
pub mod utils;
pub mod vision;

// Re-export main types for convenient access
pub use attention::{MultiHeadAttention, SelfAttention};
pub use checkpointing::{CheckpointConfig, CheckpointStrategy};
pub use config::{AttentionConfig, FeedForwardConfig, TransformerLayerConfig};
pub use decoder::{Decoder, DecoderConfig};
pub use encoder::{Encoder, EncoderConfig};
pub use error::{Result, TrustformerError};
pub use ffn::{FeedForward, GatedFeedForward};
pub use flash_attention::{
    FlashAttention, FlashAttentionConfig, FlashAttentionPreset, FlashAttentionStats,
    FlashAttentionV2Config,
};
pub use gqa::{GQAConfig, GQAPreset, GQAStats, GroupedQueryAttention};
pub use kv_cache::{CacheStats, KVCache, KVCacheConfig};
pub use layers::{DecoderLayer, DecoderLayerConfig, EncoderLayer, EncoderLayerConfig};
pub use lora::{LoRAAttention, LoRAConfig, LoRALinear, LoRAPreset, LoRAStats};
pub use moe::{MoeConfig, MoeLayer, MoePreset, MoeStats, RouterType};
pub use normalization::{LayerNorm, LayerNormConfig, RMSNorm};
pub use patterns::{
    AttentionMask, BlockSparseMask, CausalMask, GlobalLocalMask, LocalMask, RuleBasedMask,
    RulePattern, StridedMask,
};
pub use position::{
    AlibiPositionEncoding, LearnedPositionEncoding, PositionEncodingConfig, PositionEncodingType,
    RelativePositionEncoding, RotaryPositionEncoding, SinusoidalPositionEncoding,
};
pub use presets::ModelPreset;
pub use rule_attention::{
    RuleAttentionConfig, RuleAttentionType, RuleBasedAttention, StructuredAttention,
};
pub use sliding_window::{
    SlidingWindowAttention, SlidingWindowConfig, SlidingWindowPreset, SlidingWindowStats,
};
pub use sparse_attention::{
    LocalAttention, SparseAttention, SparseAttentionConfig, SparsePatternType,
};
pub use stacks::{DecoderStack, DecoderStackConfig, EncoderStack, EncoderStackConfig};
pub use trustformers_integration::{
    CheckpointData, IntegrationConfig, ModelConfig, TensorLogicModel, TrustformersConverter,
    TrustformersWeightLoader,
};
pub use utils::{decoder_stack_stats, encoder_stack_stats, ModelStats};
pub use vision::{
    PatchEmbedding, PatchEmbeddingConfig, ViTPreset, VisionTransformer, VisionTransformerConfig,
};

// Legacy compatibility (keep for backward compatibility)
#[deprecated(since = "0.1.0", note = "Use AttentionConfig instead")]
pub type AttnSpec = AttentionConfig;

#[deprecated(
    since = "0.1.0",
    note = "Use SelfAttention::build_attention_graph instead"
)]
pub fn self_attention_as_rules(_spec: &AttentionConfig) {
    // Legacy function - use SelfAttention::build_attention_graph instead
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::EinsumGraph;

    #[test]
    fn test_end_to_end_self_attention() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let attn = SelfAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = attn.build_attention_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_end_to_end_multi_head_attention() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let mha = MultiHeadAttention::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("Q");
        graph.add_tensor("K");
        graph.add_tensor("V");

        let outputs = mha.build_mha_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_end_to_end_ffn() {
        let config = FeedForwardConfig::new(512, 2048);
        let ffn = FeedForward::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");
        graph.add_tensor("W1");
        graph.add_tensor("b1");
        graph.add_tensor("W2");
        graph.add_tensor("b2");

        let outputs = ffn.build_ffn_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_end_to_end_gated_ffn() {
        let config = FeedForwardConfig::new(512, 2048);
        let glu = GatedFeedForward::new(config).unwrap();

        let mut graph = EinsumGraph::new();
        graph.add_tensor("x");
        graph.add_tensor("W_gate");
        graph.add_tensor("W_value");
        graph.add_tensor("W_out");

        let outputs = glu.build_glu_graph(&mut graph).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_transformer_layer_config() {
        let config = TransformerLayerConfig::new(512, 8, 2048).unwrap();
        assert_eq!(config.attention.d_model, 512);
        assert_eq!(config.attention.n_heads, 8);
        assert_eq!(config.feed_forward.d_ff, 2048);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = AttentionConfig::new(512, 8)
            .unwrap()
            .with_causal(true)
            .with_dropout(0.1);

        assert!(config.causal);
        assert!((config.dropout - 0.1).abs() < 1e-10);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ffn_config_builder() {
        let config = FeedForwardConfig::new(512, 2048)
            .with_activation("relu")
            .with_dropout(0.1);

        assert_eq!(config.activation, "relu");
        assert!((config.dropout - 0.1).abs() < 1e-10);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_configurations() {
        // Invalid head count
        let result = AttentionConfig::new(512, 7);
        assert!(result.is_err());

        // Valid configuration
        let result = AttentionConfig::new(512, 8);
        assert!(result.is_ok());
    }
}
