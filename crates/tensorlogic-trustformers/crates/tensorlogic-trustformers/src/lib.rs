//! # TensorLogic TrustformeRS
//!
//! Transformer components as logic-to-tensor compilations. This crate provides einsum-based
//! implementations of transformer architectures, expressing self-attention, feed-forward networks,
//! and other components as tensor logic operations.
//!
//! ## Architecture
//!
//! The crate is organized into several modules:
//! - `config`: Configuration types for transformer components
//! - `error`: Error types and handling
//! - `position`: Position encoding variants (sinusoidal, learned, relative, RoPE, ALiBi)
//! - `attention`: Self-attention and multi-head attention
//! - `feedforward`: Feed-forward networks (standard and gated)
//! - `normalization`: Layer normalization variants (LayerNorm, RMSNorm)
//! - `encoder`: Transformer encoder layers and stacks
//! - `decoder`: Transformer decoder layers and stacks
//! - `patterns`: Rule-based and sparse attention patterns
//! - `utils`: Utility functions (parameter counting, FLOP calculations)
//! - `presets`: Model presets (GPT-2/3, LLaMA, BLOOM, T5)
//!
//! ## Example
//!
//! ```rust,no_run
//! use tensorlogic_trustformers::presets::ModelPreset;
//! use tensorlogic_trustformers::encoder::EncoderStack;
//!
//! // Create a GPT-2 small configuration
//! let config = ModelPreset::Gpt2Small.to_config();
//!
//! // Build an encoder stack
//! let encoder = EncoderStack::new(
//!     config.num_layers,
//!     config.hidden_size,
//!     config.num_heads,
//!     config.intermediate_size,
//! );
//! ```

pub mod attention;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod feedforward;
pub mod normalization;
pub mod patterns;
pub mod position;
pub mod presets;
pub mod utils;

// Re-export commonly used types
pub use config::{
    AttentionConfig, DecoderConfig, EncoderConfig, FeedForwardConfig, ModelConfig,
    NormalizationConfig, PositionEncodingConfig,
};
pub use error::{TrustformersError, TrustformersResult};
pub use presets::ModelPreset;
