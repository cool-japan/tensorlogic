//! Configuration types for transformer components.

use serde::{Deserialize, Serialize};

use crate::error::{TrustformersError, TrustformersResult};

/// Position encoding type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionEncodingType {
    /// Sinusoidal position encodings (Vaswani et al., 2017)
    Sinusoidal,
    /// Learned position embeddings
    Learned,
    /// Relative position encodings (Shaw et al., 2018)
    Relative,
    /// Rotary Position Embeddings (Su et al., 2021) - Used in LLaMA, GPT-NeoX
    RoPE,
    /// Attention with Linear Biases (Press et al., 2021) - Used in BLOOM
    ALiBi,
    /// No position encoding
    None,
}

/// Configuration for position encodings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionEncodingConfig {
    /// Type of position encoding
    pub encoding_type: PositionEncodingType,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Dropout probability (for learned embeddings)
    pub dropout: f64,
    /// Base frequency (for sinusoidal and RoPE)
    pub base: f64,
}

impl Default for PositionEncodingConfig {
    fn default() -> Self {
        Self {
            encoding_type: PositionEncodingType::Sinusoidal,
            max_seq_len: 512,
            hidden_size: 512,
            dropout: 0.1,
            base: 10000.0,
        }
    }
}

impl PositionEncodingConfig {
    /// Validate the configuration
    pub fn validate(&self) -> TrustformersResult<()> {
        if self.max_seq_len == 0 {
            return Err(TrustformersError::invalid_config(
                "max_seq_len must be > 0",
            ));
        }
        if self.hidden_size == 0 {
            return Err(TrustformersError::invalid_config(
                "hidden_size must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(TrustformersError::invalid_config(
                "dropout must be in [0, 1]",
            ));
        }
        if self.base <= 0.0 {
            return Err(TrustformersError::invalid_config("base must be > 0"));
        }
        Ok(())
    }
}

/// Attention pattern type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionPattern {
    /// Full attention (all-to-all)
    Full,
    /// Causal (autoregressive) attention
    Causal,
    /// Strided sparse attention
    Strided { stride: usize },
    /// Local attention with fixed window
    Local { window_size: usize },
    /// Block-sparse attention
    BlockSparse { block_size: usize },
    /// Global-local attention (some tokens attend globally)
    GlobalLocal { global_tokens: usize },
}

/// Configuration for attention mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Attention dropout probability
    pub attention_dropout: f64,
    /// Output dropout probability
    pub output_dropout: f64,
    /// Attention pattern
    pub pattern: AttentionPattern,
    /// Whether to use bias in QKV projection
    pub use_bias: bool,
    /// Scaling factor for attention scores (default: 1/sqrt(head_dim))
    pub scale: Option<f64>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 512,
            num_heads: 8,
            attention_dropout: 0.1,
            output_dropout: 0.1,
            pattern: AttentionPattern::Full,
            use_bias: true,
            scale: None,
        }
    }
}

impl AttentionConfig {
    /// Validate the configuration
    pub fn validate(&self) -> TrustformersResult<()> {
        if self.hidden_size == 0 {
            return Err(TrustformersError::invalid_config(
                "hidden_size must be > 0",
            ));
        }
        if self.num_heads == 0 {
            return Err(TrustformersError::invalid_config(
                "num_heads must be > 0",
            ));
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(TrustformersError::invalid_head_dimension(
                self.hidden_size,
                self.num_heads,
            ));
        }
        if !(0.0..=1.0).contains(&self.attention_dropout) {
            return Err(TrustformersError::invalid_config(
                "attention_dropout must be in [0, 1]",
            ));
        }
        if !(0.0..=1.0).contains(&self.output_dropout) {
            return Err(TrustformersError::invalid_config(
                "output_dropout must be in [0, 1]",
            ));
        }
        if let Some(scale) = self.scale {
            if scale <= 0.0 {
                return Err(TrustformersError::invalid_config("scale must be > 0"));
            }
        }
        Ok(())
    }

    /// Get the dimension of each attention head
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// Feed-forward network type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedForwardType {
    /// Standard FFN with two linear layers and activation
    Standard,
    /// Gated Linear Unit (GLU) variant
    GLU,
    /// Gated GELU Unit (GeGLU) - Used in T5
    GeGLU,
    /// Swish-Gated Linear Unit (SwiGLU) - Used in LLaMA
    SwiGLU,
}

/// Configuration for feed-forward networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardConfig {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Intermediate dimension size (typically 4 * hidden_size)
    pub intermediate_size: usize,
    /// FFN type
    pub ffn_type: FeedForwardType,
    /// Dropout probability
    pub dropout: f64,
    /// Whether to use bias
    pub use_bias: bool,
}

impl Default for FeedForwardConfig {
    fn default() -> Self {
        Self {
            hidden_size: 512,
            intermediate_size: 2048,
            ffn_type: FeedForwardType::Standard,
            dropout: 0.1,
            use_bias: true,
        }
    }
}

impl FeedForwardConfig {
    /// Validate the configuration
    pub fn validate(&self) -> TrustformersResult<()> {
        if self.hidden_size == 0 {
            return Err(TrustformersError::invalid_config(
                "hidden_size must be > 0",
            ));
        }
        if self.intermediate_size == 0 {
            return Err(TrustformersError::invalid_config(
                "intermediate_size must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(TrustformersError::invalid_config(
                "dropout must be in [0, 1]",
            ));
        }
        Ok(())
    }
}

/// Normalization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationType {
    /// Layer Normalization (Ba et al., 2016)
    LayerNorm,
    /// RMS Normalization (Zhang & Sennrich, 2019) - Used in LLaMA
    RMSNorm,
}

/// Configuration for normalization layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Normalization type
    pub norm_type: NormalizationType,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Whether to use affine transformation (scale and bias)
    pub elementwise_affine: bool,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            hidden_size: 512,
            norm_type: NormalizationType::LayerNorm,
            epsilon: 1e-5,
            elementwise_affine: true,
        }
    }
}

impl NormalizationConfig {
    /// Validate the configuration
    pub fn validate(&self) -> TrustformersResult<()> {
        if self.hidden_size == 0 {
            return Err(TrustformersError::invalid_config(
                "hidden_size must be > 0",
            ));
        }
        if self.epsilon <= 0.0 {
            return Err(TrustformersError::invalid_config("epsilon must be > 0"));
        }
        Ok(())
    }
}

/// Normalization position in transformer layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPosition {
    /// Pre-normalization (norm before attention/FFN)
    Pre,
    /// Post-normalization (norm after attention/FFN)
    Post,
}

/// Configuration for transformer encoder layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Attention configuration
    pub attention: AttentionConfig,
    /// Feed-forward configuration
    pub feedforward: FeedForwardConfig,
    /// Normalization configuration
    pub normalization: NormalizationConfig,
    /// Normalization position
    pub norm_position: NormPosition,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            attention: AttentionConfig::default(),
            feedforward: FeedForwardConfig::default(),
            normalization: NormalizationConfig::default(),
            norm_position: NormPosition::Pre,
        }
    }
}

impl EncoderConfig {
    /// Validate the configuration
    pub fn validate(&self) -> TrustformersResult<()> {
        self.attention.validate()?;
        self.feedforward.validate()?;
        self.normalization.validate()?;

        // Ensure dimensions match
        if self.attention.hidden_size != self.feedforward.hidden_size {
            return Err(TrustformersError::invalid_config(
                "attention and feedforward hidden_size must match",
            ));
        }
        if self.attention.hidden_size != self.normalization.hidden_size {
            return Err(TrustformersError::invalid_config(
                "attention and normalization hidden_size must match",
            ));
        }

        Ok(())
    }
}

/// Configuration for transformer decoder layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Self-attention configuration
    pub self_attention: AttentionConfig,
    /// Cross-attention configuration
    pub cross_attention: AttentionConfig,
    /// Feed-forward configuration
    pub feedforward: FeedForwardConfig,
    /// Normalization configuration
    pub normalization: NormalizationConfig,
    /// Normalization position
    pub norm_position: NormPosition,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        let mut self_attention = AttentionConfig::default();
        self_attention.pattern = AttentionPattern::Causal;

        Self {
            self_attention,
            cross_attention: AttentionConfig::default(),
            feedforward: FeedForwardConfig::default(),
            normalization: NormalizationConfig::default(),
            norm_position: NormPosition::Pre,
        }
    }
}

impl DecoderConfig {
    /// Validate the configuration
    pub fn validate(&self) -> TrustformersResult<()> {
        self.self_attention.validate()?;
        self.cross_attention.validate()?;
        self.feedforward.validate()?;
        self.normalization.validate()?;

        // Ensure dimensions match
        if self.self_attention.hidden_size != self.cross_attention.hidden_size {
            return Err(TrustformersError::invalid_config(
                "self_attention and cross_attention hidden_size must match",
            ));
        }
        if self.self_attention.hidden_size != self.feedforward.hidden_size {
            return Err(TrustformersError::invalid_config(
                "self_attention and feedforward hidden_size must match",
            ));
        }
        if self.self_attention.hidden_size != self.normalization.hidden_size {
            return Err(TrustformersError::invalid_config(
                "self_attention and normalization hidden_size must match",
            ));
        }

        Ok(())
    }
}

/// Complete model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Intermediate dimension size in FFN
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Position encoding configuration
    pub position_encoding: PositionEncodingConfig,
    /// Encoder configuration (for encoder-only or encoder-decoder models)
    pub encoder: Option<EncoderConfig>,
    /// Decoder configuration (for decoder-only or encoder-decoder models)
    pub decoder: Option<DecoderConfig>,
    /// Dropout probability for embeddings
    pub embedding_dropout: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            hidden_size: 512,
            num_layers: 6,
            num_heads: 8,
            intermediate_size: 2048,
            max_seq_len: 512,
            position_encoding: PositionEncodingConfig::default(),
            encoder: Some(EncoderConfig::default()),
            decoder: None,
            embedding_dropout: 0.1,
        }
    }
}

impl ModelConfig {
    /// Validate the configuration
    pub fn validate(&self) -> TrustformersResult<()> {
        if self.vocab_size == 0 {
            return Err(TrustformersError::invalid_config(
                "vocab_size must be > 0",
            ));
        }
        if self.hidden_size == 0 {
            return Err(TrustformersError::invalid_config(
                "hidden_size must be > 0",
            ));
        }
        if self.num_layers == 0 {
            return Err(TrustformersError::invalid_config(
                "num_layers must be > 0",
            ));
        }
        if self.num_heads == 0 {
            return Err(TrustformersError::invalid_config(
                "num_heads must be > 0",
            ));
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(TrustformersError::invalid_head_dimension(
                self.hidden_size,
                self.num_heads,
            ));
        }
        if !(0.0..=1.0).contains(&self.embedding_dropout) {
            return Err(TrustformersError::invalid_config(
                "embedding_dropout must be in [0, 1]",
            ));
        }

        self.position_encoding.validate()?;
        if let Some(ref encoder) = self.encoder {
            encoder.validate()?;
        }
        if let Some(ref decoder) = self.decoder {
            decoder.validate()?;
        }

        if self.encoder.is_none() && self.decoder.is_none() {
            return Err(TrustformersError::invalid_config(
                "At least one of encoder or decoder must be configured",
            ));
        }

        Ok(())
    }

    /// Create an encoder-only model configuration
    pub fn encoder_only(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> Self {
        let mut config = Self {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            ..Default::default()
        };
        config.encoder = Some(EncoderConfig::default());
        config.decoder = None;
        config
    }

    /// Create a decoder-only model configuration
    pub fn decoder_only(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> Self {
        let mut config = Self {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            ..Default::default()
        };
        config.encoder = None;
        config.decoder = Some(DecoderConfig::default());
        config
    }

    /// Create an encoder-decoder model configuration
    pub fn encoder_decoder(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            encoder: Some(EncoderConfig::default()),
            decoder: Some(DecoderConfig::default()),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_encoding_config_default() {
        let config = PositionEncodingConfig::default();
        assert_eq!(config.encoding_type, PositionEncodingType::Sinusoidal);
        assert_eq!(config.max_seq_len, 512);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_position_encoding_config_validation() {
        let mut config = PositionEncodingConfig::default();
        config.max_seq_len = 0;
        assert!(config.validate().is_err());

        config = PositionEncodingConfig::default();
        config.dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_attention_config_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim(), 64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_attention_config_validation() {
        let mut config = AttentionConfig::default();
        config.hidden_size = 0;
        assert!(config.validate().is_err());

        config = AttentionConfig::default();
        config.num_heads = 0;
        assert!(config.validate().is_err());

        config = AttentionConfig::default();
        config.hidden_size = 513; // Not divisible by num_heads (8)
        assert!(config.validate().is_err());

        config = AttentionConfig::default();
        config.attention_dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_feedforward_config_default() {
        let config = FeedForwardConfig::default();
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.intermediate_size, 2048);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_feedforward_config_validation() {
        let mut config = FeedForwardConfig::default();
        config.hidden_size = 0;
        assert!(config.validate().is_err());

        config = FeedForwardConfig::default();
        config.dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_normalization_config_default() {
        let config = NormalizationConfig::default();
        assert_eq!(config.norm_type, NormalizationType::LayerNorm);
        assert_eq!(config.epsilon, 1e-5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_encoder_config_default() {
        let config = EncoderConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_decoder_config_default() {
        let config = DecoderConfig::default();
        assert_eq!(
            config.self_attention.pattern,
            AttentionPattern::Causal
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_model_config_validation() {
        let config = ModelConfig::default();
        assert!(config.validate().is_ok());

        let config = ModelConfig::encoder_only(30000, 768, 12, 12, 3072);
        assert!(config.validate().is_ok());
        assert!(config.encoder.is_some());
        assert!(config.decoder.is_none());

        let config = ModelConfig::decoder_only(50000, 1024, 24, 16, 4096);
        assert!(config.validate().is_ok());
        assert!(config.encoder.is_none());
        assert!(config.decoder.is_some());

        let config = ModelConfig::encoder_decoder(30000, 512, 6, 8, 2048);
        assert!(config.validate().is_ok());
        assert!(config.encoder.is_some());
        assert!(config.decoder.is_some());
    }

    #[test]
    fn test_attention_pattern_types() {
        let patterns = vec![
            AttentionPattern::Full,
            AttentionPattern::Causal,
            AttentionPattern::Strided { stride: 2 },
            AttentionPattern::Local { window_size: 256 },
            AttentionPattern::BlockSparse { block_size: 64 },
            AttentionPattern::GlobalLocal { global_tokens: 4 },
        ];

        for pattern in patterns {
            let mut config = AttentionConfig::default();
            config.pattern = pattern;
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_feedforward_types() {
        let types = vec![
            FeedForwardType::Standard,
            FeedForwardType::GLU,
            FeedForwardType::GeGLU,
            FeedForwardType::SwiGLU,
        ];

        for ffn_type in types {
            let mut config = FeedForwardConfig::default();
            config.ffn_type = ffn_type;
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_position_encoding_types() {
        let types = vec![
            PositionEncodingType::Sinusoidal,
            PositionEncodingType::Learned,
            PositionEncodingType::Relative,
            PositionEncodingType::RoPE,
            PositionEncodingType::ALiBi,
            PositionEncodingType::None,
        ];

        for encoding_type in types {
            let mut config = PositionEncodingConfig::default();
            config.encoding_type = encoding_type;
            assert!(config.validate().is_ok());
        }
    }
}
