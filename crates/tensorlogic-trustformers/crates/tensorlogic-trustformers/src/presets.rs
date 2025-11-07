//! Model preset configurations for popular transformer architectures.

use crate::config::{
    AttentionConfig, AttentionPattern, DecoderConfig, EncoderConfig, FeedForwardConfig,
    FeedForwardType, ModelConfig, NormPosition, NormalizationConfig, NormalizationType,
    PositionEncodingConfig, PositionEncodingType,
};

/// Preset configurations for well-known transformer models
#[derive(Debug, Clone, Copy)]
pub enum ModelPreset {
    /// GPT-2 Small (117M parameters)
    Gpt2Small,
    /// GPT-2 Medium (345M parameters)
    Gpt2Medium,
    /// GPT-2 Large (774M parameters)
    Gpt2Large,
    /// GPT-2 XL (1.5B parameters)
    Gpt2Xl,
    /// GPT-3 Small (125M parameters, similar to GPT-2 Small but with different training)
    Gpt3Small,
    /// LLaMA 7B
    LLaMA7B,
    /// LLaMA 13B
    LLaMA13B,
    /// LLaMA 33B
    LLaMA33B,
    /// LLaMA 65B
    LLaMA65B,
    /// BLOOM 176B
    Bloom176B,
    /// T5 Small (60M parameters)
    T5Small,
    /// T5 Base (220M parameters)
    T5Base,
    /// T5 Large (770M parameters)
    T5Large,
    /// T5 XL (3B parameters)
    T5Xl,
    /// T5 XXL (11B parameters)
    T5Xxl,
}

impl ModelPreset {
    /// Convert preset to ModelConfig
    pub fn to_config(self) -> ModelConfig {
        match self {
            ModelPreset::Gpt2Small => gpt2_small_config(),
            ModelPreset::Gpt2Medium => gpt2_medium_config(),
            ModelPreset::Gpt2Large => gpt2_large_config(),
            ModelPreset::Gpt2Xl => gpt2_xl_config(),
            ModelPreset::Gpt3Small => gpt3_small_config(),
            ModelPreset::LLaMA7B => llama_7b_config(),
            ModelPreset::LLaMA13B => llama_13b_config(),
            ModelPreset::LLaMA33B => llama_33b_config(),
            ModelPreset::LLaMA65B => llama_65b_config(),
            ModelPreset::Bloom176B => bloom_176b_config(),
            ModelPreset::T5Small => t5_small_config(),
            ModelPreset::T5Base => t5_base_config(),
            ModelPreset::T5Large => t5_large_config(),
            ModelPreset::T5Xl => t5_xl_config(),
            ModelPreset::T5Xxl => t5_xxl_config(),
        }
    }
}

fn gpt2_small_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 50257,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12,
        intermediate_size: 3072,
        max_seq_len: 1024,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Learned,
            max_seq_len: 1024,
            hidden_size: 768,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config(768, 12, 3072, NormalizationType::LayerNorm, true)),
        embedding_dropout: 0.1,
    }
}

fn gpt2_medium_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 50257,
        hidden_size: 1024,
        num_layers: 24,
        num_heads: 16,
        intermediate_size: 4096,
        max_seq_len: 1024,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Learned,
            max_seq_len: 1024,
            hidden_size: 1024,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config(1024, 16, 4096, NormalizationType::LayerNorm, true)),
        embedding_dropout: 0.1,
    }
}

fn gpt2_large_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 50257,
        hidden_size: 1280,
        num_layers: 36,
        num_heads: 20,
        intermediate_size: 5120,
        max_seq_len: 1024,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Learned,
            max_seq_len: 1024,
            hidden_size: 1280,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config(1280, 20, 5120, NormalizationType::LayerNorm, true)),
        embedding_dropout: 0.1,
    }
}

fn gpt2_xl_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 50257,
        hidden_size: 1600,
        num_layers: 48,
        num_heads: 25,
        intermediate_size: 6400,
        max_seq_len: 1024,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Learned,
            max_seq_len: 1024,
            hidden_size: 1600,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config(1600, 25, 6400, NormalizationType::LayerNorm, true)),
        embedding_dropout: 0.1,
    }
}

fn gpt3_small_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 50257,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12,
        intermediate_size: 3072,
        max_seq_len: 2048,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Learned,
            max_seq_len: 2048,
            hidden_size: 768,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config(768, 12, 3072, NormalizationType::LayerNorm, true)),
        embedding_dropout: 0.1,
    }
}

fn llama_7b_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32000,
        hidden_size: 4096,
        num_layers: 32,
        num_heads: 32,
        intermediate_size: 11008,
        max_seq_len: 2048,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::RoPE,
            max_seq_len: 2048,
            hidden_size: 4096,
            dropout: 0.0,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config_swiglu(4096, 32, 11008, NormalizationType::RMSNorm, false)),
        embedding_dropout: 0.0,
    }
}

fn llama_13b_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32000,
        hidden_size: 5120,
        num_layers: 40,
        num_heads: 40,
        intermediate_size: 13824,
        max_seq_len: 2048,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::RoPE,
            max_seq_len: 2048,
            hidden_size: 5120,
            dropout: 0.0,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config_swiglu(5120, 40, 13824, NormalizationType::RMSNorm, false)),
        embedding_dropout: 0.0,
    }
}

fn llama_33b_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32000,
        hidden_size: 6656,
        num_layers: 60,
        num_heads: 52,
        intermediate_size: 17920,
        max_seq_len: 2048,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::RoPE,
            max_seq_len: 2048,
            hidden_size: 6656,
            dropout: 0.0,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config_swiglu(6656, 52, 17920, NormalizationType::RMSNorm, false)),
        embedding_dropout: 0.0,
    }
}

fn llama_65b_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32000,
        hidden_size: 8192,
        num_layers: 80,
        num_heads: 64,
        intermediate_size: 22016,
        max_seq_len: 2048,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::RoPE,
            max_seq_len: 2048,
            hidden_size: 8192,
            dropout: 0.0,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config_swiglu(8192, 64, 22016, NormalizationType::RMSNorm, false)),
        embedding_dropout: 0.0,
    }
}

fn bloom_176b_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 250880,
        hidden_size: 14336,
        num_layers: 70,
        num_heads: 112,
        intermediate_size: 57344,
        max_seq_len: 2048,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::ALiBi,
            max_seq_len: 2048,
            hidden_size: 14336,
            dropout: 0.0,
            base: 10000.0,
        },
        encoder: None,
        decoder: Some(decoder_config(14336, 112, 57344, NormalizationType::LayerNorm, true)),
        embedding_dropout: 0.0,
    }
}

fn t5_small_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32128,
        hidden_size: 512,
        num_layers: 6,
        num_heads: 8,
        intermediate_size: 2048,
        max_seq_len: 512,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Relative,
            max_seq_len: 512,
            hidden_size: 512,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: Some(encoder_config(512, 8, 2048, NormalizationType::RMSNorm, false)),
        decoder: Some(decoder_config_geglu(512, 8, 2048, NormalizationType::RMSNorm, false)),
        embedding_dropout: 0.1,
    }
}

fn t5_base_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32128,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12,
        intermediate_size: 3072,
        max_seq_len: 512,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Relative,
            max_seq_len: 512,
            hidden_size: 768,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: Some(encoder_config(768, 12, 3072, NormalizationType::RMSNorm, false)),
        decoder: Some(decoder_config_geglu(768, 12, 3072, NormalizationType::RMSNorm, false)),
        embedding_dropout: 0.1,
    }
}

fn t5_large_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32128,
        hidden_size: 1024,
        num_layers: 24,
        num_heads: 16,
        intermediate_size: 4096,
        max_seq_len: 512,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Relative,
            max_seq_len: 512,
            hidden_size: 1024,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: Some(encoder_config(1024, 16, 4096, NormalizationType::RMSNorm, false)),
        decoder: Some(decoder_config_geglu(1024, 16, 4096, NormalizationType::RMSNorm, false)),
        embedding_dropout: 0.1,
    }
}

fn t5_xl_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32128,
        hidden_size: 2048,
        num_layers: 24,
        num_heads: 32,
        intermediate_size: 8192,
        max_seq_len: 512,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Relative,
            max_seq_len: 512,
            hidden_size: 2048,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: Some(encoder_config(2048, 32, 8192, NormalizationType::RMSNorm, false)),
        decoder: Some(decoder_config_geglu(2048, 32, 8192, NormalizationType::RMSNorm, false)),
        embedding_dropout: 0.1,
    }
}

fn t5_xxl_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 32128,
        hidden_size: 4096,
        num_layers: 24,
        num_heads: 64,
        intermediate_size: 16384,
        max_seq_len: 512,
        position_encoding: PositionEncodingConfig {
            encoding_type: PositionEncodingType::Relative,
            max_seq_len: 512,
            hidden_size: 4096,
            dropout: 0.1,
            base: 10000.0,
        },
        encoder: Some(encoder_config(4096, 64, 16384, NormalizationType::RMSNorm, false)),
        decoder: Some(decoder_config_geglu(4096, 64, 16384, NormalizationType::RMSNorm, false)),
        embedding_dropout: 0.1,
    }
}

// Helper functions to create encoder/decoder configs

fn encoder_config(
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    norm_type: NormalizationType,
    use_bias: bool,
) -> EncoderConfig {
    EncoderConfig {
        attention: AttentionConfig {
            hidden_size,
            num_heads,
            attention_dropout: 0.1,
            output_dropout: 0.1,
            pattern: AttentionPattern::Full,
            use_bias,
            scale: None,
        },
        feedforward: FeedForwardConfig {
            hidden_size,
            intermediate_size,
            ffn_type: FeedForwardType::Standard,
            dropout: 0.1,
            use_bias,
        },
        normalization: NormalizationConfig {
            hidden_size,
            norm_type,
            epsilon: 1e-6,
            elementwise_affine: true,
        },
        norm_position: NormPosition::Pre,
    }
}

fn decoder_config(
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    norm_type: NormalizationType,
    use_bias: bool,
) -> DecoderConfig {
    DecoderConfig {
        self_attention: AttentionConfig {
            hidden_size,
            num_heads,
            attention_dropout: 0.1,
            output_dropout: 0.1,
            pattern: AttentionPattern::Causal,
            use_bias,
            scale: None,
        },
        cross_attention: AttentionConfig {
            hidden_size,
            num_heads,
            attention_dropout: 0.1,
            output_dropout: 0.1,
            pattern: AttentionPattern::Full,
            use_bias,
            scale: None,
        },
        feedforward: FeedForwardConfig {
            hidden_size,
            intermediate_size,
            ffn_type: FeedForwardType::Standard,
            dropout: 0.1,
            use_bias,
        },
        normalization: NormalizationConfig {
            hidden_size,
            norm_type,
            epsilon: 1e-6,
            elementwise_affine: true,
        },
        norm_position: NormPosition::Pre,
    }
}

fn decoder_config_swiglu(
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    norm_type: NormalizationType,
    use_bias: bool,
) -> DecoderConfig {
    let mut config = decoder_config(hidden_size, num_heads, intermediate_size, norm_type, use_bias);
    config.feedforward.ffn_type = FeedForwardType::SwiGLU;
    config
}

fn decoder_config_geglu(
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    norm_type: NormalizationType,
    use_bias: bool,
) -> DecoderConfig {
    let mut config = decoder_config(hidden_size, num_heads, intermediate_size, norm_type, use_bias);
    config.feedforward.ffn_type = FeedForwardType::GeGLU;
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_small() {
        let config = ModelPreset::Gpt2Small.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
    }

    #[test]
    fn test_gpt2_medium() {
        let config = ModelPreset::Gpt2Medium.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 1024);
    }

    #[test]
    fn test_gpt2_large() {
        let config = ModelPreset::Gpt2Large.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 1280);
    }

    #[test]
    fn test_gpt2_xl() {
        let config = ModelPreset::Gpt2Xl.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 1600);
    }

    #[test]
    fn test_gpt3_small() {
        let config = ModelPreset::Gpt3Small.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_seq_len, 2048);
    }

    #[test]
    fn test_llama_7b() {
        let config = ModelPreset::LLaMA7B.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.position_encoding.encoding_type, PositionEncodingType::RoPE);
    }

    #[test]
    fn test_llama_13b() {
        let config = ModelPreset::LLaMA13B.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 5120);
    }

    #[test]
    fn test_llama_33b() {
        let config = ModelPreset::LLaMA33B.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 6656);
    }

    #[test]
    fn test_llama_65b() {
        let config = ModelPreset::LLaMA65B.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 8192);
    }

    #[test]
    fn test_bloom_176b() {
        let config = ModelPreset::Bloom176B.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.position_encoding.encoding_type, PositionEncodingType::ALiBi);
    }

    #[test]
    fn test_t5_small() {
        let config = ModelPreset::T5Small.to_config();
        assert!(config.validate().is_ok());
        assert!(config.encoder.is_some());
        assert!(config.decoder.is_some());
    }

    #[test]
    fn test_t5_base() {
        let config = ModelPreset::T5Base.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 768);
    }

    #[test]
    fn test_t5_large() {
        let config = ModelPreset::T5Large.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 1024);
    }

    #[test]
    fn test_t5_xl() {
        let config = ModelPreset::T5Xl.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 2048);
    }

    #[test]
    fn test_t5_xxl() {
        let config = ModelPreset::T5Xxl.to_config();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 4096);
    }
}
