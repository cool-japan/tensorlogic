//! Model presets for common transformer architectures.
//!
//! This module provides pre-configured transformer architectures matching
//! popular models like GPT-2, GPT-3, LLaMA, BLOOM, and T5.

use crate::{error::Result, DecoderStackConfig, EncoderStackConfig};

/// Enum representing common transformer model presets
#[derive(Clone, Debug, PartialEq)]
pub enum ModelPreset {
    /// GPT-2 Small (117M parameters)
    Gpt2Small,
    /// GPT-2 Medium (345M parameters)
    Gpt2Medium,
    /// GPT-2 Large (774M parameters)
    Gpt2Large,
    /// GPT-2 XL (1.5B parameters)
    Gpt2Xl,
    /// GPT-3 Small (~125M parameters)
    Gpt3Small,
    /// LLaMA 7B
    LLaMA7B,
    /// LLaMA 13B
    LLaMA13B,
    /// LLaMA 33B (30B)
    LLaMA33B,
    /// LLaMA 65B
    LLaMA65B,
    /// BLOOM 176B
    Bloom176B,
    /// T5 Small (60M parameters) - encoder-decoder
    T5Small,
    /// T5 Base (220M parameters) - encoder-decoder
    T5Base,
    /// T5 Large (770M parameters) - encoder-decoder
    T5Large,
    /// T5 XL (3B parameters) - encoder-decoder
    T5Xl,
    /// T5 XXL (11B parameters) - encoder-decoder
    T5Xxl,
}

impl ModelPreset {
    /// Convert preset to encoder stack configuration
    ///
    /// For encoder-only models (GPT, LLaMA, BLOOM), returns the full model config.
    /// For encoder-decoder models (T5), returns only the encoder config.
    /// Use `to_encoder_decoder_config()` for T5 models to get both.
    pub fn to_config(&self) -> Result<EncoderStackConfig> {
        match self {
            Self::Gpt2Small => Ok(Self::gpt2_small()),
            Self::Gpt2Medium => Ok(Self::gpt2_medium()),
            Self::Gpt2Large => Ok(Self::gpt2_large()),
            Self::Gpt2Xl => Ok(Self::gpt2_xl()),
            Self::Gpt3Small => Ok(Self::gpt3_small()),
            Self::LLaMA7B => Ok(Self::llama_7b()),
            Self::LLaMA13B => Ok(Self::llama_13b()),
            Self::LLaMA33B => Ok(Self::llama_33b()),
            Self::LLaMA65B => Ok(Self::llama_65b()),
            Self::Bloom176B => Ok(Self::bloom_176b()),
            Self::T5Small => Ok(Self::t5_small().0),
            Self::T5Base => Ok(Self::t5_base().0),
            Self::T5Large => Ok(Self::t5_large().0),
            Self::T5Xl => Ok(Self::t5_xl().0),
            Self::T5Xxl => Ok(Self::t5_xxl().0),
        }
    }

    /// Convert preset to encoder-decoder configuration (for T5 models)
    pub fn to_encoder_decoder_config(&self) -> Result<(EncoderStackConfig, DecoderStackConfig)> {
        match self {
            Self::T5Small => Ok(Self::t5_small()),
            Self::T5Base => Ok(Self::t5_base()),
            Self::T5Large => Ok(Self::t5_large()),
            Self::T5Xl => Ok(Self::t5_xl()),
            Self::T5Xxl => Ok(Self::t5_xxl()),
            _ => Err(crate::error::TrustformerError::InvalidDimension {
                expected: 2,
                got: 1,
                context: format!("{:?} is not an encoder-decoder model", self),
            }),
        }
    }

    /// GPT-2 Small (117M parameters)
    fn gpt2_small() -> EncoderStackConfig {
        EncoderStackConfig::new(
            12,   // layers
            768,  // d_model
            12,   // n_heads
            3072, // d_ff (4 * d_model)
            1024, // max_seq_len
        )
        .unwrap()
        .with_dropout(0.1)
    }

    /// GPT-2 Medium (345M parameters)
    fn gpt2_medium() -> EncoderStackConfig {
        EncoderStackConfig::new(
            24,   // layers
            1024, // d_model
            16,   // n_heads
            4096, // d_ff
            1024, // max_seq_len
        )
        .unwrap()
        .with_dropout(0.1)
    }

    /// GPT-2 Large (774M parameters)
    fn gpt2_large() -> EncoderStackConfig {
        EncoderStackConfig::new(
            36,   // layers
            1280, // d_model
            20,   // n_heads
            5120, // d_ff
            1024, // max_seq_len
        )
        .unwrap()
        .with_dropout(0.1)
    }

    /// GPT-2 XL (1.5B parameters)
    fn gpt2_xl() -> EncoderStackConfig {
        EncoderStackConfig::new(
            48,   // layers
            1600, // d_model
            25,   // n_heads
            6400, // d_ff
            1024, // max_seq_len
        )
        .unwrap()
        .with_dropout(0.1)
    }

    /// GPT-3 Small (~125M parameters)
    fn gpt3_small() -> EncoderStackConfig {
        EncoderStackConfig::new(
            12,   // layers
            768,  // d_model
            12,   // n_heads
            3072, // d_ff
            2048, // max_seq_len (longer than GPT-2)
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// LLaMA 7B
    fn llama_7b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            32,    // layers
            4096,  // d_model
            32,    // n_heads
            11008, // d_ff (uses SwiGLU, ~2.7x d_model)
            2048,  // max_seq_len (can be extended with RoPE)
        )
        .unwrap()
        .with_dropout(0.0)
        .with_learned_position_encoding() // Would use RoPE in practice
    }

    /// LLaMA 13B
    fn llama_13b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            40,    // layers
            5120,  // d_model
            40,    // n_heads
            13824, // d_ff
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
        .with_learned_position_encoding()
    }

    /// LLaMA 33B (actually 30B)
    fn llama_33b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            60,    // layers
            6656,  // d_model
            52,    // n_heads
            17920, // d_ff
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
        .with_learned_position_encoding()
    }

    /// LLaMA 65B
    fn llama_65b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            80,    // layers
            8192,  // d_model
            64,    // n_heads
            22016, // d_ff
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
        .with_learned_position_encoding()
    }

    /// BLOOM 176B (full configuration)
    fn bloom_176b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            70,    // layers
            14336, // d_model
            112,   // n_heads (d_model must be divisible by n_heads: 14336 / 112 = 128)
            57344, // d_ff (4 * d_model)
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
        // Note: BLOOM uses ALiBi position encoding
    }

    /// T5 Small (60M parameters)
    fn t5_small() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(6, 512, 8, 2048, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(6, 512, 8, 2048, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }

    /// T5 Base (220M parameters)
    fn t5_base() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(12, 768, 12, 3072, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(12, 768, 12, 3072, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }

    /// T5 Large (770M parameters)
    fn t5_large() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(24, 1024, 16, 4096, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(24, 1024, 16, 4096, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }

    /// T5 XL (3B parameters)
    fn t5_xl() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(24, 2048, 32, 8192, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(24, 2048, 32, 8192, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }

    /// T5 XXL (11B parameters)
    fn t5_xxl() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(24, 4096, 64, 16384, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(24, 4096, 64, 16384, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_small_preset() {
        let config = ModelPreset::Gpt2Small.to_config().unwrap();
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.layer_config.attention.d_model, 768);
        assert_eq!(config.layer_config.attention.n_heads, 12);
        assert_eq!(config.layer_config.feed_forward.d_ff, 3072);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gpt2_medium_preset() {
        let config = ModelPreset::Gpt2Medium.to_config().unwrap();
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.layer_config.attention.d_model, 1024);
        assert_eq!(config.layer_config.attention.n_heads, 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gpt2_large_preset() {
        let config = ModelPreset::Gpt2Large.to_config().unwrap();
        assert_eq!(config.num_layers, 36);
        assert_eq!(config.layer_config.attention.d_model, 1280);
        assert_eq!(config.layer_config.attention.n_heads, 20);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gpt2_xl_preset() {
        let config = ModelPreset::Gpt2Xl.to_config().unwrap();
        assert_eq!(config.num_layers, 48);
        assert_eq!(config.layer_config.attention.d_model, 1600);
        assert_eq!(config.layer_config.attention.n_heads, 25);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gpt3_small_preset() {
        let config = ModelPreset::Gpt3Small.to_config().unwrap();
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.layer_config.attention.d_model, 768);
        assert_eq!(config.position_encoding.max_seq_len, 2048);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_llama_7b_preset() {
        let config = ModelPreset::LLaMA7B.to_config().unwrap();
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.layer_config.attention.d_model, 4096);
        assert_eq!(config.layer_config.attention.n_heads, 32);
        assert_eq!(config.layer_config.feed_forward.d_ff, 11008);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_llama_13b_preset() {
        let config = ModelPreset::LLaMA13B.to_config().unwrap();
        assert_eq!(config.num_layers, 40);
        assert_eq!(config.layer_config.attention.d_model, 5120);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_llama_33b_preset() {
        let config = ModelPreset::LLaMA33B.to_config().unwrap();
        assert_eq!(config.num_layers, 60);
        assert_eq!(config.layer_config.attention.d_model, 6656);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_llama_65b_preset() {
        let config = ModelPreset::LLaMA65B.to_config().unwrap();
        assert_eq!(config.num_layers, 80);
        assert_eq!(config.layer_config.attention.d_model, 8192);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_bloom_176b_preset() {
        let config = ModelPreset::Bloom176B.to_config().unwrap();
        assert_eq!(config.num_layers, 70);
        assert_eq!(config.layer_config.attention.d_model, 14336);
        assert_eq!(config.layer_config.attention.n_heads, 112);
        // Verify d_model is divisible by n_heads
        assert_eq!(14336 % 112, 0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_t5_small_preset() {
        let (encoder, decoder) = ModelPreset::T5Small.to_encoder_decoder_config().unwrap();
        assert_eq!(encoder.num_layers, 6);
        assert_eq!(decoder.num_layers, 6);
        assert_eq!(encoder.layer_config.attention.d_model, 512);
        assert_eq!(decoder.layer_config.self_attention.d_model, 512);
        assert!(encoder.validate().is_ok());
        assert!(decoder.validate().is_ok());
    }

    #[test]
    fn test_t5_base_preset() {
        let (encoder, decoder) = ModelPreset::T5Base.to_encoder_decoder_config().unwrap();
        assert_eq!(encoder.num_layers, 12);
        assert_eq!(decoder.num_layers, 12);
        assert_eq!(encoder.layer_config.attention.d_model, 768);
        assert!(encoder.validate().is_ok());
        assert!(decoder.validate().is_ok());
    }

    #[test]
    fn test_t5_large_preset() {
        let (encoder, decoder) = ModelPreset::T5Large.to_encoder_decoder_config().unwrap();
        assert_eq!(encoder.num_layers, 24);
        assert_eq!(decoder.num_layers, 24);
        assert_eq!(encoder.layer_config.attention.d_model, 1024);
        assert!(encoder.validate().is_ok());
        assert!(decoder.validate().is_ok());
    }

    #[test]
    fn test_t5_xl_preset() {
        let (encoder, decoder) = ModelPreset::T5Xl.to_encoder_decoder_config().unwrap();
        assert_eq!(encoder.num_layers, 24);
        assert_eq!(decoder.num_layers, 24);
        assert_eq!(encoder.layer_config.attention.d_model, 2048);
        assert!(encoder.validate().is_ok());
        assert!(decoder.validate().is_ok());
    }

    #[test]
    fn test_t5_xxl_preset() {
        let (encoder, decoder) = ModelPreset::T5Xxl.to_encoder_decoder_config().unwrap();
        assert_eq!(encoder.num_layers, 24);
        assert_eq!(decoder.num_layers, 24);
        assert_eq!(encoder.layer_config.attention.d_model, 4096);
        assert!(encoder.validate().is_ok());
        assert!(decoder.validate().is_ok());
    }

    #[test]
    fn test_preset_enum_equality() {
        assert_eq!(ModelPreset::Gpt2Small, ModelPreset::Gpt2Small);
        assert_ne!(ModelPreset::Gpt2Small, ModelPreset::Gpt2Medium);
    }

    #[test]
    fn test_encoder_decoder_error_for_gpt() {
        let result = ModelPreset::Gpt2Small.to_encoder_decoder_config();
        assert!(result.is_err());
    }

    #[test]
    fn test_all_presets_validate() {
        // Verify all presets produce valid configurations
        assert!(ModelPreset::Gpt2Small
            .to_config()
            .unwrap()
            .validate()
            .is_ok());
        assert!(ModelPreset::Gpt2Medium
            .to_config()
            .unwrap()
            .validate()
            .is_ok());
        assert!(ModelPreset::Gpt2Large
            .to_config()
            .unwrap()
            .validate()
            .is_ok());
        assert!(ModelPreset::Gpt2Xl.to_config().unwrap().validate().is_ok());
        assert!(ModelPreset::Gpt3Small
            .to_config()
            .unwrap()
            .validate()
            .is_ok());
        assert!(ModelPreset::LLaMA7B.to_config().unwrap().validate().is_ok());
        assert!(ModelPreset::LLaMA13B
            .to_config()
            .unwrap()
            .validate()
            .is_ok());
        assert!(ModelPreset::LLaMA33B
            .to_config()
            .unwrap()
            .validate()
            .is_ok());
        assert!(ModelPreset::LLaMA65B
            .to_config()
            .unwrap()
            .validate()
            .is_ok());
        assert!(ModelPreset::Bloom176B
            .to_config()
            .unwrap()
            .validate()
            .is_ok());
    }
}
