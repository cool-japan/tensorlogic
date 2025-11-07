//! Utility functions for transformer models.
//!
//! This module provides helper functions for common transformer operations:
//! - Parameter counting
//! - Configuration validation
//! - Dimension calculations
//! - Model statistics

use crate::{
    AttentionConfig, DecoderLayerConfig, DecoderStackConfig, EncoderLayerConfig,
    EncoderStackConfig, FeedForwardConfig, LayerNormConfig,
};

/// Statistics about a transformer model
#[derive(Clone, Debug, PartialEq)]
pub struct ModelStats {
    /// Total number of parameters
    pub total_params: usize,
    /// Number of trainable parameters
    pub trainable_params: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Model dimension
    pub d_model: usize,
    /// Memory footprint estimate (bytes)
    pub memory_estimate: usize,
}

impl ModelStats {
    /// Format as human-readable string
    pub fn summary(&self) -> String {
        format!(
            "ModelStats:\n  Total params: {}\n  Trainable: {}\n  Layers: {}\n  d_model: {}\n  Memory: {} MB",
            Self::format_number(self.total_params),
            Self::format_number(self.trainable_params),
            self.num_layers,
            self.d_model,
            self.memory_estimate / (1024 * 1024)
        )
    }

    fn format_number(n: usize) -> String {
        if n >= 1_000_000_000 {
            format!("{:.2}B", n as f64 / 1_000_000_000.0)
        } else if n >= 1_000_000 {
            format!("{:.2}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.2}K", n as f64 / 1_000.0)
        } else {
            n.to_string()
        }
    }
}

/// Count parameters in attention layer
pub fn count_attention_params(config: &AttentionConfig) -> usize {
    let d_model = config.d_model;

    // Q, K, V projection matrices: 3 * (d_model * d_model)
    let qkv_params = 3 * d_model * d_model;

    // Output projection: d_model * d_model
    let out_params = d_model * d_model;

    // Biases for Q, K, V, and output (optional, typically included)
    let bias_params = 4 * d_model;

    qkv_params + out_params + bias_params
}

/// Count parameters in feed-forward network
pub fn count_ffn_params(config: &FeedForwardConfig) -> usize {
    let d_model = config.d_model;
    let d_ff = config.d_ff;

    // First layer: d_model * d_ff + d_ff (weights + bias)
    let layer1_params = d_model * d_ff + d_ff;

    // Second layer: d_ff * d_model + d_model (weights + bias)
    let layer2_params = d_ff * d_model + d_model;

    layer1_params + layer2_params
}

/// Count parameters in layer normalization
pub fn count_layernorm_params(config: &LayerNormConfig) -> usize {
    if config.elementwise_affine {
        // Gamma (scale) and beta (shift)
        2 * config.normalized_shape
    } else {
        0
    }
}

/// Count parameters in encoder layer
pub fn count_encoder_layer_params(config: &EncoderLayerConfig) -> usize {
    let attention_params = count_attention_params(&config.attention);
    let ffn_params = count_ffn_params(&config.feed_forward);
    let ln1_params = count_layernorm_params(&config.layer_norm);
    let ln2_params = count_layernorm_params(&config.layer_norm);

    attention_params + ffn_params + ln1_params + ln2_params
}

/// Count parameters in decoder layer
pub fn count_decoder_layer_params(config: &DecoderLayerConfig) -> usize {
    let self_attn_params = count_attention_params(&config.self_attention);
    let cross_attn_params = count_attention_params(&config.cross_attention);
    let ffn_params = count_ffn_params(&config.feed_forward);
    let ln1_params = count_layernorm_params(&config.layer_norm);
    let ln2_params = count_layernorm_params(&config.layer_norm);
    let ln3_params = count_layernorm_params(&config.layer_norm);

    self_attn_params + cross_attn_params + ffn_params + ln1_params + ln2_params + ln3_params
}

/// Get statistics for encoder stack
pub fn encoder_stack_stats(config: &EncoderStackConfig) -> ModelStats {
    let layer_params = count_encoder_layer_params(&config.layer_config);
    let total_layers_params = layer_params * config.num_layers;

    // Position encoding parameters (if learned)
    let pos_encoding_params = match config.position_encoding.encoding_type {
        crate::position::PositionEncodingType::Learned => {
            config.position_encoding.max_seq_len * config.position_encoding.d_model
        }
        _ => 0, // Sinusoidal and relative don't have learned parameters
    };

    // Final layer norm (if enabled)
    let final_norm_params = if config.final_layer_norm {
        count_layernorm_params(&LayerNormConfig::new(config.layer_config.attention.d_model))
    } else {
        0
    };

    let total_params = total_layers_params + pos_encoding_params + final_norm_params;

    // Memory estimate: 4 bytes per parameter (float32)
    let memory_estimate = total_params * 4;

    ModelStats {
        total_params,
        trainable_params: total_params,
        num_layers: config.num_layers,
        d_model: config.layer_config.attention.d_model,
        memory_estimate,
    }
}

/// Get statistics for decoder stack
pub fn decoder_stack_stats(config: &DecoderStackConfig) -> ModelStats {
    let layer_params = count_decoder_layer_params(&config.layer_config);
    let total_layers_params = layer_params * config.num_layers;

    // Position encoding parameters
    let pos_encoding_params = match config.position_encoding.encoding_type {
        crate::position::PositionEncodingType::Learned => {
            config.position_encoding.max_seq_len * config.position_encoding.d_model
        }
        _ => 0,
    };

    // Final layer norm
    let final_norm_params = if config.final_layer_norm {
        count_layernorm_params(&LayerNormConfig::new(
            config.layer_config.self_attention.d_model,
        ))
    } else {
        0
    };

    let total_params = total_layers_params + pos_encoding_params + final_norm_params;
    let memory_estimate = total_params * 4;

    ModelStats {
        total_params,
        trainable_params: total_params,
        num_layers: config.num_layers,
        d_model: config.layer_config.self_attention.d_model,
        memory_estimate,
    }
}

/// Calculate FLOPs for attention operation
///
/// FLOPs for attention: 4 * batch * seq_len^2 * d_model
pub fn attention_flops(batch_size: usize, seq_len: usize, d_model: usize) -> usize {
    4 * batch_size * seq_len * seq_len * d_model
}

/// Calculate FLOPs for feed-forward network
///
/// FLOPs for FFN: 2 * batch * seq_len * (d_model * d_ff + d_ff * d_model)
pub fn ffn_flops(batch_size: usize, seq_len: usize, d_model: usize, d_ff: usize) -> usize {
    2 * batch_size * seq_len * (d_model * d_ff + d_ff * d_model)
}

/// Calculate total FLOPs for transformer layer
pub fn layer_flops(batch_size: usize, seq_len: usize, config: &EncoderLayerConfig) -> usize {
    let attn = attention_flops(batch_size, seq_len, config.attention.d_model);
    let ffn = ffn_flops(
        batch_size,
        seq_len,
        config.feed_forward.d_model,
        config.feed_forward.d_ff,
    );
    attn + ffn
}

/// Validate configuration compatibility
pub fn validate_encoder_decoder_compatibility(
    encoder: &EncoderStackConfig,
    decoder: &DecoderStackConfig,
) -> Result<(), String> {
    // Check d_model compatibility
    if encoder.layer_config.attention.d_model != decoder.layer_config.self_attention.d_model {
        return Err(format!(
            "d_model mismatch: encoder={}, decoder={}",
            encoder.layer_config.attention.d_model, decoder.layer_config.self_attention.d_model
        ));
    }

    // Check that decoder uses causal masking
    if !decoder.layer_config.self_attention.causal {
        return Err("Decoder self-attention must use causal masking".to_string());
    }

    Ok(())
}

/// Helper to create common transformer configurations
pub mod presets {
    use super::*;

    /// GPT-2 Small configuration (117M parameters)
    pub fn gpt2_small() -> EncoderStackConfig {
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

    /// BERT Base configuration (110M parameters)
    pub fn bert_base() -> EncoderStackConfig {
        EncoderStackConfig::new(
            12,   // layers
            768,  // d_model
            12,   // n_heads
            3072, // d_ff
            512,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.1)
    }

    /// Transformer Base (from "Attention Is All You Need")
    pub fn transformer_base() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(6, 512, 8, 2048, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(6, 512, 8, 2048, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }

    /// Small model for testing (faster)
    pub fn tiny() -> EncoderStackConfig {
        EncoderStackConfig::new(2, 128, 4, 512, 128)
            .unwrap()
            .with_dropout(0.0)
    }

    /// BERT Large configuration (340M parameters)
    pub fn bert_large() -> EncoderStackConfig {
        EncoderStackConfig::new(
            24,   // layers
            1024, // d_model
            16,   // n_heads
            4096, // d_ff
            512,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.1)
    }

    /// GPT-2 Medium configuration (345M parameters)
    pub fn gpt2_medium() -> EncoderStackConfig {
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

    /// GPT-2 Large configuration (774M parameters)
    pub fn gpt2_large() -> EncoderStackConfig {
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

    /// GPT-2 XL configuration (1.5B parameters)
    pub fn gpt2_xl() -> EncoderStackConfig {
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

    /// GPT-3 Small configuration (~125M parameters)
    pub fn gpt3_small() -> EncoderStackConfig {
        EncoderStackConfig::new(
            12,   // layers
            768,  // d_model
            12,   // n_heads
            3072, // d_ff
            2048, // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// GPT-3 Medium configuration (~350M parameters)
    pub fn gpt3_medium() -> EncoderStackConfig {
        EncoderStackConfig::new(
            24,   // layers
            1024, // d_model
            16,   // n_heads
            4096, // d_ff
            2048, // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// GPT-3 Large configuration (~760M parameters)
    pub fn gpt3_large() -> EncoderStackConfig {
        EncoderStackConfig::new(
            24,   // layers
            1536, // d_model
            16,   // n_heads
            6144, // d_ff
            2048, // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// GPT-3 XL configuration (~1.3B parameters)
    pub fn gpt3_xl() -> EncoderStackConfig {
        EncoderStackConfig::new(
            24,   // layers
            2048, // d_model
            16,   // n_heads (d_model must be divisible by n_heads)
            8192, // d_ff
            2048, // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// GPT-3 2.7B configuration
    pub fn gpt3_2_7b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            32,    // layers
            2560,  // d_model
            32,    // n_heads
            10240, // d_ff
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// GPT-3 6.7B configuration
    pub fn gpt3_6_7b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            32,    // layers
            4096,  // d_model
            32,    // n_heads
            16384, // d_ff
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// GPT-3 13B configuration
    pub fn gpt3_13b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            40,    // layers
            5120,  // d_model (must be divisible by n_heads)
            40,    // n_heads
            20480, // d_ff (4 * d_model)
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// GPT-3 175B configuration (davinci)
    pub fn gpt3_175b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            96,    // layers
            12288, // d_model
            96,    // n_heads
            49152, // d_ff
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// LLaMA 7B configuration
    /// Uses RoPE (implemented separately in position module)
    pub fn llama_7b() -> EncoderStackConfig {
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

    /// LLaMA 13B configuration
    pub fn llama_13b() -> EncoderStackConfig {
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

    /// LLaMA 30B configuration
    pub fn llama_30b() -> EncoderStackConfig {
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

    /// LLaMA 65B configuration
    pub fn llama_65b() -> EncoderStackConfig {
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

    /// BLOOM 560M configuration (uses ALiBi)
    pub fn bloom_560m() -> EncoderStackConfig {
        EncoderStackConfig::new(
            24,   // layers
            1024, // d_model
            16,   // n_heads
            4096, // d_ff
            2048, // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
        // Note: BLOOM uses ALiBi position encoding (implemented in position module)
    }

    /// BLOOM 3B configuration
    pub fn bloom_3b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            30,    // layers
            2560,  // d_model
            32,    // n_heads
            10240, // d_ff
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// BLOOM 7B configuration
    pub fn bloom_7b() -> EncoderStackConfig {
        EncoderStackConfig::new(
            30,    // layers
            4096,  // d_model
            32,    // n_heads
            16384, // d_ff
            2048,  // max_seq_len
        )
        .unwrap()
        .with_dropout(0.0)
    }

    /// T5 Small configuration (60M parameters)
    pub fn t5_small() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(6, 512, 8, 2048, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(6, 512, 8, 2048, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }

    /// T5 Base configuration (220M parameters)
    pub fn t5_base() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(12, 768, 12, 3072, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(12, 768, 12, 3072, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }

    /// T5 Large configuration (770M parameters)
    pub fn t5_large() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(24, 1024, 16, 4096, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(24, 1024, 16, 4096, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }

    /// T5 XL configuration (3B parameters)
    pub fn t5_xl() -> (EncoderStackConfig, DecoderStackConfig) {
        let encoder = EncoderStackConfig::new(24, 2048, 32, 8192, 512)
            .unwrap()
            .with_dropout(0.1);

        let decoder = DecoderStackConfig::new(24, 2048, 32, 8192, 512)
            .unwrap()
            .with_dropout(0.1);

        (encoder, decoder)
    }

    /// T5 XXL configuration (11B parameters)
    pub fn t5_xxl() -> (EncoderStackConfig, DecoderStackConfig) {
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
    fn test_count_attention_params() {
        let config = AttentionConfig::new(512, 8).unwrap();
        let params = count_attention_params(&config);

        // QKV: 3 * 512 * 512 = 786,432
        // Output: 512 * 512 = 262,144
        // Biases: 4 * 512 = 2,048
        // Total: 1,050,624
        assert_eq!(params, 1_050_624);
    }

    #[test]
    fn test_count_ffn_params() {
        let config = FeedForwardConfig::new(512, 2048);
        let params = count_ffn_params(&config);

        // Layer 1: 512 * 2048 + 2048 = 1,050,624
        // Layer 2: 2048 * 512 + 512 = 1,049,088
        // Total: 2,099,712
        assert_eq!(params, 2_099_712);
    }

    #[test]
    fn test_count_layernorm_params() {
        let config = LayerNormConfig::new(512);
        let params = count_layernorm_params(&config);
        assert_eq!(params, 1024); // gamma + beta

        let config_no_affine = LayerNormConfig::new(512).with_elementwise_affine(false);
        let params = count_layernorm_params(&config_no_affine);
        assert_eq!(params, 0);
    }

    #[test]
    fn test_encoder_layer_params() {
        let config = EncoderLayerConfig::new(512, 8, 2048).unwrap();
        let params = count_encoder_layer_params(&config);

        // Attention: 1,050,624
        // FFN: 2,099,712
        // LN1: 1,024
        // LN2: 1,024
        // Total: 3,152,384
        assert_eq!(params, 3_152_384);
    }

    #[test]
    fn test_encoder_stack_stats() {
        let config = EncoderStackConfig::new(6, 512, 8, 2048, 512).unwrap();
        let stats = encoder_stack_stats(&config);

        assert_eq!(stats.num_layers, 6);
        assert_eq!(stats.d_model, 512);
        assert!(stats.total_params > 0);
        assert_eq!(stats.trainable_params, stats.total_params);
    }

    #[test]
    fn test_decoder_stack_stats() {
        let config = DecoderStackConfig::new(6, 512, 8, 2048, 512).unwrap();
        let stats = decoder_stack_stats(&config);

        assert_eq!(stats.num_layers, 6);
        assert_eq!(stats.d_model, 512);
        // Decoder has more params than encoder (cross-attention)
        assert!(stats.total_params > 0);
    }

    #[test]
    fn test_flops_calculations() {
        let batch = 32;
        let seq_len = 128;
        let d_model = 512;
        let d_ff = 2048;

        let attn_flops = attention_flops(batch, seq_len, d_model);
        assert!(attn_flops > 0);

        let ffn_flops = ffn_flops(batch, seq_len, d_model, d_ff);
        assert!(ffn_flops > 0);
    }

    #[test]
    fn test_validate_compatibility() {
        let encoder = EncoderStackConfig::new(6, 512, 8, 2048, 512).unwrap();
        let decoder = DecoderStackConfig::new(6, 512, 8, 2048, 512).unwrap();

        assert!(validate_encoder_decoder_compatibility(&encoder, &decoder).is_ok());

        // Mismatched d_model
        let encoder_mismatch = EncoderStackConfig::new(6, 768, 8, 2048, 512).unwrap();
        assert!(validate_encoder_decoder_compatibility(&encoder_mismatch, &decoder).is_err());
    }

    #[test]
    fn test_presets() {
        let gpt2 = presets::gpt2_small();
        assert_eq!(gpt2.num_layers, 12);
        assert_eq!(gpt2.layer_config.attention.d_model, 768);

        let bert = presets::bert_base();
        assert_eq!(bert.num_layers, 12);
        assert_eq!(bert.layer_config.attention.d_model, 768);

        let (encoder, decoder) = presets::transformer_base();
        assert_eq!(encoder.num_layers, 6);
        assert_eq!(decoder.num_layers, 6);
        assert!(validate_encoder_decoder_compatibility(&encoder, &decoder).is_ok());
    }

    #[test]
    fn test_model_stats_summary() {
        let config = EncoderStackConfig::new(6, 512, 8, 2048, 512).unwrap();
        let stats = encoder_stack_stats(&config);
        let summary = stats.summary();

        assert!(summary.contains("ModelStats"));
        assert!(summary.contains("Total params"));
        assert!(summary.contains("Layers: 6"));
    }

    #[test]
    fn test_format_number() {
        let stats = ModelStats {
            total_params: 117_000_000,
            trainable_params: 117_000_000,
            num_layers: 12,
            d_model: 768,
            memory_estimate: 468_000_000,
        };

        let summary = stats.summary();
        assert!(summary.contains("117.00M"));
    }

    #[test]
    fn test_bert_large_preset() {
        let config = presets::bert_large();
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.layer_config.attention.d_model, 1024);
        assert_eq!(config.layer_config.attention.n_heads, 16);
    }

    #[test]
    fn test_gpt2_variants() {
        let small = presets::gpt2_small();
        let medium = presets::gpt2_medium();
        let large = presets::gpt2_large();
        let xl = presets::gpt2_xl();

        // Verify parameter counts increase
        let small_stats = encoder_stack_stats(&small);
        let medium_stats = encoder_stack_stats(&medium);
        let large_stats = encoder_stack_stats(&large);
        let xl_stats = encoder_stack_stats(&xl);

        assert!(medium_stats.total_params > small_stats.total_params);
        assert!(large_stats.total_params > medium_stats.total_params);
        assert!(xl_stats.total_params > large_stats.total_params);
    }

    #[test]
    fn test_gpt3_variants() {
        let small = presets::gpt3_small();
        let medium = presets::gpt3_medium();
        let large = presets::gpt3_large();
        let xl = presets::gpt3_xl();

        assert_eq!(small.num_layers, 12);
        assert_eq!(medium.num_layers, 24);
        assert_eq!(large.num_layers, 24);
        assert_eq!(xl.num_layers, 24);

        // d_model increases
        assert!(medium.layer_config.attention.d_model > small.layer_config.attention.d_model);
        assert!(large.layer_config.attention.d_model > medium.layer_config.attention.d_model);
        assert!(xl.layer_config.attention.d_model > large.layer_config.attention.d_model);
    }

    #[test]
    fn test_gpt3_large_models() {
        let m2_7b = presets::gpt3_2_7b();
        let m6_7b = presets::gpt3_6_7b();
        let m13b = presets::gpt3_13b();
        let m175b = presets::gpt3_175b();

        assert_eq!(m2_7b.num_layers, 32);
        assert_eq!(m6_7b.num_layers, 32);
        assert_eq!(m13b.num_layers, 40);
        assert_eq!(m175b.num_layers, 96);

        // Verify d_model increases
        assert!(m6_7b.layer_config.attention.d_model > m2_7b.layer_config.attention.d_model);
        assert!(m13b.layer_config.attention.d_model > m6_7b.layer_config.attention.d_model);
        assert!(m175b.layer_config.attention.d_model > m13b.layer_config.attention.d_model);
    }

    #[test]
    fn test_llama_variants() {
        let m7b = presets::llama_7b();
        let m13b = presets::llama_13b();
        let m30b = presets::llama_30b();
        let m65b = presets::llama_65b();

        // Verify layer counts increase
        assert!(m13b.num_layers > m7b.num_layers);
        assert!(m30b.num_layers > m13b.num_layers);
        assert!(m65b.num_layers > m30b.num_layers);

        // Verify d_model increases
        assert!(m13b.layer_config.attention.d_model > m7b.layer_config.attention.d_model);
        assert!(m30b.layer_config.attention.d_model > m13b.layer_config.attention.d_model);
        assert!(m65b.layer_config.attention.d_model > m30b.layer_config.attention.d_model);

        // LLaMA uses learned PE (would be RoPE in practice)
        assert!(matches!(
            m7b.position_encoding.encoding_type,
            crate::position::PositionEncodingType::Learned
        ));
    }

    #[test]
    fn test_bloom_variants() {
        let m560m = presets::bloom_560m();
        let m3b = presets::bloom_3b();
        let m7b = presets::bloom_7b();

        assert_eq!(m560m.num_layers, 24);
        assert_eq!(m3b.num_layers, 30);
        assert_eq!(m7b.num_layers, 30);

        // Verify d_model increases
        assert!(m3b.layer_config.attention.d_model > m560m.layer_config.attention.d_model);
        assert!(m7b.layer_config.attention.d_model > m3b.layer_config.attention.d_model);
    }

    #[test]
    fn test_t5_variants() {
        let small = presets::t5_small();
        let base = presets::t5_base();
        let large = presets::t5_large();
        let xl = presets::t5_xl();
        let xxl = presets::t5_xxl();

        // Verify encoder-decoder compatibility
        assert!(validate_encoder_decoder_compatibility(&small.0, &small.1).is_ok());
        assert!(validate_encoder_decoder_compatibility(&base.0, &base.1).is_ok());
        assert!(validate_encoder_decoder_compatibility(&large.0, &large.1).is_ok());
        assert!(validate_encoder_decoder_compatibility(&xl.0, &xl.1).is_ok());
        assert!(validate_encoder_decoder_compatibility(&xxl.0, &xxl.1).is_ok());

        // Verify parameter counts increase
        let small_stats = encoder_stack_stats(&small.0);
        let base_stats = encoder_stack_stats(&base.0);
        let large_stats = encoder_stack_stats(&large.0);

        assert!(base_stats.total_params > small_stats.total_params);
        assert!(large_stats.total_params > base_stats.total_params);
    }

    #[test]
    fn test_all_presets_validate() {
        // Ensure all preset configurations are valid
        assert!(presets::tiny().validate().is_ok());
        assert!(presets::gpt2_small().validate().is_ok());
        assert!(presets::bert_base().validate().is_ok());
        assert!(presets::bert_large().validate().is_ok());
        assert!(presets::gpt2_medium().validate().is_ok());
        assert!(presets::gpt2_large().validate().is_ok());
        assert!(presets::gpt2_xl().validate().is_ok());
        assert!(presets::gpt3_small().validate().is_ok());
        assert!(presets::gpt3_medium().validate().is_ok());
        assert!(presets::gpt3_large().validate().is_ok());
        assert!(presets::gpt3_xl().validate().is_ok());
        assert!(presets::llama_7b().validate().is_ok());
        assert!(presets::llama_13b().validate().is_ok());
        assert!(presets::bloom_560m().validate().is_ok());
        assert!(presets::bloom_3b().validate().is_ok());

        let (enc, dec) = presets::transformer_base();
        assert!(enc.validate().is_ok());
        assert!(dec.validate().is_ok());
    }
}
