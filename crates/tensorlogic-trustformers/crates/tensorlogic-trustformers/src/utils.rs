//! Utility functions for parameter counting and FLOP estimation.

use crate::config::ModelConfig;

/// Count the number of parameters in a model configuration
pub fn count_parameters(config: &ModelConfig) -> usize {
    let mut total = 0;
    
    // Embedding parameters
    total += config.vocab_size * config.hidden_size; // Token embeddings
    total += config.max_seq_len * config.hidden_size; // Position embeddings (if learned)
    
    // Encoder parameters (if present)
    if let Some(ref encoder_config) = config.encoder {
        let layer_params = count_encoder_layer_parameters(
            config.hidden_size,
            config.num_heads,
            config.intermediate_size,
        );
        total += layer_params * config.num_layers;
    }
    
    // Decoder parameters (if present)
    if let Some(ref decoder_config) = config.decoder {
        let layer_params = count_decoder_layer_parameters(
            config.hidden_size,
            config.num_heads,
            config.intermediate_size,
        );
        total += layer_params * config.num_layers;
    }
    
    // Output layer
    total += config.hidden_size * config.vocab_size;
    
    total
}

/// Count parameters in a single encoder layer
fn count_encoder_layer_parameters(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> usize {
    let mut total = 0;
    
    // Multi-head attention: Q, K, V, O projections
    total += hidden_size * hidden_size * 4; // 4 weight matrices
    total += hidden_size * 4; // Biases
    
    // Feed-forward: two linear layers
    total += hidden_size * intermediate_size; // First layer
    total += intermediate_size * hidden_size; // Second layer
    total += intermediate_size + hidden_size; // Biases
    
    // Layer normalization: gamma and beta for two layers
    total += hidden_size * 2 * 2;
    
    total
}

/// Count parameters in a single decoder layer
fn count_decoder_layer_parameters(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> usize {
    let mut total = 0;
    
    // Masked self-attention
    total += hidden_size * hidden_size * 4;
    total += hidden_size * 4;
    
    // Cross-attention
    total += hidden_size * hidden_size * 4;
    total += hidden_size * 4;
    
    // Feed-forward
    total += hidden_size * intermediate_size;
    total += intermediate_size * hidden_size;
    total += intermediate_size + hidden_size;
    
    // Layer normalization: gamma and beta for three layers
    total += hidden_size * 2 * 3;
    
    total
}

/// Estimate FLOPs for a single forward pass
pub fn estimate_flops(config: &ModelConfig, seq_len: usize) -> usize {
    let mut total_flops = 0;
    
    // Embedding lookup (negligible)
    
    // Encoder FLOPs
    if config.encoder.is_some() {
        let layer_flops = estimate_encoder_layer_flops(
            config.hidden_size,
            config.num_heads,
            config.intermediate_size,
            seq_len,
        );
        total_flops += layer_flops * config.num_layers;
    }
    
    // Decoder FLOPs
    if config.decoder.is_some() {
        let layer_flops = estimate_decoder_layer_flops(
            config.hidden_size,
            config.num_heads,
            config.intermediate_size,
            seq_len,
        );
        total_flops += layer_flops * config.num_layers;
    }
    
    // Output projection
    total_flops += 2 * seq_len * config.hidden_size * config.vocab_size;
    
    total_flops
}

/// Estimate FLOPs for a single encoder layer
fn estimate_encoder_layer_flops(hidden_size: usize, _num_heads: usize, intermediate_size: usize, seq_len: usize) -> usize {
    let mut total = 0;
    
    // QKV projections: 3 * (2 * seq_len * hidden_size * hidden_size)
    total += 6 * seq_len * hidden_size * hidden_size;
    
    // Attention scores: 2 * seq_len * seq_len * hidden_size
    total += 2 * seq_len * seq_len * hidden_size;
    
    // Attention output: 2 * seq_len * seq_len * hidden_size
    total += 2 * seq_len * seq_len * hidden_size;
    
    // Output projection: 2 * seq_len * hidden_size * hidden_size
    total += 2 * seq_len * hidden_size * hidden_size;
    
    // Feed-forward: 2 * (2 * seq_len * hidden_size * intermediate_size)
    total += 4 * seq_len * hidden_size * intermediate_size;
    
    total
}

/// Estimate FLOPs for a single decoder layer
fn estimate_decoder_layer_flops(hidden_size: usize, _num_heads: usize, intermediate_size: usize, seq_len: usize) -> usize {
    let mut total = 0;
    
    // Masked self-attention (same as encoder attention)
    total += 6 * seq_len * hidden_size * hidden_size;
    total += 2 * seq_len * seq_len * hidden_size;
    total += 2 * seq_len * seq_len * hidden_size;
    total += 2 * seq_len * hidden_size * hidden_size;
    
    // Cross-attention (same as self-attention)
    total += 6 * seq_len * hidden_size * hidden_size;
    total += 2 * seq_len * seq_len * hidden_size;
    total += 2 * seq_len * seq_len * hidden_size;
    total += 2 * seq_len * hidden_size * hidden_size;
    
    // Feed-forward
    total += 4 * seq_len * hidden_size * intermediate_size;
    
    total
}

/// Estimate memory usage in bytes
pub fn estimate_memory(config: &ModelConfig, seq_len: usize, batch_size: usize) -> usize {
    let mut total_bytes = 0;
    
    // Parameters (4 bytes per float32)
    let num_params = count_parameters(config);
    total_bytes += num_params * 4;
    
    // Activations (rough estimate)
    let activation_memory = batch_size * seq_len * config.hidden_size * 4; // per layer
    total_bytes += activation_memory * config.num_layers * 2; // multiply by layers and some overhead
    
    // Attention scores
    let attention_memory = batch_size * config.num_heads * seq_len * seq_len * 4;
    total_bytes += attention_memory * config.num_layers;
    
    total_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_parameters() {
        let config = ModelConfig::encoder_only(30000, 512, 6, 8, 2048);
        let params = count_parameters(&config);
        assert!(params > 0);
    }

    #[test]
    fn test_estimate_flops() {
        let config = ModelConfig::encoder_only(30000, 512, 6, 8, 2048);
        let flops = estimate_flops(&config, 128);
        assert!(flops > 0);
    }

    #[test]
    fn test_estimate_memory() {
        let config = ModelConfig::encoder_only(30000, 512, 6, 8, 2048);
        let memory = estimate_memory(&config, 128, 32);
        assert!(memory > 0);
    }

    #[test]
    fn test_different_model_sizes() {
        for (hidden, layers) in vec![(256, 4), (512, 6), (768, 12)] {
            let config = ModelConfig::encoder_only(30000, hidden, layers, 8, hidden * 4);
            let params = count_parameters(&config);
            assert!(params > 0);
        }
    }

    #[test]
    fn test_decoder_only_parameters() {
        let config = ModelConfig::decoder_only(50000, 768, 12, 12, 3072);
        let params = count_parameters(&config);
        assert!(params > 0);
    }
}
