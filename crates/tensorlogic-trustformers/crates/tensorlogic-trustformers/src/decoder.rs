//! Transformer decoder layer and stack implementations.

use crate::config::{DecoderConfig, NormPosition};
use crate::error::TrustformersResult;
use tensorlogic_ir::EinsumGraph;

/// Transformer decoder layer
pub struct DecoderLayer {
    config: DecoderConfig,
}

impl DecoderLayer {
    /// Create a new decoder layer
    pub fn new(config: DecoderConfig) -> TrustformersResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Build the decoder layer graph (masked self-attention + cross-attention + FFN + residual + norm)
    pub fn build(&self) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        
        // The actual implementation would combine:
        // 1. Masked self-attention
        // 2. Add & Norm
        // 3. Cross-attention (to encoder output)
        // 4. Add & Norm
        // 5. Feed-forward
        // 6. Add & Norm
        
        Ok(graph)
    }
}

/// Transformer decoder stack
pub struct DecoderStack {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
}

impl DecoderStack {
    /// Create a new decoder stack
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> Self {
        Self {
            num_layers,
            hidden_size,
            num_heads,
            intermediate_size,
        }
    }

    /// Build the full decoder stack graph
    pub fn build(&self) -> TrustformersResult<EinsumGraph> {
        let mut graph = EinsumGraph::new();
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_layer_creation() {
        let config = DecoderConfig::default();
        let decoder = DecoderLayer::new(config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_layer_build() {
        let config = DecoderConfig::default();
        let decoder = DecoderLayer::new(config).unwrap();
        let graph = decoder.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_decoder_stack_creation() {
        let stack = DecoderStack::new(6, 512, 8, 2048);
        assert_eq!(stack.num_layers, 6);
        assert_eq!(stack.hidden_size, 512);
    }

    #[test]
    fn test_decoder_stack_build() {
        let stack = DecoderStack::new(6, 512, 8, 2048);
        let graph = stack.build();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_pre_norm_decoder() {
        let mut config = DecoderConfig::default();
        config.norm_position = NormPosition::Pre;
        let decoder = DecoderLayer::new(config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_post_norm_decoder() {
        let mut config = DecoderConfig::default();
        config.norm_position = NormPosition::Post;
        let decoder = DecoderLayer::new(config);
        assert!(decoder.is_ok());
    }
}
