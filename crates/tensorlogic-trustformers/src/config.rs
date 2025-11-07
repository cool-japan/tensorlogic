//! Configuration structures for transformer components.

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrustformerError};

/// Configuration for self-attention mechanism
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Model dimension (d_model)
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per head (d_k = d_model / n_heads)
    pub d_k: usize,
    /// Whether to use causal (autoregressive) masking
    pub causal: bool,
    /// Dropout probability (0.0 = no dropout)
    pub dropout: f64,
}

impl AttentionConfig {
    /// Create a new attention configuration
    pub fn new(d_model: usize, n_heads: usize) -> Result<Self> {
        if !d_model.is_multiple_of(n_heads) {
            return Err(TrustformerError::InvalidHeadCount { d_model, n_heads });
        }

        Ok(Self {
            d_model,
            n_heads,
            d_k: d_model / n_heads,
            causal: false,
            dropout: 0.0,
        })
    }

    /// Set causal masking
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "d_model must be positive".to_string(),
            });
        }

        if self.n_heads == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "n_heads must be positive".to_string(),
            });
        }

        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(TrustformerError::InvalidHeadCount {
                d_model: self.d_model,
                n_heads: self.n_heads,
            });
        }

        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: format!("dropout must be in [0,1], got {}", self.dropout),
            });
        }

        Ok(())
    }
}

/// Configuration for feed-forward network
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedForwardConfig {
    /// Model dimension (d_model)
    pub d_model: usize,
    /// Hidden dimension (typically 4 * d_model)
    pub d_ff: usize,
    /// Activation function name
    pub activation: String,
    /// Dropout probability
    pub dropout: f64,
}

impl FeedForwardConfig {
    /// Create a new feed-forward configuration
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            d_model,
            d_ff,
            activation: "gelu".to_string(),
            dropout: 0.0,
        }
    }

    /// Set activation function
    pub fn with_activation(mut self, activation: impl Into<String>) -> Self {
        self.activation = activation.into();
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "d_model must be positive".to_string(),
            });
        }

        if self.d_ff == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "d_ff must be positive".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&self.dropout) {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: format!("dropout must be in [0,1], got {}", self.dropout),
            });
        }

        Ok(())
    }
}

/// Configuration for a complete transformer layer
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformerLayerConfig {
    /// Attention configuration
    pub attention: AttentionConfig,
    /// Feed-forward configuration
    pub feed_forward: FeedForwardConfig,
    /// Whether to use pre-layer normalization (vs post)
    pub pre_norm: bool,
}

impl TransformerLayerConfig {
    /// Create a new transformer layer configuration
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            attention: AttentionConfig::new(d_model, n_heads)?,
            feed_forward: FeedForwardConfig::new(d_model, d_ff),
            pre_norm: true,
        })
    }

    /// Set pre-normalization vs post-normalization
    pub fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.pre_norm = pre_norm;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.attention.validate()?;
        self.feed_forward.validate()?;

        if self.attention.d_model != self.feed_forward.d_model {
            return Err(TrustformerError::InvalidDimension {
                expected: self.attention.d_model,
                got: self.feed_forward.d_model,
                context: "d_model mismatch between attention and FFN".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_valid() {
        let config = AttentionConfig::new(512, 8).unwrap();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.n_heads, 8);
        assert_eq!(config.d_k, 64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_attention_config_invalid_heads() {
        let result = AttentionConfig::new(512, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_config_with_causal() {
        let config = AttentionConfig::new(512, 8).unwrap().with_causal(true);
        assert!(config.causal);
    }

    #[test]
    fn test_attention_config_with_dropout() {
        let config = AttentionConfig::new(512, 8).unwrap().with_dropout(0.1);
        assert!((config.dropout - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_ffn_config() {
        let config = FeedForwardConfig::new(512, 2048);
        assert_eq!(config.d_model, 512);
        assert_eq!(config.d_ff, 2048);
        assert_eq!(config.activation, "gelu");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ffn_config_with_activation() {
        let config = FeedForwardConfig::new(512, 2048).with_activation("relu");
        assert_eq!(config.activation, "relu");
    }

    #[test]
    fn test_transformer_layer_config() {
        let config = TransformerLayerConfig::new(512, 8, 2048).unwrap();
        assert_eq!(config.attention.d_model, 512);
        assert_eq!(config.feed_forward.d_model, 512);
        assert!(config.pre_norm);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_transformer_layer_config_with_pre_norm() {
        let config = TransformerLayerConfig::new(512, 8, 2048)
            .unwrap()
            .with_pre_norm(false);
        assert!(!config.pre_norm);
    }
}
