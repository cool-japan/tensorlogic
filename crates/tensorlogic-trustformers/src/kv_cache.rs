//! Key-Value cache for efficient autoregressive inference.
//!
//! During autoregressive generation (e.g., text generation), transformers repeatedly
//! compute attention over the same prefix tokens. KV-caching stores the key and value
//! projections from previous steps, avoiding redundant computation.
//!
//! ## Performance Impact
//!
//! Without KV-cache:
//! ```text
//! Step 1: Compute attention for token 1
//! Step 2: Compute attention for tokens 1,2    (redundant!)
//! Step 3: Compute attention for tokens 1,2,3  (redundant!)
//! ```
//!
//! With KV-cache:
//! ```text
//! Step 1: Compute K,V for token 1, cache them
//! Step 2: Compute K,V for token 2, append to cache
//! Step 3: Compute K,V for token 3, append to cache
//! ```
//!
//! **Speedup**: ~10-100x for long sequences!
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tensorlogic_trustformers::KVCache;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create cache for 12-layer model with 12 heads
//! let mut cache = KVCache::new(12, 12, 64);
//!
//! // During generation, update cache for each layer
//! # let (new_keys, new_values) = (vec![], vec![]);
//! cache.update_layer(0, new_keys, new_values)?;
//!
//! // Retrieve cached keys/values for attention
//! let (cached_keys, cached_values) = cache.get_layer(0)?;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, TrustformerError};

/// Configuration for KV-cache
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KVCacheConfig {
    /// Number of layers in the model
    pub num_layers: usize,
    /// Number of attention heads per layer
    pub num_heads: usize,
    /// Dimension per attention head (d_k)
    pub head_dim: usize,
    /// Maximum sequence length to cache
    pub max_seq_len: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Whether to enable cache
    pub enabled: bool,
}

impl KVCacheConfig {
    /// Create a new KV-cache configuration
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_layers,
            num_heads,
            head_dim,
            max_seq_len: 2048,
            max_batch_size: 32,
            enabled: true,
        }
    }

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Set maximum batch size
    pub fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }

    /// Enable or disable cache
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.num_layers == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "num_layers must be > 0".to_string(),
            });
        }

        if self.num_heads == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "num_heads must be > 0".to_string(),
            });
        }

        if self.head_dim == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "head_dim must be > 0".to_string(),
            });
        }

        if self.max_seq_len == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "max_seq_len must be > 0".to_string(),
            });
        }

        if self.max_batch_size == 0 {
            return Err(TrustformerError::InvalidDimension {
                expected: 1,
                got: 0,
                context: "max_batch_size must be > 0".to_string(),
            });
        }

        Ok(())
    }

    /// Calculate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Each cache entry: [batch, num_heads, seq_len, head_dim]
        // We store both keys and values
        // Assume f32 (4 bytes per element)
        let bytes_per_element = 4;
        let elements_per_layer =
            self.max_batch_size * self.num_heads * self.max_seq_len * self.head_dim * 2; // keys + values

        elements_per_layer * self.num_layers * bytes_per_element
    }

    /// Human-readable memory usage
    pub fn memory_usage_mb(&self) -> f64 {
        self.memory_usage() as f64 / (1024.0 * 1024.0)
    }
}

/// Cache entry for a single layer
#[derive(Clone, Debug)]
pub struct CacheEntry {
    /// Cached keys: [batch, num_heads, seq_len, head_dim]
    pub keys: Vec<f32>,
    /// Cached values: [batch, num_heads, seq_len, head_dim]
    pub values: Vec<f32>,
    /// Current sequence length in cache
    pub seq_len: usize,
    /// Batch size
    pub batch_size: usize,
}

impl CacheEntry {
    /// Create a new empty cache entry
    pub fn new(batch_size: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let capacity = batch_size * num_heads * max_seq_len * head_dim;
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            seq_len: 0,
            batch_size,
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Get current sequence length
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }
}

/// Key-Value cache for efficient transformer inference
#[derive(Clone, Debug)]
pub struct KVCache {
    /// Configuration
    config: KVCacheConfig,
    /// Cache entries per layer
    cache: HashMap<usize, CacheEntry>,
    /// Current generation step
    step: usize,
}

impl KVCache {
    /// Create a new KV-cache
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> Self {
        let config = KVCacheConfig::new(num_layers, num_heads, head_dim);
        Self {
            config,
            cache: HashMap::new(),
            step: 0,
        }
    }

    /// Create KV-cache from configuration
    pub fn from_config(config: KVCacheConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            cache: HashMap::new(),
            step: 0,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Check if cache is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get current generation step
    pub fn step(&self) -> usize {
        self.step
    }

    /// Initialize cache for a layer
    pub fn init_layer(&mut self, layer_idx: usize, batch_size: usize) -> Result<()> {
        if layer_idx >= self.config.num_layers {
            return Err(TrustformerError::InvalidDimension {
                expected: self.config.num_layers,
                got: layer_idx,
                context: format!(
                    "layer_idx {} >= num_layers {}",
                    layer_idx, self.config.num_layers
                ),
            });
        }

        if batch_size > self.config.max_batch_size {
            return Err(TrustformerError::InvalidDimension {
                expected: self.config.max_batch_size,
                got: batch_size,
                context: format!(
                    "batch_size {} > max_batch_size {}",
                    batch_size, self.config.max_batch_size
                ),
            });
        }

        let entry = CacheEntry::new(
            batch_size,
            self.config.num_heads,
            self.config.head_dim,
            self.config.max_seq_len,
        );
        self.cache.insert(layer_idx, entry);
        Ok(())
    }

    /// Update cache for a layer with new keys and values
    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        new_keys: Vec<f32>,
        new_values: Vec<f32>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Initialize layer if not present
        if !self.cache.contains_key(&layer_idx) {
            // Infer batch size from keys shape
            // Assuming keys shape: [batch, num_heads, new_seq_len, head_dim]
            let expected_size_per_token = self.config.num_heads * self.config.head_dim;

            if !new_keys.len().is_multiple_of(expected_size_per_token) {
                return Err(TrustformerError::InvalidDimension {
                    expected: expected_size_per_token,
                    got: new_keys.len(),
                    context: "keys size must be divisible by num_heads * head_dim".to_string(),
                });
            }

            let batch_size = new_keys.len() / expected_size_per_token;
            self.init_layer(layer_idx, batch_size)?;
        }

        let entry = self.cache.get_mut(&layer_idx).unwrap();

        // Validate sizes
        if new_keys.len() != new_values.len() {
            return Err(TrustformerError::InvalidDimension {
                expected: new_keys.len(),
                got: new_values.len(),
                context: "keys and values must have same size".to_string(),
            });
        }

        // Append new keys and values
        entry.keys.extend_from_slice(&new_keys);
        entry.values.extend_from_slice(&new_values);

        // Update sequence length
        let new_tokens =
            new_keys.len() / (entry.batch_size * self.config.num_heads * self.config.head_dim);
        entry.seq_len += new_tokens;

        // Check if we exceeded max sequence length
        if entry.seq_len > self.config.max_seq_len {
            return Err(TrustformerError::InvalidDimension {
                expected: self.config.max_seq_len,
                got: entry.seq_len,
                context: format!(
                    "cache exceeded max_seq_len {} (current: {})",
                    self.config.max_seq_len, entry.seq_len
                ),
            });
        }

        Ok(())
    }

    /// Get cached keys and values for a layer
    pub fn get_layer(&self, layer_idx: usize) -> Result<(&[f32], &[f32])> {
        let entry =
            self.cache
                .get(&layer_idx)
                .ok_or_else(|| TrustformerError::InvalidDimension {
                    expected: 1,
                    got: 0,
                    context: format!("layer {} not found in cache", layer_idx),
                })?;

        Ok((&entry.keys, &entry.values))
    }

    /// Get sequence length for a layer
    pub fn get_seq_len(&self, layer_idx: usize) -> Result<usize> {
        let entry =
            self.cache
                .get(&layer_idx)
                .ok_or_else(|| TrustformerError::InvalidDimension {
                    expected: 1,
                    got: 0,
                    context: format!("layer {} not found in cache", layer_idx),
                })?;

        Ok(entry.seq_len)
    }

    /// Clear cache for a specific layer
    pub fn clear_layer(&mut self, layer_idx: usize) {
        if let Some(entry) = self.cache.get_mut(&layer_idx) {
            entry.clear();
        }
    }

    /// Clear all cache entries
    pub fn clear_all(&mut self) {
        for entry in self.cache.values_mut() {
            entry.clear();
        }
        self.step = 0;
    }

    /// Increment generation step
    pub fn next_step(&mut self) {
        self.step += 1;
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.cache.clear();
        self.step = 0;
    }

    /// Get number of cached layers
    pub fn num_cached_layers(&self) -> usize {
        self.cache.len()
    }

    /// Calculate current memory usage
    pub fn current_memory_usage(&self) -> usize {
        let bytes_per_element = 4; // f32
        self.cache
            .values()
            .map(|entry| (entry.keys.len() + entry.values.len()) * bytes_per_element)
            .sum()
    }

    /// Calculate memory usage in MB
    pub fn current_memory_usage_mb(&self) -> f64 {
        self.current_memory_usage() as f64 / (1024.0 * 1024.0)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            num_layers: self.cache.len(),
            total_seq_len: self
                .cache
                .values()
                .map(|entry| entry.seq_len)
                .max()
                .unwrap_or(0),
            memory_usage_mb: self.current_memory_usage_mb(),
            max_memory_mb: self.config.memory_usage_mb(),
            step: self.step,
            enabled: self.config.enabled,
        }
    }
}

/// Statistics about cache usage
#[derive(Clone, Debug)]
pub struct CacheStats {
    /// Number of cached layers
    pub num_layers: usize,
    /// Maximum sequence length across all layers
    pub total_seq_len: usize,
    /// Current memory usage in MB
    pub memory_usage_mb: f64,
    /// Maximum allowed memory in MB
    pub max_memory_mb: f64,
    /// Current generation step
    pub step: usize,
    /// Whether cache is enabled
    pub enabled: bool,
}

impl CacheStats {
    /// Format statistics as human-readable string
    pub fn summary(&self) -> String {
        format!(
            "CacheStats:\n  Layers: {}\n  Seq len: {}\n  Memory: {:.1}/{:.1} MB ({:.1}%)\n  Step: {}\n  Enabled: {}",
            self.num_layers,
            self.total_seq_len,
            self.memory_usage_mb,
            self.max_memory_mb,
            if self.max_memory_mb > 0.0 {
                (self.memory_usage_mb / self.max_memory_mb) * 100.0
            } else {
                0.0
            },
            self.step,
            self.enabled
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_config_creation() {
        let config = KVCacheConfig::new(12, 8, 64);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.enabled);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = KVCacheConfig::new(12, 8, 64)
            .with_max_seq_len(4096)
            .with_max_batch_size(16)
            .with_enabled(false);

        assert_eq!(config.max_seq_len, 4096);
        assert_eq!(config.max_batch_size, 16);
        assert!(!config.enabled);
    }

    #[test]
    fn test_config_validation() {
        let config = KVCacheConfig::new(0, 8, 64);
        assert!(config.validate().is_err());

        let config = KVCacheConfig::new(12, 0, 64);
        assert!(config.validate().is_err());

        let config = KVCacheConfig::new(12, 8, 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_memory_usage_calculation() {
        let config = KVCacheConfig::new(12, 8, 64);
        let memory = config.memory_usage();
        assert!(memory > 0);

        let memory_mb = config.memory_usage_mb();
        assert!(memory_mb > 0.0);
    }

    #[test]
    fn test_kv_cache_creation() {
        let cache = KVCache::new(12, 8, 64);
        assert_eq!(cache.config().num_layers, 12);
        assert_eq!(cache.step(), 0);
        assert!(cache.is_enabled());
    }

    #[test]
    fn test_cache_from_config() {
        let config = KVCacheConfig::new(12, 8, 64);
        let cache = KVCache::from_config(config).unwrap();
        assert_eq!(cache.config().num_layers, 12);
    }

    #[test]
    fn test_init_layer() {
        let mut cache = KVCache::new(12, 8, 64);
        assert!(cache.init_layer(0, 1).is_ok());
        assert_eq!(cache.num_cached_layers(), 1);
    }

    #[test]
    fn test_init_layer_invalid_index() {
        let mut cache = KVCache::new(12, 8, 64);
        assert!(cache.init_layer(20, 1).is_err());
    }

    #[test]
    fn test_update_and_get_layer() {
        let mut cache = KVCache::new(12, 8, 64);

        // batch=1, heads=8, tokens=1, dim=64
        let keys = vec![0.1f32; 8 * 64];
        let values = vec![0.2f32; 8 * 64];

        cache.update_layer(0, keys.clone(), values.clone()).unwrap();

        let (cached_keys, cached_values) = cache.get_layer(0).unwrap();
        assert_eq!(cached_keys.len(), keys.len());
        assert_eq!(cached_values.len(), values.len());
    }

    #[test]
    fn test_update_multiple_steps() {
        let mut cache = KVCache::new(12, 8, 64);

        // Step 1: Add first token
        let keys1 = vec![0.1f32; 8 * 64];
        let values1 = vec![0.2f32; 8 * 64];
        cache.update_layer(0, keys1, values1).unwrap();
        assert_eq!(cache.get_seq_len(0).unwrap(), 1);

        // Step 2: Add second token
        let keys2 = vec![0.3f32; 8 * 64];
        let values2 = vec![0.4f32; 8 * 64];
        cache.update_layer(0, keys2, values2).unwrap();
        assert_eq!(cache.get_seq_len(0).unwrap(), 2);

        // Verify total cached size
        let (cached_keys, _) = cache.get_layer(0).unwrap();
        assert_eq!(cached_keys.len(), 2 * 8 * 64);
    }

    #[test]
    fn test_clear_layer() {
        let mut cache = KVCache::new(12, 8, 64);
        let keys = vec![0.1f32; 8 * 64];
        let values = vec![0.2f32; 8 * 64];

        cache.update_layer(0, keys, values).unwrap();
        assert_eq!(cache.get_seq_len(0).unwrap(), 1);

        cache.clear_layer(0);
        assert_eq!(cache.get_seq_len(0).unwrap(), 0);
    }

    #[test]
    fn test_clear_all() {
        let mut cache = KVCache::new(12, 8, 64);
        let keys = vec![0.1f32; 8 * 64];
        let values = vec![0.2f32; 8 * 64];

        cache.update_layer(0, keys.clone(), values.clone()).unwrap();
        cache.update_layer(1, keys, values).unwrap();
        assert_eq!(cache.num_cached_layers(), 2);

        cache.clear_all();
        assert_eq!(cache.get_seq_len(0).unwrap(), 0);
        assert_eq!(cache.get_seq_len(1).unwrap(), 0);
        assert_eq!(cache.step(), 0);
    }

    #[test]
    fn test_reset() {
        let mut cache = KVCache::new(12, 8, 64);
        let keys = vec![0.1f32; 8 * 64];
        let values = vec![0.2f32; 8 * 64];

        cache.update_layer(0, keys, values).unwrap();
        cache.next_step();
        assert_eq!(cache.step(), 1);

        cache.reset();
        assert_eq!(cache.num_cached_layers(), 0);
        assert_eq!(cache.step(), 0);
    }

    #[test]
    fn test_next_step() {
        let mut cache = KVCache::new(12, 8, 64);
        assert_eq!(cache.step(), 0);

        cache.next_step();
        assert_eq!(cache.step(), 1);

        cache.next_step();
        assert_eq!(cache.step(), 2);
    }

    #[test]
    fn test_memory_tracking() {
        let mut cache = KVCache::new(12, 8, 64);
        assert_eq!(cache.current_memory_usage(), 0);

        let keys = vec![0.1f32; 8 * 64];
        let values = vec![0.2f32; 8 * 64];
        cache.update_layer(0, keys, values).unwrap();

        assert!(cache.current_memory_usage() > 0);
        assert!(cache.current_memory_usage_mb() > 0.0);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = KVCache::new(12, 8, 64);
        let keys = vec![0.1f32; 8 * 64];
        let values = vec![0.2f32; 8 * 64];

        cache.update_layer(0, keys, values).unwrap();
        cache.next_step();

        let stats = cache.stats();
        assert_eq!(stats.num_layers, 1);
        assert_eq!(stats.total_seq_len, 1);
        assert!(stats.memory_usage_mb > 0.0);
        assert_eq!(stats.step, 1);
        assert!(stats.enabled);
    }

    #[test]
    fn test_stats_summary() {
        let mut cache = KVCache::new(12, 8, 64);
        let keys = vec![0.1f32; 8 * 64];
        let values = vec![0.2f32; 8 * 64];

        cache.update_layer(0, keys, values).unwrap();

        let stats = cache.stats();
        let summary = stats.summary();
        assert!(summary.contains("Layers: 1"));
        assert!(summary.contains("Seq len: 1"));
    }

    #[test]
    fn test_disabled_cache() {
        let config = KVCacheConfig::new(12, 8, 64).with_enabled(false);
        let mut cache = KVCache::from_config(config).unwrap();
        assert!(!cache.is_enabled());

        let keys = vec![0.1f32; 8 * 64];
        let values = vec![0.2f32; 8 * 64];

        // Should succeed but not actually cache
        cache.update_layer(0, keys, values).unwrap();
        assert_eq!(cache.num_cached_layers(), 0);
    }

    #[test]
    fn test_mismatched_key_value_sizes() {
        let mut cache = KVCache::new(12, 8, 64);
        let keys = vec![0.1f32; 8 * 64];
        let values = vec![0.2f32; 4 * 64]; // Wrong size

        assert!(cache.update_layer(0, keys, values).is_err());
    }

    #[test]
    fn test_cache_entry_is_empty() {
        let entry = CacheEntry::new(1, 8, 64, 2048);
        assert!(entry.is_empty());
        assert_eq!(entry.len(), 0);
    }

    #[test]
    fn test_get_nonexistent_layer() {
        let cache = KVCache::new(12, 8, 64);
        assert!(cache.get_layer(0).is_err());
        assert!(cache.get_seq_len(0).is_err());
    }
}
