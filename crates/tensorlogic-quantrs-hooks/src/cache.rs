//! Caching and memoization for factor operations.
//!
//! This module provides caching mechanisms to avoid recomputing expensive factor operations
//! like product, marginalization, and division. This can significantly improve performance
//! when the same operations are performed repeatedly.

use crate::error::Result;
use crate::factor::Factor;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// A key for caching factor operations.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum CacheKey {
    /// Product of two factors
    Product(String, String),
    /// Marginalization of a factor over a variable
    Marginalize(String, String),
    /// Division of two factors
    Divide(String, String),
    /// Reduction of a factor given evidence
    Reduce(String, String, usize),
}

/// A cache for factor operations.
///
/// This cache stores the results of expensive factor operations to avoid recomputation.
/// It uses a simple LRU-like eviction policy based on size limits.
pub struct FactorCache {
    /// The cached factors
    cache: Arc<Mutex<HashMap<CacheKey, Factor>>>,
    /// Maximum number of cached entries
    max_size: usize,
    /// Statistics
    hits: Arc<Mutex<usize>>,
    misses: Arc<Mutex<usize>>,
}

impl Default for FactorCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl FactorCache {
    /// Create a new factor cache with a maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            hits: Arc::new(Mutex::new(0)),
            misses: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a cached product result.
    pub fn get_product(&self, f1_name: &str, f2_name: &str) -> Option<Factor> {
        let key = CacheKey::Product(f1_name.to_string(), f2_name.to_string());
        self.get(&key)
    }

    /// Cache a product result.
    pub fn put_product(&self, f1_name: &str, f2_name: &str, result: Factor) {
        let key = CacheKey::Product(f1_name.to_string(), f2_name.to_string());
        self.put(key, result);
    }

    /// Get a cached marginalization result.
    pub fn get_marginalize(&self, factor_name: &str, var: &str) -> Option<Factor> {
        let key = CacheKey::Marginalize(factor_name.to_string(), var.to_string());
        self.get(&key)
    }

    /// Cache a marginalization result.
    pub fn put_marginalize(&self, factor_name: &str, var: &str, result: Factor) {
        let key = CacheKey::Marginalize(factor_name.to_string(), var.to_string());
        self.put(key, result);
    }

    /// Get a cached division result.
    pub fn get_divide(&self, f1_name: &str, f2_name: &str) -> Option<Factor> {
        let key = CacheKey::Divide(f1_name.to_string(), f2_name.to_string());
        self.get(&key)
    }

    /// Cache a division result.
    pub fn put_divide(&self, f1_name: &str, f2_name: &str, result: Factor) {
        let key = CacheKey::Divide(f1_name.to_string(), f2_name.to_string());
        self.put(key, result);
    }

    /// Get a cached reduction result.
    pub fn get_reduce(&self, factor_name: &str, var: &str, value: usize) -> Option<Factor> {
        let key = CacheKey::Reduce(factor_name.to_string(), var.to_string(), value);
        self.get(&key)
    }

    /// Cache a reduction result.
    pub fn put_reduce(&self, factor_name: &str, var: &str, value: usize, result: Factor) {
        let key = CacheKey::Reduce(factor_name.to_string(), var.to_string(), value);
        self.put(key, result);
    }

    /// Get from cache.
    fn get(&self, key: &CacheKey) -> Option<Factor> {
        let cache = self.cache.lock().unwrap();
        if let Some(factor) = cache.get(key) {
            *self.hits.lock().unwrap() += 1;
            Some(factor.clone())
        } else {
            *self.misses.lock().unwrap() += 1;
            None
        }
    }

    /// Put into cache.
    fn put(&self, key: CacheKey, factor: Factor) {
        let mut cache = self.cache.lock().unwrap();

        // Simple eviction: if at max size, remove a random entry
        if cache.len() >= self.max_size {
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }

        cache.insert(key, factor);
    }

    /// Clear the cache.
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
        *self.hits.lock().unwrap() = 0;
        *self.misses.lock().unwrap() = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let hits = *self.hits.lock().unwrap();
        let misses = *self.misses.lock().unwrap();
        let size = self.cache.lock().unwrap().len();

        CacheStats {
            hits,
            misses,
            size,
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Get current cache size.
    pub fn size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Current cache size
    pub size: usize,
    /// Hit rate (hits / (hits + misses))
    pub hit_rate: f64,
}

/// A cached factor that memoizes operations.
///
/// This wraps a factor and caches the results of operations.
pub struct CachedFactor {
    /// The underlying factor
    pub factor: Factor,
    /// The cache
    cache: Arc<FactorCache>,
}

impl CachedFactor {
    /// Create a new cached factor.
    pub fn new(factor: Factor, cache: Arc<FactorCache>) -> Self {
        Self { factor, cache }
    }

    /// Compute product with caching.
    pub fn product_cached(&self, other: &CachedFactor) -> Result<Factor> {
        // Try to get from cache
        if let Some(cached) = self
            .cache
            .get_product(&self.factor.name, &other.factor.name)
        {
            return Ok(cached);
        }

        // Compute and cache
        let result = self.factor.product(&other.factor)?;
        self.cache
            .put_product(&self.factor.name, &other.factor.name, result.clone());

        Ok(result)
    }

    /// Compute marginalization with caching.
    pub fn marginalize_out_cached(&self, var: &str) -> Result<Factor> {
        // Try to get from cache
        if let Some(cached) = self.cache.get_marginalize(&self.factor.name, var) {
            return Ok(cached);
        }

        // Compute and cache
        let result = self.factor.marginalize_out(var)?;
        self.cache
            .put_marginalize(&self.factor.name, var, result.clone());

        Ok(result)
    }

    /// Compute division with caching.
    pub fn divide_cached(&self, other: &CachedFactor) -> Result<Factor> {
        // Try to get from cache
        if let Some(cached) = self.cache.get_divide(&self.factor.name, &other.factor.name) {
            return Ok(cached);
        }

        // Compute and cache
        let result = self.factor.divide(&other.factor)?;
        self.cache
            .put_divide(&self.factor.name, &other.factor.name, result.clone());

        Ok(result)
    }

    /// Compute reduction with caching.
    pub fn reduce_cached(&self, var: &str, value: usize) -> Result<Factor> {
        // Try to get from cache
        if let Some(cached) = self.cache.get_reduce(&self.factor.name, var, value) {
            return Ok(cached);
        }

        // Compute and cache
        let result = self.factor.reduce(var, value)?;
        self.cache
            .put_reduce(&self.factor.name, var, value, result.clone());

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn create_test_factor(name: &str) -> Factor {
        let values = vec![0.1, 0.2, 0.3, 0.4];
        let array = Array::from_shape_vec(vec![2, 2], values)
            .unwrap()
            .into_dyn();
        Factor::new(
            name.to_string(),
            vec!["X".to_string(), "Y".to_string()],
            array,
        )
        .unwrap()
    }

    #[test]
    fn test_cache_product() {
        let cache = Arc::new(FactorCache::new(100));
        let f1 = CachedFactor::new(create_test_factor("f1"), cache.clone());
        let f2 = CachedFactor::new(create_test_factor("f2"), cache.clone());

        // First call - cache miss
        let result1 = f1.product_cached(&f2).unwrap();
        let stats1 = cache.stats();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);

        // Second call - cache hit
        let result2 = f1.product_cached(&f2).unwrap();
        let stats2 = cache.stats();
        assert_eq!(stats2.misses, 1);
        assert_eq!(stats2.hits, 1);

        // Results should be the same
        assert_eq!(result1.name, result2.name);
    }

    #[test]
    fn test_cache_marginalize() {
        let cache = Arc::new(FactorCache::new(100));
        let f = CachedFactor::new(create_test_factor("f"), cache.clone());

        // First call - cache miss
        let _result1 = f.marginalize_out_cached("Y").unwrap();
        let stats1 = cache.stats();
        assert_eq!(stats1.misses, 1);

        // Second call - cache hit
        let _result2 = f.marginalize_out_cached("Y").unwrap();
        let stats2 = cache.stats();
        assert_eq!(stats2.hits, 1);
    }

    #[test]
    fn test_cache_stats() {
        let cache = FactorCache::new(100);
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[test]
    fn test_cache_clear() {
        let cache = Arc::new(FactorCache::new(100));
        let f = CachedFactor::new(create_test_factor("f"), cache.clone());

        // Populate cache
        let _ = f.marginalize_out_cached("Y").unwrap();
        assert_eq!(cache.size(), 1);

        // Clear cache
        cache.clear();
        assert_eq!(cache.size(), 0);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = Arc::new(FactorCache::new(2));

        // Add 3 items (should evict oldest)
        cache.put_marginalize("f1", "X", create_test_factor("result1"));
        cache.put_marginalize("f2", "Y", create_test_factor("result2"));
        cache.put_marginalize("f3", "Z", create_test_factor("result3"));

        // Size should be at most 2
        assert!(cache.size() <= 2);
    }
}
