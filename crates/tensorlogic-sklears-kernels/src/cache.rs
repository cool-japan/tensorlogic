//! Kernel caching infrastructure for performance optimization.
//!
//! Provides caching mechanisms to avoid redundant kernel computations.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::types::Kernel;

/// Hash key for caching kernel computations
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    /// Hash of first input
    x_hash: u64,
    /// Hash of second input
    y_hash: u64,
}

impl CacheKey {
    /// Create a cache key from two input vectors
    fn from_inputs(x: &[f64], y: &[f64]) -> Self {
        Self {
            x_hash: Self::hash_vector(x),
            y_hash: Self::hash_vector(y),
        }
    }

    /// Hash a vector of f64 values
    fn hash_vector(v: &[f64]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &val in v {
            // Convert to bits for consistent hashing
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Cached kernel wrapper that stores computed values
///
/// # Example
///
/// ```rust
/// use tensorlogic_sklears_kernels::{LinearKernel, CachedKernel, Kernel};
///
/// let base_kernel = LinearKernel::new();
/// let mut cached = CachedKernel::new(Box::new(base_kernel));
///
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![4.0, 5.0, 6.0];
///
/// // First call computes and caches
/// let result1 = cached.compute(&x, &y).unwrap();
///
/// // Second call retrieves from cache
/// let result2 = cached.compute(&x, &y).unwrap();
/// assert_eq!(result1, result2);
///
/// // Check cache statistics
/// let stats = cached.stats();
/// assert!(stats.hits > 0);
/// ```
pub struct CachedKernel {
    /// Underlying kernel
    inner: Box<dyn Kernel>,
    /// Cache storage
    cache: Arc<Mutex<HashMap<CacheKey, f64>>>,
    /// Cache statistics
    stats: Arc<Mutex<CacheStats>>,
}

/// Cache statistics
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of entries in cache
    pub size: usize,
}

impl CacheStats {
    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl CachedKernel {
    /// Create a new cached kernel
    pub fn new(inner: Box<dyn Kernel>) -> Self {
        Self {
            inner,
            cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.lock().unwrap().clear();
        let mut stats = self.stats.lock().unwrap();
        stats.hits = 0;
        stats.misses = 0;
        stats.size = 0;
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

impl Kernel for CachedKernel {
    fn compute(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let key = CacheKey::from_inputs(x, y);

        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(&value) = cache.get(&key) {
                let mut stats = self.stats.lock().unwrap();
                stats.hits += 1;
                return Ok(value);
            }
        }

        // Cache miss - compute value
        let value = self.inner.compute(x, y)?;

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, value);

            let mut stats = self.stats.lock().unwrap();
            stats.misses += 1;
            stats.size = cache.len();
        }

        Ok(value)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn is_psd(&self) -> bool {
        self.inner.is_psd()
    }
}

/// Kernel matrix cache for efficient matrix operations
///
/// Stores entire kernel matrices to avoid recomputation.
///
/// # Example
///
/// ```rust
/// use tensorlogic_sklears_kernels::{LinearKernel, KernelMatrixCache};
///
/// let kernel = LinearKernel::new();
/// let mut cache = KernelMatrixCache::new();
///
/// let data = vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0],
///     vec![5.0, 6.0],
/// ];
///
/// // Compute and cache
/// let matrix1 = cache.get_or_compute(&data, &kernel).unwrap();
///
/// // Retrieve from cache
/// let matrix2 = cache.get_or_compute(&data, &kernel).unwrap();
///
/// assert_eq!(matrix1.len(), matrix2.len());
/// ```
pub struct KernelMatrixCache {
    /// Cache storage
    cache: HashMap<u64, Vec<Vec<f64>>>,
}

impl KernelMatrixCache {
    /// Create a new matrix cache
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Hash input data
    fn hash_data(data: &[Vec<f64>]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for row in data {
            for &val in row {
                val.to_bits().hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Get or compute kernel matrix
    pub fn get_or_compute(
        &mut self,
        data: &[Vec<f64>],
        kernel: &dyn Kernel,
    ) -> Result<Vec<Vec<f64>>> {
        let key = Self::hash_data(data);

        if let Some(matrix) = self.cache.get(&key) {
            return Ok(matrix.clone());
        }

        // Compute matrix
        let matrix = kernel.compute_matrix(data)?;
        self.cache.insert(key, matrix.clone());

        Ok(matrix)
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

impl Default for KernelMatrixCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_kernels::LinearKernel;

    #[test]
    fn test_cached_kernel() {
        let base = LinearKernel::new();
        let cached = CachedKernel::new(Box::new(base));

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        // First call - cache miss
        let result1 = cached.compute(&x, &y).unwrap();
        let stats1 = cached.stats();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);

        // Second call - cache hit
        let result2 = cached.compute(&x, &y).unwrap();
        let stats2 = cached.stats();
        assert_eq!(stats2.misses, 1);
        assert_eq!(stats2.hits, 1);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_cached_kernel_clear() {
        let base = LinearKernel::new();
        let mut cached = CachedKernel::new(Box::new(base));

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        cached.compute(&x, &y).unwrap();
        assert_eq!(cached.cache_size(), 1);

        cached.clear();
        assert_eq!(cached.cache_size(), 0);

        let stats = cached.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 7,
            misses: 3,
            size: 10,
        };

        assert!((stats.hit_rate() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_cache_stats_empty() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_kernel_matrix_cache() {
        let kernel = LinearKernel::new();
        let mut cache = KernelMatrixCache::new();

        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        // First call - compute
        let matrix1 = cache.get_or_compute(&data, &kernel).unwrap();
        assert_eq!(cache.size(), 1);

        // Second call - retrieve from cache
        let matrix2 = cache.get_or_compute(&data, &kernel).unwrap();
        assert_eq!(cache.size(), 1);

        assert_eq!(matrix1.len(), matrix2.len());
        for i in 0..matrix1.len() {
            for j in 0..matrix1[i].len() {
                assert_eq!(matrix1[i][j], matrix2[i][j]);
            }
        }
    }

    #[test]
    fn test_kernel_matrix_cache_clear() {
        let kernel = LinearKernel::new();
        let mut cache = KernelMatrixCache::new();

        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        cache.get_or_compute(&data, &kernel).unwrap();
        assert_eq!(cache.size(), 1);

        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cached_kernel_name() {
        let base = LinearKernel::new();
        let cached = CachedKernel::new(Box::new(base));
        assert_eq!(cached.name(), "Linear");
    }

    #[test]
    fn test_cached_kernel_psd() {
        let base = LinearKernel::new();
        let cached = CachedKernel::new(Box::new(base));
        assert!(cached.is_psd());
    }
}
