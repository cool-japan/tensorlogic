//! Memory optimization utilities for large factor graphs.
//!
//! This module provides memory-efficient representations and operations for
//! probabilistic graphical models, including:
//!
//! - Memory pooling for factor allocation
//! - Sparse factor representation for factors with many zeros
//! - Lazy evaluation for factor operations
//! - Memory-mapped factors for very large models

use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{PgmError, Result};
use crate::Factor;

/// Memory pool for factor value arrays.
///
/// Reuses allocated arrays to reduce allocation overhead
/// in iterative algorithms like message passing.
#[derive(Debug)]
pub struct FactorPool {
    /// Pool of available arrays by total size
    pools: Mutex<HashMap<usize, Vec<Vec<f64>>>>,
    /// Statistics
    stats: Mutex<PoolStats>,
    /// Maximum pool size per dimension
    max_pool_size: usize,
}

/// Statistics for memory pool usage.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Number of allocations served from pool
    pub hits: usize,
    /// Number of new allocations
    pub misses: usize,
    /// Number of arrays returned to pool
    pub returns: usize,
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Current memory in pool
    pub current_bytes: usize,
}

impl Default for FactorPool {
    fn default() -> Self {
        Self::new(100)
    }
}

impl FactorPool {
    /// Create a new factor pool with maximum size per dimension.
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            stats: Mutex::new(PoolStats::default()),
            max_pool_size,
        }
    }

    /// Allocate or reuse an array of the given size.
    pub fn allocate(&self, size: usize) -> Vec<f64> {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        if let Some(pool) = pools.get_mut(&size) {
            if let Some(array) = pool.pop() {
                stats.hits += 1;
                stats.current_bytes -= size * std::mem::size_of::<f64>();
                return array;
            }
        }

        stats.misses += 1;
        vec![0.0; size]
    }

    /// Return an array to the pool for reuse.
    pub fn return_array(&self, mut array: Vec<f64>) {
        let size = array.len();
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let pool = pools.entry(size).or_default();
        if pool.len() < self.max_pool_size {
            // Clear and return to pool
            array.fill(0.0);
            pool.push(array);
            stats.returns += 1;
            stats.current_bytes += size * std::mem::size_of::<f64>();
            stats.peak_bytes = stats.peak_bytes.max(stats.current_bytes);
        }
        // Otherwise, let it drop
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all pooled arrays.
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();
        pools.clear();
        stats.current_bytes = 0;
    }

    /// Get hit rate.
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        let total = stats.hits + stats.misses;
        if total > 0 {
            stats.hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Sparse factor representation using coordinate format.
///
/// Efficient for factors where most entries are zero or near-zero.
#[derive(Debug, Clone)]
pub struct SparseFactor {
    /// Variable names
    pub variables: Vec<String>,
    /// Cardinalities for each variable
    pub cardinalities: Vec<usize>,
    /// Non-zero entries: (indices, value)
    pub entries: Vec<(Vec<usize>, f64)>,
    /// Default value for entries not in sparse representation
    pub default_value: f64,
}

impl SparseFactor {
    /// Create a new sparse factor.
    pub fn new(variables: Vec<String>, cardinalities: Vec<usize>) -> Self {
        Self {
            variables,
            cardinalities,
            entries: Vec::new(),
            default_value: 0.0,
        }
    }

    /// Create from a dense factor with sparsity threshold.
    ///
    /// Values below threshold are treated as zero.
    pub fn from_dense(factor: &Factor, threshold: f64) -> Self {
        let shape: Vec<usize> = factor.values.shape().to_vec();
        let mut sparse = Self::new(factor.variables.clone(), shape.clone());
        sparse.default_value = 0.0;

        let total_size: usize = shape.iter().product();

        for i in 0..total_size {
            let indices = Self::flat_to_indices(i, &shape);
            let value = factor.values[indices.as_slice()];

            if value.abs() > threshold {
                sparse.entries.push((indices, value));
            }
        }

        sparse
    }

    /// Convert to dense factor.
    pub fn to_dense(&self) -> Result<Factor> {
        let total_size: usize = self.cardinalities.iter().product();
        let mut values = vec![self.default_value; total_size];

        for (indices, value) in &self.entries {
            let flat_idx = Self::indices_to_flat(indices, &self.cardinalities);
            values[flat_idx] = *value;
        }

        let array = ArrayD::from_shape_vec(IxDyn(&self.cardinalities), values)?;

        Factor::new("sparse".to_string(), self.variables.clone(), array)
    }

    /// Get value at indices.
    pub fn get(&self, indices: &[usize]) -> f64 {
        for (entry_indices, value) in &self.entries {
            if entry_indices == indices {
                return *value;
            }
        }
        self.default_value
    }

    /// Set value at indices.
    pub fn set(&mut self, indices: Vec<usize>, value: f64) {
        // Check if entry exists
        for (entry_indices, entry_value) in &mut self.entries {
            if *entry_indices == indices {
                *entry_value = value;
                return;
            }
        }

        // Add new entry
        if (value - self.default_value).abs() > 1e-10 {
            self.entries.push((indices, value));
        }
    }

    /// Get sparsity ratio (fraction of non-default entries).
    pub fn sparsity(&self) -> f64 {
        let total_size: usize = self.cardinalities.iter().product();
        if total_size > 0 {
            1.0 - (self.entries.len() as f64 / total_size as f64)
        } else {
            1.0
        }
    }

    /// Memory savings compared to dense representation.
    pub fn memory_savings(&self) -> f64 {
        let dense_bytes = self.cardinalities.iter().product::<usize>() * std::mem::size_of::<f64>();
        let sparse_bytes = self.entries.len()
            * (self.variables.len() * std::mem::size_of::<usize>() + std::mem::size_of::<f64>());

        if dense_bytes > 0 {
            1.0 - (sparse_bytes as f64 / dense_bytes as f64)
        } else {
            0.0
        }
    }

    /// Convert flat index to multi-dimensional indices.
    fn flat_to_indices(flat: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; shape.len()];
        let mut remaining = flat;

        for i in (0..shape.len()).rev() {
            indices[i] = remaining % shape[i];
            remaining /= shape[i];
        }

        indices
    }

    /// Convert multi-dimensional indices to flat index.
    fn indices_to_flat(indices: &[usize], shape: &[usize]) -> usize {
        let mut flat = 0;
        let mut stride = 1;

        for i in (0..shape.len()).rev() {
            flat += indices[i] * stride;
            stride *= shape[i];
        }

        flat
    }
}

/// Lazy factor that defers computation until needed.
///
/// Useful for chaining operations without intermediate allocations.
#[derive(Clone)]
pub struct LazyFactor {
    /// The computation to perform
    computation: Arc<dyn Fn() -> Result<Factor> + Send + Sync>,
    /// Cached result
    cached: Arc<Mutex<Option<Factor>>>,
}

impl std::fmt::Debug for LazyFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyFactor")
            .field("cached", &self.cached.lock().unwrap().is_some())
            .finish()
    }
}

impl LazyFactor {
    /// Create a new lazy factor from a computation.
    pub fn new<F>(computation: F) -> Self
    where
        F: Fn() -> Result<Factor> + Send + Sync + 'static,
    {
        Self {
            computation: Arc::new(computation),
            cached: Arc::new(Mutex::new(None)),
        }
    }

    /// Create from an already computed factor.
    pub fn from_factor(factor: Factor) -> Self {
        Self {
            computation: Arc::new(move || {
                Err(PgmError::InvalidDistribution(
                    "Already computed".to_string(),
                ))
            }),
            cached: Arc::new(Mutex::new(Some(factor))),
        }
    }

    /// Evaluate the lazy factor, computing if necessary.
    pub fn evaluate(&self) -> Result<Factor> {
        let mut cached = self.cached.lock().unwrap();

        if let Some(ref factor) = *cached {
            return Ok(factor.clone());
        }

        let result = (self.computation)()?;
        *cached = Some(result.clone());
        Ok(result)
    }

    /// Check if the factor has been computed.
    pub fn is_computed(&self) -> bool {
        self.cached.lock().unwrap().is_some()
    }

    /// Clear cached result to free memory.
    pub fn clear_cache(&self) {
        let mut cached = self.cached.lock().unwrap();
        *cached = None;
    }

    /// Create a lazy product of two factors.
    pub fn lazy_product(a: LazyFactor, b: LazyFactor) -> LazyFactor {
        LazyFactor::new(move || {
            let factor_a = a.evaluate()?;
            let factor_b = b.evaluate()?;
            factor_a.product(&factor_b)
        })
    }

    /// Create a lazy marginalization.
    pub fn lazy_marginalize(factor: LazyFactor, var: String) -> LazyFactor {
        LazyFactor::new(move || {
            let f = factor.evaluate()?;
            f.marginalize_out(&var)
        })
    }
}

/// Memory-efficient factor graph for very large models.
///
/// Uses streaming computation and doesn't hold all factors in memory.
pub struct StreamingFactorGraph {
    /// Variable information
    variables: HashMap<String, VariableInfo>,
    /// Factor generators (compute on demand)
    factor_generators: Vec<Box<dyn Fn() -> Result<Factor> + Send + Sync>>,
    /// Memory pool for allocations (reserved for future use)
    #[allow(dead_code)]
    pool: Arc<FactorPool>,
}

/// Information about a variable.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct VariableInfo {
    domain: String,
    cardinality: usize,
}

impl StreamingFactorGraph {
    /// Create a new streaming factor graph.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            factor_generators: Vec::new(),
            pool: Arc::new(FactorPool::default()),
        }
    }

    /// Create with a custom memory pool.
    pub fn with_pool(pool: Arc<FactorPool>) -> Self {
        Self {
            variables: HashMap::new(),
            factor_generators: Vec::new(),
            pool,
        }
    }

    /// Add a variable.
    pub fn add_variable(&mut self, name: String, domain: String, cardinality: usize) {
        self.variables.insert(
            name,
            VariableInfo {
                domain,
                cardinality,
            },
        );
    }

    /// Add a factor generator.
    pub fn add_factor<F>(&mut self, generator: F)
    where
        F: Fn() -> Result<Factor> + Send + Sync + 'static,
    {
        self.factor_generators.push(Box::new(generator));
    }

    /// Stream factors one at a time for memory-efficient processing.
    pub fn stream_factors(&self) -> impl Iterator<Item = Result<Factor>> + '_ {
        self.factor_generators.iter().map(|gen| gen())
    }

    /// Compute the product of all factors using streaming.
    ///
    /// Memory efficient but may be slower than batch computation.
    pub fn streaming_product(&self) -> Result<Factor> {
        let mut result: Option<Factor> = None;

        for gen in &self.factor_generators {
            let factor = gen()?;

            result = match result {
                None => Some(factor),
                Some(r) => Some(r.product(&factor)?),
            };
        }

        result.ok_or_else(|| PgmError::InvalidDistribution("No factors in graph".to_string()))
    }

    /// Number of variables.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Number of factors.
    pub fn num_factors(&self) -> usize {
        self.factor_generators.len()
    }
}

impl Default for StreamingFactorGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Compressed factor using quantization.
///
/// Reduces memory usage by storing values at lower precision.
#[derive(Debug, Clone)]
pub struct CompressedFactor {
    /// Variable names
    pub variables: Vec<String>,
    /// Cardinalities
    pub cardinalities: Vec<usize>,
    /// Quantized values (16-bit)
    quantized: Vec<u16>,
    /// Minimum value for dequantization
    min_value: f64,
    /// Scale for dequantization
    scale: f64,
}

impl CompressedFactor {
    /// Create from a dense factor.
    pub fn from_factor(factor: &Factor) -> Self {
        let values: Vec<f64> = factor.values.iter().copied().collect();
        let cardinalities: Vec<usize> = factor.values.shape().to_vec();

        let min_value = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let scale = if max_value > min_value {
            (max_value - min_value) / 65535.0
        } else {
            1.0
        };

        let quantized: Vec<u16> = values
            .iter()
            .map(|&v| ((v - min_value) / scale).round() as u16)
            .collect();

        Self {
            variables: factor.variables.clone(),
            cardinalities,
            quantized,
            min_value,
            scale,
        }
    }

    /// Convert back to dense factor.
    pub fn to_factor(&self) -> Result<Factor> {
        let values: Vec<f64> = self
            .quantized
            .iter()
            .map(|&q| self.min_value + (q as f64) * self.scale)
            .collect();

        let array = ArrayD::from_shape_vec(IxDyn(&self.cardinalities), values)?;

        Factor::new("compressed".to_string(), self.variables.clone(), array)
    }

    /// Memory size in bytes.
    pub fn memory_size(&self) -> usize {
        self.quantized.len() * std::mem::size_of::<u16>()
            + self.variables.len() * std::mem::size_of::<String>()
            + self.cardinalities.len() * std::mem::size_of::<usize>()
            + 2 * std::mem::size_of::<f64>()
    }

    /// Compression ratio compared to f64 representation.
    pub fn compression_ratio(&self) -> f64 {
        let original = self.quantized.len() * std::mem::size_of::<f64>();
        let compressed = self.quantized.len() * std::mem::size_of::<u16>();

        if compressed > 0 {
            original as f64 / compressed as f64
        } else {
            1.0
        }
    }
}

/// Block-sparse factor for factors with block structure.
///
/// Efficient when non-zero entries are clustered in blocks.
#[derive(Debug, Clone)]
pub struct BlockSparseFactor {
    /// Variable names
    pub variables: Vec<String>,
    /// Cardinalities
    pub cardinalities: Vec<usize>,
    /// Block size
    pub block_size: usize,
    /// Non-zero blocks: (block_index, values)
    blocks: HashMap<Vec<usize>, Vec<f64>>,
    /// Default block (all zeros or specific value)
    default_value: f64,
}

impl BlockSparseFactor {
    /// Create a new block-sparse factor.
    pub fn new(variables: Vec<String>, cardinalities: Vec<usize>, block_size: usize) -> Self {
        Self {
            variables,
            cardinalities,
            block_size,
            blocks: HashMap::new(),
            default_value: 0.0,
        }
    }

    /// Create from dense factor with sparsity detection.
    pub fn from_factor(factor: &Factor, block_size: usize, threshold: f64) -> Self {
        let shape: Vec<usize> = factor.values.shape().to_vec();
        let mut sparse = Self::new(factor.variables.clone(), shape.clone(), block_size);
        sparse.default_value = 0.0;
        let block_dims: Vec<usize> = shape.iter().map(|&d| d.div_ceil(block_size)).collect();

        // Iterate over blocks
        let total_blocks: usize = block_dims.iter().product();
        for block_flat in 0..total_blocks {
            let block_indices = SparseFactor::flat_to_indices(block_flat, &block_dims);

            // Extract block values
            let block_total = block_size.pow(shape.len() as u32);
            let mut block_values = Vec::with_capacity(block_total);
            let mut has_nonzero = false;

            for local_flat in 0..block_total {
                let local_indices =
                    SparseFactor::flat_to_indices(local_flat, &vec![block_size; shape.len()]);

                // Compute global indices
                let global_indices: Vec<usize> = block_indices
                    .iter()
                    .zip(local_indices.iter())
                    .zip(shape.iter())
                    .map(|((&bi, &li), &s)| (bi * block_size + li).min(s - 1))
                    .collect();

                let value = factor.values[global_indices.as_slice()];
                block_values.push(value);

                if value.abs() > threshold {
                    has_nonzero = true;
                }
            }

            if has_nonzero {
                sparse.blocks.insert(block_indices, block_values);
            }
        }

        sparse
    }

    /// Get number of non-zero blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get sparsity at block level.
    pub fn block_sparsity(&self) -> f64 {
        let block_dims: Vec<usize> = self
            .cardinalities
            .iter()
            .map(|&d| d.div_ceil(self.block_size))
            .collect();
        let total_blocks: usize = block_dims.iter().product();

        if total_blocks > 0 {
            1.0 - (self.blocks.len() as f64 / total_blocks as f64)
        } else {
            1.0
        }
    }
}

/// Estimate memory usage for a factor graph.
pub fn estimate_memory_usage(
    num_variables: usize,
    avg_cardinality: usize,
    num_factors: usize,
    avg_scope_size: usize,
) -> MemoryEstimate {
    let bytes_per_entry = std::mem::size_of::<f64>();
    let avg_factor_size = avg_cardinality.pow(avg_scope_size as u32);
    let total_factor_bytes = num_factors * avg_factor_size * bytes_per_entry;

    // Message storage for belief propagation
    let edges = num_factors * avg_scope_size;
    let message_bytes = 2 * edges * avg_cardinality * bytes_per_entry;

    // Variable marginal storage
    let marginal_bytes = num_variables * avg_cardinality * bytes_per_entry;

    MemoryEstimate {
        factor_bytes: total_factor_bytes,
        message_bytes,
        marginal_bytes,
        total_bytes: total_factor_bytes + message_bytes + marginal_bytes,
    }
}

/// Memory usage estimate.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Memory for factor values
    pub factor_bytes: usize,
    /// Memory for messages in belief propagation
    pub message_bytes: usize,
    /// Memory for marginals
    pub marginal_bytes: usize,
    /// Total estimated memory
    pub total_bytes: usize,
}

impl std::fmt::Display for MemoryEstimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let to_mb = |bytes: usize| bytes as f64 / 1_048_576.0;
        write!(
            f,
            "Memory Estimate: {:.2} MB total (factors: {:.2} MB, messages: {:.2} MB, marginals: {:.2} MB)",
            to_mb(self.total_bytes),
            to_mb(self.factor_bytes),
            to_mb(self.message_bytes),
            to_mb(self.marginal_bytes)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_factor_pool_allocation() {
        let pool = FactorPool::new(10);

        let arr1 = pool.allocate(100);
        assert_eq!(arr1.len(), 100);

        pool.return_array(arr1);
        assert_eq!(pool.stats().returns, 1);

        // Should reuse from pool
        let arr2 = pool.allocate(100);
        assert_eq!(arr2.len(), 100);
        assert_eq!(pool.stats().hits, 1);
    }

    #[test]
    fn test_factor_pool_hit_rate() {
        let pool = FactorPool::new(10);

        // First allocation is miss
        let arr = pool.allocate(50);
        pool.return_array(arr);

        // Second should be hit
        let _ = pool.allocate(50);

        assert!(pool.hit_rate() > 0.4); // At least one hit
    }

    #[test]
    fn test_sparse_factor_creation() {
        let mut sparse = SparseFactor::new(vec!["x".to_string()], vec![4]);

        sparse.set(vec![0], 1.0);
        sparse.set(vec![2], 0.5);

        assert_abs_diff_eq!(sparse.get(&[0]), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(&[1]), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(sparse.get(&[2]), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sparse_factor_from_dense() {
        let factor = Factor::new(
            "test".to_string(),
            vec!["x".to_string()],
            Array::from_vec(vec![0.0, 1.0, 0.0, 0.5]).into_dyn(),
        )
        .unwrap();

        let sparse = SparseFactor::from_dense(&factor, 0.1);
        assert_eq!(sparse.entries.len(), 2); // Only 1.0 and 0.5

        let dense = sparse.to_dense().unwrap();
        assert_abs_diff_eq!(dense.values[[1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dense.values[[3]], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sparse_factor_sparsity() {
        let mut sparse = SparseFactor::new(vec!["x".to_string()], vec![100]);
        sparse.set(vec![50], 1.0);

        let sparsity = sparse.sparsity();
        assert!(sparsity > 0.98); // 99% sparse
    }

    #[test]
    fn test_lazy_factor_deferred() {
        let counter = Arc::new(Mutex::new(0));
        let counter_clone = counter.clone();

        let lazy = LazyFactor::new(move || {
            let mut count = counter_clone.lock().unwrap();
            *count += 1;
            Factor::new(
                "test".to_string(),
                vec!["x".to_string()],
                Array::from_vec(vec![0.5, 0.5]).into_dyn(),
            )
        });

        assert!(!lazy.is_computed());
        assert_eq!(*counter.lock().unwrap(), 0);

        let _ = lazy.evaluate().unwrap();
        assert!(lazy.is_computed());
        assert_eq!(*counter.lock().unwrap(), 1);

        // Second evaluation uses cache
        let _ = lazy.evaluate().unwrap();
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[test]
    fn test_lazy_factor_from_factor() {
        let factor = Factor::new(
            "test".to_string(),
            vec!["x".to_string()],
            Array::from_vec(vec![0.3, 0.7]).into_dyn(),
        )
        .unwrap();

        let lazy = LazyFactor::from_factor(factor);
        assert!(lazy.is_computed());

        let result = lazy.evaluate().unwrap();
        assert_abs_diff_eq!(result.values[[0]], 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_compressed_factor() {
        let factor = Factor::new(
            "test".to_string(),
            vec!["x".to_string()],
            Array::from_vec(vec![0.1, 0.2, 0.3, 0.4]).into_dyn(),
        )
        .unwrap();

        let compressed = CompressedFactor::from_factor(&factor);
        let decompressed = compressed.to_factor().unwrap();

        // Values should be approximately preserved
        for i in 0..4 {
            assert_abs_diff_eq!(factor.values[[i]], decompressed.values[[i]], epsilon = 0.01);
        }
    }

    #[test]
    fn test_compressed_factor_ratio() {
        let factor = Factor::new(
            "test".to_string(),
            vec!["x".to_string(), "y".to_string()],
            ArrayD::from_elem(IxDyn(&[10, 10]), 0.5),
        )
        .unwrap();

        let compressed = CompressedFactor::from_factor(&factor);
        let ratio = compressed.compression_ratio();

        // 16-bit vs 64-bit should give ~4x compression
        assert!(ratio > 3.5);
    }

    #[test]
    fn test_streaming_factor_graph() {
        let mut graph = StreamingFactorGraph::new();
        graph.add_variable("x".to_string(), "Binary".to_string(), 2);
        graph.add_variable("y".to_string(), "Binary".to_string(), 2);

        graph.add_factor(|| {
            Factor::new(
                "factor_x".to_string(),
                vec!["x".to_string()],
                Array::from_vec(vec![0.3, 0.7]).into_dyn(),
            )
        });

        graph.add_factor(|| {
            Factor::new(
                "factor_y".to_string(),
                vec!["y".to_string()],
                Array::from_vec(vec![0.4, 0.6]).into_dyn(),
            )
        });

        assert_eq!(graph.num_variables(), 2);
        assert_eq!(graph.num_factors(), 2);
    }

    #[test]
    fn test_memory_estimate() {
        let estimate = estimate_memory_usage(10, 3, 20, 3);

        assert!(estimate.total_bytes > 0);
        assert!(estimate.factor_bytes > 0);
        assert!(estimate.message_bytes > 0);
    }

    #[test]
    fn test_block_sparse_factor() {
        let factor = Factor::new(
            "test".to_string(),
            vec!["x".to_string(), "y".to_string()],
            ArrayD::from_elem(IxDyn(&[8, 8]), 0.0),
        )
        .unwrap();

        let block_sparse = BlockSparseFactor::from_factor(&factor, 4, 0.001);

        // All zeros should give high block sparsity
        let sparsity = block_sparse.block_sparsity();
        assert!(sparsity > 0.99);
    }
}
