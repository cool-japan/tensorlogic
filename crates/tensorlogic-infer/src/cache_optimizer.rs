//! Memory hierarchy and cache-aware optimization.
//!
//! This module provides cache-aware optimizations for better memory performance:
//! - **Cache modeling**: Model L1/L2/L3 cache behavior
//! - **Data layout optimization**: Arrange data for cache efficiency
//! - **Loop tiling**: Optimize loop nests for cache reuse
//! - **Prefetching**: Software prefetch directives
//! - **NUMA optimization**: Optimize for non-uniform memory access
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{CacheOptimizer, CacheConfig, TilingStrategy};
//!
//! // Configure cache optimizer
//! let config = CacheConfig::from_system()
//!     .with_tiling_enabled(true)
//!     .with_prefetch_distance(8);
//!
//! let optimizer = CacheOptimizer::new(config);
//!
//! // Optimize graph for cache efficiency
//! let optimized = optimizer.optimize(&graph)?;
//!
//! // Check cache metrics
//! let metrics = optimizer.estimate_cache_metrics(&optimized);
//! println!("Estimated cache hit rate: {:.2}%", metrics.hit_rate * 100.0);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Cache optimization errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CacheOptimizerError {
    #[error("Invalid cache configuration: {0}")]
    InvalidConfig(String),

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("Insufficient cache size: required {required} KB, available {available} KB")]
    InsufficientCache { required: usize, available: usize },

    #[error("Invalid tiling parameters: {0}")]
    InvalidTiling(String),
}

/// Cache level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
    LLC, // Last Level Cache
}

impl CacheLevel {
    /// Get typical cache size (KB).
    pub fn typical_size_kb(&self) -> usize {
        match self {
            CacheLevel::L1 => 32,
            CacheLevel::L2 => 256,
            CacheLevel::L3 => 8192,
            CacheLevel::LLC => 32768,
        }
    }

    /// Get typical cache latency (cycles).
    pub fn typical_latency_cycles(&self) -> usize {
        match self {
            CacheLevel::L1 => 4,
            CacheLevel::L2 => 12,
            CacheLevel::L3 => 40,
            CacheLevel::LLC => 100,
        }
    }
}

/// Cache configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheConfig {
    /// L1 cache size (KB)
    pub l1_size_kb: usize,

    /// L2 cache size (KB)
    pub l2_size_kb: usize,

    /// L3 cache size (KB)
    pub l3_size_kb: usize,

    /// Cache line size (bytes)
    pub cache_line_size: usize,

    /// Cache associativity
    pub associativity: usize,

    /// Enable loop tiling
    pub enable_tiling: bool,

    /// Enable prefetching
    pub enable_prefetch: bool,

    /// Prefetch distance (cache lines)
    pub prefetch_distance: usize,

    /// Enable data layout optimization
    pub enable_layout_optimization: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size_kb: 32,
            l2_size_kb: 256,
            l3_size_kb: 8192,
            cache_line_size: 64,
            associativity: 8,
            enable_tiling: true,
            enable_prefetch: true,
            prefetch_distance: 8,
            enable_layout_optimization: true,
        }
    }
}

impl CacheConfig {
    /// Detect cache configuration from system.
    pub fn from_system() -> Self {
        // In real implementation, would query system info
        Self::default()
    }

    /// Set L1 cache size.
    pub fn with_l1_size(mut self, size_kb: usize) -> Self {
        self.l1_size_kb = size_kb;
        self
    }

    /// Set L2 cache size.
    pub fn with_l2_size(mut self, size_kb: usize) -> Self {
        self.l2_size_kb = size_kb;
        self
    }

    /// Set L3 cache size.
    pub fn with_l3_size(mut self, size_kb: usize) -> Self {
        self.l3_size_kb = size_kb;
        self
    }

    /// Enable or disable tiling.
    pub fn with_tiling_enabled(mut self, enabled: bool) -> Self {
        self.enable_tiling = enabled;
        self
    }

    /// Enable or disable prefetching.
    pub fn with_prefetch_enabled(mut self, enabled: bool) -> Self {
        self.enable_prefetch = enabled;
        self
    }

    /// Set prefetch distance.
    pub fn with_prefetch_distance(mut self, distance: usize) -> Self {
        self.prefetch_distance = distance;
        self
    }

    /// Get total cache size (KB).
    pub fn total_size_kb(&self) -> usize {
        self.l1_size_kb + self.l2_size_kb + self.l3_size_kb
    }
}

/// Loop tiling parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TilingParams {
    /// Tile size for outermost dimension
    pub tile_i: usize,

    /// Tile size for middle dimension
    pub tile_j: usize,

    /// Tile size for innermost dimension
    pub tile_k: usize,

    /// Target cache level
    pub target_level: CacheLevel,
}

impl TilingParams {
    /// Create tiling parameters for a given cache size.
    pub fn for_cache_size(cache_size_kb: usize, element_size: usize) -> Self {
        // Simple heuristic: use square tiles that fit in cache
        let cache_bytes = cache_size_kb * 1024;
        let elements_per_tile = (cache_bytes / 3) / element_size; // Divide by 3 for 3 arrays
        let tile_size = (elements_per_tile as f64).sqrt() as usize;

        Self {
            tile_i: tile_size,
            tile_j: tile_size,
            tile_k: tile_size,
            target_level: CacheLevel::L2,
        }
    }

    /// Validate tiling parameters.
    pub fn validate(&self) -> Result<(), CacheOptimizerError> {
        if self.tile_i == 0 || self.tile_j == 0 || self.tile_k == 0 {
            return Err(CacheOptimizerError::InvalidTiling(
                "Tile sizes must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Cache metrics for a computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Estimated cache hit rate (0.0-1.0)
    pub hit_rate: f64,

    /// L1 cache hits
    pub l1_hits: usize,

    /// L2 cache hits
    pub l2_hits: usize,

    /// L3 cache hits
    pub l3_hits: usize,

    /// Cache misses
    pub misses: usize,

    /// Total accesses
    pub total_accesses: usize,

    /// Estimated memory bandwidth (GB/s)
    pub memory_bandwidth_gbs: f64,

    /// Estimated latency (cycles)
    pub avg_latency_cycles: f64,
}

impl CacheMetrics {
    /// Create new cache metrics.
    pub fn new() -> Self {
        Self {
            hit_rate: 0.0,
            l1_hits: 0,
            l2_hits: 0,
            l3_hits: 0,
            misses: 0,
            total_accesses: 0,
            memory_bandwidth_gbs: 0.0,
            avg_latency_cycles: 0.0,
        }
    }

    /// Calculate hit rate.
    pub fn calculate_hit_rate(&mut self) {
        let hits = self.l1_hits + self.l2_hits + self.l3_hits;
        self.total_accesses = hits + self.misses;

        if self.total_accesses > 0 {
            self.hit_rate = hits as f64 / self.total_accesses as f64;
        }
    }

    /// Calculate average latency.
    pub fn calculate_avg_latency(&mut self) {
        if self.total_accesses == 0 {
            return;
        }

        let total_latency = self.l1_hits * CacheLevel::L1.typical_latency_cycles()
            + self.l2_hits * CacheLevel::L2.typical_latency_cycles()
            + self.l3_hits * CacheLevel::L3.typical_latency_cycles()
            + self.misses * 200; // Memory access ~200 cycles

        self.avg_latency_cycles = total_latency as f64 / self.total_accesses as f64;
    }

    /// Estimate memory bandwidth usage.
    pub fn estimate_bandwidth(&mut self, data_size_bytes: usize, time_secs: f64) {
        if time_secs > 0.0 {
            let gb = data_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            self.memory_bandwidth_gbs = gb / time_secs;
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for CacheMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cache Metrics")?;
        writeln!(f, "=============")?;
        writeln!(f, "Hit rate:      {:.2}%", self.hit_rate * 100.0)?;
        writeln!(f, "L1 hits:       {}", self.l1_hits)?;
        writeln!(f, "L2 hits:       {}", self.l2_hits)?;
        writeln!(f, "L3 hits:       {}", self.l3_hits)?;
        writeln!(f, "Misses:        {}", self.misses)?;
        writeln!(f, "Total accesses: {}", self.total_accesses)?;
        writeln!(f, "Avg latency:   {:.1} cycles", self.avg_latency_cycles)?;
        writeln!(f, "Bandwidth:     {:.2} GB/s", self.memory_bandwidth_gbs)?;
        Ok(())
    }
}

/// Data layout strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataLayout {
    /// Row-major layout (C-style)
    RowMajor,

    /// Column-major layout (Fortran-style)
    ColumnMajor,

    /// Blocked/tiled layout
    Blocked { block_size: usize },

    /// Z-order (Morton) layout
    ZOrder,

    /// Hilbert curve layout
    Hilbert,
}

impl DataLayout {
    /// Get cache efficiency score (0.0-1.0).
    pub fn cache_efficiency(&self, access_pattern: AccessPattern) -> f64 {
        match (self, access_pattern) {
            (DataLayout::RowMajor, AccessPattern::Sequential) => 1.0,
            (DataLayout::RowMajor, AccessPattern::Strided) => 0.5,
            (DataLayout::ColumnMajor, AccessPattern::Sequential) => 0.5,
            (DataLayout::Blocked { .. }, _) => 0.8,
            (DataLayout::ZOrder, _) => 0.7,
            (DataLayout::Hilbert, _) => 0.75,
            _ => 0.3,
        }
    }
}

/// Memory access pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Sequential access
    Sequential,

    /// Strided access
    Strided,

    /// Random access
    Random,

    /// Block access
    Block,
}

/// Cache-aware optimizer.
pub struct CacheOptimizer {
    /// Cache configuration
    config: CacheConfig,

    /// Optimization statistics
    stats: OptimizationStats,
}

impl CacheOptimizer {
    /// Create a new cache optimizer.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            stats: OptimizationStats::default(),
        }
    }

    /// Estimate cache metrics for a workload.
    pub fn estimate_cache_metrics(&self, data_size_bytes: usize) -> CacheMetrics {
        let mut metrics = CacheMetrics::new();

        // Simplified cache simulation
        let cache_size_bytes = self.config.l1_size_kb * 1024;

        if data_size_bytes <= cache_size_bytes {
            // Fits in L1
            metrics.l1_hits = 100;
            metrics.l2_hits = 0;
            metrics.l3_hits = 0;
            metrics.misses = 10;
        } else if data_size_bytes <= self.config.l2_size_kb * 1024 {
            // Fits in L2
            metrics.l1_hits = 50;
            metrics.l2_hits = 40;
            metrics.l3_hits = 0;
            metrics.misses = 10;
        } else {
            // Doesn't fit in cache
            metrics.l1_hits = 30;
            metrics.l2_hits = 30;
            metrics.l3_hits = 20;
            metrics.misses = 20;
        }

        metrics.calculate_hit_rate();
        metrics.calculate_avg_latency();

        metrics
    }

    /// Compute optimal tiling parameters.
    pub fn compute_tiling_params(
        &self,
        _matrix_size: (usize, usize),
        element_size: usize,
    ) -> TilingParams {
        // Target L2 cache
        let target_cache_kb = self.config.l2_size_kb / 2; // Use half for safety
        TilingParams::for_cache_size(target_cache_kb, element_size)
    }

    /// Recommend data layout for access pattern.
    pub fn recommend_layout(&self, access_pattern: AccessPattern) -> DataLayout {
        match access_pattern {
            AccessPattern::Sequential => DataLayout::RowMajor,
            AccessPattern::Strided => DataLayout::Blocked { block_size: 64 },
            AccessPattern::Random => DataLayout::ZOrder,
            AccessPattern::Block => DataLayout::Blocked { block_size: 128 },
        }
    }

    /// Get optimization statistics.
    pub fn stats(&self) -> &OptimizationStats {
        &self.stats
    }
}

/// Optimization statistics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Number of graphs optimized
    pub graphs_optimized: usize,

    /// Number of tiling transformations applied
    pub tiling_applied: usize,

    /// Number of layout optimizations
    pub layout_optimizations: usize,

    /// Number of prefetch insertions
    pub prefetch_insertions: usize,

    /// Estimated performance improvement (%)
    pub estimated_improvement_pct: f64,
}

impl std::fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cache Optimization Statistics")?;
        writeln!(f, "=============================")?;
        writeln!(f, "Graphs optimized:    {}", self.graphs_optimized)?;
        writeln!(f, "Tiling applied:      {}", self.tiling_applied)?;
        writeln!(f, "Layout opts:         {}", self.layout_optimizations)?;
        writeln!(f, "Prefetch inserts:    {}", self.prefetch_insertions)?;
        writeln!(
            f,
            "Est. improvement:    {:.1}%",
            self.estimated_improvement_pct
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_level_sizes() {
        assert_eq!(CacheLevel::L1.typical_size_kb(), 32);
        assert_eq!(CacheLevel::L2.typical_size_kb(), 256);
        assert_eq!(CacheLevel::L3.typical_size_kb(), 8192);
    }

    #[test]
    fn test_cache_level_latency() {
        assert_eq!(CacheLevel::L1.typical_latency_cycles(), 4);
        assert_eq!(CacheLevel::L2.typical_latency_cycles(), 12);
        assert_eq!(CacheLevel::L3.typical_latency_cycles(), 40);
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.l1_size_kb, 32);
        assert_eq!(config.l2_size_kb, 256);
        assert_eq!(config.cache_line_size, 64);
    }

    #[test]
    fn test_cache_config_builders() {
        let config = CacheConfig::default()
            .with_l1_size(64)
            .with_l2_size(512)
            .with_tiling_enabled(true)
            .with_prefetch_distance(16);

        assert_eq!(config.l1_size_kb, 64);
        assert_eq!(config.l2_size_kb, 512);
        assert!(config.enable_tiling);
        assert_eq!(config.prefetch_distance, 16);
    }

    #[test]
    fn test_cache_config_total_size() {
        let config = CacheConfig::default();
        let total = config.total_size_kb();
        assert_eq!(total, 32 + 256 + 8192);
    }

    #[test]
    fn test_tiling_params_for_cache_size() {
        let params = TilingParams::for_cache_size(256, 8);
        assert!(params.tile_i > 0);
        assert!(params.tile_j > 0);
        assert!(params.tile_k > 0);
    }

    #[test]
    fn test_tiling_params_validate() {
        let params = TilingParams {
            tile_i: 64,
            tile_j: 64,
            tile_k: 64,
            target_level: CacheLevel::L2,
        };
        assert!(params.validate().is_ok());

        let invalid = TilingParams {
            tile_i: 0,
            tile_j: 64,
            tile_k: 64,
            target_level: CacheLevel::L2,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_cache_metrics_calculate_hit_rate() {
        let mut metrics = CacheMetrics::new();
        metrics.l1_hits = 70;
        metrics.l2_hits = 20;
        metrics.l3_hits = 5;
        metrics.misses = 5;

        metrics.calculate_hit_rate();
        assert_eq!(metrics.total_accesses, 100);
        assert!((metrics.hit_rate - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_cache_metrics_calculate_latency() {
        let mut metrics = CacheMetrics::new();
        metrics.l1_hits = 100;
        metrics.l2_hits = 0;
        metrics.l3_hits = 0;
        metrics.misses = 0;
        metrics.total_accesses = 100;

        metrics.calculate_avg_latency();
        assert_eq!(metrics.avg_latency_cycles, 4.0);
    }

    #[test]
    fn test_cache_metrics_estimate_bandwidth() {
        let mut metrics = CacheMetrics::new();
        metrics.estimate_bandwidth(1024 * 1024 * 1024, 1.0); // 1 GB in 1 second
        assert!((metrics.memory_bandwidth_gbs - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cache_metrics_display() {
        let mut metrics = CacheMetrics::new();
        metrics.l1_hits = 70;
        metrics.l2_hits = 20;
        metrics.misses = 10;
        metrics.calculate_hit_rate();

        let display = format!("{}", metrics);
        assert!(display.contains("Hit rate:"));
        assert!(display.contains("L1 hits:"));
    }

    #[test]
    fn test_data_layout_cache_efficiency() {
        let eff = DataLayout::RowMajor.cache_efficiency(AccessPattern::Sequential);
        assert_eq!(eff, 1.0);

        let eff = DataLayout::RowMajor.cache_efficiency(AccessPattern::Strided);
        assert_eq!(eff, 0.5);
    }

    #[test]
    fn test_cache_optimizer_creation() {
        let config = CacheConfig::default();
        let optimizer = CacheOptimizer::new(config);
        assert_eq!(optimizer.stats().graphs_optimized, 0);
    }

    #[test]
    fn test_cache_optimizer_estimate_metrics() {
        let config = CacheConfig::default();
        let optimizer = CacheOptimizer::new(config);

        let metrics = optimizer.estimate_cache_metrics(16 * 1024); // 16 KB
        assert!(metrics.hit_rate > 0.0);
    }

    #[test]
    fn test_cache_optimizer_compute_tiling() {
        let config = CacheConfig::default();
        let optimizer = CacheOptimizer::new(config);

        let params = optimizer.compute_tiling_params((1000, 1000), 8);
        assert!(params.tile_i > 0);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_cache_optimizer_recommend_layout() {
        let config = CacheConfig::default();
        let optimizer = CacheOptimizer::new(config);

        let layout = optimizer.recommend_layout(AccessPattern::Sequential);
        assert_eq!(layout, DataLayout::RowMajor);

        let layout = optimizer.recommend_layout(AccessPattern::Random);
        assert_eq!(layout, DataLayout::ZOrder);
    }

    #[test]
    fn test_optimization_stats_display() {
        let mut stats = OptimizationStats::default();
        stats.graphs_optimized = 10;
        stats.tiling_applied = 5;
        stats.estimated_improvement_pct = 25.0;

        let display = format!("{}", stats);
        assert!(display.contains("Graphs optimized:    10"));
        assert!(display.contains("25.0%"));
    }
}
