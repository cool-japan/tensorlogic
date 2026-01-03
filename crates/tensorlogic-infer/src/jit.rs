//! Just-In-Time (JIT) compilation infrastructure.
//!
//! This module provides runtime compilation and adaptive optimization capabilities:
//! - `JitCompiler`: Runtime compilation with hot path detection
//! - `JitCache`: Specialized caching for JIT-compiled graphs
//! - `HotPathDetector`: Identifies frequently executed code paths
//! - `AdaptiveOptimizer`: Progressively optimizes based on runtime profiling
//! - `TlJitExecutor`: Trait for executors that support JIT compilation
//!
//! # JIT Compilation Workflow
//!
//! 1. **First Execution**: Graph is compiled with minimal optimization
//! 2. **Profiling**: Runtime characteristics are collected
//! 3. **Hot Path Detection**: Frequently executed paths are identified
//! 4. **Adaptive Optimization**: Hot paths are recompiled with aggressive optimization
//! 5. **Specialization**: Graphs are specialized for observed shapes/types
//!
//! # Example
//!
//! ```
//! use tensorlogic_infer::jit::{JitCompiler, JitConfig};
//! use tensorlogic_ir::EinsumGraph;
//!
//! let mut jit = JitCompiler::new(JitConfig::default());
//! let graph = EinsumGraph::new();
//!
//! // First execution: minimal compilation
//! let compiled = jit.compile_or_retrieve(&graph, &[]).unwrap();
//!
//! // After profiling, hot paths are recompiled with aggressive optimization
//! jit.optimize_hot_paths();
//! ```

use crate::compilation::{CompilationConfig, CompiledGraph, GraphCompiler, OptimizationLevel};
use crate::error::ExecutorError;
use crate::shape::TensorShape;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tensorlogic_ir::EinsumGraph;

/// Configuration for JIT compilation.
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Initial optimization level for first compilation
    pub initial_optimization: OptimizationLevel,
    /// Hot path optimization level
    pub hot_path_optimization: OptimizationLevel,
    /// Minimum execution count to consider a path "hot"
    pub hot_path_threshold: usize,
    /// Enable shape specialization
    pub enable_specialization: bool,
    /// Maximum number of specialized versions per graph
    pub max_specializations: usize,
    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,
    /// Profiling window size for hot path detection
    pub profiling_window: usize,
    /// Cache size limit (number of compiled graphs)
    pub cache_size: usize,
    /// Enable deoptimization for rarely used paths
    pub enable_deoptimization: bool,
    /// Threshold for deoptimization (executions per time window)
    pub deoptimization_threshold: usize,
}

impl Default for JitConfig {
    fn default() -> Self {
        JitConfig {
            initial_optimization: OptimizationLevel::Basic,
            hot_path_optimization: OptimizationLevel::Aggressive,
            hot_path_threshold: 10,
            enable_specialization: true,
            max_specializations: 5,
            enable_adaptive_optimization: true,
            profiling_window: 100,
            cache_size: 1000,
            enable_deoptimization: true,
            deoptimization_threshold: 1,
        }
    }
}

/// Key for identifying graphs and their specializations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JitKey {
    /// Hash of the graph structure
    pub graph_hash: u64,
    /// Specialization context (shapes, if enabled)
    pub specialization: Option<SpecializationContext>,
}

/// Context for graph specialization.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpecializationContext {
    /// Input shapes for specialization
    pub input_shapes: Vec<Vec<usize>>,
    /// Device target (if specified)
    pub device: Option<String>,
}

impl SpecializationContext {
    /// Create a new specialization context from input shapes.
    pub fn from_shapes(shapes: &[TensorShape]) -> Self {
        SpecializationContext {
            input_shapes: shapes
                .iter()
                .map(|s| {
                    s.dims
                        .iter()
                        .filter_map(|d| d.as_static())
                        .collect::<Vec<_>>()
                })
                .collect(),
            device: None,
        }
    }

    /// Create a context with device specification.
    pub fn with_device(mut self, device: String) -> Self {
        self.device = Some(device);
        self
    }
}

/// Statistics for a compiled graph in the JIT cache.
#[derive(Debug, Clone)]
pub struct JitEntryStats {
    /// Number of times this compiled version has been executed
    pub execution_count: usize,
    /// Total execution time for this version
    pub total_execution_time: Duration,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Optimization level used
    pub optimization_level: OptimizationLevel,
    /// Timestamp of last execution
    pub last_executed: Instant,
    /// Timestamp when compiled
    pub compiled_at: Instant,
    /// Whether this is a specialized version
    pub is_specialized: bool,
}

impl Default for JitEntryStats {
    fn default() -> Self {
        JitEntryStats {
            execution_count: 0,
            total_execution_time: Duration::from_secs(0),
            avg_execution_time: Duration::from_secs(0),
            optimization_level: OptimizationLevel::Basic,
            last_executed: Instant::now(),
            compiled_at: Instant::now(),
            is_specialized: false,
        }
    }
}

impl JitEntryStats {
    /// Record an execution of this compiled graph.
    pub fn record_execution(&mut self, duration: Duration) {
        self.execution_count += 1;
        self.total_execution_time += duration;
        self.avg_execution_time = self.total_execution_time / self.execution_count as u32;
        self.last_executed = Instant::now();
    }

    /// Check if this entry is "hot" based on execution count.
    pub fn is_hot(&self, threshold: usize) -> bool {
        self.execution_count >= threshold
    }

    /// Check if this entry is cold (rarely used).
    pub fn is_cold(&self, threshold: usize, window: Duration) -> bool {
        let time_since_last = Instant::now().duration_since(self.last_executed);
        time_since_last > window && self.execution_count < threshold
    }
}

/// Entry in the JIT cache.
#[derive(Debug, Clone)]
pub struct JitCacheEntry {
    /// The compiled graph
    pub compiled: CompiledGraph,
    /// Statistics for this entry
    pub stats: JitEntryStats,
}

/// Cache for JIT-compiled graphs with profiling support.
pub struct JitCache {
    cache: Arc<RwLock<HashMap<JitKey, JitCacheEntry>>>,
    config: JitConfig,
}

impl JitCache {
    /// Create a new JIT cache.
    pub fn new(config: JitConfig) -> Self {
        JitCache {
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Insert a compiled graph into the cache.
    pub fn insert(&self, key: JitKey, compiled: CompiledGraph, is_specialized: bool) {
        let mut cache = self.cache.write().unwrap();

        // Evict old entries if cache is full
        if cache.len() >= self.config.cache_size {
            self.evict_lru(&mut cache);
        }

        let stats = JitEntryStats {
            optimization_level: compiled.config.optimization_level,
            is_specialized,
            ..Default::default()
        };

        cache.insert(key, JitCacheEntry { compiled, stats });
    }

    /// Retrieve a compiled graph from the cache.
    pub fn get(&self, key: &JitKey) -> Option<CompiledGraph> {
        let cache = self.cache.read().unwrap();
        cache.get(key).map(|entry| entry.compiled.clone())
    }

    /// Record an execution of a cached graph.
    pub fn record_execution(&self, key: &JitKey, duration: Duration) {
        let mut cache = self.cache.write().unwrap();
        if let Some(entry) = cache.get_mut(key) {
            entry.stats.record_execution(duration);
        }
    }

    /// Get statistics for a cached entry.
    pub fn get_stats(&self, key: &JitKey) -> Option<JitEntryStats> {
        let cache = self.cache.read().unwrap();
        cache.get(key).map(|entry| entry.stats.clone())
    }

    /// Get all hot paths (frequently executed graphs).
    pub fn get_hot_paths(&self) -> Vec<(JitKey, JitEntryStats)> {
        let cache = self.cache.read().unwrap();
        cache
            .iter()
            .filter(|(_, entry)| entry.stats.is_hot(self.config.hot_path_threshold))
            .map(|(key, entry)| (key.clone(), entry.stats.clone()))
            .collect()
    }

    /// Get all cold paths (rarely executed graphs).
    pub fn get_cold_paths(&self) -> Vec<(JitKey, JitEntryStats)> {
        let cache = self.cache.read().unwrap();
        let window = Duration::from_secs(300); // 5 minutes
        cache
            .iter()
            .filter(|(_, entry)| {
                entry
                    .stats
                    .is_cold(self.config.deoptimization_threshold, window)
            })
            .map(|(key, entry)| (key.clone(), entry.stats.clone()))
            .collect()
    }

    /// Evict least recently used entry.
    fn evict_lru(&self, cache: &mut HashMap<JitKey, JitCacheEntry>) {
        if let Some((key, _)) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.stats.last_executed)
        {
            let key = key.clone();
            cache.remove(&key);
        }
    }

    /// Clear the cache.
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> JitCacheStats {
        let cache = self.cache.read().unwrap();
        let total_entries = cache.len();
        let hot_entries = cache
            .values()
            .filter(|e| e.stats.is_hot(self.config.hot_path_threshold))
            .count();
        let specialized_entries = cache.values().filter(|e| e.stats.is_specialized).count();
        let total_executions = cache.values().map(|e| e.stats.execution_count).sum();

        JitCacheStats {
            total_entries,
            hot_entries,
            specialized_entries,
            total_executions,
            cache_capacity: self.config.cache_size,
        }
    }
}

/// Statistics for the JIT cache.
#[derive(Debug, Clone)]
pub struct JitCacheStats {
    /// Total number of entries in the cache
    pub total_entries: usize,
    /// Number of hot entries
    pub hot_entries: usize,
    /// Number of specialized entries
    pub specialized_entries: usize,
    /// Total number of executions across all entries
    pub total_executions: usize,
    /// Cache capacity
    pub cache_capacity: usize,
}

/// Hot path detector that identifies frequently executed code paths.
pub struct HotPathDetector {
    config: JitConfig,
}

impl HotPathDetector {
    /// Create a new hot path detector.
    pub fn new(config: JitConfig) -> Self {
        HotPathDetector { config }
    }

    /// Detect hot paths from cache statistics.
    pub fn detect_hot_paths(&self, cache: &JitCache) -> Vec<JitKey> {
        cache
            .get_hot_paths()
            .into_iter()
            .map(|(key, _)| key)
            .collect()
    }

    /// Recommend recompilation for hot paths.
    pub fn recommend_recompilation(&self, cache: &JitCache) -> Vec<(JitKey, OptimizationLevel)> {
        cache
            .get_hot_paths()
            .into_iter()
            .filter_map(|(key, stats)| {
                // Only recommend recompilation if current optimization is below hot path level
                if stats.optimization_level < self.config.hot_path_optimization {
                    Some((key, self.config.hot_path_optimization))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Recommend deoptimization for cold paths.
    pub fn recommend_deoptimization(&self, cache: &JitCache) -> Vec<JitKey> {
        if !self.config.enable_deoptimization {
            return Vec::new();
        }

        cache
            .get_cold_paths()
            .into_iter()
            .map(|(key, _)| key)
            .collect()
    }
}

/// Adaptive optimizer that progressively optimizes based on runtime profiling.
pub struct AdaptiveOptimizer {
    config: JitConfig,
    hot_path_detector: HotPathDetector,
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer.
    pub fn new(config: JitConfig) -> Self {
        AdaptiveOptimizer {
            hot_path_detector: HotPathDetector::new(config.clone()),
            config,
        }
    }

    /// Analyze runtime behavior and recommend optimizations.
    pub fn analyze_and_recommend(&self, cache: &JitCache) -> AdaptiveOptimizationPlan {
        let hot_paths = self.hot_path_detector.recommend_recompilation(cache);
        let cold_paths = self.hot_path_detector.recommend_deoptimization(cache);

        AdaptiveOptimizationPlan {
            recompile: hot_paths,
            deoptimize: cold_paths,
        }
    }

    /// Apply adaptive optimizations to the cache.
    pub fn optimize(&self, cache: &JitCache) -> Result<usize, ExecutorError> {
        let plan = self.analyze_and_recommend(cache);
        let mut optimized_count = 0;

        // Recompile hot paths with aggressive optimization
        for (key, opt_level) in plan.recompile {
            if let Some(entry) = cache.cache.read().unwrap().get(&key) {
                let graph = &entry.compiled.graph;
                let mut config = entry.compiled.config.clone();
                config.optimization_level = opt_level;

                let mut new_compiler = GraphCompiler::new(config);
                let recompiled = new_compiler.compile(graph)?;

                // Update cache with recompiled version
                cache.cache.write().unwrap().get_mut(&key).unwrap().compiled = recompiled;
                optimized_count += 1;
            }
        }

        // Deoptimize cold paths (remove from cache or downgrade)
        for key in plan.deoptimize {
            cache.cache.write().unwrap().remove(&key);
        }

        Ok(optimized_count)
    }

    /// Get the JIT configuration.
    pub fn config(&self) -> &JitConfig {
        &self.config
    }

    /// Get the hot path detector.
    pub fn hot_path_detector(&self) -> &HotPathDetector {
        &self.hot_path_detector
    }
}

/// Plan for adaptive optimization.
#[derive(Debug, Clone)]
pub struct AdaptiveOptimizationPlan {
    /// Graphs to recompile with higher optimization
    pub recompile: Vec<(JitKey, OptimizationLevel)>,
    /// Graphs to deoptimize (remove or downgrade)
    pub deoptimize: Vec<JitKey>,
}

/// JIT compiler with runtime compilation and adaptive optimization.
pub struct JitCompiler {
    config: JitConfig,
    cache: JitCache,
    adaptive_optimizer: AdaptiveOptimizer,
}

impl JitCompiler {
    /// Create a new JIT compiler.
    pub fn new(config: JitConfig) -> Self {
        JitCompiler {
            cache: JitCache::new(config.clone()),
            adaptive_optimizer: AdaptiveOptimizer::new(config.clone()),
            config,
        }
    }

    /// Create a JIT compiler with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(JitConfig::default())
    }

    /// Compile a graph or retrieve from cache.
    pub fn compile_or_retrieve(
        &mut self,
        graph: &EinsumGraph,
        input_shapes: &[TensorShape],
    ) -> Result<CompiledGraph, ExecutorError> {
        let key = self.create_key(graph, input_shapes);

        // Check cache first
        if let Some(compiled) = self.cache.get(&key) {
            return Ok(compiled);
        }

        // Compile with initial optimization level
        let config = CompilationConfig {
            optimization_level: self.config.initial_optimization,
            enable_shape_inference: true,
            enable_memory_estimation: true,
            enable_caching: true,
            enable_parallelism: true,
            ..Default::default()
        };

        let mut compiler = GraphCompiler::new(config);
        let compiled = compiler.compile(graph)?;

        // Cache the compiled graph
        let is_specialized = self.config.enable_specialization && !input_shapes.is_empty();
        self.cache.insert(key, compiled.clone(), is_specialized);

        Ok(compiled)
    }

    /// Record execution of a compiled graph.
    pub fn record_execution(
        &self,
        graph: &EinsumGraph,
        input_shapes: &[TensorShape],
        duration: Duration,
    ) {
        let key = self.create_key(graph, input_shapes);
        self.cache.record_execution(&key, duration);
    }

    /// Optimize hot paths based on profiling data.
    pub fn optimize_hot_paths(&mut self) -> Result<usize, ExecutorError> {
        if !self.config.enable_adaptive_optimization {
            return Ok(0);
        }

        self.adaptive_optimizer.optimize(&self.cache)
    }

    /// Get JIT cache statistics.
    pub fn cache_stats(&self) -> JitCacheStats {
        self.cache.cache_stats()
    }

    /// Clear the JIT cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Create a cache key for the graph.
    fn create_key(&self, graph: &EinsumGraph, input_shapes: &[TensorShape]) -> JitKey {
        let graph_hash = self.hash_graph(graph);
        let specialization = if self.config.enable_specialization && !input_shapes.is_empty() {
            Some(SpecializationContext::from_shapes(input_shapes))
        } else {
            None
        };

        JitKey {
            graph_hash,
            specialization,
        }
    }

    /// Hash a graph for caching.
    fn hash_graph(&self, graph: &EinsumGraph) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        graph.nodes.len().hash(&mut hasher);
        // Simple hash based on node count and structure
        // In production, would use more sophisticated hashing
        hasher.finish()
    }
}

/// Trait for executors that support JIT compilation.
pub trait TlJitExecutor {
    /// Get the JIT compiler for this executor.
    fn jit_compiler(&mut self) -> &mut JitCompiler;

    /// Enable JIT compilation.
    fn enable_jit(&mut self);

    /// Disable JIT compilation.
    fn disable_jit(&mut self);

    /// Check if JIT is enabled.
    fn is_jit_enabled(&self) -> bool;

    /// Trigger adaptive optimization of hot paths.
    fn optimize_hot_paths(&mut self) -> Result<usize, ExecutorError> {
        self.jit_compiler().optimize_hot_paths()
    }

    /// Get JIT statistics.
    fn jit_stats(&self) -> JitCacheStats;
}

/// Statistics for JIT compilation performance.
#[derive(Debug, Clone)]
pub struct JitStats {
    /// Total number of compilations performed
    pub total_compilations: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Number of recompilations due to hot path optimization
    pub recompilations: usize,
    /// Number of deoptimizations
    pub deoptimizations: usize,
    /// Average compilation time
    pub avg_compilation_time: Duration,
    /// Total time saved by caching
    pub total_time_saved: Duration,
}

impl Default for JitStats {
    fn default() -> Self {
        JitStats {
            total_compilations: 0,
            cache_hits: 0,
            cache_misses: 0,
            recompilations: 0,
            deoptimizations: 0,
            avg_compilation_time: Duration::from_secs(0),
            total_time_saved: Duration::from_secs(0),
        }
    }
}

impl JitStats {
    /// Calculate cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        if self.cache_hits + self.cache_misses == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
    }

    /// Get a summary of JIT statistics.
    pub fn summary(&self) -> String {
        format!(
            "JIT Stats: {} compilations, {:.1}% cache hit rate, {} recompilations, {:.2}ms avg compile time",
            self.total_compilations,
            self.cache_hit_rate() * 100.0,
            self.recompilations,
            self.avg_compilation_time.as_secs_f64() * 1000.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_config_default() {
        let config = JitConfig::default();
        assert_eq!(config.initial_optimization, OptimizationLevel::Basic);
        assert_eq!(config.hot_path_optimization, OptimizationLevel::Aggressive);
        assert_eq!(config.hot_path_threshold, 10);
        assert!(config.enable_specialization);
        assert!(config.enable_adaptive_optimization);
    }

    #[test]
    fn test_specialization_context() {
        let shapes = vec![
            TensorShape::static_shape(vec![2, 3]),
            TensorShape::static_shape(vec![3, 4]),
        ];
        let ctx = SpecializationContext::from_shapes(&shapes);
        assert_eq!(ctx.input_shapes.len(), 2);
        assert_eq!(ctx.input_shapes[0], vec![2, 3]);
        assert_eq!(ctx.input_shapes[1], vec![3, 4]);
    }

    #[test]
    fn test_jit_entry_stats() {
        let mut stats = JitEntryStats::default();
        assert_eq!(stats.execution_count, 0);
        assert!(!stats.is_hot(10));

        // Record executions
        for _ in 0..15 {
            stats.record_execution(Duration::from_millis(10));
        }

        assert_eq!(stats.execution_count, 15);
        assert!(stats.is_hot(10));
        assert_eq!(stats.total_execution_time, Duration::from_millis(150));
    }

    #[test]
    fn test_jit_cache_insert_retrieve() {
        let config = JitConfig::default();
        let cache = JitCache::new(config);

        let graph = EinsumGraph::new();
        let compiled = CompiledGraph {
            graph: graph.clone(),
            schedule: crate::scheduling::ExecutionSchedule {
                execution_order: Vec::new(),
                device_placement: HashMap::new(),
                parallel_groups: Vec::new(),
                estimated_cost: 0.0,
            },
            shapes: HashMap::new(),
            memory_usage: HashMap::new(),
            config: CompilationConfig::default(),
            stats: crate::compilation::CompilationStats::default(),
            compiled_at: std::time::SystemTime::now(),
        };

        let key = JitKey {
            graph_hash: 12345,
            specialization: None,
        };

        cache.insert(key.clone(), compiled.clone(), false);
        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_jit_cache_eviction() {
        let config = JitConfig {
            cache_size: 2, // Small cache for testing
            ..Default::default()
        };
        let cache = JitCache::new(config);

        let graph = EinsumGraph::new();
        let compiled = CompiledGraph {
            graph: graph.clone(),
            schedule: crate::scheduling::ExecutionSchedule {
                execution_order: Vec::new(),
                device_placement: HashMap::new(),
                parallel_groups: Vec::new(),
                estimated_cost: 0.0,
            },
            shapes: HashMap::new(),
            memory_usage: HashMap::new(),
            config: CompilationConfig::default(),
            stats: crate::compilation::CompilationStats::default(),
            compiled_at: std::time::SystemTime::now(),
        };

        // Insert 3 entries (should evict oldest)
        for i in 0..3 {
            let key = JitKey {
                graph_hash: i,
                specialization: None,
            };
            cache.insert(key, compiled.clone(), false);
            std::thread::sleep(Duration::from_millis(10)); // Ensure different timestamps
        }

        let stats = cache.cache_stats();
        assert_eq!(stats.total_entries, 2); // Should only have 2 due to eviction
    }

    #[test]
    fn test_hot_path_detection() {
        let config = JitConfig::default();
        let cache = JitCache::new(config.clone());
        let detector = HotPathDetector::new(config);

        let graph = EinsumGraph::new();
        let compiled = CompiledGraph {
            graph: graph.clone(),
            schedule: crate::scheduling::ExecutionSchedule {
                execution_order: Vec::new(),
                device_placement: HashMap::new(),
                parallel_groups: Vec::new(),
                estimated_cost: 0.0,
            },
            shapes: HashMap::new(),
            memory_usage: HashMap::new(),
            config: CompilationConfig::default(),
            stats: crate::compilation::CompilationStats::default(),
            compiled_at: std::time::SystemTime::now(),
        };

        let key = JitKey {
            graph_hash: 123,
            specialization: None,
        };

        cache.insert(key.clone(), compiled, false);

        // Record many executions to make it hot
        for _ in 0..15 {
            cache.record_execution(&key, Duration::from_millis(10));
        }

        let hot_paths = detector.detect_hot_paths(&cache);
        assert_eq!(hot_paths.len(), 1);
        assert_eq!(hot_paths[0].graph_hash, 123);
    }

    #[test]
    fn test_jit_compiler_basic() {
        let mut jit = JitCompiler::with_default_config();
        let graph = EinsumGraph::new();
        let shapes = vec![];

        let result = jit.compile_or_retrieve(&graph, &shapes);
        assert!(result.is_ok());

        // Second call should hit cache
        let result2 = jit.compile_or_retrieve(&graph, &shapes);
        assert!(result2.is_ok());
    }

    #[test]
    fn test_jit_stats() {
        let stats = JitStats::default();
        assert_eq!(stats.cache_hit_rate(), 0.0);

        let stats = JitStats {
            cache_hits: 8,
            cache_misses: 2,
            ..Default::default()
        };
        assert_eq!(stats.cache_hit_rate(), 0.8);
    }

    #[test]
    fn test_adaptive_optimization_plan() {
        let plan = AdaptiveOptimizationPlan {
            recompile: vec![(
                JitKey {
                    graph_hash: 123,
                    specialization: None,
                },
                OptimizationLevel::Aggressive,
            )],
            deoptimize: vec![],
        };

        assert_eq!(plan.recompile.len(), 1);
        assert_eq!(plan.deoptimize.len(), 0);
    }

    #[test]
    fn test_jit_cache_stats() {
        let config = JitConfig::default();
        let cache = JitCache::new(config);

        let stats = cache.cache_stats();
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.hot_entries, 0);
        assert_eq!(stats.total_executions, 0);
    }

    #[test]
    fn test_specialization_with_device() {
        let shapes = vec![TensorShape::static_shape(vec![2, 3])];
        let ctx = SpecializationContext::from_shapes(&shapes).with_device("cuda:0".to_string());

        assert_eq!(ctx.device, Some("cuda:0".to_string()));
        assert_eq!(ctx.input_shapes[0], vec![2, 3]);
    }

    #[test]
    fn test_jit_entry_cold_detection() {
        let mut stats = JitEntryStats::default();

        // Execute once
        stats.record_execution(Duration::from_millis(10));

        // Not cold immediately
        assert!(!stats.is_cold(5, Duration::from_millis(100)));

        // Wait and check
        std::thread::sleep(Duration::from_millis(150));
        assert!(stats.is_cold(5, Duration::from_millis(100)));
    }

    #[test]
    fn test_jit_cache_clear() {
        let config = JitConfig::default();
        let cache = JitCache::new(config);

        let graph = EinsumGraph::new();
        let compiled = CompiledGraph {
            graph: graph.clone(),
            schedule: crate::scheduling::ExecutionSchedule {
                execution_order: Vec::new(),
                device_placement: HashMap::new(),
                parallel_groups: Vec::new(),
                estimated_cost: 0.0,
            },
            shapes: HashMap::new(),
            memory_usage: HashMap::new(),
            config: CompilationConfig::default(),
            stats: crate::compilation::CompilationStats::default(),
            compiled_at: std::time::SystemTime::now(),
        };

        let key = JitKey {
            graph_hash: 123,
            specialization: None,
        };

        cache.insert(key.clone(), compiled, false);
        assert_eq!(cache.cache_stats().total_entries, 1);

        cache.clear();
        assert_eq!(cache.cache_stats().total_entries, 0);
    }
}
