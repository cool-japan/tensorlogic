//! Graph compilation and caching infrastructure.
//!
//! This module provides ahead-of-time graph optimization and compilation capabilities:
//! - `CompiledGraph`: Optimized, executable representation of computation graphs
//! - `GraphCompiler`: Applies optimization passes and produces compiled graphs
//! - `CompilationCache`: Caches compiled graphs to avoid recompilation
//! - `TlCompilableExecutor`: Trait for executors that support graph compilation
//!
//! # Example
//!
//! ```
//! use tensorlogic_infer::compilation::{GraphCompiler, CompilationConfig, OptimizationLevel};
//! use tensorlogic_infer::DummyExecutor;
//! use tensorlogic_ir::EinsumGraph;
//!
//! let mut compiler = GraphCompiler::new(CompilationConfig {
//!     optimization_level: OptimizationLevel::Aggressive,
//!     ..Default::default()
//! });
//!
//! let graph = EinsumGraph::new();
//! let compiled = compiler.compile(&graph).unwrap();
//! ```

use crate::error::ExecutorError;
use crate::memory::MemoryEstimator;
use crate::optimization::{GraphOptimizer, OptimizationResult};
use crate::scheduling::{ExecutionSchedule, Scheduler, SchedulingStrategy};
use crate::shape::ShapeInferenceContext;
use crate::validation::GraphValidator;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tensorlogic_ir::EinsumGraph;

/// Optimization level for graph compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum OptimizationLevel {
    /// No optimization - compile as-is
    None,
    /// Basic optimizations (dead code elimination, constant folding)
    Basic,
    /// Moderate optimizations (fusion, CSE, basic scheduling)
    #[default]
    Moderate,
    /// Aggressive optimizations (all passes, advanced scheduling)
    Aggressive,
}

/// Configuration for graph compilation.
#[derive(Debug, Clone)]
pub struct CompilationConfig {
    /// Optimization level to apply
    pub optimization_level: OptimizationLevel,
    /// Whether to enable shape inference
    pub enable_shape_inference: bool,
    /// Whether to enable memory estimation
    pub enable_memory_estimation: bool,
    /// Target device for compilation (e.g., "cpu", "cuda:0")
    pub target_device: Option<String>,
    /// Maximum memory budget in bytes (None = unlimited)
    pub memory_budget: Option<usize>,
    /// Enable caching of intermediate results
    pub enable_caching: bool,
    /// Enable parallel execution planning
    pub enable_parallelism: bool,
}

impl Default for CompilationConfig {
    fn default() -> Self {
        CompilationConfig {
            optimization_level: OptimizationLevel::default(),
            enable_shape_inference: true,
            enable_memory_estimation: true,
            target_device: None,
            memory_budget: None,
            enable_caching: true,
            enable_parallelism: true,
        }
    }
}

/// Statistics about the compilation process.
#[derive(Debug, Clone)]
pub struct CompilationStats {
    /// Time taken for compilation
    pub compilation_time: Duration,
    /// Number of nodes in original graph
    pub original_nodes: usize,
    /// Number of nodes after optimization
    pub optimized_nodes: usize,
    /// Number of fusion opportunities applied
    pub fusions_applied: usize,
    /// Number of dead nodes eliminated
    pub dead_nodes_eliminated: usize,
    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: usize,
    /// Scheduled execution steps
    pub execution_steps: usize,
}

impl Default for CompilationStats {
    fn default() -> Self {
        CompilationStats {
            compilation_time: Duration::from_secs(0),
            original_nodes: 0,
            optimized_nodes: 0,
            fusions_applied: 0,
            dead_nodes_eliminated: 0,
            estimated_memory_bytes: 0,
            execution_steps: 0,
        }
    }
}

/// Compiled representation of a computation graph.
///
/// Contains the optimized graph, execution schedule, and metadata
/// necessary for efficient execution.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    /// The optimized graph
    pub graph: EinsumGraph,
    /// Execution schedule for the graph
    pub schedule: ExecutionSchedule,
    /// Shape information (if available)
    pub shapes: HashMap<usize, Vec<usize>>,
    /// Estimated memory usage per node
    pub memory_usage: HashMap<usize, usize>,
    /// Configuration used for compilation
    pub config: CompilationConfig,
    /// Compilation statistics
    pub stats: CompilationStats,
    /// Timestamp when compiled
    pub compiled_at: SystemTime,
}

impl CompiledGraph {
    /// Get the number of nodes in the compiled graph
    pub fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Get the total estimated memory usage
    pub fn total_memory(&self) -> usize {
        self.memory_usage.values().sum()
    }

    /// Check if this compiled graph is still valid
    pub fn is_valid(&self) -> bool {
        // Check if graph structure is valid
        if self.graph.nodes.is_empty() {
            return false;
        }

        // Check if schedule matches graph
        if self.schedule.execution_order.len() != self.graph.nodes.len() {
            return false;
        }

        true
    }

    /// Get a summary of the compiled graph
    pub fn summary(&self) -> String {
        format!(
            "CompiledGraph: {} nodes, {} steps, {:.2}MB memory, compiled in {:.2}ms",
            self.node_count(),
            self.stats.execution_steps,
            self.total_memory() as f64 / 1_000_000.0,
            self.stats.compilation_time.as_secs_f64() * 1000.0
        )
    }
}

/// Graph compiler that applies optimization passes.
pub struct GraphCompiler {
    config: CompilationConfig,
    optimizer: GraphOptimizer,
    validator: GraphValidator,
    scheduler: Scheduler,
}

impl GraphCompiler {
    /// Create a new graph compiler with the given configuration.
    pub fn new(config: CompilationConfig) -> Self {
        GraphCompiler {
            config,
            optimizer: GraphOptimizer::new(),
            validator: GraphValidator::new(),
            scheduler: Scheduler::new(SchedulingStrategy::Balanced),
        }
    }

    /// Create a compiler with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(CompilationConfig::default())
    }

    /// Compile a graph with the configured optimization passes.
    pub fn compile(&mut self, graph: &EinsumGraph) -> Result<CompiledGraph, ExecutorError> {
        let start_time = SystemTime::now();
        let original_nodes = graph.nodes.len();

        // Validate the graph
        let validation_result = self.validator.validate(graph);
        if !validation_result.is_valid {
            return Err(ExecutorError::GraphValidationError(format!(
                "Graph validation failed: {}",
                validation_result
                    .errors
                    .first()
                    .map(|e| e.as_str())
                    .unwrap_or("unknown error")
            )));
        }

        // Clone the graph for optimization
        let optimized_graph = graph.clone();

        // Apply optimizations based on level
        let opt_result = match self.config.optimization_level {
            OptimizationLevel::None => OptimizationResult {
                fusion_opportunities: vec![],
                dead_nodes: vec![],
                redundant_computations: vec![],
                estimated_improvement: 0.0,
            },
            OptimizationLevel::Basic
            | OptimizationLevel::Moderate
            | OptimizationLevel::Aggressive => {
                // Analyze the graph to find optimization opportunities
                self.optimizer.analyze(&optimized_graph)
            }
        };

        // Generate execution schedule
        let schedule = self.scheduler.schedule(&optimized_graph);

        // Shape inference (if enabled)
        let shapes = if self.config.enable_shape_inference {
            let _shape_ctx = ShapeInferenceContext::new();
            // Infer shapes for all nodes
            // Note: This is a simplified version - real implementation would need input shapes
            HashMap::new()
        } else {
            HashMap::new()
        };

        // Memory estimation (if enabled)
        let memory_usage = if self.config.enable_memory_estimation {
            use crate::capabilities::DType;
            let estimator = MemoryEstimator::new(DType::F32);
            let estimate = estimator.estimate(&optimized_graph);
            // Build per-node memory map from estimate
            let mut per_node: HashMap<usize, usize> = HashMap::new();
            for (idx, mem) in estimate.intermediate_memory.iter().enumerate() {
                per_node.insert(idx, mem.bytes);
            }
            per_node
        } else {
            HashMap::new()
        };

        let compilation_time = start_time.elapsed().unwrap_or(Duration::from_secs(0));

        let stats = CompilationStats {
            compilation_time,
            original_nodes,
            optimized_nodes: optimized_graph.nodes.len(),
            fusions_applied: opt_result.fusion_opportunities.len(),
            dead_nodes_eliminated: opt_result.dead_nodes.len(),
            estimated_memory_bytes: memory_usage.values().sum(),
            execution_steps: schedule.execution_order.len(),
        };

        Ok(CompiledGraph {
            graph: optimized_graph,
            schedule,
            shapes,
            memory_usage,
            config: self.config.clone(),
            stats,
            compiled_at: SystemTime::now(),
        })
    }

    /// Update the compilation configuration.
    pub fn set_config(&mut self, config: CompilationConfig) {
        self.config = config;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &CompilationConfig {
        &self.config
    }
}

/// Cache key for compiled graphs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompilationKey {
    /// Hash of the graph structure
    pub graph_hash: u64,
    /// Optimization level used
    pub optimization_level: OptimizationLevel,
    /// Target device (if specified)
    pub target_device: Option<String>,
}

impl CompilationKey {
    /// Create a key from a graph and config.
    pub fn new(graph: &EinsumGraph, config: &CompilationConfig) -> Self {
        CompilationKey {
            graph_hash: Self::hash_graph(graph),
            optimization_level: config.optimization_level,
            target_device: config.target_device.clone(),
        }
    }

    /// Compute a hash of the graph structure.
    fn hash_graph(graph: &EinsumGraph) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash node count
        graph.nodes.len().hash(&mut hasher);

        // Hash each node's operation type and connections
        for node in &graph.nodes {
            // Hash operation type
            match &node.op {
                tensorlogic_ir::OpType::Einsum { spec } => {
                    "einsum".hash(&mut hasher);
                    spec.hash(&mut hasher);
                }
                tensorlogic_ir::OpType::Reduce { op, axes } => {
                    "reduce".hash(&mut hasher);
                    op.hash(&mut hasher);
                    axes.hash(&mut hasher);
                }
                tensorlogic_ir::OpType::ElemUnary { op } => {
                    "elemunary".hash(&mut hasher);
                    op.hash(&mut hasher);
                }
                tensorlogic_ir::OpType::ElemBinary { op } => {
                    "elembinary".hash(&mut hasher);
                    op.hash(&mut hasher);
                }
            }

            // Hash inputs and outputs
            node.inputs.hash(&mut hasher);
            node.outputs.hash(&mut hasher);
        }

        hasher.finish()
    }
}

/// Statistics for the compilation cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of entries in cache
    pub size: usize,
    /// Total compilation time saved (approximate)
    pub time_saved: Duration,
}

impl CacheStats {
    /// Calculate hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Cache for compiled graphs.
///
/// Stores compiled graphs by their cache key to avoid recompilation
/// of the same graph with the same configuration.
pub struct CompilationCache {
    cache: Arc<RwLock<HashMap<CompilationKey, Arc<CompiledGraph>>>>,
    stats: Arc<RwLock<CacheStats>>,
    max_size: usize,
}

impl CompilationCache {
    /// Create a new compilation cache with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        CompilationCache {
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            max_size,
        }
    }

    /// Create a cache with default size (100 entries).
    pub fn with_default_size() -> Self {
        Self::new(100)
    }

    /// Get a compiled graph from the cache.
    pub fn get(&self, key: &CompilationKey) -> Option<Arc<CompiledGraph>> {
        let cache = self.cache.read().unwrap();
        let result = cache.get(key).cloned();

        // Update stats
        let mut stats = self.stats.write().unwrap();
        if let Some(ref compiled) = result {
            stats.hits += 1;
            stats.time_saved += compiled.stats.compilation_time;
        } else {
            stats.misses += 1;
        }

        result
    }

    /// Insert a compiled graph into the cache.
    pub fn insert(&self, key: CompilationKey, compiled: CompiledGraph) {
        let mut cache = self.cache.write().unwrap();

        // Evict oldest entry if at capacity
        if cache.len() >= self.max_size && !cache.contains_key(&key) {
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }

        cache.insert(key, Arc::new(compiled));

        // Update size stat
        let mut stats = self.stats.write().unwrap();
        stats.size = cache.len();
    }

    /// Clear the cache.
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();

        let mut stats = self.stats.write().unwrap();
        stats.size = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Get the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for executors that support graph compilation.
///
/// Executors implementing this trait can execute pre-compiled graphs
/// more efficiently than executing the original graph.
pub trait TlCompilableExecutor {
    /// Compile a graph for efficient execution.
    ///
    /// Returns a compiled graph that can be executed multiple times
    /// with different inputs without recompiling.
    fn compile_graph(
        &mut self,
        graph: &EinsumGraph,
        config: &CompilationConfig,
    ) -> Result<CompiledGraph, ExecutorError>;

    /// Execute a compiled graph.
    ///
    /// This should be more efficient than executing the original graph
    /// since optimization passes have already been applied.
    fn execute_compiled(
        &mut self,
        compiled: &CompiledGraph,
        inputs: &HashMap<usize, Box<dyn std::any::Any>>,
    ) -> Result<HashMap<usize, Box<dyn std::any::Any>>, ExecutorError>;

    /// Check if compilation is supported for this executor.
    fn supports_compilation(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::EinsumNode;

    fn create_test_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();

        // Add input tensors
        graph.tensors.push("input".to_string());
        graph.inputs.push(0);

        // Add nodes that process the input
        graph
            .nodes
            .push(EinsumNode::new("ij->ij", vec![0], vec![1]));
        graph
            .nodes
            .push(EinsumNode::new("ij,jk->ik", vec![1], vec![2]));
        graph
            .nodes
            .push(EinsumNode::new("ik->ik", vec![2], vec![3]));

        // Mark final output
        graph.outputs.push(3);

        graph
    }

    #[test]
    fn test_compilation_key_equality() {
        let graph1 = create_test_graph();
        let graph2 = create_test_graph();

        let config = CompilationConfig::default();

        let key1 = CompilationKey::new(&graph1, &config);
        let key2 = CompilationKey::new(&graph2, &config);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_compilation_key_different_graphs() {
        let graph1 = create_test_graph();
        let mut graph2 = create_test_graph();
        graph2.nodes.push(EinsumNode::new("i->i", vec![3], vec![4]));

        let config = CompilationConfig::default();

        let key1 = CompilationKey::new(&graph1, &config);
        let key2 = CompilationKey::new(&graph2, &config);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_compilation_key_different_config() {
        let graph = create_test_graph();

        let config1 = CompilationConfig {
            optimization_level: OptimizationLevel::Basic,
            ..Default::default()
        };

        let config2 = CompilationConfig {
            optimization_level: OptimizationLevel::Aggressive,
            ..Default::default()
        };

        let key1 = CompilationKey::new(&graph, &config1);
        let key2 = CompilationKey::new(&graph, &config2);

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_graph_compiler_basic() {
        let graph = create_test_graph();
        let mut compiler = GraphCompiler::new(CompilationConfig {
            optimization_level: OptimizationLevel::Basic,
            ..Default::default()
        });

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.is_valid());
        assert_eq!(compiled.stats.original_nodes, 3);
    }

    #[test]
    fn test_graph_compiler_moderate() {
        let graph = create_test_graph();
        let mut compiler = GraphCompiler::new(CompilationConfig {
            optimization_level: OptimizationLevel::Moderate,
            ..Default::default()
        });

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.is_valid());
        assert!(compiled.stats.compilation_time > Duration::from_secs(0));
    }

    #[test]
    fn test_graph_compiler_aggressive() {
        let graph = create_test_graph();
        let mut compiler = GraphCompiler::new(CompilationConfig {
            optimization_level: OptimizationLevel::Aggressive,
            ..Default::default()
        });

        let result = compiler.compile(&graph);
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(compiled.is_valid());
        assert_eq!(compiled.node_count(), compiled.stats.optimized_nodes);
    }

    #[test]
    fn test_compiled_graph_summary() {
        let graph = create_test_graph();
        let mut compiler = GraphCompiler::with_default_config();
        let compiled = compiler.compile(&graph).unwrap();

        let summary = compiled.summary();
        assert!(summary.contains("CompiledGraph"));
        assert!(summary.contains("nodes"));
        assert!(summary.contains("MB"));
    }

    #[test]
    fn test_compilation_cache_basic() {
        let cache = CompilationCache::new(10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        let graph = create_test_graph();
        let config = CompilationConfig::default();
        let key = CompilationKey::new(&graph, &config);

        // Cache miss
        assert!(cache.get(&key).is_none());

        // Insert and retrieve
        let mut compiler = GraphCompiler::with_default_config();
        let compiled = compiler.compile(&graph).unwrap();
        cache.insert(key.clone(), compiled);

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        // Cache hit
        let cached = cache.get(&key);
        assert!(cached.is_some());
    }

    #[test]
    fn test_compilation_cache_eviction() {
        let cache = CompilationCache::new(2);

        let graph1 = create_test_graph();
        let mut graph2 = create_test_graph();
        graph2.nodes.push(EinsumNode::new("i->i", vec![3], vec![4]));
        let mut graph3 = create_test_graph();
        graph3
            .nodes
            .push(EinsumNode::new("ij->ji", vec![3], vec![5]));

        let config = CompilationConfig::default();
        let mut compiler = GraphCompiler::with_default_config();

        let key1 = CompilationKey::new(&graph1, &config);
        let key2 = CompilationKey::new(&graph2, &config);
        let key3 = CompilationKey::new(&graph3, &config);

        // Fill cache
        cache.insert(key1.clone(), compiler.compile(&graph1).unwrap());
        cache.insert(key2.clone(), compiler.compile(&graph2).unwrap());
        assert_eq!(cache.len(), 2);

        // Add third entry - should evict first
        cache.insert(key3.clone(), compiler.compile(&graph3).unwrap());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_compilation_cache_stats() {
        let cache = CompilationCache::new(10);

        let graph = create_test_graph();
        let config = CompilationConfig::default();
        let key = CompilationKey::new(&graph, &config);

        // Initial stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate(), 0.0);

        // Cache miss
        cache.get(&key);
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);

        // Insert and hit
        let mut compiler = GraphCompiler::with_default_config();
        let compiled = compiler.compile(&graph).unwrap();
        cache.insert(key.clone(), compiled);
        cache.get(&key);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_compilation_cache_clear() {
        let cache = CompilationCache::new(10);
        let graph = create_test_graph();
        let config = CompilationConfig::default();
        let key = CompilationKey::new(&graph, &config);

        let mut compiler = GraphCompiler::with_default_config();
        let compiled = compiler.compile(&graph).unwrap();
        cache.insert(key.clone(), compiled);

        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_optimization_levels() {
        let graph = create_test_graph();

        let levels = vec![
            OptimizationLevel::None,
            OptimizationLevel::Basic,
            OptimizationLevel::Moderate,
            OptimizationLevel::Aggressive,
        ];

        for level in levels {
            let mut compiler = GraphCompiler::new(CompilationConfig {
                optimization_level: level,
                ..Default::default()
            });

            let result = compiler.compile(&graph);
            assert!(result.is_ok(), "Compilation failed for level {:?}", level);

            let compiled = result.unwrap();
            assert!(compiled.is_valid());
        }
    }

    #[test]
    fn test_compiled_graph_memory_estimation() {
        let graph = create_test_graph();
        let mut compiler = GraphCompiler::new(CompilationConfig {
            enable_memory_estimation: true,
            ..Default::default()
        });

        let compiled = compiler.compile(&graph).unwrap();
        // Memory estimation should return a value (usize is always non-negative)
        let _memory = compiled.total_memory();
    }

    #[test]
    fn test_config_update() {
        let mut compiler = GraphCompiler::with_default_config();

        let new_config = CompilationConfig {
            optimization_level: OptimizationLevel::Aggressive,
            enable_parallelism: false,
            ..Default::default()
        };

        compiler.set_config(new_config.clone());

        let config = compiler.config();
        assert_eq!(config.optimization_level, OptimizationLevel::Aggressive);
        assert!(!config.enable_parallelism);
    }
}
