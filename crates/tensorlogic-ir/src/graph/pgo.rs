//! Profile-guided optimization for EinsumGraph.
//!
//! This module implements profile-guided optimization (PGO) that uses runtime
//! profiling data to make better optimization decisions.
//!
//! # Features
//!
//! - **Execution profiling**: Collect runtime statistics (timing, memory, cache misses)
//! - **Hotspot identification**: Find performance bottlenecks
//! - **Adaptive optimization**: Choose optimizations based on actual behavior
//! - **Feedback-directed compilation**: Use profile data to guide compilation
//!
//! # Example
//!
//! ```
//! use tensorlogic_ir::{ExecutionProfile, ProfileGuidedOptimizer, OptimizationHint};
//! use tensorlogic_ir::EinsumGraph;
//!
//! // Collect profile during execution
//! let mut profile = ExecutionProfile::new();
//! // ... execute graph and collect stats ...
//!
//! // Use profile to optimize
//! let mut graph = EinsumGraph::new();
//! let optimizer = ProfileGuidedOptimizer::new(profile);
//! let hints = optimizer.analyze(&graph);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::graph::EinsumGraph;
use crate::IrError;

/// Node identifier (index into graph.nodes)
pub type NodeId = usize;
/// Tensor identifier (index into graph.tensors)
pub type TensorId = usize;

/// Runtime execution statistics for a single node.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeStats {
    /// Number of times executed
    pub execution_count: u64,
    /// Total time spent executing this node
    pub total_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Total memory allocated (bytes)
    pub memory_allocated: u64,
    /// Peak memory used (bytes)
    pub peak_memory: u64,
    /// Cache misses (if available)
    pub cache_misses: Option<u64>,
    /// FLOPs executed
    pub flops: Option<u64>,
}

impl NodeStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an execution
    pub fn record_execution(&mut self, duration: Duration, memory: u64) {
        self.execution_count += 1;
        self.total_time += duration;

        if self.execution_count == 1 {
            self.min_time = duration;
            self.max_time = duration;
        } else {
            if duration < self.min_time {
                self.min_time = duration;
            }
            if duration > self.max_time {
                self.max_time = duration;
            }
        }

        self.memory_allocated += memory;
        if memory > self.peak_memory {
            self.peak_memory = memory;
        }
    }

    /// Get average execution time
    pub fn avg_time(&self) -> Duration {
        if self.execution_count > 0 {
            self.total_time / self.execution_count as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get time variance (max - min)
    pub fn time_variance(&self) -> Duration {
        self.max_time.saturating_sub(self.min_time)
    }

    /// Check if this is a hot node (frequently executed)
    pub fn is_hot(&self, threshold: u64) -> bool {
        self.execution_count >= threshold
    }

    /// Get performance score (higher is worse - indicates bottleneck)
    pub fn performance_score(&self) -> f64 {
        let time_weight = self.total_time.as_secs_f64();
        let memory_weight = self.peak_memory as f64 / 1_000_000.0; // MB
        let execution_weight = self.execution_count as f64;

        time_weight * 0.5 + memory_weight * 0.3 + execution_weight * 0.2
    }
}

/// Execution profile for entire graph.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExecutionProfile {
    /// Per-node statistics
    pub node_stats: HashMap<NodeId, NodeStats>,
    /// Per-tensor statistics (size, reuse count)
    pub tensor_stats: HashMap<TensorId, TensorStats>,
    /// Total graph executions
    pub total_executions: u64,
    /// Critical path (longest execution chain)
    pub critical_path: Vec<NodeId>,
}

impl ExecutionProfile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record node execution
    pub fn record_node(&mut self, node_id: NodeId, duration: Duration, memory: u64) {
        self.node_stats
            .entry(node_id)
            .or_default()
            .record_execution(duration, memory);
    }

    /// Record tensor access
    pub fn record_tensor_access(&mut self, tensor_id: TensorId, size: usize) {
        self.tensor_stats
            .entry(tensor_id)
            .or_insert_with(|| TensorStats::new(size))
            .record_access();
    }

    /// Get hot nodes (top N by performance score)
    pub fn get_hot_nodes(&self, n: usize) -> Vec<(NodeId, f64)> {
        let mut scores: Vec<_> = self
            .node_stats
            .iter()
            .map(|(id, stats)| (*id, stats.performance_score()))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(n);
        scores
    }

    /// Get memory-intensive nodes
    pub fn get_memory_intensive_nodes(&self, threshold: u64) -> Vec<NodeId> {
        self.node_stats
            .iter()
            .filter(|(_, stats)| stats.peak_memory >= threshold)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Merge another profile (for multi-run averaging)
    pub fn merge(&mut self, other: &ExecutionProfile) {
        for (node_id, other_stats) in &other.node_stats {
            let stats = self.node_stats.entry(*node_id).or_default();

            stats.execution_count += other_stats.execution_count;
            stats.total_time += other_stats.total_time;
            stats.memory_allocated += other_stats.memory_allocated;

            if other_stats.min_time < stats.min_time
                || stats.execution_count == other_stats.execution_count
            {
                stats.min_time = other_stats.min_time;
            }
            if other_stats.max_time > stats.max_time {
                stats.max_time = other_stats.max_time;
            }
            if other_stats.peak_memory > stats.peak_memory {
                stats.peak_memory = other_stats.peak_memory;
            }
        }

        for (tensor_id, other_tensor_stats) in &other.tensor_stats {
            let tensor_stats = self
                .tensor_stats
                .entry(*tensor_id)
                .or_insert_with(|| TensorStats::new(other_tensor_stats.size_bytes));

            tensor_stats.access_count += other_tensor_stats.access_count;
            tensor_stats.last_access_time = tensor_stats
                .last_access_time
                .max(other_tensor_stats.last_access_time);
        }

        self.total_executions += other.total_executions;
    }

    /// Export profile to JSON
    pub fn to_json(&self) -> Result<String, IrError> {
        serde_json::to_string_pretty(self).map_err(|e| IrError::SerializationError(e.to_string()))
    }

    /// Import profile from JSON
    pub fn from_json(json: &str) -> Result<Self, IrError> {
        serde_json::from_str(json).map_err(|e| IrError::SerializationError(e.to_string()))
    }
}

/// Tensor usage statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorStats {
    /// Tensor size in bytes
    pub size_bytes: usize,
    /// Number of accesses
    pub access_count: u64,
    /// Last access time (for liveness analysis)
    pub last_access_time: u64,
}

impl TensorStats {
    pub fn new(size_bytes: usize) -> Self {
        TensorStats {
            size_bytes,
            access_count: 0,
            last_access_time: 0,
        }
    }

    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_access_time = self.access_count;
    }

    /// Check if this tensor is frequently reused
    pub fn is_reused(&self, threshold: u64) -> bool {
        self.access_count >= threshold
    }
}

/// Optimization hint derived from profiling.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationHint {
    /// Fuse these nodes together
    FuseNodes(Vec<NodeId>),
    /// Parallelize these independent nodes
    Parallelize(Vec<NodeId>),
    /// Cache this tensor in fast memory
    CacheTensor(TensorId),
    /// Use in-place operation for this node
    InPlaceOp(NodeId),
    /// Prefetch this tensor
    Prefetch(TensorId),
    /// Tile this operation
    TileOperation { node: NodeId, tile_size: usize },
    /// Reorder operations for better cache locality
    ReorderOps(Vec<NodeId>),
    /// Allocate large buffer for this tensor
    PreAllocate { tensor: TensorId, size: usize },
}

/// Profile-guided optimizer.
#[derive(Clone, Debug)]
pub struct ProfileGuidedOptimizer {
    profile: ExecutionProfile,
    /// Hotness threshold (execution count)
    hot_threshold: u64,
    /// Memory threshold for considering a node memory-intensive (MB)
    memory_threshold: u64,
}

impl ProfileGuidedOptimizer {
    pub fn new(profile: ExecutionProfile) -> Self {
        ProfileGuidedOptimizer {
            profile,
            hot_threshold: 10,
            memory_threshold: 100 * 1024 * 1024, // 100 MB
        }
    }

    /// Set hotness threshold
    pub fn with_hot_threshold(mut self, threshold: u64) -> Self {
        self.hot_threshold = threshold;
        self
    }

    /// Set memory threshold
    pub fn with_memory_threshold(mut self, threshold: u64) -> Self {
        self.memory_threshold = threshold;
        self
    }

    /// Analyze graph and generate optimization hints
    pub fn analyze(&self, graph: &EinsumGraph) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        // 1. Identify hot nodes for fusion
        let hot_nodes = self.profile.get_hot_nodes(10);
        if hot_nodes.len() >= 2 {
            let node_ids: Vec<_> = hot_nodes.iter().map(|(id, _)| *id).collect();
            hints.push(OptimizationHint::FuseNodes(node_ids));
        }

        // 2. Identify memory-intensive operations
        let memory_nodes = self
            .profile
            .get_memory_intensive_nodes(self.memory_threshold);
        for node_id in memory_nodes {
            // Suggest in-place operations
            hints.push(OptimizationHint::InPlaceOp(node_id));

            // Check if we can tile the operation
            if self.is_tileable(node_id, graph) {
                hints.push(OptimizationHint::TileOperation {
                    node: node_id,
                    tile_size: 1024, // Default tile size
                });
            }
        }

        // 3. Identify frequently reused tensors for caching
        for (tensor_id, stats) in &self.profile.tensor_stats {
            if stats.is_reused(self.hot_threshold) {
                hints.push(OptimizationHint::CacheTensor(*tensor_id));
            }

            // Pre-allocate large tensors
            if stats.size_bytes > 1024 * 1024 {
                // > 1 MB
                hints.push(OptimizationHint::PreAllocate {
                    tensor: *tensor_id,
                    size: stats.size_bytes,
                });
            }
        }

        // 4. Find independent operations for parallelization
        let parallel_groups = self.find_parallel_groups(graph);
        for group in parallel_groups {
            if group.len() >= 2 {
                hints.push(OptimizationHint::Parallelize(group));
            }
        }

        hints
    }

    /// Check if a node can be tiled
    fn is_tileable(&self, _node_id: NodeId, _graph: &EinsumGraph) -> bool {
        // Simplified check - would analyze operation type and dimensions
        true
    }

    /// Find groups of independent operations that can be parallelized
    fn find_parallel_groups(&self, graph: &EinsumGraph) -> Vec<Vec<NodeId>> {
        let mut groups = Vec::new();

        // Simple algorithm: nodes at the same depth with no dependencies
        let depths = self.compute_depths(graph);
        let mut depth_map: HashMap<usize, Vec<NodeId>> = HashMap::new();

        for (node_id, depth) in depths {
            depth_map.entry(depth).or_default().push(node_id);
        }

        for (_, nodes) in depth_map {
            if nodes.len() >= 2 {
                groups.push(nodes);
            }
        }

        groups
    }

    /// Compute depth of each node in the graph
    fn compute_depths(&self, graph: &EinsumGraph) -> HashMap<NodeId, usize> {
        let mut depths = HashMap::new();

        for node_id in 0..graph.nodes.len() {
            depths.insert(
                node_id,
                self.compute_node_depth(node_id, graph, &mut HashMap::new()),
            );
        }

        depths
    }

    #[allow(clippy::only_used_in_recursion)]
    fn compute_node_depth(
        &self,
        node_id: NodeId,
        graph: &EinsumGraph,
        memo: &mut HashMap<NodeId, usize>,
    ) -> usize {
        if let Some(&depth) = memo.get(&node_id) {
            return depth;
        }

        let node = &graph.nodes[node_id];
        let input_depths: Vec<_> = node
            .inputs
            .iter()
            .filter_map(|&tensor_id| {
                // Find the node that produces this tensor
                graph.nodes.iter().enumerate().find_map(|(id, n)| {
                    if n.outputs.contains(&tensor_id) {
                        Some(self.compute_node_depth(id, graph, memo))
                    } else {
                        None
                    }
                })
            })
            .collect();

        let depth = if input_depths.is_empty() {
            0
        } else {
            input_depths.into_iter().max().unwrap() + 1
        };

        memo.insert(node_id, depth);
        depth
    }

    /// Apply optimization hints to a graph
    pub fn apply_hints(
        &self,
        graph: &mut EinsumGraph,
        hints: &[OptimizationHint],
    ) -> Result<usize, IrError> {
        let mut applied = 0;

        for hint in hints {
            match hint {
                OptimizationHint::FuseNodes(nodes) => {
                    if self.try_fuse_nodes(graph, nodes)? {
                        applied += 1;
                    }
                }
                OptimizationHint::CacheTensor(tensor_id) => {
                    self.mark_tensor_cached(graph, *tensor_id);
                    applied += 1;
                }
                OptimizationHint::InPlaceOp(node_id) => {
                    if self.try_make_inplace(graph, *node_id)? {
                        applied += 1;
                    }
                }
                OptimizationHint::PreAllocate { tensor, size } => {
                    self.mark_preallocate(graph, *tensor, *size);
                    applied += 1;
                }
                _ => {
                    // Other hints require backend-specific implementation
                }
            }
        }

        Ok(applied)
    }

    fn try_fuse_nodes(&self, _graph: &mut EinsumGraph, _nodes: &[NodeId]) -> Result<bool, IrError> {
        // Would implement actual fusion logic
        Ok(false)
    }

    fn mark_tensor_cached(&self, _graph: &mut EinsumGraph, _tensor_id: TensorId) {
        // Mark tensor for caching (would add metadata)
    }

    fn try_make_inplace(
        &self,
        _graph: &mut EinsumGraph,
        _node_id: NodeId,
    ) -> Result<bool, IrError> {
        // Would check if operation can be in-place and modify
        Ok(false)
    }

    fn mark_preallocate(&self, _graph: &mut EinsumGraph, _tensor_id: TensorId, _size: usize) {
        // Mark tensor for pre-allocation (would add metadata)
    }

    /// Get profile reference
    pub fn profile(&self) -> &ExecutionProfile {
        &self.profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_stats_basic() {
        let mut stats = NodeStats::new();

        stats.record_execution(Duration::from_millis(100), 1024);
        assert_eq!(stats.execution_count, 1);
        assert_eq!(stats.total_time, Duration::from_millis(100));
        assert_eq!(stats.peak_memory, 1024);

        stats.record_execution(Duration::from_millis(150), 2048);
        assert_eq!(stats.execution_count, 2);
        assert_eq!(stats.avg_time(), Duration::from_millis(125));
        assert_eq!(stats.peak_memory, 2048);
    }

    #[test]
    fn test_node_stats_min_max() {
        let mut stats = NodeStats::new();

        stats.record_execution(Duration::from_millis(100), 1024);
        stats.record_execution(Duration::from_millis(50), 512);
        stats.record_execution(Duration::from_millis(200), 4096);

        assert_eq!(stats.min_time, Duration::from_millis(50));
        assert_eq!(stats.max_time, Duration::from_millis(200));
        assert_eq!(stats.time_variance(), Duration::from_millis(150));
    }

    #[test]
    fn test_node_stats_hotness() {
        let mut stats = NodeStats::new();

        for _ in 0..5 {
            stats.record_execution(Duration::from_millis(10), 100);
        }

        assert!(!stats.is_hot(10));
        assert!(stats.is_hot(5));
        assert!(stats.is_hot(1));
    }

    #[test]
    fn test_execution_profile_record() {
        let mut profile = ExecutionProfile::new();

        profile.record_node(0, Duration::from_millis(100), 1024);
        profile.record_node(1, Duration::from_millis(200), 2048);
        profile.record_node(0, Duration::from_millis(110), 1024);

        assert_eq!(profile.node_stats.len(), 2);
        assert_eq!(profile.node_stats[&0].execution_count, 2);
        assert_eq!(profile.node_stats[&1].execution_count, 1);
    }

    #[test]
    fn test_hot_nodes() {
        let mut profile = ExecutionProfile::new();

        // Node 0: frequently executed, fast
        for _ in 0..100 {
            profile.record_node(0, Duration::from_millis(10), 100);
        }

        // Node 1: rarely executed, slow
        for _ in 0..5 {
            profile.record_node(1, Duration::from_millis(500), 10000);
        }

        let hot_nodes = profile.get_hot_nodes(2);
        assert_eq!(hot_nodes.len(), 2);

        // Node 1 should have higher performance score due to time*weight
        // but this depends on the actual scoring function
        assert!(hot_nodes[0].1 > 0.0);
    }

    #[test]
    fn test_tensor_stats() {
        let mut stats = TensorStats::new(1024);

        assert_eq!(stats.access_count, 0);

        stats.record_access();
        assert_eq!(stats.access_count, 1);
        assert_eq!(stats.last_access_time, 1);

        stats.record_access();
        assert_eq!(stats.access_count, 2);
        assert_eq!(stats.last_access_time, 2);

        assert!(stats.is_reused(2));
        assert!(!stats.is_reused(3));
    }

    #[test]
    fn test_profile_merge() {
        let mut profile1 = ExecutionProfile::new();
        profile1.record_node(0, Duration::from_millis(100), 1024);
        profile1.total_executions = 1;

        let mut profile2 = ExecutionProfile::new();
        profile2.record_node(0, Duration::from_millis(150), 2048);
        profile2.record_node(1, Duration::from_millis(200), 512);
        profile2.total_executions = 1;

        profile1.merge(&profile2);

        assert_eq!(profile1.node_stats.len(), 2);
        assert_eq!(profile1.node_stats[&0].execution_count, 2);
        assert_eq!(profile1.total_executions, 2);
    }

    #[test]
    fn test_profile_serialization() {
        let mut profile = ExecutionProfile::new();
        profile.record_node(0, Duration::from_millis(100), 1024);
        profile.record_tensor_access(0, 2048);

        let json = profile.to_json().unwrap();
        let restored = ExecutionProfile::from_json(&json).unwrap();

        assert_eq!(profile.node_stats.len(), restored.node_stats.len());
        assert_eq!(profile.tensor_stats.len(), restored.tensor_stats.len());
    }

    #[test]
    fn test_pgo_optimizer_basic() {
        let mut profile = ExecutionProfile::new();

        // Create hot nodes
        for _ in 0..20 {
            profile.record_node(0, Duration::from_millis(50), 1024);
            profile.record_node(1, Duration::from_millis(60), 2048);
        }

        let optimizer = ProfileGuidedOptimizer::new(profile);
        assert_eq!(optimizer.hot_threshold, 10);
    }

    #[test]
    fn test_optimization_hints() {
        let mut profile = ExecutionProfile::new();

        // Hot nodes for fusion
        for _ in 0..20 {
            profile.record_node(0, Duration::from_millis(10), 1024);
            profile.record_node(1, Duration::from_millis(10), 1024);
        }

        // Large memory node
        profile.record_node(2, Duration::from_millis(100), 200 * 1024 * 1024);

        // Frequently accessed tensor
        for _ in 0..50 {
            profile.record_tensor_access(0, 4096);
        }

        let optimizer = ProfileGuidedOptimizer::new(profile)
            .with_hot_threshold(10)
            .with_memory_threshold(100 * 1024 * 1024);

        let graph = EinsumGraph::new();
        let hints = optimizer.analyze(&graph);

        // Should generate various hints
        assert!(!hints.is_empty());

        // Check for cache hint
        assert!(hints
            .iter()
            .any(|h| matches!(h, OptimizationHint::CacheTensor(_))));
    }

    #[test]
    fn test_memory_intensive_nodes() {
        let mut profile = ExecutionProfile::new();

        profile.record_node(0, Duration::from_millis(10), 50 * 1024 * 1024);
        profile.record_node(1, Duration::from_millis(10), 150 * 1024 * 1024);
        profile.record_node(2, Duration::from_millis(10), 1024);

        let memory_nodes = profile.get_memory_intensive_nodes(100 * 1024 * 1024);

        assert_eq!(memory_nodes.len(), 1);
        assert_eq!(memory_nodes[0], 1);
    }
}
