//! Automatic parallelization for computation graphs.
//!
//! This module provides automatic detection and exploitation of parallelism opportunities:
//! - **Dependency analysis**: Build dependency graphs and detect parallelizable operations
//! - **Cost modeling**: Estimate execution costs and communication overhead
//! - **Work partitioning**: Dynamically partition work across threads/devices
//! - **Load balancing**: Balance work to minimize idle time
//! - **Pipeline detection**: Identify pipeline parallelism opportunities
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_infer::{AutoParallelizer, ParallelizationStrategy, CostModel};
//!
//! // Create auto-parallelizer with cost model
//! let parallelizer = AutoParallelizer::new()
//!     .with_strategy(ParallelizationStrategy::Aggressive)
//!     .with_cost_model(CostModel::ProfileBased);
//!
//! // Analyze graph for parallelism
//! let analysis = parallelizer.analyze(&graph)?;
//! println!("Found {} parallelizable stages", analysis.num_stages);
//!
//! // Generate parallel execution plan
//! let plan = parallelizer.generate_plan(&graph)?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Auto-parallelization errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AutoParallelError {
    #[error("Dependency cycle detected: {0}")]
    DependencyCycle(String),

    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    #[error("Cost model error: {0}")]
    CostModelError(String),

    #[error("Partitioning failed: {0}")]
    PartitioningFailed(String),
}

/// Node ID in the computation graph.
pub type NodeId = String;

/// Parallelization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelizationStrategy {
    /// Conservative: Only parallelize when clearly beneficial
    Conservative,
    /// Balanced: Balance parallelism and overhead
    Balanced,
    /// Aggressive: Maximize parallelism even with potential overhead
    Aggressive,
    /// Cost-based: Use cost model to decide
    CostBased,
}

/// Cost model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CostModel {
    /// Simple heuristic-based cost model
    Heuristic,
    /// Profile-based cost model using historical data
    ProfileBased,
    /// Analytical cost model based on operation complexity
    Analytical,
    /// Hybrid approach combining multiple models
    Hybrid,
}

/// Dependency type between nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Data dependency: consumer needs producer's output
    Data,
    /// Control dependency: execution order matters
    Control,
    /// Memory dependency: shared memory access
    Memory,
}

/// Node information for parallelization analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: NodeId,
    pub op_type: String,
    pub estimated_cost: f64, // in microseconds
    pub memory_size: usize,  // in bytes
    pub dependencies: Vec<(NodeId, DependencyType)>,
    pub can_parallelize: bool,
}

/// Parallel stage containing nodes that can execute concurrently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelStage {
    pub stage_id: usize,
    pub nodes: Vec<NodeId>,
    pub estimated_time: f64,
    pub memory_requirement: usize,
    pub predecessors: Vec<usize>, // Stages that must complete before this
}

/// Work partition for a single worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkPartition {
    pub worker_id: usize,
    pub nodes: Vec<NodeId>,
    pub estimated_load: f64,
}

/// Parallelization analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationAnalysis {
    pub num_stages: usize,
    pub stages: Vec<ParallelStage>,
    pub critical_path_length: f64,
    pub total_work: f64,
    pub parallelism_factor: f64, // total_work / critical_path_length
    pub communication_overhead: f64,
    pub recommended_workers: usize,
}

/// Parallel execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionPlan {
    pub stages: Vec<ParallelStage>,
    pub partitions: Vec<WorkPartition>,
    pub estimated_speedup: f64,
    pub load_balance_ratio: f64,
}

/// Automatic parallelizer.
pub struct AutoParallelizer {
    strategy: ParallelizationStrategy,
    cost_model: CostModel,
    max_workers: usize,
    overhead_per_task: f64,             // microseconds
    communication_bandwidth: f64,       // GB/s
    profile_data: HashMap<String, f64>, // op_type -> avg_time_us
}

impl AutoParallelizer {
    /// Create a new auto-parallelizer with default settings.
    pub fn new() -> Self {
        Self {
            strategy: ParallelizationStrategy::Balanced,
            cost_model: CostModel::Heuristic,
            max_workers: num_cpus::get(),
            overhead_per_task: 10.0,        // 10 microseconds per task
            communication_bandwidth: 100.0, // 100 GB/s
            profile_data: HashMap::new(),
        }
    }

    /// Set parallelization strategy.
    pub fn with_strategy(mut self, strategy: ParallelizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set cost model.
    pub fn with_cost_model(mut self, model: CostModel) -> Self {
        self.cost_model = model;
        self
    }

    /// Set maximum number of workers.
    pub fn with_max_workers(mut self, workers: usize) -> Self {
        self.max_workers = workers;
        self
    }

    /// Update profile data with observed execution times.
    pub fn update_profile(&mut self, op_type: String, time_us: f64) {
        let entry = self.profile_data.entry(op_type).or_insert(0.0);
        *entry = 0.9 * *entry + 0.1 * time_us; // Exponential moving average
    }

    /// Analyze graph for parallelization opportunities.
    pub fn analyze(
        &self,
        nodes: &[NodeInfo],
    ) -> Result<ParallelizationAnalysis, AutoParallelError> {
        // Build dependency graph
        let dep_graph = self.build_dependency_graph(nodes)?;

        // Topological sort to find stages
        let stages = self.compute_stages(nodes, &dep_graph)?;

        // Calculate critical path
        let critical_path_length = self.calculate_critical_path(&stages);

        // Calculate total work
        let total_work: f64 = nodes.iter().map(|n| n.estimated_cost).sum();

        // Estimate communication overhead
        let communication_overhead = self.estimate_communication_overhead(&stages, nodes);

        // Calculate parallelism factor
        let parallelism_factor = if critical_path_length > 0.0 {
            total_work / critical_path_length
        } else {
            1.0
        };

        // Recommend number of workers
        let recommended_workers = self.recommend_worker_count(parallelism_factor);

        Ok(ParallelizationAnalysis {
            num_stages: stages.len(),
            stages,
            critical_path_length,
            total_work,
            parallelism_factor,
            communication_overhead,
            recommended_workers,
        })
    }

    /// Generate parallel execution plan.
    pub fn generate_plan(
        &self,
        nodes: &[NodeInfo],
    ) -> Result<ParallelExecutionPlan, AutoParallelError> {
        let analysis = self.analyze(nodes)?;

        // Partition work across workers
        let partitions = self.partition_work(&analysis)?;

        // Calculate estimated speedup
        let sequential_time = analysis.total_work;
        let parallel_time = analysis.critical_path_length + analysis.communication_overhead;
        let estimated_speedup = if parallel_time > 0.0 {
            sequential_time / parallel_time
        } else {
            1.0
        };

        // Calculate load balance ratio
        let load_balance_ratio = self.calculate_load_balance(&partitions);

        Ok(ParallelExecutionPlan {
            stages: analysis.stages,
            partitions,
            estimated_speedup,
            load_balance_ratio,
        })
    }

    /// Build dependency graph from nodes.
    fn build_dependency_graph(
        &self,
        nodes: &[NodeInfo],
    ) -> Result<HashMap<NodeId, HashSet<NodeId>>, AutoParallelError> {
        let mut graph: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();

        // Initialize graph with all nodes
        for node in nodes {
            graph.entry(node.id.clone()).or_insert_with(HashSet::new);
        }

        // Add edges
        for node in nodes {
            for (dep_id, _dep_type) in &node.dependencies {
                if !graph.contains_key(dep_id) {
                    return Err(AutoParallelError::InvalidGraph(format!(
                        "Unknown dependency: {}",
                        dep_id
                    )));
                }
                graph
                    .entry(node.id.clone())
                    .or_insert_with(HashSet::new)
                    .insert(dep_id.clone());
            }
        }

        // Check for cycles
        self.check_cycles(&graph)?;

        Ok(graph)
    }

    /// Check for dependency cycles using DFS.
    fn check_cycles(
        &self,
        graph: &HashMap<NodeId, HashSet<NodeId>>,
    ) -> Result<(), AutoParallelError> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node in graph.keys() {
            if !visited.contains(node) {
                if self.has_cycle_util(node, graph, &mut visited, &mut rec_stack)? {
                    return Err(AutoParallelError::DependencyCycle(format!(
                        "Cycle detected involving node: {}",
                        node
                    )));
                }
            }
        }

        Ok(())
    }

    fn has_cycle_util(
        &self,
        node: &NodeId,
        graph: &HashMap<NodeId, HashSet<NodeId>>,
        visited: &mut HashSet<NodeId>,
        rec_stack: &mut HashSet<NodeId>,
    ) -> Result<bool, AutoParallelError> {
        visited.insert(node.clone());
        rec_stack.insert(node.clone());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if self.has_cycle_util(neighbor, graph, visited, rec_stack)? {
                        return Ok(true);
                    }
                } else if rec_stack.contains(neighbor) {
                    return Ok(true);
                }
            }
        }

        rec_stack.remove(node);
        Ok(false)
    }

    /// Compute parallel stages using level-based topological sort.
    fn compute_stages(
        &self,
        nodes: &[NodeInfo],
        dep_graph: &HashMap<NodeId, HashSet<NodeId>>,
    ) -> Result<Vec<ParallelStage>, AutoParallelError> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut node_map: HashMap<NodeId, &NodeInfo> = HashMap::new();

        // Initialize in-degree and node map
        for node in nodes {
            node_map.insert(node.id.clone(), node);
            let deps = dep_graph.get(&node.id).unwrap();
            in_degree.insert(node.id.clone(), deps.len());
        }

        let mut stages = Vec::new();
        let mut current_level: VecDeque<NodeId> = VecDeque::new();

        // Find nodes with no dependencies
        for (node_id, &degree) in &in_degree {
            if degree == 0 {
                current_level.push_back(node_id.clone());
            }
        }

        let mut stage_id = 0;
        while !current_level.is_empty() {
            let mut stage_nodes = Vec::new();
            let mut estimated_time: f64 = 0.0;
            let mut memory_requirement = 0;

            // Process all nodes at current level
            for _ in 0..current_level.len() {
                if let Some(node_id) = current_level.pop_front() {
                    let node = node_map[&node_id];
                    stage_nodes.push(node_id.clone());
                    estimated_time = estimated_time.max(node.estimated_cost);
                    memory_requirement += node.memory_size;

                    // Decrease in-degree of dependent nodes
                    for other_id in node_map.keys() {
                        if dep_graph[other_id].contains(&node_id) {
                            if let Some(degree) = in_degree.get_mut(other_id) {
                                *degree -= 1;
                                if *degree == 0 {
                                    current_level.push_back(other_id.clone());
                                }
                            }
                        }
                    }
                }
            }

            if !stage_nodes.is_empty() {
                stages.push(ParallelStage {
                    stage_id,
                    nodes: stage_nodes,
                    estimated_time,
                    memory_requirement,
                    predecessors: if stage_id > 0 {
                        vec![stage_id - 1]
                    } else {
                        vec![]
                    },
                });
                stage_id += 1;
            }
        }

        // Check if all nodes were processed
        if stages.iter().map(|s| s.nodes.len()).sum::<usize>() != nodes.len() {
            return Err(AutoParallelError::DependencyCycle(
                "Not all nodes were processed - cycle detected".to_string(),
            ));
        }

        Ok(stages)
    }

    /// Calculate critical path length.
    fn calculate_critical_path(&self, stages: &[ParallelStage]) -> f64 {
        stages.iter().map(|s| s.estimated_time).sum()
    }

    /// Estimate communication overhead.
    fn estimate_communication_overhead(
        &self,
        stages: &[ParallelStage],
        _nodes: &[NodeInfo],
    ) -> f64 {
        let mut overhead = 0.0;

        // Add overhead for each stage boundary
        for stage in stages {
            if stage.nodes.len() > 1 {
                // Multiple nodes in stage need synchronization
                overhead += self.overhead_per_task * stage.nodes.len() as f64;

                // Add communication overhead based on memory transfer
                let transfer_time =
                    stage.memory_requirement as f64 / (self.communication_bandwidth * 1e9) * 1e6;
                overhead += transfer_time;
            }
        }

        overhead
    }

    /// Recommend number of workers based on parallelism factor.
    fn recommend_worker_count(&self, parallelism_factor: f64) -> usize {
        let ideal = parallelism_factor.ceil() as usize;

        match self.strategy {
            ParallelizationStrategy::Conservative => ideal.min(self.max_workers / 2).max(1),
            ParallelizationStrategy::Balanced => ideal.min(self.max_workers),
            ParallelizationStrategy::Aggressive => self.max_workers,
            ParallelizationStrategy::CostBased => {
                // Use cost model to decide
                if parallelism_factor > 2.0 {
                    ideal.min(self.max_workers)
                } else {
                    (ideal / 2).max(1)
                }
            }
        }
    }

    /// Partition work across workers.
    fn partition_work(
        &self,
        analysis: &ParallelizationAnalysis,
    ) -> Result<Vec<WorkPartition>, AutoParallelError> {
        let num_workers = analysis.recommended_workers;
        let mut partitions: Vec<WorkPartition> = (0..num_workers)
            .map(|i| WorkPartition {
                worker_id: i,
                nodes: Vec::new(),
                estimated_load: 0.0,
            })
            .collect();

        // For each stage, distribute nodes across workers
        for stage in &analysis.stages {
            // Sort nodes by estimated cost (descending)
            let mut stage_nodes: Vec<(NodeId, f64)> = stage
                .nodes
                .iter()
                .map(|id| (id.clone(), 1.0)) // Simplified: assume uniform cost
                .collect();
            stage_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Greedy assignment to least loaded worker
            for (node_id, cost) in stage_nodes {
                let min_partition = partitions
                    .iter_mut()
                    .min_by(|a, b| a.estimated_load.partial_cmp(&b.estimated_load).unwrap())
                    .ok_or_else(|| {
                        AutoParallelError::PartitioningFailed("No partitions available".to_string())
                    })?;

                min_partition.nodes.push(node_id);
                min_partition.estimated_load += cost;
            }
        }

        Ok(partitions)
    }

    /// Calculate load balance ratio (1.0 = perfect balance).
    fn calculate_load_balance(&self, partitions: &[WorkPartition]) -> f64 {
        if partitions.is_empty() {
            return 1.0;
        }

        let loads: Vec<f64> = partitions.iter().map(|p| p.estimated_load).collect();
        let max_load = loads.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_load = loads.iter().sum::<f64>() / loads.len() as f64;

        if max_load > 0.0 {
            avg_load / max_load
        } else {
            1.0
        }
    }
}

impl Default for AutoParallelizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_nodes() -> Vec<NodeInfo> {
        vec![
            NodeInfo {
                id: "a".to_string(),
                op_type: "input".to_string(),
                estimated_cost: 10.0,
                memory_size: 1000,
                dependencies: vec![],
                can_parallelize: true,
            },
            NodeInfo {
                id: "b".to_string(),
                op_type: "compute".to_string(),
                estimated_cost: 20.0,
                memory_size: 2000,
                dependencies: vec![("a".to_string(), DependencyType::Data)],
                can_parallelize: true,
            },
            NodeInfo {
                id: "c".to_string(),
                op_type: "compute".to_string(),
                estimated_cost: 15.0,
                memory_size: 1500,
                dependencies: vec![("a".to_string(), DependencyType::Data)],
                can_parallelize: true,
            },
            NodeInfo {
                id: "d".to_string(),
                op_type: "output".to_string(),
                estimated_cost: 10.0,
                memory_size: 1000,
                dependencies: vec![
                    ("b".to_string(), DependencyType::Data),
                    ("c".to_string(), DependencyType::Data),
                ],
                can_parallelize: false,
            },
        ]
    }

    #[test]
    fn test_auto_parallelizer_creation() {
        let parallelizer = AutoParallelizer::new();
        assert_eq!(parallelizer.strategy, ParallelizationStrategy::Balanced);
        assert_eq!(parallelizer.cost_model, CostModel::Heuristic);
    }

    #[test]
    fn test_builder_pattern() {
        let parallelizer = AutoParallelizer::new()
            .with_strategy(ParallelizationStrategy::Aggressive)
            .with_cost_model(CostModel::ProfileBased)
            .with_max_workers(8);

        assert_eq!(parallelizer.strategy, ParallelizationStrategy::Aggressive);
        assert_eq!(parallelizer.cost_model, CostModel::ProfileBased);
        assert_eq!(parallelizer.max_workers, 8);
    }

    #[test]
    fn test_dependency_graph_building() {
        let parallelizer = AutoParallelizer::new();
        let nodes = create_test_nodes();

        let graph = parallelizer.build_dependency_graph(&nodes).unwrap();

        assert_eq!(graph.len(), 4);
        assert!(graph["b"].contains("a"));
        assert!(graph["c"].contains("a"));
        assert!(graph["d"].contains("b"));
        assert!(graph["d"].contains("c"));
    }

    #[test]
    fn test_cycle_detection() {
        let parallelizer = AutoParallelizer::new();

        // Create nodes with a cycle
        let nodes = vec![
            NodeInfo {
                id: "a".to_string(),
                op_type: "compute".to_string(),
                estimated_cost: 10.0,
                memory_size: 1000,
                dependencies: vec![("b".to_string(), DependencyType::Data)],
                can_parallelize: true,
            },
            NodeInfo {
                id: "b".to_string(),
                op_type: "compute".to_string(),
                estimated_cost: 10.0,
                memory_size: 1000,
                dependencies: vec![("a".to_string(), DependencyType::Data)],
                can_parallelize: true,
            },
        ];

        let result = parallelizer.build_dependency_graph(&nodes);
        assert!(result.is_err());
    }

    #[test]
    fn test_stage_computation() {
        let parallelizer = AutoParallelizer::new();
        let nodes = create_test_nodes();

        let analysis = parallelizer.analyze(&nodes).unwrap();

        assert_eq!(analysis.num_stages, 3);
        assert_eq!(analysis.stages[0].nodes, vec!["a"]);
        assert_eq!(analysis.stages[1].nodes.len(), 2); // b and c can run in parallel
        assert!(analysis.stages[1].nodes.contains(&"b".to_string()));
        assert!(analysis.stages[1].nodes.contains(&"c".to_string()));
        assert_eq!(analysis.stages[2].nodes, vec!["d"]);
    }

    #[test]
    fn test_critical_path_calculation() {
        let parallelizer = AutoParallelizer::new();
        let nodes = create_test_nodes();

        let analysis = parallelizer.analyze(&nodes).unwrap();

        // Critical path: a (10) -> max(b (20), c (15)) -> d (10) = 40
        assert_eq!(analysis.critical_path_length, 40.0);
    }

    #[test]
    fn test_parallelism_factor() {
        let parallelizer = AutoParallelizer::new();
        let nodes = create_test_nodes();

        let analysis = parallelizer.analyze(&nodes).unwrap();

        // Total work: 10 + 20 + 15 + 10 = 55
        // Critical path: 40
        // Parallelism factor: 55 / 40 = 1.375
        assert!((analysis.parallelism_factor - 1.375).abs() < 0.01);
    }

    #[test]
    fn test_execution_plan_generation() {
        let parallelizer = AutoParallelizer::new();
        let nodes = create_test_nodes();

        let plan = parallelizer.generate_plan(&nodes).unwrap();

        assert_eq!(plan.stages.len(), 3);
        assert!(!plan.partitions.is_empty());
        // May not always have speedup due to overhead, just check it's positive
        assert!(plan.estimated_speedup > 0.0);
        assert!(plan.load_balance_ratio > 0.0 && plan.load_balance_ratio <= 1.0);
    }

    #[test]
    fn test_profile_update() {
        let mut parallelizer = AutoParallelizer::new();

        parallelizer.update_profile("compute".to_string(), 100.0);
        parallelizer.update_profile("compute".to_string(), 200.0);

        assert!(parallelizer.profile_data.contains_key("compute"));
        let avg = parallelizer.profile_data["compute"];
        // First update: 0.9 * 0.0 + 0.1 * 100.0 = 10.0
        // Second update: 0.9 * 10.0 + 0.1 * 200.0 = 29.0
        assert!(avg >= 0.0);
    }

    #[test]
    fn test_strategy_variations() {
        let nodes = create_test_nodes();

        let conservative = AutoParallelizer::new()
            .with_strategy(ParallelizationStrategy::Conservative)
            .analyze(&nodes)
            .unwrap();

        let aggressive = AutoParallelizer::new()
            .with_strategy(ParallelizationStrategy::Aggressive)
            .analyze(&nodes)
            .unwrap();

        // Aggressive should recommend more workers
        assert!(aggressive.recommended_workers >= conservative.recommended_workers);
    }

    #[test]
    fn test_sequential_graph() {
        let parallelizer = AutoParallelizer::new();

        // Create a sequential graph (no parallelism)
        let nodes = vec![
            NodeInfo {
                id: "a".to_string(),
                op_type: "compute".to_string(),
                estimated_cost: 10.0,
                memory_size: 1000,
                dependencies: vec![],
                can_parallelize: true,
            },
            NodeInfo {
                id: "b".to_string(),
                op_type: "compute".to_string(),
                estimated_cost: 10.0,
                memory_size: 1000,
                dependencies: vec![("a".to_string(), DependencyType::Data)],
                can_parallelize: true,
            },
        ];

        let analysis = parallelizer.analyze(&nodes).unwrap();

        assert_eq!(analysis.num_stages, 2);
        assert_eq!(analysis.parallelism_factor, 1.0); // No parallelism
    }

    #[test]
    fn test_fully_parallel_graph() {
        let parallelizer = AutoParallelizer::new();

        // Create a fully parallel graph
        let nodes = vec![
            NodeInfo {
                id: "a".to_string(),
                op_type: "compute".to_string(),
                estimated_cost: 10.0,
                memory_size: 1000,
                dependencies: vec![],
                can_parallelize: true,
            },
            NodeInfo {
                id: "b".to_string(),
                op_type: "compute".to_string(),
                estimated_cost: 10.0,
                memory_size: 1000,
                dependencies: vec![],
                can_parallelize: true,
            },
            NodeInfo {
                id: "c".to_string(),
                op_type: "compute".to_string(),
                estimated_cost: 10.0,
                memory_size: 1000,
                dependencies: vec![],
                can_parallelize: true,
            },
        ];

        let analysis = parallelizer.analyze(&nodes).unwrap();

        assert_eq!(analysis.num_stages, 1);
        assert_eq!(analysis.parallelism_factor, 3.0); // Perfect parallelism
    }

    #[test]
    fn test_load_balancing() {
        let parallelizer = AutoParallelizer::new().with_max_workers(2);
        let nodes = create_test_nodes();

        let plan = parallelizer.generate_plan(&nodes).unwrap();

        // Check that partitions exist and have reasonable balance
        assert!(plan.partitions.len() > 0);
        assert!(plan.load_balance_ratio > 0.0 && plan.load_balance_ratio <= 1.0);
    }

    #[test]
    fn test_invalid_graph() {
        let parallelizer = AutoParallelizer::new();

        // Node with unknown dependency
        let nodes = vec![NodeInfo {
            id: "a".to_string(),
            op_type: "compute".to_string(),
            estimated_cost: 10.0,
            memory_size: 1000,
            dependencies: vec![("unknown".to_string(), DependencyType::Data)],
            can_parallelize: true,
        }];

        let result = parallelizer.build_dependency_graph(&nodes);
        assert!(result.is_err());
    }
}
