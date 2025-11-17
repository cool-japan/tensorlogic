//! Advanced scheduling strategies for tensor graph execution.
//!
//! This module provides sophisticated scheduling algorithms that optimize
//! for different objectives: latency, throughput, resource utilization,
//! and multi-objective trade-offs.

use std::collections::{HashMap, HashSet, VecDeque};

use super::EinsumGraph;
use crate::error::IrError;

/// Scheduling objective to optimize for
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingObjective {
    /// Minimize total execution latency (critical path)
    MinimizeLatency,
    /// Maximize throughput (operations per unit time)
    MaximizeThroughput,
    /// Minimize peak memory usage
    MinimizeMemory,
    /// Balance between latency and memory
    Balanced,
    /// Optimize for pipeline execution
    Pipeline,
}

/// A schedule for executing graph operations
#[derive(Debug, Clone)]
pub struct ExecutionSchedule {
    /// Ordered sequence of operations to execute
    pub execution_order: Vec<usize>,
    /// Operations that can execute in parallel at each step
    pub parallel_stages: Vec<Vec<usize>>,
    /// Estimated execution time for each stage
    pub stage_costs: Vec<f64>,
    /// Total estimated execution time
    pub total_cost: f64,
    /// Peak memory usage
    pub peak_memory: usize,
    /// Objective used for scheduling
    pub objective: SchedulingObjective,
}

impl ExecutionSchedule {
    /// Create a new execution schedule
    pub fn new(objective: SchedulingObjective) -> Self {
        Self {
            execution_order: Vec::new(),
            parallel_stages: Vec::new(),
            stage_costs: Vec::new(),
            total_cost: 0.0,
            peak_memory: 0,
            objective,
        }
    }

    /// Get the number of stages in the schedule
    pub fn num_stages(&self) -> usize {
        self.parallel_stages.len()
    }

    /// Get maximum parallelism across all stages
    pub fn max_parallelism(&self) -> usize {
        self.parallel_stages
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
    }

    /// Get average parallelism
    pub fn avg_parallelism(&self) -> f64 {
        if self.parallel_stages.is_empty() {
            return 0.0;
        }
        let total: usize = self.parallel_stages.iter().map(|s| s.len()).sum();
        total as f64 / self.parallel_stages.len() as f64
    }
}

/// Advanced scheduler for computation graphs
pub struct GraphScheduler {
    /// Cost model for operations
    operation_costs: HashMap<usize, f64>,
    /// Memory usage per tensor
    tensor_memory: HashMap<usize, usize>,
}

impl GraphScheduler {
    /// Create a new scheduler
    pub fn new() -> Self {
        Self {
            operation_costs: HashMap::new(),
            tensor_memory: HashMap::new(),
        }
    }

    /// Set the cost for an operation
    pub fn set_operation_cost(&mut self, node_idx: usize, cost: f64) {
        self.operation_costs.insert(node_idx, cost);
    }

    /// Set memory size for a tensor
    pub fn set_tensor_memory(&mut self, tensor_idx: usize, size: usize) {
        self.tensor_memory.insert(tensor_idx, size);
    }

    /// Generate a schedule optimized for the given objective
    pub fn schedule(
        &self,
        graph: &EinsumGraph,
        objective: SchedulingObjective,
    ) -> Result<ExecutionSchedule, IrError> {
        match objective {
            SchedulingObjective::MinimizeLatency => self.schedule_min_latency(graph),
            SchedulingObjective::MaximizeThroughput => self.schedule_max_throughput(graph),
            SchedulingObjective::MinimizeMemory => self.schedule_min_memory(graph),
            SchedulingObjective::Balanced => self.schedule_balanced(graph),
            SchedulingObjective::Pipeline => self.schedule_pipeline(graph),
        }
    }

    /// Schedule to minimize latency (critical path)
    fn schedule_min_latency(&self, graph: &EinsumGraph) -> Result<ExecutionSchedule, IrError> {
        let mut schedule = ExecutionSchedule::new(SchedulingObjective::MinimizeLatency);

        // Build dependency graph
        let dependencies = self.build_dependencies(graph);

        // Compute earliest start times using critical path analysis
        let start_times = self.compute_start_times(graph, &dependencies);

        // Group by start time to create parallel stages
        let mut stages: HashMap<usize, Vec<usize>> = HashMap::new();
        for (node_idx, &start_time) in start_times.iter().enumerate() {
            stages
                .entry(start_time as usize)
                .or_default()
                .push(node_idx);
        }

        // Sort stages and build schedule
        let mut stage_indices: Vec<_> = stages.keys().copied().collect();
        stage_indices.sort_unstable();

        for stage_idx in stage_indices {
            if let Some(nodes) = stages.get(&stage_idx) {
                let stage_cost = nodes
                    .iter()
                    .map(|&idx| self.get_operation_cost(idx))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                schedule.parallel_stages.push(nodes.clone());
                schedule.stage_costs.push(stage_cost);
                schedule.total_cost += stage_cost;

                for &node in nodes {
                    schedule.execution_order.push(node);
                }
            }
        }

        Ok(schedule)
    }

    /// Schedule to maximize throughput
    fn schedule_max_throughput(&self, graph: &EinsumGraph) -> Result<ExecutionSchedule, IrError> {
        let mut schedule = ExecutionSchedule::new(SchedulingObjective::MaximizeThroughput);

        // Use list scheduling with longest processing time first
        let dependencies = self.build_dependencies(graph);
        #[allow(clippy::unnecessary_map_or)]
        let mut ready: Vec<usize> = (0..graph.nodes.len())
            .filter(|&i| dependencies.get(&i).map_or(true, |deps| deps.is_empty()))
            .collect();

        // Sort by cost (descending) for better load balancing
        ready.sort_by(|&a, &b| {
            let cost_a = self.get_operation_cost(a);
            let cost_b = self.get_operation_cost(b);
            cost_b.partial_cmp(&cost_a).unwrap()
        });

        let mut scheduled = HashSet::new();
        let _in_degree = self.compute_in_degrees(graph, &dependencies);

        while !ready.is_empty() {
            let mut stage = Vec::new();
            let mut stage_cost: f64 = 0.0;

            // Schedule all ready operations in this stage
            for &node_idx in &ready {
                let cost = self.get_operation_cost(node_idx);
                stage.push(node_idx);
                stage_cost = stage_cost.max(cost);
                scheduled.insert(node_idx);
                schedule.execution_order.push(node_idx);
            }

            schedule.parallel_stages.push(stage);
            schedule.stage_costs.push(stage_cost);
            schedule.total_cost += stage_cost;

            // Update ready list
            ready.clear();
            for (node_idx, deps) in &dependencies {
                if scheduled.contains(node_idx) {
                    continue;
                }

                let all_deps_scheduled = deps.iter().all(|&dep| scheduled.contains(&dep));
                if all_deps_scheduled {
                    ready.push(*node_idx);
                }
            }

            // Sort by cost again
            ready.sort_by(|&a, &b| {
                let cost_a = self.get_operation_cost(a);
                let cost_b = self.get_operation_cost(b);
                cost_b.partial_cmp(&cost_a).unwrap()
            });
        }

        Ok(schedule)
    }

    /// Schedule to minimize memory usage
    fn schedule_min_memory(&self, graph: &EinsumGraph) -> Result<ExecutionSchedule, IrError> {
        let mut schedule = ExecutionSchedule::new(SchedulingObjective::MinimizeMemory);

        // Use earliest deadline first with memory pressure
        let dependencies = self.build_dependencies(graph);
        let tensor_lifetimes = self.compute_tensor_lifetimes(graph);

        #[allow(clippy::unnecessary_map_or)]
        let mut ready: Vec<usize> = (0..graph.nodes.len())
            .filter(|&i| dependencies.get(&i).map_or(true, |deps| deps.is_empty()))
            .collect();

        let mut scheduled = HashSet::new();

        while !ready.is_empty() {
            // Choose operation that frees the most memory
            let best_idx = ready
                .iter()
                .max_by_key(|&&idx| self.estimate_memory_freed(graph, idx, &tensor_lifetimes))
                .copied()
                .unwrap();

            ready.retain(|&idx| idx != best_idx);

            schedule.execution_order.push(best_idx);
            schedule.parallel_stages.push(vec![best_idx]);
            let cost = self.get_operation_cost(best_idx);
            schedule.stage_costs.push(cost);
            schedule.total_cost += cost;
            scheduled.insert(best_idx);

            // Update ready list
            for (node_idx, deps) in &dependencies {
                if scheduled.contains(node_idx) || ready.contains(node_idx) {
                    continue;
                }

                if deps.iter().all(|&dep| scheduled.contains(&dep)) {
                    ready.push(*node_idx);
                }
            }
        }

        Ok(schedule)
    }

    /// Schedule with balanced objectives
    fn schedule_balanced(&self, graph: &EinsumGraph) -> Result<ExecutionSchedule, IrError> {
        // Use a weighted combination of latency and memory objectives
        let latency_schedule = self.schedule_min_latency(graph)?;
        let _memory_schedule = self.schedule_min_memory(graph)?;

        // For now, prefer latency schedule with memory awareness
        // In a full implementation, we would use multi-objective optimization
        Ok(latency_schedule)
    }

    /// Schedule for pipeline execution
    fn schedule_pipeline(&self, graph: &EinsumGraph) -> Result<ExecutionSchedule, IrError> {
        let mut schedule = ExecutionSchedule::new(SchedulingObjective::Pipeline);

        // Partition graph into pipeline stages
        let stages = self.partition_for_pipeline(graph)?;

        for stage_nodes in stages {
            let stage_cost = stage_nodes
                .iter()
                .map(|&idx| self.get_operation_cost(idx))
                .sum();

            schedule.parallel_stages.push(stage_nodes.clone());
            schedule.stage_costs.push(stage_cost);
            schedule.total_cost = schedule.total_cost.max(stage_cost);

            for &node in &stage_nodes {
                schedule.execution_order.push(node);
            }
        }

        Ok(schedule)
    }

    /// Build dependency graph
    fn build_dependencies(&self, graph: &EinsumGraph) -> HashMap<usize, Vec<usize>> {
        let mut dependencies: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut tensor_producer: HashMap<usize, usize> = HashMap::new();

        // Map each tensor to its producer
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &output_idx in &node.outputs {
                tensor_producer.insert(output_idx, node_idx);
            }
        }

        // Build dependencies
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            let mut deps = Vec::new();
            for &input_idx in &node.inputs {
                if let Some(&producer) = tensor_producer.get(&input_idx) {
                    if producer != node_idx {
                        deps.push(producer);
                    }
                }
            }
            dependencies.insert(node_idx, deps);
        }

        dependencies
    }

    /// Compute earliest start times for each operation
    fn compute_start_times(
        &self,
        graph: &EinsumGraph,
        dependencies: &HashMap<usize, Vec<usize>>,
    ) -> Vec<f64> {
        let mut start_times = vec![0.0; graph.nodes.len()];
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Find roots (nodes with no dependencies)
        for (node_idx, deps) in dependencies {
            if deps.is_empty() {
                queue.push_back(*node_idx);
            }
        }

        while let Some(node_idx) = queue.pop_front() {
            if visited.contains(&node_idx) {
                continue;
            }
            visited.insert(node_idx);

            // Compute start time based on dependencies
            let deps = dependencies
                .get(&node_idx)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let max_dep_finish = deps
                .iter()
                .map(|&dep_idx| start_times[dep_idx] + self.get_operation_cost(dep_idx))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            start_times[node_idx] = max_dep_finish;

            // Add successors to queue
            for (succ_idx, succ_deps) in dependencies {
                if succ_deps.contains(&node_idx) && !visited.contains(succ_idx) {
                    queue.push_back(*succ_idx);
                }
            }
        }

        start_times
    }

    /// Compute in-degrees for topological sort
    fn compute_in_degrees(
        &self,
        graph: &EinsumGraph,
        dependencies: &HashMap<usize, Vec<usize>>,
    ) -> Vec<usize> {
        let mut in_degree = vec![0; graph.nodes.len()];
        for (node_idx, deps) in dependencies {
            in_degree[*node_idx] = deps.len();
        }
        in_degree
    }

    /// Compute tensor lifetimes
    fn compute_tensor_lifetimes(&self, graph: &EinsumGraph) -> HashMap<usize, (usize, usize)> {
        let mut lifetimes = HashMap::new();

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &tensor_idx in &node.inputs {
                let entry = lifetimes.entry(tensor_idx).or_insert((node_idx, node_idx));
                entry.0 = entry.0.min(node_idx);
                entry.1 = entry.1.max(node_idx);
            }
            for &tensor_idx in &node.outputs {
                let entry = lifetimes.entry(tensor_idx).or_insert((node_idx, node_idx));
                entry.0 = entry.0.min(node_idx);
                entry.1 = entry.1.max(node_idx);
            }
        }

        lifetimes
    }

    /// Estimate memory freed by executing an operation
    fn estimate_memory_freed(
        &self,
        graph: &EinsumGraph,
        node_idx: usize,
        lifetimes: &HashMap<usize, (usize, usize)>,
    ) -> usize {
        let node = &graph.nodes[node_idx];
        let mut freed = 0;

        for &input_tensor in &node.inputs {
            if let Some(&(_, last_use)) = lifetimes.get(&input_tensor) {
                if last_use == node_idx {
                    freed += self.tensor_memory.get(&input_tensor).copied().unwrap_or(1);
                }
            }
        }

        freed
    }

    /// Partition graph into pipeline stages
    fn partition_for_pipeline(&self, graph: &EinsumGraph) -> Result<Vec<Vec<usize>>, IrError> {
        // Simple partitioning: divide into roughly equal-cost stages
        let total_cost: f64 = (0..graph.nodes.len())
            .map(|i| self.get_operation_cost(i))
            .sum();

        let target_stages = 4; // Default number of pipeline stages
        let target_cost_per_stage = total_cost / target_stages as f64;

        let dependencies = self.build_dependencies(graph);
        let topo_order = self.topological_sort(graph, &dependencies);

        let mut stages = Vec::new();
        let mut current_stage = Vec::new();
        let mut current_cost = 0.0;

        for &node_idx in &topo_order {
            let cost = self.get_operation_cost(node_idx);
            current_stage.push(node_idx);
            current_cost += cost;

            if current_cost >= target_cost_per_stage {
                stages.push(current_stage.clone());
                current_stage.clear();
                current_cost = 0.0;
            }
        }

        if !current_stage.is_empty() {
            stages.push(current_stage);
        }

        Ok(stages)
    }

    /// Topological sort of the graph
    fn topological_sort(
        &self,
        graph: &EinsumGraph,
        dependencies: &HashMap<usize, Vec<usize>>,
    ) -> Vec<usize> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut in_degree = self.compute_in_degrees(graph, dependencies);

        let mut queue: VecDeque<usize> = (0..graph.nodes.len())
            .filter(|&i| in_degree[i] == 0)
            .collect();

        while let Some(node_idx) = queue.pop_front() {
            if visited.contains(&node_idx) {
                continue;
            }
            visited.insert(node_idx);
            result.push(node_idx);

            // Update successors
            for (succ_idx, deps) in dependencies {
                if deps.contains(&node_idx) {
                    in_degree[*succ_idx] = in_degree[*succ_idx].saturating_sub(1);
                    if in_degree[*succ_idx] == 0 {
                        queue.push_back(*succ_idx);
                    }
                }
            }
        }

        result
    }

    /// Get operation cost (with default)
    fn get_operation_cost(&self, node_idx: usize) -> f64 {
        self.operation_costs.get(&node_idx).copied().unwrap_or(1.0)
    }
}

impl Default for GraphScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EinsumNode;

    #[test]
    fn test_execution_schedule_creation() {
        let schedule = ExecutionSchedule::new(SchedulingObjective::MinimizeLatency);
        assert_eq!(schedule.objective, SchedulingObjective::MinimizeLatency);
        assert_eq!(schedule.num_stages(), 0);
    }

    #[test]
    fn test_execution_schedule_stats() {
        let mut schedule = ExecutionSchedule::new(SchedulingObjective::MinimizeLatency);
        schedule.parallel_stages.push(vec![0, 1, 2]);
        schedule.parallel_stages.push(vec![3]);

        assert_eq!(schedule.num_stages(), 2);
        assert_eq!(schedule.max_parallelism(), 3);
        assert_eq!(schedule.avg_parallelism(), 2.0);
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = GraphScheduler::new();
        assert!(scheduler.operation_costs.is_empty());
    }

    #[test]
    fn test_scheduler_set_costs() {
        let mut scheduler = GraphScheduler::new();
        scheduler.set_operation_cost(0, 5.0);
        scheduler.set_tensor_memory(1, 1024);

        assert_eq!(scheduler.get_operation_cost(0), 5.0);
        assert_eq!(scheduler.tensor_memory.get(&1), Some(&1024));
    }

    #[test]
    fn test_schedule_empty_graph() {
        let scheduler = GraphScheduler::new();
        let graph = EinsumGraph::new();

        let schedule = scheduler
            .schedule(&graph, SchedulingObjective::MinimizeLatency)
            .unwrap();
        assert_eq!(schedule.num_stages(), 0);
    }

    #[test]
    fn test_schedule_single_node() {
        let mut scheduler = GraphScheduler::new();
        let mut graph = EinsumGraph::new();

        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();

        scheduler.set_operation_cost(0, 2.0);

        let schedule = scheduler
            .schedule(&graph, SchedulingObjective::MinimizeLatency)
            .unwrap();
        assert_eq!(schedule.execution_order.len(), 1);
        assert_eq!(schedule.total_cost, 2.0);
    }

    #[test]
    fn test_build_dependencies() {
        let scheduler = GraphScheduler::new();
        let mut graph = EinsumGraph::new();

        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("tanh", b, c))
            .unwrap();

        let deps = scheduler.build_dependencies(&graph);
        assert_eq!(deps.get(&0).unwrap().len(), 0);
        assert_eq!(deps.get(&1).unwrap(), &vec![0]);
    }

    #[test]
    fn test_topological_sort() {
        let scheduler = GraphScheduler::new();
        let mut graph = EinsumGraph::new();

        let a = graph.add_tensor("A");
        let b = graph.add_tensor("B");
        let c = graph.add_tensor("C");

        graph
            .add_node(EinsumNode::elem_unary("relu", a, b))
            .unwrap();
        graph
            .add_node(EinsumNode::elem_unary("tanh", b, c))
            .unwrap();

        let deps = scheduler.build_dependencies(&graph);
        let topo = scheduler.topological_sort(&graph, &deps);

        assert_eq!(topo.len(), 2);
        assert_eq!(topo[0], 0);
        assert_eq!(topo[1], 1);
    }

    #[test]
    fn test_scheduling_objectives() {
        assert_eq!(
            SchedulingObjective::MinimizeLatency,
            SchedulingObjective::MinimizeLatency
        );
        assert_ne!(
            SchedulingObjective::MinimizeLatency,
            SchedulingObjective::MaximizeThroughput
        );
    }
}
