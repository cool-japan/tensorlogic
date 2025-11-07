//! Execution scheduling and optimization for efficient graph execution.

use std::collections::{HashMap, HashSet, VecDeque};

use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

use crate::capabilities::DeviceType;

/// Execution schedule for a graph
#[derive(Debug, Clone)]
pub struct ExecutionSchedule {
    /// Ordered list of node indices to execute
    pub execution_order: Vec<usize>,
    /// Device placement for each node
    pub device_placement: HashMap<usize, DeviceType>,
    /// Parallel execution groups (nodes that can run concurrently)
    pub parallel_groups: Vec<Vec<usize>>,
    /// Estimated execution cost (arbitrary units)
    pub estimated_cost: f64,
}

impl ExecutionSchedule {
    pub fn new() -> Self {
        ExecutionSchedule {
            execution_order: Vec::new(),
            device_placement: HashMap::new(),
            parallel_groups: Vec::new(),
            estimated_cost: 0.0,
        }
    }

    pub fn sequential(num_nodes: usize, device: DeviceType) -> Self {
        let execution_order: Vec<usize> = (0..num_nodes).collect();
        let device_placement: HashMap<_, _> = (0..num_nodes).map(|i| (i, device)).collect();
        let parallel_groups: Vec<Vec<usize>> = execution_order.iter().map(|&i| vec![i]).collect();

        ExecutionSchedule {
            execution_order,
            device_placement,
            parallel_groups,
            estimated_cost: num_nodes as f64,
        }
    }

    pub fn len(&self) -> usize {
        self.execution_order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.execution_order.is_empty()
    }

    pub fn get_device(&self, node_idx: usize) -> Option<DeviceType> {
        self.device_placement.get(&node_idx).copied()
    }

    pub fn num_parallel_stages(&self) -> usize {
        self.parallel_groups.len()
    }
}

impl Default for ExecutionSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Scheduling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// Execute nodes in topological order
    Sequential,
    /// Maximize parallelism (minimize depth)
    MaximizeParallelism,
    /// Minimize memory usage (reuse tensors aggressively)
    MinimizeMemory,
    /// Balance parallelism and memory
    Balanced,
    /// Custom cost-based optimization
    CostBased,
}

/// Node cost model for scheduling decisions
#[derive(Debug, Clone)]
pub struct NodeCost {
    pub compute_cost: f64,
    pub memory_cost: usize,
    pub communication_cost: f64,
}

impl NodeCost {
    pub fn new() -> Self {
        NodeCost {
            compute_cost: 1.0,
            memory_cost: 0,
            communication_cost: 0.0,
        }
    }

    pub fn estimate_from_node(node: &EinsumNode) -> Self {
        let compute_cost = match &node.op {
            OpType::Einsum { spec } => {
                // Estimate based on einsum complexity
                let num_indices = spec.chars().filter(|c| c.is_alphabetic()).count();
                (num_indices as f64).powi(2) // Rough O(n²) estimate
            }
            OpType::ElemUnary { .. } => 1.0,
            OpType::ElemBinary { .. } => 1.5,
            OpType::Reduce { axes, .. } => 2.0 + axes.len() as f64,
        };

        NodeCost {
            compute_cost,
            memory_cost: 1024, // Default 1KB estimate
            communication_cost: 0.0,
        }
    }

    pub fn total_cost(&self) -> f64 {
        self.compute_cost + self.communication_cost + (self.memory_cost as f64 / 1024.0)
    }
}

impl Default for NodeCost {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph scheduler
pub struct Scheduler {
    strategy: SchedulingStrategy,
}

impl Scheduler {
    pub fn new(strategy: SchedulingStrategy) -> Self {
        Scheduler { strategy }
    }

    /// Generate execution schedule for a graph
    pub fn schedule(&self, graph: &EinsumGraph) -> ExecutionSchedule {
        match self.strategy {
            SchedulingStrategy::Sequential => self.schedule_sequential(graph),
            SchedulingStrategy::MaximizeParallelism => self.schedule_parallel(graph),
            SchedulingStrategy::MinimizeMemory => self.schedule_memory_efficient(graph),
            SchedulingStrategy::Balanced => self.schedule_balanced(graph),
            SchedulingStrategy::CostBased => self.schedule_cost_based(graph),
        }
    }

    fn schedule_sequential(&self, graph: &EinsumGraph) -> ExecutionSchedule {
        ExecutionSchedule::sequential(graph.nodes.len(), DeviceType::CPU)
    }

    fn schedule_parallel(&self, graph: &EinsumGraph) -> ExecutionSchedule {
        let mut schedule = ExecutionSchedule::new();
        let num_nodes = graph.nodes.len();
        let _num_tensors = graph.tensors.len();

        // Build dependency graph
        let deps = self.build_dependency_graph(graph);

        // Compute levels (maximum distance from input)
        let levels = self.compute_node_levels(graph, &deps);

        // Group nodes by level for parallel execution
        let max_level = *levels.values().max().unwrap_or(&0);
        let mut level_groups: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];

        for (node_idx, &level) in &levels {
            level_groups[level].push(*node_idx);
        }

        // Create execution order (level-by-level)
        for group in &level_groups {
            schedule.execution_order.extend(group);
            if !group.is_empty() {
                schedule.parallel_groups.push(group.clone());
            }
        }

        // Assign all nodes to CPU (default)
        for i in 0..num_nodes {
            schedule.device_placement.insert(i, DeviceType::CPU);
        }

        // Estimate cost as number of levels (critical path length)
        schedule.estimated_cost = (max_level + 1) as f64;

        schedule
    }

    fn schedule_memory_efficient(&self, graph: &EinsumGraph) -> ExecutionSchedule {
        let mut schedule = ExecutionSchedule::new();
        let num_nodes = graph.nodes.len();
        let num_tensors = graph.tensors.len();

        // Build dependency graph
        let deps = self.build_dependency_graph(graph);

        // Greedy scheduling: execute nodes that free the most memory first
        let mut executed = HashSet::new();
        let mut ready_queue = VecDeque::new();

        // Find initial ready nodes (no dependencies or all deps satisfied)
        for node_idx in 0..num_nodes {
            if self.is_ready(node_idx, &deps, &executed, num_tensors) {
                ready_queue.push_back(node_idx);
            }
        }

        while let Some(node_idx) = ready_queue.pop_front() {
            if executed.contains(&node_idx) {
                continue;
            }

            schedule.execution_order.push(node_idx);
            schedule.parallel_groups.push(vec![node_idx]);
            schedule.device_placement.insert(node_idx, DeviceType::CPU);
            executed.insert(node_idx);

            // Add newly ready nodes
            for next_idx in 0..num_nodes {
                if !executed.contains(&next_idx)
                    && self.is_ready(next_idx, &deps, &executed, num_tensors)
                {
                    ready_queue.push_back(next_idx);
                }
            }
        }

        schedule.estimated_cost = num_nodes as f64;
        schedule
    }

    fn schedule_balanced(&self, graph: &EinsumGraph) -> ExecutionSchedule {
        // Compromise between parallelism and memory
        // Use parallel scheduling but limit group sizes
        let mut parallel_schedule = self.schedule_parallel(graph);

        // Merge small groups to reduce overhead
        let mut merged_groups = Vec::new();
        let mut current_group = Vec::new();

        for group in parallel_schedule.parallel_groups {
            if group.len() > 4 {
                // Large group: keep separate
                if !current_group.is_empty() {
                    merged_groups.push(current_group.clone());
                    current_group.clear();
                }
                merged_groups.push(group);
            } else {
                // Small group: accumulate
                current_group.extend(group);
                if current_group.len() >= 4 {
                    merged_groups.push(current_group.clone());
                    current_group.clear();
                }
            }
        }

        if !current_group.is_empty() {
            merged_groups.push(current_group);
        }

        parallel_schedule.parallel_groups = merged_groups;
        parallel_schedule.estimated_cost *= 1.2; // Slight overhead from merging

        parallel_schedule
    }

    fn schedule_cost_based(&self, graph: &EinsumGraph) -> ExecutionSchedule {
        let mut schedule = ExecutionSchedule::new();
        let num_nodes = graph.nodes.len();

        // Estimate costs for each node
        let costs: Vec<NodeCost> = graph
            .nodes
            .iter()
            .map(NodeCost::estimate_from_node)
            .collect();

        // Build dependency graph
        let deps = self.build_dependency_graph(graph);

        // Compute critical path costs
        let critical_costs = self.compute_critical_path_costs(graph, &costs, &deps);

        // Sort by critical path cost (highest first for better parallelism)
        let mut node_priorities: Vec<(usize, f64)> = critical_costs
            .iter()
            .enumerate()
            .map(|(i, &cost)| (i, cost))
            .collect();
        node_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Schedule using priority-based topological sort
        let mut executed = HashSet::new();
        let num_tensors = graph.tensors.len();

        while executed.len() < num_nodes {
            let mut current_wave = Vec::new();

            for &(node_idx, _) in &node_priorities {
                if executed.contains(&node_idx) {
                    continue;
                }

                if self.is_ready(node_idx, &deps, &executed, num_tensors) {
                    current_wave.push(node_idx);
                    executed.insert(node_idx);
                }
            }

            if current_wave.is_empty() {
                break; // Avoid infinite loop on cyclic graphs
            }

            schedule.execution_order.extend(&current_wave);
            schedule.parallel_groups.push(current_wave);
        }

        // Assign devices (all CPU for now)
        for i in 0..num_nodes {
            schedule.device_placement.insert(i, DeviceType::CPU);
        }

        // Estimate total cost
        schedule.estimated_cost = costs.iter().map(|c| c.total_cost()).sum();

        schedule
    }

    fn build_dependency_graph(&self, graph: &EinsumGraph) -> HashMap<usize, Vec<usize>> {
        let mut deps: HashMap<usize, Vec<usize>> = HashMap::new();

        // Build a mapping from tensor index to the node that produces it
        let mut tensor_producers: HashMap<usize, usize> = HashMap::new();
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &output_idx in &node.outputs {
                tensor_producers.insert(output_idx, node_idx);
            }
        }

        // For each node, find which other nodes it depends on
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            let mut node_deps = Vec::new();
            for &input_idx in &node.inputs {
                // Check if this tensor is produced by another node
                if let Some(&producer_idx) = tensor_producers.get(&input_idx) {
                    node_deps.push(producer_idx);
                }
            }
            deps.insert(node_idx, node_deps);
        }

        deps
    }

    fn compute_node_levels(
        &self,
        graph: &EinsumGraph,
        deps: &HashMap<usize, Vec<usize>>,
    ) -> HashMap<usize, usize> {
        let mut levels = HashMap::new();
        let num_nodes = graph.nodes.len();

        // Compute levels iteratively
        for _ in 0..num_nodes {
            for node_idx in 0..num_nodes {
                let max_dep_level = deps
                    .get(&node_idx)
                    .map(|d| d.iter().filter_map(|&i| levels.get(&i)).max().copied())
                    .unwrap_or(None);

                let level = max_dep_level.map(|l| l + 1).unwrap_or(0);
                levels.insert(node_idx, level);
            }
        }

        levels
    }

    fn compute_critical_path_costs(
        &self,
        graph: &EinsumGraph,
        costs: &[NodeCost],
        deps: &HashMap<usize, Vec<usize>>,
    ) -> Vec<f64> {
        let num_nodes = graph.nodes.len();
        let mut critical_costs = vec![0.0; num_nodes];

        // Compute critical path costs iteratively (reverse topological order)
        for _ in 0..num_nodes {
            for node_idx in (0..num_nodes).rev() {
                let node_cost = costs[node_idx].total_cost();

                // Find max cost among dependent nodes
                let max_successor_cost = (0..num_nodes)
                    .filter(|&i| deps.get(&i).map(|d| d.contains(&node_idx)).unwrap_or(false))
                    .map(|i| critical_costs[i])
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                critical_costs[node_idx] = node_cost + max_successor_cost;
            }
        }

        critical_costs
    }

    fn is_ready(
        &self,
        _node_idx: usize,
        deps: &HashMap<usize, Vec<usize>>,
        executed: &HashSet<usize>,
        _num_tensors: usize,
    ) -> bool {
        let node_idx = _node_idx;
        deps.get(&node_idx)
            .map(|d| d.iter().all(|&dep| executed.contains(&dep)))
            .unwrap_or(true)
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(SchedulingStrategy::Balanced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();
        graph.tensors.push("x".to_string());
        graph.tensors.push("y".to_string());
        graph.tensors.push("t2".to_string()); // Output of node 0
        graph.tensors.push("t3".to_string()); // Output of node 1
        graph.tensors.push("t4".to_string()); // Output of node 2

        // Node 0: einsum (depends on tensors 0, 1)
        graph.nodes.push(EinsumNode {
            inputs: vec![0, 1],
            outputs: vec![2],
            op: OpType::Einsum {
                spec: "ab,bc->ac".into(),
            },
            metadata: None,
        });

        // Node 1: unary op (depends on node 0)
        graph.nodes.push(EinsumNode {
            inputs: vec![2], // Output of node 0
            outputs: vec![3],
            op: OpType::ElemUnary { op: "relu".into() },
            metadata: None,
        });

        // Node 2: reduce (depends on node 1)
        graph.nodes.push(EinsumNode {
            inputs: vec![3], // Output of node 1
            outputs: vec![4],
            op: OpType::Reduce {
                op: "sum".into(),
                axes: vec![0],
            },
            metadata: None,
        });

        graph
    }

    #[test]
    fn test_execution_schedule_creation() {
        let schedule = ExecutionSchedule::new();
        assert!(schedule.is_empty());
        assert_eq!(schedule.num_parallel_stages(), 0);
    }

    #[test]
    fn test_sequential_schedule() {
        let schedule = ExecutionSchedule::sequential(5, DeviceType::CPU);
        assert_eq!(schedule.len(), 5);
        assert_eq!(schedule.execution_order, vec![0, 1, 2, 3, 4]);
        assert_eq!(schedule.num_parallel_stages(), 5);

        for i in 0..5 {
            assert_eq!(schedule.get_device(i), Some(DeviceType::CPU));
        }
    }

    #[test]
    fn test_node_cost_estimation() {
        let node = EinsumNode {
            inputs: vec![0, 1],
            outputs: vec![2],
            op: OpType::Einsum {
                spec: "ab,bc->ac".into(),
            },
            metadata: None,
        };

        let cost = NodeCost::estimate_from_node(&node);
        assert!(cost.compute_cost > 0.0);
        assert!(cost.total_cost() > 0.0);
    }

    #[test]
    fn test_scheduler_sequential() {
        let graph = create_test_graph();
        let scheduler = Scheduler::new(SchedulingStrategy::Sequential);
        let schedule = scheduler.schedule(&graph);

        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule.execution_order, vec![0, 1, 2]);
    }

    #[test]
    fn test_scheduler_parallel() {
        let graph = create_test_graph();
        let scheduler = Scheduler::new(SchedulingStrategy::MaximizeParallelism);
        let schedule = scheduler.schedule(&graph);

        assert_eq!(schedule.len(), 3);
        // Parallel schedule should group independent nodes
        assert!(schedule.num_parallel_stages() <= 3);
    }

    #[test]
    fn test_scheduler_memory_efficient() {
        let graph = create_test_graph();
        let scheduler = Scheduler::new(SchedulingStrategy::MinimizeMemory);
        let schedule = scheduler.schedule(&graph);

        assert_eq!(schedule.len(), 3);
        // Should execute in topological order
        assert!(schedule.execution_order.contains(&0));
        assert!(schedule.execution_order.contains(&1));
        assert!(schedule.execution_order.contains(&2));
    }

    #[test]
    fn test_scheduler_balanced() {
        let graph = create_test_graph();
        let scheduler = Scheduler::new(SchedulingStrategy::Balanced);
        let schedule = scheduler.schedule(&graph);

        assert_eq!(schedule.len(), 3);
        assert!(schedule.estimated_cost > 0.0);
    }

    #[test]
    fn test_scheduler_cost_based() {
        let graph = create_test_graph();
        let scheduler = Scheduler::new(SchedulingStrategy::CostBased);
        let schedule = scheduler.schedule(&graph);

        assert_eq!(schedule.len(), 3);
        assert!(schedule.estimated_cost > 0.0);
    }

    #[test]
    fn test_dependency_graph_building() {
        let graph = create_test_graph();
        let scheduler = Scheduler::default();
        let deps = scheduler.build_dependency_graph(&graph);

        assert_eq!(deps.len(), 3);
        assert!(deps[&0].is_empty()); // Node 0 has no node dependencies
        assert_eq!(deps[&1], vec![0]); // Node 1 depends on node 0
        assert_eq!(deps[&2], vec![1]); // Node 2 depends on node 1
    }

    #[test]
    fn test_node_levels() {
        let graph = create_test_graph();
        let scheduler = Scheduler::default();
        let deps = scheduler.build_dependency_graph(&graph);
        let levels = scheduler.compute_node_levels(&graph, &deps);

        assert_eq!(levels[&0], 0); // Node 0 is at level 0
        assert_eq!(levels[&1], 1); // Node 1 is at level 1
        assert_eq!(levels[&2], 2); // Node 2 is at level 2
    }

    #[test]
    fn test_scheduling_strategies() {
        let strategies = vec![
            SchedulingStrategy::Sequential,
            SchedulingStrategy::MaximizeParallelism,
            SchedulingStrategy::MinimizeMemory,
            SchedulingStrategy::Balanced,
            SchedulingStrategy::CostBased,
        ];

        let graph = create_test_graph();

        for strategy in strategies {
            let scheduler = Scheduler::new(strategy);
            let schedule = scheduler.schedule(&graph);
            assert_eq!(schedule.len(), 3, "Strategy {:?} failed", strategy);
        }
    }
}
