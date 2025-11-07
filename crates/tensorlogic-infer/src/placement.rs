//! Device placement and multi-device execution coordination.

use std::collections::HashMap;

use tensorlogic_ir::EinsumGraph;

use crate::capabilities::DeviceType;
use crate::scheduling::ExecutionSchedule;

/// Device specification with optional id
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    pub device_type: DeviceType,
    pub device_id: usize,
}

impl Device {
    pub fn new(device_type: DeviceType, device_id: usize) -> Self {
        Device {
            device_type,
            device_id,
        }
    }

    pub fn cpu(id: usize) -> Self {
        Device::new(DeviceType::CPU, id)
    }

    pub fn gpu(id: usize) -> Self {
        Device::new(DeviceType::GPU, id)
    }

    pub fn default_cpu() -> Self {
        Device::cpu(0)
    }

    pub fn as_str(&self) -> String {
        format!("{}:{}", self.device_type.as_str(), self.device_id)
    }
}

/// Placement strategy for multi-device execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementStrategy {
    /// Place all operations on a single device
    SingleDevice,
    /// Round-robin placement across devices
    RoundRobin,
    /// Place based on operation cost
    CostBased,
    /// Place to minimize data transfer
    MinimizeTransfer,
    /// Custom placement via callback
    Custom,
}

/// Device placement plan
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    /// Node index -> Device mapping
    pub node_placement: HashMap<usize, Device>,
    /// Tensor index -> Device mapping
    pub tensor_placement: HashMap<usize, Device>,
    /// Estimated transfer cost
    pub transfer_cost: f64,
}

impl PlacementPlan {
    pub fn new() -> Self {
        PlacementPlan {
            node_placement: HashMap::new(),
            tensor_placement: HashMap::new(),
            transfer_cost: 0.0,
        }
    }

    /// Create a single-device placement plan
    pub fn single_device(num_nodes: usize, num_tensors: usize, device: Device) -> Self {
        let mut plan = PlacementPlan::new();

        for i in 0..num_nodes {
            plan.node_placement.insert(i, device);
        }

        for i in 0..num_tensors {
            plan.tensor_placement.insert(i, device);
        }

        plan
    }

    /// Get device for a node
    pub fn get_node_device(&self, node_idx: usize) -> Option<Device> {
        self.node_placement.get(&node_idx).copied()
    }

    /// Get device for a tensor
    pub fn get_tensor_device(&self, tensor_idx: usize) -> Option<Device> {
        self.tensor_placement.get(&tensor_idx).copied()
    }

    /// Count number of cross-device transfers
    pub fn count_transfers(&self, graph: &EinsumGraph) -> usize {
        let mut transfers = 0;

        // Build a mapping from tensor index to the node that produces it
        let mut tensor_producers: HashMap<usize, usize> = HashMap::new();
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            for &output_idx in &node.outputs {
                tensor_producers.insert(output_idx, node_idx);
            }
        }

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            let node_device = self.get_node_device(node_idx);

            for &input_idx in &node.inputs {
                // Determine the device of the input tensor
                let input_device = if let Some(&producer_idx) = tensor_producers.get(&input_idx) {
                    // Tensor is produced by another node
                    self.get_node_device(producer_idx)
                } else {
                    // Tensor is an input tensor
                    self.get_tensor_device(input_idx)
                };

                if node_device != input_device {
                    transfers += 1;
                }
            }
        }

        transfers
    }

    /// Get list of all devices used in this plan
    pub fn devices(&self) -> Vec<Device> {
        let mut devices: Vec<_> = self.node_placement.values().copied().collect();
        devices.sort_by(|a, b| {
            a.device_id
                .cmp(&b.device_id)
                .then_with(|| format!("{:?}", a.device_type).cmp(&format!("{:?}", b.device_type)))
        });
        devices.dedup();
        devices
    }

    /// Summary of the placement plan
    pub fn summary(&self) -> String {
        let devices = self.devices();
        format!(
            "Placement Plan:\n\
             - Nodes: {}\n\
             - Tensors: {}\n\
             - Devices: {} ({:?})\n\
             - Transfer cost: {:.2}",
            self.node_placement.len(),
            self.tensor_placement.len(),
            devices.len(),
            devices.iter().map(|d| d.as_str()).collect::<Vec<_>>(),
            self.transfer_cost
        )
    }
}

impl Default for PlacementPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Device placement optimizer
pub struct PlacementOptimizer {
    strategy: PlacementStrategy,
    available_devices: Vec<Device>,
}

impl PlacementOptimizer {
    pub fn new(strategy: PlacementStrategy, available_devices: Vec<Device>) -> Self {
        PlacementOptimizer {
            strategy,
            available_devices,
        }
    }

    /// Create optimizer for single device
    pub fn single_device(device: Device) -> Self {
        PlacementOptimizer {
            strategy: PlacementStrategy::SingleDevice,
            available_devices: vec![device],
        }
    }

    /// Compute placement plan for a graph
    pub fn place(&self, graph: &EinsumGraph) -> PlacementPlan {
        match self.strategy {
            PlacementStrategy::SingleDevice => self.place_single_device(graph),
            PlacementStrategy::RoundRobin => self.place_round_robin(graph),
            PlacementStrategy::CostBased => self.place_cost_based(graph),
            PlacementStrategy::MinimizeTransfer => self.place_minimize_transfer(graph),
            PlacementStrategy::Custom => self.place_single_device(graph), // Fallback
        }
    }

    /// Compute placement with an execution schedule
    pub fn place_with_schedule(
        &self,
        graph: &EinsumGraph,
        schedule: &ExecutionSchedule,
    ) -> PlacementPlan {
        let mut plan = self.place(graph);

        // Use schedule's device placement if available
        for (node_idx, device_type) in &schedule.device_placement {
            if let Some(device) = self.find_device(*device_type) {
                plan.node_placement.insert(*node_idx, device);
            }
        }

        // Recompute transfer cost
        plan.transfer_cost = self.estimate_transfer_cost(graph, &plan);

        plan
    }

    fn place_single_device(&self, graph: &EinsumGraph) -> PlacementPlan {
        let device = self
            .available_devices
            .first()
            .copied()
            .unwrap_or(Device::default_cpu());
        PlacementPlan::single_device(graph.nodes.len(), graph.tensors.len(), device)
    }

    fn place_round_robin(&self, graph: &EinsumGraph) -> PlacementPlan {
        let mut plan = PlacementPlan::new();

        if self.available_devices.is_empty() {
            return plan;
        }

        // Place input tensors
        for (idx, _) in graph.tensors.iter().enumerate() {
            let device = self.available_devices[idx % self.available_devices.len()];
            plan.tensor_placement.insert(idx, device);
        }

        // Place nodes
        for (idx, _) in graph.nodes.iter().enumerate() {
            let device = self.available_devices[idx % self.available_devices.len()];
            plan.node_placement.insert(idx, device);
        }

        plan.transfer_cost = self.estimate_transfer_cost(graph, &plan);
        plan
    }

    fn place_cost_based(&self, graph: &EinsumGraph) -> PlacementPlan {
        use crate::scheduling::NodeCost;

        let mut plan = PlacementPlan::new();

        if self.available_devices.is_empty() {
            return plan;
        }

        // Compute node costs
        let costs: Vec<f64> = graph
            .nodes
            .iter()
            .map(|node| NodeCost::estimate_from_node(node).total_cost())
            .collect();

        // Use greedy assignment to balance load across devices
        let mut device_loads = vec![0.0; self.available_devices.len()];

        // Sort nodes by cost (descending)
        let mut node_order: Vec<_> = costs.iter().enumerate().collect();
        node_order.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        // Assign each node to the least loaded device
        for (node_idx, &cost) in node_order {
            let min_device_idx = device_loads
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            device_loads[min_device_idx] += cost;
            plan.node_placement
                .insert(node_idx, self.available_devices[min_device_idx]);
        }

        // Place tensors on same device as their first consumer
        let _num_tensors = graph.tensors.len();
        for (tensor_idx, _) in graph.tensors.iter().enumerate() {
            // Find first node that uses this tensor
            let consumer_device = graph
                .nodes
                .iter()
                .enumerate()
                .find(|(_, node)| node.inputs.contains(&tensor_idx))
                .and_then(|(node_idx, _)| plan.node_placement.get(&node_idx))
                .copied()
                .unwrap_or(self.available_devices[0]);

            plan.tensor_placement.insert(tensor_idx, consumer_device);
        }

        plan.transfer_cost = self.estimate_transfer_cost(graph, &plan);
        plan
    }

    fn place_minimize_transfer(&self, graph: &EinsumGraph) -> PlacementPlan {
        let mut plan = PlacementPlan::new();

        if self.available_devices.is_empty() {
            return plan;
        }

        // Start with single device placement
        let default_device = self.available_devices[0];
        plan = PlacementPlan::single_device(graph.nodes.len(), graph.tensors.len(), default_device);

        // Iteratively try to reduce transfers by moving nodes
        let mut improved = true;
        let max_iterations = 10;
        let mut iteration = 0;

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            let current_transfers = plan.count_transfers(graph);

            for node_idx in 0..graph.nodes.len() {
                let current_device = plan.get_node_device(node_idx).unwrap();

                // Try each alternative device
                for &candidate_device in &self.available_devices {
                    if candidate_device == current_device {
                        continue;
                    }

                    // Temporarily change placement
                    plan.node_placement.insert(node_idx, candidate_device);
                    let new_transfers = plan.count_transfers(graph);

                    if new_transfers < current_transfers {
                        // Keep the change
                        improved = true;
                        break;
                    } else {
                        // Revert
                        plan.node_placement.insert(node_idx, current_device);
                    }
                }
            }
        }

        plan.transfer_cost = self.estimate_transfer_cost(graph, &plan);
        plan
    }

    fn estimate_transfer_cost(&self, graph: &EinsumGraph, plan: &PlacementPlan) -> f64 {
        // Simple cost model: 1.0 per transfer
        plan.count_transfers(graph) as f64
    }

    fn find_device(&self, device_type: DeviceType) -> Option<Device> {
        self.available_devices
            .iter()
            .find(|d| d.device_type == device_type)
            .copied()
    }

    pub fn strategy(&self) -> PlacementStrategy {
        self.strategy
    }

    pub fn available_devices(&self) -> &[Device] {
        &self.available_devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::{EinsumNode, OpType};

    fn create_test_graph() -> EinsumGraph {
        let mut graph = EinsumGraph::new();
        graph.tensors.push("x".to_string());
        graph.tensors.push("y".to_string());
        graph.tensors.push("t2".to_string()); // Output of node 0
        graph.tensors.push("t3".to_string()); // Output of node 1

        graph.nodes.push(EinsumNode {
            inputs: vec![0, 1],
            outputs: vec![2],
            op: OpType::Einsum {
                spec: "ab,bc->ac".into(),
            },
            metadata: None,
        });

        graph.nodes.push(EinsumNode {
            inputs: vec![2],
            outputs: vec![3],
            op: OpType::ElemUnary { op: "relu".into() },
            metadata: None,
        });

        graph
    }

    #[test]
    fn test_device_creation() {
        let cpu = Device::cpu(0);
        assert_eq!(cpu.device_type, DeviceType::CPU);
        assert_eq!(cpu.device_id, 0);
        assert_eq!(cpu.as_str(), "CPU:0");

        let gpu = Device::gpu(1);
        assert_eq!(gpu.device_type, DeviceType::GPU);
        assert_eq!(gpu.device_id, 1);
        assert_eq!(gpu.as_str(), "GPU:1");
    }

    #[test]
    fn test_placement_plan_single_device() {
        let device = Device::cpu(0);
        let plan = PlacementPlan::single_device(3, 2, device);

        assert_eq!(plan.node_placement.len(), 3);
        assert_eq!(plan.tensor_placement.len(), 2);
        assert_eq!(plan.get_node_device(0), Some(device));
        assert_eq!(plan.get_tensor_device(0), Some(device));
    }

    #[test]
    fn test_placement_plan_devices() {
        let mut plan = PlacementPlan::new();
        plan.node_placement.insert(0, Device::cpu(0));
        plan.node_placement.insert(1, Device::gpu(0));
        plan.node_placement.insert(2, Device::gpu(1));

        let devices = plan.devices();
        assert!(devices.len() >= 2); // At least CPU and GPU
    }

    #[test]
    fn test_single_device_placement() {
        let graph = create_test_graph();
        let optimizer = PlacementOptimizer::single_device(Device::cpu(0));
        let plan = optimizer.place(&graph);

        assert_eq!(plan.node_placement.len(), 2);
        assert_eq!(plan.count_transfers(&graph), 0); // No transfers on single device
    }

    #[test]
    fn test_round_robin_placement() {
        let graph = create_test_graph();
        let devices = vec![Device::cpu(0), Device::cpu(1)];
        let optimizer = PlacementOptimizer::new(PlacementStrategy::RoundRobin, devices);
        let plan = optimizer.place(&graph);

        assert_eq!(plan.node_placement.len(), 2);
        // Different nodes should be on different devices
        let dev0 = plan.get_node_device(0);
        let dev1 = plan.get_node_device(1);
        assert_ne!(dev0, dev1);
    }

    #[test]
    fn test_cost_based_placement() {
        let graph = create_test_graph();
        let devices = vec![Device::cpu(0), Device::gpu(0)];
        let optimizer = PlacementOptimizer::new(PlacementStrategy::CostBased, devices);
        let plan = optimizer.place(&graph);

        assert_eq!(plan.node_placement.len(), 2);
        assert!(plan.transfer_cost >= 0.0);
    }

    #[test]
    fn test_minimize_transfer_placement() {
        let graph = create_test_graph();
        let devices = vec![Device::cpu(0), Device::cpu(1)];
        let optimizer = PlacementOptimizer::new(PlacementStrategy::MinimizeTransfer, devices);
        let plan = optimizer.place(&graph);

        assert_eq!(plan.node_placement.len(), 2);
        // Should minimize transfers
        let single_device_plan = PlacementOptimizer::single_device(Device::cpu(0)).place(&graph);
        assert!(plan.count_transfers(&graph) <= single_device_plan.count_transfers(&graph) + 2);
    }

    #[test]
    fn test_transfer_counting() {
        let graph = create_test_graph();

        // Single device: no transfers
        let plan1 = PlacementPlan::single_device(2, 4, Device::cpu(0));
        assert_eq!(plan1.count_transfers(&graph), 0);

        // Different devices: some transfers
        let mut plan2 = PlacementPlan::new();
        plan2.node_placement.insert(0, Device::cpu(0));
        plan2.node_placement.insert(1, Device::gpu(0));
        plan2.tensor_placement.insert(0, Device::cpu(0));
        plan2.tensor_placement.insert(1, Device::cpu(0));
        plan2.tensor_placement.insert(2, Device::cpu(0)); // Output of node 0
        plan2.tensor_placement.insert(3, Device::gpu(0)); // Output of node 1

        let transfers = plan2.count_transfers(&graph);
        assert!(transfers > 0); // Should have at least one transfer
    }

    #[test]
    fn test_placement_strategies() {
        let strategies = vec![
            PlacementStrategy::SingleDevice,
            PlacementStrategy::RoundRobin,
            PlacementStrategy::CostBased,
            PlacementStrategy::MinimizeTransfer,
        ];

        let graph = create_test_graph();
        let devices = vec![Device::cpu(0), Device::cpu(1)];

        for strategy in strategies {
            let optimizer = PlacementOptimizer::new(strategy, devices.clone());
            let plan = optimizer.place(&graph);
            assert!(
                !plan.node_placement.is_empty(),
                "Strategy {:?} failed",
                strategy
            );
        }
    }

    #[test]
    fn test_placement_summary() {
        let plan = PlacementPlan::single_device(5, 3, Device::cpu(0));
        let summary = plan.summary();

        assert!(summary.contains("Placement Plan"));
        assert!(summary.contains("Nodes: 5"));
        assert!(summary.contains("Tensors: 3"));
    }
}
