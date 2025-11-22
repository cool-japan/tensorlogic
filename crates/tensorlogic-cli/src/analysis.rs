//! Graph analysis and metrics for TensorLogic CLI

use std::collections::{HashMap, HashSet};
use tensorlogic_ir::{EinsumGraph, OpType};

/// Graph complexity metrics
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    /// Total number of tensors
    pub tensor_count: usize,
    /// Total number of nodes
    pub node_count: usize,
    /// Number of input tensors
    pub input_count: usize,
    /// Number of output tensors
    pub output_count: usize,
    /// Operation type breakdown
    pub op_breakdown: HashMap<String, usize>,
    /// Graph depth (longest path)
    pub depth: usize,
    /// Average fanout (outputs per node)
    pub avg_fanout: f64,
    /// Estimated computational complexity (FLOPs)
    pub estimated_flops: u64,
    /// Estimated memory usage (bytes)
    pub estimated_memory: u64,
}

impl GraphMetrics {
    /// Analyze an einsum graph
    pub fn analyze(graph: &EinsumGraph) -> Self {
        let tensor_count = graph.tensors.len();
        let node_count = graph.nodes.len();
        let input_count = graph.inputs.len();
        let output_count = graph.outputs.len();

        // Operation breakdown
        let mut op_breakdown = HashMap::new();
        for node in &graph.nodes {
            let op_name = match &node.op {
                OpType::Einsum { .. } => "Einsum",
                OpType::ElemUnary { .. } => "ElemUnary",
                OpType::ElemBinary { .. } => "ElemBinary",
                OpType::Reduce { .. } => "Reduce",
            };
            *op_breakdown.entry(op_name.to_string()).or_insert(0) += 1;
        }

        // Calculate depth
        let depth = calculate_depth(graph);

        // Calculate average fanout
        let total_outputs: usize = graph.nodes.iter().map(|n| n.outputs.len()).sum();
        let avg_fanout = if node_count > 0 {
            total_outputs as f64 / node_count as f64
        } else {
            0.0
        };

        // Estimate FLOPs and memory
        let estimated_flops = estimate_flops(graph);
        let estimated_memory = estimate_memory(graph);

        Self {
            tensor_count,
            node_count,
            input_count,
            output_count,
            op_breakdown,
            depth,
            avg_fanout,
            estimated_flops,
            estimated_memory,
        }
    }

    /// Print metrics in human-readable format
    pub fn print(&self) {
        println!("Graph Metrics:");
        println!("  Tensors: {}", self.tensor_count);
        println!("  Nodes: {}", self.node_count);
        println!("  Inputs: {}", self.input_count);
        println!("  Outputs: {}", self.output_count);
        println!("  Depth: {}", self.depth);
        println!("  Avg Fanout: {:.2}", self.avg_fanout);
        println!("\nOperation Breakdown:");
        for (op, count) in &self.op_breakdown {
            println!("  {}: {}", op, count);
        }
        println!("\nEstimates:");
        println!("  FLOPs: {}", format_number(self.estimated_flops));
        println!("  Memory: {}", format_bytes(self.estimated_memory));
    }
}

fn calculate_depth(graph: &EinsumGraph) -> usize {
    let mut depths = HashMap::new();

    // Initialize input tensors with depth 0
    for input_id in &graph.inputs {
        depths.insert(*input_id, 0);
    }

    // Topologically process nodes
    let mut processed = HashSet::new();
    let mut changed = true;

    while changed {
        changed = false;
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            if processed.contains(&node_idx) {
                continue;
            }

            // Check if all inputs are processed
            let all_inputs_ready = node
                .inputs
                .iter()
                .all(|input_id| depths.contains_key(input_id));

            if all_inputs_ready {
                // Calculate depth as max(input depths) + 1
                let max_input_depth = node
                    .inputs
                    .iter()
                    .map(|id| *depths.get(id).unwrap_or(&0))
                    .max()
                    .unwrap_or(0);

                let node_depth = max_input_depth + 1;

                // Set depth for all output tensors
                for output_id in &node.outputs {
                    depths.insert(*output_id, node_depth);
                }

                processed.insert(node_idx);
                changed = true;
            }
        }
    }

    // Return maximum depth
    *depths.values().max().unwrap_or(&0)
}

fn estimate_flops(graph: &EinsumGraph) -> u64 {
    let mut total_flops = 0u64;

    for node in &graph.nodes {
        let flops = match &node.op {
            OpType::Einsum { .. } => {
                // Rough estimate: 2 FLOPs per element (multiply-add)
                // Assume 1000 elements per tensor (very rough)
                2000
            }
            OpType::ElemUnary { .. } => {
                // 1 FLOP per element
                1000
            }
            OpType::ElemBinary { .. } => {
                // 1 FLOP per element
                1000
            }
            OpType::Reduce { .. } => {
                // Sum reduction: n-1 additions
                999
            }
        };
        total_flops += flops;
    }

    total_flops
}

fn estimate_memory(graph: &EinsumGraph) -> u64 {
    // Assume f64 (8 bytes) and 1000 elements per tensor
    let bytes_per_tensor = 8 * 1000;
    (graph.tensors.len() as u64) * bytes_per_tensor
}

pub fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1500), "1.50K");
        assert_eq!(format_number(1500000), "1.50M");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 bytes");
        assert_eq!(format_bytes(2048), "2.00 KB");
        assert_eq!(format_bytes(2 * 1024 * 1024), "2.00 MB");
    }
}
