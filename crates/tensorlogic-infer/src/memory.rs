//! Memory estimation utilities for execution planning.

use tensorlogic_ir::{EinsumGraph, OpType};

use crate::capabilities::DType;

/// Memory usage estimate for a tensor
#[derive(Debug, Clone)]
pub struct TensorMemory {
    pub tensor_idx: usize,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub bytes: usize,
}

impl TensorMemory {
    pub fn new(tensor_idx: usize, shape: Vec<usize>, dtype: DType) -> Self {
        let element_count: usize = shape.iter().product();
        let bytes = element_count * dtype.byte_size();

        TensorMemory {
            tensor_idx,
            shape,
            element_count,
            bytes,
        }
    }

    pub fn megabytes(&self) -> f64 {
        self.bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Complete memory profile for graph execution
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub input_memory: Vec<TensorMemory>,
    pub intermediate_memory: Vec<TensorMemory>,
    pub output_memory: Vec<TensorMemory>,
    pub total_bytes: usize,
    pub peak_bytes: usize,
}

impl MemoryEstimate {
    pub fn new() -> Self {
        MemoryEstimate {
            input_memory: Vec::new(),
            intermediate_memory: Vec::new(),
            output_memory: Vec::new(),
            total_bytes: 0,
            peak_bytes: 0,
        }
    }

    pub fn total_megabytes(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn peak_megabytes(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn summary(&self) -> String {
        format!(
            "Memory Estimate:\n\
             - Inputs: {} tensors, {:.2} MB\n\
             - Intermediates: {} tensors, {:.2} MB\n\
             - Outputs: {} tensors, {:.2} MB\n\
             - Total: {:.2} MB\n\
             - Peak: {:.2} MB",
            self.input_memory.len(),
            self.input_memory.iter().map(|t| t.megabytes()).sum::<f64>(),
            self.intermediate_memory.len(),
            self.intermediate_memory
                .iter()
                .map(|t| t.megabytes())
                .sum::<f64>(),
            self.output_memory.len(),
            self.output_memory
                .iter()
                .map(|t| t.megabytes())
                .sum::<f64>(),
            self.total_megabytes(),
            self.peak_megabytes()
        )
    }
}

impl Default for MemoryEstimate {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory estimator for execution graphs
pub struct MemoryEstimator {
    dtype: DType,
}

impl MemoryEstimator {
    pub fn new(dtype: DType) -> Self {
        MemoryEstimator { dtype }
    }

    /// Estimate memory usage for a graph
    /// Note: Uses default shape [10] for all tensors since graph only stores names
    pub fn estimate(&self, graph: &EinsumGraph) -> MemoryEstimate {
        let mut estimate = MemoryEstimate::new();
        let default_shape = vec![10]; // Default shape for estimation

        // Estimate input tensors
        for idx in 0..graph.tensors.len() {
            let mem = TensorMemory::new(idx, default_shape.clone(), self.dtype);
            estimate.total_bytes += mem.bytes;
            estimate.input_memory.push(mem);
        }

        // Estimate intermediate/output tensors from nodes
        let num_inputs = graph.tensors.len();
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            let tensor_idx = num_inputs + node_idx;
            let shape = self.estimate_output_shape(node, graph);

            let mem = TensorMemory::new(tensor_idx, shape, self.dtype);
            estimate.total_bytes += mem.bytes;

            // Last node is output, others are intermediates
            if node_idx == graph.nodes.len() - 1 {
                estimate.output_memory.push(mem);
            } else {
                estimate.intermediate_memory.push(mem);
            }
        }

        // Peak memory is when all tensors are alive
        // (simplified - doesn't account for tensor lifetime)
        estimate.peak_bytes = estimate.total_bytes;

        estimate
    }

    /// Estimate memory with tensor lifetime analysis for peak usage
    pub fn estimate_with_lifetime(&self, graph: &EinsumGraph) -> MemoryEstimate {
        let mut estimate = self.estimate(graph);

        // Track which tensors are alive at each point
        let num_tensors = graph.tensors.len() + graph.nodes.len();
        let mut alive = vec![false; num_tensors];

        // Input tensors are initially alive
        for item in alive.iter_mut().take(graph.tensors.len()) {
            *item = true;
        }

        let mut peak_bytes = 0;

        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Mark output tensor as alive
            let output_idx = graph.tensors.len() + node_idx;
            alive[output_idx] = true;

            // Calculate current memory usage
            let current_bytes: usize = alive
                .iter()
                .enumerate()
                .filter(|(_, &is_alive)| is_alive)
                .map(|(idx, _)| {
                    if idx < graph.tensors.len() {
                        // Input tensor
                        &estimate.input_memory[idx]
                    } else {
                        // Intermediate/output tensor
                        let node_offset = idx - graph.tensors.len();
                        if node_offset < estimate.intermediate_memory.len() {
                            &estimate.intermediate_memory[node_offset]
                        } else {
                            &estimate.output_memory[0]
                        }
                    }
                })
                .map(|mem| mem.bytes)
                .sum();

            peak_bytes = peak_bytes.max(current_bytes);

            // Mark input tensors as dead if no longer needed
            // (simplified: assume each tensor is used only once)
            for &input_idx in &node.inputs {
                if self.is_last_use(input_idx, node_idx, graph) {
                    alive[input_idx] = false;
                }
            }
        }

        estimate.peak_bytes = peak_bytes;
        estimate
    }

    fn estimate_output_shape(
        &self,
        node: &tensorlogic_ir::EinsumNode,
        _graph: &EinsumGraph,
    ) -> Vec<usize> {
        // Since graph only stores tensor names, we use default shapes for estimation
        match &node.op {
            OpType::Einsum { spec } => {
                // Simplified: parse einsum spec to estimate shape
                if let Some(arrow_pos) = spec.find("->") {
                    let output_axes = &spec[arrow_pos + 2..];
                    // Estimate each dimension as 10 (placeholder)
                    vec![10; output_axes.len()]
                } else {
                    // Default shape
                    vec![10]
                }
            }
            OpType::ElemUnary { op: _ } | OpType::ElemBinary { op: _ } => {
                // Shape preserved for element-wise ops - use default
                vec![10]
            }
            OpType::Reduce { op: _, axes } => {
                // Remove reduced axes from default shape
                let default_shape = vec![10, 10]; // Assume 2D default
                let mut shape = default_shape.clone();
                for &axis in axes.iter().rev() {
                    if axis < shape.len() {
                        shape.remove(axis);
                    }
                }
                if shape.is_empty() {
                    vec![1]
                } else {
                    shape
                }
            }
        }
    }

    fn is_last_use(&self, tensor_idx: usize, current_node: usize, graph: &EinsumGraph) -> bool {
        // Check if any later nodes use this tensor
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            if node_idx > current_node && node.inputs.contains(&tensor_idx) {
                return false;
            }
        }
        true
    }

    /// Estimate memory for a batch of inputs
    pub fn estimate_batch(&self, graph: &EinsumGraph, batch_size: usize) -> MemoryEstimate {
        let single_estimate = self.estimate(graph);

        let mut batch_estimate = MemoryEstimate::new();
        batch_estimate.total_bytes = single_estimate.total_bytes * batch_size;
        batch_estimate.peak_bytes = single_estimate.peak_bytes * batch_size;

        // Scale all tensors by batch size
        for input in &single_estimate.input_memory {
            let mut batched = input.clone();
            batched.bytes *= batch_size;
            batch_estimate.input_memory.push(batched);
        }

        for intermediate in &single_estimate.intermediate_memory {
            let mut batched = intermediate.clone();
            batched.bytes *= batch_size;
            batch_estimate.intermediate_memory.push(batched);
        }

        for output in &single_estimate.output_memory {
            let mut batched = output.clone();
            batched.bytes *= batch_size;
            batch_estimate.output_memory.push(batched);
        }

        batch_estimate
    }
}

impl Default for MemoryEstimator {
    fn default() -> Self {
        Self::new(DType::F64)
    }
}
