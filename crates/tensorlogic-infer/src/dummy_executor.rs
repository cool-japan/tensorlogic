//! Dummy executor implementation for testing.

use std::collections::HashMap;

use tensorlogic_ir::{EinsumGraph, OpType};

use crate::batch::{BatchResult, TlBatchExecutor};
use crate::capabilities::{BackendCapabilities, DType, DeviceType, Feature, TlCapabilities};
use crate::dummy_tensor::DummyTensor;
use crate::error::ExecutorError;
use crate::ops::{ElemOp, ReduceOp};
use crate::profiling::{Profiler, TlProfiledExecutor};
use crate::traits::{TlAutodiff, TlExecutor};

/// Minimal executor implementation for testing and prototyping.
///
/// This provides a simple, reference implementation that verifies
/// the execution logic without requiring heavy dependencies.
pub struct DummyExecutor {
    pub tensors: HashMap<String, DummyTensor>,
    capabilities: BackendCapabilities,
    profiler: Option<Profiler>,
}

impl DummyExecutor {
    pub fn new() -> Self {
        let capabilities = BackendCapabilities::new("DummyExecutor", "0.1.0")
            .with_device(DeviceType::CPU)
            .with_dtype(DType::F64)
            .with_feature(Feature::Autodiff)
            .with_feature(Feature::BatchExecution)
            .with_max_dims(16);

        DummyExecutor {
            tensors: HashMap::new(),
            capabilities,
            profiler: None,
        }
    }

    pub fn add_tensor(&mut self, name: impl Into<String>, tensor: DummyTensor) {
        self.tensors.insert(name.into(), tensor);
    }

    pub fn get_tensor(&self, name: &str) -> Option<&DummyTensor> {
        self.tensors.get(name)
    }
}

impl Default for DummyExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl TlExecutor for DummyExecutor {
    type Tensor = DummyTensor;
    type Error = ExecutorError;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error> {
        if inputs.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "No input tensors".to_string(),
            ));
        }

        // Simple stub: just return a tensor with the same shape as the first input
        let output_shape = inputs[0].shape.clone();
        let output_size: usize = output_shape.iter().product();

        let result_data = vec![1.0; output_size];

        Ok(DummyTensor {
            name: format!("einsum({})", spec),
            shape: output_shape,
            data: result_data,
        })
    }

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        // Check if this is actually a unary operation
        match op {
            ElemOp::Relu | ElemOp::Sigmoid | ElemOp::OneMinus => {}
            _ => {
                return Err(ExecutorError::UnsupportedOperation(format!(
                    "Operation {:?} is not a unary operation",
                    op
                )))
            }
        }

        let result_data: Vec<f64> = x
            .data
            .iter()
            .map(|&val| match op {
                ElemOp::Relu => val.max(0.0),
                ElemOp::Sigmoid => 1.0 / (1.0 + (-val).exp()),
                ElemOp::OneMinus => 1.0 - val,
                _ => unreachable!(),
            })
            .collect();

        Ok(DummyTensor {
            name: format!("{:?}({})", op, x.name),
            shape: x.shape.clone(),
            data: result_data,
        })
    }

    fn elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        if x.shape != y.shape {
            return Err(ExecutorError::ShapeMismatch(format!(
                "{:?} vs {:?}",
                x.shape, y.shape
            )));
        }

        let result_data: Vec<f64> = x
            .data
            .iter()
            .zip(y.data.iter())
            .map(|(&a, &b)| match op {
                // Arithmetic operations
                ElemOp::Add => a + b,
                ElemOp::Subtract => a - b,
                ElemOp::Multiply => a * b,
                ElemOp::Divide => {
                    if b.abs() < 1e-10 {
                        0.0 // Avoid division by zero
                    } else {
                        a / b
                    }
                }
                ElemOp::Min => a.min(b),
                ElemOp::Max => a.max(b),

                // Comparison operations (return 0.0 or 1.0)
                ElemOp::Eq => {
                    if (a - b).abs() < 1e-10 {
                        1.0
                    } else {
                        0.0
                    }
                }
                ElemOp::Lt => {
                    if a < b {
                        1.0
                    } else {
                        0.0
                    }
                }
                ElemOp::Gt => {
                    if a > b {
                        1.0
                    } else {
                        0.0
                    }
                }
                ElemOp::Lte => {
                    if a <= b {
                        1.0
                    } else {
                        0.0
                    }
                }
                ElemOp::Gte => {
                    if a >= b {
                        1.0
                    } else {
                        0.0
                    }
                }

                // Extended logical operations
                ElemOp::OrMax => a.max(b),
                ElemOp::OrProbSum => a + b - a * b, // Probabilistic sum: 1 - (1-a)(1-b)
                ElemOp::Nand => 1.0 - (a * b),
                ElemOp::Nor => 1.0 - a.max(b),
                ElemOp::Xor => (a - b).abs(), // Soft XOR: |a - b|

                // Unary operations shouldn't be called on binary
                ElemOp::Relu | ElemOp::Sigmoid | ElemOp::OneMinus => {
                    unreachable!("Unary operation {:?} called on binary", op)
                }
            })
            .collect();

        Ok(DummyTensor {
            name: format!("{:?}({},{})", op, x.name, y.name),
            shape: x.shape.clone(),
            data: result_data,
        })
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &Self::Tensor,
        axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error> {
        if axes.is_empty() {
            return Ok(x.clone());
        }

        let rank = x.shape.len();
        let mut output_shape = x.shape.clone();
        for &axis in axes.iter().rev() {
            if axis >= rank {
                return Err(ExecutorError::InvalidAxis { axis, rank });
            }
            output_shape.remove(axis);
        }

        let output_size: usize = if output_shape.is_empty() {
            1
        } else {
            output_shape.iter().product()
        };

        let result_data = match op {
            ReduceOp::Sum => vec![x.data.iter().sum::<f64>(); output_size],
            ReduceOp::Max => {
                vec![x.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)); output_size]
            }
            ReduceOp::Min => vec![x.data.iter().fold(f64::INFINITY, |a, &b| a.min(b)); output_size],
            ReduceOp::Mean => vec![x.data.iter().sum::<f64>() / x.size() as f64; output_size],
            ReduceOp::Product => vec![x.data.iter().product::<f64>(); output_size],
        };

        Ok(DummyTensor {
            name: format!("{:?}({},axes={:?})", op, x.name, axes),
            shape: if output_shape.is_empty() {
                vec![1]
            } else {
                output_shape
            },
            data: result_data,
        })
    }
}

// TlCapabilities implementation
impl TlCapabilities for DummyExecutor {
    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn supports_elem_op(&self, _op: ElemOp) -> bool {
        true // DummyExecutor supports all element ops
    }

    fn supports_reduce_op(&self, _op: ReduceOp) -> bool {
        true // DummyExecutor supports all reduce ops
    }

    fn supports_einsum(&self, _spec: &str) -> bool {
        true // DummyExecutor has basic einsum support
    }
}

// TlProfiledExecutor implementation
impl TlProfiledExecutor for DummyExecutor {
    fn profiler(&self) -> Option<&Profiler> {
        self.profiler.as_ref()
    }

    fn profiler_mut(&mut self) -> Option<&mut Profiler> {
        self.profiler.as_mut()
    }

    fn enable_profiling(&mut self) {
        let mut profiler = Profiler::new();
        profiler.start();
        self.profiler = Some(profiler);
    }

    fn disable_profiling(&mut self) {
        if let Some(mut profiler) = self.profiler.take() {
            profiler.stop();
        }
    }
}

// TlBatchExecutor implementation
impl TlBatchExecutor for DummyExecutor {
    type Tensor = DummyTensor;
    type Error = ExecutorError;

    fn execute_batch(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<Vec<Self::Tensor>>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error> {
        if batch_inputs.is_empty() {
            return Err(ExecutorError::EmptyInput(
                "Batch inputs cannot be empty".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(batch_inputs.len());
        for inputs in batch_inputs {
            let output = self.execute_graph_internal(graph, &inputs)?;
            outputs.push(output);
        }

        Ok(BatchResult::new(outputs))
    }

    fn execute_batch_parallel(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<Vec<Self::Tensor>>,
        _num_threads: Option<usize>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error> {
        // DummyExecutor doesn't support true parallel execution
        // Fall back to sequential execution
        self.execute_batch(graph, batch_inputs)
    }

    fn optimal_batch_size(&self) -> usize {
        16 // Conservative batch size for dummy executor
    }
}

// TlAutodiff implementation
impl TlAutodiff for DummyExecutor {
    type Tape = HashMap<usize, DummyTensor>;

    fn forward(&mut self, graph: &EinsumGraph) -> Result<Self::Tensor, Self::Error> {
        if graph.nodes.is_empty() {
            return Err(ExecutorError::EmptyInput(
                "Graph has no nodes to execute".to_string(),
            ));
        }

        // Execute the graph and return the last tensor
        let mut tensors: HashMap<usize, DummyTensor> = HashMap::new();

        // Initialize input tensors (first N tensors in the graph)
        // Note: In a real implementation, these would be provided as inputs
        for (idx, tensor_name) in graph.tensors.iter().enumerate() {
            // Create dummy tensors with default shape [10]
            tensors.insert(idx, DummyTensor::ones(tensor_name.clone(), vec![10]));
        }

        // Execute each node
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            let output_idx = graph.tensors.len() + node_idx;
            let output = self.execute_node_internal(node, &tensors)?;
            tensors.insert(output_idx, output);
        }

        // Return the last computed tensor (or from outputs if specified)
        let output_idx = if graph.outputs.is_empty() {
            graph.tensors.len() + graph.nodes.len() - 1
        } else {
            graph.outputs[0]
        };

        tensors
            .remove(&output_idx)
            .ok_or_else(|| ExecutorError::TensorNotFound("Output tensor".to_string()))
    }

    fn backward(
        &mut self,
        graph: &EinsumGraph,
        _loss: &Self::Tensor,
    ) -> Result<Self::Tape, Self::Error> {
        // Simplified backward pass: just return unit gradients for all tensors
        let mut gradients = HashMap::new();

        for (idx, tensor_name) in graph.tensors.iter().enumerate() {
            gradients.insert(
                idx,
                DummyTensor::ones(format!("grad_{}", tensor_name), vec![10]),
            );
        }

        Ok(gradients)
    }
}

// Helper methods for DummyExecutor
impl DummyExecutor {
    fn execute_graph_internal(
        &mut self,
        graph: &EinsumGraph,
        _inputs: &[DummyTensor],
    ) -> Result<DummyTensor, ExecutorError> {
        // Simplified: just execute forward pass
        self.forward(graph)
    }

    fn execute_node_internal(
        &mut self,
        node: &tensorlogic_ir::EinsumNode,
        tensors: &HashMap<usize, DummyTensor>,
    ) -> Result<DummyTensor, ExecutorError> {
        match &node.op {
            OpType::Einsum { spec } => {
                let inputs: Vec<DummyTensor> =
                    node.inputs
                        .iter()
                        .map(|&idx| {
                            tensors.get(&idx).cloned().ok_or_else(|| {
                                ExecutorError::TensorNotFound(format!("Tensor {}", idx))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                self.einsum(spec, &inputs)
            }
            OpType::ElemUnary { op } => {
                if node.inputs.is_empty() {
                    return Err(ExecutorError::EmptyInput(
                        "ElemUnary requires an input".to_string(),
                    ));
                }
                let input = tensors.get(&node.inputs[0]).ok_or_else(|| {
                    ExecutorError::TensorNotFound(format!("Tensor {}", node.inputs[0]))
                })?;
                let elem_op = Self::parse_elem_op(op)?;
                self.elem_op(elem_op, input)
            }
            OpType::ElemBinary { op } => {
                if node.inputs.len() < 2 {
                    return Err(ExecutorError::EmptyInput(
                        "ElemBinary requires two inputs".to_string(),
                    ));
                }
                let input1 = tensors.get(&node.inputs[0]).ok_or_else(|| {
                    ExecutorError::TensorNotFound(format!("Tensor {}", node.inputs[0]))
                })?;
                let input2 = tensors.get(&node.inputs[1]).ok_or_else(|| {
                    ExecutorError::TensorNotFound(format!("Tensor {}", node.inputs[1]))
                })?;
                let elem_op = Self::parse_elem_op(op)?;
                self.elem_op_binary(elem_op, input1, input2)
            }
            OpType::Reduce { op, axes } => {
                if node.inputs.is_empty() {
                    return Err(ExecutorError::EmptyInput(
                        "Reduce requires an input".to_string(),
                    ));
                }
                let input = tensors.get(&node.inputs[0]).ok_or_else(|| {
                    ExecutorError::TensorNotFound(format!("Tensor {}", node.inputs[0]))
                })?;
                let reduce_op = Self::parse_reduce_op(op)?;
                self.reduce(reduce_op, input, axes)
            }
        }
    }

    fn parse_elem_op(op_str: &str) -> Result<ElemOp, ExecutorError> {
        match op_str.to_lowercase().as_str() {
            "relu" => Ok(ElemOp::Relu),
            "sigmoid" => Ok(ElemOp::Sigmoid),
            "oneminus" | "one_minus" => Ok(ElemOp::OneMinus),
            "add" => Ok(ElemOp::Add),
            "subtract" | "sub" => Ok(ElemOp::Subtract),
            "multiply" | "mul" => Ok(ElemOp::Multiply),
            "divide" | "div" => Ok(ElemOp::Divide),
            "eq" | "equal" => Ok(ElemOp::Eq),
            "lt" | "less" => Ok(ElemOp::Lt),
            "gt" | "greater" => Ok(ElemOp::Gt),
            "lte" | "le" => Ok(ElemOp::Lte),
            "gte" | "ge" => Ok(ElemOp::Gte),
            "ormax" | "or_max" => Ok(ElemOp::OrMax),
            "orprobsum" | "or_prob_sum" => Ok(ElemOp::OrProbSum),
            "nand" => Ok(ElemOp::Nand),
            "nor" => Ok(ElemOp::Nor),
            "xor" => Ok(ElemOp::Xor),
            _ => Err(ExecutorError::UnsupportedOperation(format!(
                "Unknown element operation: {}",
                op_str
            ))),
        }
    }

    fn parse_reduce_op(op_str: &str) -> Result<ReduceOp, ExecutorError> {
        match op_str.to_lowercase().as_str() {
            "sum" => Ok(ReduceOp::Sum),
            "max" => Ok(ReduceOp::Max),
            "min" => Ok(ReduceOp::Min),
            "mean" => Ok(ReduceOp::Mean),
            "product" | "prod" => Ok(ReduceOp::Product),
            _ => Err(ExecutorError::UnsupportedOperation(format!(
                "Unknown reduce operation: {}",
                op_str
            ))),
        }
    }
}
