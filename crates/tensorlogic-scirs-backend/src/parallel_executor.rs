//! Parallel executor implementation using Rayon for multi-threaded execution.
//!
//! This module provides a parallel implementation of the TensorLogic executor
//! that can execute independent operations concurrently using thread pools.
//!
//! ## Key Features
//!
//! - **Level-by-level execution**: Operations are grouped by execution level
//! - **Rayon thread pools**: Configurable thread pool for parallel execution
//! - **Automatic dependency handling**: Uses DependencyAnalysis for safe parallelization
//! - **Performance monitoring**: Tracks parallel vs sequential execution times
//!
//! ## Example
//!
//! ```rust,ignore
//! use tensorlogic_scirs_backend::ParallelScirs2Exec;
//! use tensorlogic_infer::TlAutodiff;
//!
//! let mut executor = ParallelScirs2Exec::new();
//! executor.set_num_threads(4); // Use 4 threads
//!
//! let result = executor.forward(&graph)?;
//! ```

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

#[cfg(feature = "parallel")]
use std::sync::{Arc, Mutex};
use tensorlogic_infer::{ElemOp, ExecutorError, ReduceOp, TlAutodiff, TlExecutor};
#[cfg(not(feature = "parallel"))]
use tensorlogic_ir::EinsumGraph;
#[cfg(feature = "parallel")]
use tensorlogic_ir::{EinsumGraph, OpType};

use crate::autodiff::ForwardTape;
#[cfg(feature = "parallel")]
use crate::dependency_analyzer::DependencyAnalysis;
#[cfg(feature = "parallel")]
use crate::ops::{parse_elem_op, parse_reduce_op};
use crate::Scirs2Tensor;

/// Configuration for parallel execution.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use (None = use all available cores)
    pub num_threads: Option<usize>,
    /// Minimum number of operations per level to enable parallelization
    /// (levels with fewer ops run sequentially to avoid overhead)
    pub min_parallel_ops: usize,
    /// Enable memory pooling for tensor reuse
    pub enable_pooling: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Use all available cores
            min_parallel_ops: 2,
            enable_pooling: true,
        }
    }
}

/// Statistics about parallel execution.
#[derive(Debug, Clone)]
pub struct ParallelStats {
    /// Number of execution levels
    pub num_levels: usize,
    /// Number of operations executed in parallel
    pub parallel_ops: usize,
    /// Number of operations executed sequentially
    pub sequential_ops: usize,
    /// Maximum number of concurrent operations in any level
    pub max_parallelism: usize,
    /// Estimated speedup from parallelization
    pub estimated_speedup: f64,
}

/// Parallel executor using Rayon for multi-threaded execution.
pub struct ParallelScirs2Exec {
    /// Base executor for sequential operations
    pub(crate) base: crate::executor::Scirs2Exec,
    /// Configuration for parallel execution
    pub config: ParallelConfig,
    /// Statistics from last execution
    pub stats: Option<ParallelStats>,
}

impl ParallelScirs2Exec {
    /// Create a new parallel executor with default configuration.
    pub fn new() -> Self {
        Self {
            base: crate::executor::Scirs2Exec::new(),
            config: ParallelConfig::default(),
            stats: None,
        }
    }

    /// Create a parallel executor with custom configuration.
    pub fn with_config(config: ParallelConfig) -> Self {
        let base = if config.enable_pooling {
            crate::executor::Scirs2Exec::with_memory_pool()
        } else {
            crate::executor::Scirs2Exec::new()
        };

        Self {
            base,
            config,
            stats: None,
        }
    }

    /// Set the number of threads to use.
    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.config.num_threads = Some(num_threads);
    }

    /// Get the number of threads configured (returns actual thread count).
    #[cfg(feature = "parallel")]
    pub fn num_threads(&self) -> usize {
        self.config.num_threads.unwrap_or_else(current_num_threads)
    }

    #[cfg(not(feature = "parallel"))]
    pub fn num_threads(&self) -> usize {
        self.config.num_threads.unwrap_or(1)
    }

    /// Enable or disable memory pooling.
    pub fn set_pooling(&mut self, enable: bool) {
        self.config.enable_pooling = enable;
        if enable {
            self.base.enable_pooling();
        } else {
            self.base.disable_pooling();
        }
    }

    /// Get pool statistics if pooling is enabled.
    pub fn pool_stats(&self) -> Option<crate::memory_pool::PoolStats> {
        self.base.pool_stats()
    }

    /// Get statistics from the last execution.
    pub fn execution_stats(&self) -> Option<&ParallelStats> {
        self.stats.as_ref()
    }

    /// Add a named tensor to the executor.
    pub fn add_tensor(&mut self, name: impl Into<String>, tensor: Scirs2Tensor) {
        self.base.add_tensor(name, tensor);
    }

    /// Get a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&Scirs2Tensor> {
        self.base.get_tensor(name)
    }

    /// Execute a single operation (helper function).
    #[cfg(feature = "parallel")]
    fn execute_operation(
        &self,
        node: &tensorlogic_ir::EinsumNode,
        input_tensors: &[Scirs2Tensor],
    ) -> Result<Scirs2Tensor, ExecutorError> {
        // Dispatch based on operation type
        match &node.op {
            OpType::Einsum { spec } => {
                // Need to use a mutable executor for einsum
                // For now, we'll use the sequential path through self.base
                // In a real parallel implementation, we'd need to handle this differently
                let views: Vec<_> = input_tensors.iter().map(|t| t.view()).collect();
                let view_refs: Vec<_> = views.iter().collect();
                scirs2_linalg::einsum(spec, &view_refs)
                    .map_err(|e| ExecutorError::InvalidEinsumSpec(format!("Einsum error: {}", e)))
            }
            OpType::ElemUnary { op } => {
                if input_tensors.len() != 1 {
                    return Err(ExecutorError::InvalidEinsumSpec(format!(
                        "Unary operation requires 1 input, got {}",
                        input_tensors.len()
                    )));
                }
                let elem_op = parse_elem_op(op)?;
                match elem_op {
                    ElemOp::Relu => Ok(input_tensors[0].mapv(|v| v.max(0.0))),
                    ElemOp::Sigmoid => Ok(input_tensors[0].mapv(|v| 1.0 / (1.0 + (-v).exp()))),
                    ElemOp::OneMinus => Ok(input_tensors[0].mapv(|v| 1.0 - v)),
                    _ => Err(ExecutorError::UnsupportedOperation(format!(
                        "Unary operation {:?} not supported",
                        elem_op
                    ))),
                }
            }
            OpType::ElemBinary { op } => {
                if input_tensors.len() != 2 {
                    return Err(ExecutorError::InvalidEinsumSpec(format!(
                        "Binary operation requires 2 inputs, got {}",
                        input_tensors.len()
                    )));
                }
                let elem_op = parse_elem_op(op)?;
                let x = &input_tensors[0];
                let y = &input_tensors[1];

                // Handle scalar broadcasting
                let x_is_scalar = x.ndim() == 0;
                let y_is_scalar = y.ndim() == 0;

                let (x_broadcast, y_broadcast);
                let (x_ref, y_ref) = if x_is_scalar && !y_is_scalar {
                    let scalar_value = x.iter().next().unwrap();
                    x_broadcast =
                        scirs2_core::ndarray::Array::from_elem(y.raw_dim(), *scalar_value);
                    (&x_broadcast.view(), &y.view())
                } else if y_is_scalar && !x_is_scalar {
                    let scalar_value = y.iter().next().unwrap();
                    y_broadcast =
                        scirs2_core::ndarray::Array::from_elem(x.raw_dim(), *scalar_value);
                    (&x.view(), &y_broadcast.view())
                } else if x.shape() != y.shape() {
                    return Err(ExecutorError::ShapeMismatch(format!(
                        "Shape mismatch: {:?} vs {:?}",
                        x.shape(),
                        y.shape()
                    )));
                } else {
                    (&x.view(), &y.view())
                };

                let result = match elem_op {
                    ElemOp::Add => x_ref + y_ref,
                    ElemOp::Subtract => x_ref - y_ref,
                    ElemOp::Multiply => x_ref * y_ref,
                    ElemOp::Divide => x_ref / y_ref,
                    ElemOp::Min => scirs2_core::ndarray::Zip::from(x_ref)
                        .and(y_ref)
                        .map_collect(|&a, &b| a.min(b)),
                    ElemOp::Max => scirs2_core::ndarray::Zip::from(x_ref)
                        .and(y_ref)
                        .map_collect(|&a, &b| a.max(b)),
                    ElemOp::OrMax => scirs2_core::ndarray::Zip::from(x_ref)
                        .and(y_ref)
                        .map_collect(|&a, &b| a.max(b)),
                    ElemOp::OrProbSum => scirs2_core::ndarray::Zip::from(x_ref)
                        .and(y_ref)
                        .map_collect(|&a, &b| a + b - a * b),
                    ElemOp::Nand => scirs2_core::ndarray::Zip::from(x_ref)
                        .and(y_ref)
                        .map_collect(|&a, &b| 1.0 - (a * b)),
                    ElemOp::Nor => scirs2_core::ndarray::Zip::from(x_ref)
                        .and(y_ref)
                        .map_collect(|&a, &b| 1.0 - a.max(b)),
                    ElemOp::Xor => scirs2_core::ndarray::Zip::from(x_ref)
                        .and(y_ref)
                        .map_collect(|&a, &b| a + b - 2.0 * a * b),
                    _ => {
                        return Err(ExecutorError::UnsupportedOperation(format!(
                            "Binary operation {:?} not supported",
                            elem_op
                        )))
                    }
                };
                Ok(result)
            }
            OpType::Reduce { op, axes } => {
                if input_tensors.len() != 1 {
                    return Err(ExecutorError::InvalidEinsumSpec(format!(
                        "Reduce operation requires 1 input, got {}",
                        input_tensors.len()
                    )));
                }
                let reduce_op = parse_reduce_op(op)?;
                let x = &input_tensors[0];

                use scirs2_core::ndarray::Axis;
                let mut result = x.clone();
                for &axis in axes.iter().rev() {
                    result = match reduce_op {
                        ReduceOp::Sum => result.sum_axis(Axis(axis)),
                        ReduceOp::Max => result.map_axis(Axis(axis), |view| {
                            view.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                        }),
                        ReduceOp::Min => result.map_axis(Axis(axis), |view| {
                            view.iter().fold(f64::INFINITY, |a, &b| a.min(b))
                        }),
                        ReduceOp::Mean => {
                            let sum = result.sum_axis(Axis(axis));
                            let count = result.len_of(Axis(axis)) as f64;
                            sum / count
                        }
                        ReduceOp::Product => {
                            result.map_axis(Axis(axis), |view| view.iter().product())
                        }
                    };
                }
                Ok(result)
            }
        }
    }
}

impl Default for ParallelScirs2Exec {
    fn default() -> Self {
        Self::new()
    }
}

// Delegate basic TlExecutor methods to the base executor
impl TlExecutor for ParallelScirs2Exec {
    type Tensor = Scirs2Tensor;
    type Error = ExecutorError;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error> {
        self.base.einsum(spec, inputs)
    }

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        self.base.elem_op(op, x)
    }

    fn elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        self.base.elem_op_binary(op, x, y)
    }

    fn reduce(
        &mut self,
        op: ReduceOp,
        x: &Self::Tensor,
        axes: &[usize],
    ) -> Result<Self::Tensor, Self::Error> {
        self.base.reduce(op, x, axes)
    }
}

#[cfg(feature = "parallel")]
impl TlAutodiff for ParallelScirs2Exec {
    type Tape = ForwardTape;

    fn forward(&mut self, graph: &EinsumGraph) -> Result<Self::Tensor, Self::Error> {
        if graph.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "Empty graph provided".to_string(),
            ));
        }

        if graph.outputs.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "No output tensors specified".to_string(),
            ));
        }

        // Analyze dependencies
        let analysis = DependencyAnalysis::analyze(graph);

        // Initialize tensor storage
        let computed_tensors: Arc<Mutex<Vec<Option<Scirs2Tensor>>>> =
            Arc::new(Mutex::new(vec![None; graph.tensors.len()]));

        let node_inputs: Arc<Mutex<Vec<Vec<Scirs2Tensor>>>> =
            Arc::new(Mutex::new(Vec::with_capacity(graph.nodes.len())));

        // Initialize input tensors from our stored tensors
        {
            let mut storage = computed_tensors.lock().unwrap();
            for (idx, tensor_name) in graph.tensors.iter().enumerate() {
                if let Some(tensor) = self.base.tensors.get(tensor_name) {
                    storage[idx] = Some(tensor.clone());
                } else {
                    // Handle tensors with axes notation (e.g., "age[a]" -> "age")
                    let base_name = tensor_name.split('[').next().unwrap_or(tensor_name);
                    if let Some(tensor) = self.base.tensors.get(base_name) {
                        storage[idx] = Some(tensor.clone());
                    } else if tensor_name.starts_with("const_") || base_name.starts_with("const_") {
                        // Handle constant tensors
                        let const_name = if tensor_name.starts_with("const_") {
                            tensor_name
                        } else {
                            base_name
                        };
                        if let Some(value_str) = const_name.strip_prefix("const_") {
                            if let Ok(value) = value_str.parse::<f64>() {
                                use scirs2_core::ndarray::arr0;
                                storage[idx] = Some(arr0(value).into_dyn());
                            }
                        }
                    }
                }
            }
        }

        // Track statistics
        let mut parallel_ops = 0;
        let mut sequential_ops = 0;

        // Execute operations level by level
        for level_ops in &analysis.execution_levels {
            let should_parallelize = level_ops.len() >= self.config.min_parallel_ops;

            if should_parallelize {
                // Parallel execution for this level
                parallel_ops += level_ops.len();

                // Execute all operations in this level in parallel
                let results: Vec<_> = level_ops
                    .par_iter()
                    .map(|&op_idx| {
                        let node = &graph.nodes[op_idx];

                        // Read inputs from shared storage
                        let inputs: Result<Vec<_>, _> = {
                            let storage = computed_tensors.lock().unwrap();
                            node.inputs
                                .iter()
                                .map(|&idx| {
                                    storage
                                        .get(idx)
                                        .and_then(|t| t.as_ref())
                                        .cloned()
                                        .ok_or_else(|| {
                                            ExecutorError::TensorNotFound(format!(
                                                "Tensor at index {} not found",
                                                idx
                                            ))
                                        })
                                })
                                .collect()
                        };

                        let input_tensors = inputs?;
                        let result = self.execute_operation(node, &input_tensors)?;

                        Ok((op_idx, node.outputs.clone(), input_tensors, result))
                    })
                    .collect::<Result<Vec<_>, ExecutorError>>()?;

                // Store results
                {
                    let mut storage = computed_tensors.lock().unwrap();
                    let mut inputs_vec = node_inputs.lock().unwrap();

                    // Ensure node_inputs has enough capacity
                    while inputs_vec.len()
                        <= results.iter().map(|(idx, _, _, _)| *idx).max().unwrap_or(0)
                    {
                        inputs_vec.push(Vec::new());
                    }

                    for (op_idx, outputs, inputs, tensor) in results {
                        // Store in tensor storage
                        if let Some(&output_idx) = outputs.first() {
                            storage[output_idx] = Some(tensor);
                        }

                        // Store inputs for backward pass
                        inputs_vec[op_idx] = inputs;
                    }
                }
            } else {
                // Sequential execution for this level
                sequential_ops += level_ops.len();

                let mut storage = computed_tensors.lock().unwrap();
                let mut inputs_vec = node_inputs.lock().unwrap();

                for &op_idx in level_ops {
                    let node = &graph.nodes[op_idx];

                    let inputs: Result<Vec<_>, _> = node
                        .inputs
                        .iter()
                        .map(|&idx| {
                            storage
                                .get(idx)
                                .and_then(|t| t.as_ref())
                                .cloned()
                                .ok_or_else(|| {
                                    ExecutorError::TensorNotFound(format!(
                                        "Tensor at index {} not found",
                                        idx
                                    ))
                                })
                        })
                        .collect();

                    let input_tensors = inputs?;
                    let result = self.execute_operation(node, &input_tensors)?;

                    // Store result
                    if let Some(&output_idx) = node.outputs.first() {
                        storage[output_idx] = Some(result);
                    }

                    // Store inputs for backward pass
                    while inputs_vec.len() <= op_idx {
                        inputs_vec.push(Vec::new());
                    }
                    inputs_vec[op_idx] = input_tensors;
                }
            }
        }

        // Store tape for backward pass
        let final_tensors = Arc::try_unwrap(computed_tensors)
            .unwrap()
            .into_inner()
            .unwrap();
        let final_inputs = Arc::try_unwrap(node_inputs).unwrap().into_inner().unwrap();

        self.base.tape = Some(ForwardTape {
            tensors: final_tensors.clone(),
            node_inputs: final_inputs,
        });

        // Store statistics
        self.stats = Some(ParallelStats {
            num_levels: analysis.num_levels,
            parallel_ops,
            sequential_ops,
            max_parallelism: analysis.max_parallelism,
            estimated_speedup: analysis.estimated_speedup(),
        });

        // Return the output tensor
        let output_idx = graph.outputs[0];
        final_tensors
            .get(output_idx)
            .and_then(|t| t.clone())
            .ok_or_else(|| ExecutorError::TensorNotFound("Output tensor not computed".to_string()))
    }

    fn backward(
        &mut self,
        graph: &EinsumGraph,
        loss_grad: &Self::Tensor,
    ) -> Result<Self::Tape, Self::Error> {
        // Use the base executor's backward implementation
        // (backward pass is typically more sequential due to dependency chains)
        self.base.backward(graph, loss_grad)
    }
}

// If parallel feature is not enabled, provide a non-parallel implementation
#[cfg(not(feature = "parallel"))]
impl TlAutodiff for ParallelScirs2Exec {
    type Tape = ForwardTape;

    fn forward(&mut self, graph: &EinsumGraph) -> Result<Self::Tensor, Self::Error> {
        // Fall back to sequential execution
        self.base.forward(graph)
    }

    fn backward(
        &mut self,
        graph: &EinsumGraph,
        loss_grad: &Self::Tensor,
    ) -> Result<Self::Tape, Self::Error> {
        self.base.backward(graph, loss_grad)
    }
}

#[cfg(test)]
#[cfg(feature = "parallel")]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use tensorlogic_ir::EinsumNode;

    fn create_parallel_test_graph() -> EinsumGraph {
        // Create a graph with parallelizable operations:
        // Tensors: 0=a, 1=b, 2=c, 3=d, 4=e, 5=f
        // Op0: c = relu(a)    (level 0, independent)
        // Op1: d = sigmoid(b) (level 0, independent)
        // Op2: e = c + d      (level 1, depends on Op0, Op1)
        // Op3: f = relu(e)    (level 2, depends on Op2)

        let mut graph = EinsumGraph::new();

        let a_idx = graph.add_tensor("a"); // 0
        let b_idx = graph.add_tensor("b"); // 1
        let c_idx = graph.add_tensor("c"); // 2
        let d_idx = graph.add_tensor("d"); // 3
        let e_idx = graph.add_tensor("e"); // 4
        let f_idx = graph.add_tensor("f"); // 5

        graph.add_input(a_idx).unwrap();
        graph.add_input(b_idx).unwrap();

        // Op0: c = relu(a)
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "relu".to_string(),
                },
                inputs: vec![a_idx],
                outputs: vec![c_idx],
                metadata: None,
            })
            .unwrap();

        // Op1: d = sigmoid(b)
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "sigmoid".to_string(),
                },
                inputs: vec![b_idx],
                outputs: vec![d_idx],
                metadata: None,
            })
            .unwrap();

        // Op2: e = c + d
        graph
            .add_node(EinsumNode {
                op: OpType::ElemBinary {
                    op: "add".to_string(),
                },
                inputs: vec![c_idx, d_idx],
                outputs: vec![e_idx],
                metadata: None,
            })
            .unwrap();

        // Op3: f = relu(e)
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "relu".to_string(),
                },
                inputs: vec![e_idx],
                outputs: vec![f_idx],
                metadata: None,
            })
            .unwrap();

        graph.add_output(f_idx).unwrap();

        graph
    }

    #[test]
    fn test_parallel_executor_creation() {
        let executor = ParallelScirs2Exec::new();
        assert_eq!(executor.config.min_parallel_ops, 2);
        assert!(executor.config.enable_pooling);
    }

    #[test]
    fn test_set_num_threads() {
        let mut executor = ParallelScirs2Exec::new();
        executor.set_num_threads(4);
        assert_eq!(executor.config.num_threads, Some(4));
    }

    #[test]
    fn test_parallel_forward_pass() {
        let graph = create_parallel_test_graph();
        let mut executor = ParallelScirs2Exec::new();

        executor.add_tensor("a", array![-1.0, 2.0, -3.0].into_dyn());
        executor.add_tensor("b", array![0.0, 1.0, 2.0].into_dyn());

        let result = executor.forward(&graph).unwrap();

        // Verify output shape
        assert_eq!(result.shape(), &[3]);

        // Verify statistics
        let stats = executor.execution_stats().unwrap();
        assert_eq!(stats.num_levels, 3);
        assert!(stats.parallel_ops >= 2); // Op0 and Op1 should run in parallel
    }

    #[test]
    fn test_parallel_vs_sequential_correctness() {
        let graph = create_parallel_test_graph();

        // Execute with parallel executor
        let mut parallel_exec = ParallelScirs2Exec::new();
        parallel_exec.add_tensor("a", array![-1.0, 2.0, -3.0].into_dyn());
        parallel_exec.add_tensor("b", array![0.0, 1.0, 2.0].into_dyn());
        let parallel_result = parallel_exec.forward(&graph).unwrap();

        // Execute with sequential executor
        let mut sequential_exec = crate::executor::Scirs2Exec::new();
        sequential_exec.add_tensor("a", array![-1.0, 2.0, -3.0].into_dyn());
        sequential_exec.add_tensor("b", array![0.0, 1.0, 2.0].into_dyn());
        let sequential_result = sequential_exec.forward(&graph).unwrap();

        // Results should match
        assert_eq!(parallel_result.shape(), sequential_result.shape());

        for (p, s) in parallel_result.iter().zip(sequential_result.iter()) {
            assert!((p - s).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parallel_stats() {
        let graph = create_parallel_test_graph();
        let mut executor = ParallelScirs2Exec::new();

        executor.add_tensor("a", array![1.0, 2.0].into_dyn());
        executor.add_tensor("b", array![3.0, 4.0].into_dyn());

        executor.forward(&graph).unwrap();

        let stats = executor.execution_stats().unwrap();
        assert_eq!(stats.num_levels, 3);
        assert!(stats.max_parallelism >= 2);
        assert!(stats.estimated_speedup > 1.0);
    }

    #[test]
    fn test_pooling_integration() {
        let graph = create_parallel_test_graph();
        let mut executor = ParallelScirs2Exec::new();
        executor.set_pooling(true);

        executor.add_tensor("a", array![1.0, 2.0].into_dyn());
        executor.add_tensor("b", array![3.0, 4.0].into_dyn());

        executor.forward(&graph).unwrap();

        // Pool should have some statistics (if pooling is used)
        let _pool_stats = executor.pool_stats();
        // Note: pool might not be used in forward pass, so this is optional
    }

    #[test]
    fn test_min_parallel_ops_threshold() {
        // Create a graph with only 1 independent operation
        let mut graph = EinsumGraph::new();

        let a_idx = graph.add_tensor("a");
        let b_idx = graph.add_tensor("b");

        graph.add_input(a_idx).unwrap();

        // Single operation
        graph
            .add_node(EinsumNode {
                op: OpType::ElemUnary {
                    op: "relu".to_string(),
                },
                inputs: vec![a_idx],
                outputs: vec![b_idx],
                metadata: None,
            })
            .unwrap();

        graph.add_output(b_idx).unwrap();

        let mut executor = ParallelScirs2Exec::new();
        executor.add_tensor("a", array![1.0, 2.0, 3.0].into_dyn());

        executor.forward(&graph).unwrap();

        let stats = executor.execution_stats().unwrap();
        // Since there's only 1 op, it should run sequentially
        assert_eq!(stats.sequential_ops, 1);
        assert_eq!(stats.parallel_ops, 0);
    }

    #[test]
    fn test_backward_pass_with_parallel() {
        let graph = create_parallel_test_graph();
        let mut executor = ParallelScirs2Exec::new();

        executor.add_tensor("a", array![1.0, 2.0, 3.0].into_dyn());
        executor.add_tensor("b", array![0.5, 1.0, 1.5].into_dyn());

        executor.forward(&graph).unwrap();

        // Backward pass
        let loss_grad = array![1.0, 1.0, 1.0].into_dyn();

        let tape = executor.backward(&graph, &loss_grad).unwrap();

        // Should have gradients for inputs
        assert!(!tape.is_empty());
    }
}
