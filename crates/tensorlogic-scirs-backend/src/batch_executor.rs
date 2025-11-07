//! Batch execution support for parallel processing.

use crate::{Scirs2Exec, Scirs2Tensor};
use tensorlogic_infer::{BatchResult, ExecutorError, TlAutodiff, TlBatchExecutor};
use tensorlogic_ir::EinsumGraph;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

impl TlBatchExecutor for Scirs2Exec {
    type Tensor = Scirs2Tensor;
    type Error = ExecutorError;

    fn execute_batch(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<Vec<Self::Tensor>>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error> {
        if batch_inputs.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "Empty batch provided".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(batch_inputs.len());

        for input_batch in batch_inputs {
            // Store tensors for this batch item
            for (idx, tensor) in input_batch.iter().enumerate() {
                if idx < graph.tensors.len() {
                    self.add_tensor(graph.tensors[idx].clone(), tensor.clone());
                }
            }

            let output = self.forward(graph)?;
            outputs.push(output);
        }

        Ok(BatchResult::new(outputs))
    }

    fn execute_batch_parallel(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<Vec<Self::Tensor>>,
        num_threads: Option<usize>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error> {
        #[cfg(feature = "parallel")]
        {
            if batch_inputs.is_empty() {
                return Err(ExecutorError::InvalidEinsumSpec(
                    "Empty batch provided".to_string(),
                ));
            }

            // Configure thread pool if requested
            if let Some(threads) = num_threads {
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build_global()
                    .ok(); // Ignore if already initialized
            }

            // Execute batch items in parallel
            let results: Result<Vec<_>, _> = batch_inputs
                .par_iter()
                .map(|input_batch| {
                    let mut executor = self.clone();

                    for (idx, tensor) in input_batch.iter().enumerate() {
                        if idx < graph.tensors.len() {
                            executor.add_tensor(graph.tensors[idx].clone(), tensor.clone());
                        }
                    }

                    executor.forward(graph)
                })
                .collect();

            let outputs = results?;
            Ok(BatchResult::new(outputs))
        }

        #[cfg(not(feature = "parallel"))]
        {
            let _ = num_threads; // Avoid unused variable warning
                                 // Fall back to sequential execution when parallel feature is disabled
            self.execute_batch(graph, batch_inputs)
        }
    }

    fn optimal_batch_size(&self) -> usize {
        // For CPU execution, a moderate batch size balances parallelism and overhead
        let num_cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        // Use 2x the number of CPUs as a heuristic
        num_cpus * 2
    }
}

/// Parallel batch executor using rayon for CPU parallelism
pub struct ParallelBatchExecutor {
    /// Base executor template
    base: Scirs2Exec,
}

impl ParallelBatchExecutor {
    /// Create a new parallel batch executor
    pub fn new() -> Self {
        ParallelBatchExecutor {
            base: Scirs2Exec::new(),
        }
    }

    /// Create parallel batch executor with memory pooling
    pub fn with_memory_pool() -> Self {
        ParallelBatchExecutor {
            base: Scirs2Exec::with_memory_pool(),
        }
    }

    /// Execute batch in parallel using rayon
    pub fn execute_parallel(
        &self,
        graph: &EinsumGraph,
        batch_inputs: Vec<Vec<Scirs2Tensor>>,
    ) -> Result<BatchResult<Scirs2Tensor>, ExecutorError> {
        if batch_inputs.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "Empty batch provided".to_string(),
            ));
        }

        #[cfg(feature = "parallel")]
        {
            // Execute batch items in parallel using rayon
            let results: Result<Vec<_>, _> = batch_inputs
                .par_iter()
                .map(|input_batch| {
                    let mut executor = self.base.clone();

                    for (idx, tensor) in input_batch.iter().enumerate() {
                        if idx < graph.tensors.len() {
                            executor.add_tensor(graph.tensors[idx].clone(), tensor.clone());
                        }
                    }

                    executor.forward(graph)
                })
                .collect();

            let outputs = results?;
            Ok(BatchResult::new(outputs))
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Fall back to sequential execution when parallel feature is disabled
            let mut outputs = Vec::with_capacity(batch_inputs.len());

            for input_batch in batch_inputs {
                let mut executor = self.base.clone();

                for (idx, tensor) in input_batch.iter().enumerate() {
                    if idx < graph.tensors.len() {
                        executor.add_tensor(graph.tensors[idx].clone(), tensor.clone());
                    }
                }

                let output = executor.forward(graph)?;
                outputs.push(output);
            }

            Ok(BatchResult::new(outputs))
        }
    }
}

impl Default for ParallelBatchExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Scirs2Exec {
    fn clone(&self) -> Self {
        Scirs2Exec {
            tensors: self.tensors.clone(),
            tape: self.tape.clone(),
            pool: None, // Don't clone pool to avoid shared state issues
        }
    }
}

#[cfg(all(test, feature = "integration-tests"))]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;
    use tensorlogic_compiler::compile_to_einsum;
    use tensorlogic_ir::{TLExpr, Term};

    fn create_test_tensor(shape: &[usize], value: f64) -> ArrayD<f64> {
        ArrayD::from_elem(shape.to_vec(), value)
    }

    #[test]
    fn test_batch_executor_basic() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let y = TLExpr::pred("y", vec![Term::var("i")]);
        let expr = TLExpr::add(x, y);
        let graph = compile_to_einsum(&expr).unwrap();

        let mut executor = Scirs2Exec::new();

        // Create batch of 3 items
        let batch_inputs = vec![
            vec![create_test_tensor(&[5], 1.0), create_test_tensor(&[5], 2.0)],
            vec![create_test_tensor(&[5], 3.0), create_test_tensor(&[5], 4.0)],
            vec![create_test_tensor(&[5], 5.0), create_test_tensor(&[5], 6.0)],
        ];

        let result = executor.execute_batch(&graph, batch_inputs).unwrap();

        assert_eq!(result.len(), 3);
        assert!((result.outputs[0][0] - 3.0).abs() < 1e-6); // 1 + 2
        assert!((result.outputs[1][0] - 7.0).abs() < 1e-6); // 3 + 4
        assert!((result.outputs[2][0] - 11.0).abs() < 1e-6); // 5 + 6
        assert_eq!(result.batch_size, 3);
    }

    #[test]
    fn test_optimal_batch_size() {
        let executor = Scirs2Exec::new();

        let batch_size = executor.optimal_batch_size();
        assert!(batch_size > 0);
        assert!(batch_size <= 128); // Reasonable upper bound
    }

    #[test]
    fn test_parallel_batch_executor() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let expr = TLExpr::mul(x.clone(), x);
        let graph = compile_to_einsum(&expr).unwrap();

        let executor = ParallelBatchExecutor::new();

        let batch_inputs = vec![
            vec![create_test_tensor(&[3], 2.0)],
            vec![create_test_tensor(&[3], 3.0)],
        ];

        let result = executor.execute_parallel(&graph, batch_inputs).unwrap();

        assert_eq!(result.len(), 2);
        assert!((result.outputs[0][0] - 4.0).abs() < 1e-6); // 2 * 2
        assert!((result.outputs[1][0] - 9.0).abs() < 1e-6); // 3 * 3
    }

    #[test]
    fn test_empty_batch_error() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let graph = compile_to_einsum(&x).unwrap();

        let mut executor = Scirs2Exec::new();
        let batch_inputs: Vec<Vec<ArrayD<f64>>> = vec![];

        let result = executor.execute_batch(&graph, batch_inputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_parallel_same_as_sequential() {
        let x = TLExpr::pred("x", vec![Term::var("i")]);
        let y = TLExpr::pred("y", vec![Term::var("i")]);
        let expr = TLExpr::add(x, y);
        let graph = compile_to_einsum(&expr).unwrap();

        let batch_inputs = vec![
            vec![create_test_tensor(&[3], 1.0), create_test_tensor(&[3], 2.0)],
            vec![create_test_tensor(&[3], 3.0), create_test_tensor(&[3], 4.0)],
        ];

        let mut executor = Scirs2Exec::new();
        let result_seq = executor
            .execute_batch(&graph, batch_inputs.clone())
            .unwrap();

        let mut executor2 = Scirs2Exec::new();
        let result_par = executor2
            .execute_batch_parallel(&graph, batch_inputs, None)
            .unwrap();

        assert_eq!(result_seq.len(), result_par.len());
        for (seq, par) in result_seq.outputs.iter().zip(result_par.outputs.iter()) {
            assert_eq!(seq.shape(), par.shape());
            for (s, p) in seq.iter().zip(par.iter()) {
                assert!((s - p).abs() < 1e-10);
            }
        }
    }
}
