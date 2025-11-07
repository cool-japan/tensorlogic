//! Batch execution support for processing multiple inputs efficiently.

use tensorlogic_ir::EinsumGraph;

/// Batch execution result containing outputs and optional metadata
#[derive(Debug, Clone)]
pub struct BatchResult<T> {
    pub outputs: Vec<T>,
    pub batch_size: usize,
}

impl<T> BatchResult<T> {
    pub fn new(outputs: Vec<T>) -> Self {
        let batch_size = outputs.len();
        BatchResult {
            outputs,
            batch_size,
        }
    }

    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.outputs.iter()
    }
}

/// Extension trait for batch execution
pub trait TlBatchExecutor {
    type Tensor;
    type Error;

    /// Execute a graph on a batch of inputs
    fn execute_batch(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<Vec<Self::Tensor>>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error>;

    /// Execute a graph on a batch of inputs with parallel processing
    fn execute_batch_parallel(
        &mut self,
        graph: &EinsumGraph,
        batch_inputs: Vec<Vec<Self::Tensor>>,
        num_threads: Option<usize>,
    ) -> Result<BatchResult<Self::Tensor>, Self::Error>;

    /// Get maximum recommended batch size for this executor
    fn max_batch_size(&self) -> Option<usize> {
        None // Default: no limit
    }

    /// Get optimal batch size for this executor
    fn optimal_batch_size(&self) -> usize {
        32 // Default recommendation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_result() {
        let outputs = vec![1, 2, 3, 4];
        let result = BatchResult::new(outputs.clone());

        assert_eq!(result.len(), 4);
        assert_eq!(result.batch_size, 4);
        assert!(!result.is_empty());

        let collected: Vec<&i32> = result.iter().collect();
        assert_eq!(collected.len(), 4);
    }

    #[test]
    fn test_empty_batch_result() {
        let result: BatchResult<i32> = BatchResult::new(vec![]);
        assert_eq!(result.len(), 0);
        assert!(result.is_empty());
    }
}
