//! Tests for memory module

#[cfg(test)]
mod tests {
    use crate::capabilities::DType;
    use crate::memory::{MemoryEstimator, TensorMemory};
    use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

    #[test]
    fn test_tensor_memory() {
        let mem = TensorMemory::new(0, vec![100, 100], DType::F64);
        assert_eq!(mem.element_count, 10000);
        assert_eq!(mem.bytes, 10000 * 8);
    }

    #[test]
    fn test_memory_estimator_simple() {
        let mut graph = EinsumGraph::new();
        let _ = graph.add_tensor("input");
        let node = EinsumNode {
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            inputs: vec![0],
            outputs: vec![1],
            metadata: None,
        };
        let _ = graph.add_node(node);

        let estimator = MemoryEstimator::new(DType::F64);
        let estimate = estimator.estimate(&graph);

        assert_eq!(estimate.input_memory.len(), 1);
    }

    #[test]
    fn test_batch_memory_estimate() {
        let mut graph = EinsumGraph::new();
        let _ = graph.add_tensor("input");
        let node = EinsumNode {
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            inputs: vec![0],
            outputs: vec![1],
            metadata: None,
        };
        let _ = graph.add_node(node);

        let estimator = MemoryEstimator::new(DType::F64);
        let single_estimate = estimator.estimate(&graph);
        let batch_estimate = estimator.estimate_batch(&graph, 32);

        assert_eq!(batch_estimate.total_bytes, single_estimate.total_bytes * 32);
    }
}
