//! Tests for validation module

#[cfg(test)]
mod tests {
    use crate::validation::{GraphValidator, ValidationResult};
    use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

    #[test]
    fn test_validation_result_new() {
        let result = ValidationResult::new();
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_validation_result_add_error() {
        let mut result = ValidationResult::new();
        result.add_error("Test error");
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_empty_graph() {
        let graph = EinsumGraph::new();
        let validator = GraphValidator::new();
        let result = validator.validate(&graph);
        assert!(result.is_valid);
    }

    #[test]
    fn test_valid_simple_graph() {
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

        let validator = GraphValidator::new();
        let result = validator.validate(&graph);
        assert!(result.is_valid);
    }

    #[test]
    fn test_invalid_tensor_index() {
        let mut graph = EinsumGraph::new();
        let _ = graph.add_tensor("input");
        let node = EinsumNode {
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            inputs: vec![10], // Invalid
            outputs: vec![1],
            metadata: None,
        };
        graph.nodes.push(node); // Bypass validation

        let validator = GraphValidator::new();
        let result = validator.validate(&graph);
        assert!(!result.is_valid);
    }
}
