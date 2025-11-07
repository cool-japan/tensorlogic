//! Shape inference and validation support.

use crate::Scirs2Exec;
use std::collections::HashMap;
use tensorlogic_infer::{ExecutorError, ShapeInferenceContext, TensorShape};
use tensorlogic_ir::{EinsumGraph, OpType};

/// Shape inference engine for SciRS2 backend
pub struct Scirs2ShapeInference {
    /// Known tensor shapes
    shapes: HashMap<String, Vec<usize>>,
}

impl Scirs2ShapeInference {
    /// Create a new shape inference engine
    pub fn new() -> Self {
        Scirs2ShapeInference {
            shapes: HashMap::new(),
        }
    }

    /// Register a tensor shape
    pub fn register_shape(&mut self, name: String, shape: Vec<usize>) {
        self.shapes.insert(name, shape);
    }

    /// Infer shapes for all tensors in a graph
    pub fn infer_graph_shapes(
        &mut self,
        graph: &EinsumGraph,
        executor: &Scirs2Exec,
    ) -> Result<ShapeInferenceContext, ExecutorError> {
        let mut context = ShapeInferenceContext::new();

        // Register input tensor shapes from executor
        for (idx, tensor_name) in graph.tensors.iter().enumerate() {
            if let Some(tensor) = executor.get_tensor(tensor_name) {
                let shape = tensor.shape().to_vec();
                context.set_tensor_shape(idx, TensorShape::static_shape(shape.clone()));
                self.shapes.insert(tensor_name.clone(), shape);
            }
        }

        // Infer shapes for each operation in the graph
        for node in &graph.nodes {
            self.infer_node_shape(node, &mut context)?;
        }

        Ok(context)
    }

    /// Infer output shape for a single node
    fn infer_node_shape(
        &self,
        node: &tensorlogic_ir::EinsumNode,
        context: &mut ShapeInferenceContext,
    ) -> Result<(), ExecutorError> {
        let input_shapes: Vec<TensorShape> = node
            .inputs
            .iter()
            .filter_map(|&idx| context.get_tensor_shape(idx).cloned())
            .collect();

        if input_shapes.len() != node.inputs.len() {
            return Err(ExecutorError::ShapeMismatch(
                "Not all input shapes are known".to_string(),
            ));
        }

        // Infer output shape based on operation type
        let output_shape = match &node.op {
            OpType::Einsum { spec } => self.infer_einsum_shape(spec, &input_shapes)?,
            OpType::ElemUnary { .. } => {
                // Unary operations preserve shape
                input_shapes[0].clone()
            }
            OpType::ElemBinary { .. } => {
                // Binary operations require compatible shapes
                self.infer_binary_shape(&input_shapes[0], &input_shapes[1])?
            }
            OpType::Reduce { axes, .. } => {
                // Reduction removes specified axes
                self.infer_reduce_shape(&input_shapes[0], axes)?
            }
        };

        // Register output shape
        if let Some(&output_idx) = node.outputs.first() {
            context.set_tensor_shape(output_idx, output_shape);
        }

        Ok(())
    }

    /// Infer shape for einsum operation
    fn infer_einsum_shape(
        &self,
        spec: &str,
        _input_shapes: &[TensorShape],
    ) -> Result<TensorShape, ExecutorError> {
        // Parse einsum specification
        let parts: Vec<&str> = spec.split("->").collect();
        if parts.len() != 2 {
            return Err(ExecutorError::InvalidEinsumSpec(format!(
                "Invalid einsum spec: {}",
                spec
            )));
        }

        let output_spec = parts[1].trim();

        // For now, return a dynamic shape for einsum
        // Full shape inference would require parsing the spec and input shapes
        Ok(TensorShape::dynamic(output_spec.len()))
    }

    /// Infer shape for binary element-wise operation
    fn infer_binary_shape(
        &self,
        shape1: &TensorShape,
        shape2: &TensorShape,
    ) -> Result<TensorShape, ExecutorError> {
        // Check if both shapes are static
        if let (Some(s1), Some(s2)) = (shape1.as_static(), shape2.as_static()) {
            if s1 == s2 {
                return Ok(TensorShape::static_shape(s1));
            } else if s1.is_empty() {
                // Scalar broadcast
                return Ok(TensorShape::static_shape(s2));
            } else if s2.is_empty() {
                // Scalar broadcast
                return Ok(TensorShape::static_shape(s1));
            } else {
                return Err(ExecutorError::ShapeMismatch(format!(
                    "Incompatible shapes: {:?} and {:?}",
                    s1, s2
                )));
            }
        }

        // If either shape is dynamic, return dynamic
        Ok(TensorShape::dynamic(shape1.rank().max(shape2.rank())))
    }

    /// Infer shape for reduction operation
    fn infer_reduce_shape(
        &self,
        shape: &TensorShape,
        axes: &[usize],
    ) -> Result<TensorShape, ExecutorError> {
        if let Some(dims) = shape.as_static() {
            let mut result_dims = dims.clone();
            // Remove reduced axes (in reverse order to maintain indices)
            for &axis in axes.iter().rev() {
                if axis < result_dims.len() {
                    result_dims.remove(axis);
                }
            }
            return Ok(TensorShape::static_shape(result_dims));
        }

        // Dynamic or symbolic shape
        let new_rank = shape.rank().saturating_sub(axes.len());
        Ok(TensorShape::dynamic(new_rank))
    }
}

impl Default for Scirs2ShapeInference {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate tensor shapes match expected shapes
pub fn validate_tensor_shapes(
    executor: &Scirs2Exec,
    expected_shapes: &HashMap<String, Vec<usize>>,
) -> Result<(), ExecutorError> {
    for (name, expected_shape) in expected_shapes {
        if let Some(tensor) = executor.get_tensor(name) {
            let actual_shape = tensor.shape();
            if actual_shape != expected_shape.as_slice() {
                return Err(ExecutorError::ShapeMismatch(format!(
                    "Tensor '{}': expected shape {:?}, got {:?}",
                    name, expected_shape, actual_shape
                )));
            }
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "integration-tests"))]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;
    use tensorlogic_compiler::compile_to_einsum;
    use tensorlogic_ir::{TLExpr, Term};

    fn create_test_tensor(shape: &[usize]) -> ArrayD<f64> {
        ArrayD::zeros(shape.to_vec())
    }

    #[test]
    fn test_shape_inference_basic() {
        let x = TLExpr::pred("x", vec![Term::var("i"), Term::var("j")]);
        let y = TLExpr::pred("y", vec![Term::var("i"), Term::var("j")]);
        let expr = TLExpr::add(x, y);
        let graph = compile_to_einsum(&expr).unwrap();

        let mut executor = Scirs2Exec::new();
        executor.add_tensor(graph.tensors[0].clone(), create_test_tensor(&[3, 4]));
        executor.add_tensor(graph.tensors[1].clone(), create_test_tensor(&[3, 4]));

        let mut inference = Scirs2ShapeInference::new();
        let context = inference.infer_graph_shapes(&graph, &executor).unwrap();

        // Check that shapes were inferred
        assert!(context.get_tensor_shape(0).is_some());
        assert!(context.get_tensor_shape(1).is_some());
    }

    #[test]
    fn test_validate_shapes_success() {
        let mut executor = Scirs2Exec::new();
        executor.add_tensor("x".to_string(), create_test_tensor(&[2, 3]));
        executor.add_tensor("y".to_string(), create_test_tensor(&[4, 5]));

        let mut expected = HashMap::new();
        expected.insert("x".to_string(), vec![2, 3]);
        expected.insert("y".to_string(), vec![4, 5]);

        let result = validate_tensor_shapes(&executor, &expected);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_shapes_mismatch() {
        let mut executor = Scirs2Exec::new();
        executor.add_tensor("x".to_string(), create_test_tensor(&[2, 3]));

        let mut expected = HashMap::new();
        expected.insert("x".to_string(), vec![3, 4]); // Wrong shape

        let result = validate_tensor_shapes(&executor, &expected);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_unary_shape() {
        let inference = Scirs2ShapeInference::new();
        let input_shape = TensorShape::static_shape(vec![2, 3, 4]);

        // Unary operations preserve shape
        let node = tensorlogic_ir::EinsumNode {
            inputs: vec![0],
            outputs: vec![1],
            op: OpType::ElemUnary {
                op: "relu".to_string(),
            },
            metadata: None,
        };

        let mut context = ShapeInferenceContext::new();
        context.set_tensor_shape(0, input_shape.clone());

        inference.infer_node_shape(&node, &mut context).unwrap();

        let output_shape = context.get_tensor_shape(1).unwrap();
        assert_eq!(output_shape, &input_shape);
    }

    #[test]
    fn test_infer_reduce_shape() {
        let inference = Scirs2ShapeInference::new();

        // Reduce along axis 1: [2, 3, 4] -> [2, 4]
        let result = inference
            .infer_reduce_shape(&TensorShape::static_shape(vec![2, 3, 4]), &[1])
            .unwrap();

        let result_dims = result.as_static().unwrap();
        assert_eq!(result_dims, vec![2, 4]);
    }

    #[test]
    fn test_infer_binary_shape_matching() {
        let inference = Scirs2ShapeInference::new();

        let shape1 = TensorShape::static_shape(vec![2, 3]);
        let shape2 = TensorShape::static_shape(vec![2, 3]);

        let result = inference.infer_binary_shape(&shape1, &shape2).unwrap();

        let result_dims = result.as_static().unwrap();
        assert_eq!(result_dims, vec![2, 3]);
    }

    #[test]
    fn test_infer_binary_shape_scalar_broadcast() {
        let inference = Scirs2ShapeInference::new();

        let shape1 = TensorShape::static_shape(vec![]); // Scalar
        let shape2 = TensorShape::static_shape(vec![2, 3]);

        let result = inference.infer_binary_shape(&shape1, &shape2).unwrap();

        let result_dims = result.as_static().unwrap();
        assert_eq!(result_dims, vec![2, 3]);
    }

    #[test]
    fn test_infer_binary_shape_mismatch() {
        let inference = Scirs2ShapeInference::new();

        let shape1 = TensorShape::static_shape(vec![2, 3]);
        let shape2 = TensorShape::static_shape(vec![4, 5]);

        let result = inference.infer_binary_shape(&shape1, &shape2);
        assert!(result.is_err());
    }
}
