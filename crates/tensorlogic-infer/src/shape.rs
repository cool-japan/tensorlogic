//! Tensor shape inference and validation.

use std::collections::HashMap;

use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType};

/// Shape information for a tensor dimension
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimSize {
    /// Static known size
    Static(usize),
    /// Dynamic size (known at runtime)
    Dynamic,
    /// Symbolic dimension (e.g., batch size)
    Symbolic(String),
}

impl DimSize {
    pub fn is_static(&self) -> bool {
        matches!(self, DimSize::Static(_))
    }

    pub fn as_static(&self) -> Option<usize> {
        match self {
            DimSize::Static(size) => Some(*size),
            _ => None,
        }
    }
}

/// Tensor shape representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<DimSize>,
}

impl TensorShape {
    pub fn new(dims: Vec<DimSize>) -> Self {
        TensorShape { dims }
    }

    pub fn static_shape(sizes: Vec<usize>) -> Self {
        TensorShape {
            dims: sizes.into_iter().map(DimSize::Static).collect(),
        }
    }

    pub fn dynamic(rank: usize) -> Self {
        TensorShape {
            dims: vec![DimSize::Dynamic; rank],
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn is_static(&self) -> bool {
        self.dims.iter().all(|d| d.is_static())
    }

    pub fn as_static(&self) -> Option<Vec<usize>> {
        self.dims.iter().map(|d| d.as_static()).collect()
    }

    /// Check if two shapes are compatible (can broadcast or are equal)
    pub fn compatible_with(&self, other: &TensorShape) -> bool {
        if self.rank() != other.rank() {
            return false;
        }

        for (a, b) in self.dims.iter().zip(other.dims.iter()) {
            match (a, b) {
                (DimSize::Static(size_a), DimSize::Static(size_b)) => {
                    if size_a != size_b && *size_a != 1 && *size_b != 1 {
                        return false;
                    }
                }
                _ => {
                    // Dynamic or symbolic dims are always compatible
                }
            }
        }

        true
    }
}

/// Shape inference context
pub struct ShapeInferenceContext {
    tensor_shapes: HashMap<usize, TensorShape>,
}

impl ShapeInferenceContext {
    pub fn new() -> Self {
        ShapeInferenceContext {
            tensor_shapes: HashMap::new(),
        }
    }

    pub fn set_tensor_shape(&mut self, tensor_idx: usize, shape: TensorShape) {
        self.tensor_shapes.insert(tensor_idx, shape);
    }

    pub fn get_tensor_shape(&self, tensor_idx: usize) -> Option<&TensorShape> {
        self.tensor_shapes.get(&tensor_idx)
    }

    /// Infer shapes for all tensors in a graph
    pub fn infer_graph_shapes(
        &mut self,
        graph: &EinsumGraph,
        input_shapes: &HashMap<usize, TensorShape>,
    ) -> Result<(), String> {
        // Copy input shapes
        for (idx, shape) in input_shapes {
            self.tensor_shapes.insert(*idx, shape.clone());
        }

        // Infer shapes for each node
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            let output_idx = node_idx + graph.tensors.len(); // Simplified
            let output_shape = self.infer_node_shape(node)?;
            self.tensor_shapes.insert(output_idx, output_shape);
        }

        Ok(())
    }

    fn infer_node_shape(&self, node: &EinsumNode) -> Result<TensorShape, String> {
        match &node.op {
            OpType::Einsum { spec } => {
                // Parse einsum spec to infer output shape
                self.infer_einsum_shape(spec, &node.inputs)
            }
            OpType::ElemUnary { op: _ } => {
                // Unary ops preserve shape
                if let Some(input_shape) = self.get_tensor_shape(node.inputs[0]) {
                    Ok(input_shape.clone())
                } else {
                    Err("Input shape not available for unary op".to_string())
                }
            }
            OpType::ElemBinary { op: _ } => {
                // Binary ops require compatible shapes
                if node.inputs.len() < 2 {
                    return Err("Binary op requires 2 inputs".to_string());
                }

                let shape_a = self
                    .get_tensor_shape(node.inputs[0])
                    .ok_or("Input 0 shape not available")?;
                let shape_b = self
                    .get_tensor_shape(node.inputs[1])
                    .ok_or("Input 1 shape not available")?;

                if !shape_a.compatible_with(shape_b) {
                    return Err(format!(
                        "Incompatible shapes for binary op: {:?} vs {:?}",
                        shape_a, shape_b
                    ));
                }

                // Return the broadcasted shape
                Ok(shape_a.clone())
            }
            OpType::Reduce { op: _, axes } => {
                if let Some(input_shape) = self.get_tensor_shape(node.inputs[0]) {
                    // Remove reduced dimensions
                    let mut output_dims = input_shape.dims.clone();
                    for &axis in axes.iter().rev() {
                        if axis < output_dims.len() {
                            output_dims.remove(axis);
                        }
                    }
                    Ok(TensorShape::new(output_dims))
                } else {
                    Err("Input shape not available for reduce op".to_string())
                }
            }
        }
    }

    fn infer_einsum_shape(&self, spec: &str, inputs: &[usize]) -> Result<TensorShape, String> {
        // Parse einsum specification
        let (input_specs, output_spec) = if let Some(arrow_pos) = spec.find("->") {
            let input_part = &spec[..arrow_pos];
            let output_part = &spec[arrow_pos + 2..];
            (input_part, Some(output_part))
        } else {
            (spec, None)
        };

        // Parse input specs (e.g., "ab,bc" -> ["ab", "bc"])
        let input_specs: Vec<&str> = input_specs.split(',').map(|s| s.trim()).collect();

        if input_specs.len() != inputs.len() {
            return Err(format!(
                "Einsum spec has {} inputs but {} tensors provided",
                input_specs.len(),
                inputs.len()
            ));
        }

        // Build dimension size map from input tensors
        let mut dim_sizes: std::collections::HashMap<char, DimSize> =
            std::collections::HashMap::new();

        for (spec_idx, &input_idx) in inputs.iter().enumerate() {
            let input_shape = self
                .get_tensor_shape(input_idx)
                .ok_or_else(|| format!("Input {} shape not available", input_idx))?;

            let axes = input_specs[spec_idx].chars().collect::<Vec<_>>();

            if axes.len() != input_shape.rank() {
                return Err(format!(
                    "Input {} spec '{}' has {} axes but tensor has rank {}",
                    spec_idx,
                    input_specs[spec_idx],
                    axes.len(),
                    input_shape.rank()
                ));
            }

            // Map each axis character to its dimension size
            for (axis_idx, axis_char) in axes.iter().enumerate() {
                let dim_size = input_shape.dims[axis_idx].clone();

                if let Some(existing) = dim_sizes.get(axis_char) {
                    // Check consistency
                    if let (DimSize::Static(size1), DimSize::Static(size2)) = (existing, &dim_size)
                    {
                        if size1 != size2 {
                            return Err(format!(
                                "Dimension '{}' has inconsistent sizes: {} vs {}",
                                axis_char, size1, size2
                            ));
                        }
                    }
                } else {
                    dim_sizes.insert(*axis_char, dim_size);
                }
            }
        }

        // Determine output shape
        let output_dims = if let Some(output_axes) = output_spec {
            // Explicit output specification
            output_axes
                .chars()
                .map(|c| dim_sizes.get(&c).cloned().unwrap_or(DimSize::Dynamic))
                .collect()
        } else {
            // Implicit output: all non-repeated indices in alphabetical order
            let mut all_axes: Vec<char> = dim_sizes.keys().copied().collect();
            all_axes.sort();
            all_axes
                .into_iter()
                .map(|c| dim_sizes[&c].clone())
                .collect()
        };

        Ok(TensorShape::new(output_dims))
    }
}

impl Default for ShapeInferenceContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_static() {
        let shape = TensorShape::static_shape(vec![3, 4, 5]);
        assert_eq!(shape.rank(), 3);
        assert!(shape.is_static());
        assert_eq!(shape.as_static(), Some(vec![3, 4, 5]));
    }

    #[test]
    fn test_tensor_shape_dynamic() {
        let shape = TensorShape::dynamic(3);
        assert_eq!(shape.rank(), 3);
        assert!(!shape.is_static());
        assert_eq!(shape.as_static(), None);
    }

    #[test]
    fn test_shape_compatibility() {
        let shape1 = TensorShape::static_shape(vec![3, 4]);
        let shape2 = TensorShape::static_shape(vec![3, 4]);
        assert!(shape1.compatible_with(&shape2));

        let shape3 = TensorShape::static_shape(vec![3, 1]);
        assert!(shape1.compatible_with(&shape3)); // Broadcasting

        let shape4 = TensorShape::static_shape(vec![3, 5]);
        assert!(!shape1.compatible_with(&shape4));
    }

    #[test]
    fn test_shape_inference_context() {
        let mut ctx = ShapeInferenceContext::new();
        let shape = TensorShape::static_shape(vec![2, 3]);

        ctx.set_tensor_shape(0, shape.clone());
        assert_eq!(ctx.get_tensor_shape(0), Some(&shape));
        assert_eq!(ctx.get_tensor_shape(1), None);
    }

    #[test]
    fn test_einsum_shape_inference() {
        let mut ctx = ShapeInferenceContext::new();

        // Set up input shapes
        ctx.set_tensor_shape(0, TensorShape::static_shape(vec![3, 4]));
        ctx.set_tensor_shape(1, TensorShape::static_shape(vec![4, 5]));

        // "ab,bc->ac" should produce shape [3, 5]
        let shape = ctx.infer_einsum_shape("ab,bc->ac", &[0, 1]).unwrap();
        assert_eq!(shape.rank(), 2);
        assert_eq!(shape.as_static(), Some(vec![3, 5]));
    }

    #[test]
    fn test_einsum_shape_inference_explicit() {
        let mut ctx = ShapeInferenceContext::new();
        ctx.set_tensor_shape(0, TensorShape::static_shape(vec![2, 3, 4]));

        // "abc->ab" should produce shape [2, 3]
        let shape = ctx.infer_einsum_shape("abc->ab", &[0]).unwrap();
        assert_eq!(shape.rank(), 2);
        assert_eq!(shape.as_static(), Some(vec![2, 3]));
    }

    #[test]
    fn test_einsum_shape_inference_diagonal() {
        let mut ctx = ShapeInferenceContext::new();
        ctx.set_tensor_shape(0, TensorShape::static_shape(vec![3, 3]));

        // "aa->a" should produce shape [3]
        let shape = ctx.infer_einsum_shape("aa->a", &[0]).unwrap();
        assert_eq!(shape.rank(), 1);
        assert_eq!(shape.as_static(), Some(vec![3]));
    }

    #[test]
    fn test_einsum_shape_inference_batch_matmul() {
        let mut ctx = ShapeInferenceContext::new();
        ctx.set_tensor_shape(0, TensorShape::static_shape(vec![10, 3, 4]));
        ctx.set_tensor_shape(1, TensorShape::static_shape(vec![10, 4, 5]));

        // "bik,bkj->bij" should produce shape [10, 3, 5]
        let shape = ctx.infer_einsum_shape("bik,bkj->bij", &[0, 1]).unwrap();
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.as_static(), Some(vec![10, 3, 5]));
    }

    #[test]
    fn test_einsum_shape_inference_inconsistent_dims() {
        let mut ctx = ShapeInferenceContext::new();
        ctx.set_tensor_shape(0, TensorShape::static_shape(vec![3, 4]));
        ctx.set_tensor_shape(1, TensorShape::static_shape(vec![5, 6]));

        // "ab,bc->ac" should fail because 'b' has different sizes (4 vs 5)
        let result = ctx.infer_einsum_shape("ab,bc->ac", &[0, 1]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("inconsistent"));
    }
}
