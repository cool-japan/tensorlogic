//! SciRS2 executor implementation.

use scirs2_core::ndarray::Axis;
use std::collections::HashMap;
use tensorlogic_infer::{ElemOp, ExecutorError, ReduceOp, TlExecutor};

use crate::autodiff::ForwardTape;
use crate::memory_pool::TensorPool;
use crate::Scirs2Tensor;

pub struct Scirs2Exec {
    pub tensors: HashMap<String, Scirs2Tensor>,
    pub(crate) tape: Option<ForwardTape>,
    /// Optional memory pool for tensor reuse
    pub(crate) pool: Option<TensorPool>,
}

impl Default for Scirs2Exec {
    fn default() -> Self {
        Self::new()
    }
}

impl Scirs2Exec {
    pub fn new() -> Self {
        Scirs2Exec {
            tensors: HashMap::new(),
            tape: None,
            pool: None,
        }
    }

    /// Create executor with memory pooling enabled
    pub fn with_memory_pool() -> Self {
        Scirs2Exec {
            tensors: HashMap::new(),
            tape: None,
            pool: Some(TensorPool::new()),
        }
    }

    /// Enable memory pooling
    pub fn enable_pooling(&mut self) {
        if self.pool.is_none() {
            self.pool = Some(TensorPool::new());
        }
    }

    /// Disable memory pooling
    pub fn disable_pooling(&mut self) {
        self.pool = None;
    }

    /// Get pool statistics if pooling is enabled
    pub fn pool_stats(&self) -> Option<crate::memory_pool::PoolStats> {
        self.pool.as_ref().map(|p| p.stats())
    }

    pub fn add_tensor(&mut self, name: impl Into<String>, tensor: Scirs2Tensor) {
        self.tensors.insert(name.into(), tensor);
    }

    pub fn get_tensor(&self, name: &str) -> Option<&Scirs2Tensor> {
        self.tensors.get(name)
    }
}

impl TlExecutor for Scirs2Exec {
    type Tensor = Scirs2Tensor;
    type Error = ExecutorError;

    fn einsum(&mut self, spec: &str, inputs: &[Self::Tensor]) -> Result<Self::Tensor, Self::Error> {
        if inputs.is_empty() {
            return Err(ExecutorError::InvalidEinsumSpec(
                "No input tensors provided".to_string(),
            ));
        }

        let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
        let view_refs: Vec<_> = views.iter().collect();

        scirs2_linalg::einsum(spec, &view_refs)
            .map_err(|e| ExecutorError::InvalidEinsumSpec(format!("Einsum error: {}", e)))
    }

    fn elem_op(&mut self, op: ElemOp, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        let result = match op {
            ElemOp::Relu => x.mapv(|v| v.max(0.0)),
            ElemOp::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ElemOp::OneMinus => x.mapv(|v| 1.0 - v),
            _ => {
                return Err(ExecutorError::UnsupportedOperation(format!(
                    "Unary operation {:?} not supported",
                    op
                )))
            }
        };

        Ok(result)
    }

    fn elem_op_binary(
        &mut self,
        op: ElemOp,
        x: &Self::Tensor,
        y: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        // Handle scalar broadcasting: if one tensor is scalar (shape []) and the other isn't,
        // broadcast the scalar to match the shape of the other tensor
        let x_is_scalar = x.ndim() == 0;
        let y_is_scalar = y.ndim() == 0;

        let (x_broadcast, y_broadcast);
        let (x_ref, y_ref) = if x_is_scalar && !y_is_scalar {
            // x is scalar, broadcast to y's shape
            let scalar_value = x.iter().next().unwrap();
            x_broadcast = scirs2_core::ndarray::Array::from_elem(y.raw_dim(), *scalar_value);
            (&x_broadcast.view(), &y.view())
        } else if y_is_scalar && !x_is_scalar {
            // y is scalar, broadcast to x's shape
            let scalar_value = y.iter().next().unwrap();
            y_broadcast = scirs2_core::ndarray::Array::from_elem(x.raw_dim(), *scalar_value);
            (&x.view(), &y_broadcast.view())
        } else if x.shape() != y.shape() {
            // Shapes don't match and neither is a scalar
            return Err(ExecutorError::ShapeMismatch(format!(
                "Shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            )));
        } else {
            // Shapes match exactly (including both being scalars)
            (&x.view(), &y.view())
        };

        let result = match op {
            // Arithmetic operations
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

            // Comparison operations (return 0.0 or 1.0)
            ElemOp::Eq => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if (a - b).abs() < 1e-10 { 1.0 } else { 0.0 }),
            ElemOp::Lt => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if a < b { 1.0 } else { 0.0 }),
            ElemOp::Gt => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if a > b { 1.0 } else { 0.0 }),
            ElemOp::Lte => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if a <= b { 1.0 } else { 0.0 }),
            ElemOp::Gte => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| if a >= b { 1.0 } else { 0.0 }),

            // Extended logical operations
            ElemOp::OrMax => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| a.max(b)),
            ElemOp::OrProbSum => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| a + b - a * b), // 1 - (1-a)(1-b) = a + b - ab
            ElemOp::Nand => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| 1.0 - a * b),
            ElemOp::Nor => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| 1.0 - a.max(b)),
            ElemOp::Xor => scirs2_core::ndarray::Zip::from(x_ref)
                .and(y_ref)
                .map_collect(|&a, &b| a + b - 2.0 * a * b), // Soft XOR: (a XOR b) = a + b - 2ab

            _ => {
                return Err(ExecutorError::UnsupportedOperation(format!(
                    "Binary operation {:?} not supported",
                    op
                )))
            }
        };

        Ok(result)
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

        for &axis in axes {
            if axis >= x.ndim() {
                return Err(ExecutorError::ShapeMismatch(format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    x.ndim()
                )));
            }
        }

        let mut result = x.clone();
        for &axis in axes.iter().rev() {
            result = match op {
                ReduceOp::Sum => result.sum_axis(Axis(axis)),
                ReduceOp::Max => result.fold_axis(Axis(axis), f64::NEG_INFINITY, |&a, &b| a.max(b)),
                ReduceOp::Min => result.fold_axis(Axis(axis), f64::INFINITY, |&a, &b| a.min(b)),
                ReduceOp::Mean => result.mean_axis(Axis(axis)).unwrap(),
                ReduceOp::Product => result.fold_axis(Axis(axis), 1.0, |&a, &b| a * b),
            };
        }

        Ok(result)
    }
}
