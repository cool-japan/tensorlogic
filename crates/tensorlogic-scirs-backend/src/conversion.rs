//! Tensor conversion and construction utilities.

use scirs2_core::ndarray::{Array, IxDyn};
use tensorlogic_infer::ExecutorError;

use crate::{Scirs2Exec, Scirs2Tensor};

impl Scirs2Exec {
    /// Create a tensor from a flat vector and shape.
    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Result<Scirs2Tensor, ExecutorError> {
        let total_size: usize = shape.iter().product();
        if total_size != data.len() {
            return Err(ExecutorError::ShapeMismatch(format!(
                "Data length {} doesn't match shape {:?} (total: {})",
                data.len(),
                shape,
                total_size
            )));
        }

        Array::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| ExecutorError::ShapeMismatch(e.to_string()))
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Scirs2Tensor {
        Array::zeros(IxDyn(&shape))
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Scirs2Tensor {
        Array::ones(IxDyn(&shape))
    }
}
