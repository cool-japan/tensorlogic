//! Model interface for training with Tensorlogic.

use crate::{TrainError, TrainResult};
use scirs2_core::ndarray::{Array, ArrayView, Ix2, IxDyn};
use std::collections::HashMap;

/// Trait for trainable models.
///
/// This trait defines the interface for models that can be trained with the
/// Tensorlogic training infrastructure. Models must implement forward and
/// backward passes, parameter management, and optional save/load functionality.
pub trait Model {
    /// Perform a forward pass through the model.
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// Output tensor from the model
    fn forward(&self, input: &ArrayView<f64, Ix2>) -> TrainResult<Array<f64, Ix2>>;

    /// Perform a backward pass to compute gradients.
    ///
    /// # Arguments
    /// * `input` - Input tensor used in forward pass
    /// * `grad_output` - Gradient of loss with respect to model output
    ///
    /// # Returns
    /// Gradients for each model parameter
    fn backward(
        &self,
        input: &ArrayView<f64, Ix2>,
        grad_output: &ArrayView<f64, Ix2>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>>;

    /// Get a reference to the model's parameters.
    fn parameters(&self) -> &HashMap<String, Array<f64, Ix2>>;

    /// Get a mutable reference to the model's parameters.
    fn parameters_mut(&mut self) -> &mut HashMap<String, Array<f64, Ix2>>;

    /// Set the model's parameters.
    fn set_parameters(&mut self, parameters: HashMap<String, Array<f64, Ix2>>);

    /// Get the number of parameters in the model.
    fn num_parameters(&self) -> usize {
        self.parameters().values().map(|p| p.len()).sum()
    }

    /// Save model state to a dictionary.
    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        self.parameters()
            .iter()
            .map(|(name, param)| (name.clone(), param.iter().copied().collect()))
            .collect()
    }

    /// Load model state from a dictionary.
    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) -> TrainResult<()> {
        let parameters = self.parameters_mut();

        for (name, values) in state {
            if let Some(param) = parameters.get_mut(&name) {
                if param.len() != values.len() {
                    return Err(TrainError::InvalidParameter(format!(
                        "Parameter '{}' size mismatch: expected {}, got {}",
                        name,
                        param.len(),
                        values.len()
                    )));
                }

                for (p, v) in param.iter_mut().zip(values.iter()) {
                    *p = *v;
                }
            } else {
                return Err(TrainError::InvalidParameter(format!(
                    "Parameter '{}' not found in model",
                    name
                )));
            }
        }

        Ok(())
    }

    /// Reset model parameters (optional, for retraining).
    fn reset_parameters(&mut self) {
        // Default implementation does nothing
        // Models can override this to implement custom initialization
    }
}

/// Trait for models that support automatic differentiation via scirs2-autograd.
///
/// This trait extends the base Model trait with support for training using
/// SciRS2's automatic differentiation system.
///
/// Note: This trait is currently a placeholder for future scirs2-autograd integration.
/// The actual Variable type will be specified once scirs2-autograd is fully integrated.
pub trait AutodiffModel: Model {
    /// Forward pass with autodiff tracking (placeholder).
    ///
    /// # Arguments
    /// * `input` - Input data array
    ///
    /// # Returns
    /// Success indicator (actual implementation will return autodiff Variable)
    fn forward_autodiff(&self, input: &ArrayView<f64, Ix2>) -> TrainResult<()> {
        // Placeholder implementation
        let _ = input;
        Ok(())
    }

    /// Compute gradients automatically using backward pass (placeholder).
    ///
    /// # Returns
    /// Gradients for all parameters
    fn compute_gradients(&self) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        // Placeholder implementation
        Ok(HashMap::new())
    }
}

/// Trait for models with dynamic computation graphs.
///
/// This extends the model interface to support variable-sized inputs
/// and dynamic graph construction (e.g., for RNNs, variable-length sequences).
pub trait DynamicModel {
    /// Forward pass with dynamic input dimensions.
    fn forward_dynamic(&self, input: &ArrayView<f64, IxDyn>) -> TrainResult<Array<f64, IxDyn>>;

    /// Backward pass with dynamic input dimensions.
    fn backward_dynamic(
        &self,
        input: &ArrayView<f64, IxDyn>,
        grad_output: &ArrayView<f64, IxDyn>,
    ) -> TrainResult<HashMap<String, Array<f64, IxDyn>>>;
}

/// A simple linear model for testing and demonstration.
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Model parameters (weights and biases).
    parameters: HashMap<String, Array<f64, Ix2>>,
    /// Input dimension.
    input_dim: usize,
    /// Output dimension.
    output_dim: usize,
}

impl LinearModel {
    /// Create a new linear model.
    ///
    /// # Arguments
    /// * `input_dim` - Input dimension
    /// * `output_dim` - Output dimension
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut parameters = HashMap::new();

        // Initialize weights with small random values (simplified)
        let weights = Array::zeros((input_dim, output_dim));
        let biases = Array::zeros((1, output_dim));

        parameters.insert("weight".to_string(), weights);
        parameters.insert("bias".to_string(), biases);

        Self {
            parameters,
            input_dim,
            output_dim,
        }
    }

    /// Initialize parameters with Xavier/Glorot uniform initialization.
    pub fn xavier_init(&mut self) {
        let limit = (6.0 / (self.input_dim + self.output_dim) as f64).sqrt();

        if let Some(weights) = self.parameters.get_mut("weight") {
            // Simplified initialization (in practice, use proper random)
            weights.mapv_inplace(|_| (limit * 2.0 * 0.5) - limit);
        }
    }

    /// Get input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

impl Model for LinearModel {
    fn forward(&self, input: &ArrayView<f64, Ix2>) -> TrainResult<Array<f64, Ix2>> {
        let weights = self
            .parameters
            .get("weight")
            .ok_or_else(|| TrainError::InvalidParameter("weight not found".to_string()))?;
        let biases = self
            .parameters
            .get("bias")
            .ok_or_else(|| TrainError::InvalidParameter("bias not found".to_string()))?;

        // Linear transformation: Y = X @ W + b
        let output = input.dot(weights) + biases;
        Ok(output)
    }

    fn backward(
        &self,
        input: &ArrayView<f64, Ix2>,
        grad_output: &ArrayView<f64, Ix2>,
    ) -> TrainResult<HashMap<String, Array<f64, Ix2>>> {
        let mut gradients = HashMap::new();

        // Gradient w.r.t. weights: dL/dW = X^T @ dL/dY
        let grad_weights = input.t().dot(grad_output);
        gradients.insert("weight".to_string(), grad_weights);

        // Gradient w.r.t. biases: dL/db = sum(dL/dY, axis=0)
        let grad_biases = grad_output
            .sum_axis(scirs2_core::ndarray::Axis(0))
            .insert_axis(scirs2_core::ndarray::Axis(0));
        gradients.insert("bias".to_string(), grad_biases);

        Ok(gradients)
    }

    fn parameters(&self) -> &HashMap<String, Array<f64, Ix2>> {
        &self.parameters
    }

    fn parameters_mut(&mut self) -> &mut HashMap<String, Array<f64, Ix2>> {
        &mut self.parameters
    }

    fn set_parameters(&mut self, parameters: HashMap<String, Array<f64, Ix2>>) {
        self.parameters = parameters;
    }

    fn reset_parameters(&mut self) {
        self.xavier_init();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_linear_model_creation() {
        let model = LinearModel::new(10, 5);
        assert_eq!(model.input_dim(), 10);
        assert_eq!(model.output_dim(), 5);
        assert_eq!(model.parameters().len(), 2);
    }

    #[test]
    fn test_linear_model_forward() {
        let model = LinearModel::new(3, 2);
        let input = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let output = model.forward(&input.view()).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_linear_model_backward() {
        let model = LinearModel::new(3, 2);
        let input = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let grad_output = arr2(&[[1.0, 1.0], [1.0, 1.0]]);

        let gradients = model.backward(&input.view(), &grad_output.view()).unwrap();

        assert!(gradients.contains_key("weight"));
        assert!(gradients.contains_key("bias"));
        assert_eq!(gradients["weight"].shape(), &[3, 2]);
        assert_eq!(gradients["bias"].shape(), &[1, 2]);
    }

    #[test]
    fn test_model_state_dict() {
        let model = LinearModel::new(2, 2);
        let state = model.state_dict();
        assert_eq!(state.len(), 2);
        assert!(state.contains_key("weight"));
        assert!(state.contains_key("bias"));
    }

    #[test]
    fn test_model_load_state() {
        let mut model = LinearModel::new(2, 2);
        let state = model.state_dict();

        // Modify parameters
        model.parameters_mut().get_mut("weight").unwrap()[[0, 0]] = 99.0;

        // Load original state
        model.load_state_dict(state.clone()).unwrap();

        // Verify state was restored
        assert_eq!(model.parameters().get("weight").unwrap()[[0, 0]], 0.0);
    }

    #[test]
    fn test_num_parameters() {
        let model = LinearModel::new(10, 5);
        // 10*5 weights + 5 biases = 55
        assert_eq!(model.num_parameters(), 55);
    }
}
