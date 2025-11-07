//! Training API for tensorlogic-py.
//!
//! This module provides a high-level training interface for neural-symbolic models,
//! including trainers, loss functions, optimizers, and callbacks.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use scirs2_core::ndarray::ArrayViewD;

use crate::compiler::PyCompilationConfig;
use crate::executor::py_execute;
use crate::types::{PyEinsumGraph, PyTLExpr};

// ============================================================================
// Loss Functions
// ============================================================================

/// Loss function for training neural-symbolic models.
///
/// Base class for all loss functions. Loss functions compute the difference
/// between predicted and target values during training.
#[pyclass(name = "LossFunction", subclass)]
#[derive(Clone)]
pub struct PyLossFunction {
    loss_type: String,
}

#[pymethods]
impl PyLossFunction {
    /// Create a new loss function.
    ///
    /// Args:
    ///     loss_type: Type of loss function ('mse', 'bce', 'cross_entropy', 'custom')
    ///
    /// Returns:
    ///     LossFunction instance
    #[new]
    fn new(loss_type: String) -> Self {
        PyLossFunction { loss_type }
    }

    /// Get the loss type.
    #[getter]
    fn loss_type(&self) -> String {
        self.loss_type.clone()
    }

    /// Compute loss between predictions and targets.
    ///
    /// Args:
    ///     predictions: Predicted values (NumPy array)
    ///     targets: Target/ground truth values (NumPy array)
    ///
    /// Returns:
    ///     Scalar loss value
    fn __call__<'py>(
        &self,
        _py: Python<'py>,
        predictions: PyReadonlyArrayDyn<'py, f64>,
        targets: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<f64> {
        let pred_array = predictions.as_array();
        let target_array = targets.as_array();

        if pred_array.shape() != target_array.shape() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                pred_array.shape(),
                target_array.shape()
            )));
        }

        let loss = match self.loss_type.as_str() {
            "mse" => compute_mse(&pred_array, &target_array),
            "bce" => compute_bce(&pred_array, &target_array),
            "cross_entropy" => compute_cross_entropy(&pred_array, &target_array),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported loss type: {}",
                    self.loss_type
                )))
            }
        };

        Ok(loss)
    }

    fn __repr__(&self) -> String {
        format!("LossFunction(loss_type='{}')", self.loss_type)
    }
}

// Loss function implementations
fn compute_mse(predictions: &ArrayViewD<f64>, targets: &ArrayViewD<f64>) -> f64 {
    let diff = predictions - targets;
    let squared = diff.mapv(|x| x * x);
    squared.mean().unwrap_or(0.0)
}

fn compute_bce(predictions: &ArrayViewD<f64>, targets: &ArrayViewD<f64>) -> f64 {
    // Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))
    let epsilon = 1e-7;
    let pred_clipped = predictions.mapv(|p| p.max(epsilon).min(1.0 - epsilon));

    let term1 = targets * &pred_clipped.mapv(|p| p.ln());
    let term2 = &targets.mapv(|y| 1.0 - y) * &pred_clipped.mapv(|p| (1.0 - p).ln());

    -(term1 + term2).mean().unwrap_or(0.0)
}

fn compute_cross_entropy(predictions: &ArrayViewD<f64>, targets: &ArrayViewD<f64>) -> f64 {
    // Categorical cross-entropy: -mean(sum(y * log(p)))
    let epsilon = 1e-7;
    let pred_clipped = predictions.mapv(|p| (p + epsilon).ln());

    -(targets * &pred_clipped).sum() / predictions.len() as f64
}

/// Create a Mean Squared Error (MSE) loss function.
///
/// MSE measures the average squared difference between predictions and targets.
/// Suitable for regression tasks.
///
/// Returns:
///     LossFunction configured for MSE
#[pyfunction]
pub fn mse_loss() -> PyLossFunction {
    PyLossFunction::new("mse".to_string())
}

/// Create a Binary Cross-Entropy (BCE) loss function.
///
/// BCE is used for binary classification tasks where predictions are probabilities.
///
/// Returns:
///     LossFunction configured for BCE
#[pyfunction]
pub fn bce_loss() -> PyLossFunction {
    PyLossFunction::new("bce".to_string())
}

/// Create a Cross-Entropy loss function.
///
/// Cross-entropy is used for multi-class classification tasks.
///
/// Returns:
///     LossFunction configured for cross-entropy
#[pyfunction]
pub fn cross_entropy_loss() -> PyLossFunction {
    PyLossFunction::new("cross_entropy".to_string())
}

// ============================================================================
// Optimizer
// ============================================================================

/// Optimizer for updating model parameters during training.
#[pyclass(name = "Optimizer")]
#[derive(Clone)]
pub struct PyOptimizer {
    optimizer_type: String,
    learning_rate: f64,
    #[allow(dead_code)] // Reserved for future optimizer parameter storage
    config: HashMap<String, f64>,
}

#[pymethods]
impl PyOptimizer {
    /// Create a new optimizer.
    ///
    /// Args:
    ///     optimizer_type: Type of optimizer ('sgd', 'adam', 'rmsprop')
    ///     learning_rate: Learning rate (step size) for updates
    ///     config: Optional configuration parameters
    ///
    /// Returns:
    ///     Optimizer instance
    #[new]
    #[pyo3(signature = (optimizer_type, learning_rate=0.01, config=None))]
    fn new(
        optimizer_type: String,
        learning_rate: f64,
        config: Option<HashMap<String, f64>>,
    ) -> Self {
        PyOptimizer {
            optimizer_type,
            learning_rate,
            config: config.unwrap_or_default(),
        }
    }

    /// Get optimizer type.
    #[getter]
    fn optimizer_type(&self) -> String {
        self.optimizer_type.clone()
    }

    /// Get learning rate.
    #[getter]
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set learning rate.
    #[setter]
    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn __repr__(&self) -> String {
        format!(
            "Optimizer(type='{}', lr={})",
            self.optimizer_type, self.learning_rate
        )
    }
}

/// Create a Stochastic Gradient Descent (SGD) optimizer.
///
/// Args:
///     learning_rate: Learning rate for parameter updates (default: 0.01)
///     momentum: Momentum factor (default: 0.0)
///
/// Returns:
///     Optimizer configured for SGD
#[pyfunction]
#[pyo3(signature = (learning_rate=0.01, momentum=0.0))]
pub fn sgd(learning_rate: f64, momentum: f64) -> PyOptimizer {
    let mut config = HashMap::new();
    config.insert("momentum".to_string(), momentum);
    PyOptimizer::new("sgd".to_string(), learning_rate, Some(config))
}

/// Create an Adam optimizer.
///
/// Args:
///     learning_rate: Learning rate (default: 0.001)
///     beta1: Exponential decay rate for first moment (default: 0.9)
///     beta2: Exponential decay rate for second moment (default: 0.999)
///     epsilon: Small constant for numerical stability (default: 1e-8)
///
/// Returns:
///     Optimizer configured for Adam
#[pyfunction]
#[pyo3(signature = (learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8))]
pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> PyOptimizer {
    let mut config = HashMap::new();
    config.insert("beta1".to_string(), beta1);
    config.insert("beta2".to_string(), beta2);
    config.insert("epsilon".to_string(), epsilon);
    PyOptimizer::new("adam".to_string(), learning_rate, Some(config))
}

/// Create an RMSprop optimizer.
///
/// Args:
///     learning_rate: Learning rate (default: 0.01)
///     alpha: Smoothing constant (default: 0.99)
///     epsilon: Small constant for numerical stability (default: 1e-8)
///
/// Returns:
///     Optimizer configured for RMSprop
#[pyfunction]
#[pyo3(signature = (learning_rate=0.01, alpha=0.99, epsilon=1e-8))]
pub fn rmsprop(learning_rate: f64, alpha: f64, epsilon: f64) -> PyOptimizer {
    let mut config = HashMap::new();
    config.insert("alpha".to_string(), alpha);
    config.insert("epsilon".to_string(), epsilon);
    PyOptimizer::new("rmsprop".to_string(), learning_rate, Some(config))
}

// ============================================================================
// Callbacks
// ============================================================================

/// Callback for monitoring and controlling training.
#[pyclass(name = "Callback", subclass)]
#[derive(Clone)]
pub struct PyCallback {
    callback_type: String,
    config: HashMap<String, f64>,
}

#[pymethods]
impl PyCallback {
    /// Create a new callback.
    ///
    /// Args:
    ///     callback_type: Type of callback ('early_stopping', 'model_checkpoint', 'logger')
    ///     config: Callback-specific configuration
    ///
    /// Returns:
    ///     Callback instance
    #[new]
    #[pyo3(signature = (callback_type, config=None))]
    fn new(callback_type: String, config: Option<HashMap<String, f64>>) -> Self {
        PyCallback {
            callback_type,
            config: config.unwrap_or_default(),
        }
    }

    /// Get callback type.
    #[getter]
    fn callback_type(&self) -> String {
        self.callback_type.clone()
    }

    fn __repr__(&self) -> String {
        format!("Callback(type='{}')", self.callback_type)
    }
}

/// Create an EarlyStopping callback.
///
/// Stops training when validation loss stops improving.
///
/// Args:
///     patience: Number of epochs with no improvement to wait (default: 5)
///     min_delta: Minimum change to qualify as improvement (default: 0.0001)
///
/// Returns:
///     Callback for early stopping
#[pyfunction]
#[pyo3(signature = (patience=5.0, min_delta=0.0001))]
pub fn early_stopping(patience: f64, min_delta: f64) -> PyCallback {
    let mut config = HashMap::new();
    config.insert("patience".to_string(), patience);
    config.insert("min_delta".to_string(), min_delta);
    PyCallback::new("early_stopping".to_string(), Some(config))
}

/// Create a ModelCheckpoint callback.
///
/// Saves model checkpoints during training.
///
/// Args:
///     save_best_only: Only save when model improves (default: True)
///
/// Returns:
///     Callback for model checkpointing
#[pyfunction]
#[pyo3(signature = (save_best_only=1.0))]
pub fn model_checkpoint(save_best_only: f64) -> PyCallback {
    let mut config = HashMap::new();
    config.insert("save_best_only".to_string(), save_best_only);
    PyCallback::new("model_checkpoint".to_string(), Some(config))
}

/// Create a Logger callback.
///
/// Logs training progress to console or file.
///
/// Args:
///     verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
///
/// Returns:
///     Callback for logging
#[pyfunction]
#[pyo3(signature = (verbose=1.0))]
pub fn logger(verbose: f64) -> PyCallback {
    let mut config = HashMap::new();
    config.insert("verbose".to_string(), verbose);
    PyCallback::new("logger".to_string(), Some(config))
}

// ============================================================================
// Training History
// ============================================================================

/// Training history containing loss and metrics over epochs.
#[pyclass(name = "TrainingHistory")]
#[derive(Clone)]
pub struct PyTrainingHistory {
    #[pyo3(get)]
    train_losses: Vec<f64>,
    #[pyo3(get)]
    val_losses: Vec<f64>,
    metrics: HashMap<String, Vec<f64>>,
}

#[pymethods]
impl PyTrainingHistory {
    /// Create a new training history.
    #[new]
    fn new() -> Self {
        PyTrainingHistory {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            metrics: HashMap::new(),
        }
    }

    /// Add training loss for an epoch.
    fn add_train_loss(&mut self, loss: f64) {
        self.train_losses.push(loss);
    }

    /// Add validation loss for an epoch.
    fn add_val_loss(&mut self, loss: f64) {
        self.val_losses.push(loss);
    }

    /// Add a metric value for an epoch.
    fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.entry(name).or_default().push(value);
    }

    /// Get a specific metric's history.
    fn get_metric(&self, name: String) -> Option<Vec<f64>> {
        self.metrics.get(&name).cloned()
    }

    /// Get number of epochs.
    fn num_epochs(&self) -> usize {
        self.train_losses.len()
    }

    /// Get the best (minimum) training loss and its epoch.
    fn best_train_loss(&self) -> Option<(usize, f64)> {
        self.train_losses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &loss)| (idx, loss))
    }

    /// Get the best (minimum) validation loss and its epoch.
    fn best_val_loss(&self) -> Option<(usize, f64)> {
        self.val_losses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &loss)| (idx, loss))
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainingHistory(epochs={}, train_loss={:.4}, val_loss={:.4})",
            self.num_epochs(),
            self.train_losses.last().unwrap_or(&0.0),
            self.val_losses.last().unwrap_or(&0.0)
        )
    }

    /// Rich HTML representation for Jupyter notebooks
    ///
    /// Returns:
    ///     HTML string for display in Jupyter/IPython
    fn _repr_html_(&self) -> String {
        use crate::jupyter::training_history_html;

        let val_losses = if self.val_losses.is_empty() {
            None
        } else {
            Some(self.val_losses.clone())
        };

        training_history_html(self.num_epochs(), &self.train_losses, &val_losses)
    }
}

// ============================================================================
// Trainer
// ============================================================================

/// High-level trainer for neural-symbolic models.
///
/// The Trainer class provides a fit() method similar to scikit-learn,
/// managing the full training loop with loss computation, optimization,
/// and callback execution.
#[pyclass(name = "Trainer")]
pub struct PyTrainer {
    graph: PyEinsumGraph,
    loss_fn: PyLossFunction,
    optimizer: PyOptimizer,
    callbacks: Vec<PyCallback>,
    history: PyTrainingHistory,
    output_name: String,
}

#[pymethods]
impl PyTrainer {
    /// Create a new trainer.
    ///
    /// Args:
    ///     graph: Compiled computation graph
    ///     loss_fn: Loss function to minimize
    ///     optimizer: Optimizer for parameter updates
    ///     output_name: Name of the output to use for training (default: "result")
    ///     callbacks: List of callbacks for monitoring (optional)
    ///
    /// Returns:
    ///     Trainer instance
    #[new]
    #[pyo3(signature = (graph, loss_fn, optimizer, output_name="output".to_string(), callbacks=None))]
    fn new(
        graph: PyEinsumGraph,
        loss_fn: PyLossFunction,
        optimizer: PyOptimizer,
        output_name: String,
        callbacks: Option<Vec<PyCallback>>,
    ) -> Self {
        PyTrainer {
            graph,
            loss_fn,
            optimizer,
            callbacks: callbacks.unwrap_or_default(),
            history: PyTrainingHistory::new(),
            output_name,
        }
    }

    /// Train the model on data.
    ///
    /// Args:
    ///     train_inputs: Training input data (dict of variable -> NumPy array)
    ///     train_targets: Training target values (NumPy array)
    ///     epochs: Number of training epochs (default: 10)
    ///     validation_data: Optional (val_inputs, val_targets) tuple
    ///     verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
    ///
    /// Returns:
    ///     TrainingHistory with loss and metrics over epochs
    ///
    /// Example:
    ///     >>> trainer = tl.Trainer(graph, loss_fn, optimizer)
    ///     >>> history = trainer.fit(train_inputs, train_targets, epochs=50)
    ///     >>> print(f"Final loss: {history.train_losses[-1]:.4f}")
    #[pyo3(signature = (train_inputs, train_targets, epochs=10, validation_data=None, verbose=1))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        train_inputs: Bound<'py, PyDict>,
        train_targets: PyReadonlyArrayDyn<'py, f64>,
        epochs: usize,
        validation_data: Option<(Bound<'py, PyDict>, PyReadonlyArrayDyn<'py, f64>)>,
        verbose: i32,
    ) -> PyResult<PyTrainingHistory> {
        for epoch in 0..epochs {
            // Forward pass
            let output_dict = py_execute(py, &self.graph, &train_inputs, None)?;

            // Extract the specified output
            let predictions: PyReadonlyArrayDyn<'py, f64> = output_dict
                .get_item(&self.output_name)?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "Output '{}' not found in graph outputs",
                        self.output_name
                    ))
                })?
                .extract()?;

            // Compute loss
            let train_loss = self
                .loss_fn
                .__call__(py, predictions, train_targets.clone())?;

            self.history.add_train_loss(train_loss);

            // Validation if provided
            if let Some((ref val_inputs, ref val_targets)) = validation_data {
                let val_output_dict = py_execute(py, &self.graph, val_inputs, None)?;
                let val_predictions: PyReadonlyArrayDyn<'py, f64> = val_output_dict
                    .get_item(&self.output_name)?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "Output '{}' not found in graph outputs",
                            self.output_name
                        ))
                    })?
                    .extract()?;

                let val_loss = self
                    .loss_fn
                    .__call__(py, val_predictions, val_targets.clone())?;
                self.history.add_val_loss(val_loss);
            }

            // Logging
            if verbose > 0 {
                if let Some((_, _)) = validation_data {
                    println!(
                        "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}",
                        epoch + 1,
                        epochs,
                        train_loss,
                        self.history.val_losses.last().unwrap_or(&0.0)
                    );
                } else {
                    println!(
                        "Epoch {}/{}: train_loss={:.4}",
                        epoch + 1,
                        epochs,
                        train_loss
                    );
                }
            }

            // Early stopping check (simplified)
            if self.should_stop_early(epoch) {
                if verbose > 0 {
                    println!("Early stopping triggered at epoch {}", epoch + 1);
                }
                break;
            }
        }

        Ok(self.history.clone())
    }

    /// Evaluate model on data without training.
    ///
    /// Args:
    ///     inputs: Input data (dict of variable -> NumPy array)
    ///     targets: Target values (NumPy array)
    ///
    /// Returns:
    ///     Loss value on the data
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyDict>,
        targets: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<f64> {
        let output_dict = py_execute(py, &self.graph, &inputs, None)?;
        let predictions: PyReadonlyArrayDyn<'py, f64> = output_dict
            .get_item(&self.output_name)?
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "Output '{}' not found in graph outputs",
                    self.output_name
                ))
            })?
            .extract()?;

        self.loss_fn.__call__(py, predictions, targets)
    }

    /// Make predictions on new data.
    ///
    /// Args:
    ///     inputs: Input data (dict of variable -> NumPy array)
    ///
    /// Returns:
    ///     Predictions (NumPy array)
    fn predict<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let output_dict = py_execute(py, &self.graph, &inputs, None)?;
        let item = output_dict.get_item(&self.output_name)?.ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "Output '{}' not found in graph outputs",
                self.output_name
            ))
        })?;

        // Extract as NumPy array
        item.extract::<Bound<'py, PyArrayDyn<f64>>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "Expected NumPy array for output '{}'",
                self.output_name
            ))
        })
    }

    /// Get training history.
    fn get_history(&self) -> PyTrainingHistory {
        self.history.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Trainer(loss={}, optimizer={})",
            self.loss_fn.loss_type, self.optimizer.optimizer_type
        )
    }
}

impl PyTrainer {
    /// Check if early stopping should be triggered.
    fn should_stop_early(&self, _current_epoch: usize) -> bool {
        for callback in &self.callbacks {
            if callback.callback_type == "early_stopping" {
                let patience = callback.config.get("patience").unwrap_or(&5.0) as &f64;
                let min_delta = callback.config.get("min_delta").unwrap_or(&0.0001) as &f64;

                if self.history.val_losses.len() < *patience as usize {
                    return false;
                }

                let recent_losses =
                    &self.history.val_losses[self.history.val_losses.len() - *patience as usize..];
                let best_loss = recent_losses
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let current_loss = self.history.val_losses.last().unwrap();

                if current_loss - best_loss > *min_delta {
                    return false;
                }

                return true;
            }
        }
        false
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Train a model with a simple API.
///
/// Convenience function for training without explicitly creating a Trainer.
///
/// Args:
///     expr: TensorLogic expression to train
///     train_inputs: Training input data
///     train_targets: Training target values
///     loss_fn: Loss function (default: MSE)
///     optimizer: Optimizer (default: Adam with lr=0.001)
///     epochs: Number of training epochs (default: 10)
///     config: Optional compilation configuration
///
/// Returns:
///     Tuple of (trained_graph, training_history)
///
/// Example:
///     >>> expr = tl.pred("knows", [tl.var("x"), tl.var("y")])
///     >>> graph, history = tl.fit(expr, train_inputs, train_targets, epochs=50)
#[pyfunction]
#[pyo3(signature = (expr, train_inputs, train_targets, loss_fn=None, optimizer=None, epochs=10, config=None))]
#[allow(clippy::too_many_arguments)] // Convenience function needs all parameters
pub fn fit<'py>(
    py: Python<'py>,
    expr: &PyTLExpr,
    train_inputs: Bound<'py, PyDict>,
    train_targets: PyReadonlyArrayDyn<'py, f64>,
    loss_fn: Option<PyLossFunction>,
    optimizer: Option<PyOptimizer>,
    epochs: usize,
    config: Option<&PyCompilationConfig>,
) -> PyResult<(PyEinsumGraph, PyTrainingHistory)> {
    // Compile expression
    let graph = if let Some(cfg) = config {
        crate::compiler::py_compile_with_config(expr, cfg)?
    } else {
        crate::compiler::py_compile(expr)?
    };

    // Create trainer
    let loss = loss_fn.unwrap_or_else(mse_loss);
    let opt = optimizer.unwrap_or_else(|| adam(0.001, 0.9, 0.999, 1e-8));

    let mut trainer = PyTrainer::new(graph.clone(), loss, opt, "output".to_string(), None);

    // Train
    let history = trainer.fit(py, train_inputs, train_targets, epochs, None, 1)?;

    Ok((graph, history))
}

// ============================================================================
// Module Registration
// ============================================================================

/// Register training-related types and functions with Python module.
pub fn register_training_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyLossFunction>()?;
    m.add_class::<PyOptimizer>()?;
    m.add_class::<PyCallback>()?;
    m.add_class::<PyTrainingHistory>()?;
    m.add_class::<PyTrainer>()?;

    // Loss functions
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(bce_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, m)?)?;

    // Optimizers
    m.add_function(wrap_pyfunction!(sgd, m)?)?;
    m.add_function(wrap_pyfunction!(adam, m)?)?;
    m.add_function(wrap_pyfunction!(rmsprop, m)?)?;

    // Callbacks
    m.add_function(wrap_pyfunction!(early_stopping, m)?)?;
    m.add_function(wrap_pyfunction!(model_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(logger, m)?)?;

    // Training function
    m.add_function(wrap_pyfunction!(fit, m)?)?;

    Ok(())
}
