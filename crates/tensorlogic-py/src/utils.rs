//! Utility functions and context managers for TensorLogic Python bindings
//!
//! This module provides convenient helpers, context managers, and custom
//! exception types for better Python integration.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;
use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::EinsumGraph;
use tensorlogic_scirs_backend::Scirs2Exec;

use crate::compiler::PyCompilationConfig;
use crate::numpy_conversion::arrayd_to_numpy;
use crate::types::{PyEinsumGraph, PyTLExpr};

// Custom exceptions for better error handling
create_exception!(pytensorlogic, CompilationError, PyException);
create_exception!(pytensorlogic, ExecutionError, PyException);
create_exception!(pytensorlogic, ValidationError, PyException);
create_exception!(pytensorlogic, BackendError, PyException);
create_exception!(pytensorlogic, ConfigurationError, PyException);

/// Execution context manager for managed graph execution
///
/// Provides automatic resource management and cleanup for graph execution.
#[pyclass(name = "ExecutionContext")]
pub struct PyExecutionContext {
    graph: PyEinsumGraph,
    executor: Option<Scirs2Exec>,
    results: Vec<HashMap<String, ArrayD<f64>>>,
}

#[pymethods]
impl PyExecutionContext {
    #[new]
    fn new(graph: PyEinsumGraph) -> Self {
        PyExecutionContext {
            graph,
            executor: Some(Scirs2Exec::new()),
            results: Vec::new(),
        }
    }

    /// Execute the graph with given inputs
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let executor = self
            .executor
            .as_mut()
            .ok_or_else(|| ExecutionError::new_err("ExecutionContext has been closed"))?;

        // Clear executor state for new execution
        *executor = Scirs2Exec::new();

        // Add input tensors
        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;
            let array: PyReadonlyArrayDyn<f64> = value.extract()?;
            let owned = array.as_array().to_owned();
            executor.add_tensor(name, owned);
        }

        // Execute
        let result = executor
            .forward(&self.graph.inner)
            .map_err(|e| ExecutionError::new_err(format!("Execution failed: {}", e)))?;

        // Store result
        let mut result_map = HashMap::new();
        result_map.insert("output".to_string(), result.clone());
        self.results.push(result_map);

        // Convert to Python dict
        let output_dict = PyDict::new(py);
        let py_array = arrayd_to_numpy(py, &result)?;
        output_dict.set_item("output", py_array)?;

        Ok(output_dict)
    }

    /// Get all results from this context
    fn get_results<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let results = pyo3::types::PyList::empty(py);
        for result_map in &self.results {
            let dict = PyDict::new(py);
            for (name, array) in result_map {
                let py_array = arrayd_to_numpy(py, array)?;
                dict.set_item(name, py_array)?;
            }
            results.append(dict)?;
        }
        Ok(results)
    }

    /// Get number of executions performed
    fn execution_count(&self) -> usize {
        self.results.len()
    }

    /// Clear stored results
    fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Context manager enter
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager exit
    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, pyo3::types::PyAny>,
        _exc_val: &Bound<'_, pyo3::types::PyAny>,
        _exc_tb: &Bound<'_, pyo3::types::PyAny>,
    ) -> bool {
        // Cleanup: drop executor
        self.executor = None;
        false
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecutionContext(executions={}, active={})",
            self.results.len(),
            self.executor.is_some()
        )
    }
}

/// Compilation context manager for managed compilation
#[pyclass(name = "CompilationContext")]
pub struct PyCompilationContext {
    config: Option<PyCompilationConfig>,
    compiled_graphs: Vec<(String, EinsumGraph)>,
}

#[pymethods]
impl PyCompilationContext {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyCompilationConfig>) -> Self {
        PyCompilationContext {
            config,
            compiled_graphs: Vec::new(),
        }
    }

    /// Compile an expression within this context
    #[pyo3(signature = (expr, name=None))]
    fn compile(&mut self, expr: &PyTLExpr, name: Option<String>) -> PyResult<PyEinsumGraph> {
        // Note: config is stored but not used in basic compile_to_einsum
        // Future: use compile_to_einsum_with_context for full config support
        let _ = &self.config;

        let graph = compile_to_einsum(&expr.inner)
            .map_err(|e| CompilationError::new_err(format!("Compilation failed: {}", e)))?;

        // Store with name
        let graph_name = name.unwrap_or_else(|| format!("graph_{}", self.compiled_graphs.len()));
        self.compiled_graphs.push((graph_name, graph.clone()));

        Ok(PyEinsumGraph { inner: graph })
    }

    /// Get all compiled graphs
    fn get_graphs(&self) -> Vec<(String, PyEinsumGraph)> {
        self.compiled_graphs
            .iter()
            .map(|(name, graph)| {
                (
                    name.clone(),
                    PyEinsumGraph {
                        inner: graph.clone(),
                    },
                )
            })
            .collect()
    }

    /// Get a specific graph by name
    fn get_graph(&self, name: &str) -> Option<PyEinsumGraph> {
        self.compiled_graphs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, graph)| PyEinsumGraph {
                inner: graph.clone(),
            })
    }

    /// Get number of compiled graphs
    fn graph_count(&self) -> usize {
        self.compiled_graphs.len()
    }

    /// Context manager enter
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager exit
    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, pyo3::types::PyAny>,
        _exc_val: &Bound<'_, pyo3::types::PyAny>,
        _exc_tb: &Bound<'_, pyo3::types::PyAny>,
    ) -> bool {
        false
    }

    fn __repr__(&self) -> String {
        format!(
            "CompilationContext(graphs={}, config={})",
            self.compiled_graphs.len(),
            self.config.is_some()
        )
    }
}

/// Quick compile and execute in one step
///
/// Convenience function for simple one-shot computations.
///
/// Args:
///     expr: TensorLogic expression to compile and execute
///     inputs: Dictionary of input tensors
///     config: Optional compilation configuration
///
/// Returns:
///     Dictionary of output tensors
///
/// Example:
///     >>> result = tl.quick_execute(expr, {"data": np.array([1, 2, 3])})
#[pyfunction(name = "quick_execute")]
#[pyo3(signature = (expr, inputs, config=None))]
pub fn py_quick_execute<'py>(
    py: Python<'py>,
    expr: &PyTLExpr,
    inputs: &Bound<'py, PyDict>,
    config: Option<PyCompilationConfig>,
) -> PyResult<Bound<'py, PyDict>> {
    // Note: config is accepted but not used in basic compile_to_einsum
    let _ = config;

    let graph = compile_to_einsum(&expr.inner)
        .map_err(|e| CompilationError::new_err(format!("Compilation failed: {}", e)))?;

    // Execute
    let mut executor = Scirs2Exec::new();
    for (key, value) in inputs.iter() {
        let name: String = key.extract()?;
        let array: PyReadonlyArrayDyn<f64> = value.extract()?;
        let owned = array.as_array().to_owned();
        executor.add_tensor(name, owned);
    }

    let result = executor
        .forward(&graph)
        .map_err(|e| ExecutionError::new_err(format!("Execution failed: {}", e)))?;

    // Convert to Python dict
    let output_dict = PyDict::new(py);
    let py_array = arrayd_to_numpy(py, &result)?;
    output_dict.set_item("output", py_array)?;

    Ok(output_dict)
}

/// Validate input tensors against expected schema
///
/// Args:
///     inputs: Dictionary of input tensors
///     expected_names: List of expected input names
///     min_dims: Minimum number of dimensions (optional)
///     max_dims: Maximum number of dimensions (optional)
///
/// Raises:
///     ValidationError: If inputs don't match expected schema
#[pyfunction(name = "validate_inputs")]
#[pyo3(signature = (inputs, expected_names, min_dims=None, max_dims=None))]
pub fn py_validate_inputs<'py>(
    inputs: &Bound<'py, PyDict>,
    expected_names: Vec<String>,
    min_dims: Option<usize>,
    max_dims: Option<usize>,
) -> PyResult<()> {
    // Check all expected names are present
    for name in &expected_names {
        if !inputs.contains(name)? {
            return Err(ValidationError::new_err(format!(
                "Missing required input: '{}'",
                name
            )));
        }
    }

    // Validate dimensions
    for (key, value) in inputs.iter() {
        let name: String = key.extract()?;
        let array: PyReadonlyArrayDyn<f64> = value.extract()?;
        let ndim = array.ndim();

        if let Some(min) = min_dims {
            if ndim < min {
                return Err(ValidationError::new_err(format!(
                    "Input '{}' has {} dimensions, expected at least {}",
                    name, ndim, min
                )));
            }
        }

        if let Some(max) = max_dims {
            if ndim > max {
                return Err(ValidationError::new_err(format!(
                    "Input '{}' has {} dimensions, expected at most {}",
                    name, ndim, max
                )));
            }
        }
    }

    Ok(())
}

/// Create multiple graphs from expressions in batch
///
/// Args:
///     expressions: List of expressions to compile
///     config: Optional compilation configuration
///
/// Returns:
///     List of compiled graphs
#[pyfunction(name = "batch_compile")]
#[pyo3(signature = (expressions, config=None))]
pub fn py_batch_compile(
    expressions: Vec<PyTLExpr>,
    config: Option<PyCompilationConfig>,
) -> PyResult<Vec<PyEinsumGraph>> {
    // Note: config is accepted but not used in basic compile_to_einsum
    let _ = config;

    let mut graphs = Vec::new();
    for (i, expr) in expressions.iter().enumerate() {
        let graph = compile_to_einsum(&expr.inner).map_err(|e| {
            CompilationError::new_err(format!("Compilation failed for expression {}: {}", i, e))
        })?;

        graphs.push(PyEinsumGraph { inner: graph });
    }

    Ok(graphs)
}

/// Predict on multiple inputs using the same graph
///
/// Args:
///     graph: Compiled graph
///     inputs_list: List of input dictionaries
///
/// Returns:
///     List of output dictionaries
#[pyfunction(name = "batch_predict")]
pub fn py_batch_predict<'py>(
    py: Python<'py>,
    graph: &PyEinsumGraph,
    inputs_list: Vec<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let results = pyo3::types::PyList::empty(py);

    for inputs in inputs_list {
        let mut executor = Scirs2Exec::new();

        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;
            let array: PyReadonlyArrayDyn<f64> = value.extract()?;
            let owned = array.as_array().to_owned();
            executor.add_tensor(name, owned);
        }

        let result = executor
            .forward(&graph.inner)
            .map_err(|e| ExecutionError::new_err(format!("Batch prediction failed: {}", e)))?;

        let output_dict = PyDict::new(py);
        let py_array = arrayd_to_numpy(py, &result)?;
        output_dict.set_item("output", py_array)?;
        results.append(output_dict)?;
    }

    Ok(results)
}

/// Create an execution context
#[pyfunction(name = "execution_context")]
pub fn py_execution_context(graph: PyEinsumGraph) -> PyExecutionContext {
    PyExecutionContext::new(graph)
}

/// Create a compilation context
#[pyfunction(name = "compilation_context")]
#[pyo3(signature = (config=None))]
pub fn py_compilation_context(config: Option<PyCompilationConfig>) -> PyCompilationContext {
    PyCompilationContext::new(config)
}

/// Register utility module functions and exceptions
pub fn register_utils_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register custom exceptions
    m.add("CompilationError", m.py().get_type::<CompilationError>())?;
    m.add("ExecutionError", m.py().get_type::<ExecutionError>())?;
    m.add("ValidationError", m.py().get_type::<ValidationError>())?;
    m.add("BackendError", m.py().get_type::<BackendError>())?;
    m.add(
        "ConfigurationError",
        m.py().get_type::<ConfigurationError>(),
    )?;

    // Register classes
    m.add_class::<PyExecutionContext>()?;
    m.add_class::<PyCompilationContext>()?;

    // Register functions
    m.add_function(wrap_pyfunction!(py_quick_execute, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_compile, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_predict, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution_context, m)?)?;
    m.add_function(wrap_pyfunction!(py_compilation_context, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_context_creation() {
        // Basic structure test
    }
}
