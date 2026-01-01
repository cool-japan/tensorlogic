//! Compilation functions for Python

use crate::adapters::PyCompilerContext;
use crate::types::{PyEinsumGraph, PyTLExpr};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tensorlogic_compiler::{
    compile_to_einsum, compile_to_einsum_with_config, compile_to_einsum_with_context,
    CompilationConfig,
};

/// Compilation configuration
#[pyclass(name = "CompilationConfig")]
#[derive(Clone)]
pub struct PyCompilationConfig {
    pub inner: CompilationConfig,
}

impl Default for PyCompilationConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyCompilationConfig {
    #[new]
    pub fn new() -> Self {
        PyCompilationConfig {
            inner: CompilationConfig::soft_differentiable(),
        }
    }

    /// Create a configuration for soft differentiable logic (default)
    ///
    /// Best for neural network training with gradient descent.
    /// Uses smooth approximations that are differentiable everywhere.
    #[staticmethod]
    fn soft_differentiable() -> Self {
        PyCompilationConfig {
            inner: CompilationConfig::soft_differentiable(),
        }
    }

    /// Create a configuration for hard Boolean logic
    ///
    /// Uses discrete Boolean operations (min/max).
    /// Not differentiable but provides exact Boolean semantics.
    #[staticmethod]
    fn hard_boolean() -> Self {
        PyCompilationConfig {
            inner: CompilationConfig::hard_boolean(),
        }
    }

    /// Create a configuration for Gödel fuzzy logic
    ///
    /// Uses min for AND, max for OR, standard complement for NOT.
    #[staticmethod]
    fn fuzzy_godel() -> Self {
        PyCompilationConfig {
            inner: CompilationConfig::fuzzy_godel(),
        }
    }

    /// Create a configuration for Product fuzzy logic
    ///
    /// Uses product for AND, probabilistic sum for OR.
    #[staticmethod]
    fn fuzzy_product() -> Self {
        PyCompilationConfig {
            inner: CompilationConfig::fuzzy_product(),
        }
    }

    /// Create a configuration for Łukasiewicz fuzzy logic
    ///
    /// Uses Łukasiewicz t-norm and t-conorm.
    #[staticmethod]
    fn fuzzy_lukasiewicz() -> Self {
        PyCompilationConfig {
            inner: CompilationConfig::fuzzy_lukasiewicz(),
        }
    }

    /// Create a configuration for probabilistic logic
    ///
    /// Interprets logical operations as probabilistic operations.
    #[staticmethod]
    fn probabilistic() -> Self {
        PyCompilationConfig {
            inner: CompilationConfig::probabilistic(),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// Rich HTML representation for Jupyter notebooks
    ///
    /// Returns:
    ///     HTML string for display in Jupyter/IPython
    fn _repr_html_(&self) -> String {
        use crate::jupyter::compilation_config_html;

        // Determine config name and description based on the config
        let (name, desc) = (
            "Compilation Config",
            "Custom configuration for logic-to-tensor compilation",
        );

        compilation_config_html(name, desc)
    }

    /// Export to JSON string
    ///
    /// Serializes the compilation configuration to a JSON string.
    /// This can be used to save configurations for later use or to transfer
    /// them between different environments.
    ///
    /// Returns:
    ///     JSON representation of the configuration
    ///
    /// Raises:
    ///     RuntimeError: If serialization fails
    ///
    /// Example:
    ///     >>> config = CompilationConfig.soft_differentiable()
    ///     >>> json_str = config.to_json()
    ///     >>> print(json_str)
    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to serialize CompilationConfig: {}", e))
        })
    }

    /// Import from JSON string
    ///
    /// Deserializes a compilation configuration from a JSON string.
    /// This allows loading previously saved configurations.
    ///
    /// Args:
    ///     json: JSON string representing a CompilationConfig
    ///
    /// Returns:
    ///     CompilationConfig instance
    ///
    /// Raises:
    ///     RuntimeError: If deserialization fails
    ///
    /// Example:
    ///     >>> json_str = '{"and_strategy":"Product",...}'
    ///     >>> config = CompilationConfig.from_json(json_str)
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let inner = serde_json::from_str(&json).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to deserialize CompilationConfig: {}", e))
        })?;
        Ok(PyCompilationConfig { inner })
    }
}

/// Compile a TLExpr into an EinsumGraph
///
/// This is the main compilation function. It takes a logical expression
/// and compiles it into a tensor computation graph that can be executed.
///
/// Args:
///     expr: The logical expression to compile
///
/// Returns:
///     An EinsumGraph representing the compiled computation
///
/// Raises:
///     RuntimeError: If compilation fails
///
/// Example:
///     >>> expr = pred("knows", [var("x"), var("y")])
///     >>> graph = compile(expr)
#[pyfunction(name = "compile")]
pub fn py_compile(expr: &PyTLExpr) -> PyResult<PyEinsumGraph> {
    match compile_to_einsum(&expr.inner) {
        Ok(graph) => Ok(PyEinsumGraph { inner: graph }),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Compilation failed: {}",
            e
        ))),
    }
}

/// Compile a TLExpr with custom configuration
///
/// Allows fine-grained control over how logical operations are compiled
/// to tensor operations.
///
/// Args:
///     expr: The logical expression to compile
///     config: Compilation configuration
///
/// Returns:
///     An EinsumGraph representing the compiled computation
///
/// Raises:
///     RuntimeError: If compilation fails
///
/// Example:
///     >>> config = CompilationConfig.hard_boolean()
///     >>> graph = compile_with_config(expr, config)
#[pyfunction(name = "compile_with_config")]
pub fn py_compile_with_config(
    expr: &PyTLExpr,
    config: &PyCompilationConfig,
) -> PyResult<PyEinsumGraph> {
    match compile_to_einsum_with_config(&expr.inner, &config.inner) {
        Ok(graph) => Ok(PyEinsumGraph { inner: graph }),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Compilation failed: {}",
            e
        ))),
    }
}

/// Compile a TensorLogic expression with a compiler context.
///
/// This allows you to specify domains and other compilation context
/// before compiling, which is necessary for expressions with quantifiers.
///
/// Args:
///     expr: The TLExpr to compile
///     context: A CompilerContext with registered domains
///
/// Returns:
///     An EinsumGraph representing the compiled computation
///
/// Raises:
///     RuntimeError: If compilation fails
///
/// Example:
///     >>> ctx = compiler_context()
///     >>> ctx.add_domain("Person", 100)
///     >>> expr = exists("y", "Person", pred("knows", [var("x"), var("y")]))
///     >>> graph = compile_with_context(expr, ctx)
#[pyfunction(name = "compile_with_context")]
pub fn py_compile_with_context(
    expr: &PyTLExpr,
    context: &mut PyCompilerContext,
) -> PyResult<PyEinsumGraph> {
    match compile_to_einsum_with_context(&expr.inner, &mut context.inner) {
        Ok(graph) => Ok(PyEinsumGraph { inner: graph }),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Compilation failed: {}",
            e
        ))),
    }
}
