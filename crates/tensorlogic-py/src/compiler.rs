//! Compilation functions for Python

use crate::adapters::PyCompilerContext;
use crate::types::{PyEinsumGraph, PyTLExpr};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tensorlogic_compiler::{compile_to_einsum, compile_to_einsum_with_context, CompilationConfig};

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

    /// Export to JSON string (placeholder for now)
    ///
    /// Note: Full serialization not yet implemented.
    /// Returns a placeholder JSON object.
    ///
    /// Returns:
    ///     JSON representation of the configuration
    ///
    /// Example:
    ///     >>> json_str = config.to_json()
    pub fn to_json(&self) -> PyResult<String> {
        // TODO: Implement proper serialization once CompilationConfig has Serialize derive
        Ok("{}".to_string())
    }

    /// Import from JSON string (placeholder for now)
    ///
    /// Note: Full deserialization not yet implemented.
    /// Returns default configuration.
    ///
    /// Args:
    ///     json: JSON string
    ///
    /// Returns:
    ///     CompilationConfig (default for now)
    ///
    /// Example:
    ///     >>> config = CompilationConfig.from_json(json_str)
    #[staticmethod]
    pub fn from_json(_json: String) -> PyResult<Self> {
        // TODO: Implement proper deserialization once CompilationConfig has Serialize derive
        Ok(PyCompilationConfig::new())
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
    _config: &PyCompilationConfig,
) -> PyResult<PyEinsumGraph> {
    // Note: We need to extend the compiler API to accept config
    // For now, compile with default config
    // TODO: Add compile_to_einsum_with_config to tensorlogic-compiler
    match compile_to_einsum(&expr.inner) {
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
