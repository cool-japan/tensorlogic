//! Python bindings for TensorLogic
//!
//! This module exposes TensorLogic's compilation and execution APIs to Python,
//! enabling researchers and practitioners to use TensorLogic from Jupyter notebooks
//! and Python workflows.

use pyo3::prelude::*;

mod adapters;
mod backend;
mod compiler;
mod dsl;
mod executor;
mod jupyter;
mod numpy_conversion;
mod persistence;
mod provenance;
mod training;
mod types;

use adapters::{py_compiler_context, py_domain_info, py_predicate_info, py_symbol_table};
use adapters::{PyCompilerContext, PyDomainInfo, PyPredicateInfo, PySymbolTable};
use backend::{
    py_get_backend_capabilities, py_get_default_backend, py_get_system_info,
    py_list_available_backends, PyBackend, PyBackendCapabilities,
};
use compiler::{py_compile, py_compile_with_config, py_compile_with_context, PyCompilationConfig};
use executor::py_execute;
use persistence::{
    py_load_full_model, py_load_model, py_model_package, py_save_full_model, py_save_model,
    PyModelPackage,
};
use provenance::{
    py_get_metadata, py_get_provenance, py_provenance_tracker, PyProvenance, PyProvenanceTracker,
    PySourceLocation, PySourceSpan,
};
use types::{PyEinsumGraph, PyTLExpr, PyTerm};

/// TensorLogic: Logic-as-tensor planning layer
///
/// Compile logical rules into tensor equations and execute them with various backends.
///
/// Example:
///     >>> import pytensorlogic as tl
///     >>> # Create a predicate
///     >>> expr = tl.pred("knows", [tl.var("x"), tl.var("y")])
///     >>> # Compile to tensor graph
///     >>> graph = tl.compile(expr)
///     >>> # Execute with inputs
///     >>> result = tl.execute(graph, {"knows": knows_matrix})
#[pymodule]
fn pytensorlogic(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register types
    m.add_class::<PyTerm>()?;
    m.add_class::<PyTLExpr>()?;
    m.add_class::<PyEinsumGraph>()?;
    m.add_class::<PyCompilationConfig>()?;

    // Register backend types
    m.add_class::<PyBackend>()?;
    m.add_class::<PyBackendCapabilities>()?;

    // Register adapter types
    m.add_class::<PyDomainInfo>()?;
    m.add_class::<PyPredicateInfo>()?;
    m.add_class::<PySymbolTable>()?;
    m.add_class::<PyCompilerContext>()?;

    // Register provenance types
    m.add_class::<PySourceLocation>()?;
    m.add_class::<PySourceSpan>()?;
    m.add_class::<PyProvenance>()?;
    m.add_class::<PyProvenanceTracker>()?;

    // Register persistence types
    m.add_class::<PyModelPackage>()?;

    // Register functions - Logical operations
    m.add_function(wrap_pyfunction!(types::py_var, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_const, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_pred, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_and, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_or, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_not, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_exists, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_forall, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_imply, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_constant, m)?)?;

    // Register arithmetic operations
    m.add_function(wrap_pyfunction!(types::py_add, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_sub, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_mul, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_div, m)?)?;

    // Register comparison operations
    m.add_function(wrap_pyfunction!(types::py_eq, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_lt, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_gt, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_lte, m)?)?;
    m.add_function(wrap_pyfunction!(types::py_gte, m)?)?;

    // Register conditional operations
    m.add_function(wrap_pyfunction!(types::py_if_then_else, m)?)?;

    // Register compilation and execution
    m.add_function(wrap_pyfunction!(py_compile, m)?)?;
    m.add_function(wrap_pyfunction!(py_compile_with_config, m)?)?;
    m.add_function(wrap_pyfunction!(py_compile_with_context, m)?)?;
    m.add_function(wrap_pyfunction!(py_execute, m)?)?;

    // Register adapter functions
    m.add_function(wrap_pyfunction!(py_domain_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_predicate_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_symbol_table, m)?)?;
    m.add_function(wrap_pyfunction!(py_compiler_context, m)?)?;

    // Register backend functions
    m.add_function(wrap_pyfunction!(py_get_backend_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(py_list_available_backends, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_default_backend, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_system_info, m)?)?;

    // Register provenance functions
    m.add_function(wrap_pyfunction!(py_get_provenance, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_provenance_tracker, m)?)?;

    // Register persistence functions
    m.add_function(wrap_pyfunction!(py_save_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_save_full_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_full_model, m)?)?;
    m.add_function(wrap_pyfunction!(py_model_package, m)?)?;

    // Register training module
    training::register_training_module(m)?;

    // Register DSL module
    dsl::register_dsl_module(m)?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
