//! Execution functions for Python

use crate::backend::PyBackend;
use crate::numpy_conversion::arrayd_to_numpy;
use crate::types::PyEinsumGraph;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_core::ndarray::ArrayD;
use tensorlogic_infer::TlAutodiff;
use tensorlogic_scirs_backend::Scirs2Exec;

/// Execute an EinsumGraph with given inputs
///
/// Args:
///     graph: The compiled EinsumGraph to execute
///     inputs: Dictionary mapping input names to NumPy arrays
///     backend: Optional backend selection (defaults to Auto, which selects SciRS2CPU)
///
/// Returns:
///     Dictionary mapping output names to NumPy arrays
///
/// Raises:
///     RuntimeError: If execution fails or backend is not available
///
/// Example:
///     >>> import numpy as np
///     >>> from pytensorlogic import Backend
///     >>> graph = compile(expr)
///     >>> inputs = {"knows": np.random.rand(100, 100)}
///     >>> # Use default backend (Auto/SciRS2CPU)
///     >>> outputs = execute(graph, inputs)
///     >>> # Explicitly select CPU backend
///     >>> outputs = execute(graph, inputs, backend=Backend.SciRS2CPU)
#[pyfunction(name = "execute", signature = (graph, inputs, backend = None))]
pub fn py_execute<'py>(
    py: Python<'py>,
    graph: &PyEinsumGraph,
    inputs: &Bound<'py, PyDict>,
    backend: Option<PyBackend>,
) -> PyResult<Bound<'py, PyDict>> {
    // Select backend (default to Auto, which uses SciRS2CPU)
    let backend = backend.unwrap_or_default();

    // Validate backend availability
    if !backend.is_available() {
        return Err(PyRuntimeError::new_err(format!(
            "Backend {:?} is not available on this system. \
             Check available backends with list_available_backends()",
            backend
        )));
    }

    // All CPU-based backends (CPU, SIMD) use the same executor
    // The SIMD acceleration is handled automatically by scirs2-core when built with simd feature
    // GPU backend would require different initialization when supported
    let backend_name = match backend {
        PyBackend::Auto => "SciRS2",
        PyBackend::SciRS2CPU => "SciRS2",
        PyBackend::SciRS2SIMD => "SciRS2",  // Same as CPU, SIMD handled by scirs2-core
        PyBackend::SciRS2GPU => {
            return Err(PyRuntimeError::new_err(
                "GPU backend is not yet implemented. Use Backend.SciRS2CPU or Backend.SciRS2SIMD instead.",
            ))
        }
    };

    // Create executor based on backend selection
    // Note: SIMD acceleration is transparent when scirs2-core is built with simd feature
    let mut executor = match backend_name {
        "SciRS2" => Scirs2Exec::new(),
        _ => unreachable!("Invalid backend after validation"),
    };

    // Add input tensors to executor
    for (key, value) in inputs.iter() {
        let name: String = key.extract()?;

        // Extract as dynamic array to handle any dimensions
        let array: PyReadonlyArrayDyn<f64> = value.extract()?;
        let array_ref = array.as_array();

        // Convert to owned ArrayD
        let owned_array: ArrayD<f64> = array_ref.to_owned();
        executor.add_tensor(name, owned_array);
    }

    // Execute forward pass
    let result = executor
        .forward(&graph.inner)
        .map_err(|e| PyRuntimeError::new_err(format!("Execution failed: {}", e)))?;

    // Convert result back to Python dict with NumPy array
    let output_dict = PyDict::new(py);

    // Get output tensor names from graph
    if graph.inner.outputs.is_empty() {
        return Err(PyRuntimeError::new_err("No outputs in graph"));
    }

    // Use actual tensor name from graph
    let output_idx = graph.inner.outputs[0];
    let output_name = if output_idx < graph.inner.tensors.len() {
        &graph.inner.tensors[output_idx]
    } else {
        // Fall back to default name if index is out of bounds
        "output"
    };

    let py_array = arrayd_to_numpy(py, &result)?;
    output_dict.set_item(output_name, &py_array)?;

    // Also keep "output" key for backward compatibility
    if output_name != "output" {
        output_dict.set_item("output", &py_array)?;
    }

    Ok(output_dict)
}
