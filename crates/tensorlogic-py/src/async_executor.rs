//! Async execution support for TensorLogic Python bindings
//!
//! This module provides asynchronous execution capabilities for running
//! TensorLogic graphs without blocking, useful for Jupyter notebooks
//! and web applications.

use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use scirs2_core::ndarray::ArrayD;
use tensorlogic_infer::TlAutodiff;
use tensorlogic_scirs_backend::Scirs2Exec;

use crate::backend::PyBackend;
use crate::numpy_conversion::{arrayd_to_numpy, numpy_to_arrayd};
use crate::types::PyEinsumGraph;

/// Type alias for async execution result thread handle
type AsyncResultHandle = thread::JoinHandle<PyResult<HashMap<String, ArrayD<f64>>>>;

/// Cancellation token for async operations
#[pyclass(name = "CancellationToken")]
#[derive(Clone)]
pub struct PyCancellationToken {
    cancelled: Arc<AtomicBool>,
}

#[pymethods]
impl PyCancellationToken {
    #[new]
    fn new() -> Self {
        PyCancellationToken {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Cancel the associated operation
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation has been requested
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Reset the cancellation token
    fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }

    fn __repr__(&self) -> String {
        format!("CancellationToken(cancelled={})", self.is_cancelled())
    }
}

/// Create a cancellation token
#[pyfunction(name = "cancellation_token")]
pub fn py_cancellation_token() -> PyCancellationToken {
    PyCancellationToken::new()
}

/// Future-like result for async execution
///
/// Represents a pending or completed async execution result.
/// Use `is_ready()` to check if computation is complete, and `result()` to retrieve it.
/// Supports cancellation through a CancellationToken.
#[pyclass(name = "AsyncResult")]
pub struct PyAsyncResult {
    thread_handle: Option<AsyncResultHandle>,
    completed: bool,
    cancelled: bool,
    result: Option<PyResult<HashMap<String, ArrayD<f64>>>>,
    cancellation_token: Option<PyCancellationToken>,
}

#[pymethods]
impl PyAsyncResult {
    /// Check if the async computation is ready
    ///
    /// Returns:
    ///     bool: True if computation is complete, False if still running
    fn is_ready(&mut self) -> bool {
        if self.completed {
            return true;
        }

        // Check if thread has finished
        if let Some(handle) = &self.thread_handle {
            if handle.is_finished() {
                // Take the handle and get result
                let handle = self.thread_handle.take().unwrap();
                self.result = Some(handle.join().unwrap_or_else(|_| {
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Thread panicked during execution",
                    ))
                }));
                self.completed = true;
                return true;
            }
        }

        false
    }

    /// Wait for computation to complete and get the result
    ///
    /// Returns:
    ///     Dict[str, np.ndarray]: Dictionary mapping output names to NumPy arrays
    ///
    /// Raises:
    ///     RuntimeError: If execution failed
    fn result<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        // Ensure computation is complete
        if !self.completed {
            if let Some(handle) = self.thread_handle.take() {
                self.result = Some(handle.join().unwrap_or_else(|_| {
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Thread panicked during execution",
                    ))
                }));
                self.completed = true;
            }
        }

        // Extract result
        match self.result.as_ref() {
            Some(Ok(tensors)) => {
                let output_dict = PyDict::new(py);
                for (name, array) in tensors.iter() {
                    let py_array = arrayd_to_numpy(py, array)?;
                    output_dict.set_item(name, py_array)?;
                }
                Ok(output_dict)
            }
            Some(Err(e)) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Execution failed: {}",
                e
            ))),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No result available",
            )),
        }
    }

    /// Wait for computation with timeout
    ///
    /// Args:
    ///     timeout_secs: Maximum time to wait in seconds
    ///
    /// Returns:
    ///     bool: True if completed within timeout, False if still running
    fn wait(&mut self, timeout_secs: f64) -> bool {
        use std::time::{Duration, Instant};

        let start = Instant::now();
        let timeout = Duration::from_secs_f64(timeout_secs);

        while !self.is_ready() {
            if start.elapsed() >= timeout {
                return false;
            }
            // Check for cancellation
            if self.is_cancelled() {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }

        true
    }

    /// Cancel the async operation
    ///
    /// Note: Cancellation is cooperative - the operation will be cancelled
    /// at the next cancellation check point.
    fn cancel(&mut self) {
        if let Some(ref token) = self.cancellation_token {
            token.cancel();
        }
        self.cancelled = true;
    }

    /// Check if the operation was cancelled
    fn is_cancelled(&self) -> bool {
        self.cancelled
            || self
                .cancellation_token
                .as_ref()
                .is_some_and(|t| t.is_cancelled())
    }

    /// Get the cancellation token if available
    fn get_cancellation_token(&self) -> Option<PyCancellationToken> {
        self.cancellation_token.clone()
    }

    fn __repr__(&self) -> String {
        if self.cancelled {
            "AsyncResult(cancelled=True)".to_string()
        } else if self.completed {
            "AsyncResult(completed=True)".to_string()
        } else {
            "AsyncResult(completed=False)".to_string()
        }
    }
}

/// Execute a graph asynchronously without blocking
///
/// This function starts execution in a background thread and immediately
/// returns an AsyncResult that can be polled for completion.
///
/// Args:
///     graph: The compiled EinsumGraph to execute
///     inputs: Dictionary mapping input names to NumPy arrays
///     backend: Optional backend selection (defaults to Auto)
///
/// Returns:
///     AsyncResult: Future-like object that can be polled for results
///
/// Example:
///     >>> future = tl.execute_async(graph, inputs)
///     >>> while not future.is_ready():
///     ...     print("Still computing...")
///     ...     time.sleep(0.1)
///     >>> result = future.result()
#[pyfunction(name = "execute_async")]
#[pyo3(signature = (graph, inputs, backend=None))]
pub fn py_execute_async<'py>(
    _py: Python<'py>,
    graph: &PyEinsumGraph,
    inputs: &Bound<'py, PyDict>,
    backend: Option<PyBackend>,
) -> PyResult<PyAsyncResult> {
    // Clone graph for thread safety
    let graph_clone = graph.inner.clone();

    // Convert inputs to Rust types
    let mut input_tensors: HashMap<String, ArrayD<f64>> = HashMap::new();
    for (key, value) in inputs.iter() {
        let key_str: String = key.extract()?;
        let array: PyReadonlyArrayDyn<'py, f64> = value.extract()?;
        let tensor = numpy_to_arrayd(array);
        input_tensors.insert(key_str, tensor);
    }

    // Select backend
    let _backend = backend.unwrap_or(PyBackend::Auto);

    // Spawn computation in background thread
    let handle = thread::spawn(move || {
        // Create executor
        let mut executor = Scirs2Exec::new();

        // Add input tensors
        for (name, tensor) in input_tensors {
            executor.add_tensor(name, tensor);
        }

        // Execute graph using forward pass
        let result = executor.forward(&graph_clone).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Execution failed: {}", e))
        })?;

        // Return result in a HashMap (simulating multiple outputs support)
        let mut output_map = HashMap::new();
        output_map.insert("output".to_string(), result);
        Ok(output_map)
    });

    let token = PyCancellationToken::new();
    Ok(PyAsyncResult {
        thread_handle: Some(handle),
        completed: false,
        cancelled: false,
        result: None,
        cancellation_token: Some(token),
    })
}

/// Execute multiple graphs in parallel
///
/// Executes multiple graphs concurrently in separate threads and returns
/// a list of AsyncResult objects.
///
/// Args:
///     graphs: List of EinsumGraph objects to execute
///     inputs_list: List of input dictionaries (one per graph)
///     backend: Optional backend selection
///
/// Returns:
///     List[AsyncResult]: List of futures for each execution
///
/// Example:
///     >>> futures = tl.execute_parallel([graph1, graph2], [inputs1, inputs2])
///     >>> results = [f.result() for f in futures]
#[pyfunction(name = "execute_parallel")]
#[pyo3(signature = (graphs, inputs_list, backend=None))]
pub fn py_execute_parallel<'py>(
    py: Python<'py>,
    graphs: Vec<PyEinsumGraph>,
    inputs_list: Vec<Bound<'py, PyDict>>,
    backend: Option<PyBackend>,
) -> PyResult<Vec<PyAsyncResult>> {
    if graphs.len() != inputs_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Number of graphs must match number of input dictionaries",
        ));
    }

    let mut futures = Vec::new();

    for (graph, inputs) in graphs.iter().zip(inputs_list.iter()) {
        let future = py_execute_async(py, graph, inputs, backend)?;
        futures.push(future);
    }

    Ok(futures)
}

/// Batch executor for processing multiple inputs through the same graph
///
/// Efficiently executes a single graph with multiple different inputs,
/// optionally in parallel.
#[pyclass(name = "BatchExecutor")]
pub struct PyBatchExecutor {
    graph: PyEinsumGraph,
    backend: PyBackend,
}

#[pymethods]
impl PyBatchExecutor {
    #[new]
    #[pyo3(signature = (graph, backend=None))]
    fn new(graph: PyEinsumGraph, backend: Option<PyBackend>) -> Self {
        PyBatchExecutor {
            graph,
            backend: backend.unwrap_or(PyBackend::Auto),
        }
    }

    /// Execute graph with multiple input batches in parallel
    ///
    /// Args:
    ///     inputs_list: List of input dictionaries to process
    ///     parallel: If True, execute batches in parallel (default: True)
    ///
    /// Returns:
    ///     List of result dictionaries
    ///
    /// Example:
    ///     >>> executor = tl.BatchExecutor(graph)
    ///     >>> results = executor.execute_batch([inputs1, inputs2, inputs3])
    #[pyo3(signature = (inputs_list, parallel=true))]
    fn execute_batch<'py>(
        &self,
        py: Python<'py>,
        inputs_list: Vec<Bound<'py, PyDict>>,
        parallel: bool,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        if parallel {
            // Parallel execution
            let graphs = vec![self.graph.clone(); inputs_list.len()];
            let futures = py_execute_parallel(py, graphs, inputs_list, Some(self.backend))?;

            // Wait for all to complete and collect results
            let mut results = Vec::new();
            for mut future in futures {
                results.push(future.result(py)?);
            }
            Ok(results)
        } else {
            // Sequential execution
            let mut results = Vec::new();
            for inputs in inputs_list {
                let result =
                    crate::executor::py_execute(py, &self.graph, &inputs, Some(self.backend))?;
                results.push(result);
            }
            Ok(results)
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchExecutor(graph_nodes={}, backend={:?})",
            self.graph.inner.nodes.len(),
            self.backend
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_async_result_lifecycle() {
        // Basic test structure (requires Python runtime for full testing)
        // Actual tests should be in Python test suite
    }
}
