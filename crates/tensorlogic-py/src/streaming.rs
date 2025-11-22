//! Streaming execution support for TensorLogic Python bindings
//!
//! This module provides streaming/iterator interfaces for processing
//! large datasets that don't fit in memory.

use numpy::{PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList};
use scirs2_core::ndarray::{s, Array1, ArrayD};
use std::collections::HashMap;
use tensorlogic_infer::TlAutodiff;
use tensorlogic_scirs_backend::Scirs2Exec;

use crate::backend::PyBackend;
use crate::numpy_conversion::arrayd_to_numpy;
use crate::types::PyEinsumGraph;

/// Streaming executor for processing large datasets in chunks
///
/// Processes data in memory-efficient chunks, yielding results
/// as they are computed.
#[pyclass(name = "StreamingExecutor")]
pub struct PyStreamingExecutor {
    graph: PyEinsumGraph,
    backend: PyBackend,
    chunk_size: usize,
    overlap: usize,
}

#[pymethods]
impl PyStreamingExecutor {
    #[new]
    #[pyo3(signature = (graph, chunk_size=1000, overlap=0, backend=None))]
    fn new(
        graph: PyEinsumGraph,
        chunk_size: usize,
        overlap: usize,
        backend: Option<PyBackend>,
    ) -> Self {
        PyStreamingExecutor {
            graph,
            backend: backend.unwrap_or(PyBackend::Auto),
            chunk_size,
            overlap,
        }
    }

    /// Execute on data in streaming fashion
    ///
    /// Args:
    ///     inputs: Dictionary with input arrays
    ///     output_key: Name of input to chunk (default: first input)
    ///
    /// Returns:
    ///     List of result chunks
    #[pyo3(signature = (inputs, output_key=None))]
    fn execute_streaming<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        output_key: Option<String>,
    ) -> PyResult<Bound<'py, PyList>> {
        // Get input arrays
        let mut input_tensors: HashMap<String, ArrayD<f64>> = HashMap::new();
        let mut chunk_key = output_key.clone();
        let mut total_length = 0;

        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;
            let array: PyReadonlyArrayDyn<f64> = value.extract()?;
            let owned = array.as_array().to_owned();

            // Use first key if not specified
            if chunk_key.is_none() {
                chunk_key = Some(name.clone());
                total_length = owned.shape().first().copied().unwrap_or(0);
            }

            input_tensors.insert(name, owned);
        }

        let chunk_key = chunk_key
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No input data provided"))?;

        // Process in chunks
        let results = PyList::empty(py);
        let mut offset = 0;

        while offset < total_length {
            let end = (offset + self.chunk_size).min(total_length);
            let mut chunk_inputs: HashMap<String, ArrayD<f64>> = HashMap::new();

            // Extract chunks
            for (name, tensor) in &input_tensors {
                if name == &chunk_key {
                    // Slice the chunked input
                    let sliced = tensor.slice(s![offset..end, ..]).to_owned().into_dyn();
                    chunk_inputs.insert(name.clone(), sliced);
                } else {
                    // Keep full tensor for non-chunked inputs
                    chunk_inputs.insert(name.clone(), tensor.clone());
                }
            }

            // Execute chunk
            let mut executor = Scirs2Exec::new();
            for (name, tensor) in chunk_inputs {
                executor.add_tensor(name, tensor);
            }

            let result = executor.forward(&self.graph.inner).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Chunk execution failed: {}", e))
            })?;

            let py_array = arrayd_to_numpy(py, &result)?;
            results.append(py_array)?;

            // Move to next chunk with overlap handling
            offset = end - self.overlap;
            if offset >= end {
                break;
            }
        }

        Ok(results)
    }

    /// Get the chunk size
    #[getter]
    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Set the chunk size
    #[setter]
    fn set_chunk_size(&mut self, size: usize) {
        self.chunk_size = size;
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingExecutor(chunk_size={}, overlap={}, backend={:?})",
            self.chunk_size, self.overlap, self.backend
        )
    }
}

/// Data generator for memory-efficient data loading
#[pyclass(name = "DataGenerator")]
pub struct PyDataGenerator {
    data_sources: Vec<String>,
    batch_size: usize,
    shuffle: bool,
    current_index: usize,
}

#[pymethods]
impl PyDataGenerator {
    #[new]
    #[pyo3(signature = (data_sources, batch_size=32, shuffle=false))]
    fn new(data_sources: Vec<String>, batch_size: usize, shuffle: bool) -> Self {
        PyDataGenerator {
            data_sources,
            batch_size,
            shuffle,
            current_index: 0,
        }
    }

    /// Get the number of batches
    fn __len__(&self) -> usize {
        self.data_sources.len().div_ceil(self.batch_size)
    }

    /// Reset the generator
    fn reset(&mut self) {
        self.current_index = 0;
    }

    fn __repr__(&self) -> String {
        format!(
            "DataGenerator(sources={}, batch_size={}, shuffle={})",
            self.data_sources.len(),
            self.batch_size,
            self.shuffle
        )
    }
}

/// Accumulator for streaming results
#[pyclass(name = "ResultAccumulator")]
pub struct PyResultAccumulator {
    results: Vec<ArrayD<f64>>,
    total_count: usize,
}

#[pymethods]
impl PyResultAccumulator {
    #[new]
    fn new() -> Self {
        PyResultAccumulator {
            results: Vec::new(),
            total_count: 0,
        }
    }

    /// Add a result chunk
    fn add<'py>(
        &mut self,
        py: Python<'py>,
        result: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<()> {
        let array: PyReadonlyArrayDyn<f64> = result.extract()?;
        let owned = array.as_array().to_owned();
        self.total_count += owned.shape().first().copied().unwrap_or(1);
        self.results.push(owned);
        let _ = py; // Silence unused warning
        Ok(())
    }

    /// Get the number of accumulated results
    fn count(&self) -> usize {
        self.results.len()
    }

    /// Get total number of elements accumulated
    fn total_elements(&self) -> usize {
        self.total_count
    }

    /// Combine all results into a single array
    fn combine<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        if self.results.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No results to combine",
            ));
        }

        // Calculate total size
        let total_len: usize = self.results.iter().map(|r| r.len()).sum();

        // Flatten and combine
        let mut combined = Array1::zeros(total_len);
        let mut offset = 0;
        for result in &self.results {
            let flat = result.iter().cloned().collect::<Vec<_>>();
            for (i, val) in flat.into_iter().enumerate() {
                if offset + i < total_len {
                    combined[offset + i] = val;
                }
            }
            offset += result.len();
        }

        let py_array = PyArray1::from_array(py, &combined);
        Ok(py_array.into_any())
    }

    /// Clear all accumulated results
    fn clear(&mut self) {
        self.results.clear();
        self.total_count = 0;
    }

    /// Get statistics about accumulated results
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("num_chunks", self.results.len())?;
        dict.set_item("total_elements", self.total_count)?;

        // Calculate mean chunk size
        if !self.results.is_empty() {
            let total_size: usize = self.results.iter().map(|r| r.len()).sum();
            let mean_size = total_size as f64 / self.results.len() as f64;
            dict.set_item("mean_chunk_size", mean_size)?;
        }

        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "ResultAccumulator(chunks={}, total_elements={})",
            self.results.len(),
            self.total_count
        )
    }
}

/// Create a streaming executor
#[pyfunction(name = "streaming_executor")]
#[pyo3(signature = (graph, chunk_size=1000, overlap=0, backend=None))]
pub fn py_streaming_executor(
    graph: PyEinsumGraph,
    chunk_size: usize,
    overlap: usize,
    backend: Option<PyBackend>,
) -> PyStreamingExecutor {
    PyStreamingExecutor::new(graph, chunk_size, overlap, backend)
}

/// Create a result accumulator
#[pyfunction(name = "result_accumulator")]
pub fn py_result_accumulator() -> PyResultAccumulator {
    PyResultAccumulator::new()
}

/// Process an iterator of inputs through a graph
///
/// Memory-efficient processing of data streams.
#[pyfunction(name = "process_stream")]
#[pyo3(signature = (graph, input_iterator, input_name, backend=None))]
pub fn py_process_stream<'py>(
    py: Python<'py>,
    graph: &PyEinsumGraph,
    input_iterator: &Bound<'py, PyIterator>,
    input_name: String,
    backend: Option<PyBackend>,
) -> PyResult<Bound<'py, PyList>> {
    let _backend = backend.unwrap_or(PyBackend::Auto);
    let results = PyList::empty(py);

    for item in input_iterator {
        let item = item?;
        let array: PyReadonlyArrayDyn<f64> = item.extract()?;
        let tensor = array.as_array().to_owned();

        // Execute
        let mut executor = Scirs2Exec::new();
        executor.add_tensor(input_name.clone(), tensor);

        let result = executor.forward(&graph.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Stream execution failed: {}", e))
        })?;

        let py_array = arrayd_to_numpy(py, &result)?;
        results.append(py_array)?;
    }

    Ok(results)
}

/// Register streaming module functions
pub fn register_streaming_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStreamingExecutor>()?;
    m.add_class::<PyDataGenerator>()?;
    m.add_class::<PyResultAccumulator>()?;
    m.add_function(wrap_pyfunction!(py_streaming_executor, m)?)?;
    m.add_function(wrap_pyfunction!(py_result_accumulator, m)?)?;
    m.add_function(wrap_pyfunction!(py_process_stream, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_accumulator() {
        let acc = PyResultAccumulator::new();
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.total_elements(), 0);
    }
}
