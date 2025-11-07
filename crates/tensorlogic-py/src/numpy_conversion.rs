//! NumPy array conversion utilities

use numpy::{PyArray, PyArray2, PyArrayMethods, PyReadonlyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};

/// Convert a NumPy array to an SciRS2 Array2
#[allow(dead_code)]
pub fn numpy_to_array2(array: PyReadonlyArray2<f64>) -> Array2<f64> {
    array.as_array().to_owned()
}

/// Convert an SciRS2 Array2 to a NumPy array
#[allow(dead_code)]
pub fn array2_to_numpy<'py>(
    py: Python<'py>,
    array: &Array2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = array.shape();
    let data: Vec<f64> = array.iter().copied().collect();

    // Create PyArray from raw data
    unsafe {
        let py_array = PyArray2::new(py, [shape[0], shape[1]], false);
        let mut slice = py_array.readwrite();
        let slice_mut = slice.as_slice_mut()?;
        slice_mut.copy_from_slice(&data);
        Ok(py_array)
    }
}

/// Convert a NumPy array to an SciRS2 ArrayD (dynamic dimensions)
#[allow(dead_code)]
pub fn numpy_to_arrayd(array: PyReadonlyArray<f64, IxDyn>) -> ArrayD<f64> {
    array.as_array().to_owned()
}

/// Convert an SciRS2 ArrayD to a NumPy array (dynamic dimensions)
pub fn arrayd_to_numpy<'py>(
    py: Python<'py>,
    array: &ArrayD<f64>,
) -> PyResult<Bound<'py, PyArray<f64, IxDyn>>> {
    let shape = array.shape();
    let data: Vec<f64> = array.iter().copied().collect();

    // Create PyArray from raw data
    unsafe {
        let py_array = PyArray::new(py, shape, false);
        let mut slice = py_array.readwrite();
        let slice_mut = slice.as_slice_mut()?;
        slice_mut.copy_from_slice(&data);
        Ok(py_array)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_conversion_roundtrip() {
        // This test requires Python runtime, so we can't run it in pure Rust tests
        // It should be tested from Python side
    }
}
