//! Backend selection and capability queries for Python

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use tensorlogic_infer::{BackendCapabilities, DType, DeviceType, Feature, TlCapabilities};
use tensorlogic_scirs_backend::Scirs2Exec;

/// Backend selection for execution
///
/// The backend determines which execution engine is used to run compiled graphs.
/// Different backends may have different performance characteristics and hardware support.
///
/// Available backends:
/// - Auto: Automatically selects the best available backend
/// - SciRS2CPU: SciRS2 backend with CPU execution (always available)
/// - SciRS2SIMD: SciRS2 backend with SIMD acceleration (always available)
/// - SciRS2GPU: SciRS2 backend with GPU execution (requires 'gpu' feature)
///
/// Example:
///     >>> from pytensorlogic import Backend
///     >>> # Use default backend
///     >>> result = execute(graph, inputs)
///     >>> # Explicitly select SIMD backend for better performance
///     >>> result = execute(graph, inputs, backend=Backend.SciRS2SIMD)
#[pyclass(name = "Backend")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyBackend {
    /// Automatically select the best available backend
    Auto,
    /// SciRS2 backend with CPU execution
    SciRS2CPU,
    /// SciRS2 backend with SIMD acceleration
    SciRS2SIMD,
    /// SciRS2 backend with GPU execution (requires 'gpu' feature)
    SciRS2GPU,
}

#[pymethods]
impl PyBackend {
    /// Auto: Automatically select the best available backend
    #[classattr]
    const AUTO: PyBackend = PyBackend::Auto;

    /// SciRS2CPU: SciRS2 backend with CPU execution
    #[classattr]
    const SCIRS2_CPU: PyBackend = PyBackend::SciRS2CPU;

    /// SciRS2SIMD: SciRS2 backend with SIMD acceleration
    #[classattr]
    const SCIRS2_SIMD: PyBackend = PyBackend::SciRS2SIMD;

    /// SciRS2GPU: SciRS2 backend with GPU execution
    #[classattr]
    const SCIRS2_GPU: PyBackend = PyBackend::SciRS2GPU;

    /// Get the backend name as a string
    fn __str__(&self) -> &'static str {
        match self {
            PyBackend::Auto => "Auto",
            PyBackend::SciRS2CPU => "SciRS2CPU",
            PyBackend::SciRS2SIMD => "SciRS2SIMD",
            PyBackend::SciRS2GPU => "SciRS2GPU",
        }
    }

    /// Get the backend representation
    fn __repr__(&self) -> String {
        format!("Backend.{}", self.__str__())
    }
}

// Default implementation is conditional based on feature flags
#[allow(clippy::derivable_impls)]
impl Default for PyBackend {
    fn default() -> Self {
        // Default to SIMD if available, otherwise CPU
        #[cfg(feature = "simd")]
        {
            PyBackend::SciRS2SIMD
        }
        #[cfg(not(feature = "simd"))]
        {
            PyBackend::SciRS2CPU
        }
    }
}

impl PyBackend {
    /// Check if this backend is available on the current system
    ///
    /// Returns:
    ///     bool: True if the backend is available, False otherwise
    pub fn is_available(&self) -> bool {
        match self {
            PyBackend::Auto => true,       // Auto is always available
            PyBackend::SciRS2CPU => true,  // CPU is always available
            PyBackend::SciRS2SIMD => true, // SIMD is always available (CPU fallback)
            // For now, GPU is not yet implemented
            PyBackend::SciRS2GPU => false,
        }
    }
}

/// Backend capabilities information
///
/// Provides detailed information about a backend's supported features,
/// devices, data types, and limitations.
///
/// Example:
///     >>> caps = get_backend_capabilities(Backend.SciRS2CPU)
///     >>> print(f"Backend: {caps.name} v{caps.version}")
///     >>> print(f"Devices: {caps.devices}")
///     >>> print(f"Features: {caps.features}")
#[pyclass(name = "BackendCapabilities")]
#[derive(Clone)]
pub struct PyBackendCapabilities {
    inner: BackendCapabilities,
}

#[pymethods]
impl PyBackendCapabilities {
    /// Get the backend name
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get the backend version
    #[getter]
    fn version(&self) -> String {
        self.inner.version.clone()
    }

    /// Get list of supported device types
    #[getter]
    fn devices(&self) -> Vec<String> {
        self.inner
            .supported_devices
            .iter()
            .map(|d| d.as_str().to_string())
            .collect()
    }

    /// Get list of supported data types
    #[getter]
    fn dtypes(&self) -> Vec<String> {
        self.inner
            .supported_dtypes
            .iter()
            .map(|d| d.as_str().to_string())
            .collect()
    }

    /// Get list of supported features
    #[getter]
    fn features(&self) -> Vec<String> {
        self.inner
            .features
            .iter()
            .map(|f| f.as_str().to_string())
            .collect()
    }

    /// Get maximum number of tensor dimensions supported
    #[getter]
    fn max_dims(&self) -> usize {
        self.inner.max_tensor_dims
    }

    /// Check if a specific device type is supported
    ///
    /// Args:
    ///     device: Device type to check (e.g., "CPU", "GPU", "TPU")
    ///
    /// Returns:
    ///     bool: True if the device is supported
    fn supports_device(&self, device: &str) -> bool {
        let device_type = match device.to_uppercase().as_str() {
            "CPU" => DeviceType::CPU,
            "GPU" => DeviceType::GPU,
            "TPU" => DeviceType::TPU,
            _ => return false,
        };
        self.inner.supports_device(device_type)
    }

    /// Check if a specific data type is supported
    ///
    /// Args:
    ///     dtype: Data type to check (e.g., "f32", "f64", "i32")
    ///
    /// Returns:
    ///     bool: True if the data type is supported
    fn supports_dtype(&self, dtype: &str) -> bool {
        let dtype_enum = match dtype.to_lowercase().as_str() {
            "f32" => DType::F32,
            "f64" => DType::F64,
            "i32" => DType::I32,
            "i64" => DType::I64,
            "bool" => DType::Bool,
            _ => return false,
        };
        self.inner.supports_dtype(dtype_enum)
    }

    /// Check if a specific feature is supported
    ///
    /// Args:
    ///     feature: Feature name (e.g., "Autodiff", "BatchExecution", "GPUAcceleration")
    ///
    /// Returns:
    ///     bool: True if the feature is supported
    fn supports_feature(&self, feature: &str) -> bool {
        let feature_enum = match feature {
            "Autodiff" => Feature::Autodiff,
            "BatchExecution" => Feature::BatchExecution,
            "SparseTensors" => Feature::SparseTensors,
            "MixedPrecision" => Feature::MixedPrecision,
            "GPUAcceleration" => Feature::GPUAcceleration,
            "DistributedExecution" => Feature::DistributedExecution,
            "JIT" => Feature::JIT,
            _ => return false,
        };
        self.inner.supports_feature(feature_enum)
    }

    /// Get a summary of the backend capabilities
    ///
    /// Returns:
    ///     str: Human-readable summary of capabilities
    fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Get capabilities as a dictionary
    ///
    /// Returns:
    ///     dict: Dictionary with all capability information
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("name", &self.inner.name)?;
        dict.set_item("version", &self.inner.version)?;
        dict.set_item("devices", self.devices())?;
        dict.set_item("dtypes", self.dtypes())?;
        dict.set_item("features", self.features())?;
        dict.set_item("max_dims", self.inner.max_tensor_dims)?;
        Ok(dict)
    }

    /// String representation
    fn __str__(&self) -> String {
        format!("{} v{}", self.inner.name, self.inner.version)
    }

    /// Detailed representation
    fn __repr__(&self) -> String {
        format!(
            "BackendCapabilities(name='{}', version='{}', devices={}, features={})",
            self.inner.name,
            self.inner.version,
            self.devices().len(),
            self.features().len()
        )
    }
}

impl PyBackendCapabilities {
    pub fn from_backend_capabilities(caps: BackendCapabilities) -> Self {
        PyBackendCapabilities { inner: caps }
    }
}

/// Get capabilities for a specific backend
///
/// Args:
///     backend: The backend to query (defaults to Auto)
///
/// Returns:
///     BackendCapabilities: Detailed capability information
///
/// Raises:
///     RuntimeError: If the backend is not available
///
/// Example:
///     >>> from pytensorlogic import get_backend_capabilities, Backend
///     >>> caps = get_backend_capabilities(Backend.SciRS2CPU)
///     >>> print(caps.summary())
#[pyfunction(name = "get_backend_capabilities", signature = (backend = None))]
pub fn py_get_backend_capabilities(backend: Option<PyBackend>) -> PyResult<PyBackendCapabilities> {
    let backend = backend.unwrap_or_default();

    // Check if backend is available
    if !backend.is_available() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Backend {:?} is not available on this system",
            backend
        )));
    }

    // For now, we only have SciRS2 backends
    let executor = Scirs2Exec::new();
    let caps = executor.capabilities().clone();

    Ok(PyBackendCapabilities::from_backend_capabilities(caps))
}

/// List all available backends
///
/// Returns:
///     dict: Dictionary mapping backend names to their availability status
///
/// Example:
///     >>> from pytensorlogic import list_available_backends
///     >>> backends = list_available_backends()
///     >>> print(backends)
///     {'Auto': True, 'SciRS2CPU': True, 'SciRS2SIMD': True, 'SciRS2GPU': False}
#[pyfunction(name = "list_available_backends")]
pub fn py_list_available_backends(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("Auto", PyBackend::Auto.is_available())?;
    dict.set_item("SciRS2CPU", PyBackend::SciRS2CPU.is_available())?;
    dict.set_item("SciRS2SIMD", PyBackend::SciRS2SIMD.is_available())?;
    dict.set_item("SciRS2GPU", PyBackend::SciRS2GPU.is_available())?;
    Ok(dict)
}

/// Get the default backend for this system
///
/// Returns:
///     Backend: The default backend (SciRS2SIMD if available, otherwise SciRS2CPU)
///
/// Example:
///     >>> from pytensorlogic import get_default_backend
///     >>> backend = get_default_backend()
///     >>> print(backend)
///     Backend.SciRS2SIMD
#[pyfunction(name = "get_default_backend")]
pub fn py_get_default_backend() -> PyBackend {
    PyBackend::default()
}

/// Get detailed system information
///
/// Returns:
///     dict: Dictionary with system and backend information
///
/// Example:
///     >>> from pytensorlogic import get_system_info
///     >>> info = get_system_info()
///     >>> print(info)
#[pyfunction(name = "get_system_info")]
pub fn py_get_system_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);

    // Version information
    dict.set_item("tensorlogic_version", env!("CARGO_PKG_VERSION"))?;
    dict.set_item("rust_version", env!("CARGO_PKG_RUST_VERSION"))?;

    // Backend information
    let executor = Scirs2Exec::new();
    let caps = executor.capabilities();
    dict.set_item("default_backend", caps.name.as_str())?;
    dict.set_item("backend_version", caps.version.as_str())?;

    // Available backends
    let mut backends = HashMap::new();
    backends.insert("Auto", PyBackend::Auto.is_available());
    backends.insert("SciRS2CPU", PyBackend::SciRS2CPU.is_available());
    backends.insert("SciRS2SIMD", PyBackend::SciRS2SIMD.is_available());
    backends.insert("SciRS2GPU", PyBackend::SciRS2GPU.is_available());

    let backends_dict = PyDict::new(py);
    for (name, available) in backends {
        backends_dict.set_item(name, available)?;
    }
    dict.set_item("available_backends", backends_dict)?;

    // CPU backend capabilities
    let cpu_caps = PyDict::new(py);
    cpu_caps.set_item(
        "devices",
        caps.supported_devices
            .iter()
            .map(|d| d.as_str())
            .collect::<Vec<_>>(),
    )?;
    cpu_caps.set_item(
        "dtypes",
        caps.supported_dtypes
            .iter()
            .map(|d| d.as_str())
            .collect::<Vec<_>>(),
    )?;
    cpu_caps.set_item(
        "features",
        caps.features.iter().map(|f| f.as_str()).collect::<Vec<_>>(),
    )?;
    cpu_caps.set_item("max_dims", caps.max_tensor_dims)?;
    dict.set_item("cpu_capabilities", cpu_caps)?;

    Ok(dict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_str() {
        assert_eq!(PyBackend::Auto.__str__(), "Auto");
        assert_eq!(PyBackend::SciRS2CPU.__str__(), "SciRS2CPU");
        assert_eq!(PyBackend::SciRS2SIMD.__str__(), "SciRS2SIMD");
        assert_eq!(PyBackend::SciRS2GPU.__str__(), "SciRS2GPU");
    }

    #[test]
    fn test_backend_availability() {
        assert!(PyBackend::Auto.is_available());
        assert!(PyBackend::SciRS2CPU.is_available());
        assert!(PyBackend::SciRS2SIMD.is_available());
        // GPU is not yet implemented, so it's never available
        assert!(!PyBackend::SciRS2GPU.is_available());
    }

    #[test]
    fn test_default_backend() {
        // Default depends on SIMD feature
        #[cfg(feature = "simd")]
        {
            assert_eq!(PyBackend::default(), PyBackend::SciRS2SIMD);
        }
        #[cfg(not(feature = "simd"))]
        {
            assert_eq!(PyBackend::default(), PyBackend::SciRS2CPU);
        }
    }

    #[test]
    fn test_get_capabilities() {
        let caps = py_get_backend_capabilities(Some(PyBackend::SciRS2CPU));
        assert!(caps.is_ok());
        let caps = caps.unwrap();
        assert_eq!(caps.name(), "SciRS2 Backend");
        assert!(!caps.version().is_empty());
    }

    #[test]
    fn test_capabilities_devices() {
        let caps = py_get_backend_capabilities(Some(PyBackend::SciRS2CPU)).unwrap();
        let devices = caps.devices();
        assert!(devices.contains(&"CPU".to_string()));
    }

    #[test]
    fn test_capabilities_dtypes() {
        let caps = py_get_backend_capabilities(Some(PyBackend::SciRS2CPU)).unwrap();
        let dtypes = caps.dtypes();
        assert!(dtypes.contains(&"f64".to_string()));
        assert!(dtypes.contains(&"f32".to_string()));
    }

    #[test]
    fn test_capabilities_features() {
        let caps = py_get_backend_capabilities(Some(PyBackend::SciRS2CPU)).unwrap();
        let features = caps.features();
        assert!(features.contains(&"Autodiff".to_string()));
        assert!(features.contains(&"BatchExecution".to_string()));
    }

    #[test]
    fn test_supports_device() {
        let caps = py_get_backend_capabilities(Some(PyBackend::SciRS2CPU)).unwrap();
        assert!(caps.supports_device("CPU"));
        assert!(caps.supports_device("cpu")); // Case insensitive
    }

    #[test]
    fn test_supports_dtype() {
        let caps = py_get_backend_capabilities(Some(PyBackend::SciRS2CPU)).unwrap();
        assert!(caps.supports_dtype("f64"));
        assert!(caps.supports_dtype("f32"));
        assert!(caps.supports_dtype("F64")); // Case insensitive
    }

    #[test]
    fn test_supports_feature() {
        let caps = py_get_backend_capabilities(Some(PyBackend::SciRS2CPU)).unwrap();
        assert!(caps.supports_feature("Autodiff"));
        assert!(caps.supports_feature("BatchExecution"));
        assert!(!caps.supports_feature("NonExistentFeature"));
    }

    #[test]
    fn test_capabilities_summary() {
        let caps = py_get_backend_capabilities(Some(PyBackend::SciRS2CPU)).unwrap();
        let summary = caps.summary();
        assert!(summary.contains("SciRS2 Backend"));
        assert!(summary.contains("CPU"));
        assert!(summary.contains("Autodiff"));
    }
}
