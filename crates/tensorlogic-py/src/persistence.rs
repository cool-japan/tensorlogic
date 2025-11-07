//! Model persistence for saving and loading TensorLogic models
//!
//! This module provides functionality to save and load:
//! - Compiled EinsumGraph instances
//! - Compilation configurations
//! - Symbol tables and domain information
//! - Training state and parameters
//!
//! Supports multiple formats:
//! - JSON: Human-readable, cross-platform
//! - Binary: Compact, efficient (bincode)
//! - Pickle: Python-native serialization

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::adapters::{PyCompilerContext, PySymbolTable};
use crate::compiler::PyCompilationConfig;
use crate::types::PyEinsumGraph;

/// Model package containing everything needed to save/load a model
#[pyclass(name = "ModelPackage", module = "pytensorlogic")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyModelPackage {
    /// The compiled einsum graph
    #[pyo3(get, set)]
    pub graph: Option<String>, // Serialized graph

    /// Compilation configuration used
    #[pyo3(get, set)]
    pub config: Option<String>, // Serialized config

    /// Symbol table (domain and predicate info)
    #[pyo3(get, set)]
    pub symbol_table: Option<String>, // Serialized symbol table

    /// Compiler context (variable bindings, axis assignments)
    #[pyo3(get, set)]
    pub compiler_context: Option<String>, // Serialized compiler context

    /// Training parameters (if trained)
    #[pyo3(get, set)]
    pub parameters: Option<HashMap<String, Vec<u8>>>, // Tensor parameters

    /// Metadata (version, creation date, description, etc.)
    #[pyo3(get, set)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl PyModelPackage {
    #[new]
    fn new() -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
        metadata.insert("created_at".to_string(), chrono::Utc::now().to_rfc3339());

        Self {
            graph: None,
            config: None,
            symbol_table: None,
            compiler_context: None,
            parameters: None,
            metadata,
        }
    }

    /// Add metadata key-value pair
    ///
    /// Args:
    ///     key: Metadata key
    ///     value: Metadata value
    fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata value by key
    ///
    /// Args:
    ///     key: Metadata key
    ///
    /// Returns:
    ///     Metadata value or None
    fn get_metadata(&self, key: String) -> Option<String> {
        self.metadata.get(&key).cloned()
    }

    /// Save package to JSON file
    ///
    /// Args:
    ///     path: File path to save to
    ///
    /// Example:
    ///     >>> package = tl.ModelPackage()
    ///     >>> package.save_json("model.json")
    fn save_json(&self, path: String) -> PyResult<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "JSON serialization failed: {}",
                e
            ))
        })?;

        std::fs::write(path, json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write file: {}", e))
        })?;

        Ok(())
    }

    /// Load package from JSON file
    ///
    /// Args:
    ///     path: File path to load from
    ///
    /// Returns:
    ///     ModelPackage instance
    ///
    /// Example:
    ///     >>> package = tl.ModelPackage.load_json("model.json")
    #[staticmethod]
    fn load_json(path: String) -> PyResult<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read file: {}", e))
        })?;

        let package: PyModelPackage = serde_json::from_str(&json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "JSON deserialization failed: {}",
                e
            ))
        })?;

        Ok(package)
    }

    /// Save package to binary file (bincode format)
    ///
    /// Args:
    ///     path: File path to save to
    ///
    /// Example:
    ///     >>> package = tl.ModelPackage()
    ///     >>> package.save_binary("model.bin")
    fn save_binary(&self, path: String) -> PyResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Binary serialization failed: {}",
                e
            ))
        })?;

        std::fs::write(path, bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write file: {}", e))
        })?;

        Ok(())
    }

    /// Load package from binary file (bincode format)
    ///
    /// Args:
    ///     path: File path to load from
    ///
    /// Returns:
    ///     ModelPackage instance
    ///
    /// Example:
    ///     >>> package = tl.ModelPackage.load_binary("model.bin")
    #[staticmethod]
    fn load_binary(path: String) -> PyResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read file: {}", e))
        })?;

        let package: PyModelPackage = bincode::deserialize(&bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Binary deserialization failed: {}",
                e
            ))
        })?;

        Ok(package)
    }

    /// Convert to JSON string
    ///
    /// Returns:
    ///     JSON representation
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "JSON serialization failed: {}",
                e
            ))
        })
    }

    /// Create from JSON string
    ///
    /// Args:
    ///     json: JSON string
    ///
    /// Returns:
    ///     ModelPackage instance
    #[staticmethod]
    fn from_json(json: String) -> PyResult<Self> {
        serde_json::from_str(&json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "JSON deserialization failed: {}",
                e
            ))
        })
    }

    /// Convert to binary bytes (for pickle support)
    ///
    /// Returns:
    ///     Binary representation as bytes
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bincode::serialize(self).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Binary serialization failed: {}",
                e
            ))
        })?;

        Ok(PyBytes::new(py, &bytes))
    }

    /// Create from binary bytes (for pickle support)
    ///
    /// Args:
    ///     bytes: Binary bytes
    ///
    /// Returns:
    ///     ModelPackage instance
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let result: Self = bincode::deserialize(bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Binary deserialization failed: {}",
                e
            ))
        })?;
        Ok(result)
    }

    /// Python pickle support: __getstate__
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.to_bytes(py)
    }

    /// Python pickle support: __setstate__
    fn __setstate__(&mut self, bytes: &[u8]) -> PyResult<()> {
        let package = Self::from_bytes(bytes)?;
        *self = package;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelPackage(graph={}, config={}, symbol_table={}, parameters={}, metadata={})",
            self.graph.is_some(),
            self.config.is_some(),
            self.symbol_table.is_some(),
            self.parameters.is_some(),
            self.metadata.len()
        )
    }

    fn __str__(&self) -> String {
        let mut parts = vec![];

        if self.graph.is_some() {
            parts.push("graph");
        }
        if self.config.is_some() {
            parts.push("config");
        }
        if self.symbol_table.is_some() {
            parts.push("symbol_table");
        }
        if self.parameters.is_some() {
            parts.push("parameters");
        }

        format!(
            "ModelPackage with: [{}], {} metadata entries",
            parts.join(", "),
            self.metadata.len()
        )
    }

    /// Rich HTML representation for Jupyter notebooks
    ///
    /// Returns:
    ///     HTML string for display in Jupyter/IPython
    fn _repr_html_(&self) -> String {
        use crate::jupyter::model_package_html;

        model_package_html(
            self.graph.is_some(),
            self.config.is_some(),
            self.symbol_table.is_some(),
            self.parameters.is_some(),
            &self.metadata,
        )
    }
}

/// Save a compiled graph to file
///
/// Args:
///     graph: EinsumGraph to save
///     path: File path to save to
///     format: Format to use ("json" or "binary", default: "json")
///
/// Example:
///     >>> graph = tl.compile(expr)
///     >>> tl.save_model(graph, "model.json")
#[pyfunction(name = "save_model")]
#[pyo3(signature = (graph, path, format="json"))]
pub fn py_save_model(graph: &PyEinsumGraph, path: String, format: &str) -> PyResult<()> {
    let mut package = PyModelPackage::new();

    // Serialize graph
    let graph_json = serde_json::to_string(&graph.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Graph serialization failed: {}",
            e
        ))
    })?;
    package.graph = Some(graph_json);

    // Add metadata
    package.add_metadata("type".to_string(), "einsum_graph".to_string());

    // Save based on format
    match format {
        "json" => package.save_json(path),
        "binary" | "bin" => package.save_binary(path),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown format: {}. Use 'json' or 'binary'",
            format
        ))),
    }
}

/// Load a compiled graph from file
///
/// Args:
///     path: File path to load from
///     format: Format to use ("json" or "binary", default: auto-detect from extension)
///
/// Returns:
///     EinsumGraph instance
///
/// Example:
///     >>> graph = tl.load_model("model.json")
#[pyfunction(name = "load_model")]
#[pyo3(signature = (path, format=None))]
pub fn py_load_model(path: String, format: Option<String>) -> PyResult<PyEinsumGraph> {
    // Auto-detect format from extension if not provided
    let format = format.unwrap_or_else(|| {
        Path::new(&path)
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "json".to_string())
    });

    // Load package
    let package = match format.as_str() {
        "json" => PyModelPackage::load_json(path)?,
        "bin" | "binary" => PyModelPackage::load_binary(path)?,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown format: {}. Use 'json' or 'binary'",
                format
            )));
        }
    };

    // Deserialize graph
    let graph_json = package.graph.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No graph found in package")
    })?;

    let graph = serde_json::from_str(&graph_json).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Graph deserialization failed: {}",
            e
        ))
    })?;

    Ok(PyEinsumGraph { inner: graph })
}

/// Save a complete model with configuration and symbol table
///
/// Args:
///     graph: EinsumGraph to save
///     path: File path to save to
///     config: Optional compilation config
///     symbol_table: Optional symbol table
///     compiler_context: Optional compiler context
///     metadata: Optional metadata dictionary
///     format: Format to use ("json" or "binary", default: "json")
///
/// Example:
///     >>> tl.save_full_model(
///     ...     graph,
///     ...     "model.json",
///     ...     config=config,
///     ...     symbol_table=sym_table,
///     ...     metadata={"description": "My model"}
///     ... )
#[pyfunction(name = "save_full_model")]
#[pyo3(signature = (graph, path, config=None, symbol_table=None, compiler_context=None, metadata=None, format="json"))]
#[allow(clippy::too_many_arguments)]
pub fn py_save_full_model(
    _py: Python,
    graph: &PyEinsumGraph,
    path: String,
    config: Option<&PyCompilationConfig>,
    symbol_table: Option<&PySymbolTable>,
    compiler_context: Option<&PyCompilerContext>,
    metadata: Option<Bound<'_, PyDict>>,
    format: &str,
) -> PyResult<()> {
    let mut package = PyModelPackage::new();

    // Serialize graph
    let graph_json = serde_json::to_string(&graph.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Graph serialization failed: {}",
            e
        ))
    })?;
    package.graph = Some(graph_json);

    // Serialize config if provided
    // Note: We store the config as preset name since CompilationConfig doesn't have Serialize
    // This is a temporary solution until we add Serialize to CompilationConfig
    if let Some(_cfg) = config {
        // For now, just store a placeholder
        // TODO: Add proper serialization once CompilationConfig has Serialize derive
        package.config = Some("{}".to_string());
    }

    // Serialize symbol table if provided
    if let Some(st) = symbol_table {
        let st_json = st.to_json()?;
        package.symbol_table = Some(st_json);
    }

    // Serialize compiler context if provided
    if let Some(_cc) = compiler_context {
        // Note: We'll need to add to_json method to PyCompilerContext
        // For now, serialize as empty
        package.compiler_context = Some("{}".to_string());
    }

    // Add custom metadata
    if let Some(meta_dict) = metadata {
        for (key, value) in meta_dict.iter() {
            let key_str = key.to_string();
            let value_str = value.to_string();
            package.add_metadata(key_str, value_str);
        }
    }

    // Save based on format
    match format {
        "json" => package.save_json(path),
        "binary" | "bin" => package.save_binary(path),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown format: {}. Use 'json' or 'binary'",
            format
        ))),
    }
}

/// Load a complete model with configuration and symbol table
///
/// Args:
///     path: File path to load from
///     format: Format to use ("json" or "binary", default: auto-detect)
///
/// Returns:
///     Dictionary with keys: 'graph', 'config', 'symbol_table', 'metadata'
///
/// Example:
///     >>> model = tl.load_full_model("model.json")
///     >>> graph = model['graph']
///     >>> config = model['config']
#[pyfunction(name = "load_full_model")]
#[pyo3(signature = (path, format=None))]
pub fn py_load_full_model<'py>(
    py: Python<'py>,
    path: String,
    format: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    // Auto-detect format from extension if not provided
    let format = format.unwrap_or_else(|| {
        Path::new(&path)
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "json".to_string())
    });

    // Load package
    let package = match format.as_str() {
        "json" => PyModelPackage::load_json(path)?,
        "bin" | "binary" => PyModelPackage::load_binary(path)?,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown format: {}. Use 'json' or 'binary'",
                format
            )));
        }
    };

    let result = PyDict::new(py);

    // Deserialize graph
    if let Some(graph_json) = package.graph {
        let graph: tensorlogic_ir::EinsumGraph =
            serde_json::from_str(&graph_json).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Graph deserialization failed: {}",
                    e
                ))
            })?;
        result.set_item("graph", PyEinsumGraph { inner: graph })?;
    }

    // Deserialize config
    // Note: Config serialization is not yet fully implemented
    // For now, return None for config
    if let Some(_config_json) = package.config {
        // TODO: Deserialize config once CompilationConfig has Serialize derive
        // For now, use default config
        result.set_item("config", PyCompilationConfig::new())?;
    }

    // Deserialize symbol table
    if let Some(st_json) = package.symbol_table {
        let symbol_table = PySymbolTable::from_json(st_json)?;
        result.set_item("symbol_table", symbol_table)?;
    }

    // Add metadata
    let metadata_dict = PyDict::new(py);
    for (key, value) in package.metadata {
        metadata_dict.set_item(key, value)?;
    }
    result.set_item("metadata", metadata_dict)?;

    Ok(result)
}

/// Create a model package helper function
///
/// Returns:
///     Empty ModelPackage instance
///
/// Example:
///     >>> package = tl.model_package()
///     >>> package.add_metadata("author", "John Doe")
#[pyfunction(name = "model_package")]
pub fn py_model_package() -> PyModelPackage {
    PyModelPackage::new()
}

// Additional dependencies needed in Cargo.toml:
// chrono = { version = "0.4", features = ["serde"] }
// bincode = "1"
