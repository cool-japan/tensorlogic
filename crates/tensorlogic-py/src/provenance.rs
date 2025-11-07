//! Python bindings for provenance tracking
//!
//! This module exposes TensorLogic's provenance tracking capabilities to Python,
//! enabling users to track the origin and lineage of tensor computations.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tensorlogic_ir::{Provenance, SourceLocation, SourceSpan};
use tensorlogic_oxirs_bridge::ProvenanceTracker;

/// Source location in code
///
/// Represents a specific location in source code (file, line, column).
///
/// Example:
///     >>> from pytensorlogic import SourceLocation
///     >>> loc = SourceLocation("rules.tl", 10, 5)
///     >>> print(loc)
///     rules.tl:10:5
#[pyclass(name = "SourceLocation")]
#[derive(Clone)]
pub struct PySourceLocation {
    inner: SourceLocation,
}

#[pymethods]
impl PySourceLocation {
    /// Create a new source location
    ///
    /// Args:
    ///     file: Source file name or path
    ///     line: Line number (1-indexed)
    ///     column: Column number (1-indexed)
    ///
    /// Returns:
    ///     SourceLocation: New source location
    #[new]
    fn new(file: String, line: usize, column: usize) -> Self {
        PySourceLocation {
            inner: SourceLocation::new(file, line, column),
        }
    }

    /// Get the file name
    #[getter]
    fn file(&self) -> String {
        self.inner.file.clone()
    }

    /// Get the line number
    #[getter]
    fn line(&self) -> usize {
        self.inner.line
    }

    /// Get the column number
    #[getter]
    fn column(&self) -> usize {
        self.inner.column
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceLocation('{}', {}, {})",
            self.inner.file, self.inner.line, self.inner.column
        )
    }
}

/// Source span representing a range in source code
///
/// Represents a range of code from start location to end location.
///
/// Example:
///     >>> from pytensorlogic import SourceLocation, SourceSpan
///     >>> start = SourceLocation("rules.tl", 10, 1)
///     >>> end = SourceLocation("rules.tl", 15, 40)
///     >>> span = SourceSpan(start, end)
///     >>> print(span)
///     rules.tl (lines 10-15)
#[pyclass(name = "SourceSpan")]
#[derive(Clone)]
pub struct PySourceSpan {
    inner: SourceSpan,
}

#[pymethods]
impl PySourceSpan {
    /// Create a new source span
    ///
    /// Args:
    ///     start: Starting location
    ///     end: Ending location
    ///
    /// Returns:
    ///     SourceSpan: New source span
    #[new]
    fn new(start: PySourceLocation, end: PySourceLocation) -> Self {
        PySourceSpan {
            inner: SourceSpan::new(start.inner.clone(), end.inner.clone()),
        }
    }

    /// Get the start location
    #[getter]
    fn start(&self) -> PySourceLocation {
        PySourceLocation {
            inner: self.inner.start.clone(),
        }
    }

    /// Get the end location
    #[getter]
    fn end(&self) -> PySourceLocation {
        PySourceLocation {
            inner: self.inner.end.clone(),
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("SourceSpan({:?}, {:?})", self.inner.start, self.inner.end)
    }
}

/// Provenance information for IR nodes
///
/// Tracks the origin and lineage of logical expressions and tensor computations.
/// Includes rule IDs, source locations, and custom attributes.
///
/// Example:
///     >>> from pytensorlogic import Provenance
///     >>> prov = Provenance()
///     >>> prov.set_rule_id("social_network_rule_1")
///     >>> prov.set_source_file("social_rules.tl")
///     >>> prov.add_attribute("author", "alice")
///     >>> print(prov.rule_id)
///     social_network_rule_1
#[pyclass(name = "Provenance")]
#[derive(Clone)]
pub struct PyProvenance {
    inner: Provenance,
}

#[pymethods]
impl PyProvenance {
    /// Create a new empty provenance record
    ///
    /// Returns:
    ///     Provenance: New empty provenance
    #[new]
    fn new() -> Self {
        PyProvenance {
            inner: Provenance::new(),
        }
    }

    /// Get the rule ID
    #[getter]
    fn rule_id(&self) -> Option<String> {
        self.inner.rule_id.clone()
    }

    /// Set the rule ID
    ///
    /// Args:
    ///     rule_id: Unique identifier for the rule
    fn set_rule_id(&mut self, rule_id: String) {
        self.inner.rule_id = Some(rule_id);
    }

    /// Get the source file
    #[getter]
    fn source_file(&self) -> Option<String> {
        self.inner.source_file.clone()
    }

    /// Set the source file
    ///
    /// Args:
    ///     source_file: Path or name of the source file
    fn set_source_file(&mut self, source_file: String) {
        self.inner.source_file = Some(source_file);
    }

    /// Get the source span
    #[getter]
    fn span(&self) -> Option<PySourceSpan> {
        self.inner
            .span
            .as_ref()
            .map(|s| PySourceSpan { inner: s.clone() })
    }

    /// Set the source span
    ///
    /// Args:
    ///     span: Source code span
    fn set_span(&mut self, span: PySourceSpan) {
        self.inner.span = Some(span.inner.clone());
    }

    /// Add a custom attribute
    ///
    /// Args:
    ///     key: Attribute key
    ///     value: Attribute value
    fn add_attribute(&mut self, key: String, value: String) {
        self.inner.attributes.push((key, value));
    }

    /// Get an attribute value by key
    ///
    /// Args:
    ///     key: Attribute key to look up
    ///
    /// Returns:
    ///     str or None: Attribute value if found
    fn get_attribute(&self, key: &str) -> Option<String> {
        self.inner.get_attribute(key).map(|s| s.to_string())
    }

    /// Get all attributes as a dictionary
    ///
    /// Returns:
    ///     dict: All attributes as key-value pairs
    fn get_attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in &self.inner.attributes {
            dict.set_item(key, value)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "Provenance(rule_id={:?}, source_file={:?}, attributes={})",
            self.inner.rule_id,
            self.inner.source_file,
            self.inner.attributes.len()
        )
    }

    /// Rich HTML representation for Jupyter notebooks
    ///
    /// Returns:
    ///     HTML string for display in Jupyter/IPython
    fn _repr_html_(&self) -> String {
        use crate::jupyter::provenance_html;
        use std::collections::HashMap;

        // Convert Vec<(String, String)> to HashMap
        let attrs_map: HashMap<String, String> = self
            .inner
            .attributes
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        provenance_html(&self.inner.rule_id, &self.inner.source_file, &attrs_map)
    }
}

/// Provenance tracker for RDF and tensor computation mappings
///
/// Tracks the relationships between RDF entities, SHACL shapes, and tensor computations.
/// Provides bidirectional mapping and RDF* statement generation.
///
/// Example:
///     >>> from pytensorlogic import ProvenanceTracker
///     >>> tracker = ProvenanceTracker()
///     >>> tracker.track_entity("http://example.org/alice", 0)
///     >>> tracker.track_shape("http://example.org/PersonShape", "Person(x)", 1)
///     >>> entity = tracker.get_entity(0)
///     >>> print(entity)
///     http://example.org/alice
#[pyclass(name = "ProvenanceTracker")]
pub struct PyProvenanceTracker {
    inner: ProvenanceTracker,
}

#[pymethods]
impl PyProvenanceTracker {
    /// Create a new provenance tracker
    ///
    /// Args:
    ///     enable_rdfstar: Enable RDF* statement-level provenance (default: False)
    ///
    /// Returns:
    ///     ProvenanceTracker: New provenance tracker
    #[new]
    #[pyo3(signature = (enable_rdfstar = false))]
    fn new(enable_rdfstar: bool) -> Self {
        PyProvenanceTracker {
            inner: if enable_rdfstar {
                ProvenanceTracker::with_rdfstar()
            } else {
                ProvenanceTracker::new()
            },
        }
    }

    /// Track an RDF entity to tensor index mapping
    ///
    /// Args:
    ///     entity_iri: RDF entity IRI (e.g., "http://example.org/alice")
    ///     tensor_idx: Tensor index
    fn track_entity(&mut self, entity_iri: String, tensor_idx: usize) {
        self.inner.track_entity(entity_iri, tensor_idx);
    }

    /// Track a SHACL shape to rule and node mapping
    ///
    /// Args:
    ///     shape_iri: SHACL shape IRI
    ///     rule_expr: Rule expression string
    ///     node_idx: Tensor node index
    fn track_shape(&mut self, shape_iri: String, rule_expr: String, node_idx: usize) {
        self.inner.track_shape(shape_iri, rule_expr, node_idx);
    }

    /// Get the RDF entity IRI for a tensor index
    ///
    /// Args:
    ///     tensor_idx: Tensor index
    ///
    /// Returns:
    ///     str or None: Entity IRI if found
    fn get_entity(&self, tensor_idx: usize) -> Option<String> {
        self.inner.get_entity(tensor_idx).map(|s| s.to_string())
    }

    /// Get the tensor index for an RDF entity IRI
    ///
    /// Args:
    ///     entity_iri: RDF entity IRI
    ///
    /// Returns:
    ///     int or None: Tensor index if found
    fn get_tensor(&self, entity_iri: &str) -> Option<usize> {
        self.inner.get_tensor(entity_iri)
    }

    /// Track an inferred triple with metadata
    ///
    /// Args:
    ///     subject: Subject IRI
    ///     predicate: Predicate IRI
    ///     object: Object IRI
    ///     rule_id: Rule ID (optional)
    ///     confidence: Confidence score 0.0-1.0 (optional)
    #[pyo3(signature = (subject, predicate, object, rule_id = None, confidence = None))]
    fn track_inferred_triple(
        &mut self,
        subject: String,
        predicate: String,
        object: String,
        rule_id: Option<String>,
        confidence: Option<f64>,
    ) {
        self.inner
            .track_inferred_triple(subject, predicate, object, rule_id, confidence);
    }

    /// Get all tracked entity mappings
    ///
    /// Returns:
    ///     dict: Dictionary mapping entity IRIs to tensor indices
    fn get_entity_mappings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (entity, idx) in &self.inner.entity_to_tensor {
            dict.set_item(entity, idx)?;
        }
        Ok(dict)
    }

    /// Get all tracked shape mappings
    ///
    /// Returns:
    ///     dict: Dictionary mapping shape IRIs to rule expressions
    fn get_shape_mappings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (shape, rule) in &self.inner.shape_to_rule {
            dict.set_item(shape, rule)?;
        }
        Ok(dict)
    }

    /// Export provenance as RDF* statements
    ///
    /// Returns:
    ///     list: List of RDF* statement strings
    fn to_rdf_star<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let statements = self.inner.to_rdf_star();
        let list = PyList::empty(py);
        for stmt in statements {
            list.append(stmt)?;
        }
        Ok(list)
    }

    /// Export provenance as RDF* Turtle format
    ///
    /// Returns:
    ///     str: Complete Turtle document with provenance
    fn to_rdfstar_turtle(&self) -> String {
        self.inner.to_rdfstar_turtle()
    }

    /// Export provenance as JSON
    ///
    /// Returns:
    ///     str: JSON representation of provenance
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Import provenance from JSON
    ///
    /// Args:
    ///     json: JSON string
    ///
    /// Returns:
    ///     ProvenanceTracker: Restored provenance tracker
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        ProvenanceTracker::from_json(json)
            .map(|inner| PyProvenanceTracker { inner })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get high-confidence inferred triples
    ///
    /// Args:
    ///     min_confidence: Minimum confidence threshold (0.0-1.0)
    ///
    /// Returns:
    ///     list: List of high-confidence inference metadata
    #[pyo3(signature = (min_confidence = 0.8))]
    fn get_high_confidence_inferences<'py>(
        &self,
        py: Python<'py>,
        min_confidence: f64,
    ) -> PyResult<Bound<'py, PyList>> {
        let inferences = self.inner.get_high_confidence_inferences(min_confidence);
        let list = PyList::empty(py);

        for metadata in inferences {
            let dict = PyDict::new(py);
            dict.set_item("subject", &metadata.statement.subject)?;
            dict.set_item("predicate", &metadata.statement.predicate)?;
            dict.set_item("object", &metadata.statement.object)?;
            if let Some(rule_id) = &metadata.rule_id {
                dict.set_item("rule_id", rule_id)?;
            }
            if let Some(conf) = metadata.confidence {
                dict.set_item("confidence", conf)?;
            }
            if let Some(generated_by) = &metadata.generated_by {
                dict.set_item("generated_by", generated_by)?;
            }
            if let Some(generated_at) = &metadata.generated_at {
                dict.set_item("generated_at", generated_at)?;
            }
            list.append(dict)?;
        }

        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "ProvenanceTracker(entities={}, shapes={})",
            self.inner.entity_to_tensor.len(),
            self.inner.shape_to_rule.len()
        )
    }
}

/// Get provenance information from an einsum graph
///
/// Extracts provenance metadata from all nodes in the graph.
///
/// Args:
///     graph: EinsumGraph to extract provenance from
///
/// Returns:
///     list: List of provenance records for each node with metadata
///
/// Example:
///     >>> import pytensorlogic as tl
///     >>> expr = tl.pred("knows", [tl.var("x"), tl.var("y")])
///     >>> graph = tl.compile(expr)
///     >>> provenance = tl.get_provenance(graph)
///     >>> for prov in provenance:
///     ...     if prov:
///     ...         print(f"Rule: {prov.rule_id}")
#[pyfunction(name = "get_provenance")]
pub fn py_get_provenance<'py>(
    py: Python<'py>,
    graph: &crate::types::PyEinsumGraph,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);

    // Get the inner graph
    let inner_graph = &graph.inner;

    // Extract provenance from each node's metadata
    for node in &inner_graph.nodes {
        if let Some(metadata) = &node.metadata {
            if let Some(provenance) = &metadata.provenance {
                let py_prov = PyProvenance {
                    inner: provenance.clone(),
                };
                list.append(py_prov)?;
            } else {
                list.append(py.None())?;
            }
        } else {
            list.append(py.None())?;
        }
    }

    Ok(list)
}

/// Get metadata from an einsum graph
///
/// Extracts all metadata (names, spans, provenance, attributes) from graph nodes.
///
/// Args:
///     graph: EinsumGraph to extract metadata from
///
/// Returns:
///     list: List of metadata dictionaries for each node
///
/// Example:
///     >>> import pytensorlogic as tl
///     >>> expr = tl.pred("knows", [tl.var("x"), tl.var("y")])
///     >>> graph = tl.compile(expr)
///     >>> metadata = tl.get_metadata(graph)
///     >>> for meta in metadata:
///     ...     if meta:
///     ...         print(f"Node: {meta.get('name', 'unnamed')}")
#[pyfunction(name = "get_metadata")]
pub fn py_get_metadata<'py>(
    py: Python<'py>,
    graph: &crate::types::PyEinsumGraph,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);

    // Get the inner graph
    let inner_graph = &graph.inner;

    // Extract metadata from each node
    for node in &inner_graph.nodes {
        if let Some(metadata) = &node.metadata {
            let dict = PyDict::new(py);

            if let Some(name) = &metadata.name {
                dict.set_item("name", name)?;
            }

            if let Some(span) = &metadata.span {
                let py_span = PySourceSpan {
                    inner: span.clone(),
                };
                dict.set_item("span", py_span)?;
            }

            if let Some(provenance) = &metadata.provenance {
                let py_prov = PyProvenance {
                    inner: provenance.clone(),
                };
                dict.set_item("provenance", py_prov)?;
            }

            let attrs_dict = PyDict::new(py);
            for (key, value) in &metadata.attributes {
                attrs_dict.set_item(key, value)?;
            }
            dict.set_item("attributes", attrs_dict)?;

            list.append(dict)?;
        } else {
            list.append(py.None())?;
        }
    }

    Ok(list)
}

/// Create a provenance tracker
///
/// Helper function to create a new provenance tracker.
///
/// Args:
///     enable_rdfstar: Enable RDF* support (default: False)
///
/// Returns:
///     ProvenanceTracker: New provenance tracker
///
/// Example:
///     >>> from pytensorlogic import provenance_tracker
///     >>> tracker = provenance_tracker(enable_rdfstar=True)
///     >>> tracker.track_entity("http://example.org/alice", 0)
#[pyfunction(name = "provenance_tracker", signature = (enable_rdfstar = false))]
pub fn py_provenance_tracker(enable_rdfstar: bool) -> PyProvenanceTracker {
    PyProvenanceTracker::new(enable_rdfstar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location() {
        let loc = PySourceLocation::new("test.tl".to_string(), 10, 5);
        assert_eq!(loc.file(), "test.tl");
        assert_eq!(loc.line(), 10);
        assert_eq!(loc.column(), 5);
    }

    #[test]
    fn test_source_span() {
        let start = PySourceLocation::new("test.tl".to_string(), 10, 1);
        let end = PySourceLocation::new("test.tl".to_string(), 15, 40);
        let span = PySourceSpan::new(start.clone(), end.clone());
        assert_eq!(span.start().line(), 10);
        assert_eq!(span.end().line(), 15);
    }

    #[test]
    fn test_provenance() {
        let mut prov = PyProvenance::new();
        prov.set_rule_id("rule_1".to_string());
        prov.set_source_file("test.tl".to_string());
        prov.add_attribute("author".to_string(), "alice".to_string());

        assert_eq!(prov.rule_id(), Some("rule_1".to_string()));
        assert_eq!(prov.source_file(), Some("test.tl".to_string()));
        assert_eq!(prov.get_attribute("author"), Some("alice".to_string()));
    }

    #[test]
    fn test_provenance_tracker() {
        let mut tracker = PyProvenanceTracker::new(false);

        tracker.track_entity("http://example.org/alice".to_string(), 0);
        tracker.track_entity("http://example.org/bob".to_string(), 1);

        assert_eq!(
            tracker.get_entity(0),
            Some("http://example.org/alice".to_string())
        );
        assert_eq!(tracker.get_tensor("http://example.org/bob"), Some(1));
    }

    #[test]
    fn test_provenance_tracker_shapes() {
        let mut tracker = PyProvenanceTracker::new(false);

        tracker.track_shape(
            "http://example.org/PersonShape".to_string(),
            "Person(x)".to_string(),
            0,
        );

        // Verify shape was tracked
        assert_eq!(tracker.inner.shape_to_rule.len(), 1);
        assert_eq!(tracker.inner.node_to_shape.len(), 1);
    }

    #[test]
    fn test_provenance_tracker_json() {
        let mut tracker = PyProvenanceTracker::new(false);
        tracker.track_entity("http://example.org/alice".to_string(), 0);

        let json = tracker.to_json().unwrap();
        assert!(json.contains("alice"));

        let restored = PyProvenanceTracker::from_json(&json).unwrap();
        assert_eq!(
            restored.get_entity(0),
            Some("http://example.org/alice".to_string())
        );
    }
}
