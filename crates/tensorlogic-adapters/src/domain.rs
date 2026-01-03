//! Domain information and management.

use serde::{Deserialize, Serialize};

use crate::metadata::Metadata;
use crate::parametric::ParametricType;

/// Domain information including cardinality and optional element enumeration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainInfo {
    pub name: String,
    pub cardinality: usize,
    pub elements: Option<Vec<String>>,
    pub description: Option<String>,
    /// Rich metadata including provenance, documentation, tags
    pub metadata: Option<Metadata>,
    /// Parametric type definition (e.g., `List<T>`)
    pub parametric_type: Option<ParametricType>,
}

impl DomainInfo {
    pub fn new(name: impl Into<String>, cardinality: usize) -> Self {
        DomainInfo {
            name: name.into(),
            cardinality,
            elements: None,
            description: None,
            metadata: None,
            parametric_type: None,
        }
    }

    pub fn with_elements(name: impl Into<String>, elements: Vec<String>) -> Self {
        let cardinality = elements.len();
        DomainInfo {
            name: name.into(),
            cardinality,
            elements: Some(elements),
            description: None,
            metadata: None,
            parametric_type: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_parametric_type(mut self, ptype: ParametricType) -> Self {
        self.parametric_type = Some(ptype);
        self
    }

    pub fn has_element(&self, element: &str) -> bool {
        self.elements
            .as_ref()
            .map(|elems| elems.contains(&element.to_string()))
            .unwrap_or(false)
    }

    pub fn get_index(&self, element: &str) -> Option<usize> {
        self.elements
            .as_ref()
            .and_then(|elems| elems.iter().position(|e| e == element))
    }
}
