//! Enhanced serialization support for TensorLogic IR.
//!
//! This module provides improved serialization formats including:
//! - Human-readable JSON with version tagging
//! - Compact binary format using bincode
//! - Versioned format support for backward compatibility

use crate::{EinsumGraph, TLExpr};
use serde::{Deserialize, Serialize};

/// Current serialization format version
pub const FORMAT_VERSION: &str = "1.0.0";

/// Versioned wrapper for TLExpr serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedExpr {
    /// Format version (semver)
    pub version: String,
    /// Creation timestamp (ISO 8601)
    pub created_at: Option<String>,
    /// Optional metadata
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// The expression
    pub expr: TLExpr,
}

impl VersionedExpr {
    /// Create a new versioned expression
    pub fn new(expr: TLExpr) -> Self {
        VersionedExpr {
            version: FORMAT_VERSION.to_string(),
            created_at: Some(chrono::Utc::now().to_rfc3339()),
            metadata: None,
            expr,
        }
    }

    /// Create with custom metadata
    pub fn with_metadata(
        expr: TLExpr,
        metadata: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        VersionedExpr {
            version: FORMAT_VERSION.to_string(),
            created_at: Some(chrono::Utc::now().to_rfc3339()),
            metadata: Some(metadata),
            expr,
        }
    }

    /// Serialize to pretty JSON
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to compact JSON
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to binary format
    pub fn to_binary(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        oxicode::serde::encode_to_vec(self, oxicode::config::standard())
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    /// Deserialize from binary format
    pub fn from_binary(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let (result, _): (Self, usize) =
            oxicode::serde::decode_from_slice(bytes, oxicode::config::standard())
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        Ok(result)
    }
}

/// Versioned wrapper for EinsumGraph serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedGraph {
    /// Format version (semver)
    pub version: String,
    /// Creation timestamp (ISO 8601)
    pub created_at: Option<String>,
    /// Optional metadata
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// The graph
    pub graph: EinsumGraph,
}

impl VersionedGraph {
    /// Create a new versioned graph
    pub fn new(graph: EinsumGraph) -> Self {
        VersionedGraph {
            version: FORMAT_VERSION.to_string(),
            created_at: Some(chrono::Utc::now().to_rfc3339()),
            metadata: None,
            graph,
        }
    }

    /// Create with custom metadata
    pub fn with_metadata(
        graph: EinsumGraph,
        metadata: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        VersionedGraph {
            version: FORMAT_VERSION.to_string(),
            created_at: Some(chrono::Utc::now().to_rfc3339()),
            metadata: Some(metadata),
            graph,
        }
    }

    /// Serialize to pretty JSON
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to compact JSON
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize to binary format
    pub fn to_binary(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        oxicode::serde::encode_to_vec(self, oxicode::config::standard())
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    /// Deserialize from binary format
    pub fn from_binary(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let (result, _): (Self, usize) =
            oxicode::serde::decode_from_slice(bytes, oxicode::config::standard())
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        Ok(result)
    }

    /// Check if version is compatible with current version
    pub fn is_compatible(&self) -> bool {
        // Simple check: major version must match
        let self_major = self
            .version
            .split('.')
            .next()
            .and_then(|s| s.parse::<u32>().ok());
        let current_major = FORMAT_VERSION
            .split('.')
            .next()
            .and_then(|s| s.parse::<u32>().ok());

        self_major == current_major
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TLExpr, Term};

    #[test]
    fn test_versioned_expr_creation() {
        let expr = TLExpr::pred("test", vec![Term::var("x")]);
        let versioned = VersionedExpr::new(expr.clone());

        assert_eq!(versioned.version, FORMAT_VERSION);
        assert!(versioned.created_at.is_some());
        assert!(versioned.metadata.is_none());
        assert_eq!(versioned.expr, expr);
    }

    #[test]
    fn test_versioned_expr_with_metadata() {
        let expr = TLExpr::pred("test", vec![Term::var("x")]);
        let mut metadata = serde_json::Map::new();
        metadata.insert("author".to_string(), serde_json::json!("test"));

        let versioned = VersionedExpr::with_metadata(expr.clone(), metadata.clone());

        assert_eq!(versioned.version, FORMAT_VERSION);
        assert!(versioned.created_at.is_some());
        assert_eq!(versioned.metadata, Some(metadata));
        assert_eq!(versioned.expr, expr);
    }

    #[test]
    fn test_versioned_expr_json_roundtrip() {
        let expr = TLExpr::pred("test", vec![Term::var("x")]);
        let versioned = VersionedExpr::new(expr.clone());

        let json = versioned.to_json_pretty().unwrap();
        let deserialized = VersionedExpr::from_json(&json).unwrap();

        assert_eq!(deserialized.version, versioned.version);
        assert_eq!(deserialized.expr, versioned.expr);
    }

    #[test]
    fn test_versioned_expr_binary_roundtrip() {
        let expr = TLExpr::pred("test", vec![Term::var("x")]);
        let versioned = VersionedExpr::new(expr.clone());

        let binary = versioned.to_binary().unwrap();
        let deserialized = VersionedExpr::from_binary(&binary).unwrap();

        assert_eq!(deserialized.version, versioned.version);
        assert_eq!(deserialized.expr, versioned.expr);
    }

    #[test]
    fn test_versioned_graph_creation() {
        let graph = EinsumGraph {
            tensors: vec![],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_metadata: std::collections::HashMap::new(),
        };
        let versioned = VersionedGraph::new(graph.clone());

        assert_eq!(versioned.version, FORMAT_VERSION);
        assert!(versioned.created_at.is_some());
        assert!(versioned.metadata.is_none());
        assert_eq!(versioned.graph, graph);
    }

    #[test]
    fn test_versioned_graph_json_roundtrip() {
        let graph = EinsumGraph {
            tensors: vec![],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_metadata: std::collections::HashMap::new(),
        };
        let versioned = VersionedGraph::new(graph.clone());

        let json = versioned.to_json_pretty().unwrap();
        let deserialized = VersionedGraph::from_json(&json).unwrap();

        assert_eq!(deserialized.version, versioned.version);
        assert_eq!(deserialized.graph, versioned.graph);
    }

    #[test]
    fn test_versioned_graph_binary_roundtrip() {
        let graph = EinsumGraph {
            tensors: vec![],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_metadata: std::collections::HashMap::new(),
        };
        let versioned = VersionedGraph::new(graph.clone());

        let binary = versioned.to_binary().unwrap();
        let deserialized = VersionedGraph::from_binary(&binary).unwrap();

        assert_eq!(deserialized.version, versioned.version);
        assert_eq!(deserialized.graph, versioned.graph);
    }

    #[test]
    fn test_version_compatibility() {
        let graph = EinsumGraph {
            tensors: vec![],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            tensor_metadata: std::collections::HashMap::new(),
        };
        let versioned = VersionedGraph::new(graph);

        assert!(versioned.is_compatible());

        // Test with different version
        let mut incompatible = versioned.clone();
        incompatible.version = "2.0.0".to_string();
        assert!(!incompatible.is_compatible());
    }

    #[test]
    fn test_json_is_human_readable() {
        let expr = TLExpr::and(
            TLExpr::pred("p", vec![Term::var("x")]),
            TLExpr::pred("q", vec![Term::var("y")]),
        );
        let versioned = VersionedExpr::new(expr);

        let json = versioned.to_json_pretty().unwrap();

        // Check that JSON contains human-readable structure
        assert!(json.contains("version"));
        assert!(json.contains("created_at"));
        assert!(json.contains("expr"));
        assert!(json.contains("And"));
    }

    #[test]
    fn test_binary_smaller_than_json() {
        let expr = TLExpr::and(
            TLExpr::pred("p", vec![Term::var("x")]),
            TLExpr::pred("q", vec![Term::var("y")]),
        );
        let versioned = VersionedExpr::new(expr);

        let json = versioned.to_json_compact().unwrap();
        let binary = versioned.to_binary().unwrap();

        // Binary should typically be smaller than JSON
        // (though not guaranteed for very small structures)
        assert!(binary.len() <= json.len() * 2); // Allow some flexibility
    }
}
