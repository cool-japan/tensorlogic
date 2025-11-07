//! RDF* (RDF-star) support for statement-level metadata and provenance tracking.
//!
//! RDF* extends RDF with the ability to make statements about statements using
//! quoted triples. This is particularly useful for:
//! - Provenance tracking (who asserted what, when)
//! - Confidence/certainty scores
//! - Temporal validity
//! - Source attribution
//!
//! ## Example
//!
//! ```turtle
//! @prefix ex: <http://example.org/> .
//! @prefix prov: <http://www.w3.org/ns/prov#> .
//!
//! # A regular triple
//! ex:alice ex:knows ex:bob .
//!
//! # An RDF* statement about the triple above
//! <<ex:alice ex:knows ex:bob>> prov:wasGeneratedBy ex:inference_rule_42 ;
//!                               ex:confidence 0.95 ;
//!                               prov:generatedAtTime "2025-11-06T10:00:00Z" .
//! ```

use anyhow::{Context, Result};
use indexmap::IndexMap;
use oxrdf::Triple;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a quoted triple (RDF* triple reference)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuotedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl QuotedTriple {
    /// Create a new quoted triple
    pub fn new(subject: String, predicate: String, object: String) -> Self {
        QuotedTriple {
            subject,
            predicate,
            object,
        }
    }

    /// Create from an oxrdf Triple
    pub fn from_triple(triple: &Triple) -> Self {
        QuotedTriple {
            subject: triple.subject.to_string(),
            predicate: triple.predicate.to_string(),
            object: triple.object.to_string(),
        }
    }

    /// Convert to RDF* Turtle syntax
    pub fn to_turtle_syntax(&self) -> String {
        format!("<<{} {} {}>>", self.subject, self.predicate, self.object)
    }

    /// Convert to a unique identifier string
    pub fn to_id(&self) -> String {
        format!("{}|{}|{}", self.subject, self.predicate, self.object)
    }
}

/// Metadata about a statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatementMetadata {
    /// The quoted triple this metadata refers to
    pub statement: QuotedTriple,
    /// Confidence/certainty score (0.0 to 1.0)
    pub confidence: Option<f64>,
    /// Source of the statement
    pub source: Option<String>,
    /// Generation method (e.g., "inference_rule_42", "user_input", "owl_reasoning")
    pub generated_by: Option<String>,
    /// Timestamp when the statement was generated
    pub generated_at: Option<String>,
    /// Rule ID that generated this statement (for TensorLogic integration)
    pub rule_id: Option<String>,
    /// Custom metadata as key-value pairs
    pub custom: HashMap<String, String>,
}

impl StatementMetadata {
    /// Create new empty metadata for a statement
    pub fn new(statement: QuotedTriple) -> Self {
        StatementMetadata {
            statement,
            confidence: None,
            source: None,
            generated_by: None,
            generated_at: None,
            rule_id: None,
            custom: HashMap::new(),
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set source
    pub fn with_source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Set generation method
    pub fn with_generated_by(mut self, generated_by: String) -> Self {
        self.generated_by = Some(generated_by);
        self
    }

    /// Set generation timestamp
    pub fn with_generated_at(mut self, generated_at: String) -> Self {
        self.generated_at = Some(generated_at);
        self
    }

    /// Set rule ID
    pub fn with_rule_id(mut self, rule_id: String) -> Self {
        self.rule_id = Some(rule_id);
        self
    }

    /// Add custom metadata
    pub fn with_custom(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }

    /// Export to RDF* Turtle format
    pub fn to_turtle(&self) -> String {
        let mut turtle = String::new();
        turtle.push_str(&self.statement.to_turtle_syntax());
        turtle.push(' ');

        let mut predicates = Vec::new();

        if let Some(conf) = self.confidence {
            predicates.push(format!(
                "<http://example.org/confidence> \"{}\"^^<http://www.w3.org/2001/XMLSchema#double>",
                conf
            ));
        }

        if let Some(ref source) = self.source {
            predicates.push(format!(
                "<http://www.w3.org/ns/prov#hadPrimarySource> <{}>",
                source
            ));
        }

        if let Some(ref gen_by) = self.generated_by {
            predicates.push(format!(
                "<http://www.w3.org/ns/prov#wasGeneratedBy> <{}>",
                gen_by
            ));
        }

        if let Some(ref gen_at) = self.generated_at {
            predicates.push(format!(
                "<http://www.w3.org/ns/prov#generatedAtTime> \"{}\"^^<http://www.w3.org/2001/XMLSchema#dateTime>",
                gen_at
            ));
        }

        if let Some(ref rule_id) = self.rule_id {
            predicates.push(format!("<http://tensorlogic.org/rule> <{}>", rule_id));
        }

        for (key, value) in &self.custom {
            predicates.push(format!("<{}> \"{}\"", key, value));
        }

        turtle.push_str(&predicates.join(" ;\n    "));
        turtle.push_str(" .");

        turtle
    }
}

/// RDF* provenance store for tracking statement-level metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfStarProvenanceStore {
    /// Map from statement ID to metadata
    metadata: IndexMap<String, StatementMetadata>,
    /// Reverse index: predicate → statements
    predicate_index: HashMap<String, Vec<String>>,
    /// Reverse index: source → statements
    source_index: HashMap<String, Vec<String>>,
    /// Reverse index: rule_id → statements
    rule_index: HashMap<String, Vec<String>>,
}

impl RdfStarProvenanceStore {
    /// Create a new empty provenance store
    pub fn new() -> Self {
        RdfStarProvenanceStore {
            metadata: IndexMap::new(),
            predicate_index: HashMap::new(),
            source_index: HashMap::new(),
            rule_index: HashMap::new(),
        }
    }

    /// Add metadata for a statement
    pub fn add_metadata(&mut self, metadata: StatementMetadata) {
        let stmt_id = metadata.statement.to_id();
        let predicate = metadata.statement.predicate.clone();

        // Update predicate index
        self.predicate_index
            .entry(predicate)
            .or_default()
            .push(stmt_id.clone());

        // Update source index
        if let Some(ref source) = metadata.source {
            self.source_index
                .entry(source.clone())
                .or_default()
                .push(stmt_id.clone());
        }

        // Update rule index
        if let Some(ref rule_id) = metadata.rule_id {
            self.rule_index
                .entry(rule_id.clone())
                .or_default()
                .push(stmt_id.clone());
        }

        self.metadata.insert(stmt_id, metadata);
    }

    /// Get metadata for a statement
    pub fn get_metadata(&self, statement: &QuotedTriple) -> Option<&StatementMetadata> {
        self.metadata.get(&statement.to_id())
    }

    /// Get all statements with a given predicate
    pub fn get_by_predicate(&self, predicate: &str) -> Vec<&StatementMetadata> {
        self.predicate_index
            .get(predicate)
            .map(|ids| ids.iter().filter_map(|id| self.metadata.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all statements from a given source
    pub fn get_by_source(&self, source: &str) -> Vec<&StatementMetadata> {
        self.source_index
            .get(source)
            .map(|ids| ids.iter().filter_map(|id| self.metadata.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all statements generated by a given rule
    pub fn get_by_rule(&self, rule_id: &str) -> Vec<&StatementMetadata> {
        self.rule_index
            .get(rule_id)
            .map(|ids| ids.iter().filter_map(|id| self.metadata.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get statements with confidence above a threshold
    pub fn get_by_min_confidence(&self, min_confidence: f64) -> Vec<&StatementMetadata> {
        self.metadata
            .values()
            .filter(|m| m.confidence.map(|c| c >= min_confidence).unwrap_or(false))
            .collect()
    }

    /// Get all metadata entries
    pub fn all_metadata(&self) -> Vec<&StatementMetadata> {
        self.metadata.values().collect()
    }

    /// Get the number of tracked statements
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// Export all metadata to RDF* Turtle format
    pub fn to_turtle(&self) -> String {
        let mut turtle = String::new();
        turtle.push_str("# RDF* Provenance Store\n");
        turtle.push_str("@prefix prov: <http://www.w3.org/ns/prov#> .\n");
        turtle.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n");
        turtle.push_str("@prefix tl: <http://tensorlogic.org/> .\n\n");

        for metadata in self.metadata.values() {
            turtle.push_str(&metadata.to_turtle());
            turtle.push_str("\n\n");
        }

        turtle
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.metadata).context("Failed to serialize to JSON")
    }

    /// Clear all metadata
    pub fn clear(&mut self) {
        self.metadata.clear();
        self.predicate_index.clear();
        self.source_index.clear();
        self.rule_index.clear();
    }

    /// Get statistics about the provenance store
    pub fn get_stats(&self) -> ProvenanceStats {
        let total_statements = self.metadata.len();
        let with_confidence = self
            .metadata
            .values()
            .filter(|m| m.confidence.is_some())
            .count();
        let with_source = self
            .metadata
            .values()
            .filter(|m| m.source.is_some())
            .count();
        let with_rule = self
            .metadata
            .values()
            .filter(|m| m.rule_id.is_some())
            .count();
        let unique_rules = self.rule_index.len();
        let unique_sources = self.source_index.len();

        ProvenanceStats {
            total_statements,
            with_confidence,
            with_source,
            with_rule,
            unique_rules,
            unique_sources,
        }
    }
}

impl Default for RdfStarProvenanceStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the provenance store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceStats {
    pub total_statements: usize,
    pub with_confidence: usize,
    pub with_source: usize,
    pub with_rule: usize,
    pub unique_rules: usize,
    pub unique_sources: usize,
}

/// Builder for creating statement metadata with provenance information
pub struct MetadataBuilder {
    statement: QuotedTriple,
    confidence: Option<f64>,
    source: Option<String>,
    generated_by: Option<String>,
    generated_at: Option<String>,
    rule_id: Option<String>,
    custom: HashMap<String, String>,
}

impl MetadataBuilder {
    /// Create a new metadata builder for a triple
    pub fn for_triple(subject: String, predicate: String, object: String) -> Self {
        MetadataBuilder {
            statement: QuotedTriple::new(subject, predicate, object),
            confidence: None,
            source: None,
            generated_by: None,
            generated_at: None,
            rule_id: None,
            custom: HashMap::new(),
        }
    }

    /// Create a new metadata builder from a quoted triple
    pub fn for_quoted_triple(statement: QuotedTriple) -> Self {
        MetadataBuilder {
            statement,
            confidence: None,
            source: None,
            generated_by: None,
            generated_at: None,
            rule_id: None,
            custom: HashMap::new(),
        }
    }

    /// Set confidence score (0.0 to 1.0)
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set source IRI
    pub fn source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Set generation method
    pub fn generated_by(mut self, generated_by: String) -> Self {
        self.generated_by = Some(generated_by);
        self
    }

    /// Set generation timestamp (ISO 8601 format)
    pub fn generated_at(mut self, generated_at: String) -> Self {
        self.generated_at = Some(generated_at);
        self
    }

    /// Set rule ID for TensorLogic integration
    pub fn rule_id(mut self, rule_id: String) -> Self {
        self.rule_id = Some(rule_id);
        self
    }

    /// Add custom metadata
    pub fn custom(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }

    /// Build the metadata
    pub fn build(self) -> StatementMetadata {
        StatementMetadata {
            statement: self.statement,
            confidence: self.confidence,
            source: self.source,
            generated_by: self.generated_by,
            generated_at: self.generated_at,
            rule_id: self.rule_id,
            custom: self.custom,
        }
    }
}
