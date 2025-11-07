//! Enhanced metadata preservation for RDF entities.
//!
//! Supports multilingual labels, comments, and custom annotation properties.

use anyhow::{Context, Result};
use indexmap::IndexMap;
use oxrdf::{Graph, NamedNode, NamedOrBlankNodeRef, TermRef};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Language-tagged string value
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LangString {
    pub value: String,
    pub lang: Option<String>,
}

impl LangString {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            lang: None,
        }
    }

    pub fn with_lang(value: impl Into<String>, lang: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            lang: Some(lang.into()),
        }
    }

    /// Get the value in a preferred language, or fallback to any language
    pub fn preferred<'a>(items: &'a [Self], preferred_lang: Option<&str>) -> Option<&'a Self> {
        if let Some(lang) = preferred_lang {
            // First try exact match
            if let Some(item) = items.iter().find(|item| item.lang.as_deref() == Some(lang)) {
                return Some(item);
            }
        }

        // Then try English as fallback
        if let Some(item) = items.iter().find(|item| item.lang.as_deref() == Some("en")) {
            return Some(item);
        }

        // Then try any with no language tag
        if let Some(item) = items.iter().find(|item| item.lang.is_none()) {
            return Some(item);
        }

        // Finally, any value
        items.first()
    }
}

/// Rich metadata for an RDF entity (class or property)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityMetadata {
    /// Original IRI
    pub iri: String,

    /// Labels in multiple languages
    pub labels: Vec<LangString>,

    /// Comments/descriptions in multiple languages
    pub comments: Vec<LangString>,

    /// Custom annotation properties
    pub annotations: HashMap<String, Vec<String>>,

    /// When this entity was first seen (ISO 8601 timestamp)
    pub created_at: Option<String>,

    /// When this entity was last updated
    pub updated_at: Option<String>,

    /// Version information
    pub version: Option<String>,

    /// Source file/location where this was defined
    pub source: Option<String>,
}

impl EntityMetadata {
    pub fn new(iri: impl Into<String>) -> Self {
        Self {
            iri: iri.into(),
            labels: Vec::new(),
            comments: Vec::new(),
            annotations: HashMap::new(),
            created_at: None,
            updated_at: None,
            version: None,
            source: None,
        }
    }

    /// Add a label with optional language tag
    pub fn add_label(&mut self, value: impl Into<String>, lang: Option<String>) {
        self.labels.push(LangString {
            value: value.into(),
            lang,
        });
    }

    /// Add a comment with optional language tag
    pub fn add_comment(&mut self, value: impl Into<String>, lang: Option<String>) {
        self.comments.push(LangString {
            value: value.into(),
            lang,
        });
    }

    /// Add a custom annotation
    pub fn add_annotation(&mut self, property: impl Into<String>, value: impl Into<String>) {
        self.annotations
            .entry(property.into())
            .or_default()
            .push(value.into());
    }

    /// Get the preferred label for a language
    pub fn get_label(&self, preferred_lang: Option<&str>) -> Option<&str> {
        LangString::preferred(&self.labels, preferred_lang).map(|ls| ls.value.as_str())
    }

    /// Get the preferred comment for a language
    pub fn get_comment(&self, preferred_lang: Option<&str>) -> Option<&str> {
        LangString::preferred(&self.comments, preferred_lang).map(|ls| ls.value.as_str())
    }

    /// Get all values for an annotation property
    pub fn get_annotation(&self, property: &str) -> Option<&Vec<String>> {
        self.annotations.get(property)
    }

    /// Mark entity as created now
    pub fn mark_created(&mut self) {
        let now = chrono::Utc::now().to_rfc3339();
        self.created_at = Some(now.clone());
        self.updated_at = Some(now);
    }

    /// Mark entity as updated now
    pub fn mark_updated(&mut self) {
        self.updated_at = Some(chrono::Utc::now().to_rfc3339());
    }
}

/// Enhanced metadata store for all RDF entities
#[derive(Debug, Clone, Default)]
pub struct MetadataStore {
    /// Entity IRI → Metadata
    entities: IndexMap<String, EntityMetadata>,

    /// Known annotation properties and their IRIs
    annotation_properties: HashMap<String, String>,

    /// Preferred language for lookups
    preferred_language: Option<String>,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self {
            entities: IndexMap::new(),
            annotation_properties: HashMap::new(),
            preferred_language: None,
        }
    }

    /// Set the preferred language for label/comment lookups
    pub fn set_preferred_language(&mut self, lang: impl Into<String>) {
        self.preferred_language = Some(lang.into());
    }

    /// Register a custom annotation property
    pub fn register_annotation_property(
        &mut self,
        iri: impl Into<String>,
        name: impl Into<String>,
    ) {
        self.annotation_properties.insert(name.into(), iri.into());
    }

    /// Get or create metadata for an entity
    pub fn get_or_create(&mut self, iri: impl Into<String>) -> &mut EntityMetadata {
        let iri = iri.into();
        self.entities
            .entry(iri.clone())
            .or_insert_with(|| EntityMetadata::new(iri))
    }

    /// Get metadata for an entity
    pub fn get(&self, iri: &str) -> Option<&EntityMetadata> {
        self.entities.get(iri)
    }

    /// Get mutable metadata for an entity
    pub fn get_mut(&mut self, iri: &str) -> Option<&mut EntityMetadata> {
        self.entities.get_mut(iri)
    }

    /// Extract metadata from an RDF graph
    pub fn extract_from_graph(&mut self, graph: &Graph) -> Result<()> {
        let rdfs_label =
            NamedNode::new("http://www.w3.org/2000/01/rdf-schema#label").context("Invalid IRI")?;
        let rdfs_comment = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#comment")
            .context("Invalid IRI")?;

        for triple in graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subject) = triple.subject {
                let subject_iri = subject.as_str().to_string();

                // Extract labels
                if triple.predicate == rdfs_label.as_ref() {
                    if let TermRef::Literal(lit) = triple.object {
                        let meta = self.get_or_create(&subject_iri);
                        meta.add_label(
                            lit.value().to_string(),
                            lit.language().map(|s| s.to_string()),
                        );
                    }
                }

                // Extract comments
                if triple.predicate == rdfs_comment.as_ref() {
                    if let TermRef::Literal(lit) = triple.object {
                        let meta = self.get_or_create(&subject_iri);
                        meta.add_comment(
                            lit.value().to_string(),
                            lit.language().map(|s| s.to_string()),
                        );
                    }
                }

                // Extract custom annotations
                let predicate_iri = triple.predicate.as_str().to_string();
                let annotation_name = self
                    .annotation_properties
                    .iter()
                    .find(|(_, iri)| **iri == predicate_iri)
                    .map(|(name, _)| name.clone());

                if let Some(name) = annotation_name {
                    if let TermRef::Literal(lit) = triple.object {
                        let meta = self.get_or_create(&subject_iri);
                        meta.add_annotation(name, lit.value().to_string());
                    }
                }
            }
        }

        Ok(())
    }

    /// Get all entities with metadata
    pub fn all_entities(&self) -> &IndexMap<String, EntityMetadata> {
        &self.entities
    }

    /// Export metadata to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.entities).context("Failed to serialize metadata")
    }

    /// Load metadata from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        let entities: IndexMap<String, EntityMetadata> =
            serde_json::from_str(json).context("Failed to deserialize metadata")?;
        Ok(Self {
            entities,
            annotation_properties: HashMap::new(),
            preferred_language: None,
        })
    }

    /// Find entities by label (case-insensitive partial match)
    pub fn find_by_label(&self, query: &str) -> Vec<&EntityMetadata> {
        let query_lower = query.to_lowercase();
        self.entities
            .values()
            .filter(|meta| {
                meta.labels
                    .iter()
                    .any(|label| label.value.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Find entities with missing labels
    pub fn find_missing_labels(&self) -> Vec<&str> {
        self.entities
            .values()
            .filter(|meta| meta.labels.is_empty())
            .map(|meta| meta.iri.as_str())
            .collect()
    }

    /// Find entities with missing comments
    pub fn find_missing_comments(&self) -> Vec<&str> {
        self.entities
            .values()
            .filter(|meta| meta.comments.is_empty())
            .map(|meta| meta.iri.as_str())
            .collect()
    }

    /// Get statistics about metadata coverage
    pub fn stats(&self) -> MetadataStats {
        let total = self.entities.len();
        let with_labels = self
            .entities
            .values()
            .filter(|m| !m.labels.is_empty())
            .count();
        let with_comments = self
            .entities
            .values()
            .filter(|m| !m.comments.is_empty())
            .count();
        let with_annotations = self
            .entities
            .values()
            .filter(|m| !m.annotations.is_empty())
            .count();

        let total_labels: usize = self.entities.values().map(|m| m.labels.len()).sum();
        let total_comments: usize = self.entities.values().map(|m| m.comments.len()).sum();

        MetadataStats {
            total_entities: total,
            entities_with_labels: with_labels,
            entities_with_comments: with_comments,
            entities_with_annotations: with_annotations,
            total_labels,
            total_comments,
        }
    }
}

/// Statistics about metadata coverage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataStats {
    pub total_entities: usize,
    pub entities_with_labels: usize,
    pub entities_with_comments: usize,
    pub entities_with_annotations: usize,
    pub total_labels: usize,
    pub total_comments: usize,
}

impl std::fmt::Display for MetadataStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Metadata Statistics:")?;
        writeln!(f, "  Total entities: {}", self.total_entities)?;
        writeln!(
            f,
            "  Entities with labels: {} ({:.1}%)",
            self.entities_with_labels,
            (self.entities_with_labels as f64 / self.total_entities as f64) * 100.0
        )?;
        writeln!(
            f,
            "  Entities with comments: {} ({:.1}%)",
            self.entities_with_comments,
            (self.entities_with_comments as f64 / self.total_entities as f64) * 100.0
        )?;
        writeln!(
            f,
            "  Entities with annotations: {} ({:.1}%)",
            self.entities_with_annotations,
            (self.entities_with_annotations as f64 / self.total_entities as f64) * 100.0
        )?;
        writeln!(f, "  Total labels: {}", self.total_labels)?;
        writeln!(f, "  Total comments: {}", self.total_comments)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxttl::TurtleParser;

    #[test]
    fn test_lang_string_preferred() {
        let items = vec![
            LangString::with_lang("Hello", "en"),
            LangString::with_lang("Bonjour", "fr"),
            LangString::with_lang("Hola", "es"),
        ];

        assert_eq!(
            LangString::preferred(&items, Some("fr")).unwrap().value,
            "Bonjour"
        );
        assert_eq!(
            LangString::preferred(&items, Some("en")).unwrap().value,
            "Hello"
        );
        assert_eq!(
            LangString::preferred(&items, Some("de")).unwrap().value,
            "Hello"
        ); // Fallback to English
    }

    #[test]
    fn test_entity_metadata() {
        let mut meta = EntityMetadata::new("http://example.org/Person");
        meta.add_label("Person", Some("en".to_string()));
        meta.add_label("Personne", Some("fr".to_string()));
        meta.add_comment("A human being", Some("en".to_string()));
        meta.add_annotation("creator", "John Doe");

        assert_eq!(meta.get_label(Some("en")), Some("Person"));
        assert_eq!(meta.get_label(Some("fr")), Some("Personne"));
        assert_eq!(meta.get_comment(Some("en")), Some("A human being"));
        assert_eq!(meta.get_annotation("creator").unwrap()[0], "John Doe");
    }

    #[test]
    fn test_metadata_store() {
        let mut store = MetadataStore::new();

        let meta = store.get_or_create("http://example.org/Person");
        meta.add_label("Person", Some("en".to_string()));

        let meta = store.get_or_create("http://example.org/Organization");
        meta.add_label("Organization", Some("en".to_string()));

        assert_eq!(store.all_entities().len(), 2);
        assert!(store.get("http://example.org/Person").is_some());
    }

    #[test]
    fn test_extract_from_graph() {
        let turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Person rdfs:label "Person"@en ;
                      rdfs:label "Personne"@fr ;
                      rdfs:comment "A human being"@en .

            ex:Organization rdfs:label "Organization"@en .
        "#;

        let mut graph = Graph::new();
        let parser = TurtleParser::new().for_slice(turtle.as_bytes());
        for triple in parser {
            graph.insert(&triple.unwrap());
        }

        let mut store = MetadataStore::new();
        store.extract_from_graph(&graph).unwrap();

        let person = store.get("http://example.org/Person").unwrap();
        assert_eq!(person.labels.len(), 2);
        assert_eq!(person.get_label(Some("en")), Some("Person"));
        assert_eq!(person.get_label(Some("fr")), Some("Personne"));
        assert_eq!(person.comments.len(), 1);
    }

    #[test]
    fn test_find_by_label() {
        let mut store = MetadataStore::new();

        let meta = store.get_or_create("http://example.org/Person");
        meta.add_label("Person", Some("en".to_string()));

        let meta = store.get_or_create("http://example.org/Organization");
        meta.add_label("Organization", Some("en".to_string()));

        let results = store.find_by_label("person");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].iri, "http://example.org/Person");

        let results = store.find_by_label("org");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_missing_labels() {
        let mut store = MetadataStore::new();

        let meta = store.get_or_create("http://example.org/Person");
        meta.add_label("Person", Some("en".to_string()));

        store.get_or_create("http://example.org/NoLabel");

        let missing = store.find_missing_labels();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0], "http://example.org/NoLabel");
    }

    #[test]
    fn test_metadata_stats() {
        let mut store = MetadataStore::new();

        let meta = store.get_or_create("http://example.org/Person");
        meta.add_label("Person", Some("en".to_string()));
        meta.add_comment("A human being", Some("en".to_string()));

        store.get_or_create("http://example.org/NoMetadata");

        let stats = store.stats();
        assert_eq!(stats.total_entities, 2);
        assert_eq!(stats.entities_with_labels, 1);
        assert_eq!(stats.entities_with_comments, 1);
    }

    #[test]
    fn test_json_serialization() {
        let mut store = MetadataStore::new();

        let meta = store.get_or_create("http://example.org/Person");
        meta.add_label("Person", Some("en".to_string()));
        meta.add_comment("A human being", Some("en".to_string()));

        let json = store.to_json().unwrap();
        let loaded = MetadataStore::from_json(&json).unwrap();

        assert_eq!(loaded.all_entities().len(), 1);
        let person = loaded.get("http://example.org/Person").unwrap();
        assert_eq!(person.get_label(Some("en")), Some("Person"));
    }
}
