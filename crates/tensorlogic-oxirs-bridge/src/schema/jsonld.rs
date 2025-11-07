//! JSON-LD serialization support for RDF data export
//!
//! JSON-LD (JavaScript Object Notation for Linked Data) is a JSON-based
//! format for serializing Linked Data. It provides a way to express RDF
//! in JSON that is both human-readable and machine-processable.
//!
//! ## Format Specification
//!
//! JSON-LD uses special keywords starting with `@` to encode RDF semantics:
//! - `@context`: Defines namespace prefixes and type coercion
//! - `@id`: The IRI identifier for a resource
//! - `@type`: The RDF type(s) of a resource
//! - `@value`: The value of a literal
//! - `@language`: Language tag for literals
//!
//! ## Example
//!
//! ```json
//! {
//!   "@context": {
//!     "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
//!     "ex": "http://example.org/"
//!   },
//!   "@graph": [
//!     {
//!       "@id": "ex:Person",
//!       "@type": "rdfs:Class",
//!       "rdfs:label": "Person",
//!       "rdfs:comment": "A human being"
//!     }
//!   ]
//! }
//! ```

use super::{ClassInfo, PropertyInfo, SchemaAnalyzer};
use anyhow::Result;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// JSON-LD context builder for managing namespace prefixes
#[derive(Debug, Clone)]
pub struct JsonLdContext {
    prefixes: HashMap<String, String>,
}

impl JsonLdContext {
    /// Create a new JSON-LD context with standard prefixes
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();

        // Add standard RDF/RDFS/OWL prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert("sh".to_string(), "http://www.w3.org/ns/shacl#".to_string());

        JsonLdContext { prefixes }
    }

    /// Create a context from a JSON @context object
    pub fn from_json(context_value: &Value) -> Result<Self> {
        let mut context = Self::new();

        if let Some(context_obj) = context_value.as_object() {
            for (key, value) in context_obj {
                if let Some(namespace) = value.as_str() {
                    context.add_prefix(key.clone(), namespace.to_string());
                }
            }
        }

        Ok(context)
    }

    /// Add a custom prefix mapping
    pub fn add_prefix(&mut self, prefix: String, namespace: String) {
        self.prefixes.insert(prefix, namespace);
    }

    /// Get all prefix mappings
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Convert to JSON-LD @context object
    pub fn to_json(&self) -> Value {
        let mut context = Map::new();
        for (prefix, namespace) in &self.prefixes {
            context.insert(prefix.clone(), Value::String(namespace.clone()));
        }
        Value::Object(context)
    }

    /// Try to compact an IRI using registered prefixes
    pub fn compact_iri(&self, iri: &str) -> String {
        for (prefix, namespace) in &self.prefixes {
            if let Some(local_name) = iri.strip_prefix(namespace) {
                return format!("{}:{}", prefix, local_name);
            }
        }
        iri.to_string()
    }

    /// Expand a compacted IRI using registered prefixes
    pub fn expand_iri(&self, compact: &str) -> String {
        if let Some(colon_pos) = compact.find(':') {
            let prefix = &compact[..colon_pos];
            let local_name = &compact[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                return format!("{}{}", namespace, local_name);
            }
        }
        compact.to_string()
    }
}

impl Default for JsonLdContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaAnalyzer {
    /// Import RDF schema from JSON-LD format
    ///
    /// Parses a JSON-LD document and extracts RDF triples,
    /// adding them to the analyzer's graph.
    ///
    /// ## Example
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    ///
    /// let jsonld = r#"{
    ///   "@context": {
    ///     "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    ///     "ex": "http://example.org/"
    ///   },
    ///   "@graph": [
    ///     {
    ///       "@id": "ex:Person",
    ///       "@type": "rdfs:Class",
    ///       "rdfs:label": "Person"
    ///     }
    ///   ]
    /// }"#;
    ///
    /// let mut analyzer = SchemaAnalyzer::new();
    /// analyzer.load_jsonld(jsonld).unwrap();
    /// analyzer.analyze().unwrap();
    ///
    /// assert_eq!(analyzer.classes.len(), 1);
    /// ```
    pub fn load_jsonld(&mut self, jsonld: &str) -> Result<()> {
        // Parse JSON-LD document
        let document: Value = serde_json::from_str(jsonld)
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON-LD: {}", e))?;

        // Extract context
        let context = if let Some(context_value) = document.get("@context") {
            JsonLdContext::from_json(context_value)?
        } else {
            JsonLdContext::new()
        };

        // Extract graph
        let graph_array = document
            .get("@graph")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("JSON-LD document missing @graph"))?;

        // Convert each item in the graph to Turtle and load it
        let mut turtle_statements = Vec::new();

        // Add prefix declarations
        turtle_statements.push(self.context_to_turtle_prefixes(&context));

        // Convert graph items to Turtle
        for item in graph_array {
            if let Some(obj) = item.as_object() {
                turtle_statements.push(self.jsonld_object_to_turtle(obj, &context)?);
            }
        }

        // Join and load as Turtle
        let turtle = turtle_statements.join("\n\n");
        self.load_turtle(&turtle)?;

        Ok(())
    }

    /// Convert JSON-LD context to Turtle prefix declarations
    fn context_to_turtle_prefixes(&self, context: &JsonLdContext) -> String {
        let mut prefixes = Vec::new();
        for (prefix, namespace) in context.prefixes() {
            prefixes.push(format!("@prefix {}: <{}> .", prefix, namespace));
        }
        prefixes.join("\n")
    }

    /// Convert a JSON-LD object to Turtle statements
    fn jsonld_object_to_turtle(
        &self,
        obj: &Map<String, Value>,
        context: &JsonLdContext,
    ) -> Result<String> {
        let mut statements = Vec::new();

        // Get subject IRI
        let subject = obj
            .get("@id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("JSON-LD object missing @id"))?;

        let subject_iri = context.expand_iri(subject);

        // Process @type
        if let Some(type_value) = obj.get("@type") {
            let types = match type_value {
                Value::String(s) => vec![s.as_str()],
                Value::Array(arr) => arr.iter().filter_map(|v| v.as_str()).collect(),
                _ => vec![],
            };

            for type_str in types {
                let type_iri = context.expand_iri(type_str);
                statements.push(format!("<{}> a <{}> .", subject_iri, type_iri));
            }
        }

        // Process other properties
        for (key, value) in obj {
            if key.starts_with('@') {
                continue; // Skip JSON-LD keywords
            }

            let predicate_iri = context.expand_iri(key);

            match value {
                Value::String(s) => {
                    statements.push(format!(
                        "<{}> <{}> \"{}\" .",
                        subject_iri,
                        predicate_iri,
                        escape_turtle_literal(s)
                    ));
                }
                Value::Object(inner_obj) => {
                    // Handle @id references
                    if let Some(id_str) = inner_obj.get("@id").and_then(|v| v.as_str()) {
                        let object_iri = context.expand_iri(id_str);
                        statements.push(format!(
                            "<{}> <{}> <{}> .",
                            subject_iri, predicate_iri, object_iri
                        ));
                    }
                    // Handle @value literals
                    else if let Some(val_str) = inner_obj.get("@value").and_then(|v| v.as_str()) {
                        let mut literal = format!("\"{}\"", escape_turtle_literal(val_str));

                        // Add language tag if present
                        if let Some(lang) = inner_obj.get("@language").and_then(|v| v.as_str()) {
                            literal.push_str(&format!("@{}", lang));
                        }

                        statements.push(format!(
                            "<{}> <{}> {} .",
                            subject_iri, predicate_iri, literal
                        ));
                    }
                }
                Value::Array(arr) => {
                    for item in arr {
                        match item {
                            Value::String(s) => {
                                statements.push(format!(
                                    "<{}> <{}> \"{}\" .",
                                    subject_iri,
                                    predicate_iri,
                                    escape_turtle_literal(s)
                                ));
                            }
                            Value::Object(item_obj) => {
                                // Handle @id references
                                if let Some(id_str) = item_obj.get("@id").and_then(|v| v.as_str()) {
                                    let object_iri = context.expand_iri(id_str);
                                    statements.push(format!(
                                        "<{}> <{}> <{}> .",
                                        subject_iri, predicate_iri, object_iri
                                    ));
                                }
                                // Handle @value literals
                                else if let Some(val_str) =
                                    item_obj.get("@value").and_then(|v| v.as_str())
                                {
                                    let mut literal =
                                        format!("\"{}\"", escape_turtle_literal(val_str));

                                    // Add language tag if present
                                    if let Some(lang) =
                                        item_obj.get("@language").and_then(|v| v.as_str())
                                    {
                                        literal.push_str(&format!("@{}", lang));
                                    }

                                    statements.push(format!(
                                        "<{}> <{}> {} .",
                                        subject_iri, predicate_iri, literal
                                    ));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(statements.join("\n"))
    }

    /// Export the schema as JSON-LD format
    ///
    /// This generates a JSON-LD document with:
    /// - @context for namespace prefixes
    /// - @graph containing all classes and properties
    /// - Compacted IRIs where possible
    ///
    /// ## Example
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::SchemaAnalyzer;
    ///
    /// let mut analyzer = SchemaAnalyzer::new();
    /// analyzer.load_turtle(r#"
    ///     @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ///     @prefix ex: <http://example.org/> .
    ///
    ///     ex:Person a rdfs:Class ;
    ///         rdfs:label "Person" .
    /// "#).unwrap();
    /// analyzer.analyze().unwrap();
    ///
    /// let jsonld = analyzer.to_jsonld().unwrap();
    /// assert!(jsonld.contains("@context"));
    /// assert!(jsonld.contains("@graph"));
    /// ```
    pub fn to_jsonld(&self) -> Result<String> {
        self.to_jsonld_with_context(JsonLdContext::new())
    }

    /// Export the schema as JSON-LD with a custom context
    ///
    /// This allows you to provide custom prefix mappings.
    ///
    /// ## Example
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::{SchemaAnalyzer, JsonLdContext};
    ///
    /// let mut analyzer = SchemaAnalyzer::new();
    /// analyzer.load_turtle(r#"
    ///     @prefix ex: <http://example.org/> .
    ///     ex:Thing a <http://www.w3.org/2000/01/rdf-schema#Class> .
    /// "#).unwrap();
    /// analyzer.analyze().unwrap();
    ///
    /// let mut context = JsonLdContext::new();
    /// context.add_prefix("ex".to_string(), "http://example.org/".to_string());
    ///
    /// let jsonld = analyzer.to_jsonld_with_context(context).unwrap();
    /// assert!(jsonld.contains("\"ex\": \"http://example.org/\""));
    /// ```
    pub fn to_jsonld_with_context(&self, mut context: JsonLdContext) -> Result<String> {
        // Auto-detect namespaces from IRIs
        let namespaces = self.detect_namespaces();
        for (prefix, namespace) in namespaces {
            // Only add if namespace is not already mapped
            let already_mapped = context.prefixes().values().any(|ns| ns == &namespace);
            if !already_mapped && !context.prefixes().contains_key(&prefix) {
                context.add_prefix(prefix, namespace);
            }
        }

        let mut graph = Vec::new();

        // Export classes
        for (iri, class_info) in &self.classes {
            graph.push(self.class_to_jsonld(iri, class_info, &context));
        }

        // Export properties
        for (iri, prop_info) in &self.properties {
            graph.push(self.property_to_jsonld(iri, prop_info, &context));
        }

        // Build the complete JSON-LD document
        let document = json!({
            "@context": context.to_json(),
            "@graph": graph
        });

        // Pretty-print JSON
        serde_json::to_string_pretty(&document)
            .map_err(|e| anyhow::anyhow!("Failed to serialize JSON-LD: {}", e))
    }

    /// Convert a class to JSON-LD object
    fn class_to_jsonld(&self, iri: &str, class_info: &ClassInfo, context: &JsonLdContext) -> Value {
        let mut obj = Map::new();

        obj.insert("@id".to_string(), Value::String(context.compact_iri(iri)));
        obj.insert(
            "@type".to_string(),
            Value::String(context.compact_iri("http://www.w3.org/2000/01/rdf-schema#Class")),
        );

        if let Some(label) = &class_info.label {
            obj.insert(
                context.compact_iri("http://www.w3.org/2000/01/rdf-schema#label"),
                Value::String(label.clone()),
            );
        }

        if let Some(comment) = &class_info.comment {
            obj.insert(
                context.compact_iri("http://www.w3.org/2000/01/rdf-schema#comment"),
                Value::String(comment.clone()),
            );
        }

        if !class_info.subclass_of.is_empty() {
            let subclass_values: Vec<Value> = class_info
                .subclass_of
                .iter()
                .map(|parent| json!({"@id": context.compact_iri(parent)}))
                .collect();

            obj.insert(
                context.compact_iri("http://www.w3.org/2000/01/rdf-schema#subClassOf"),
                if subclass_values.len() == 1 {
                    subclass_values[0].clone()
                } else {
                    Value::Array(subclass_values)
                },
            );
        }

        Value::Object(obj)
    }

    /// Convert a property to JSON-LD object
    fn property_to_jsonld(
        &self,
        iri: &str,
        prop_info: &PropertyInfo,
        context: &JsonLdContext,
    ) -> Value {
        let mut obj = Map::new();

        obj.insert("@id".to_string(), Value::String(context.compact_iri(iri)));
        obj.insert(
            "@type".to_string(),
            Value::String(
                context.compact_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"),
            ),
        );

        if let Some(label) = &prop_info.label {
            obj.insert(
                context.compact_iri("http://www.w3.org/2000/01/rdf-schema#label"),
                Value::String(label.clone()),
            );
        }

        if let Some(comment) = &prop_info.comment {
            obj.insert(
                context.compact_iri("http://www.w3.org/2000/01/rdf-schema#comment"),
                Value::String(comment.clone()),
            );
        }

        if !prop_info.domain.is_empty() {
            let domain_values: Vec<Value> = prop_info
                .domain
                .iter()
                .map(|d| json!({"@id": context.compact_iri(d)}))
                .collect();

            obj.insert(
                context.compact_iri("http://www.w3.org/2000/01/rdf-schema#domain"),
                if domain_values.len() == 1 {
                    domain_values[0].clone()
                } else {
                    Value::Array(domain_values)
                },
            );
        }

        if !prop_info.range.is_empty() {
            let range_values: Vec<Value> = prop_info
                .range
                .iter()
                .map(|r| json!({"@id": context.compact_iri(r)}))
                .collect();

            obj.insert(
                context.compact_iri("http://www.w3.org/2000/01/rdf-schema#range"),
                if range_values.len() == 1 {
                    range_values[0].clone()
                } else {
                    Value::Array(range_values)
                },
            );
        }

        Value::Object(obj)
    }

    /// Detect namespaces used in the schema for auto-prefix generation
    fn detect_namespaces(&self) -> HashMap<String, String> {
        let mut namespaces = HashMap::new();

        // Extract namespaces from class IRIs
        for iri in self.classes.keys() {
            if let Some((namespace, _)) = Self::split_iri(iri) {
                let prefix = Self::guess_prefix(&namespace);
                namespaces.insert(prefix, namespace);
            }
        }

        // Extract namespaces from property IRIs
        for iri in self.properties.keys() {
            if let Some((namespace, _)) = Self::split_iri(iri) {
                let prefix = Self::guess_prefix(&namespace);
                namespaces.insert(prefix, namespace);
            }
        }

        namespaces
    }

    /// Split an IRI into (namespace, local_name)
    fn split_iri(iri: &str) -> Option<(String, String)> {
        iri.rfind('#')
            .map(|hash_pos| {
                (
                    iri[..=hash_pos].to_string(),
                    iri[hash_pos + 1..].to_string(),
                )
            })
            .or_else(|| {
                iri.rfind('/').map(|slash_pos| {
                    (
                        iri[..=slash_pos].to_string(),
                        iri[slash_pos + 1..].to_string(),
                    )
                })
            })
    }

    /// Guess a prefix from a namespace IRI
    fn guess_prefix(namespace: &str) -> String {
        // Try to extract a meaningful prefix from the namespace
        if let Some(domain_start) = namespace.find("://") {
            let after_protocol = &namespace[domain_start + 3..];

            // Extract domain segments
            if let Some(domain_end) = after_protocol.find('/') {
                let domain = &after_protocol[..domain_end];
                let segments: Vec<&str> = domain.split('.').collect();

                // Use the domain name before TLD as prefix
                if segments.len() >= 2 {
                    return segments[segments.len() - 2].to_string();
                }
            }
        }

        // Fallback: use "ns" prefix
        "ns".to_string()
    }
}

/// Escape special characters in Turtle literals
fn escape_turtle_literal(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonld_context_creation() {
        let context = JsonLdContext::new();
        assert!(context.prefixes().contains_key("rdf"));
        assert!(context.prefixes().contains_key("rdfs"));
        assert!(context.prefixes().contains_key("owl"));
    }

    #[test]
    fn test_jsonld_context_add_prefix() {
        let mut context = JsonLdContext::new();
        context.add_prefix("ex".to_string(), "http://example.org/".to_string());
        assert_eq!(
            context.prefixes().get("ex"),
            Some(&"http://example.org/".to_string())
        );
    }

    #[test]
    fn test_jsonld_context_compact_iri() {
        let mut context = JsonLdContext::new();
        context.add_prefix("ex".to_string(), "http://example.org/".to_string());

        assert_eq!(
            context.compact_iri("http://example.org/Person"),
            "ex:Person"
        );
        assert_eq!(
            context.compact_iri("http://www.w3.org/2000/01/rdf-schema#Class"),
            "rdfs:Class"
        );
        assert_eq!(
            context.compact_iri("http://unknown.org/Thing"),
            "http://unknown.org/Thing"
        );
    }

    #[test]
    fn test_to_jsonld_basic() {
        let mut analyzer = SchemaAnalyzer::new();
        analyzer
            .load_turtle(
                r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Person a rdfs:Class ;
                rdfs:label "Person" ;
                rdfs:comment "A human being" .
        "#,
            )
            .unwrap();
        analyzer.analyze().unwrap();

        let jsonld = analyzer.to_jsonld().unwrap();

        // Verify JSON-LD structure
        assert!(jsonld.contains("@context"));
        assert!(jsonld.contains("@graph"));
        assert!(jsonld.contains("Person"));
        assert!(jsonld.contains("A human being"));
    }

    #[test]
    fn test_to_jsonld_with_hierarchy() {
        let mut analyzer = SchemaAnalyzer::new();
        analyzer
            .load_turtle(
                r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Animal a rdfs:Class .
            ex:Dog a rdfs:Class ;
                rdfs:subClassOf ex:Animal .
        "#,
            )
            .unwrap();
        analyzer.analyze().unwrap();

        let jsonld = analyzer.to_jsonld().unwrap();

        assert!(jsonld.contains("Animal"));
        assert!(jsonld.contains("Dog"));
        assert!(jsonld.contains("subClassOf"));
    }

    #[test]
    fn test_to_jsonld_with_properties() {
        let mut analyzer = SchemaAnalyzer::new();
        analyzer
            .load_turtle(
                r#"
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:knows a rdf:Property ;
                rdfs:label "knows" ;
                rdfs:domain ex:Person ;
                rdfs:range ex:Person .
        "#,
            )
            .unwrap();
        analyzer.analyze().unwrap();

        let jsonld = analyzer.to_jsonld().unwrap();

        assert!(jsonld.contains("knows"));
        assert!(jsonld.contains("domain"));
        assert!(jsonld.contains("range"));
    }

    #[test]
    fn test_to_jsonld_with_custom_context() {
        let mut analyzer = SchemaAnalyzer::new();
        analyzer
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Thing a <http://www.w3.org/2000/01/rdf-schema#Class> .
        "#,
            )
            .unwrap();
        analyzer.analyze().unwrap();

        let mut context = JsonLdContext::new();
        context.add_prefix("ex".to_string(), "http://example.org/".to_string());

        let jsonld = analyzer.to_jsonld_with_context(context).unwrap();

        assert!(jsonld.contains("\"ex\": \"http://example.org/\""));
        assert!(jsonld.contains("ex:Thing"));
    }

    #[test]
    fn test_jsonld_roundtrip_structure() {
        let mut analyzer = SchemaAnalyzer::new();
        analyzer
            .load_turtle(
                r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Person a rdfs:Class ;
                rdfs:label "Person" .
        "#,
            )
            .unwrap();
        analyzer.analyze().unwrap();

        let jsonld_str = analyzer.to_jsonld().unwrap();

        // Parse back as JSON to verify structure
        let parsed: Value = serde_json::from_str(&jsonld_str).unwrap();

        assert!(parsed.get("@context").is_some());
        assert!(parsed.get("@graph").is_some());

        let graph = parsed.get("@graph").unwrap().as_array().unwrap();
        assert!(!graph.is_empty());

        let first_item = &graph[0];
        assert!(first_item.get("@id").is_some());
        assert!(first_item.get("@type").is_some());
    }

    #[test]
    fn test_split_iri() {
        assert_eq!(
            SchemaAnalyzer::split_iri("http://example.org/ns#Person"),
            Some(("http://example.org/ns#".to_string(), "Person".to_string()))
        );

        assert_eq!(
            SchemaAnalyzer::split_iri("http://example.org/ns/Person"),
            Some(("http://example.org/ns/".to_string(), "Person".to_string()))
        );

        assert_eq!(SchemaAnalyzer::split_iri("simple"), None);
    }

    #[test]
    fn test_guess_prefix() {
        assert_eq!(
            SchemaAnalyzer::guess_prefix("http://example.org/"),
            "example"
        );

        assert_eq!(
            SchemaAnalyzer::guess_prefix("http://xmlns.com/foaf/0.1/"),
            "xmlns"
        );

        assert_eq!(
            SchemaAnalyzer::guess_prefix("http://www.w3.org/2000/01/rdf-schema#"),
            "w3"
        );
    }

    #[test]
    fn test_detect_namespaces() {
        let mut analyzer = SchemaAnalyzer::new();
        analyzer
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            @prefix foaf: <http://xmlns.com/foaf/0.1/> .

            ex:Person a <http://www.w3.org/2000/01/rdf-schema#Class> .
            foaf:knows a <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> .
        "#,
            )
            .unwrap();
        analyzer.analyze().unwrap();

        let namespaces = analyzer.detect_namespaces();

        assert!(namespaces.contains_key("example"));
        assert!(namespaces.contains_key("xmlns"));
    }

    #[test]
    fn test_jsonld_context_from_json() {
        let context_json = json!({
            "ex": "http://example.org/",
            "foaf": "http://xmlns.com/foaf/0.1/"
        });

        let context = JsonLdContext::from_json(&context_json).unwrap();
        assert_eq!(
            context.prefixes().get("ex"),
            Some(&"http://example.org/".to_string())
        );
        assert_eq!(
            context.prefixes().get("foaf"),
            Some(&"http://xmlns.com/foaf/0.1/".to_string())
        );
    }

    #[test]
    fn test_jsonld_context_expand_iri() {
        let mut context = JsonLdContext::new();
        context.add_prefix("ex".to_string(), "http://example.org/".to_string());

        assert_eq!(context.expand_iri("ex:Person"), "http://example.org/Person");
        assert_eq!(
            context.expand_iri("rdfs:Class"),
            "http://www.w3.org/2000/01/rdf-schema#Class"
        );
        assert_eq!(
            context.expand_iri("http://full.iri/Thing"),
            "http://full.iri/Thing"
        );
    }

    #[test]
    fn test_load_jsonld_basic() {
        let jsonld = r#"{
            "@context": {
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "ex": "http://example.org/"
            },
            "@graph": [
                {
                    "@id": "ex:Person",
                    "@type": "rdfs:Class",
                    "rdfs:label": "Person",
                    "rdfs:comment": "A human being"
                }
            ]
        }"#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_jsonld(jsonld).unwrap();
        analyzer.analyze().unwrap();

        assert_eq!(analyzer.classes.len(), 1);
        let person = analyzer.classes.get("http://example.org/Person").unwrap();
        assert_eq!(person.label, Some("Person".to_string()));
        assert_eq!(person.comment, Some("A human being".to_string()));
    }

    #[test]
    fn test_load_jsonld_with_properties() {
        let jsonld = r#"{
            "@context": {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "ex": "http://example.org/"
            },
            "@graph": [
                {
                    "@id": "ex:knows",
                    "@type": "rdf:Property",
                    "rdfs:label": "knows",
                    "rdfs:domain": {"@id": "ex:Person"},
                    "rdfs:range": {"@id": "ex:Person"}
                }
            ]
        }"#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_jsonld(jsonld).unwrap();
        analyzer.analyze().unwrap();

        assert_eq!(analyzer.properties.len(), 1);
        let knows = analyzer.properties.get("http://example.org/knows").unwrap();
        assert_eq!(knows.label, Some("knows".to_string()));
        assert!(knows
            .domain
            .contains(&"http://example.org/Person".to_string()));
        assert!(knows
            .range
            .contains(&"http://example.org/Person".to_string()));
    }

    #[test]
    fn test_jsonld_roundtrip() {
        // Create original schema
        let mut analyzer1 = SchemaAnalyzer::new();
        analyzer1
            .load_turtle(
                r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Person a rdfs:Class ;
                      rdfs:label "Person" ;
                      rdfs:comment "A human being" .

            ex:Organization a rdfs:Class ;
                            rdfs:label "Organization" .
        "#,
            )
            .unwrap();
        analyzer1.analyze().unwrap();

        // Export to JSON-LD
        let jsonld = analyzer1.to_jsonld().unwrap();

        // Import back
        let mut analyzer2 = SchemaAnalyzer::new();
        analyzer2.load_jsonld(&jsonld).unwrap();
        analyzer2.analyze().unwrap();

        // Verify roundtrip
        assert_eq!(analyzer1.classes.len(), analyzer2.classes.len());
        assert!(analyzer2.classes.contains_key("http://example.org/Person"));
        assert!(analyzer2
            .classes
            .contains_key("http://example.org/Organization"));

        let person2 = analyzer2.classes.get("http://example.org/Person").unwrap();
        assert_eq!(person2.label, Some("Person".to_string()));
        assert_eq!(person2.comment, Some("A human being".to_string()));
    }

    #[test]
    fn test_load_jsonld_with_language_tags() {
        let jsonld = r#"{
            "@context": {
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "ex": "http://example.org/"
            },
            "@graph": [
                {
                    "@id": "ex:Person",
                    "@type": "rdfs:Class",
                    "rdfs:label": [
                        {"@value": "Person", "@language": "en"},
                        {"@value": "Personne", "@language": "fr"}
                    ]
                }
            ]
        }"#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_jsonld(jsonld).unwrap();
        analyzer.analyze().unwrap();

        assert_eq!(analyzer.classes.len(), 1);
        // Note: The analyzer will pick one of the labels
        let person = analyzer.classes.get("http://example.org/Person").unwrap();
        assert!(person.label.is_some());
    }

    #[test]
    fn test_escape_turtle_literal() {
        assert_eq!(escape_turtle_literal("hello"), "hello");
        assert_eq!(escape_turtle_literal("hello\"world"), "hello\\\"world");
        assert_eq!(escape_turtle_literal("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_turtle_literal("tab\there"), "tab\\there");
        assert_eq!(escape_turtle_literal("back\\slash"), "back\\\\slash");
    }
}
