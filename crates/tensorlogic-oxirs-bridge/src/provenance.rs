//! Provenance tracking for mapping RDF entities to tensor computations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::rdfstar::{MetadataBuilder, RdfStarProvenanceStore};

/// Provenance tracker: maps RDF entities to tensor computation graph
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenanceTracker {
    /// RDF entity IRI → Tensor index
    pub entity_to_tensor: HashMap<String, usize>,
    /// Tensor index → RDF entity IRI
    pub tensor_to_entity: HashMap<usize, String>,
    /// SHACL shape IRI → Rule expression
    pub shape_to_rule: HashMap<String, String>,
    /// Tensor node index → SHACL shape IRI
    pub node_to_shape: HashMap<usize, String>,
    /// RDF* provenance store for statement-level metadata
    #[serde(skip)]
    pub rdfstar_store: Option<RdfStarProvenanceStore>,
}

impl ProvenanceTracker {
    pub fn new() -> Self {
        ProvenanceTracker {
            entity_to_tensor: HashMap::new(),
            tensor_to_entity: HashMap::new(),
            shape_to_rule: HashMap::new(),
            node_to_shape: HashMap::new(),
            rdfstar_store: None,
        }
    }

    /// Create a new provenance tracker with RDF* support enabled
    pub fn with_rdfstar() -> Self {
        ProvenanceTracker {
            entity_to_tensor: HashMap::new(),
            tensor_to_entity: HashMap::new(),
            shape_to_rule: HashMap::new(),
            node_to_shape: HashMap::new(),
            rdfstar_store: Some(RdfStarProvenanceStore::new()),
        }
    }

    /// Get or create the RDF* provenance store
    pub fn rdfstar_store_mut(&mut self) -> &mut RdfStarProvenanceStore {
        if self.rdfstar_store.is_none() {
            self.rdfstar_store = Some(RdfStarProvenanceStore::new());
        }
        self.rdfstar_store.as_mut().unwrap()
    }

    /// Get the RDF* provenance store (read-only)
    pub fn rdfstar_store(&self) -> Option<&RdfStarProvenanceStore> {
        self.rdfstar_store.as_ref()
    }

    pub fn track_entity(&mut self, entity_iri: String, tensor_idx: usize) {
        self.entity_to_tensor.insert(entity_iri.clone(), tensor_idx);
        self.tensor_to_entity.insert(tensor_idx, entity_iri);
    }

    pub fn track_shape(&mut self, shape_iri: String, rule_expr: String, node_idx: usize) {
        self.shape_to_rule.insert(shape_iri.clone(), rule_expr);
        self.node_to_shape.insert(node_idx, shape_iri);
    }

    pub fn get_entity(&self, tensor_idx: usize) -> Option<&str> {
        self.tensor_to_entity.get(&tensor_idx).map(|s| s.as_str())
    }

    pub fn get_tensor(&self, entity_iri: &str) -> Option<usize> {
        self.entity_to_tensor.get(entity_iri).copied()
    }

    pub fn to_rdf_star(&self) -> Vec<String> {
        // Generate RDF* statements for provenance
        let mut statements = Vec::new();

        for (tensor_idx, entity_iri) in &self.tensor_to_entity {
            statements.push(format!(
                "<< <{}> <http://tensorlogic.org/tensor> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> >> <http://tensorlogic.org/computedBy> <http://tensorlogic.org/engine> .",
                entity_iri, tensor_idx
            ));
        }

        for (node_idx, shape_iri) in &self.node_to_shape {
            if let Some(rule) = self.shape_to_rule.get(shape_iri) {
                statements.push(format!(
                    "<< <http://tensorlogic.org/node/{}> <http://tensorlogic.org/derivedFrom> <{}> >> <http://tensorlogic.org/rule> \"{}\" .",
                    node_idx, shape_iri, rule.replace('"', "\\\"")
                ));
            }
        }

        statements
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Track a computed triple with confidence score using RDF*
    pub fn track_inferred_triple(
        &mut self,
        subject: String,
        predicate: String,
        object: String,
        rule_id: Option<String>,
        confidence: Option<f64>,
    ) {
        let store = self.rdfstar_store_mut();

        let mut builder = MetadataBuilder::for_triple(subject, predicate, object)
            .generated_by("http://tensorlogic.org/inference".to_string());

        if let Some(rule) = rule_id {
            builder = builder.rule_id(rule);
        }

        if let Some(conf) = confidence {
            builder = builder.confidence(conf);
        }

        // Add timestamp
        let now = chrono::Utc::now().to_rfc3339();
        builder = builder.generated_at(now);

        store.add_metadata(builder.build());
    }

    /// Get all inferred triples with confidence above a threshold
    pub fn get_high_confidence_inferences(
        &self,
        min_confidence: f64,
    ) -> Vec<&crate::rdfstar::StatementMetadata> {
        self.rdfstar_store
            .as_ref()
            .map(|store| store.get_by_min_confidence(min_confidence))
            .unwrap_or_default()
    }

    /// Export combined provenance (legacy + RDF*) to Turtle
    pub fn to_rdfstar_turtle(&self) -> String {
        let mut turtle = String::new();
        turtle.push_str("# Combined Provenance Export\n");
        turtle.push_str("@prefix tl: <http://tensorlogic.org/> .\n");
        turtle.push_str("@prefix prov: <http://www.w3.org/ns/prov#> .\n");
        turtle.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");

        // Add legacy provenance statements
        turtle.push_str("# Legacy Provenance\n");
        for stmt in self.to_rdf_star() {
            turtle.push_str(&stmt);
            turtle.push('\n');
        }

        // Add RDF* provenance if available
        if let Some(store) = &self.rdfstar_store {
            turtle.push_str("\n# RDF* Statement-Level Provenance\n");
            turtle.push_str(&store.to_turtle());
        }

        turtle
    }
}

impl Default for ProvenanceTracker {
    fn default() -> Self {
        Self::new()
    }
}
