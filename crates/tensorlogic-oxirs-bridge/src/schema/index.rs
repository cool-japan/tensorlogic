//! RDF graph indexing for fast triple lookups.
//!
//! Provides Subject-Predicate-Object (SPO) indexes and related query capabilities
//! for efficient RDF graph operations.

use oxrdf::{Graph, TermRef};
use std::collections::{HashMap, HashSet};

/// Statistics about the index
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub total_triples: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub unique_objects: usize,
    pub subject_index_size: usize,
    pub predicate_index_size: usize,
    pub object_index_size: usize,
}

/// Triple index for fast lookups by subject, predicate, or object.
///
/// Maintains three separate indexes:
/// - Subject → [(Predicate, Object)]
/// - Predicate → [(Subject, Object)]
/// - Object → [(Subject, Predicate)]
///
/// This enables O(1) lookups for common query patterns.
#[derive(Debug, Clone)]
pub struct TripleIndex {
    /// Subject → Vec<(Predicate, Object)>
    subject_index: HashMap<String, Vec<(String, String)>>,

    /// Predicate → Vec<(Subject, Object)>
    predicate_index: HashMap<String, Vec<(String, String)>>,

    /// Object → Vec<(Subject, Predicate)>
    object_index: HashMap<String, Vec<(String, String)>>,

    /// Prefix → `Vec<IRI>` for fast prefix-based searches
    prefix_index: HashMap<String, Vec<String>>,

    /// Statistics
    stats: IndexStats,
}

impl TripleIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self {
            subject_index: HashMap::new(),
            predicate_index: HashMap::new(),
            object_index: HashMap::new(),
            prefix_index: HashMap::new(),
            stats: IndexStats::default(),
        }
    }

    /// Build indexes from an RDF graph
    pub fn from_graph(graph: &Graph) -> Self {
        let mut index = Self::new();
        index.build_from_graph(graph);
        index
    }

    /// Build all indexes from the graph
    pub fn build_from_graph(&mut self, graph: &Graph) {
        // Clear existing indexes
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
        self.prefix_index.clear();

        let mut unique_subjects = HashSet::new();
        let mut unique_predicates = HashSet::new();
        let mut unique_objects = HashSet::new();

        // Index all triples
        for triple in graph.iter() {
            let subject = self.term_to_string(triple.subject.into());
            let predicate = triple.predicate.as_str().to_string();
            let object = self.term_to_string(triple.object);

            // Track unique values
            unique_subjects.insert(subject.clone());
            unique_predicates.insert(predicate.clone());
            unique_objects.insert(object.clone());

            // Index by subject
            self.subject_index
                .entry(subject.clone())
                .or_default()
                .push((predicate.clone(), object.clone()));

            // Index by predicate
            self.predicate_index
                .entry(predicate.clone())
                .or_default()
                .push((subject.clone(), object.clone()));

            // Index by object
            self.object_index
                .entry(object.clone())
                .or_default()
                .push((subject.clone(), predicate.clone()));

            // Build prefix index for subject
            if let Some(prefix) = self.extract_prefix(&subject) {
                self.prefix_index
                    .entry(prefix)
                    .or_default()
                    .push(subject.clone());
            }
        }

        // Update statistics
        self.stats = IndexStats {
            total_triples: graph.len(),
            unique_subjects: unique_subjects.len(),
            unique_predicates: unique_predicates.len(),
            unique_objects: unique_objects.len(),
            subject_index_size: self.subject_index.len(),
            predicate_index_size: self.predicate_index.len(),
            object_index_size: self.object_index.len(),
        };
    }

    /// Convert a TermRef to a string representation
    fn term_to_string(&self, term: TermRef) -> String {
        match term {
            TermRef::NamedNode(node) => node.as_str().to_string(),
            TermRef::BlankNode(node) => format!("_:{}", node.as_str()),
            TermRef::Literal(lit) => lit.value().to_string(),
        }
    }

    /// Extract namespace prefix from IRI
    fn extract_prefix(&self, iri: &str) -> Option<String> {
        iri.rfind('#')
            .or_else(|| iri.rfind('/'))
            .map(|pos| iri[..pos + 1].to_string())
    }

    /// Find all triples with the given subject
    pub fn find_by_subject(&self, subject: &str) -> Vec<(String, String, String)> {
        self.subject_index
            .get(subject)
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|(p, o)| (subject.to_string(), p.clone(), o.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find all triples with the given predicate
    pub fn find_by_predicate(&self, predicate: &str) -> Vec<(String, String, String)> {
        self.predicate_index
            .get(predicate)
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|(s, o)| (s.clone(), predicate.to_string(), o.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find all triples with the given object
    pub fn find_by_object(&self, object: &str) -> Vec<(String, String, String)> {
        self.object_index
            .get(object)
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|(s, p)| (s.clone(), p.clone(), object.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find all triples matching the pattern (None = wildcard)
    pub fn find_by_pattern(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<(String, String, String)> {
        match (subject, predicate, object) {
            (Some(s), None, None) => self.find_by_subject(s),
            (None, Some(p), None) => self.find_by_predicate(p),
            (None, None, Some(o)) => self.find_by_object(o),
            (Some(s), Some(p), None) => self
                .find_by_subject(s)
                .into_iter()
                .filter(|(_, pred, _)| pred == p)
                .collect(),
            (Some(s), None, Some(o)) => self
                .find_by_subject(s)
                .into_iter()
                .filter(|(_, _, obj)| obj == o)
                .collect(),
            (None, Some(p), Some(o)) => self
                .find_by_predicate(p)
                .into_iter()
                .filter(|(_, _, obj)| obj == o)
                .collect(),
            (Some(s), Some(p), Some(o)) => {
                // Exact match - check if it exists
                if self
                    .find_by_subject(s)
                    .iter()
                    .any(|(_, pred, obj)| pred == p && obj == o)
                {
                    vec![(s.to_string(), p.to_string(), o.to_string())]
                } else {
                    vec![]
                }
            }
            (None, None, None) => {
                // Return all triples
                self.subject_index
                    .iter()
                    .flat_map(|(s, pairs)| {
                        pairs
                            .iter()
                            .map(move |(p, o)| (s.clone(), p.clone(), o.clone()))
                    })
                    .collect()
            }
        }
    }

    /// Find all IRIs with the given prefix
    pub fn find_by_prefix(&self, prefix: &str) -> Vec<String> {
        self.prefix_index.get(prefix).cloned().unwrap_or_default()
    }

    /// Get all unique subjects
    pub fn get_all_subjects(&self) -> Vec<String> {
        self.subject_index.keys().cloned().collect()
    }

    /// Get all unique predicates
    pub fn get_all_predicates(&self) -> Vec<String> {
        self.predicate_index.keys().cloned().collect()
    }

    /// Get all unique objects
    pub fn get_all_objects(&self) -> Vec<String> {
        self.object_index.keys().cloned().collect()
    }

    /// Get all unique prefixes
    pub fn get_all_prefixes(&self) -> Vec<String> {
        self.prefix_index.keys().cloned().collect()
    }

    /// Get index statistics
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Check if the index contains a specific triple
    pub fn contains(&self, subject: &str, predicate: &str, object: &str) -> bool {
        !self
            .find_by_pattern(Some(subject), Some(predicate), Some(object))
            .is_empty()
    }

    /// Count triples matching a pattern
    pub fn count_by_pattern(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> usize {
        self.find_by_pattern(subject, predicate, object).len()
    }

    /// Get the degree (number of outgoing edges) of a subject
    pub fn subject_degree(&self, subject: &str) -> usize {
        self.subject_index
            .get(subject)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get the number of triples using a predicate
    pub fn predicate_frequency(&self, predicate: &str) -> usize {
        self.predicate_index
            .get(predicate)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Clear all indexes
    pub fn clear(&mut self) {
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
        self.prefix_index.clear();
        self.stats = IndexStats::default();
    }
}

impl Default for TripleIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxttl::TurtleParser;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new();

        let turtle = r#"
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Person a rdfs:Class ;
                      rdfs:label "Person" ;
                      rdfs:comment "A human being" .

            ex:Organization a rdfs:Class ;
                           rdfs:label "Organization" .

            ex:name a rdf:Property ;
                    rdfs:domain ex:Person ;
                    rdfs:range rdfs:Literal .

            ex:worksFor a rdf:Property ;
                       rdfs:domain ex:Person ;
                       rdfs:range ex:Organization .
        "#;

        let parser = TurtleParser::new().for_slice(turtle.as_bytes());
        for triple in parser {
            graph.insert(&triple.unwrap());
        }

        graph
    }

    #[test]
    fn test_index_creation() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        assert!(index.stats().total_triples > 0);
        assert!(index.stats().unique_subjects > 0);
        assert!(index.stats().unique_predicates > 0);
        assert!(index.stats().unique_objects > 0);
    }

    #[test]
    fn test_find_by_subject() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        let triples = index.find_by_subject("http://example.org/Person");
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_find_by_predicate() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        let triples = index.find_by_predicate("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        assert!(!triples.is_empty());
    }

    #[test]
    fn test_find_by_object() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        let triples = index.find_by_object("http://www.w3.org/2000/01/rdf-schema#Class");
        assert!(triples.len() >= 2); // Person and Organization
    }

    #[test]
    fn test_find_by_pattern() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        // Subject + Predicate
        let triples = index.find_by_pattern(
            Some("http://example.org/Person"),
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            None,
        );
        assert_eq!(triples.len(), 1);

        // Predicate + Object
        let triples = index.find_by_pattern(
            None,
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            Some("http://www.w3.org/2000/01/rdf-schema#Class"),
        );
        assert!(triples.len() >= 2);
    }

    #[test]
    fn test_contains() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        assert!(index.contains(
            "http://example.org/Person",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://www.w3.org/2000/01/rdf-schema#Class"
        ));

        assert!(!index.contains(
            "http://example.org/NonExistent",
            "http://example.org/unknownPredicate",
            "http://example.org/unknownObject"
        ));
    }

    #[test]
    fn test_prefix_search() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        let iris = index.find_by_prefix("http://example.org/");
        assert!(!iris.is_empty());
        assert!(iris
            .iter()
            .all(|iri| iri.starts_with("http://example.org/")));
    }

    #[test]
    fn test_subject_degree() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        let degree = index.subject_degree("http://example.org/Person");
        assert!(degree > 0);
    }

    #[test]
    fn test_predicate_frequency() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        let freq = index.predicate_frequency("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
        assert!(freq > 0);
    }

    #[test]
    fn test_get_all_subjects() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        let subjects = index.get_all_subjects();
        assert!(!subjects.is_empty());
        assert!(subjects.contains(&"http://example.org/Person".to_string()));
    }

    #[test]
    fn test_get_all_predicates() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        let predicates = index.get_all_predicates();
        assert!(!predicates.is_empty());
        assert!(predicates.contains(&"http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()));
    }

    #[test]
    fn test_count_by_pattern() {
        let graph = create_test_graph();
        let index = TripleIndex::from_graph(&graph);

        let count = index.count_by_pattern(
            None,
            Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            None,
        );
        assert!(count > 0);
    }

    #[test]
    fn test_clear() {
        let graph = create_test_graph();
        let mut index = TripleIndex::from_graph(&graph);

        assert!(index.stats().total_triples > 0);

        index.clear();

        assert_eq!(index.stats().total_triples, 0);
        assert!(index.get_all_subjects().is_empty());
    }
}
