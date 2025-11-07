//! Unit tests for the OxiRS bridge.

use crate::{compilation::compile_rules, provenance::ProvenanceTracker, schema::SchemaAnalyzer};

#[test]
fn test_provenance_tracker() {
    let mut tracker = ProvenanceTracker::new();

    tracker.track_entity("http://example.org/Person".to_string(), 0);
    tracker.track_entity("http://example.org/knows".to_string(), 1);

    assert_eq!(tracker.get_tensor("http://example.org/Person"), Some(0));
    assert_eq!(tracker.get_entity(0), Some("http://example.org/Person"));

    let rdf_star = tracker.to_rdf_star();
    assert!(!rdf_star.is_empty());
}

#[test]
fn test_provenance_json() {
    let mut tracker = ProvenanceTracker::new();
    tracker.track_entity("http://example.org/Alice".to_string(), 42);

    let json = tracker.to_json().unwrap();
    let restored = ProvenanceTracker::from_json(&json).unwrap();

    assert_eq!(restored.get_tensor("http://example.org/Alice"), Some(42));
}

#[test]
fn test_iri_to_name() {
    assert_eq!(
        SchemaAnalyzer::iri_to_name("http://example.org/Person"),
        "Person"
    );
    assert_eq!(
        SchemaAnalyzer::iri_to_name("http://xmlns.com/foaf/0.1#knows"),
        "knows"
    );
    assert_eq!(SchemaAnalyzer::iri_to_name("Person"), "Person");
}

#[test]
fn test_empty_schema_analyzer() {
    let analyzer = SchemaAnalyzer::new();

    assert_eq!(analyzer.classes.len(), 0);
    assert_eq!(analyzer.properties.len(), 0);
}

#[test]
fn test_symbol_table_from_empty_schema() {
    let analyzer = SchemaAnalyzer::new();

    let table = analyzer.to_symbol_table().unwrap();
    // Now includes standard RDF/RDFS types: Literal, Resource, Entity
    assert_eq!(table.domains.len(), 3);
    assert_eq!(table.predicates.len(), 0);
    assert!(table.domains.contains_key("Literal"));
    assert!(table.domains.contains_key("Resource"));
    assert!(table.domains.contains_key("Entity"));
}

#[test]
fn test_schema_analyzer_with_simple_rdf() {
    let mut analyzer = SchemaAnalyzer::new();

    // Simple RDF schema
    let turtle_data = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .

        ex:Person rdf:type rdfs:Class ;
                 rdfs:label "Person" ;
                 rdfs:comment "A human being" .

        ex:knows rdf:type rdf:Property ;
                rdfs:label "knows" ;
                rdfs:domain ex:Person ;
                rdfs:range ex:Person .
    "#;

    analyzer.load_turtle(turtle_data).unwrap();
    analyzer.analyze().unwrap();

    assert_eq!(analyzer.classes.len(), 1);
    assert!(analyzer.classes.contains_key("http://example.org/Person"));

    assert_eq!(analyzer.properties.len(), 1);
    assert!(analyzer.properties.contains_key("http://example.org/knows"));
}

#[test]
fn test_compile_empty_rules() {
    let result = compile_rules(&[]);
    assert!(result.is_err());
}
