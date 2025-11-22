//! Tests with real-world RDF ontologies.
//!
//! These tests verify that the crate works correctly with actual
//! RDF vocabularies used in production.

use tensorlogic_oxirs_bridge::SchemaAnalyzer;

/// Test with FOAF (Friend of a Friend) ontology subset.
///
/// FOAF is one of the most widely used RDF vocabularies for describing
/// people and their social connections.
#[test]
fn test_foaf_ontology() {
    let foaf = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .

        # Core FOAF classes
        foaf:Agent a rdfs:Class ;
            rdfs:label "Agent"@en ;
            rdfs:comment "An agent (eg. person, group, software or physical artifact)."@en .

        foaf:Person a rdfs:Class ;
            rdfs:subClassOf foaf:Agent ;
            rdfs:label "Person"@en ;
            rdfs:comment "A person."@en .

        foaf:Organization a rdfs:Class ;
            rdfs:subClassOf foaf:Agent ;
            rdfs:label "Organization"@en ;
            rdfs:comment "An organization."@en .

        foaf:Group a rdfs:Class ;
            rdfs:subClassOf foaf:Agent ;
            rdfs:label "Group"@en .

        foaf:Document a rdfs:Class ;
            rdfs:label "Document"@en ;
            rdfs:comment "A document."@en .

        foaf:Image a rdfs:Class ;
            rdfs:subClassOf foaf:Document ;
            rdfs:label "Image"@en .

        # FOAF properties
        foaf:name a rdf:Property ;
            rdfs:domain foaf:Agent ;
            rdfs:label "name"@en ;
            rdfs:comment "A name for some thing."@en .

        foaf:knows a rdf:Property ;
            rdfs:domain foaf:Person ;
            rdfs:range foaf:Person ;
            rdfs:label "knows"@en ;
            rdfs:comment "A person known by this person."@en .

        foaf:mbox a rdf:Property ;
            rdfs:domain foaf:Agent ;
            rdfs:label "personal mailbox"@en .

        foaf:homepage a rdf:Property ;
            rdfs:domain foaf:Agent ;
            rdfs:range foaf:Document ;
            rdfs:label "homepage"@en .

        foaf:depiction a rdf:Property ;
            rdfs:domain foaf:Agent ;
            rdfs:range foaf:Image ;
            rdfs:label "depiction"@en .

        foaf:member a rdf:Property ;
            rdfs:domain foaf:Group ;
            rdfs:range foaf:Agent ;
            rdfs:label "member"@en .

        foaf:age a rdf:Property ;
            rdfs:domain foaf:Agent ;
            rdfs:label "age"@en .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(foaf).unwrap();
    analyzer.analyze().unwrap();

    // Verify classes
    assert_eq!(analyzer.classes.len(), 6);
    assert!(analyzer
        .classes
        .contains_key("http://xmlns.com/foaf/0.1/Agent"));
    assert!(analyzer
        .classes
        .contains_key("http://xmlns.com/foaf/0.1/Person"));
    assert!(analyzer
        .classes
        .contains_key("http://xmlns.com/foaf/0.1/Organization"));
    assert!(analyzer
        .classes
        .contains_key("http://xmlns.com/foaf/0.1/Group"));
    assert!(analyzer
        .classes
        .contains_key("http://xmlns.com/foaf/0.1/Document"));
    assert!(analyzer
        .classes
        .contains_key("http://xmlns.com/foaf/0.1/Image"));

    // Verify subclass relationships
    let person = &analyzer.classes["http://xmlns.com/foaf/0.1/Person"];
    assert!(person
        .subclass_of
        .contains(&"http://xmlns.com/foaf/0.1/Agent".to_string()));

    let image = &analyzer.classes["http://xmlns.com/foaf/0.1/Image"];
    assert!(image
        .subclass_of
        .contains(&"http://xmlns.com/foaf/0.1/Document".to_string()));

    // Verify properties
    assert_eq!(analyzer.properties.len(), 7);
    assert!(analyzer
        .properties
        .contains_key("http://xmlns.com/foaf/0.1/name"));
    assert!(analyzer
        .properties
        .contains_key("http://xmlns.com/foaf/0.1/knows"));

    // Verify property domains and ranges
    let knows = &analyzer.properties["http://xmlns.com/foaf/0.1/knows"];
    assert!(knows
        .domain
        .contains(&"http://xmlns.com/foaf/0.1/Person".to_string()));
    assert!(knows
        .range
        .contains(&"http://xmlns.com/foaf/0.1/Person".to_string()));

    // Convert to SymbolTable
    let symbol_table = analyzer.to_symbol_table().unwrap();
    assert!(symbol_table.domains.contains_key("Person"));
    assert!(symbol_table.domains.contains_key("Organization"));
    assert!(symbol_table.predicates.contains_key("knows"));
    assert!(symbol_table.predicates.contains_key("name"));

    // Verify knows predicate structure
    let knows_pred = &symbol_table.predicates["knows"];
    assert_eq!(knows_pred.arg_domains, vec!["Person", "Person"]);
}

/// Test with Dublin Core metadata vocabulary subset.
///
/// Dublin Core is a standard for cross-domain resource description,
/// widely used for library and document metadata.
#[test]
fn test_dublin_core_ontology() {
    let dc = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix dcterms: <http://purl.org/dc/terms/> .
        @prefix dctype: <http://purl.org/dc/dcmitype/> .

        # Dublin Core Types
        dctype:Collection a rdfs:Class ;
            rdfs:label "Collection"@en ;
            rdfs:comment "An aggregation of resources."@en .

        dctype:Dataset a rdfs:Class ;
            rdfs:label "Dataset"@en ;
            rdfs:comment "Data encoded in a defined structure."@en .

        dctype:Event a rdfs:Class ;
            rdfs:label "Event"@en ;
            rdfs:comment "A non-persistent, time-based occurrence."@en .

        dctype:Image a rdfs:Class ;
            rdfs:label "Image"@en ;
            rdfs:comment "A visual representation other than text."@en .

        dctype:Text a rdfs:Class ;
            rdfs:label "Text"@en ;
            rdfs:comment "A resource consisting primarily of words for reading."@en .

        dctype:Software a rdfs:Class ;
            rdfs:label "Software"@en ;
            rdfs:comment "A computer program in source or compiled form."@en .

        # Dublin Core Terms (Properties)
        dcterms:title a rdf:Property ;
            rdfs:label "Title"@en ;
            rdfs:comment "A name given to the resource."@en .

        dcterms:creator a rdf:Property ;
            rdfs:label "Creator"@en ;
            rdfs:comment "An entity primarily responsible for making the resource."@en .

        dcterms:subject a rdf:Property ;
            rdfs:label "Subject"@en ;
            rdfs:comment "The topic of the resource."@en .

        dcterms:description a rdf:Property ;
            rdfs:label "Description"@en ;
            rdfs:comment "An account of the resource."@en .

        dcterms:publisher a rdf:Property ;
            rdfs:label "Publisher"@en ;
            rdfs:comment "An entity responsible for making the resource available."@en .

        dcterms:date a rdf:Property ;
            rdfs:label "Date"@en ;
            rdfs:comment "A point or period of time associated with an event."@en .

        dcterms:format a rdf:Property ;
            rdfs:label "Format"@en ;
            rdfs:comment "The file format, physical medium, or dimensions."@en .

        dcterms:identifier a rdf:Property ;
            rdfs:label "Identifier"@en ;
            rdfs:comment "An unambiguous reference to the resource."@en .

        dcterms:language a rdf:Property ;
            rdfs:label "Language"@en ;
            rdfs:comment "A language of the resource."@en .

        dcterms:rights a rdf:Property ;
            rdfs:label "Rights"@en ;
            rdfs:comment "Information about rights held in and over the resource."@en .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(dc).unwrap();
    analyzer.analyze().unwrap();

    // Verify classes
    assert_eq!(analyzer.classes.len(), 6);
    assert!(analyzer
        .classes
        .contains_key("http://purl.org/dc/dcmitype/Collection"));
    assert!(analyzer
        .classes
        .contains_key("http://purl.org/dc/dcmitype/Dataset"));
    assert!(analyzer
        .classes
        .contains_key("http://purl.org/dc/dcmitype/Text"));
    assert!(analyzer
        .classes
        .contains_key("http://purl.org/dc/dcmitype/Software"));

    // Verify properties
    assert_eq!(analyzer.properties.len(), 10);
    assert!(analyzer
        .properties
        .contains_key("http://purl.org/dc/terms/title"));
    assert!(analyzer
        .properties
        .contains_key("http://purl.org/dc/terms/creator"));
    assert!(analyzer
        .properties
        .contains_key("http://purl.org/dc/terms/identifier"));

    // Verify labels
    let title = &analyzer.properties["http://purl.org/dc/terms/title"];
    assert_eq!(title.label, Some("Title".to_string()));

    // Convert to SymbolTable
    let symbol_table = analyzer.to_symbol_table().unwrap();
    assert!(symbol_table.domains.contains_key("Collection"));
    assert!(symbol_table.domains.contains_key("Dataset"));
    assert!(symbol_table.predicates.contains_key("title"));
    assert!(symbol_table.predicates.contains_key("creator"));
}

/// Test with SKOS (Simple Knowledge Organization System) vocabulary subset.
///
/// SKOS is used for representing thesauri, classification schemes,
/// taxonomies, and other types of controlled vocabulary.
#[test]
fn test_skos_ontology() {
    let skos = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix skos: <http://www.w3.org/2004/02/skos/core#> .

        # SKOS Classes (using rdfs:Class for compatibility)
        skos:Concept a rdfs:Class ;
            rdfs:label "Concept"@en ;
            rdfs:comment "An idea or notion; a unit of thought."@en .

        skos:ConceptScheme a rdfs:Class ;
            rdfs:label "Concept Scheme"@en ;
            rdfs:comment "A set of concepts, optionally including statements about semantic relationships between those concepts."@en .

        skos:Collection a rdfs:Class ;
            rdfs:label "Collection"@en ;
            rdfs:comment "A meaningful collection of concepts."@en .

        # SKOS Properties - Labeling
        skos:prefLabel a rdf:Property ;
            rdfs:domain skos:Concept ;
            rdfs:label "preferred label"@en ;
            rdfs:comment "The preferred lexical label for a resource."@en .

        skos:altLabel a rdf:Property ;
            rdfs:domain skos:Concept ;
            rdfs:label "alternative label"@en ;
            rdfs:comment "An alternative lexical label for a resource."@en .

        skos:hiddenLabel a rdf:Property ;
            rdfs:domain skos:Concept ;
            rdfs:label "hidden label"@en ;
            rdfs:comment "A lexical label for a resource that should be hidden."@en .

        # SKOS Properties - Semantic Relations
        skos:broader a rdf:Property ;
            rdfs:domain skos:Concept ;
            rdfs:range skos:Concept ;
            rdfs:label "has broader"@en ;
            rdfs:comment "Relates a concept to a concept that is more general."@en .

        skos:narrower a rdf:Property ;
            rdfs:domain skos:Concept ;
            rdfs:range skos:Concept ;
            rdfs:label "has narrower"@en ;
            rdfs:comment "Relates a concept to a concept that is more specific."@en .

        skos:related a rdf:Property ;
            rdfs:domain skos:Concept ;
            rdfs:range skos:Concept ;
            rdfs:label "has related"@en ;
            rdfs:comment "Relates a concept to a concept with which there is an associative semantic relationship."@en .

        # SKOS Properties - Documentation
        skos:definition a rdf:Property ;
            rdfs:domain skos:Concept ;
            rdfs:label "definition"@en ;
            rdfs:comment "A statement of the meaning of a concept."@en .

        skos:note a rdf:Property ;
            rdfs:domain skos:Concept ;
            rdfs:label "note"@en ;
            rdfs:comment "A general note."@en .

        # SKOS Properties - Structure
        skos:inScheme a rdf:Property ;
            rdfs:domain skos:Concept ;
            rdfs:range skos:ConceptScheme ;
            rdfs:label "is in scheme"@en .

        skos:hasTopConcept a rdf:Property ;
            rdfs:domain skos:ConceptScheme ;
            rdfs:range skos:Concept ;
            rdfs:label "has top concept"@en .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(skos).unwrap();
    analyzer.analyze().unwrap();

    // Verify classes
    assert_eq!(analyzer.classes.len(), 3);
    assert!(analyzer
        .classes
        .contains_key("http://www.w3.org/2004/02/skos/core#Concept"));
    assert!(analyzer
        .classes
        .contains_key("http://www.w3.org/2004/02/skos/core#ConceptScheme"));
    assert!(analyzer
        .classes
        .contains_key("http://www.w3.org/2004/02/skos/core#Collection"));

    // Verify properties (10 properties defined: prefLabel, altLabel, hiddenLabel, broader, narrower, related, definition, note, inScheme, hasTopConcept)
    assert_eq!(analyzer.properties.len(), 10);
    assert!(analyzer
        .properties
        .contains_key("http://www.w3.org/2004/02/skos/core#prefLabel"));
    assert!(analyzer
        .properties
        .contains_key("http://www.w3.org/2004/02/skos/core#broader"));
    assert!(analyzer
        .properties
        .contains_key("http://www.w3.org/2004/02/skos/core#narrower"));
    assert!(analyzer
        .properties
        .contains_key("http://www.w3.org/2004/02/skos/core#related"));

    // Verify semantic relations have proper domain/range
    let broader = &analyzer.properties["http://www.w3.org/2004/02/skos/core#broader"];
    assert!(broader
        .domain
        .contains(&"http://www.w3.org/2004/02/skos/core#Concept".to_string()));
    assert!(broader
        .range
        .contains(&"http://www.w3.org/2004/02/skos/core#Concept".to_string()));

    // Convert to SymbolTable
    let symbol_table = analyzer.to_symbol_table().unwrap();
    assert!(symbol_table.domains.contains_key("Concept"));
    assert!(symbol_table.domains.contains_key("ConceptScheme"));
    assert!(symbol_table.predicates.contains_key("prefLabel"));
    assert!(symbol_table.predicates.contains_key("broader"));
    assert!(symbol_table.predicates.contains_key("narrower"));

    // Verify broader predicate structure
    let broader_pred = &symbol_table.predicates["broader"];
    assert_eq!(broader_pred.arg_domains, vec!["Concept", "Concept"]);
}

/// Test combining multiple ontologies.
///
/// Real-world applications often combine vocabularies from multiple ontologies.
#[test]
fn test_combined_ontologies() {
    let combined = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        @prefix dcterms: <http://purl.org/dc/terms/> .
        @prefix schema: <http://schema.org/> .

        # FOAF classes
        foaf:Person a rdfs:Class ;
            rdfs:label "Person"@en .

        foaf:Organization a rdfs:Class ;
            rdfs:label "Organization"@en .

        # Schema.org classes
        schema:CreativeWork a rdfs:Class ;
            rdfs:label "Creative Work"@en .

        schema:Book a rdfs:Class ;
            rdfs:subClassOf schema:CreativeWork ;
            rdfs:label "Book"@en .

        # Cross-vocabulary properties
        dcterms:creator a rdf:Property ;
            rdfs:domain schema:CreativeWork ;
            rdfs:range foaf:Person ;
            rdfs:label "creator"@en .

        schema:author a rdf:Property ;
            rdfs:domain schema:Book ;
            rdfs:range foaf:Person ;
            rdfs:label "author"@en .

        schema:publisher a rdf:Property ;
            rdfs:domain schema:Book ;
            rdfs:range foaf:Organization ;
            rdfs:label "publisher"@en .

        foaf:knows a rdf:Property ;
            rdfs:domain foaf:Person ;
            rdfs:range foaf:Person ;
            rdfs:label "knows"@en .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(combined).unwrap();
    analyzer.analyze().unwrap();

    // Verify classes from different namespaces
    assert_eq!(analyzer.classes.len(), 4);
    assert!(analyzer
        .classes
        .contains_key("http://xmlns.com/foaf/0.1/Person"));
    assert!(analyzer
        .classes
        .contains_key("http://xmlns.com/foaf/0.1/Organization"));
    assert!(analyzer
        .classes
        .contains_key("http://schema.org/CreativeWork"));
    assert!(analyzer.classes.contains_key("http://schema.org/Book"));

    // Verify cross-vocabulary properties
    assert_eq!(analyzer.properties.len(), 4);

    let creator = &analyzer.properties["http://purl.org/dc/terms/creator"];
    assert!(creator
        .domain
        .contains(&"http://schema.org/CreativeWork".to_string()));
    assert!(creator
        .range
        .contains(&"http://xmlns.com/foaf/0.1/Person".to_string()));

    // Convert to SymbolTable
    let symbol_table = analyzer.to_symbol_table().unwrap();

    // All domain names should be extracted correctly
    assert!(symbol_table.domains.contains_key("Person"));
    assert!(symbol_table.domains.contains_key("Organization"));
    assert!(symbol_table.domains.contains_key("CreativeWork"));
    assert!(symbol_table.domains.contains_key("Book"));

    // Predicates should work across vocabularies
    let creator_pred = &symbol_table.predicates["creator"];
    assert_eq!(creator_pred.arg_domains, vec!["CreativeWork", "Person"]);
}

/// Test with Schema.org vocabulary subset.
///
/// Schema.org is used for structured data on web pages,
/// widely adopted by search engines.
#[test]
fn test_schema_org_ontology() {
    let schema = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix schema: <http://schema.org/> .

        # Schema.org Classes
        schema:Thing a rdfs:Class ;
            rdfs:label "Thing"@en ;
            rdfs:comment "The most generic type of item."@en .

        schema:Person a rdfs:Class ;
            rdfs:subClassOf schema:Thing ;
            rdfs:label "Person"@en .

        schema:Place a rdfs:Class ;
            rdfs:subClassOf schema:Thing ;
            rdfs:label "Place"@en .

        schema:Organization a rdfs:Class ;
            rdfs:subClassOf schema:Thing ;
            rdfs:label "Organization"@en .

        schema:Event a rdfs:Class ;
            rdfs:subClassOf schema:Thing ;
            rdfs:label "Event"@en .

        schema:Product a rdfs:Class ;
            rdfs:subClassOf schema:Thing ;
            rdfs:label "Product"@en .

        # Schema.org Properties
        schema:name a rdf:Property ;
            rdfs:domain schema:Thing ;
            rdfs:label "name"@en .

        schema:description a rdf:Property ;
            rdfs:domain schema:Thing ;
            rdfs:label "description"@en .

        schema:url a rdf:Property ;
            rdfs:domain schema:Thing ;
            rdfs:label "url"@en .

        schema:location a rdf:Property ;
            rdfs:domain schema:Event ;
            rdfs:range schema:Place ;
            rdfs:label "location"@en .

        schema:organizer a rdf:Property ;
            rdfs:domain schema:Event ;
            rdfs:range schema:Organization ;
            rdfs:label "organizer"@en .

        schema:worksFor a rdf:Property ;
            rdfs:domain schema:Person ;
            rdfs:range schema:Organization ;
            rdfs:label "works for"@en .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(schema).unwrap();
    analyzer.analyze().unwrap();

    // Verify class hierarchy
    let person = &analyzer.classes["http://schema.org/Person"];
    assert!(person
        .subclass_of
        .contains(&"http://schema.org/Thing".to_string()));

    let event = &analyzer.classes["http://schema.org/Event"];
    assert!(event
        .subclass_of
        .contains(&"http://schema.org/Thing".to_string()));

    // Convert to SymbolTable and verify
    let symbol_table = analyzer.to_symbol_table().unwrap();

    let works_for = &symbol_table.predicates["worksFor"];
    assert_eq!(works_for.arg_domains, vec!["Person", "Organization"]);

    let location = &symbol_table.predicates["location"];
    assert_eq!(location.arg_domains, vec!["Event", "Place"]);
}
