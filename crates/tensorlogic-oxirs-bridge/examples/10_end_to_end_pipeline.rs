//! End-to-End Pipeline Example: RDF → TLExpr → Validation → Export
//!
//! This comprehensive example demonstrates the full integration capabilities of
//! tensorlogic-oxirs-bridge, showing how to:
//! 1. Load and analyze RDF schemas
//! 2. Apply RDFS inference
//! 3. Parse SHACL constraints
//! 4. Compile SPARQL queries to TensorLogic
//! 5. Track provenance with RDF*
//! 6. Validate data and generate reports
//! 7. Export results in multiple formats

use anyhow::Result;
use tensorlogic_oxirs_bridge::{
    schema::cache::SchemaCache,
    shacl::validation::{ShaclValidator, ValidationSeverity},
    GraphQLConverter, MetadataBuilder, ProvenanceTracker, RdfStarProvenanceStore, SchemaAnalyzer,
    ShaclConverter, SparqlCompiler,
};

fn main() -> Result<()> {
    println!("=== TensorLogic OxiRS Bridge: End-to-End Pipeline ===\n");

    // Phase 1: Schema Loading and Analysis
    phase1_schema_loading()?;

    // Phase 2: RDFS Inference
    phase2_rdfs_inference()?;

    // Phase 3: SHACL Constraint Compilation
    phase3_shacl_constraints()?;

    // Phase 4: SPARQL Query Compilation
    phase4_sparql_queries()?;

    // Phase 5: Provenance Tracking
    phase5_provenance_tracking()?;

    // Phase 6: Validation Pipeline
    phase6_validation_pipeline()?;

    // Phase 7: Caching and Performance
    phase7_caching_performance()?;

    // Phase 8: GraphQL Integration
    phase8_graphql_integration()?;

    println!("\n=== Pipeline Complete ===");
    println!("All phases executed successfully!");

    Ok(())
}

/// Phase 1: Load and analyze RDF schemas
fn phase1_schema_loading() -> Result<()> {
    println!("--- Phase 1: Schema Loading and Analysis ---");

    let turtle = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix ex: <http://example.org/> .

        # Class hierarchy
        ex:Entity a rdfs:Class ;
            rdfs:label "Entity"@en, "Entite"@fr ;
            rdfs:comment "Base class for all entities" .

        ex:Person a rdfs:Class ;
            rdfs:subClassOf ex:Entity ;
            rdfs:label "Person"@en, "Personne"@fr .

        ex:Organization a rdfs:Class ;
            rdfs:subClassOf ex:Entity ;
            rdfs:label "Organization"@en .

        ex:Employee a rdfs:Class ;
            rdfs:subClassOf ex:Person ;
            rdfs:label "Employee"@en .

        # Properties (with entity ranges for SymbolTable compatibility)
        ex:name a rdf:Property ;
            rdfs:domain ex:Entity ;
            rdfs:label "name"@en .

        ex:age a rdf:Property ;
            rdfs:domain ex:Person .

        ex:worksFor a rdf:Property ;
            rdfs:domain ex:Employee ;
            rdfs:range ex:Organization .

        ex:knows a rdf:Property ;
            rdfs:domain ex:Person ;
            rdfs:range ex:Person .
    "#;

    // Load with indexing and metadata
    let mut analyzer = SchemaAnalyzer::new().with_indexing().with_metadata();
    analyzer.load_turtle(turtle)?;
    analyzer.analyze()?;

    // Show analysis results
    println!("Loaded schema:");
    println!("  - Classes: {}", analyzer.classes.len());
    println!("  - Properties: {}", analyzer.properties.len());

    // Show subclass hierarchy
    println!("\nClass hierarchy:");
    for (iri, info) in &analyzer.classes {
        let name = iri.split('/').next_back().unwrap_or(iri);
        if !info.subclass_of.is_empty() {
            let parents: Vec<_> = info
                .subclass_of
                .iter()
                .map(|s| s.split('/').next_back().unwrap_or(s))
                .collect();
            println!("  {} -> {:?}", name, parents);
        }
    }

    // Show indexed lookups
    if let Some(index) = analyzer.index() {
        let person_triples = index.find_by_subject("http://example.org/Person");
        println!("\nTriples about Person: {}", person_triples.len());
    }

    // Show multilingual metadata
    if let Some(metadata) = analyzer.metadata() {
        if let Some(meta) = metadata.get("http://example.org/Person") {
            println!("\nPerson labels:");
            println!("  EN: {}", meta.get_label(Some("en")).unwrap_or("N/A"));
            println!("  FR: {}", meta.get_label(Some("fr")).unwrap_or("N/A"));
        }
    }

    // Convert to SymbolTable
    let symbol_table = analyzer.to_symbol_table()?;
    println!(
        "\nConverted to SymbolTable: {} domains, {} predicates",
        symbol_table.domains.len(),
        symbol_table.predicates.len()
    );

    println!("Phase 1 complete!\n");
    Ok(())
}

/// Phase 2: Apply RDFS inference
fn phase2_rdfs_inference() -> Result<()> {
    println!("--- Phase 2: RDFS Inference ---");

    let turtle = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .

        # Class hierarchy
        ex:Animal a rdfs:Class .
        ex:Mammal a rdfs:Class ; rdfs:subClassOf ex:Animal .
        ex:Dog a rdfs:Class ; rdfs:subClassOf ex:Mammal .
        ex:Cat a rdfs:Class ; rdfs:subClassOf ex:Mammal .

        # Property hierarchy
        ex:hasRelative a rdf:Property .
        ex:hasParent a rdf:Property ; rdfs:subPropertyOf ex:hasRelative .
        ex:hasFather a rdf:Property ; rdfs:subPropertyOf ex:hasParent .
        ex:hasMother a rdf:Property ; rdfs:subPropertyOf ex:hasParent .
    "#;

    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(turtle)?;
    analyzer.analyze()?;

    // Apply RDFS inference
    let engine = analyzer.create_inference_engine();

    // Check transitive subclass relationships
    println!("Transitive subclass checks:");
    let dog_is_mammal =
        engine.is_subclass_of("http://example.org/Dog", "http://example.org/Mammal");
    let dog_is_animal =
        engine.is_subclass_of("http://example.org/Dog", "http://example.org/Animal");
    println!("  Dog subClassOf Mammal: {}", dog_is_mammal);
    println!("  Dog subClassOf Animal: {} (inferred)", dog_is_animal);

    // Check transitive subproperty relationships
    println!("\nTransitive subproperty checks:");
    let father_is_parent = engine.is_subproperty_of(
        "http://example.org/hasFather",
        "http://example.org/hasParent",
    );
    let father_is_relative = engine.is_subproperty_of(
        "http://example.org/hasFather",
        "http://example.org/hasRelative",
    );
    println!("  hasFather subPropertyOf hasParent: {}", father_is_parent);
    println!(
        "  hasFather subPropertyOf hasRelative: {} (inferred)",
        father_is_relative
    );

    // Get all superclasses
    let dog_supers = engine.get_all_superclasses("http://example.org/Dog");
    let super_names: Vec<_> = dog_supers
        .iter()
        .map(|s| s.split('/').next_back().unwrap_or(s))
        .collect();
    println!("\nAll superclasses of Dog: {:?}", super_names);

    // Show inference statistics
    let stats = engine.get_inference_stats();
    println!("\nInference statistics:");
    println!("  Original triples: {}", stats.original_triples);
    println!("  Inferred triples: {}", stats.inferred_triples);
    println!("  Subclass relations: {}", stats.subclass_relations);
    println!("  Subproperty relations: {}", stats.subproperty_relations);

    println!("Phase 2 complete!\n");
    Ok(())
}

/// Phase 3: Parse and compile SHACL constraints
fn phase3_shacl_constraints() -> Result<()> {
    println!("--- Phase 3: SHACL Constraint Compilation ---");

    let shacl = r#"
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        # Person validation shape
        ex:PersonShape a sh:NodeShape ;
            sh:targetClass ex:Person ;
            sh:property [
                sh:path ex:name ;
                sh:minCount 1 ;
                sh:maxCount 1 ;
                sh:datatype xsd:string ;
                sh:minLength 1 ;
                sh:maxLength 100 ;
            ] ;
            sh:property [
                sh:path ex:age ;
                sh:minCount 0 ;
                sh:maxCount 1 ;
                sh:datatype xsd:integer ;
                sh:minInclusive 0 ;
                sh:maxInclusive 150 ;
            ] ;
            sh:property [
                sh:path ex:email ;
                sh:pattern "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$" ;
            ] .

        # Employee validation shape
        ex:EmployeeShape a sh:NodeShape ;
            sh:targetClass ex:Employee ;
            sh:property [
                sh:path ex:employeeId ;
                sh:minCount 1 ;
                sh:datatype xsd:string ;
            ] ;
            sh:property [
                sh:path ex:department ;
                sh:in ( "Engineering" "Sales" "Marketing" "HR" ) ;
            ] ;
            sh:and (
                ex:PersonShape
            ) .
    "#;

    // Parse SHACL shapes
    let symbol_table = tensorlogic_adapters::SymbolTable::new();
    let mut converter = ShaclConverter::new(symbol_table);
    converter.parse_shapes(shacl)?;

    println!("Parsed SHACL shapes:");
    println!("  - Node shapes: {}", converter.shapes().len());

    // Show constraint types
    for (iri, shape) in converter.shapes() {
        let shape_name = iri.split('/').next_back().unwrap_or(iri);
        println!("\n  Shape: {}", shape_name);
        if let Some(target) = &shape.target_class {
            let target_name = target.split('/').next_back().unwrap_or(target);
            println!("    Target class: {}", target_name);
        }
        println!("    Property constraints: {}", shape.properties.len());
    }

    // Compile to TensorLogic expressions
    let expressions = converter.convert_to_rules(shacl)?;
    println!("\nCompiled {} TensorLogic expressions", expressions.len());

    // Show a sample expression
    if let Some(expr) = expressions.first() {
        let expr_str = format!("{:?}", expr);
        let truncated = if expr_str.len() > 100 {
            format!("{}...", &expr_str[..100])
        } else {
            expr_str
        };
        println!("Sample expression: {}", truncated);
    }

    println!("Phase 3 complete!\n");
    Ok(())
}

/// Phase 4: Compile SPARQL queries to TensorLogic
fn phase4_sparql_queries() -> Result<()> {
    println!("--- Phase 4: SPARQL Query Compilation ---");

    let mut compiler = SparqlCompiler::new();

    // Add predicate mappings
    compiler.add_predicate_mapping("http://example.org/knows".to_string(), "knows".to_string());
    compiler.add_predicate_mapping("http://example.org/name".to_string(), "name".to_string());
    compiler.add_predicate_mapping("http://example.org/age".to_string(), "age".to_string());
    compiler.add_predicate_mapping(
        "http://example.org/worksFor".to_string(),
        "worksFor".to_string(),
    );

    // Test different query types
    println!("Query compilation examples:\n");

    // 1. SELECT query with OPTIONAL
    let select_query = r#"
        SELECT ?person ?name ?age WHERE {
            ?person <http://example.org/name> ?name .
            OPTIONAL { ?person <http://example.org/age> ?age }
            FILTER(?age > 18)
        } LIMIT 100
    "#;

    let parsed = compiler.parse_query(select_query)?;
    let tl_expr = compiler.compile_to_tensorlogic(&parsed)?;

    println!("1. SELECT with OPTIONAL and FILTER:");
    if let tensorlogic_oxirs_bridge::sparql::QueryType::Select {
        select_vars,
        distinct,
        ..
    } = &parsed.query_type
    {
        println!("   Variables: {:?}", select_vars);
        println!("   DISTINCT: {}", distinct);
    }
    println!("   LIMIT: {:?}", parsed.limit);
    println!(
        "   Compiled to TLExpr: {} bytes",
        format!("{:?}", tl_expr).len()
    );

    // 2. ASK query
    let ask_query = r#"
        ASK WHERE {
            ?x <http://example.org/knows> ?y .
        }
    "#;

    let parsed_ask = compiler.parse_query(ask_query)?;
    let ask_expr = compiler.compile_to_tensorlogic(&parsed_ask)?;

    println!("\n2. ASK query (existence check):");
    println!(
        "   Compiled to TLExpr: {} bytes",
        format!("{:?}", ask_expr).len()
    );

    // 3. CONSTRUCT query
    let construct_query = r#"
        CONSTRUCT { ?x <http://example.org/colleague> ?y }
        WHERE {
            ?x <http://example.org/worksFor> ?org .
            ?y <http://example.org/worksFor> ?org .
            FILTER(?x != ?y)
        }
    "#;

    let parsed_construct = compiler.parse_query(construct_query)?;
    let construct_expr = compiler.compile_to_tensorlogic(&parsed_construct)?;

    println!("\n3. CONSTRUCT query (infer new triples):");
    if let tensorlogic_oxirs_bridge::sparql::QueryType::Construct { template } =
        &parsed_construct.query_type
    {
        println!("   Template patterns: {}", template.len());
    }
    println!(
        "   Compiled to TLExpr: {} bytes",
        format!("{:?}", construct_expr).len()
    );

    // 4. UNION query
    let union_query = r#"
        SELECT ?x ?rel WHERE {
            { ?x <http://example.org/knows> ?rel }
            UNION
            { ?x <http://example.org/worksFor> ?rel }
        } ORDER BY ?x
    "#;

    let parsed_union = compiler.parse_query(union_query)?;
    let union_expr = compiler.compile_to_tensorlogic(&parsed_union)?;

    println!("\n4. UNION query (disjunction):");
    println!("   ORDER BY: {:?}", parsed_union.order_by);
    println!(
        "   Compiled to TLExpr: {} bytes",
        format!("{:?}", union_expr).len()
    );

    println!("\nPhase 4 complete!\n");
    Ok(())
}

/// Phase 5: Track provenance with RDF*
fn phase5_provenance_tracking() -> Result<()> {
    println!("--- Phase 5: Provenance Tracking ---");

    // Create a provenance tracker with RDF* support
    let mut tracker = ProvenanceTracker::with_rdfstar();

    // Track entity-tensor mappings
    tracker.track_entity("http://example.org/alice".to_string(), 0);
    tracker.track_entity("http://example.org/bob".to_string(), 1);
    tracker.track_entity("http://example.org/charlie".to_string(), 2);

    println!("Entity-to-tensor mappings:");
    println!("  alice -> tensor index 0");
    println!("  bob -> tensor index 1");
    println!("  charlie -> tensor index 2");

    // Create RDF* provenance store
    let mut rdfstar_store = RdfStarProvenanceStore::new();

    // Track inferred statements with metadata
    let metadata1 = MetadataBuilder::for_triple(
        "http://example.org/alice".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
        "http://example.org/Person".to_string(),
    )
    .confidence(0.95)
    .source("rdfs-inference".to_string())
    .rule_id("rule-001".to_string())
    .build();

    rdfstar_store.add_metadata(metadata1);

    let metadata2 = MetadataBuilder::for_triple(
        "http://example.org/alice".to_string(),
        "http://example.org/knows".to_string(),
        "http://example.org/bob".to_string(),
    )
    .confidence(0.87)
    .source("shacl-validation".to_string())
    .rule_id("rule-002".to_string())
    .build();

    rdfstar_store.add_metadata(metadata2);

    let metadata3 = MetadataBuilder::for_triple(
        "http://example.org/bob".to_string(),
        "http://example.org/knows".to_string(),
        "http://example.org/charlie".to_string(),
    )
    .confidence(0.92)
    .source("sparql-construct".to_string())
    .rule_id("rule-003".to_string())
    .build();

    rdfstar_store.add_metadata(metadata3);

    println!("\nRDF* provenance tracking:");
    println!("  Total statements: {}", rdfstar_store.len());

    // Query by confidence
    let high_confidence = rdfstar_store.get_by_min_confidence(0.90);
    println!(
        "  High confidence (>= 0.90): {} statements",
        high_confidence.len()
    );

    // Query by source
    let from_inference = rdfstar_store.get_by_source("rdfs-inference");
    println!("  From RDFS inference: {} statements", from_inference.len());

    // Query by rule
    let from_rule_002 = rdfstar_store.get_by_rule("rule-002");
    println!("  From rule-002: {} statements", from_rule_002.len());

    // Export provenance statistics
    let stats = rdfstar_store.get_stats();
    println!("\nProvenance statistics:");
    println!("  Total statements: {}", stats.total_statements);
    println!("  With confidence: {}", stats.with_confidence);
    println!("  With source: {}", stats.with_source);
    println!("  With rule_id: {}", stats.with_rule);

    // Export to RDF* Turtle
    let turtle_export = rdfstar_store.to_turtle();
    println!("\nRDF* Turtle export: {} bytes", turtle_export.len());

    println!("Phase 5 complete!\n");
    Ok(())
}

/// Phase 6: Validation pipeline
fn phase6_validation_pipeline() -> Result<()> {
    println!("--- Phase 6: Validation Pipeline ---");

    // Create a validator (typically used for mock validation demos)
    let _validator = ShaclValidator::new();

    // Simulate validation results
    let mut report = tensorlogic_oxirs_bridge::shacl::validation::ValidationReport::new();

    // Add some validation results
    let result1 = tensorlogic_oxirs_bridge::shacl::validation::ValidationResult::new(
        "http://example.org/person1",
        "http://example.org/PersonShape",
        "http://www.w3.org/ns/shacl#MinCountConstraintComponent",
        "Missing required property: name",
    )
    .with_severity(ValidationSeverity::Violation)
    .with_path("http://example.org/name");

    let result2 = tensorlogic_oxirs_bridge::shacl::validation::ValidationResult::new(
        "http://example.org/person2",
        "http://example.org/PersonShape",
        "http://www.w3.org/ns/shacl#MaxInclusiveConstraintComponent",
        "Age value is unusually high: 200",
    )
    .with_severity(ValidationSeverity::Warning)
    .with_path("http://example.org/age")
    .with_value("200");

    let result3 = tensorlogic_oxirs_bridge::shacl::validation::ValidationResult::new(
        "http://example.org/person3",
        "http://example.org/PersonShape",
        "http://www.w3.org/ns/shacl#PatternConstraintComponent",
        "Email pattern matched",
    )
    .with_severity(ValidationSeverity::Info)
    .with_path("http://example.org/email");

    report.add_result(result1);
    report.add_result(result2);
    report.add_result(result3);

    println!("Validation report:");
    println!("  Conforms: {}", report.conforms);
    println!("  Total results: {}", report.results.len());
    println!(
        "  Violations: {}",
        report
            .results
            .iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Violation))
            .count()
    );
    println!(
        "  Warnings: {}",
        report
            .results
            .iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Warning))
            .count()
    );
    println!(
        "  Info: {}",
        report
            .results
            .iter()
            .filter(|r| matches!(r.severity, ValidationSeverity::Info))
            .count()
    );

    // Export to Turtle
    let turtle_report = report.to_turtle();
    println!("\nTurtle report: {} bytes", turtle_report.len());

    // Export to JSON
    let json_report = report.to_json()?;
    println!("JSON report: {} bytes", json_report.len());

    println!("Phase 6 complete!\n");
    Ok(())
}

/// Phase 7: Caching and performance optimization
fn phase7_caching_performance() -> Result<()> {
    println!("--- Phase 7: Caching and Performance ---");

    // Create a schema cache
    let mut cache = SchemaCache::new();

    let turtle = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .

        ex:Person a rdfs:Class .
        ex:knows a rdf:Property .
    "#;

    // First access - cache miss
    let start = std::time::Instant::now();
    let symbol_table = if let Some(cached) = cache.get_symbol_table(turtle) {
        println!("Cache HIT (first access)");
        cached
    } else {
        println!("Cache MISS (first access)");
        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(turtle)?;
        analyzer.analyze()?;
        let table = analyzer.to_symbol_table()?;
        cache.put_symbol_table(turtle, table.clone());
        table
    };
    let first_duration = start.elapsed();

    // Second access - cache hit
    let start = std::time::Instant::now();
    let _cached_table = if let Some(cached) = cache.get_symbol_table(turtle) {
        println!("Cache HIT (second access)");
        cached
    } else {
        println!("Cache MISS (second access)");
        symbol_table.clone()
    };
    let second_duration = start.elapsed();

    // Show performance improvement
    println!("\nPerformance comparison:");
    println!("  First access: {:?}", first_duration);
    println!("  Second access: {:?}", second_duration);
    if first_duration.as_nanos() > 0 {
        let speedup = first_duration.as_nanos() as f64 / second_duration.as_nanos().max(1) as f64;
        println!("  Speedup: {:.1}x", speedup);
    }

    // Cache statistics
    let stats = cache.stats();
    println!("\nCache statistics:");
    println!("  Hits: {}", stats.total_hits);
    println!("  Misses: {}", stats.total_misses);
    println!("  Hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("  Symbol table entries: {}", stats.symbol_table_entries);

    println!("Phase 7 complete!\n");
    Ok(())
}

/// Phase 8: GraphQL integration
fn phase8_graphql_integration() -> Result<()> {
    println!("--- Phase 8: GraphQL Integration ---");

    let graphql_schema = r#"
        type Person {
            id: ID!
            name: String!
            age: Int
            email: String
            friends: [Person!]
        }

        type Organization {
            id: ID!
            name: String!
            employees: [Person!]!
            location: String
        }

        type Query {
            person(id: ID!): Person
            organization(id: ID!): Organization
            allPersons: [Person!]!
        }

        type Mutation {
            createPerson(name: String!, age: Int): Person
            updatePerson(id: ID!, name: String, age: Int): Person
        }
    "#;

    // Convert GraphQL schema to TensorLogic
    let mut converter = GraphQLConverter::new();
    converter.parse_schema(graphql_schema)?;

    // Show conversion results
    println!("GraphQL to TensorLogic conversion:");
    println!("  Types discovered: {}", converter.types().len());

    for (type_name, type_def) in converter.types() {
        println!("\n  Type: {}", type_name);
        println!("    Fields: {}", type_def.fields.len());
        for field in &type_def.fields {
            let required = if field.is_required { "!" } else { "" };
            let list = if field.is_list { "[]" } else { "" };
            println!(
                "      - {}: {}{}{}",
                field.name, list, field.field_type, required
            );
        }
    }

    // Convert to SymbolTable
    let symbol_table = converter.to_symbol_table()?;
    println!("\nConverted to SymbolTable:");
    println!("  Domains: {}", symbol_table.domains.len());
    println!("  Predicates: {}", symbol_table.predicates.len());

    println!("Phase 8 complete!\n");
    Ok(())
}
