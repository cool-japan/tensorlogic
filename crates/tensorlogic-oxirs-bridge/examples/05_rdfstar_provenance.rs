//! RDF* provenance tracking example
//!
//! This example demonstrates how to:
//! 1. Track provenance of tensor computations
//! 2. Use RDF* quoted triples for metadata
//! 3. Store and query provenance information
//! 4. Export provenance as Turtle and JSON
//!
//! Run with: cargo run --example 05_rdfstar_provenance -p tensorlogic-oxirs-bridge

use anyhow::Result;
use tensorlogic_oxirs_bridge::{
    rdfstar::{MetadataBuilder, RdfStarProvenanceStore},
    ProvenanceTracker,
};

fn main() -> Result<()> {
    println!("=== RDF* Provenance Tracking ===\n");

    // Step 1: Create provenance tracker for basic tracking
    println!("Step 1: Basic Provenance Tracking");
    println!("{}", "=".repeat(70));

    let mut tracker = ProvenanceTracker::new();

    // Track entity-to-tensor-index mappings
    tracker.track_entity("http://example.org/Person".to_string(), 0);
    tracker.track_entity("http://example.org/Organization".to_string(), 1);
    tracker.track_entity("http://example.org/Document".to_string(), 2);
    tracker.track_entity("http://example.org/knows".to_string(), 3);
    tracker.track_entity("http://example.org/worksFor".to_string(), 4);
    tracker.track_entity("http://example.org/authored".to_string(), 5);

    println!(
        "Tracked {} entities to tensor indices",
        tracker.entity_to_tensor.len()
    );

    // Track shape-to-rule mappings
    tracker.track_shape(
        "http://example.org/shapes#PersonShape".to_string(),
        "∀x. Person(x) → ∃y. knows(x, y)".to_string(),
        0,
    );

    tracker.track_shape(
        "http://example.org/shapes#OrganizationShape".to_string(),
        "∀x. Organization(x) → ∃y. worksFor(y, x)".to_string(),
        1,
    );

    println!("Tracked {} shapes to rules", tracker.shape_to_rule.len());

    // Export basic provenance
    println!("\nBasic Provenance (JSON):");
    let json = tracker.to_json()?;
    println!("{}", json);

    // Step 2: Advanced RDF* provenance with metadata
    println!("\n\nStep 2: Advanced RDF* Provenance Store");
    println!("{}", "=".repeat(70));

    let mut store = RdfStarProvenanceStore::new();

    // Add statements with rich metadata
    println!("\nAdding provenance statements...");

    // Statement 1: Person knows Person (inferred with high confidence)
    let metadata1 = MetadataBuilder::for_triple(
        "http://example.org/Alice".to_string(),
        "http://example.org/knows".to_string(),
        "http://example.org/Bob".to_string(),
    )
    .source("http://example.org/reasoning-engine".to_string())
    .rule_id("http://example.org/rules#transitivity".to_string())
    .confidence(0.95)
    .generated_at("2025-01-07T10:30:00Z".to_string())
    .custom(
        "inference_method".to_string(),
        "forward-chaining".to_string(),
    )
    .build();

    store.add_metadata(metadata1);

    // Statement 2: Person worksFor Organization (directly asserted)
    let metadata2 = MetadataBuilder::for_triple(
        "http://example.org/Alice".to_string(),
        "http://example.org/worksFor".to_string(),
        "http://example.org/ACME_Corp".to_string(),
    )
    .source("http://example.org/hr-database".to_string())
    .rule_id("http://example.org/rules#direct-assertion".to_string())
    .confidence(1.0)
    .generated_at("2025-01-06T14:20:00Z".to_string())
    .custom("verified_by".to_string(), "hr_admin".to_string())
    .build();

    store.add_metadata(metadata2);

    // Statement 3: Person authored Document (extracted from text)
    let metadata3 = MetadataBuilder::for_triple(
        "http://example.org/Alice".to_string(),
        "http://example.org/authored".to_string(),
        "http://example.org/TechReport_2025".to_string(),
    )
    .source("http://example.org/nlp-extractor".to_string())
    .rule_id("http://example.org/rules#entity-extraction".to_string())
    .confidence(0.87)
    .generated_at("2025-01-07T09:15:00Z".to_string())
    .custom("extraction_model".to_string(), "bert-base-ner".to_string())
    .custom(
        "context_sentence".to_string(),
        "Alice wrote the technical report".to_string(),
    )
    .build();

    store.add_metadata(metadata3);

    println!("✓ Added {} statements with metadata", store.len());

    // Step 3: Query provenance by various criteria
    println!("\n\nStep 3: Querying Provenance");
    println!("{}", "=".repeat(70));

    // Query by source
    println!("\nStatements from reasoning engine:");
    let by_source = store.get_by_source("http://example.org/reasoning-engine");
    for stmt in &by_source {
        println!(
            "  << {} {} {} >>",
            stmt.statement.subject, stmt.statement.predicate, stmt.statement.object
        );
        if let Some(conf) = stmt.confidence {
            println!("    Confidence: {:.2}", conf);
        }
    }

    // Query by rule
    println!("\nStatements inferred by transitivity rule:");
    let by_rule = store.get_by_rule("http://example.org/rules#transitivity");
    for stmt in &by_rule {
        println!(
            "  << {} {} {} >>",
            stmt.statement.subject, stmt.statement.predicate, stmt.statement.object
        );
        if let Some(ref source) = stmt.source {
            println!("    Source: {}", source);
        }
    }

    // Query by confidence threshold
    println!("\nHigh confidence statements (>= 0.9):");
    let high_conf = store.get_by_min_confidence(0.9);
    for stmt in &high_conf {
        println!(
            "  << {} {} {} >>",
            stmt.statement.subject, stmt.statement.predicate, stmt.statement.object
        );
        if let Some(conf) = stmt.confidence {
            println!("    Confidence: {:.2}", conf);
        }
    }

    // Query by predicate
    println!("\nStatements with 'authored' predicate:");
    let by_pred = store.get_by_predicate("http://example.org/authored");
    for stmt in &by_pred {
        println!(
            "  << {} {} {} >>",
            stmt.statement.subject, stmt.statement.predicate, stmt.statement.object
        );
        if let Some(model) = stmt.custom.get("extraction_model") {
            println!("    Extraction model: {}", model);
        }
    }

    // Step 4: Export as RDF* Turtle
    println!("\n\nStep 4: RDF* Turtle Export");
    println!("{}", "=".repeat(70));

    let turtle = store.to_turtle();
    println!("\n{}", turtle);

    // Step 5: Export as JSON
    println!("\n\nStep 5: JSON Export");
    println!("{}", "=".repeat(70));

    let json = store.to_json()?;
    println!("\n{}", json);

    // Step 6: Provenance statistics
    println!("\n\nStep 6: Provenance Statistics");
    println!("{}", "=".repeat(70));

    let stats = store.get_stats();
    println!("Total statements: {}", stats.total_statements);
    println!("Statements with confidence: {}", stats.with_confidence);
    println!("Statements with source: {}", stats.with_source);
    println!("Statements with rule: {}", stats.with_rule);
    println!("Unique sources: {}", stats.unique_sources);
    println!("Unique rules: {}", stats.unique_rules);

    // Step 7: Clear and reset
    println!("\n\nStep 7: Clear Provenance Store");
    println!("{}", "=".repeat(70));
    println!("Before clear: {} statements", store.len());
    store.clear();
    println!("After clear: {} statements", store.len());

    println!("\n=== Example Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("  - Entity-to-tensor tracking");
    println!("  - Shape-to-rule mapping");
    println!("  - RDF* quoted triples with metadata");
    println!("  - Confidence scoring");
    println!("  - Timestamp tracking");
    println!("  - Custom attributes");
    println!("  - Multiple query methods");
    println!("  - Turtle and JSON export");
    println!("  - Statistical summaries");

    Ok(())
}
