//! Example: RDF* Provenance Tracking with TensorLogic
//!
//! This example demonstrates:
//! - Tracking inferred triples with confidence scores using RDF*
//! - Recording provenance metadata (source, generation method, timestamps)
//! - Querying provenance by confidence, source, and rule ID
//! - Exporting provenance to RDF* Turtle format
//! - Integration with RDFS inference engine

use anyhow::Result;
use tensorlogic_oxirs_bridge::{
    MetadataBuilder, ProvenanceTracker, QuotedTriple, RdfsInferenceEngine, SchemaAnalyzer,
};

fn main() -> Result<()> {
    println!("=== RDF* Provenance Tracking Example ===\n");

    // Example: A simple knowledge base with person relationships
    let kb_turtle = r#"
        @prefix ex: <http://example.org/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

        # Class hierarchy
        ex:Person rdfs:subClassOf ex:LivingThing .
        ex:Student rdfs:subClassOf ex:Person .
        ex:Professor rdfs:subClassOf ex:Person .

        # Properties
        ex:knows rdfs:domain ex:Person ;
                 rdfs:range ex:Person .

        ex:teaches rdfs:domain ex:Professor ;
                   rdfs:range ex:Student .

        # Instance data
        ex:alice rdf:type ex:Student .
        ex:bob rdf:type ex:Professor .
        ex:alice ex:knows ex:bob .
        ex:bob ex:teaches ex:alice .
    "#;

    println!("1. Loading knowledge base...");
    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(kb_turtle)?;
    analyzer.analyze()?;
    println!("   Loaded {} triples\n", analyzer.graph.len());

    // 2. Run RDFS inference
    println!("2. Running RDFS inference...");
    let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
    engine.materialize()?;

    let stats = engine.get_inference_stats();
    println!("   Original triples: {}", stats.original_triples);
    println!("   Inferred triples: {}", stats.inferred_triples);
    println!("   Total triples: {}\n", stats.total_triples);

    // 3. Create provenance tracker with RDF* support
    println!("3. Tracking inference provenance with RDF*...");
    let mut tracker = ProvenanceTracker::with_rdfstar();

    // Track some inferred triples with metadata
    // Simulate different confidence levels for different inferences

    // High confidence: Type inference from subClassOf
    tracker.track_inferred_triple(
        "http://example.org/alice".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
        "http://example.org/Person".to_string(),
        Some("rdfs9_subclass_inference".to_string()),
        Some(0.99),
    );

    tracker.track_inferred_triple(
        "http://example.org/alice".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
        "http://example.org/LivingThing".to_string(),
        Some("rdfs9_subclass_inference".to_string()),
        Some(0.99),
    );

    // Medium confidence: Domain inference
    tracker.track_inferred_triple(
        "http://example.org/alice".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
        "http://example.org/Person".to_string(),
        Some("rdfs2_domain_inference".to_string()),
        Some(0.95),
    );

    // Medium-high confidence: Range inference
    tracker.track_inferred_triple(
        "http://example.org/bob".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
        "http://example.org/Person".to_string(),
        Some("rdfs3_range_inference".to_string()),
        Some(0.95),
    );

    // Lower confidence: Uncertain inference from noisy data
    tracker.track_inferred_triple(
        "http://example.org/alice".to_string(),
        "http://example.org/friendOf".to_string(),
        "http://example.org/bob".to_string(),
        Some("probabilistic_inference".to_string()),
        Some(0.75),
    );

    println!(
        "   Tracked {} statements with provenance\n",
        tracker.rdfstar_store().unwrap().len()
    );

    // 4. Query provenance by confidence
    println!("4. Querying high-confidence inferences (>= 0.95)...");
    let high_conf = tracker.get_high_confidence_inferences(0.95);
    println!("   Found {} high-confidence inferences:", high_conf.len());
    for metadata in &high_conf {
        println!(
            "     - {} {} {} (confidence: {:.2}, rule: {:?})",
            extract_local_name(&metadata.statement.subject),
            extract_local_name(&metadata.statement.predicate),
            extract_local_name(&metadata.statement.object),
            metadata.confidence.unwrap_or(0.0),
            metadata.rule_id.as_deref().unwrap_or("N/A"),
        );
    }
    println!();

    // 5. Query by rule ID
    if let Some(store) = tracker.rdfstar_store() {
        println!("5. Querying inferences by rule...");

        let subclass_inferences = store.get_by_rule("rdfs9_subclass_inference");
        println!("   Subclass inferences: {}", subclass_inferences.len());

        let domain_inferences = store.get_by_rule("rdfs2_domain_inference");
        println!("   Domain inferences: {}", domain_inferences.len());

        let probabilistic_inferences = store.get_by_rule("probabilistic_inference");
        println!(
            "   Probabilistic inferences: {}",
            probabilistic_inferences.len()
        );
        println!();

        // 6. Show detailed provenance for one statement
        println!("6. Detailed provenance for a specific statement:");
        if let Some(metadata) = subclass_inferences.first() {
            println!(
                "   Statement: {} {} {}",
                extract_local_name(&metadata.statement.subject),
                extract_local_name(&metadata.statement.predicate),
                extract_local_name(&metadata.statement.object),
            );
            println!("   Confidence: {:.3}", metadata.confidence.unwrap_or(0.0));
            println!(
                "   Generated by: {}",
                metadata.generated_by.as_deref().unwrap_or("N/A")
            );
            println!(
                "   Rule ID: {}",
                metadata.rule_id.as_deref().unwrap_or("N/A")
            );
            if let Some(ref timestamp) = metadata.generated_at {
                println!("   Generated at: {}", timestamp);
            }
        }
        println!();

        // 7. Provenance statistics
        println!("7. Provenance statistics:");
        let prov_stats = store.get_stats();
        println!(
            "   Total tracked statements: {}",
            prov_stats.total_statements
        );
        println!("   With confidence scores: {}", prov_stats.with_confidence);
        println!("   With source attribution: {}", prov_stats.with_source);
        println!("   With rule IDs: {}", prov_stats.with_rule);
        println!("   Unique rules: {}", prov_stats.unique_rules);
        println!();
    }

    // 8. Advanced: Add custom metadata using the builder
    println!("8. Adding custom provenance metadata...");
    let custom_qt = QuotedTriple::new(
        "http://example.org/alice".to_string(),
        "http://example.org/expertIn".to_string(),
        "http://example.org/MachineLearning".to_string(),
    );

    let custom_metadata = MetadataBuilder::for_quoted_triple(custom_qt)
        .confidence(0.85)
        .source("http://example.org/linkedin_profile".to_string())
        .generated_by("http://tensorlogic.org/nlp_extraction".to_string())
        .rule_id("entity_extraction_rule".to_string())
        .custom("extraction_method".to_string(), "BERT-NER".to_string())
        .custom("model_version".to_string(), "v2.1".to_string())
        .build();

    tracker.rdfstar_store_mut().add_metadata(custom_metadata);
    println!("   Added statement with custom metadata\n");

    // 9. Export provenance to RDF* Turtle
    println!("9. Exporting provenance to RDF* Turtle format...");
    let turtle_export = tracker.to_rdfstar_turtle();

    println!("   First 500 characters of export:");
    println!("   {}", &turtle_export[..turtle_export.len().min(500)]);
    println!("   ...\n");

    // 10. Show how provenance helps with uncertainty quantification
    println!("10. Using provenance for uncertainty quantification:");

    if let Some(store) = tracker.rdfstar_store() {
        let all_metadata = store.all_metadata();

        // Calculate average confidence by rule type
        let mut rule_confidences: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();

        for metadata in all_metadata {
            if let (Some(rule), Some(conf)) = (&metadata.rule_id, metadata.confidence) {
                rule_confidences.entry(rule.clone()).or_default().push(conf);
            }
        }

        for (rule, confidences) in rule_confidences {
            let avg: f64 = confidences.iter().sum::<f64>() / confidences.len() as f64;
            println!(
                "   {}: avg confidence = {:.3} ({} inferences)",
                rule,
                avg,
                confidences.len()
            );
        }
    }

    println!("\n=== Provenance Tracking Complete ===");

    Ok(())
}

fn extract_local_name(iri: &str) -> String {
    iri.split(['/', '#'])
        .rfind(|s| !s.is_empty())
        .unwrap_or(iri)
        .to_string()
}
