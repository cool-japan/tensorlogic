//! Example demonstrating performance features: indexing, caching, and metadata preservation.
//!
//! This example shows how to use:
//! - Triple indexing for fast lookups
//! - Schema caching for repeated operations
//! - Multilingual metadata preservation
//! - Enhanced error handling

use anyhow::Result;
use tensorlogic_oxirs_bridge::{PersistentCache, SchemaAnalyzer, SchemaCache};

fn main() -> Result<()> {
    println!("=== TensorLogic OxiRS Bridge: Performance Features ===\n");

    // Part 1: Triple Indexing for Fast Lookups
    println!("Part 1: Triple Indexing");
    println!("{}", "-".repeat(50));
    demo_triple_indexing()?;

    // Part 2: Schema Caching
    println!("\nPart 2: Schema Caching");
    println!("{}", "-".repeat(50));
    demo_schema_caching()?;

    // Part 3: Metadata Preservation
    println!("\nPart 3: Metadata Preservation");
    println!("{}", "-".repeat(50));
    demo_metadata_preservation()?;

    // Part 4: Performance Comparison
    println!("\nPart 4: Performance Comparison");
    println!("{}", "-".repeat(50));
    demo_performance_comparison()?;

    Ok(())
}

/// Demonstrate triple indexing for fast lookups
fn demo_triple_indexing() -> Result<()> {
    let turtle = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        @prefix ex: <http://example.org/> .

        ex:Alice a foaf:Person ;
                 foaf:name "Alice Smith" ;
                 foaf:knows ex:Bob ;
                 foaf:mbox <mailto:alice@example.org> .

        ex:Bob a foaf:Person ;
               foaf:name "Bob Jones" ;
               foaf:knows ex:Alice ;
               foaf:knows ex:Charlie .

        ex:Charlie a foaf:Person ;
                   foaf:name "Charlie Brown" ;
                   foaf:knows ex:Bob .

        foaf:Person a rdfs:Class ;
                    rdfs:label "Person"@en ;
                    rdfs:label "Personne"@fr ;
                    rdfs:comment "A person"@en .

        foaf:knows a rdf:Property ;
                   rdfs:label "knows"@en ;
                   rdfs:domain foaf:Person ;
                   rdfs:range foaf:Person .
    "#;

    // Create analyzer with indexing enabled
    let mut analyzer = SchemaAnalyzer::new().with_indexing();
    analyzer.load_turtle(turtle)?;

    // Access the index
    if let Some(index) = analyzer.index() {
        println!("Index Statistics:");
        let stats = index.stats();
        println!("  Total triples: {}", stats.total_triples);
        println!("  Unique subjects: {}", stats.unique_subjects);
        println!("  Unique predicates: {}", stats.unique_predicates);
        println!("  Unique objects: {}", stats.unique_objects);

        // Find all triples about Alice
        println!("\nAll triples about Alice:");
        let alice_triples = index.find_by_subject("http://example.org/Alice");
        for (s, p, o) in alice_triples.iter().take(3) {
            println!(
                "  {} -> {} -> {}",
                shorten_iri(s),
                shorten_iri(p),
                shorten_iri(o)
            );
        }

        // Find all "knows" relationships
        println!("\nAll 'knows' relationships:");
        let knows_triples = index.find_by_predicate("http://xmlns.com/foaf/0.1/knows");
        for (s, _p, o) in &knows_triples {
            println!("  {} knows {}", shorten_iri(s), shorten_iri(o));
        }

        // Find by pattern (wildcard queries)
        println!("\nFind all people who know someone:");
        let pattern_triples =
            index.find_by_pattern(None, Some("http://xmlns.com/foaf/0.1/knows"), None);
        println!("  Found {} relationships", pattern_triples.len());

        // Prefix-based search
        println!("\nAll entities in example.org namespace:");
        let example_iris = index.find_by_prefix("http://example.org/");
        for iri in example_iris.iter().take(5) {
            println!("  {}", iri);
        }

        // Graph analytics
        println!("\nGraph Analytics:");
        println!(
            "  Alice's degree (outgoing edges): {}",
            index.subject_degree("http://example.org/Alice")
        );
        println!(
            "  'knows' predicate frequency: {}",
            index.predicate_frequency("http://xmlns.com/foaf/0.1/knows")
        );
    }

    Ok(())
}

/// Demonstrate schema caching for repeated operations
fn demo_schema_caching() -> Result<()> {
    let turtle = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .

        ex:Product a rdfs:Class ;
                   rdfs:label "Product" .

        ex:price a rdf:Property ;
                 rdfs:domain ex:Product .
    "#;

    // In-memory cache
    let mut cache = SchemaCache::new();

    // First access - cache miss
    println!("First parse (cache miss):");
    let table1 = if let Some(cached) = cache.get_symbol_table(turtle) {
        println!("  ✓ Cache hit!");
        cached
    } else {
        println!("  ✗ Cache miss - parsing...");
        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(turtle)?;
        analyzer.analyze()?;
        let table = analyzer.to_symbol_table()?;
        cache.put_symbol_table(turtle, table.clone());
        table
    };

    // Second access - cache hit
    println!("\nSecond parse (cache hit):");
    let _table2 = if let Some(cached) = cache.get_symbol_table(turtle) {
        println!("  ✓ Cache hit!");
        cached
    } else {
        println!("  ✗ Cache miss");
        table1.clone()
    };

    // Show cache statistics
    println!("\nCache Statistics:");
    let stats = cache.stats();
    println!("  {}", stats);

    // Persistent cache example
    println!("\nPersistent Cache (file-based):");
    let cache_dir = std::env::temp_dir().join("tensorlogic_cache_demo");
    let mut persistent = PersistentCache::new(&cache_dir)?;

    // Save to disk
    persistent.save_symbol_table(turtle, &table1)?;
    println!("  ✓ Saved to disk: {:?}", cache_dir);

    // Load from disk
    if let Some(loaded) = persistent.load_symbol_table(turtle)? {
        println!(
            "  ✓ Loaded from disk: {} domains, {} predicates",
            loaded.domains.len(),
            loaded.predicates.len()
        );
    }

    // Cleanup
    persistent.clear_all()?;
    std::fs::remove_dir_all(cache_dir)?;

    Ok(())
}

/// Demonstrate metadata preservation with multilingual support
fn demo_metadata_preservation() -> Result<()> {
    let turtle = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .
        @prefix dc: <http://purl.org/dc/terms/> .

        foaf:Person a rdfs:Class ;
                    rdfs:label "Person"@en ;
                    rdfs:label "Personne"@fr ;
                    rdfs:label "Persona"@es ;
                    rdfs:label "人"@ja ;
                    rdfs:comment "A person, living or dead"@en ;
                    rdfs:comment "Une personne, vivante ou morte"@fr ;
                    dc:creator "FOAF Project" .

        foaf:Organization a rdfs:Class ;
                         rdfs:label "Organization"@en ;
                         rdfs:label "Organisation"@fr .

        foaf:name a rdf:Property ;
                  rdfs:label "name"@en ;
                  rdfs:label "nom"@fr ;
                  rdfs:comment "A name for some thing"@en ;
                  dc:creator "FOAF Project" .
    "#;

    // Create analyzer with metadata preservation
    let mut analyzer = SchemaAnalyzer::new().with_metadata();
    analyzer.load_turtle(turtle)?;

    // Access the metadata store
    if let Some(store) = analyzer.metadata() {
        println!("Metadata Statistics:");
        let stats = store.stats();
        println!("{}", stats);

        // Access multilingual labels
        println!("\nMultilingual Labels for foaf:Person:");
        if let Some(meta) = store.get("http://xmlns.com/foaf/0.1/Person") {
            println!("  English: {}", meta.get_label(Some("en")).unwrap_or("N/A"));
            println!("  French: {}", meta.get_label(Some("fr")).unwrap_or("N/A"));
            println!("  Spanish: {}", meta.get_label(Some("es")).unwrap_or("N/A"));
            println!(
                "  Japanese: {}",
                meta.get_label(Some("ja")).unwrap_or("N/A")
            );

            // Show all labels
            println!("\n  All labels:");
            for label in &meta.labels {
                println!(
                    "    {} ({})",
                    label.value,
                    label.lang.as_deref().unwrap_or("no-lang")
                );
            }
        }

        // Find entities by label
        println!("\nSearch for 'person' (case-insensitive):");
        let results = store.find_by_label("person");
        for meta in results {
            println!(
                "  Found: {} = {}",
                shorten_iri(&meta.iri),
                meta.get_label(Some("en")).unwrap_or("N/A")
            );
        }

        // Find missing metadata
        println!("\nMetadata Quality:");
        let missing_labels = store.find_missing_labels();
        let missing_comments = store.find_missing_comments();
        if missing_labels.is_empty() {
            println!("  ✓ All entities have labels");
        } else {
            println!("  ✗ {} entities missing labels", missing_labels.len());
        }
        if missing_comments.is_empty() {
            println!("  ✓ All entities have comments");
        } else {
            println!("  ⚠ {} entities missing comments", missing_comments.len());
        }

        // Export metadata to JSON
        let json = store.to_json()?;
        println!("\nMetadata JSON export size: {} bytes", json.len());
    }

    Ok(())
}

/// Demonstrate performance comparison with and without optimizations
fn demo_performance_comparison() -> Result<()> {
    use std::time::Instant;

    // Generate a moderately-sized RDF graph
    let turtle = r#"
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <http://example.org/> .

        ex:Class1 a rdfs:Class .
        ex:Class2 a rdfs:Class .
        ex:Class3 a rdfs:Class .
        ex:prop1 a rdf:Property ; rdfs:domain ex:Class1 .
        ex:prop2 a rdf:Property ; rdfs:domain ex:Class2 .
        ex:prop3 a rdf:Property ; rdfs:domain ex:Class3 .
    "#;

    // Benchmark without optimizations
    let start = Instant::now();
    let mut analyzer1 = SchemaAnalyzer::new();
    analyzer1.load_turtle(turtle)?;
    analyzer1.analyze()?;
    let _table1 = analyzer1.to_symbol_table()?;
    let time_no_opt = start.elapsed();

    // Benchmark with indexing
    let start = Instant::now();
    let mut analyzer2 = SchemaAnalyzer::new().with_indexing();
    analyzer2.load_turtle(turtle)?;
    analyzer2.analyze()?;
    let _table2 = analyzer2.to_symbol_table()?;
    let time_with_index = start.elapsed();

    // Benchmark with caching (second access)
    let mut cache = SchemaCache::new();
    let start = Instant::now();
    cache.get_symbol_table(turtle); // Miss
    cache.put_symbol_table(turtle, _table1.clone());
    let _ = cache.get_symbol_table(turtle); // Hit
    let time_cache_hit = start.elapsed();

    println!("Performance Benchmarks:");
    println!("  No optimizations: {:?}", time_no_opt);
    println!("  With indexing: {:?}", time_with_index);
    println!("  Cache hit: {:?}", time_cache_hit);

    println!(
        "\nSpeedup with caching: {:.2}x",
        time_no_opt.as_secs_f64() / time_cache_hit.as_secs_f64()
    );

    Ok(())
}

/// Helper to shorten IRIs for display
fn shorten_iri(iri: &str) -> String {
    if let Some(hash_pos) = iri.rfind('#') {
        iri[hash_pos + 1..].to_string()
    } else if let Some(slash_pos) = iri.rfind('/') {
        iri[slash_pos + 1..].to_string()
    } else {
        iri.to_string()
    }
}
