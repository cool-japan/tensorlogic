//! Property-based tests for tensorlogic-oxirs-bridge
//!
//! These tests use proptest to generate random inputs and verify
//! that various invariants hold across all inputs.

use proptest::prelude::*;
use tensorlogic_oxirs_bridge::{
    schema::{nquads::NQuadsProcessor, streaming::StreamAnalyzer},
    Quad,
};

// Strategy for generating valid IRIs
fn valid_iri_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("http://example\\.org/[a-zA-Z][a-zA-Z0-9_]{0,20}")
        .expect("Invalid regex")
}

// Strategy for generating simple literal values
fn simple_literal_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z0-9 ]{1,50}").expect("Invalid regex")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test that N-Quads roundtrip preserves data
    #[test]
    fn nquads_roundtrip_preserves_data(
        subject in valid_iri_strategy(),
        predicate in valid_iri_strategy(),
        object_iri in valid_iri_strategy(),
        graph in valid_iri_strategy()
    ) {
        let mut processor = NQuadsProcessor::new();

        // Add a quad
        let quad = Quad::new(
            subject.clone(),
            predicate.clone(),
            format!("<{}>", object_iri),
            Some(graph.clone()),
        );
        processor.add_quad(quad);

        // Export and re-import
        let nquads = processor.to_nquads();
        let mut processor2 = NQuadsProcessor::new();
        prop_assert!(processor2.load_nquads(&nquads).is_ok());

        // Verify counts match
        prop_assert_eq!(processor.total_quads(), processor2.total_quads());
    }

    /// Test that stream analyzer counts correctly
    #[test]
    fn stream_analyzer_counts_correctly(
        num_subjects in 1usize..=10,
        predicates_per_subject in 1usize..=5
    ) {
        let mut analyzer = StreamAnalyzer::new();

        // Generate triples with some overlap in subjects
        let mut total = 0;
        for i in 0..num_subjects {
            for j in 0..predicates_per_subject {
                let subject = format!("http://example.org/s{}", i);
                let predicate = format!("http://example.org/p{}", j);
                let object = format!("http://example.org/o{}_{}", i, j);
                analyzer.process_triple(&subject, &predicate, &object);
                total += 1;
            }
        }

        prop_assert_eq!(analyzer.total_triples(), total);
        prop_assert_eq!(analyzer.unique_subject_count(), num_subjects);
    }

    /// Test that literal escape/unescape are inverse operations
    #[test]
    fn literal_escape_unescape_roundtrip(value in simple_literal_strategy()) {
        let escaped = escape_literal(&value);
        let unescaped = unescape_literal(&escaped);
        prop_assert_eq!(value, unescaped);
    }

    /// Test that N-Quads with multiple graphs maintains separation
    #[test]
    fn nquads_graph_separation(
        num_graphs in 1usize..=5,
        quads_per_graph in 1usize..=10
    ) {
        let mut processor = NQuadsProcessor::new();

        for g in 0..num_graphs {
            let graph = format!("http://example.org/graph{}", g);
            for q in 0..quads_per_graph {
                let quad = Quad::new(
                    format!("http://example.org/s{}_{}", g, q),
                    "http://example.org/p".to_string(),
                    format!("<http://example.org/o{}_{}>", g, q),
                    Some(graph.clone()),
                );
                processor.add_quad(quad);
            }
        }

        prop_assert_eq!(processor.graph_count(), num_graphs);
        prop_assert_eq!(processor.total_quads(), num_graphs * quads_per_graph);

        // Verify each graph has the right count
        for g in 0..num_graphs {
            let graph = format!("http://example.org/graph{}", g);
            let graph_quads = processor.get_graph(Some(&graph)).expect("Graph should exist");
            prop_assert_eq!(graph_quads.len(), quads_per_graph);
        }
    }

    /// Test that top predicates are sorted by frequency
    #[test]
    fn stream_analyzer_top_predicates_sorted(
        num_predicates in 2usize..=10,
    ) {
        let mut analyzer = StreamAnalyzer::new();

        // Add predicates with varying frequencies
        for i in 0..num_predicates {
            let predicate = format!("http://example.org/p{}", i);
            // More frequent predicates at lower indices
            let count = num_predicates - i;
            for _ in 0..count {
                analyzer.process_triple("http://example.org/s", &predicate, "http://example.org/o");
            }
        }

        let top = analyzer.top_predicates(num_predicates);

        // Verify they're sorted by frequency (descending)
        for i in 0..top.len() - 1 {
            prop_assert!(top[i].1 >= top[i + 1].1, "Not sorted: {:?}", top);
        }
    }

    /// Test that default graph quads have no graph IRI
    #[test]
    fn nquads_default_graph_has_no_iri(
        subject in valid_iri_strategy(),
        predicate in valid_iri_strategy(),
        object_iri in valid_iri_strategy()
    ) {
        let mut processor = NQuadsProcessor::new();

        // Add a quad to default graph
        let quad = Quad::default_graph(
            subject,
            predicate,
            format!("<{}>", object_iri),
        );
        processor.add_quad(quad);

        // Verify it's in the default graph
        let default_quads = processor.get_graph(None).expect("Default graph should exist");
        prop_assert_eq!(default_quads.len(), 1);
        prop_assert!(default_quads[0].graph.is_none());
    }

    /// Test that Quad serialization preserves all fields
    #[test]
    fn quad_serialization_preserves_fields(
        subject in valid_iri_strategy(),
        predicate in valid_iri_strategy(),
        object_iri in valid_iri_strategy(),
        graph in valid_iri_strategy()
    ) {
        let quad = Quad::new(
            subject.clone(),
            predicate.clone(),
            format!("<{}>", object_iri.clone()),
            Some(graph.clone()),
        );

        let nquads = quad.to_nquads();

        // Verify all parts are in the output
        prop_assert!(nquads.contains(&subject), "Missing subject");
        prop_assert!(nquads.contains(&predicate), "Missing predicate");
        prop_assert!(nquads.contains(&object_iri), "Missing object");
        prop_assert!(nquads.contains(&graph), "Missing graph");
        prop_assert!(nquads.ends_with(" .\n"), "Missing terminator");
    }

    /// Test that predicate stats are accurate
    #[test]
    fn stream_analyzer_predicate_stats_accurate(
        counts in prop::collection::vec(1usize..=20, 1..=5)
    ) {
        let mut analyzer = StreamAnalyzer::new();

        for (i, &count) in counts.iter().enumerate() {
            let predicate = format!("http://example.org/p{}", i);
            for _ in 0..count {
                analyzer.process_triple("http://example.org/s", &predicate, "http://example.org/o");
            }
        }

        let stats = analyzer.predicate_stats();
        prop_assert_eq!(stats.len(), counts.len());

        for (i, &count) in counts.iter().enumerate() {
            let predicate = format!("http://example.org/p{}", i);
            prop_assert_eq!(*stats.get(&predicate).unwrap(), count);
        }
    }
}

// Helper functions for testing

/// Escape a string literal for N-Triples/N-Quads format
fn escape_literal(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{}\"", escaped)
}

/// Unescape a literal value from N-Triples/N-Quads format
fn unescape_literal(s: &str) -> String {
    // Remove quotes
    let s = s.trim_matches('"');
    s.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace("\\\"", "\"")
        .replace("\\\\", "\\")
}

// Additional unit tests that complement property tests

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_escape_special_chars() {
        assert_eq!(escape_literal("hello"), r#""hello""#);
        assert_eq!(escape_literal("hello\"world"), r#""hello\"world""#);
        assert_eq!(escape_literal("line1\nline2"), r#""line1\nline2""#);
    }

    #[test]
    fn test_unescape_special_chars() {
        assert_eq!(unescape_literal(r#""hello""#), "hello");
        assert_eq!(unescape_literal(r#""hello\"world""#), "hello\"world");
        assert_eq!(unescape_literal(r#""line1\nline2""#), "line1\nline2");
    }

    #[test]
    fn test_empty_analyzer() {
        let analyzer = StreamAnalyzer::new();
        assert_eq!(analyzer.total_triples(), 0);
        assert_eq!(analyzer.unique_subject_count(), 0);
        assert!(analyzer.predicate_stats().is_empty());
    }

    #[test]
    fn test_empty_processor() {
        let processor = NQuadsProcessor::new();
        assert_eq!(processor.total_quads(), 0);
        assert_eq!(processor.graph_count(), 0);
    }
}
