//! Tests for RDF* support

#[cfg(test)]
mod tests {
    use crate::rdfstar::{
        MetadataBuilder, QuotedTriple, RdfStarProvenanceStore, StatementMetadata,
    };

    #[test]
    fn test_quoted_triple_creation() {
        let qt = QuotedTriple::new(
            "http://example.org/alice".to_string(),
            "http://example.org/knows".to_string(),
            "http://example.org/bob".to_string(),
        );

        assert_eq!(qt.subject, "http://example.org/alice");
        assert_eq!(qt.predicate, "http://example.org/knows");
        assert_eq!(qt.object, "http://example.org/bob");
    }

    #[test]
    fn test_quoted_triple_to_turtle() {
        let qt = QuotedTriple::new(
            "<http://example.org/alice>".to_string(),
            "<http://example.org/knows>".to_string(),
            "<http://example.org/bob>".to_string(),
        );

        let turtle = qt.to_turtle_syntax();
        assert!(turtle.contains("alice"));
        assert!(turtle.contains("knows"));
        assert!(turtle.contains("bob"));
    }

    #[test]
    fn test_statement_metadata_creation() {
        let qt = QuotedTriple::new(
            "ex:alice".to_string(),
            "ex:knows".to_string(),
            "ex:bob".to_string(),
        );

        let metadata = StatementMetadata::new(qt.clone())
            .with_confidence(0.95)
            .with_source("http://example.org/source1".to_string())
            .with_generated_by("http://example.org/rule42".to_string());

        assert_eq!(metadata.statement, qt);
        assert_eq!(metadata.confidence, Some(0.95));
        assert_eq!(
            metadata.source,
            Some("http://example.org/source1".to_string())
        );
        assert_eq!(
            metadata.generated_by,
            Some("http://example.org/rule42".to_string())
        );
    }

    #[test]
    fn test_metadata_to_turtle() {
        let qt = QuotedTriple::new(
            "<http://example.org/alice>".to_string(),
            "<http://example.org/knows>".to_string(),
            "<http://example.org/bob>".to_string(),
        );

        let metadata = StatementMetadata::new(qt)
            .with_confidence(0.9)
            .with_source("http://example.org/source1".to_string());

        let turtle = metadata.to_turtle();
        assert!(turtle.contains("confidence"));
        assert!(turtle.contains("0.9"));
        assert!(turtle.contains("hadPrimarySource"));
    }

    #[test]
    fn test_provenance_store_creation() {
        let store = RdfStarProvenanceStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_provenance_store_add_metadata() {
        let mut store = RdfStarProvenanceStore::new();

        let qt = QuotedTriple::new(
            "ex:alice".to_string(),
            "ex:knows".to_string(),
            "ex:bob".to_string(),
        );

        let metadata = StatementMetadata::new(qt.clone()).with_confidence(0.95);

        store.add_metadata(metadata);

        assert_eq!(store.len(), 1);
        assert!(store.get_metadata(&qt).is_some());
    }

    #[test]
    fn test_provenance_store_get_by_confidence() {
        let mut store = RdfStarProvenanceStore::new();

        // Add statements with different confidence scores
        for (i, conf) in [0.5, 0.7, 0.9, 0.95, 0.99].iter().enumerate() {
            let qt = QuotedTriple::new(
                format!("ex:subject{}", i),
                "ex:predicate".to_string(),
                format!("ex:object{}", i),
            );
            let metadata = StatementMetadata::new(qt).with_confidence(*conf);
            store.add_metadata(metadata);
        }

        let high_conf = store.get_by_min_confidence(0.9);
        assert_eq!(high_conf.len(), 3); // 0.9, 0.95, 0.99

        let very_high_conf = store.get_by_min_confidence(0.95);
        assert_eq!(very_high_conf.len(), 2); // 0.95, 0.99
    }

    #[test]
    fn test_provenance_store_get_by_predicate() {
        let mut store = RdfStarProvenanceStore::new();

        // Add statements with different predicates
        for pred in ["ex:knows", "ex:likes", "ex:knows"].iter() {
            let qt = QuotedTriple::new(
                "ex:alice".to_string(),
                pred.to_string(),
                "ex:bob".to_string(),
            );
            let metadata = StatementMetadata::new(qt);
            store.add_metadata(metadata);
        }

        let knows_stmts = store.get_by_predicate("ex:knows");
        assert_eq!(knows_stmts.len(), 2);

        let likes_stmts = store.get_by_predicate("ex:likes");
        assert_eq!(likes_stmts.len(), 1);
    }

    #[test]
    fn test_provenance_store_get_by_source() {
        let mut store = RdfStarProvenanceStore::new();

        // Add statements from different sources
        for (i, source) in ["source1", "source2", "source1"].iter().enumerate() {
            let qt = QuotedTriple::new(
                format!("ex:subject{}", i),
                "ex:predicate".to_string(),
                format!("ex:object{}", i),
            );
            let metadata = StatementMetadata::new(qt).with_source(source.to_string());
            store.add_metadata(metadata);
        }

        let source1_stmts = store.get_by_source("source1");
        assert_eq!(source1_stmts.len(), 2);

        let source2_stmts = store.get_by_source("source2");
        assert_eq!(source2_stmts.len(), 1);
    }

    #[test]
    fn test_provenance_store_get_by_rule() {
        let mut store = RdfStarProvenanceStore::new();

        // Add statements generated by different rules
        for (i, rule) in ["rule1", "rule2", "rule1"].iter().enumerate() {
            let qt = QuotedTriple::new(
                format!("ex:subject{}", i),
                "ex:predicate".to_string(),
                format!("ex:object{}", i),
            );
            let metadata = StatementMetadata::new(qt).with_rule_id(rule.to_string());
            store.add_metadata(metadata);
        }

        let rule1_stmts = store.get_by_rule("rule1");
        assert_eq!(rule1_stmts.len(), 2);

        let rule2_stmts = store.get_by_rule("rule2");
        assert_eq!(rule2_stmts.len(), 1);
    }

    #[test]
    fn test_provenance_store_to_turtle() {
        let mut store = RdfStarProvenanceStore::new();

        let qt = QuotedTriple::new(
            "<http://example.org/alice>".to_string(),
            "<http://example.org/knows>".to_string(),
            "<http://example.org/bob>".to_string(),
        );

        let metadata = StatementMetadata::new(qt)
            .with_confidence(0.95)
            .with_generated_by("http://example.org/rule42".to_string());

        store.add_metadata(metadata);

        let turtle = store.to_turtle();
        assert!(turtle.contains("@prefix prov"));
        assert!(turtle.contains("confidence"));
        assert!(turtle.contains("wasGeneratedBy"));
    }

    #[test]
    fn test_provenance_store_to_json() {
        let mut store = RdfStarProvenanceStore::new();

        let qt = QuotedTriple::new(
            "ex:alice".to_string(),
            "ex:knows".to_string(),
            "ex:bob".to_string(),
        );

        let metadata = StatementMetadata::new(qt).with_confidence(0.95);
        store.add_metadata(metadata);

        let json = store.to_json().unwrap();
        assert!(json.contains("alice"));
        assert!(json.contains("knows"));
        assert!(json.contains("0.95"));
    }

    #[test]
    fn test_provenance_store_statistics() {
        let mut store = RdfStarProvenanceStore::new();

        // Add various statements
        for i in 0..10 {
            let qt = QuotedTriple::new(
                format!("ex:subject{}", i),
                "ex:predicate".to_string(),
                format!("ex:object{}", i),
            );

            let mut metadata = StatementMetadata::new(qt);

            if i < 5 {
                metadata = metadata.with_confidence(0.9);
            }
            if i < 3 {
                metadata = metadata.with_source(format!("source{}", i % 2));
            }
            if i < 7 {
                metadata = metadata.with_rule_id(format!("rule{}", i % 3));
            }

            store.add_metadata(metadata);
        }

        let stats = store.get_stats();
        assert_eq!(stats.total_statements, 10);
        assert_eq!(stats.with_confidence, 5);
        assert_eq!(stats.with_source, 3);
        assert_eq!(stats.with_rule, 7);
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = MetadataBuilder::for_triple(
            "ex:alice".to_string(),
            "ex:knows".to_string(),
            "ex:bob".to_string(),
        )
        .confidence(0.95)
        .source("http://example.org/source1".to_string())
        .generated_by("http://example.org/inference".to_string())
        .rule_id("rule42".to_string())
        .custom("custom_key".to_string(), "custom_value".to_string())
        .build();

        assert_eq!(metadata.confidence, Some(0.95));
        assert_eq!(
            metadata.source,
            Some("http://example.org/source1".to_string())
        );
        assert_eq!(
            metadata.generated_by,
            Some("http://example.org/inference".to_string())
        );
        assert_eq!(metadata.rule_id, Some("rule42".to_string()));
        assert_eq!(
            metadata.custom.get("custom_key"),
            Some(&"custom_value".to_string())
        );
    }

    #[test]
    fn test_metadata_builder_with_quoted_triple() {
        let qt = QuotedTriple::new(
            "ex:alice".to_string(),
            "ex:knows".to_string(),
            "ex:bob".to_string(),
        );

        let metadata = MetadataBuilder::for_quoted_triple(qt.clone())
            .confidence(0.9)
            .build();

        assert_eq!(metadata.statement, qt);
        assert_eq!(metadata.confidence, Some(0.9));
    }

    #[test]
    fn test_confidence_clamping() {
        // Confidence should be clamped to [0.0, 1.0]
        let metadata1 =
            MetadataBuilder::for_triple("ex:a".to_string(), "ex:b".to_string(), "ex:c".to_string())
                .confidence(1.5) // Above 1.0
                .build();

        assert_eq!(metadata1.confidence, Some(1.0));

        let metadata2 =
            MetadataBuilder::for_triple("ex:a".to_string(), "ex:b".to_string(), "ex:c".to_string())
                .confidence(-0.5) // Below 0.0
                .build();

        assert_eq!(metadata2.confidence, Some(0.0));
    }

    #[test]
    fn test_provenance_store_clear() {
        let mut store = RdfStarProvenanceStore::new();

        // Add some metadata
        for i in 0..5 {
            let qt = QuotedTriple::new(
                format!("ex:s{}", i),
                "ex:p".to_string(),
                format!("ex:o{}", i),
            );
            store.add_metadata(StatementMetadata::new(qt));
        }

        assert_eq!(store.len(), 5);

        store.clear();

        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_complex_provenance_scenario() {
        let mut store = RdfStarProvenanceStore::new();

        // Simulate a complex inference scenario
        // Rule 1 infers 3 triples
        for i in 0..3 {
            let qt = QuotedTriple::new(
                format!("ex:person{}", i),
                "ex:type".to_string(),
                "ex:Human".to_string(),
            );
            let metadata = StatementMetadata::new(qt)
                .with_confidence(0.99)
                .with_rule_id("rule1".to_string())
                .with_source("ontology1".to_string());
            store.add_metadata(metadata);
        }

        // Rule 2 infers 2 triples with lower confidence
        for i in 0..2 {
            let qt = QuotedTriple::new(
                format!("ex:person{}", i),
                "ex:likesCoffee".to_string(),
                "\"true\"".to_string(),
            );
            let metadata = StatementMetadata::new(qt)
                .with_confidence(0.7)
                .with_rule_id("rule2".to_string())
                .with_source("survey_data".to_string());
            store.add_metadata(metadata);
        }

        // Verify different queries
        assert_eq!(store.len(), 5);

        let rule1_inferences = store.get_by_rule("rule1");
        assert_eq!(rule1_inferences.len(), 3);

        let high_confidence = store.get_by_min_confidence(0.9);
        assert_eq!(high_confidence.len(), 3);

        let from_ontology = store.get_by_source("ontology1");
        assert_eq!(from_ontology.len(), 3);

        let type_statements = store.get_by_predicate("ex:type");
        assert_eq!(type_statements.len(), 3);

        // Check statistics
        let stats = store.get_stats();
        assert_eq!(stats.total_statements, 5);
        assert_eq!(stats.unique_rules, 2);
        assert_eq!(stats.unique_sources, 2);
    }
}
