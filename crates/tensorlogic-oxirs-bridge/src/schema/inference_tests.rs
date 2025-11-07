//! Tests for RDFS inference engine

#[cfg(test)]
mod tests {
    use crate::schema::{inference::RdfsInferenceEngine, SchemaAnalyzer};
    use oxrdf::{NamedNode, Triple};

    #[test]
    fn test_subclass_transitivity() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:Mammal rdfs:subClassOf ex:Animal .
            ex:Dog rdfs:subClassOf ex:Mammal .
            ex:Poodle rdfs:subClassOf ex:Dog .

            ex:fluffy rdf:type ex:Poodle .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        let complete_graph = engine.get_complete_graph();

        // Check that fluffy is inferred to be an Animal (transitive closure)
        let fluffy = NamedNode::new("http://example.org/fluffy").unwrap();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let animal = NamedNode::new("http://example.org/Animal").unwrap();

        let expected = Triple::new(fluffy.clone(), rdf_type.clone(), animal.clone());
        assert!(
            complete_graph.contains(&expected),
            "Expected fluffy to be inferred as an Animal"
        );

        // Check that fluffy is also a Mammal
        let mammal = NamedNode::new("http://example.org/Mammal").unwrap();
        let mammal_triple = Triple::new(fluffy.clone(), rdf_type.clone(), mammal.clone());
        assert!(
            complete_graph.contains(&mammal_triple),
            "Expected fluffy to be inferred as a Mammal"
        );

        // Check that fluffy is also a Dog
        let dog = NamedNode::new("http://example.org/Dog").unwrap();
        let dog_triple = Triple::new(fluffy.clone(), rdf_type.clone(), dog.clone());
        assert!(
            complete_graph.contains(&dog_triple),
            "Expected fluffy to be inferred as a Dog"
        );
    }

    #[test]
    fn test_domain_inference() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:hasFather rdfs:domain ex:Person .

            ex:john ex:hasFather ex:bob .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        let complete_graph = engine.get_complete_graph();

        // Check that john is inferred to be a Person
        let john = NamedNode::new("http://example.org/john").unwrap();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let person = NamedNode::new("http://example.org/Person").unwrap();

        let expected = Triple::new(john.clone(), rdf_type.clone(), person.clone());
        assert!(
            complete_graph.contains(&expected),
            "Expected john to be inferred as a Person via domain"
        );
    }

    #[test]
    fn test_range_inference() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:hasFather rdfs:range ex:Person .

            ex:john ex:hasFather ex:bob .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        let complete_graph = engine.get_complete_graph();

        // Check that bob is inferred to be a Person
        let bob = NamedNode::new("http://example.org/bob").unwrap();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let person = NamedNode::new("http://example.org/Person").unwrap();

        let expected = Triple::new(bob.clone(), rdf_type.clone(), person.clone());
        assert!(
            complete_graph.contains(&expected),
            "Expected bob to be inferred as a Person via range"
        );
    }

    #[test]
    fn test_subproperty_inference() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:hasFather rdfs:subPropertyOf ex:hasParent .
            ex:hasParent rdfs:subPropertyOf ex:hasAncestor .

            ex:john ex:hasFather ex:bob .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        let complete_graph = engine.get_complete_graph();

        // Check that john hasParent bob is inferred
        let john = NamedNode::new("http://example.org/john").unwrap();
        let has_parent = NamedNode::new("http://example.org/hasParent").unwrap();
        let bob = NamedNode::new("http://example.org/bob").unwrap();

        let parent_triple = Triple::new(john.clone(), has_parent.clone(), bob.clone());
        assert!(
            complete_graph.contains(&parent_triple),
            "Expected john hasParent bob to be inferred"
        );

        // Check that john hasAncestor bob is inferred (transitive)
        let has_ancestor = NamedNode::new("http://example.org/hasAncestor").unwrap();
        let ancestor_triple = Triple::new(john.clone(), has_ancestor.clone(), bob.clone());
        assert!(
            complete_graph.contains(&ancestor_triple),
            "Expected john hasAncestor bob to be inferred transitively"
        );
    }

    #[test]
    fn test_combined_inference() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:Employee rdfs:subClassOf ex:Person .
            ex:Person rdfs:subClassOf ex:LegalEntity .

            ex:worksFor rdfs:domain ex:Employee .
            ex:worksFor rdfs:range ex:Organization .

            ex:john ex:worksFor ex:acme .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        let complete_graph = engine.get_complete_graph();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();

        // Check domain inference: john is an Employee
        let john = NamedNode::new("http://example.org/john").unwrap();
        let employee = NamedNode::new("http://example.org/Employee").unwrap();
        let employee_triple = Triple::new(john.clone(), rdf_type.clone(), employee.clone());
        assert!(
            complete_graph.contains(&employee_triple),
            "Expected john to be inferred as an Employee"
        );

        // Check subclass inference: john is a Person
        let person = NamedNode::new("http://example.org/Person").unwrap();
        let person_triple = Triple::new(john.clone(), rdf_type.clone(), person.clone());
        assert!(
            complete_graph.contains(&person_triple),
            "Expected john to be inferred as a Person"
        );

        // Check transitive subclass: john is a LegalEntity
        let legal_entity = NamedNode::new("http://example.org/LegalEntity").unwrap();
        let entity_triple = Triple::new(john.clone(), rdf_type.clone(), legal_entity.clone());
        assert!(
            complete_graph.contains(&entity_triple),
            "Expected john to be inferred as a LegalEntity"
        );

        // Check range inference: acme is an Organization
        let acme = NamedNode::new("http://example.org/acme").unwrap();
        let organization = NamedNode::new("http://example.org/Organization").unwrap();
        let org_triple = Triple::new(acme.clone(), rdf_type.clone(), organization.clone());
        assert!(
            complete_graph.contains(&org_triple),
            "Expected acme to be inferred as an Organization"
        );
    }

    #[test]
    fn test_is_subclass_of() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Poodle rdfs:subClassOf ex:Dog .
            ex:Dog rdfs:subClassOf ex:Mammal .
            ex:Mammal rdfs:subClassOf ex:Animal .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        // Direct subclass
        assert!(engine.is_subclass_of("http://example.org/Poodle", "http://example.org/Dog"));

        // Transitive subclass
        assert!(engine.is_subclass_of("http://example.org/Poodle", "http://example.org/Mammal"));
        assert!(engine.is_subclass_of("http://example.org/Poodle", "http://example.org/Animal"));

        // Not a subclass
        assert!(!engine.is_subclass_of("http://example.org/Animal", "http://example.org/Poodle"));
    }

    #[test]
    fn test_is_subproperty_of() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:hasMother rdfs:subPropertyOf ex:hasParent .
            ex:hasParent rdfs:subPropertyOf ex:hasAncestor .
            ex:hasAncestor rdfs:subPropertyOf ex:hasRelative .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        // Direct subproperty
        assert!(engine.is_subproperty_of(
            "http://example.org/hasMother",
            "http://example.org/hasParent"
        ));

        // Transitive subproperty
        assert!(engine.is_subproperty_of(
            "http://example.org/hasMother",
            "http://example.org/hasAncestor"
        ));
        assert!(engine.is_subproperty_of(
            "http://example.org/hasMother",
            "http://example.org/hasRelative"
        ));

        // Not a subproperty
        assert!(!engine.is_subproperty_of(
            "http://example.org/hasRelative",
            "http://example.org/hasMother"
        ));
    }

    #[test]
    fn test_get_all_superclasses() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:Poodle rdfs:subClassOf ex:Dog .
            ex:Dog rdfs:subClassOf ex:Mammal .
            ex:Mammal rdfs:subClassOf ex:Animal .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        let superclasses = engine.get_all_superclasses("http://example.org/Poodle");

        assert!(superclasses.contains("http://example.org/Dog"));
        assert!(superclasses.contains("http://example.org/Mammal"));
        assert!(superclasses.contains("http://example.org/Animal"));
        assert_eq!(superclasses.len(), 3);
    }

    #[test]
    fn test_get_all_superproperties() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .

            ex:hasMother rdfs:subPropertyOf ex:hasParent .
            ex:hasParent rdfs:subPropertyOf ex:hasAncestor .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        let superprops = engine.get_all_superproperties("http://example.org/hasMother");

        assert!(superprops.contains("http://example.org/hasParent"));
        assert!(superprops.contains("http://example.org/hasAncestor"));
        assert_eq!(superprops.len(), 2);
    }

    #[test]
    fn test_inference_stats() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:Dog rdfs:subClassOf ex:Mammal .
            ex:Mammal rdfs:subClassOf ex:Animal .

            ex:hasParent rdfs:subPropertyOf ex:hasAncestor .

            ex:hasFather rdfs:domain ex:Person .
            ex:john ex:hasFather ex:bob .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let original_count = analyzer.graph.len();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        let stats = engine.get_inference_stats();

        assert_eq!(stats.original_triples, original_count);
        assert!(stats.inferred_triples > 0, "Expected some inferred triples");
        assert_eq!(
            stats.total_triples,
            stats.original_triples + stats.inferred_triples
        );
        assert!(stats.subclass_relations > 0);
        assert!(stats.subproperty_relations > 0);
    }

    #[test]
    fn test_complex_hierarchy_with_multiple_inheritance() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:FlyingMammal rdfs:subClassOf ex:Mammal .
            ex:FlyingMammal rdfs:subClassOf ex:FlyingAnimal .
            ex:Mammal rdfs:subClassOf ex:Animal .
            ex:FlyingAnimal rdfs:subClassOf ex:Animal .
            ex:Bat rdfs:subClassOf ex:FlyingMammal .

            ex:stella rdf:type ex:Bat .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
        engine.materialize().unwrap();

        let complete_graph = engine.get_complete_graph();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let stella = NamedNode::new("http://example.org/stella").unwrap();

        // stella should be inferred to be all parent classes
        let types_to_check = vec!["FlyingMammal", "Mammal", "FlyingAnimal", "Animal"];

        for type_name in types_to_check {
            let type_node = NamedNode::new(format!("http://example.org/{}", type_name)).unwrap();
            let type_triple = Triple::new(stella.clone(), rdf_type.clone(), type_node.clone());
            assert!(
                complete_graph.contains(&type_triple),
                "Expected stella to be inferred as a {}",
                type_name
            );
        }
    }

    #[test]
    fn test_no_infinite_loop_on_circular_hierarchy() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:A rdfs:subClassOf ex:B .
            ex:B rdfs:subClassOf ex:A .

            ex:entity rdf:type ex:A .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());

        // Should not hang or panic
        let result = engine.materialize();
        assert!(
            result.is_ok(),
            "Inference should handle circular hierarchies"
        );

        let stats = engine.get_inference_stats();
        assert!(
            stats.total_triples < 1000,
            "Should not create excessive triples"
        );
    }

    #[test]
    fn test_integration_with_schema_analyzer() {
        let rdfs_turtle = r#"
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix ex: <http://example.org/> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:Employee rdfs:subClassOf ex:Person .
            ex:worksFor rdfs:domain ex:Employee .

            ex:john ex:worksFor ex:acme .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(rdfs_turtle).unwrap();

        let materialized = analyzer.materialize_rdfs_entailments().unwrap();

        // Check that john is inferred to be a Person
        let john = NamedNode::new("http://example.org/john").unwrap();
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap();
        let person = NamedNode::new("http://example.org/Person").unwrap();

        let person_triple = Triple::new(john.clone(), rdf_type.clone(), person.clone());
        assert!(
            materialized.contains(&person_triple),
            "Expected john to be inferred as a Person via SchemaAnalyzer integration"
        );
    }
}
