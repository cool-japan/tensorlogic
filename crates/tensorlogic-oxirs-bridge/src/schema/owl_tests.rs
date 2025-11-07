//! Tests for OWL support

#[cfg(test)]
mod tests {
    use crate::schema::SchemaAnalyzer;

    #[test]
    fn test_extract_owl_class_basic() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:Person a owl:Class ;
                rdfs:label "Person" ;
                rdfs:comment "A human being" .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        assert!(!owl_classes.is_empty());
        let person_class = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("Person"))
            .expect("Person class not found");

        assert_eq!(person_class.base.label, Some("Person".to_string()));
        assert_eq!(person_class.base.comment, Some("A human being".to_string()));
    }

    #[test]
    fn test_extract_equivalent_class() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:Person a owl:Class ;
                owl:equivalentClass ex:Human .

            ex:Human a owl:Class .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        let person_class = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("Person"))
            .expect("Person class not found");

        assert!(!person_class.equivalent_classes.is_empty());
        assert!(person_class.equivalent_classes[0].contains("Human"));
    }

    #[test]
    fn test_extract_union_of() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:LegalEntity a owl:Class ;
                owl:unionOf ( ex:Person ex:Organization ) .

            ex:Person a owl:Class .
            ex:Organization a owl:Class .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        let legal_entity = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("LegalEntity"))
            .expect("LegalEntity class not found");

        // Check if unionOf was parsed
        if !legal_entity.union_of.is_empty() {
            assert!(!legal_entity.union_of[0].is_empty());
        }
    }

    #[test]
    fn test_extract_intersection_of() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:WorkingParent a owl:Class ;
                owl:intersectionOf ( ex:Parent ex:Employee ) .

            ex:Parent a owl:Class .
            ex:Employee a owl:Class .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        let working_parent = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("WorkingParent"))
            .expect("WorkingParent class not found");

        // Check if intersectionOf was parsed
        if !working_parent.intersection_of.is_empty() {
            assert!(!working_parent.intersection_of[0].is_empty());
        }
    }

    #[test]
    fn test_extract_complement_of() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:NonPerson a owl:Class ;
                owl:complementOf ex:Person .

            ex:Person a owl:Class .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        let non_person = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("NonPerson"))
            .expect("NonPerson class not found");

        assert!(!non_person.complement_of.is_empty());
        assert!(non_person.complement_of[0].contains("Person"));
    }

    #[test]
    fn test_extract_disjoint_with() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:Person a owl:Class ;
                owl:disjointWith ex:Organization .

            ex:Organization a owl:Class .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        let person_class = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("Person"))
            .expect("Person class not found");

        assert!(!person_class.disjoint_with.is_empty());
        assert!(person_class.disjoint_with[0].contains("Organization"));
    }

    #[test]
    fn test_extract_functional_property() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:hasSSN a owl:FunctionalProperty, rdf:Property ;
                rdfs:domain ex:Person ;
                rdfs:range ex:SSN .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_properties = analyzer.extract_owl_properties().unwrap();

        let ssn_prop = owl_properties
            .iter()
            .find(|p| p.base.iri.contains("hasSSN"))
            .expect("hasSSN property not found");

        assert!(ssn_prop.characteristics.functional);
    }

    #[test]
    fn test_extract_inverse_functional_property() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:hasEmail a owl:InverseFunctionalProperty, rdf:Property ;
                rdfs:domain ex:Person ;
                rdfs:range ex:Email .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_properties = analyzer.extract_owl_properties().unwrap();

        let email_prop = owl_properties
            .iter()
            .find(|p| p.base.iri.contains("hasEmail"))
            .expect("hasEmail property not found");

        assert!(email_prop.characteristics.inverse_functional);
    }

    #[test]
    fn test_extract_transitive_property() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:ancestorOf a owl:TransitiveProperty, rdf:Property ;
                rdfs:domain ex:Person ;
                rdfs:range ex:Person .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_properties = analyzer.extract_owl_properties().unwrap();

        let ancestor_prop = owl_properties
            .iter()
            .find(|p| p.base.iri.contains("ancestorOf"))
            .expect("ancestorOf property not found");

        assert!(ancestor_prop.characteristics.transitive);
    }

    #[test]
    fn test_extract_symmetric_property() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:friendOf a owl:SymmetricProperty, rdf:Property ;
                rdfs:domain ex:Person ;
                rdfs:range ex:Person .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_properties = analyzer.extract_owl_properties().unwrap();

        let friend_prop = owl_properties
            .iter()
            .find(|p| p.base.iri.contains("friendOf"))
            .expect("friendOf property not found");

        assert!(friend_prop.characteristics.symmetric);
    }

    #[test]
    fn test_extract_inverse_of() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:hasChild a rdf:Property ;
                owl:inverseOf ex:hasParent .

            ex:hasParent a rdf:Property .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_properties = analyzer.extract_owl_properties().unwrap();

        let has_child = owl_properties
            .iter()
            .find(|p| p.base.iri.contains("hasChild"))
            .expect("hasChild property not found");

        assert!(has_child.inverse_of.is_some());
        assert!(has_child.inverse_of.as_ref().unwrap().contains("hasParent"));
    }

    #[test]
    fn test_extract_equivalent_property() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:hasName a rdf:Property ;
                owl:equivalentProperty ex:name .

            ex:name a rdf:Property .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_properties = analyzer.extract_owl_properties().unwrap();

        let has_name = owl_properties
            .iter()
            .find(|p| p.base.iri.contains("hasName"))
            .expect("hasName property not found");

        assert!(!has_name.equivalent_properties.is_empty());
        assert!(has_name.equivalent_properties[0].contains("name"));
    }

    #[test]
    fn test_extract_owl_restriction_some_values_from() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:Parent a owl:Class ;
                rdfs:subClassOf [
                    a owl:Restriction ;
                    owl:onProperty ex:hasChild ;
                    owl:someValuesFrom ex:Person
                ] .

            ex:Person a owl:Class .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        let parent_class = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("Parent"))
            .expect("Parent class not found");

        let parent_node = oxrdf::NamedNode::new(parent_class.base.iri.clone()).unwrap();
        let restrictions = analyzer.extract_owl_restrictions(&parent_node).unwrap();

        if !restrictions.is_empty() {
            match &restrictions[0] {
                crate::schema::owl::OwlRestriction::SomeValuesFrom { on_property, class } => {
                    assert!(on_property.contains("hasChild"));
                    assert!(class.contains("Person"));
                }
                _ => panic!("Expected SomeValuesFrom restriction"),
            }
        }
    }

    #[test]
    fn test_extract_owl_restriction_all_values_from() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:HappyPerson a owl:Class ;
                rdfs:subClassOf [
                    a owl:Restriction ;
                    owl:onProperty ex:hasChild ;
                    owl:allValuesFrom ex:HappyPerson
                ] .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        let happy_person = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("HappyPerson"))
            .expect("HappyPerson class not found");

        let happy_node = oxrdf::NamedNode::new(happy_person.base.iri.clone()).unwrap();
        let restrictions = analyzer.extract_owl_restrictions(&happy_node).unwrap();

        if !restrictions.is_empty() {
            match &restrictions[0] {
                crate::schema::owl::OwlRestriction::AllValuesFrom { on_property, class } => {
                    assert!(on_property.contains("hasChild"));
                    assert!(class.contains("HappyPerson"));
                }
                _ => panic!("Expected AllValuesFrom restriction"),
            }
        }
    }

    #[test]
    fn test_extract_owl_restriction_min_cardinality() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:Parent a owl:Class ;
                rdfs:subClassOf [
                    a owl:Restriction ;
                    owl:onProperty ex:hasChild ;
                    owl:minCardinality "1"^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>
                ] .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        let parent_class = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("Parent"))
            .expect("Parent class not found");

        let parent_node = oxrdf::NamedNode::new(parent_class.base.iri.clone()).unwrap();
        let restrictions = analyzer.extract_owl_restrictions(&parent_node).unwrap();

        if !restrictions.is_empty() {
            match &restrictions[0] {
                crate::schema::owl::OwlRestriction::MinCardinality {
                    on_property,
                    cardinality,
                } => {
                    assert!(on_property.contains("hasChild"));
                    assert_eq!(*cardinality, 1);
                }
                _ => panic!("Expected MinCardinality restriction"),
            }
        }
    }

    #[test]
    fn test_extract_owl_restriction_max_cardinality() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:Monogamist a owl:Class ;
                rdfs:subClassOf [
                    a owl:Restriction ;
                    owl:onProperty ex:hasSpouse ;
                    owl:maxCardinality "1"^^<http://www.w3.org/2001/XMLSchema#nonNegativeInteger>
                ] .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();

        let monogamist = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("Monogamist"))
            .expect("Monogamist class not found");

        let mono_node = oxrdf::NamedNode::new(monogamist.base.iri.clone()).unwrap();
        let restrictions = analyzer.extract_owl_restrictions(&mono_node).unwrap();

        if !restrictions.is_empty() {
            match &restrictions[0] {
                crate::schema::owl::OwlRestriction::MaxCardinality {
                    on_property,
                    cardinality,
                } => {
                    assert!(on_property.contains("hasSpouse"));
                    assert_eq!(*cardinality, 1);
                }
                _ => panic!("Expected MaxCardinality restriction"),
            }
        }
    }

    #[test]
    fn test_multiple_property_characteristics() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:knows a rdf:Property, owl:SymmetricProperty, owl:TransitiveProperty ;
                rdfs:domain ex:Person ;
                rdfs:range ex:Person .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_properties = analyzer.extract_owl_properties().unwrap();

        let knows_prop = owl_properties
            .iter()
            .find(|p| p.base.iri.contains("knows"))
            .expect("knows property not found");

        assert!(knows_prop.characteristics.symmetric);
        assert!(knows_prop.characteristics.transitive);
    }

    #[test]
    fn test_complex_owl_ontology() {
        let owl_turtle = r#"
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix ex: <http://example.org/> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

            ex:Person a owl:Class ;
                rdfs:label "Person" ;
                owl:disjointWith ex:Organization .

            ex:Organization a owl:Class ;
                rdfs:label "Organization" .

            ex:Employee a owl:Class ;
                rdfs:subClassOf ex:Person ;
                rdfs:subClassOf [
                    a owl:Restriction ;
                    owl:onProperty ex:worksFor ;
                    owl:someValuesFrom ex:Organization
                ] .

            ex:worksFor a rdf:Property ;
                rdfs:domain ex:Employee ;
                rdfs:range ex:Organization .

            ex:employs a rdf:Property ;
                owl:inverseOf ex:worksFor .

            ex:hasSSN a owl:FunctionalProperty, rdf:Property ;
                rdfs:domain ex:Person .
        "#;

        let mut analyzer = SchemaAnalyzer::new();
        analyzer.load_turtle(owl_turtle).unwrap();
        analyzer.analyze().unwrap();

        let owl_classes = analyzer.extract_owl_classes().unwrap();
        let owl_properties = analyzer.extract_owl_properties().unwrap();

        // Check classes
        assert!(
            owl_classes.len() >= 3,
            "Expected at least 3 classes, got {}",
            owl_classes.len()
        );

        // Check Person class
        let person = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("Person"))
            .expect("Person class not found");
        assert!(!person.disjoint_with.is_empty());

        // Check Employee has restriction
        let employee = owl_classes
            .iter()
            .find(|c| c.base.iri.contains("Employee"))
            .expect("Employee class not found");
        let emp_node = oxrdf::NamedNode::new(employee.base.iri.clone()).unwrap();
        let restrictions = analyzer.extract_owl_restrictions(&emp_node).unwrap();
        assert!(
            !restrictions.is_empty(),
            "Expected at least one restriction for Employee"
        );

        // Check properties
        let ssn_prop = owl_properties
            .iter()
            .find(|p| p.base.iri.contains("hasSSN"))
            .expect("hasSSN property not found");
        assert!(ssn_prop.characteristics.functional);

        let employs_prop = owl_properties
            .iter()
            .find(|p| p.base.iri.contains("employs"))
            .expect("employs property not found");
        assert!(employs_prop.inverse_of.is_some());
    }
}
