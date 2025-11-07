//! Example: OWL and RDFS inference with TensorLogic
//!
//! This example demonstrates:
//! - Loading OWL ontologies with class hierarchies and property characteristics
//! - Running RDFS inference to materialize entailed triples
//! - Extracting OWL restrictions and converting them to TensorLogic rules
//! - Using the inference engine's query API

use anyhow::Result;
use tensorlogic_oxirs_bridge::{RdfsInferenceEngine, SchemaAnalyzer};

fn main() -> Result<()> {
    println!("=== OWL and RDFS Inference Example ===\n");

    // Example OWL ontology for a university domain
    let university_ontology = r#"
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix uni: <http://example.org/university#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        # ============ Classes ============

        uni:Person a owl:Class ;
            rdfs:label "Person" ;
            rdfs:comment "A human being" .

        uni:Student a owl:Class ;
            rdfs:subClassOf uni:Person ;
            rdfs:label "Student" ;
            rdfs:comment "A person enrolled in courses" .

        uni:GraduateStudent a owl:Class ;
            rdfs:subClassOf uni:Student ;
            rdfs:label "Graduate Student" ;
            rdfs:comment "A student in a graduate program" .

        uni:Faculty a owl:Class ;
            rdfs:subClassOf uni:Person ;
            rdfs:label "Faculty" ;
            rdfs:comment "A teaching or research staff member" .

        uni:Professor a owl:Class ;
            rdfs:subClassOf uni:Faculty ;
            rdfs:label "Professor" ;
            rdfs:comment "A senior faculty member" .

        uni:Course a owl:Class ;
            rdfs:label "Course" ;
            rdfs:comment "An educational course" .

        uni:Department a owl:Class ;
            rdfs:label "Department" ;
            rdfs:comment "An academic department" .

        # Disjoint classes
        uni:Student owl:disjointWith uni:Faculty .

        # ============ Properties ============

        # teaches: Faculty → Course (functional)
        uni:teaches a owl:ObjectProperty, owl:FunctionalProperty ;
            rdfs:domain uni:Faculty ;
            rdfs:range uni:Course ;
            rdfs:label "teaches" .

        # enrolledIn: Student → Course
        uni:enrolledIn a owl:ObjectProperty ;
            rdfs:domain uni:Student ;
            rdfs:range uni:Course ;
            rdfs:label "enrolled in" .

        # advisedBy: GraduateStudent → Professor
        uni:advisedBy a owl:ObjectProperty ;
            rdfs:domain uni:GraduateStudent ;
            rdfs:range uni:Professor ;
            rdfs:label "advised by" .

        # supervises (inverse of advisedBy)
        uni:supervises a owl:ObjectProperty ;
            owl:inverseOf uni:advisedBy ;
            rdfs:label "supervises" .

        # worksIn: Faculty → Department
        uni:worksIn a owl:ObjectProperty ;
            rdfs:domain uni:Faculty ;
            rdfs:range uni:Department ;
            rdfs:label "works in" .

        # Property hierarchy
        uni:advisedBy rdfs:subPropertyOf uni:worksWithFaculty .
        uni:worksWithFaculty rdfs:subPropertyOf uni:relatedToPerson .

        # Symmetric property
        uni:collaboratesWith a owl:SymmetricProperty ;
            rdfs:domain uni:Faculty ;
            rdfs:range uni:Faculty ;
            rdfs:label "collaborates with" .

        # Transitive property
        uni:ancestorDepartmentOf a owl:TransitiveProperty ;
            rdfs:domain uni:Department ;
            rdfs:range uni:Department ;
            rdfs:label "ancestor department of" .

        # ============ OWL Restrictions ============

        # A Professor must teach at least one course
        uni:Professor rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty uni:teaches ;
            owl:minCardinality "1"^^xsd:nonNegativeInteger
        ] .

        # A GraduateStudent must be advised by some Professor
        uni:GraduateStudent rdfs:subClassOf [
            a owl:Restriction ;
            owl:onProperty uni:advisedBy ;
            owl:someValuesFrom uni:Professor
        ] .

        # ============ Instance Data ============

        uni:alice rdf:type uni:GraduateStudent ;
            uni:enrolledIn uni:cs101 ;
            uni:advisedBy uni:drSmith .

        uni:drSmith rdf:type uni:Professor ;
            uni:teaches uni:cs101 ;
            uni:worksIn uni:csDepartment .

        uni:cs101 rdf:type uni:Course .
        uni:csDepartment rdf:type uni:Department .
    "#;

    // 1. Load and analyze the ontology
    println!("1. Loading OWL ontology...");
    let mut analyzer = SchemaAnalyzer::new();
    analyzer.load_turtle(university_ontology)?;
    analyzer.analyze()?;

    println!("   Loaded {} classes", analyzer.classes.len());
    println!("   Loaded {} properties", analyzer.properties.len());

    // 2. Extract OWL-specific information
    println!("\n2. Extracting OWL features...");

    let owl_classes = analyzer.extract_owl_classes()?;
    println!("   Found {} OWL classes", owl_classes.len());

    // Show class hierarchy
    println!("\n   Class Hierarchy:");
    for owl_class in &owl_classes {
        if !owl_class.base.subclass_of.is_empty() {
            let local_name = extract_local_name(&owl_class.base.iri);
            println!("   - {} subclassOf:", local_name);
            for parent in &owl_class.base.subclass_of {
                println!("     → {}", extract_local_name(parent));
            }
        }
        if !owl_class.disjoint_with.is_empty() {
            let local_name = extract_local_name(&owl_class.base.iri);
            println!("   - {} disjointWith:", local_name);
            for disjoint in &owl_class.disjoint_with {
                println!("     → {}", extract_local_name(disjoint));
            }
        }
    }

    let owl_properties = analyzer.extract_owl_properties()?;
    println!("\n   Found {} OWL properties", owl_properties.len());

    // Show property characteristics
    println!("\n   Property Characteristics:");
    for prop in &owl_properties {
        let local_name = extract_local_name(&prop.base.iri);
        let mut characteristics = Vec::new();

        if prop.characteristics.functional {
            characteristics.push("Functional");
        }
        if prop.characteristics.symmetric {
            characteristics.push("Symmetric");
        }
        if prop.characteristics.transitive {
            characteristics.push("Transitive");
        }
        if prop.inverse_of.is_some() {
            characteristics.push("HasInverse");
        }

        if !characteristics.is_empty() {
            println!("   - {}: {}", local_name, characteristics.join(", "));
        }
    }

    // Show OWL restrictions
    println!("\n   OWL Restrictions:");
    for owl_class in &owl_classes {
        let class_node = oxrdf::NamedNode::new(owl_class.base.iri.clone())?;
        let restrictions = analyzer.extract_owl_restrictions(&class_node)?;

        if !restrictions.is_empty() {
            let local_name = extract_local_name(&owl_class.base.iri);
            println!("   - {} has restrictions:", local_name);
            for restriction in &restrictions {
                println!("     {:?}", restriction);
            }
        }
    }

    // 3. Run RDFS inference
    println!("\n3. Running RDFS inference...");
    let mut engine = RdfsInferenceEngine::new(analyzer.graph.clone());
    engine.materialize()?;

    let stats = engine.get_inference_stats();
    println!("   Original triples: {}", stats.original_triples);
    println!("   Inferred triples: {}", stats.inferred_triples);
    println!("   Total triples: {}", stats.total_triples);
    println!("   Subclass relations: {}", stats.subclass_relations);
    println!("   Subproperty relations: {}", stats.subproperty_relations);

    // 4. Query the inference engine
    println!("\n4. Querying inference results...");

    // Check class hierarchy
    let graduate_student = "http://example.org/university#GraduateStudent";
    let person = "http://example.org/university#Person";

    if engine.is_subclass_of(graduate_student, person) {
        println!("   ✓ GraduateStudent is a subclass of Person (transitive)");
    }

    let all_superclasses = engine.get_all_superclasses(graduate_student);
    println!("\n   All superclasses of GraduateStudent:");
    for superclass in &all_superclasses {
        println!("   - {}", extract_local_name(superclass));
    }

    // Check property hierarchy
    let advised_by = "http://example.org/university#advisedBy";
    let related_to_person = "http://example.org/university#relatedToPerson";

    if engine.is_subproperty_of(advised_by, related_to_person) {
        println!("\n   ✓ advisedBy is a subproperty of relatedToPerson (transitive)");
    }

    let all_superprops = engine.get_all_superproperties(advised_by);
    println!("\n   All superproperties of advisedBy:");
    for superprop in &all_superprops {
        println!("   - {}", extract_local_name(superprop));
    }

    // 5. Check inferred types
    println!("\n5. Checking inferred types...");

    let complete_graph = engine.get_complete_graph();
    let alice = oxrdf::NamedNode::new("http://example.org/university#alice")?;
    let rdf_type = oxrdf::NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;

    println!("\n   Alice's inferred types:");
    for triple in complete_graph.iter() {
        if let oxrdf::NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
            if subj == alice.as_ref() && triple.predicate == rdf_type.as_ref() {
                if let oxrdf::TermRef::NamedNode(obj) = triple.object {
                    println!("   - {}", extract_local_name(obj.as_str()));
                }
            }
        }
    }

    let dr_smith = oxrdf::NamedNode::new("http://example.org/university#drSmith")?;
    println!("\n   Dr. Smith's inferred types:");
    for triple in complete_graph.iter() {
        if let oxrdf::NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
            if subj == dr_smith.as_ref() && triple.predicate == rdf_type.as_ref() {
                if let oxrdf::TermRef::NamedNode(obj) = triple.object {
                    println!("   - {}", extract_local_name(obj.as_str()));
                }
            }
        }
    }

    println!("\n=== Inference Complete ===");

    Ok(())
}

fn extract_local_name(iri: &str) -> String {
    iri.split(['/', '#']).next_back().unwrap_or(iri).to_string()
}
