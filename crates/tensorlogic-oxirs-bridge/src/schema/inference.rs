//! RDFS inference engine for materializing entailed triples.
//!
//! This module implements RDFS entailment rules to infer new triples from
//! existing RDF data. The inference engine supports:
//! - rdfs:subClassOf transitivity
//! - Property domain/range inheritance
//! - Type propagation through class hierarchies
//! - Property inheritance through property hierarchies

use anyhow::Result;
use oxrdf::{Graph, NamedNode, NamedOrBlankNodeRef, TermRef, Triple};
use std::collections::{HashMap, HashSet};

use super::SchemaAnalyzer;

/// RDFS inference engine that materializes entailed triples
pub struct RdfsInferenceEngine {
    /// Original graph
    pub graph: Graph,
    /// Inferred triples (materialized)
    pub inferred: Graph,
    /// Subclass hierarchy cache
    subclass_hierarchy: HashMap<String, HashSet<String>>,
    /// Subproperty hierarchy cache
    subproperty_hierarchy: HashMap<String, HashSet<String>>,
    /// Property domains cache
    property_domains: HashMap<String, HashSet<String>>,
    /// Property ranges cache
    property_ranges: HashMap<String, HashSet<String>>,
}

impl RdfsInferenceEngine {
    /// Create a new inference engine from a graph
    pub fn new(graph: Graph) -> Self {
        RdfsInferenceEngine {
            graph,
            inferred: Graph::new(),
            subclass_hierarchy: HashMap::new(),
            subproperty_hierarchy: HashMap::new(),
            property_domains: HashMap::new(),
            property_ranges: HashMap::new(),
        }
    }

    /// Run all RDFS inference rules and materialize entailed triples
    pub fn materialize(&mut self) -> Result<()> {
        // Build hierarchy caches first
        self.build_subclass_hierarchy()?;
        self.build_subproperty_hierarchy()?;
        self.build_property_constraints()?;

        // Apply inference rules iteratively until fixpoint
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            // Rule: rdfs2 (domain type inference)
            // If (x, p, y) and (p, rdfs:domain, D) then (x, rdf:type, D)
            if self.apply_domain_inference()? {
                changed = true;
            }

            // Rule: rdfs3 (range type inference)
            // If (x, p, y) and (p, rdfs:range, R) then (y, rdf:type, R)
            if self.apply_range_inference()? {
                changed = true;
            }

            // Rule: rdfs9 (subclass inheritance)
            // If (x, rdf:type, C) and (C, rdfs:subClassOf, D) then (x, rdf:type, D)
            if self.apply_subclass_inference()? {
                changed = true;
            }

            // Rule: rdfs7 (subproperty inference)
            // If (x, p, y) and (p, rdfs:subPropertyOf, q) then (x, q, y)
            if self.apply_subproperty_inference()? {
                changed = true;
            }
        }

        Ok(())
    }

    /// Get the complete graph (original + inferred triples)
    pub fn get_complete_graph(&self) -> Graph {
        let mut complete = self.graph.clone();
        for triple in self.inferred.iter() {
            complete.insert(triple);
        }
        complete
    }

    /// Get only the inferred triples
    pub fn get_inferred_triples(&self) -> &Graph {
        &self.inferred
    }

    /// Build transitive closure of subclass hierarchy
    fn build_subclass_hierarchy(&mut self) -> Result<()> {
        let rdfs_subclass = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf")?;

        // First, collect direct subclass relationships
        let mut direct_subclasses: HashMap<String, HashSet<String>> = HashMap::new();

        for triple in self.graph.iter() {
            if triple.predicate == rdfs_subclass.as_ref() {
                if let (NamedOrBlankNodeRef::NamedNode(subj), TermRef::NamedNode(obj)) =
                    (triple.subject, triple.object)
                {
                    direct_subclasses
                        .entry(subj.as_str().to_string())
                        .or_default()
                        .insert(obj.as_str().to_string());
                }
            }
        }

        // Compute transitive closure
        for class in direct_subclasses.keys() {
            let mut visited = HashSet::new();
            let mut stack = vec![class.clone()];

            while let Some(current) = stack.pop() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current.clone());

                if let Some(parents) = direct_subclasses.get(&current) {
                    for parent in parents {
                        stack.push(parent.clone());
                    }
                }
            }

            visited.remove(class);
            self.subclass_hierarchy.insert(class.clone(), visited);
        }

        Ok(())
    }

    /// Build transitive closure of subproperty hierarchy
    fn build_subproperty_hierarchy(&mut self) -> Result<()> {
        let rdfs_subproperty =
            NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")?;

        // First, collect direct subproperty relationships
        let mut direct_subprops: HashMap<String, HashSet<String>> = HashMap::new();

        for triple in self.graph.iter() {
            if triple.predicate == rdfs_subproperty.as_ref() {
                if let (NamedOrBlankNodeRef::NamedNode(subj), TermRef::NamedNode(obj)) =
                    (triple.subject, triple.object)
                {
                    direct_subprops
                        .entry(subj.as_str().to_string())
                        .or_default()
                        .insert(obj.as_str().to_string());
                }
            }
        }

        // Compute transitive closure
        for prop in direct_subprops.keys() {
            let mut visited = HashSet::new();
            let mut stack = vec![prop.clone()];

            while let Some(current) = stack.pop() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current.clone());

                if let Some(parents) = direct_subprops.get(&current) {
                    for parent in parents {
                        stack.push(parent.clone());
                    }
                }
            }

            visited.remove(prop);
            self.subproperty_hierarchy.insert(prop.clone(), visited);
        }

        Ok(())
    }

    /// Build property domain and range constraints
    fn build_property_constraints(&mut self) -> Result<()> {
        let rdfs_domain = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#domain")?;
        let rdfs_range = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#range")?;

        for triple in self.graph.iter() {
            if triple.predicate == rdfs_domain.as_ref() {
                if let (NamedOrBlankNodeRef::NamedNode(subj), TermRef::NamedNode(obj)) =
                    (triple.subject, triple.object)
                {
                    self.property_domains
                        .entry(subj.as_str().to_string())
                        .or_default()
                        .insert(obj.as_str().to_string());
                }
            } else if triple.predicate == rdfs_range.as_ref() {
                if let (NamedOrBlankNodeRef::NamedNode(subj), TermRef::NamedNode(obj)) =
                    (triple.subject, triple.object)
                {
                    self.property_ranges
                        .entry(subj.as_str().to_string())
                        .or_default()
                        .insert(obj.as_str().to_string());
                }
            }
        }

        Ok(())
    }

    /// Apply domain inference rule (rdfs2)
    fn apply_domain_inference(&mut self) -> Result<bool> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
        let mut new_triples = Vec::new();

        for triple in self.graph.iter() {
            let pred_str = triple.predicate.as_str().to_string();

            if let Some(domains) = self.property_domains.get(&pred_str) {
                if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                    for domain in domains {
                        let domain_node = NamedNode::new(domain.clone())?;
                        let type_triple =
                            Triple::new(subj.into_owned(), rdf_type.clone(), domain_node.clone());

                        // Check if triple already exists
                        if !self.graph.contains(&type_triple)
                            && !self.inferred.contains(&type_triple)
                        {
                            new_triples.push(type_triple);
                        }
                    }
                }
            }
        }

        let changed = !new_triples.is_empty();
        for triple in new_triples {
            self.inferred.insert(&triple);
            self.graph.insert(&triple);
        }

        Ok(changed)
    }

    /// Apply range inference rule (rdfs3)
    fn apply_range_inference(&mut self) -> Result<bool> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
        let mut new_triples = Vec::new();

        for triple in self.graph.iter() {
            let pred_str = triple.predicate.as_str().to_string();

            if let Some(ranges) = self.property_ranges.get(&pred_str) {
                if let TermRef::NamedNode(obj) = triple.object {
                    for range in ranges {
                        let range_node = NamedNode::new(range.clone())?;
                        let type_triple =
                            Triple::new(obj.into_owned(), rdf_type.clone(), range_node.clone());

                        if !self.graph.contains(&type_triple)
                            && !self.inferred.contains(&type_triple)
                        {
                            new_triples.push(type_triple);
                        }
                    }
                }
            }
        }

        let changed = !new_triples.is_empty();
        for triple in new_triples {
            self.inferred.insert(&triple);
            self.graph.insert(&triple);
        }

        Ok(changed)
    }

    /// Apply subclass inference rule (rdfs9)
    fn apply_subclass_inference(&mut self) -> Result<bool> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
        let mut new_triples = Vec::new();

        for triple in self.graph.iter() {
            if triple.predicate == rdf_type.as_ref() {
                if let (NamedOrBlankNodeRef::NamedNode(subj), TermRef::NamedNode(class_obj)) =
                    (triple.subject, triple.object)
                {
                    let class_str = class_obj.as_str().to_string();

                    // Get all superclasses
                    if let Some(superclasses) = self.subclass_hierarchy.get(&class_str) {
                        for superclass in superclasses {
                            let superclass_node = NamedNode::new(superclass.clone())?;
                            let type_triple = Triple::new(
                                subj.into_owned(),
                                rdf_type.clone(),
                                superclass_node.clone(),
                            );

                            if !self.graph.contains(&type_triple)
                                && !self.inferred.contains(&type_triple)
                            {
                                new_triples.push(type_triple);
                            }
                        }
                    }
                }
            }
        }

        let changed = !new_triples.is_empty();
        for triple in new_triples {
            self.inferred.insert(&triple);
            self.graph.insert(&triple);
        }

        Ok(changed)
    }

    /// Apply subproperty inference rule (rdfs7)
    fn apply_subproperty_inference(&mut self) -> Result<bool> {
        let mut new_triples = Vec::new();

        for triple in self.graph.iter() {
            let pred_str = triple.predicate.as_str().to_string();

            if let Some(superprops) = self.subproperty_hierarchy.get(&pred_str) {
                for superprop in superprops {
                    let superprop_node = NamedNode::new(superprop.clone())?;
                    let new_triple = Triple::new(
                        triple.subject.into_owned(),
                        superprop_node.clone(),
                        triple.object.into_owned(),
                    );

                    if !self.graph.contains(&new_triple) && !self.inferred.contains(&new_triple) {
                        new_triples.push(new_triple);
                    }
                }
            }
        }

        let changed = !new_triples.is_empty();
        for triple in new_triples {
            self.inferred.insert(&triple);
            self.graph.insert(&triple);
        }

        Ok(changed)
    }

    /// Get all superclasses of a given class (including transitive)
    pub fn get_all_superclasses(&self, class_iri: &str) -> HashSet<String> {
        self.subclass_hierarchy
            .get(class_iri)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all superproperties of a given property (including transitive)
    pub fn get_all_superproperties(&self, prop_iri: &str) -> HashSet<String> {
        self.subproperty_hierarchy
            .get(prop_iri)
            .cloned()
            .unwrap_or_default()
    }

    /// Check if class A is a subclass of class B (direct or transitive)
    pub fn is_subclass_of(&self, class_a: &str, class_b: &str) -> bool {
        if let Some(superclasses) = self.subclass_hierarchy.get(class_a) {
            superclasses.contains(class_b)
        } else {
            false
        }
    }

    /// Check if property A is a subproperty of property B (direct or transitive)
    pub fn is_subproperty_of(&self, prop_a: &str, prop_b: &str) -> bool {
        if let Some(superprops) = self.subproperty_hierarchy.get(prop_a) {
            superprops.contains(prop_b)
        } else {
            false
        }
    }

    /// Get statistics about inferred triples
    pub fn get_inference_stats(&self) -> InferenceStats {
        InferenceStats {
            original_triples: self.graph.len() - self.inferred.len(),
            inferred_triples: self.inferred.len(),
            total_triples: self.graph.len(),
            subclass_relations: self
                .subclass_hierarchy
                .values()
                .map(|s| s.len())
                .sum::<usize>(),
            subproperty_relations: self
                .subproperty_hierarchy
                .values()
                .map(|s| s.len())
                .sum::<usize>(),
        }
    }
}

/// Statistics about RDFS inference
#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub original_triples: usize,
    pub inferred_triples: usize,
    pub total_triples: usize,
    pub subclass_relations: usize,
    pub subproperty_relations: usize,
}

impl SchemaAnalyzer {
    /// Create an RDFS inference engine from this analyzer's graph
    pub fn create_inference_engine(&self) -> RdfsInferenceEngine {
        RdfsInferenceEngine::new(self.graph.clone())
    }

    /// Run RDFS inference and return the materialized graph
    pub fn materialize_rdfs_entailments(&self) -> Result<Graph> {
        let mut engine = self.create_inference_engine();
        engine.materialize()?;
        Ok(engine.get_complete_graph())
    }
}
