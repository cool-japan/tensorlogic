//! OWL (Web Ontology Language) support for schema analysis.
//!
//! This module extends the basic RDF schema analysis with OWL constructs including:
//! - OWL class definitions (owl:Class, owl:equivalentClass)
//! - Class combinations (owl:unionOf, owl:intersectionOf, owl:complementOf)
//! - Property restrictions (owl:Restriction, owl:onProperty, etc.)
//! - Property characteristics (owl:FunctionalProperty, owl:TransitiveProperty, etc.)

use anyhow::Result;
use oxrdf::{NamedNode, NamedOrBlankNodeRef, TermRef};

use super::{ClassInfo, PropertyInfo, SchemaAnalyzer};

/// OWL namespace
pub const OWL_NS: &str = "http://www.w3.org/2002/07/owl#";

/// Extended class information with OWL-specific constructs
#[derive(Clone, Debug)]
pub struct OwlClassInfo {
    pub base: ClassInfo,
    pub equivalent_classes: Vec<String>,
    pub union_of: Vec<Vec<String>>,
    pub intersection_of: Vec<Vec<String>>,
    pub complement_of: Vec<String>,
    pub disjoint_with: Vec<String>,
}

/// OWL property characteristics
#[derive(Clone, Debug, Default)]
pub struct OwlPropertyCharacteristics {
    pub functional: bool,
    pub inverse_functional: bool,
    pub transitive: bool,
    pub symmetric: bool,
    pub asymmetric: bool,
    pub reflexive: bool,
    pub irreflexive: bool,
}

/// Extended property information with OWL-specific constructs
#[derive(Clone, Debug)]
pub struct OwlPropertyInfo {
    pub base: PropertyInfo,
    pub characteristics: OwlPropertyCharacteristics,
    pub inverse_of: Option<String>,
    pub equivalent_properties: Vec<String>,
    pub sub_property_of: Vec<String>,
}

/// OWL restriction types
#[derive(Clone, Debug)]
pub enum OwlRestriction {
    SomeValuesFrom {
        on_property: String,
        class: String,
    },
    AllValuesFrom {
        on_property: String,
        class: String,
    },
    HasValue {
        on_property: String,
        value: String,
    },
    MinCardinality {
        on_property: String,
        cardinality: u32,
    },
    MaxCardinality {
        on_property: String,
        cardinality: u32,
    },
    ExactCardinality {
        on_property: String,
        cardinality: u32,
    },
    MinQualifiedCardinality {
        on_property: String,
        cardinality: u32,
        on_class: String,
    },
    MaxQualifiedCardinality {
        on_property: String,
        cardinality: u32,
        on_class: String,
    },
    ExactQualifiedCardinality {
        on_property: String,
        cardinality: u32,
        on_class: String,
    },
}

impl SchemaAnalyzer {
    /// Extract OWL class definitions from the graph
    pub fn extract_owl_classes(&self) -> Result<Vec<OwlClassInfo>> {
        let mut owl_classes = Vec::new();

        let owl_class = NamedNode::new(format!("{}Class", OWL_NS))?;
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;

        // Find all OWL classes
        for triple in self.graph.iter() {
            if triple.predicate == rdf_type.as_ref() {
                if let TermRef::NamedNode(obj) = triple.object {
                    if obj == owl_class.as_ref() {
                        if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                            let class_iri = subj.as_str().to_string();
                            let class_node = NamedNode::new(class_iri.clone())?;

                            let owl_info = OwlClassInfo {
                                base: ClassInfo {
                                    iri: class_iri.clone(),
                                    label: self.get_label(&class_node)?,
                                    comment: self.get_comment(&class_node)?,
                                    subclass_of: self.get_subclasses(&class_node)?,
                                },
                                equivalent_classes: self.get_equivalent_classes(&class_node)?,
                                union_of: self.get_union_of(&class_node)?,
                                intersection_of: self.get_intersection_of(&class_node)?,
                                complement_of: self.get_complement_of(&class_node)?,
                                disjoint_with: self.get_disjoint_with(&class_node)?,
                            };

                            owl_classes.push(owl_info);
                        }
                    }
                }
            }
        }

        // Also check for classes defined through RDFS that may have OWL properties
        for (class_iri, class_info) in &self.classes {
            let class_node = NamedNode::new(class_iri.clone())?;

            // Check if this class has OWL-specific properties
            let has_owl_properties = self.has_owl_class_properties(&class_node)?;

            if has_owl_properties {
                let owl_info = OwlClassInfo {
                    base: class_info.clone(),
                    equivalent_classes: self.get_equivalent_classes(&class_node)?,
                    union_of: self.get_union_of(&class_node)?,
                    intersection_of: self.get_intersection_of(&class_node)?,
                    complement_of: self.get_complement_of(&class_node)?,
                    disjoint_with: self.get_disjoint_with(&class_node)?,
                };

                // Only add if not already in the list
                if !owl_classes.iter().any(|c| c.base.iri == *class_iri) {
                    owl_classes.push(owl_info);
                }
            }
        }

        Ok(owl_classes)
    }

    /// Extract OWL property definitions from the graph
    pub fn extract_owl_properties(&self) -> Result<Vec<OwlPropertyInfo>> {
        let mut owl_properties = Vec::new();

        // Check each property for OWL characteristics
        for (prop_iri, prop_info) in &self.properties {
            let prop_node = NamedNode::new(prop_iri.clone())?;

            let owl_info = OwlPropertyInfo {
                base: prop_info.clone(),
                characteristics: self.get_property_characteristics(&prop_node)?,
                inverse_of: self.get_inverse_of(&prop_node)?,
                equivalent_properties: self.get_equivalent_properties(&prop_node)?,
                sub_property_of: self.get_sub_property_of(&prop_node)?,
            };

            owl_properties.push(owl_info);
        }

        // Also check for properties with explicit OWL types
        let property_types = vec![
            format!("{}ObjectProperty", OWL_NS),
            format!("{}DatatypeProperty", OWL_NS),
            format!("{}FunctionalProperty", OWL_NS),
            format!("{}InverseFunctionalProperty", OWL_NS),
            format!("{}TransitiveProperty", OWL_NS),
            format!("{}SymmetricProperty", OWL_NS),
            format!("{}AsymmetricProperty", OWL_NS),
            format!("{}ReflexiveProperty", OWL_NS),
            format!("{}IrreflexiveProperty", OWL_NS),
        ];

        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;

        for type_iri in property_types {
            let prop_type = NamedNode::new(type_iri)?;

            for triple in self.graph.iter() {
                if triple.predicate == rdf_type.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        if obj == prop_type.as_ref() {
                            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                                let prop_iri = subj.as_str().to_string();

                                // Skip if already processed
                                if owl_properties.iter().any(|p| p.base.iri == prop_iri) {
                                    continue;
                                }

                                let prop_node = NamedNode::new(prop_iri.clone())?;

                                let base = if let Some(existing) = self.properties.get(&prop_iri) {
                                    existing.clone()
                                } else {
                                    PropertyInfo {
                                        iri: prop_iri.clone(),
                                        label: self.get_label(&prop_node)?,
                                        comment: self.get_comment(&prop_node)?,
                                        domain: self.get_domain(&prop_node)?,
                                        range: self.get_range(&prop_node)?,
                                    }
                                };

                                let owl_info = OwlPropertyInfo {
                                    base,
                                    characteristics: self
                                        .get_property_characteristics(&prop_node)?,
                                    inverse_of: self.get_inverse_of(&prop_node)?,
                                    equivalent_properties: self
                                        .get_equivalent_properties(&prop_node)?,
                                    sub_property_of: self.get_sub_property_of(&prop_node)?,
                                };

                                owl_properties.push(owl_info);
                            }
                        }
                    }
                }
            }
        }

        Ok(owl_properties)
    }

    /// Extract OWL restrictions for a class
    pub fn extract_owl_restrictions(&self, class_node: &NamedNode) -> Result<Vec<OwlRestriction>> {
        let mut restrictions = Vec::new();

        let rdfs_subclass = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf")?;
        let owl_restriction = NamedNode::new(format!("{}Restriction", OWL_NS))?;
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;

        // Find restrictions in subClassOf relationships
        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == class_node.as_ref() && triple.predicate == rdfs_subclass.as_ref() {
                    if let TermRef::BlankNode(blank) = triple.object {
                        // Check if this blank node is a restriction
                        let is_restriction = self.graph.iter().any(|t| {
                            matches!(t.subject, NamedOrBlankNodeRef::BlankNode(b) if b.as_str() == blank.as_str())
                                && t.predicate == rdf_type.as_ref()
                                && matches!(t.object, TermRef::NamedNode(n) if n == owl_restriction.as_ref())
                        });

                        if is_restriction {
                            if let Some(restriction) = self.parse_restriction(blank.as_str())? {
                                restrictions.push(restriction);
                            }
                        }
                    }
                }
            }
        }

        Ok(restrictions)
    }

    /// Parse a restriction from a blank node
    fn parse_restriction(&self, blank_id: &str) -> Result<Option<OwlRestriction>> {
        let on_property = NamedNode::new(format!("{}onProperty", OWL_NS))?;
        let some_values = NamedNode::new(format!("{}someValuesFrom", OWL_NS))?;
        let all_values = NamedNode::new(format!("{}allValuesFrom", OWL_NS))?;
        let has_value = NamedNode::new(format!("{}hasValue", OWL_NS))?;
        let min_card = NamedNode::new(format!("{}minCardinality", OWL_NS))?;
        let max_card = NamedNode::new(format!("{}maxCardinality", OWL_NS))?;
        let exact_card = NamedNode::new(format!("{}cardinality", OWL_NS))?;
        let min_qualified = NamedNode::new(format!("{}minQualifiedCardinality", OWL_NS))?;
        let max_qualified = NamedNode::new(format!("{}maxQualifiedCardinality", OWL_NS))?;
        let exact_qualified = NamedNode::new(format!("{}qualifiedCardinality", OWL_NS))?;
        let on_class = NamedNode::new(format!("{}onClass", OWL_NS))?;

        let mut property = None;
        let mut some_class = None;
        let mut all_class = None;
        let mut value = None;
        let mut min_c = None;
        let mut max_c = None;
        let mut exact_c = None;
        let mut min_q = None;
        let mut max_q = None;
        let mut exact_q = None;
        let mut qualified_class = None;

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::BlankNode(subj) = triple.subject {
                if subj.as_str() == blank_id {
                    if triple.predicate == on_property.as_ref() {
                        if let TermRef::NamedNode(prop) = triple.object {
                            property = Some(self.extract_local_name(prop.as_str()));
                        }
                    } else if triple.predicate == some_values.as_ref() {
                        if let TermRef::NamedNode(cls) = triple.object {
                            some_class = Some(self.extract_local_name(cls.as_str()));
                        }
                    } else if triple.predicate == all_values.as_ref() {
                        if let TermRef::NamedNode(cls) = triple.object {
                            all_class = Some(self.extract_local_name(cls.as_str()));
                        }
                    } else if triple.predicate == has_value.as_ref() {
                        match triple.object {
                            TermRef::NamedNode(n) => {
                                value = Some(self.extract_local_name(n.as_str()));
                            }
                            TermRef::Literal(lit) => {
                                value = Some(lit.value().to_string());
                            }
                            _ => {}
                        }
                    } else if triple.predicate == min_card.as_ref() {
                        if let TermRef::Literal(lit) = triple.object {
                            min_c = lit.value().parse().ok();
                        }
                    } else if triple.predicate == max_card.as_ref() {
                        if let TermRef::Literal(lit) = triple.object {
                            max_c = lit.value().parse().ok();
                        }
                    } else if triple.predicate == exact_card.as_ref() {
                        if let TermRef::Literal(lit) = triple.object {
                            exact_c = lit.value().parse().ok();
                        }
                    } else if triple.predicate == min_qualified.as_ref() {
                        if let TermRef::Literal(lit) = triple.object {
                            min_q = lit.value().parse().ok();
                        }
                    } else if triple.predicate == max_qualified.as_ref() {
                        if let TermRef::Literal(lit) = triple.object {
                            max_q = lit.value().parse().ok();
                        }
                    } else if triple.predicate == exact_qualified.as_ref() {
                        if let TermRef::Literal(lit) = triple.object {
                            exact_q = lit.value().parse().ok();
                        }
                    } else if triple.predicate == on_class.as_ref() {
                        if let TermRef::NamedNode(cls) = triple.object {
                            qualified_class = Some(self.extract_local_name(cls.as_str()));
                        }
                    }
                }
            }
        }

        if let Some(prop) = property {
            if let Some(cls) = some_class {
                return Ok(Some(OwlRestriction::SomeValuesFrom {
                    on_property: prop,
                    class: cls,
                }));
            }
            if let Some(cls) = all_class {
                return Ok(Some(OwlRestriction::AllValuesFrom {
                    on_property: prop,
                    class: cls,
                }));
            }
            if let Some(val) = value {
                return Ok(Some(OwlRestriction::HasValue {
                    on_property: prop,
                    value: val,
                }));
            }
            if let Some(card) = min_c {
                return Ok(Some(OwlRestriction::MinCardinality {
                    on_property: prop,
                    cardinality: card,
                }));
            }
            if let Some(card) = max_c {
                return Ok(Some(OwlRestriction::MaxCardinality {
                    on_property: prop,
                    cardinality: card,
                }));
            }
            if let Some(card) = exact_c {
                return Ok(Some(OwlRestriction::ExactCardinality {
                    on_property: prop,
                    cardinality: card,
                }));
            }
            if let (Some(card), Some(cls)) = (min_q, qualified_class.as_ref()) {
                return Ok(Some(OwlRestriction::MinQualifiedCardinality {
                    on_property: prop,
                    cardinality: card,
                    on_class: cls.clone(),
                }));
            }
            if let (Some(card), Some(cls)) = (max_q, qualified_class.as_ref()) {
                return Ok(Some(OwlRestriction::MaxQualifiedCardinality {
                    on_property: prop,
                    cardinality: card,
                    on_class: cls.clone(),
                }));
            }
            if let (Some(card), Some(cls)) = (exact_q, qualified_class) {
                return Ok(Some(OwlRestriction::ExactQualifiedCardinality {
                    on_property: prop,
                    cardinality: card,
                    on_class: cls,
                }));
            }
        }

        Ok(None)
    }

    /// Extract local name from full IRI
    fn extract_local_name(&self, iri: &str) -> String {
        iri.split(['/', '#']).next_back().unwrap_or(iri).to_string()
    }

    /// Check if a class has OWL-specific properties
    fn has_owl_class_properties(&self, node: &NamedNode) -> Result<bool> {
        let owl_predicates = vec![
            format!("{}equivalentClass", OWL_NS),
            format!("{}unionOf", OWL_NS),
            format!("{}intersectionOf", OWL_NS),
            format!("{}complementOf", OWL_NS),
            format!("{}disjointWith", OWL_NS),
        ];

        for pred_iri in owl_predicates {
            let pred = NamedNode::new(pred_iri)?;
            for triple in self.graph.iter() {
                if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                    if subj == node.as_ref() && triple.predicate == pred.as_ref() {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Get equivalent classes for a node
    fn get_equivalent_classes(&self, node: &NamedNode) -> Result<Vec<String>> {
        let equivalent = NamedNode::new(format!("{}equivalentClass", OWL_NS))?;
        let mut classes = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == equivalent.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        classes.push(obj.as_str().to_string());
                    }
                }
            }
        }

        Ok(classes)
    }

    /// Get unionOf classes for a node
    fn get_union_of(&self, node: &NamedNode) -> Result<Vec<Vec<String>>> {
        let union_pred = NamedNode::new(format!("{}unionOf", OWL_NS))?;
        let mut unions = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == union_pred.as_ref() {
                    if let TermRef::BlankNode(list) = triple.object {
                        let classes = self.extract_rdf_list_named_nodes(list.as_str());
                        if !classes.is_empty() {
                            unions.push(classes);
                        }
                    }
                }
            }
        }

        Ok(unions)
    }

    /// Get intersectionOf classes for a node
    fn get_intersection_of(&self, node: &NamedNode) -> Result<Vec<Vec<String>>> {
        let intersection_pred = NamedNode::new(format!("{}intersectionOf", OWL_NS))?;
        let mut intersections = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == intersection_pred.as_ref() {
                    if let TermRef::BlankNode(list) = triple.object {
                        let classes = self.extract_rdf_list_named_nodes(list.as_str());
                        if !classes.is_empty() {
                            intersections.push(classes);
                        }
                    }
                }
            }
        }

        Ok(intersections)
    }

    /// Get complementOf classes for a node
    fn get_complement_of(&self, node: &NamedNode) -> Result<Vec<String>> {
        let complement = NamedNode::new(format!("{}complementOf", OWL_NS))?;
        let mut classes = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == complement.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        classes.push(obj.as_str().to_string());
                    }
                }
            }
        }

        Ok(classes)
    }

    /// Get disjointWith classes for a node
    fn get_disjoint_with(&self, node: &NamedNode) -> Result<Vec<String>> {
        let disjoint = NamedNode::new(format!("{}disjointWith", OWL_NS))?;
        let mut classes = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == disjoint.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        classes.push(obj.as_str().to_string());
                    }
                }
            }
        }

        Ok(classes)
    }

    /// Get property characteristics
    fn get_property_characteristics(&self, node: &NamedNode) -> Result<OwlPropertyCharacteristics> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;

        let mut chars = OwlPropertyCharacteristics::default();

        let char_types = vec![
            (format!("{}FunctionalProperty", OWL_NS), "functional"),
            (
                format!("{}InverseFunctionalProperty", OWL_NS),
                "inverse_functional",
            ),
            (format!("{}TransitiveProperty", OWL_NS), "transitive"),
            (format!("{}SymmetricProperty", OWL_NS), "symmetric"),
            (format!("{}AsymmetricProperty", OWL_NS), "asymmetric"),
            (format!("{}ReflexiveProperty", OWL_NS), "reflexive"),
            (format!("{}IrreflexiveProperty", OWL_NS), "irreflexive"),
        ];

        for (type_iri, char_name) in char_types {
            let type_node = NamedNode::new(type_iri)?;

            for triple in self.graph.iter() {
                if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                    if subj == node.as_ref() && triple.predicate == rdf_type.as_ref() {
                        if let TermRef::NamedNode(obj) = triple.object {
                            if obj == type_node.as_ref() {
                                match char_name {
                                    "functional" => chars.functional = true,
                                    "inverse_functional" => chars.inverse_functional = true,
                                    "transitive" => chars.transitive = true,
                                    "symmetric" => chars.symmetric = true,
                                    "asymmetric" => chars.asymmetric = true,
                                    "reflexive" => chars.reflexive = true,
                                    "irreflexive" => chars.irreflexive = true,
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(chars)
    }

    /// Get inverse property
    fn get_inverse_of(&self, node: &NamedNode) -> Result<Option<String>> {
        let inverse = NamedNode::new(format!("{}inverseOf", OWL_NS))?;

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == inverse.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        return Ok(Some(obj.as_str().to_string()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Get equivalent properties
    fn get_equivalent_properties(&self, node: &NamedNode) -> Result<Vec<String>> {
        let equivalent = NamedNode::new(format!("{}equivalentProperty", OWL_NS))?;
        let mut properties = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == equivalent.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        properties.push(obj.as_str().to_string());
                    }
                }
            }
        }

        Ok(properties)
    }

    /// Get sub-properties
    fn get_sub_property_of(&self, node: &NamedNode) -> Result<Vec<String>> {
        let sub_property = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")?;
        let mut properties = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == sub_property.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        properties.push(obj.as_str().to_string());
                    }
                }
            }
        }

        Ok(properties)
    }

    /// Extract named nodes from an RDF list
    fn extract_rdf_list_named_nodes(&self, list_id: &str) -> Vec<String> {
        let mut values = Vec::new();
        let rdf_first = match NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#first") {
            Ok(n) => n,
            Err(_) => return values,
        };
        let rdf_rest = match NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest") {
            Ok(n) => n,
            Err(_) => return values,
        };
        let rdf_nil = match NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil") {
            Ok(n) => n,
            Err(_) => return values,
        };

        let mut current = list_id.to_string();
        loop {
            let mut found_first = false;
            let mut next_node = None;

            for triple in self.graph.iter() {
                if let NamedOrBlankNodeRef::BlankNode(subj) = triple.subject {
                    if subj.as_str() == current {
                        if triple.predicate == rdf_first.as_ref() {
                            if let TermRef::NamedNode(n) = triple.object {
                                values.push(n.as_str().to_string());
                                found_first = true;
                            }
                        } else if triple.predicate == rdf_rest.as_ref() {
                            match triple.object {
                                TermRef::BlankNode(rest) => {
                                    next_node = Some(rest.as_str().to_string());
                                }
                                TermRef::NamedNode(n) if n == rdf_nil.as_ref() => {
                                    return values;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            if !found_first {
                break;
            }

            if let Some(next) = next_node {
                current = next;
            } else {
                break;
            }
        }

        values
    }
}
