//! RDF schema analysis and extraction.

use anyhow::{Context, Result};
use oxrdf::{NamedNode, NamedOrBlankNodeRef, TermRef};

use super::{ClassInfo, PropertyInfo, SchemaAnalyzer};

impl SchemaAnalyzer {
    pub fn extract_classes(&mut self) -> Result<()> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .context("Invalid IRI")?;
        let rdfs_class =
            NamedNode::new("http://www.w3.org/2000/01/rdf-schema#Class").context("Invalid IRI")?;

        for triple in self.graph.iter() {
            if triple.predicate == rdf_type.as_ref() {
                if let TermRef::NamedNode(obj) = triple.object {
                    if obj == rdfs_class.as_ref() {
                        if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                            let class_iri = subj.as_str().to_string();
                            let class_node =
                                NamedNode::new(class_iri.clone()).context("Invalid class IRI")?;
                            let class_info = ClassInfo {
                                iri: class_iri.clone(),
                                label: self.get_label(&class_node)?,
                                comment: self.get_comment(&class_node)?,
                                subclass_of: self.get_subclasses(&class_node)?,
                            };
                            self.classes.insert(class_iri, class_info);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn extract_properties(&mut self) -> Result<()> {
        let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .context("Invalid IRI")?;
        let rdf_property = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property")
            .context("Invalid IRI")?;

        for triple in self.graph.iter() {
            if triple.predicate == rdf_type.as_ref() {
                if let TermRef::NamedNode(obj) = triple.object {
                    if obj == rdf_property.as_ref() {
                        if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                            let prop_iri = subj.as_str().to_string();
                            let prop_node =
                                NamedNode::new(prop_iri.clone()).context("Invalid property IRI")?;
                            let prop_info = PropertyInfo {
                                iri: prop_iri.clone(),
                                label: self.get_label(&prop_node)?,
                                comment: self.get_comment(&prop_node)?,
                                domain: self.get_domain(&prop_node)?,
                                range: self.get_range(&prop_node)?,
                            };
                            self.properties.insert(prop_iri, prop_info);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn get_label(&self, node: &NamedNode) -> Result<Option<String>> {
        let rdfs_label =
            NamedNode::new("http://www.w3.org/2000/01/rdf-schema#label").context("Invalid IRI")?;

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == rdfs_label.as_ref() {
                    if let TermRef::Literal(lit) = triple.object {
                        return Ok(Some(lit.value().to_string()));
                    }
                }
            }
        }

        Ok(None)
    }

    pub(crate) fn get_comment(&self, node: &NamedNode) -> Result<Option<String>> {
        let rdfs_comment = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#comment")
            .context("Invalid IRI")?;

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == rdfs_comment.as_ref() {
                    if let TermRef::Literal(lit) = triple.object {
                        return Ok(Some(lit.value().to_string()));
                    }
                }
            }
        }

        Ok(None)
    }

    pub(crate) fn get_subclasses(&self, node: &NamedNode) -> Result<Vec<String>> {
        let rdfs_subclass = NamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf")
            .context("Invalid IRI")?;
        let mut subclasses = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == rdfs_subclass.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        subclasses.push(obj.as_str().to_string());
                    }
                }
            }
        }

        Ok(subclasses)
    }

    pub(crate) fn get_domain(&self, node: &NamedNode) -> Result<Vec<String>> {
        let rdfs_domain =
            NamedNode::new("http://www.w3.org/2000/01/rdf-schema#domain").context("Invalid IRI")?;
        let mut domains = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == rdfs_domain.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        domains.push(obj.as_str().to_string());
                    }
                }
            }
        }

        Ok(domains)
    }

    pub(crate) fn get_range(&self, node: &NamedNode) -> Result<Vec<String>> {
        let rdfs_range =
            NamedNode::new("http://www.w3.org/2000/01/rdf-schema#range").context("Invalid IRI")?;
        let mut ranges = Vec::new();

        for triple in self.graph.iter() {
            if let NamedOrBlankNodeRef::NamedNode(subj) = triple.subject {
                if subj == node.as_ref() && triple.predicate == rdfs_range.as_ref() {
                    if let TermRef::NamedNode(obj) = triple.object {
                        ranges.push(obj.as_str().to_string());
                    }
                }
            }
        }

        Ok(ranges)
    }
}
