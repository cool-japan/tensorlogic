//! OxiRS SPARQL execution integration.
//!
//! This module provides a bridge to OxiRS's SPARQL execution capabilities,
//! enabling execution of SPARQL queries against RDF data stores.
//!
//! # Overview
//!
//! The `OxirsSparqlExecutor` provides:
//! - Loading RDF data from Turtle, N-Triples, and other formats
//! - Executing SPARQL queries (SELECT, ASK, CONSTRUCT, DESCRIBE)
//! - Converting query results to TensorLogic expressions
//!
//! # Example
//!
//! ```no_run
//! use tensorlogic_oxirs_bridge::oxirs_executor::OxirsSparqlExecutor;
//!
//! let mut executor = OxirsSparqlExecutor::new().expect("Failed to create executor");
//!
//! // Load RDF data
//! executor.load_turtle(r#"
//!     @prefix ex: <http://example.org/> .
//!     ex:Alice ex:knows ex:Bob .
//!     ex:Bob ex:knows ex:Carol .
//! "#).unwrap();
//!
//! // Execute a query
//! let results = executor.execute("SELECT ?s ?o WHERE { ?s <http://example.org/knows> ?o }").unwrap();
//! ```

use crate::sparql::{SparqlCompiler, SparqlQuery};
use anyhow::{anyhow, Context, Result};
use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject};
use oxirs_core::{RdfStore, Triple};
use oxrdf::Term as OxrdfTerm;
use oxttl::TurtleParser as OxTtlParser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tensorlogic_ir::TLExpr;

/// Query results from SPARQL execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResults {
    /// SELECT query results: variable bindings
    Select {
        /// Variable names in the result set
        variables: Vec<String>,
        /// Result rows, each containing bindings for each variable
        bindings: Vec<HashMap<String, QueryValue>>,
    },
    /// ASK query result: boolean
    Ask(bool),
    /// CONSTRUCT query result: generated triples
    Construct { triples: Vec<TripleResult> },
    /// DESCRIBE query result: description triples
    Describe { triples: Vec<TripleResult> },
}

/// A value in query results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryValue {
    /// IRI value
    Iri(String),
    /// Literal value with optional datatype and language
    Literal {
        value: String,
        datatype: Option<String>,
        language: Option<String>,
    },
    /// Blank node
    BlankNode(String),
}

impl QueryValue {
    /// Get the string value.
    pub fn as_str(&self) -> &str {
        match self {
            QueryValue::Iri(s) => s,
            QueryValue::Literal { value, .. } => value,
            QueryValue::BlankNode(s) => s,
        }
    }

    /// Check if this is an IRI.
    pub fn is_iri(&self) -> bool {
        matches!(self, QueryValue::Iri(_))
    }

    /// Check if this is a literal.
    pub fn is_literal(&self) -> bool {
        matches!(self, QueryValue::Literal { .. })
    }
}

/// A triple in query results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleResult {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// OxiRS SPARQL executor.
///
/// This executor uses OxiRS's storage backend and query execution
/// capabilities to process SPARQL queries against RDF data.
pub struct OxirsSparqlExecutor {
    /// Internal RDF store
    store: RdfStore,
    /// SPARQL compiler for parsing queries
    compiler: SparqlCompiler,
    /// Predicate mappings for TensorLogic conversion
    predicate_mappings: HashMap<String, String>,
    /// Base IRI for relative references
    base_iri: Option<String>,
}

impl OxirsSparqlExecutor {
    /// Create a new SPARQL executor.
    pub fn new() -> Result<Self> {
        Ok(Self {
            store: RdfStore::new()?,
            compiler: SparqlCompiler::new(),
            predicate_mappings: HashMap::new(),
            base_iri: None,
        })
    }

    /// Set the base IRI for relative references.
    pub fn set_base_iri(&mut self, base: &str) {
        self.base_iri = Some(base.to_string());
    }

    /// Add a predicate mapping for TensorLogic conversion.
    pub fn add_predicate_mapping(&mut self, iri: &str, name: &str) {
        self.predicate_mappings
            .insert(iri.to_string(), name.to_string());
        self.compiler
            .add_predicate_mapping(iri.to_string(), name.to_string());
    }

    /// Load RDF data from Turtle format.
    pub fn load_turtle(&mut self, turtle: &str) -> Result<usize> {
        let parser = OxTtlParser::new().for_slice(turtle.as_bytes());
        let mut count = 0;

        for result in parser {
            let triple = result.context("Failed to parse Turtle")?;

            // Convert to OxiRS triple and store
            let triple = self.oxttl_to_oxirs_triple(&triple)?;
            self.store.insert_triple(triple)?;
            count += 1;
        }

        Ok(count)
    }

    /// Convert an oxttl Triple to an OxiRS Triple.
    #[allow(deprecated)]
    fn oxttl_to_oxirs_triple(&self, triple: &oxrdf::Triple) -> Result<Triple> {
        let subject: Subject = match &triple.subject {
            oxrdf::NamedOrBlankNode::NamedNode(n) => {
                Subject::NamedNode(NamedNode::new(n.as_str())?)
            }
            oxrdf::NamedOrBlankNode::BlankNode(b) => {
                Subject::BlankNode(BlankNode::new(b.as_str())?)
            }
        };

        let predicate: Predicate = Predicate::NamedNode(NamedNode::new(triple.predicate.as_str())?);

        let object: Object = match &triple.object {
            OxrdfTerm::NamedNode(n) => Object::NamedNode(NamedNode::new(n.as_str())?),
            OxrdfTerm::BlankNode(b) => Object::BlankNode(BlankNode::new(b.as_str())?),
            OxrdfTerm::Literal(l) => Object::Literal(Literal::new(l.value())),
            // RDF 1.2 quoted triples - convert to empty literal as fallback
            #[allow(unreachable_patterns)]
            _ => Object::Literal(Literal::new("")),
        };

        Ok(Triple::new(subject, predicate, object))
    }

    /// Load RDF data from N-Triples format.
    pub fn load_ntriples(&mut self, ntriples: &str) -> Result<usize> {
        use oxttl::NTriplesParser;
        let parser = NTriplesParser::new().for_slice(ntriples.as_bytes());
        let mut count = 0;

        for result in parser {
            let triple = result.context("Failed to parse N-Triples")?;
            let triple = self.oxttl_to_oxirs_triple(&triple)?;
            self.store.insert_triple(triple)?;
            count += 1;
        }

        Ok(count)
    }

    /// Get the number of triples in the store.
    pub fn num_triples(&self) -> usize {
        self.store.len().unwrap_or(0)
    }

    /// Convert Subject to string representation.
    fn subject_to_string(subject: &Subject) -> String {
        match subject {
            Subject::NamedNode(n) => n.as_str().to_string(),
            Subject::BlankNode(b) => format!("_:{}", b.as_str()),
            Subject::Variable(v) => format!("?{}", v.as_str()),
            Subject::QuotedTriple(_) => "[QuotedTriple]".to_string(),
        }
    }

    /// Convert Predicate to string representation.
    fn predicate_to_string(predicate: &Predicate) -> String {
        match predicate {
            Predicate::NamedNode(n) => n.as_str().to_string(),
            Predicate::Variable(v) => format!("?{}", v.as_str()),
        }
    }

    /// Convert Object to string representation.
    fn object_to_string(object: &Object) -> String {
        match object {
            Object::NamedNode(n) => n.as_str().to_string(),
            Object::BlankNode(b) => format!("_:{}", b.as_str()),
            Object::Literal(l) => l.value().to_string(),
            Object::Variable(v) => format!("?{}", v.as_str()),
            Object::QuotedTriple(_) => "[QuotedTriple]".to_string(),
        }
    }

    /// Get all triples from the store.
    fn get_all_triples(&self) -> Vec<Triple> {
        self.store.triples().unwrap_or_default()
    }

    /// Query triples by subject.
    fn query_by_subject(&self, subject_str: &str) -> Vec<Triple> {
        if let Ok(subject_node) = NamedNode::new(subject_str) {
            let subject = Subject::NamedNode(subject_node);
            self.store
                .query_triples(Some(&subject), None, None)
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Query triples by predicate.
    fn query_by_predicate(&self, predicate_str: &str) -> Vec<Triple> {
        if let Ok(predicate_node) = NamedNode::new(predicate_str) {
            let predicate = Predicate::NamedNode(predicate_node);
            self.store
                .query_triples(None, Some(&predicate), None)
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Query triples by object (IRI only).
    fn query_by_object(&self, object_str: &str) -> Vec<Triple> {
        if let Ok(object_node) = NamedNode::new(object_str) {
            let object = Object::NamedNode(object_node);
            self.store
                .query_triples(None, None, Some(&object))
                .unwrap_or_default()
        } else {
            // Try as literal
            let object = Object::Literal(Literal::new(object_str));
            self.store
                .query_triples(None, None, Some(&object))
                .unwrap_or_default()
        }
    }

    /// Execute a SPARQL query.
    pub fn execute(&self, sparql: &str) -> Result<QueryResults> {
        // Parse the query
        let query = self.compiler.parse_query(sparql)?;

        // Execute based on query type
        self.execute_query(&query)
    }

    /// Execute a parsed query.
    fn execute_query(&self, query: &SparqlQuery) -> Result<QueryResults> {
        use crate::sparql::QueryType;

        match &query.query_type {
            QueryType::Select {
                select_vars,
                distinct,
                ..
            } => {
                let bindings = self.execute_pattern(&query.where_pattern)?;

                // Project to selected variables
                let projected: Vec<HashMap<String, QueryValue>> =
                    if select_vars.contains(&"*".to_string()) {
                        bindings
                    } else {
                        bindings
                            .into_iter()
                            .map(|b| {
                                b.into_iter()
                                    .filter(|(k, _)| select_vars.contains(k))
                                    .collect()
                            })
                            .collect()
                    };

                // Handle DISTINCT
                let results = if *distinct {
                    self.deduplicate_bindings(projected)
                } else {
                    projected
                };

                // Apply LIMIT and OFFSET
                let results = self.apply_modifiers(results, query.limit, query.offset);

                Ok(QueryResults::Select {
                    variables: select_vars.clone(),
                    bindings: results,
                })
            }
            QueryType::Ask => {
                let bindings = self.execute_pattern(&query.where_pattern)?;
                Ok(QueryResults::Ask(!bindings.is_empty()))
            }
            QueryType::Construct { template } => {
                use crate::sparql::PatternElement;
                let bindings = self.execute_pattern(&query.where_pattern)?;
                let mut triples = Vec::new();

                for binding in bindings {
                    for pattern in template {
                        let subject = match &pattern.subject {
                            PatternElement::Variable(v) => binding
                                .get(v)
                                .map(|q| q.as_str().to_string())
                                .unwrap_or_default(),
                            PatternElement::Constant(c) => c.clone(),
                        };
                        let predicate = match &pattern.predicate {
                            PatternElement::Variable(v) => binding
                                .get(v)
                                .map(|q| q.as_str().to_string())
                                .unwrap_or_default(),
                            PatternElement::Constant(c) => c.clone(),
                        };
                        let object = match &pattern.object {
                            PatternElement::Variable(v) => binding
                                .get(v)
                                .map(|q| q.as_str().to_string())
                                .unwrap_or_default(),
                            PatternElement::Constant(c) => c.clone(),
                        };

                        if !subject.is_empty() && !predicate.is_empty() && !object.is_empty() {
                            triples.push(TripleResult {
                                subject,
                                predicate,
                                object,
                            });
                        }
                    }
                }

                Ok(QueryResults::Construct { triples })
            }
            QueryType::Describe { resources } => {
                let mut triples = Vec::new();

                for resource in resources {
                    // Get all triples where resource is subject
                    let outgoing = self.query_by_subject(resource);
                    for triple in outgoing {
                        triples.push(TripleResult {
                            subject: Self::subject_to_string(triple.subject()),
                            predicate: Self::predicate_to_string(triple.predicate()),
                            object: Self::object_to_string(triple.object()),
                        });
                    }

                    // Get all triples where resource is object
                    let incoming = self.query_by_object(resource);
                    for triple in incoming {
                        triples.push(TripleResult {
                            subject: Self::subject_to_string(triple.subject()),
                            predicate: Self::predicate_to_string(triple.predicate()),
                            object: Self::object_to_string(triple.object()),
                        });
                    }
                }

                Ok(QueryResults::Describe { triples })
            }
        }
    }

    /// Execute a graph pattern and return bindings.
    fn execute_pattern(
        &self,
        pattern: &crate::sparql::GraphPattern,
    ) -> Result<Vec<HashMap<String, QueryValue>>> {
        use crate::sparql::{GraphPattern, PatternElement};

        match pattern {
            GraphPattern::Triple(triple) => {
                let mut results = Vec::new();

                // Query the store based on bound elements
                let triples = match (&triple.subject, &triple.predicate, &triple.object) {
                    (PatternElement::Constant(s), _, _) => self.query_by_subject(s),
                    (_, PatternElement::Constant(p), _) => self.query_by_predicate(p),
                    (_, _, PatternElement::Constant(o)) => self.query_by_object(o),
                    _ => self.get_all_triples(),
                };

                for t in triples {
                    let mut binding = HashMap::new();
                    let t_subject = Self::subject_to_string(t.subject());
                    let t_predicate = Self::predicate_to_string(t.predicate());
                    let t_object = Self::object_to_string(t.object());

                    // Bind subject
                    if let PatternElement::Variable(v) = &triple.subject {
                        binding.insert(v.clone(), QueryValue::Iri(t_subject.clone()));
                    } else if let PatternElement::Constant(c) = &triple.subject {
                        if t_subject != *c {
                            continue;
                        }
                    }

                    // Bind predicate
                    if let PatternElement::Variable(v) = &triple.predicate {
                        binding.insert(v.clone(), QueryValue::Iri(t_predicate.clone()));
                    } else if let PatternElement::Constant(c) = &triple.predicate {
                        if t_predicate != *c {
                            continue;
                        }
                    }

                    // Bind object
                    if let PatternElement::Variable(v) = &triple.object {
                        let value = if t_object.starts_with("http") {
                            QueryValue::Iri(t_object.clone())
                        } else {
                            QueryValue::Literal {
                                value: t_object.clone(),
                                datatype: None,
                                language: None,
                            }
                        };
                        binding.insert(v.clone(), value);
                    } else if let PatternElement::Constant(c) = &triple.object {
                        if t_object != *c {
                            continue;
                        }
                    }

                    results.push(binding);
                }

                Ok(results)
            }
            GraphPattern::Group(patterns) => {
                if patterns.is_empty() {
                    return Ok(vec![HashMap::new()]);
                }

                let mut current_results = self.execute_pattern(&patterns[0])?;

                for pattern in patterns.iter().skip(1) {
                    let new_results = self.execute_pattern(pattern)?;
                    current_results = self.join_bindings(current_results, new_results);
                }

                Ok(current_results)
            }
            GraphPattern::Optional(inner) => {
                // OPTIONAL: left outer join semantics
                let inner_results = self.execute_pattern(inner)?;
                if inner_results.is_empty() {
                    Ok(vec![HashMap::new()])
                } else {
                    Ok(inner_results)
                }
            }
            GraphPattern::Union(left, right) => {
                let mut left_results = self.execute_pattern(left)?;
                let right_results = self.execute_pattern(right)?;
                left_results.extend(right_results);
                Ok(left_results)
            }
            GraphPattern::Filter(_condition) => {
                // Filters are applied during pattern matching
                // For now, return empty (would need context from parent pattern)
                Ok(Vec::new())
            }
        }
    }

    /// Join two sets of bindings.
    fn join_bindings(
        &self,
        left: Vec<HashMap<String, QueryValue>>,
        right: Vec<HashMap<String, QueryValue>>,
    ) -> Vec<HashMap<String, QueryValue>> {
        let mut results = Vec::new();

        for l in &left {
            for r in &right {
                // Check for compatible bindings
                let mut compatible = true;
                for (key, lval) in l {
                    if let Some(rval) = r.get(key) {
                        if lval.as_str() != rval.as_str() {
                            compatible = false;
                            break;
                        }
                    }
                }

                if compatible {
                    // Merge bindings
                    let mut merged = l.clone();
                    for (key, rval) in r {
                        merged.entry(key.clone()).or_insert_with(|| rval.clone());
                    }
                    results.push(merged);
                }
            }
        }

        results
    }

    /// Remove duplicate bindings.
    fn deduplicate_bindings(
        &self,
        bindings: Vec<HashMap<String, QueryValue>>,
    ) -> Vec<HashMap<String, QueryValue>> {
        let mut seen = std::collections::HashSet::new();
        let mut results = Vec::new();

        for binding in bindings {
            let key: Vec<_> = binding
                .iter()
                .map(|(k, v)| format!("{}={}", k, v.as_str()))
                .collect();
            let key = key.join(",");

            if !seen.contains(&key) {
                seen.insert(key);
                results.push(binding);
            }
        }

        results
    }

    /// Apply LIMIT and OFFSET modifiers.
    fn apply_modifiers(
        &self,
        bindings: Vec<HashMap<String, QueryValue>>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Vec<HashMap<String, QueryValue>> {
        let offset = offset.unwrap_or(0);
        let mut results: Vec<_> = bindings.into_iter().skip(offset).collect();

        if let Some(limit) = limit {
            results.truncate(limit);
        }

        results
    }

    /// Execute a SPARQL query and convert results to TensorLogic expressions.
    pub fn execute_to_tlexpr(&self, sparql: &str) -> Result<TLExpr> {
        let query = self.compiler.parse_query(sparql)?;
        self.compiler.compile_to_tensorlogic(&query)
    }

    /// Convert query results to TensorLogic expression.
    pub fn results_to_tlexpr(&self, results: &QueryResults) -> Result<TLExpr> {
        match results {
            QueryResults::Ask(value) => {
                if *value {
                    Ok(TLExpr::pred("true", vec![]))
                } else {
                    Ok(TLExpr::pred("false", vec![]))
                }
            }
            QueryResults::Select { bindings, .. } => {
                if bindings.is_empty() {
                    return Ok(TLExpr::pred("empty", vec![]));
                }

                // Convert bindings to conjunctions of predicates
                let mut exprs = Vec::new();
                for binding in bindings {
                    let mut terms = Vec::new();
                    for (var, value) in binding {
                        let pred_name = self
                            .predicate_mappings
                            .get(value.as_str())
                            .cloned()
                            .unwrap_or_else(|| var.clone());
                        terms.push(TLExpr::pred(
                            &pred_name,
                            vec![tensorlogic_ir::Term::constant(value.as_str())],
                        ));
                    }

                    if !terms.is_empty() {
                        let conjunction = terms.into_iter().reduce(TLExpr::and);
                        if let Some(expr) = conjunction {
                            exprs.push(expr);
                        }
                    }
                }

                // Return disjunction of all bindings
                exprs
                    .into_iter()
                    .reduce(TLExpr::or)
                    .ok_or_else(|| anyhow!("Empty results"))
            }
            QueryResults::Construct { triples } | QueryResults::Describe { triples } => {
                // Convert triples to predicates
                let mut exprs = Vec::new();
                for triple in triples {
                    let pred_name = self
                        .predicate_mappings
                        .get(&triple.predicate)
                        .cloned()
                        .unwrap_or_else(|| Self::iri_to_name(&triple.predicate));
                    let expr = TLExpr::pred(
                        &pred_name,
                        vec![
                            tensorlogic_ir::Term::constant(&triple.subject),
                            tensorlogic_ir::Term::constant(&triple.object),
                        ],
                    );
                    exprs.push(expr);
                }

                exprs
                    .into_iter()
                    .reduce(TLExpr::and)
                    .ok_or_else(|| anyhow!("Empty results"))
            }
        }
    }

    /// Extract local name from an IRI.
    fn iri_to_name(iri: &str) -> String {
        iri.split(['/', '#']).next_back().unwrap_or(iri).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = OxirsSparqlExecutor::new().expect("Failed to create executor");
        assert_eq!(executor.num_triples(), 0);
    }

    #[test]
    fn test_load_turtle() {
        let mut executor = OxirsSparqlExecutor::new().expect("Failed to create executor");
        let turtle = r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
            ex:Bob ex:knows ex:Carol .
        "#;

        let result = executor.load_turtle(turtle);
        assert!(result.is_ok());
        assert_eq!(executor.num_triples(), 2);
    }

    #[test]
    fn test_execute_select() {
        let mut executor = OxirsSparqlExecutor::new().expect("Failed to create executor");
        executor
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
        "#,
            )
            .expect("Load failed");

        let query = "SELECT ?s ?o WHERE { ?s <http://example.org/knows> ?o }";
        let result = executor.execute(query);

        assert!(result.is_ok());
        match result.expect("Query failed") {
            QueryResults::Select { bindings, .. } => {
                assert!(!bindings.is_empty());
            }
            _ => panic!("Expected SELECT results"),
        }
    }

    #[test]
    fn test_execute_ask() {
        let mut executor = OxirsSparqlExecutor::new().expect("Failed to create executor");
        executor
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
        "#,
            )
            .expect("Load failed");

        let query = "ASK WHERE { ?s <http://example.org/knows> ?o }";
        let result = executor.execute(query);

        assert!(result.is_ok());
        match result.expect("Query failed") {
            QueryResults::Ask(value) => {
                assert!(value);
            }
            _ => panic!("Expected ASK results"),
        }
    }

    #[test]
    fn test_predicate_mapping() {
        let mut executor = OxirsSparqlExecutor::new().expect("Failed to create executor");
        executor.add_predicate_mapping("http://example.org/knows", "knows");

        assert!(executor
            .predicate_mappings
            .contains_key("http://example.org/knows"));
    }

    #[test]
    fn test_query_value_methods() {
        let iri = QueryValue::Iri("http://example.org/Alice".to_string());
        assert!(iri.is_iri());
        assert!(!iri.is_literal());
        assert_eq!(iri.as_str(), "http://example.org/Alice");

        let literal = QueryValue::Literal {
            value: "Hello".to_string(),
            datatype: None,
            language: Some("en".to_string()),
        };
        assert!(!literal.is_iri());
        assert!(literal.is_literal());
    }

    #[test]
    fn test_execute_to_tlexpr() {
        let mut executor = OxirsSparqlExecutor::new().expect("Failed to create executor");
        executor.add_predicate_mapping("http://example.org/knows", "knows");
        executor
            .load_turtle(
                r#"
            @prefix ex: <http://example.org/> .
            ex:Alice ex:knows ex:Bob .
        "#,
            )
            .expect("Load failed");

        let query = "SELECT ?s ?o WHERE { ?s <http://example.org/knows> ?o }";
        let result = executor.execute_to_tlexpr(query);

        assert!(result.is_ok());
    }

    #[test]
    fn test_load_ntriples() {
        let mut executor = OxirsSparqlExecutor::new().expect("Failed to create executor");
        let ntriples = r#"
            <http://example.org/Alice> <http://example.org/knows> <http://example.org/Bob> .
        "#;

        let result = executor.load_ntriples(ntriples);
        assert!(result.is_ok());
        assert_eq!(executor.num_triples(), 1);
    }
}
