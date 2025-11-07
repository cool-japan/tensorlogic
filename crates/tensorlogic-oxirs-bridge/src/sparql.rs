//! Basic SPARQL query compilation to TensorLogic operations
//!
//! This module provides basic support for compiling SPARQL queries
//! into TensorLogic expressions. Currently supports:
//! - Simple SELECT queries
//! - Triple patterns
//! - Filter constraints
//!
//! For full SPARQL support, consider using a dedicated SPARQL engine.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tensorlogic_ir::{TLExpr, Term};

/// Represents a SPARQL triple pattern
#[derive(Debug, Clone, PartialEq)]
pub struct TriplePattern {
    pub subject: PatternElement,
    pub predicate: PatternElement,
    pub object: PatternElement,
}

/// Element in a triple pattern (variable or constant)
#[derive(Debug, Clone, PartialEq)]
pub enum PatternElement {
    Variable(String),
    Constant(String),
}

/// Filter condition in SPARQL
#[derive(Debug, Clone)]
pub enum FilterCondition {
    Equals(String, String),
    NotEquals(String, String),
    GreaterThan(String, String),
    LessThan(String, String),
    Regex(String, String),
}

/// Compiled SPARQL query
#[derive(Debug, Clone)]
pub struct SparqlQuery {
    pub select_vars: Vec<String>,
    pub patterns: Vec<TriplePattern>,
    pub filters: Vec<FilterCondition>,
}

/// SPARQL query parser and compiler
pub struct SparqlCompiler {
    /// Map of predicate IRIs to TensorLogic predicate names
    predicate_mapping: HashMap<String, String>,
}

impl SparqlCompiler {
    pub fn new() -> Self {
        SparqlCompiler {
            predicate_mapping: HashMap::new(),
        }
    }

    /// Add a mapping from IRI to predicate name
    ///
    /// Example: map `"http://example.org/knows"` to `"knows"`
    pub fn add_predicate_mapping(&mut self, iri: String, predicate_name: String) {
        self.predicate_mapping.insert(iri, predicate_name);
    }

    /// Parse a simple SPARQL SELECT query
    ///
    /// Supports basic syntax like:
    /// ```sparql
    /// SELECT ?x ?y WHERE {
    ///   ?x <http://example.org/knows> ?y .
    ///   ?x <http://example.org/age> ?age .
    ///   FILTER(?age > 18)
    /// }
    /// ```
    ///
    /// Note: This is a simplified parser for demonstration.
    /// For production, use a dedicated SPARQL parser.
    pub fn parse_query(&self, sparql: &str) -> Result<SparqlQuery> {
        let mut select_vars = Vec::new();
        let mut patterns = Vec::new();
        let mut filters = Vec::new();

        // Normalize the query by collapsing whitespace and removing newlines within clauses
        let normalized = sparql
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        // Find SELECT variables
        if let Some(select_start) = normalized.find("SELECT") {
            if let Some(where_start) = normalized.find("WHERE") {
                let select_part = normalized[select_start + 6..where_start].trim();
                for var in select_part.split_whitespace() {
                    if let Some(var_name) = var.strip_prefix('?') {
                        select_vars.push(var_name.to_string());
                    }
                }
            }
        }

        // Find WHERE clause content (between { and })
        if let Some(where_start) = normalized.find("WHERE") {
            if let Some(brace_start) = normalized[where_start..].find('{') {
                let content_start = where_start + brace_start + 1;
                if let Some(brace_end) = normalized[content_start..].find('}') {
                    let where_content = normalized[content_start..content_start + brace_end].trim();

                    // Split by statement-terminating '.' (not periods inside URIs)
                    // We parse carefully to avoid splitting on dots inside <...> URIs
                    for statement in self.split_sparql_statements(where_content) {
                        let statement = statement.trim();
                        if statement.is_empty() {
                            continue;
                        }

                        if statement.starts_with("FILTER") {
                            if let Some(filter) = self.parse_filter(statement)? {
                                filters.push(filter);
                            }
                        } else {
                            // Parse triple pattern
                            if let Some(pattern) = self.parse_triple_pattern(statement)? {
                                patterns.push(pattern);
                            }
                        }
                    }
                }
            }
        }

        Ok(SparqlQuery {
            select_vars,
            patterns,
            filters,
        })
    }

    /// Split SPARQL WHERE content into statements, respecting URI boundaries
    ///
    /// This splits on '.' that are statement terminators, not on '.' inside <...> URIs
    fn split_sparql_statements<'a>(&self, content: &'a str) -> Vec<&'a str> {
        let mut statements = Vec::new();
        let mut current_start = 0;
        let mut inside_uri = false;
        let mut inside_string = false;
        let chars: Vec<char> = content.chars().collect();

        for i in 0..chars.len() {
            match chars[i] {
                '<' if !inside_string => inside_uri = true,
                '>' if !inside_string => inside_uri = false,
                '"' if !inside_uri => inside_string = !inside_string,
                '.' if !inside_uri && !inside_string => {
                    // Found a statement-terminating period
                    let statement = &content[current_start..i];
                    if !statement.trim().is_empty() {
                        statements.push(statement);
                    }
                    current_start = i + 1;
                }
                _ => {}
            }
        }

        // Add the last statement if there's anything left
        if current_start < content.len() {
            let statement = &content[current_start..];
            if !statement.trim().is_empty() {
                statements.push(statement);
            }
        }

        statements
    }

    /// Parse a triple pattern
    fn parse_triple_pattern(&self, line: &str) -> Result<Option<TriplePattern>> {
        // Remove trailing dot and split by whitespace
        let line = line.trim_end_matches('.').trim();
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Ok(None);
        }

        let subject = self.parse_pattern_element(parts[0])?;
        let predicate = self.parse_pattern_element(parts[1])?;
        let object = self.parse_pattern_element(parts[2])?;

        Ok(Some(TriplePattern {
            subject,
            predicate,
            object,
        }))
    }

    /// Parse a pattern element (variable or constant)
    fn parse_pattern_element(&self, s: &str) -> Result<PatternElement> {
        if let Some(var_name) = s.strip_prefix('?') {
            Ok(PatternElement::Variable(var_name.to_string()))
        } else if let Some(iri) = s.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
            Ok(PatternElement::Constant(iri.to_string()))
        } else if let Some(literal) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
            Ok(PatternElement::Constant(literal.to_string()))
        } else {
            Ok(PatternElement::Constant(s.to_string()))
        }
    }

    /// Parse a FILTER clause
    fn parse_filter(&self, line: &str) -> Result<Option<FilterCondition>> {
        let filter_content = line
            .strip_prefix("FILTER")
            .and_then(|s| s.trim().strip_prefix('('))
            .and_then(|s| s.trim().strip_suffix(')'))
            .map(|s| s.trim());

        if let Some(content) = filter_content {
            if content.contains(">") {
                let parts: Vec<&str> = content.split('>').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::GreaterThan(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            } else if content.contains("<") {
                let parts: Vec<&str> = content.split('<').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::LessThan(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            } else if content.contains("!=") {
                let parts: Vec<&str> = content.split("!=").map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::NotEquals(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            } else if content.contains("=") {
                let parts: Vec<&str> = content.split('=').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    return Ok(Some(FilterCondition::Equals(
                        parts[0].trim_start_matches('?').to_string(),
                        parts[1].trim_matches('"').to_string(),
                    )));
                }
            }
        }

        Ok(None)
    }

    /// Compile a SPARQL query to TensorLogic expression
    ///
    /// Converts SPARQL patterns to TLExpr predicates and filters to constraints.
    ///
    /// ## Example
    ///
    /// ```
    /// use tensorlogic_oxirs_bridge::sparql::SparqlCompiler;
    ///
    /// let mut compiler = SparqlCompiler::new();
    /// compiler.add_predicate_mapping(
    ///     "http://example.org/knows".to_string(),
    ///     "knows".to_string()
    /// );
    ///
    /// let query = r#"
    ///     SELECT ?x ?y WHERE {
    ///       ?x <http://example.org/knows> ?y .
    ///     }
    /// "#;
    ///
    /// let sparql_query = compiler.parse_query(query).unwrap();
    /// let tl_expr = compiler.compile_to_tensorlogic(&sparql_query).unwrap();
    /// ```
    pub fn compile_to_tensorlogic(&self, query: &SparqlQuery) -> Result<TLExpr> {
        if query.patterns.is_empty() {
            return Err(anyhow!("No patterns to compile"));
        }

        // Convert each pattern to a TLExpr predicate
        let mut expr_list = Vec::new();

        for pattern in &query.patterns {
            let pred_name = match &pattern.predicate {
                PatternElement::Constant(iri) => {
                    // Try to map IRI to predicate name
                    self.predicate_mapping
                        .get(iri)
                        .cloned()
                        .unwrap_or_else(|| Self::iri_to_name(iri))
                }
                PatternElement::Variable(v) => {
                    return Err(anyhow!("Variable predicates not supported: ?{}", v));
                }
            };

            let subj_term = match &pattern.subject {
                PatternElement::Variable(v) => Term::var(v),
                PatternElement::Constant(c) => Term::constant(c),
            };

            let obj_term = match &pattern.object {
                PatternElement::Variable(v) => Term::var(v),
                PatternElement::Constant(c) => Term::constant(c),
            };

            let pred_expr = TLExpr::pred(&pred_name, vec![subj_term, obj_term]);
            expr_list.push(pred_expr);
        }

        // Combine patterns with AND
        let mut combined = expr_list
            .into_iter()
            .reduce(TLExpr::and)
            .ok_or_else(|| anyhow!("Failed to combine patterns"))?;

        // Add filters as additional constraints
        for filter in &query.filters {
            let filter_expr = match filter {
                FilterCondition::Equals(var, val) => {
                    TLExpr::pred("equals", vec![Term::var(var), Term::constant(val)])
                }
                FilterCondition::NotEquals(var, val) => TLExpr::negate(TLExpr::pred(
                    "equals",
                    vec![Term::var(var), Term::constant(val)],
                )),
                FilterCondition::GreaterThan(var, val) => {
                    TLExpr::pred("greaterThan", vec![Term::var(var), Term::constant(val)])
                }
                FilterCondition::LessThan(var, val) => {
                    TLExpr::pred("lessThan", vec![Term::var(var), Term::constant(val)])
                }
                FilterCondition::Regex(var, pattern) => {
                    TLExpr::pred("matches", vec![Term::var(var), Term::constant(pattern)])
                }
            };

            combined = TLExpr::and(combined, filter_expr);
        }

        Ok(combined)
    }

    /// Extract local name from IRI
    fn iri_to_name(iri: &str) -> String {
        iri.split(['/', '#']).next_back().unwrap_or(iri).to_string()
    }
}

impl Default for SparqlCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_query() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?x ?y WHERE {
              ?x <http://example.org/knows> ?y .
            }
        "#;

        let parsed = compiler.parse_query(query).unwrap();
        assert_eq!(parsed.select_vars, vec!["x", "y"]);
        assert_eq!(parsed.patterns.len(), 1);
        assert_eq!(parsed.filters.len(), 0);

        let pattern = &parsed.patterns[0];
        assert_eq!(pattern.subject, PatternElement::Variable("x".to_string()));
        assert_eq!(
            pattern.predicate,
            PatternElement::Constant("http://example.org/knows".to_string())
        );
        assert_eq!(pattern.object, PatternElement::Variable("y".to_string()));
    }

    #[test]
    fn test_parse_query_with_filter() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?x ?age WHERE {
              ?x <http://example.org/age> ?age .
              FILTER(?age > 18)
            }
        "#;

        let parsed = compiler.parse_query(query).unwrap();
        assert_eq!(parsed.select_vars, vec!["x", "age"]);
        assert_eq!(parsed.patterns.len(), 1);
        assert_eq!(parsed.filters.len(), 1);

        match &parsed.filters[0] {
            FilterCondition::GreaterThan(var, val) => {
                assert_eq!(var, "age");
                assert_eq!(val, "18");
            }
            _ => panic!("Expected GreaterThan filter"),
        }
    }

    #[test]
    fn test_compile_simple_query() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/knows".to_string(), "knows".to_string());

        let query = r#"
            SELECT ?x ?y WHERE {
              ?x <http://example.org/knows> ?y .
            }
        "#;

        let parsed = compiler.parse_query(query).unwrap();
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).unwrap();

        // Should generate a predicate expression
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("knows"));
    }

    #[test]
    fn test_compile_query_with_multiple_patterns() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/knows".to_string(), "knows".to_string());
        compiler.add_predicate_mapping("http://example.org/age".to_string(), "age".to_string());

        let query = r#"
            SELECT ?x ?y ?z WHERE {
              ?x <http://example.org/knows> ?y .
              ?y <http://example.org/knows> ?z .
            }
        "#;

        let parsed = compiler.parse_query(query).unwrap();
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).unwrap();

        // Should generate AND of predicates
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("knows"));
        assert!(expr_str.contains("And"));
    }

    #[test]
    fn test_compile_query_with_filter() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/age".to_string(), "age".to_string());

        let query = r#"
            SELECT ?x ?a WHERE {
              ?x <http://example.org/age> ?a .
              FILTER(?a > 18)
            }
        "#;

        let parsed = compiler.parse_query(query).unwrap();
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).unwrap();

        // Should include both predicate and filter
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("age"));
        assert!(expr_str.contains("greaterThan"));
    }

    #[test]
    fn test_iri_to_name() {
        assert_eq!(
            SparqlCompiler::iri_to_name("http://example.org/knows"),
            "knows"
        );
        assert_eq!(
            SparqlCompiler::iri_to_name("http://xmlns.com/foaf/0.1#Person"),
            "Person"
        );
        assert_eq!(SparqlCompiler::iri_to_name("simple"), "simple");
    }
}
