//! GraphQL integration for RDF knowledge graphs.
//!
//! This module provides a bridge between OxiRS's RDF capabilities and GraphQL,
//! enabling GraphQL queries over knowledge graphs stored in RDF format.
//!
//! # Overview
//!
//! The `OxirsGraphQLBridge` provides:
//! - Automatic GraphQL schema generation from RDF ontologies
//! - Query execution translating GraphQL to SPARQL
//! - Type mapping between RDF classes and GraphQL types
//!
//! # Example
//!
//! ```no_run
//! use tensorlogic_oxirs_bridge::oxirs_graphql::OxirsGraphQLBridge;
//!
//! let mut bridge = OxirsGraphQLBridge::new().expect("Failed to create bridge");
//!
//! // Load RDF data
//! bridge.load_turtle(r#"
//!     @prefix ex: <http://example.org/> .
//!     @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
//!     ex:Person a rdfs:Class .
//!     ex:Alice a ex:Person .
//!     ex:Alice ex:name "Alice" .
//! "#).unwrap();
//!
//! // Generate GraphQL schema from the data
//! bridge.generate_schema().unwrap();
//! ```

use crate::oxirs_executor::OxirsSparqlExecutor;
use anyhow::Result;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tensorlogic_adapters::{DomainInfo, PredicateInfo, SymbolTable};
use tensorlogic_ir::{TLExpr, Term};

/// GraphQL type definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLType {
    /// Type name
    pub name: String,
    /// Whether this is a scalar type
    pub is_scalar: bool,
    /// Whether this is a list type
    pub is_list: bool,
    /// Whether this is non-null
    pub is_non_null: bool,
    /// Inner type for lists
    pub inner_type: Option<Box<GraphQLType>>,
}

impl GraphQLType {
    /// Create a scalar type.
    pub fn scalar(name: &str) -> Self {
        Self {
            name: name.to_string(),
            is_scalar: true,
            is_list: false,
            is_non_null: false,
            inner_type: None,
        }
    }

    /// Create an object type reference.
    pub fn object(name: &str) -> Self {
        Self {
            name: name.to_string(),
            is_scalar: false,
            is_list: false,
            is_non_null: false,
            inner_type: None,
        }
    }

    /// Create a list type.
    pub fn list(inner: GraphQLType) -> Self {
        Self {
            name: format!("[{}]", inner.name),
            is_scalar: false,
            is_list: true,
            is_non_null: false,
            inner_type: Some(Box::new(inner)),
        }
    }

    /// Make this type non-null.
    pub fn non_null(mut self) -> Self {
        self.is_non_null = true;
        self
    }

    /// Convert to SDL type string.
    pub fn to_sdl(&self) -> String {
        let base = if self.is_list {
            format!("[{}]", self.inner_type.as_ref().map_or("ID", |t| &t.name))
        } else {
            self.name.clone()
        };
        if self.is_non_null {
            format!("{}!", base)
        } else {
            base
        }
    }
}

/// GraphQL field definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLField {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: GraphQLType,
    /// Optional description
    pub description: Option<String>,
    /// Arguments
    pub arguments: Vec<GraphQLArgument>,
}

/// GraphQL argument definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLArgument {
    /// Argument name
    pub name: String,
    /// Argument type
    pub arg_type: GraphQLType,
    /// Default value
    pub default_value: Option<String>,
}

/// GraphQL object type definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLObjectType {
    /// Type name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Fields
    pub fields: IndexMap<String, GraphQLField>,
    /// Interfaces this type implements
    pub interfaces: Vec<String>,
}

impl GraphQLObjectType {
    /// Create a new object type.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            fields: IndexMap::new(),
            interfaces: Vec::new(),
        }
    }

    /// Add a field to this type.
    pub fn add_field(&mut self, field: GraphQLField) {
        self.fields.insert(field.name.clone(), field);
    }

    /// Convert to SDL.
    pub fn to_sdl(&self) -> String {
        let mut sdl = format!("type {} {{\n", self.name);
        for (_, field) in &self.fields {
            if let Some(desc) = &field.description {
                sdl.push_str(&format!("  \"\"\"{}\"\"\"\n", desc));
            }
            sdl.push_str(&format!(
                "  {}: {}\n",
                field.name,
                field.field_type.to_sdl()
            ));
        }
        sdl.push_str("}\n");
        sdl
    }
}

/// GraphQL schema definition (internal representation).
#[derive(Debug, Clone, Default)]
pub struct GraphQLSchema {
    /// Type definitions
    pub types: IndexMap<String, GraphQLObjectType>,
    /// Query type name
    pub query_type: Option<String>,
    /// Mutation type name
    pub mutation_type: Option<String>,
    /// Schema SDL string
    pub sdl: String,
}

impl GraphQLSchema {
    /// Create a new empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse schema from SDL (simplified - stores the SDL string).
    pub fn parse(sdl: &str) -> Result<Self> {
        Ok(Self {
            types: IndexMap::new(),
            query_type: Some("Query".to_string()),
            mutation_type: None,
            sdl: sdl.to_string(),
        })
    }

    /// Add a type to the schema.
    pub fn add_type(&mut self, object_type: GraphQLObjectType) {
        self.types.insert(object_type.name.clone(), object_type);
    }

    /// Generate SDL from types.
    pub fn to_sdl(&self) -> String {
        if !self.sdl.is_empty() {
            return self.sdl.clone();
        }

        let mut sdl = String::new();
        for (_, object_type) in &self.types {
            sdl.push_str(&object_type.to_sdl());
            sdl.push('\n');
        }
        sdl
    }
}

/// OxiRS GraphQL Bridge.
///
/// Bridges RDF data access through a GraphQL interface.
pub struct OxirsGraphQLBridge {
    /// Internal SPARQL executor
    executor: OxirsSparqlExecutor,
    /// Generated GraphQL schema
    schema: Option<GraphQLSchema>,
    /// Type definitions
    types: IndexMap<String, GraphQLObjectType>,
    /// IRI to type name mappings
    iri_to_type: HashMap<String, String>,
    /// Prefix mappings
    prefixes: HashMap<String, String>,
}

impl OxirsGraphQLBridge {
    /// Create a new GraphQL bridge.
    pub fn new() -> Result<Self> {
        Ok(Self {
            executor: OxirsSparqlExecutor::new()?,
            schema: None,
            types: IndexMap::new(),
            iri_to_type: HashMap::new(),
            prefixes: HashMap::new(),
        })
    }

    /// Load RDF data from Turtle format.
    pub fn load_turtle(&mut self, turtle: &str) -> Result<usize> {
        self.executor.load_turtle(turtle)
    }

    /// Add a prefix mapping.
    pub fn add_prefix(&mut self, prefix: &str, iri: &str) {
        self.prefixes.insert(prefix.to_string(), iri.to_string());
    }

    /// Generate GraphQL schema from loaded RDF data.
    pub fn generate_schema(&mut self) -> Result<()> {
        // Query for all classes
        let classes_query = r#"
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT DISTINCT ?class WHERE {
                ?class a rdfs:Class .
            }
        "#;

        let class_results = self.executor.execute(classes_query)?;

        // Create types for each class
        if let crate::oxirs_executor::QueryResults::Select { bindings, .. } = class_results {
            for binding in bindings {
                if let Some(class_value) = binding.get("class") {
                    let class_iri = class_value.as_str();
                    let type_name = Self::iri_to_type_name(class_iri);
                    self.iri_to_type
                        .insert(class_iri.to_string(), type_name.clone());

                    let object_type = GraphQLObjectType::new(&type_name);
                    self.types.insert(type_name, object_type);
                }
            }
        }

        // Query for properties of each class
        for (iri, type_name) in &self.iri_to_type.clone() {
            let props_query = format!(
                r#"
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT DISTINCT ?prop WHERE {{
                    ?prop rdfs:domain <{}> .
                }}
            "#,
                iri
            );

            let prop_results = self.executor.execute(&props_query)?;

            if let crate::oxirs_executor::QueryResults::Select { bindings, .. } = prop_results {
                for binding in bindings {
                    if let Some(prop_value) = binding.get("prop") {
                        let prop_iri = prop_value.as_str();
                        let field_name = Self::iri_to_field_name(prop_iri);

                        let field = GraphQLField {
                            name: field_name,
                            field_type: GraphQLType::scalar("String"),
                            description: None,
                            arguments: Vec::new(),
                        };

                        if let Some(object_type) = self.types.get_mut(type_name) {
                            object_type.add_field(field);
                        }
                    }
                }
            }
        }

        // Build Query type
        let mut query_type = GraphQLObjectType::new("Query");

        // Add a query field for each type
        for (type_name, _) in &self.types {
            if type_name == "Query" {
                continue;
            }

            // Singular query
            let field = GraphQLField {
                name: Self::type_to_query_name(type_name),
                field_type: GraphQLType::object(type_name),
                description: Some(format!("Get a single {} by ID", type_name)),
                arguments: vec![GraphQLArgument {
                    name: "id".to_string(),
                    arg_type: GraphQLType::scalar("ID").non_null(),
                    default_value: None,
                }],
            };
            query_type.add_field(field);

            // Plural query
            let list_field = GraphQLField {
                name: format!("all{}s", type_name),
                field_type: GraphQLType::list(GraphQLType::object(type_name)),
                description: Some(format!("List all {}s", type_name)),
                arguments: vec![
                    GraphQLArgument {
                        name: "limit".to_string(),
                        arg_type: GraphQLType::scalar("Int"),
                        default_value: Some("10".to_string()),
                    },
                    GraphQLArgument {
                        name: "offset".to_string(),
                        arg_type: GraphQLType::scalar("Int"),
                        default_value: Some("0".to_string()),
                    },
                ],
            };
            query_type.add_field(list_field);
        }

        self.types.insert("Query".to_string(), query_type);

        // Generate SDL
        let mut schema = GraphQLSchema::new();
        for (_, obj_type) in &self.types {
            schema.add_type(obj_type.clone());
        }
        schema.query_type = Some("Query".to_string());

        self.schema = Some(schema);

        Ok(())
    }

    /// Execute a GraphQL query.
    pub fn execute_query(&self, query: &str) -> Result<serde_json::Value> {
        // Parse the GraphQL query (simplified - just handle basic queries)
        let query_trimmed = query.trim();

        // Extract operation (simplified parser)
        if query_trimmed.starts_with("query") || query_trimmed.starts_with('{') {
            // This is a query operation
            self.execute_graphql_query(query_trimmed)
        } else {
            Err(anyhow::anyhow!("Unsupported GraphQL operation"))
        }
    }

    /// Execute a parsed GraphQL query (simplified).
    fn execute_graphql_query(&self, query: &str) -> Result<serde_json::Value> {
        // Very simplified GraphQL parser - just extract field names
        // In production, you'd want to use a proper GraphQL parser

        let mut result = serde_json::Map::new();
        let data = serde_json::Map::new();

        // For now, return an empty result with the schema info
        result.insert("data".to_string(), serde_json::Value::Object(data));

        // If introspection query, return schema info
        if query.contains("__schema") || query.contains("__type") {
            if let Some(schema) = &self.schema {
                let mut schema_info = serde_json::Map::new();
                let types: Vec<serde_json::Value> = schema
                    .types
                    .keys()
                    .map(|k| serde_json::json!({"name": k}))
                    .collect();
                schema_info.insert("types".to_string(), serde_json::Value::Array(types));
                result.insert(
                    "data".to_string(),
                    serde_json::json!({"__schema": schema_info}),
                );
            }
        }

        Ok(serde_json::Value::Object(result))
    }

    /// Convert schema to TensorLogic symbol table.
    pub fn schema_to_symbol_table(&self) -> Result<SymbolTable> {
        let mut symbol_table = SymbolTable::new();

        // Add a default "Entity" domain for RDF entities
        // Using a large cardinality as placeholder (unknown actual count)
        let entity_domain = DomainInfo::new("Entity", usize::MAX);
        let _domain_added = symbol_table.add_domain(entity_domain);

        for (name, object_type) in &self.types {
            // Add type as a unary predicate
            let pred_info = PredicateInfo::new(name.clone(), vec!["Entity".to_string()]);
            let _result = symbol_table.add_predicate(pred_info);

            // Add fields as binary predicates
            for (field_name, _field) in &object_type.fields {
                let pred_name = format!("{}_{}", name, field_name);
                let field_pred_info =
                    PredicateInfo::new(pred_name, vec!["Entity".to_string(), "Entity".to_string()]);
                let _result = symbol_table.add_predicate(field_pred_info);
            }
        }

        Ok(symbol_table)
    }

    /// Get the generated GraphQL schema as SDL.
    pub fn get_schema_sdl(&self) -> Option<String> {
        self.schema.as_ref().map(|s: &GraphQLSchema| s.to_sdl())
    }

    /// Convert IRI to GraphQL type name.
    fn iri_to_type_name(iri: &str) -> String {
        let local = iri.split(['/', '#']).next_back().unwrap_or(iri);
        // Capitalize first letter
        let mut chars = local.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    }

    /// Convert IRI to GraphQL field name.
    fn iri_to_field_name(iri: &str) -> String {
        let local = iri.split(['/', '#']).next_back().unwrap_or(iri);
        // Keep as lowercase
        local.to_lowercase()
    }

    /// Convert type name to query field name.
    fn type_to_query_name(type_name: &str) -> String {
        // Lowercase first letter
        let mut chars = type_name.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
        }
    }

    /// Convert a GraphQL query result to TensorLogic expression.
    pub fn result_to_tlexpr(&self, result: &serde_json::Value) -> Result<TLExpr> {
        // Convert JSON result to TLExpr
        match result {
            serde_json::Value::Object(obj) => {
                if let Some(data) = obj.get("data") {
                    self.json_to_tlexpr(data)
                } else {
                    Ok(TLExpr::pred("empty", vec![]))
                }
            }
            _ => Ok(TLExpr::pred("empty", vec![])),
        }
    }

    /// Convert JSON value to TensorLogic expression.
    fn json_to_tlexpr(&self, value: &serde_json::Value) -> Result<TLExpr> {
        match value {
            serde_json::Value::Null => Ok(TLExpr::pred("null", vec![])),
            serde_json::Value::Bool(b) => {
                if *b {
                    Ok(TLExpr::pred("true", vec![]))
                } else {
                    Ok(TLExpr::pred("false", vec![]))
                }
            }
            serde_json::Value::Number(n) => {
                Ok(TLExpr::pred("number", vec![Term::constant(n.to_string())]))
            }
            serde_json::Value::String(s) => Ok(TLExpr::pred("string", vec![Term::constant(s)])),
            serde_json::Value::Array(arr) => {
                let mut exprs = Vec::new();
                for item in arr {
                    exprs.push(self.json_to_tlexpr(item)?);
                }
                exprs
                    .into_iter()
                    .reduce(TLExpr::and)
                    .ok_or_else(|| anyhow::anyhow!("Empty array"))
            }
            serde_json::Value::Object(obj) => {
                let mut exprs = Vec::new();
                for (key, val) in obj {
                    let val_expr = self.json_to_tlexpr(val)?;
                    let pred = TLExpr::pred(key, vec![]);
                    exprs.push(TLExpr::and(pred, val_expr));
                }
                exprs
                    .into_iter()
                    .reduce(TLExpr::and)
                    .ok_or_else(|| anyhow::anyhow!("Empty object"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = OxirsGraphQLBridge::new();
        assert!(bridge.is_ok());
    }

    #[test]
    fn test_graphql_type_scalar() {
        let t = GraphQLType::scalar("String");
        assert!(t.is_scalar);
        assert_eq!(t.to_sdl(), "String");
    }

    #[test]
    fn test_graphql_type_list() {
        let inner = GraphQLType::object("Person");
        let t = GraphQLType::list(inner);
        assert!(t.is_list);
        assert_eq!(t.to_sdl(), "[Person]");
    }

    #[test]
    fn test_graphql_type_non_null() {
        let t = GraphQLType::scalar("ID").non_null();
        assert!(t.is_non_null);
        assert_eq!(t.to_sdl(), "ID!");
    }

    #[test]
    fn test_object_type_creation() {
        let mut object_type = GraphQLObjectType::new("Person");
        object_type.add_field(GraphQLField {
            name: "name".to_string(),
            field_type: GraphQLType::scalar("String"),
            description: None,
            arguments: Vec::new(),
        });

        assert_eq!(object_type.name, "Person");
        assert!(object_type.fields.contains_key("name"));
    }

    #[test]
    fn test_iri_to_type_name() {
        assert_eq!(
            OxirsGraphQLBridge::iri_to_type_name("http://example.org/Person"),
            "Person"
        );
        assert_eq!(
            OxirsGraphQLBridge::iri_to_type_name("http://example.org/schema#person"),
            "Person"
        );
    }

    #[test]
    fn test_type_to_query_name() {
        assert_eq!(OxirsGraphQLBridge::type_to_query_name("Person"), "person");
        assert_eq!(
            OxirsGraphQLBridge::type_to_query_name("BookAuthor"),
            "bookAuthor"
        );
    }

    #[test]
    fn test_execute_introspection_query() {
        let mut bridge = OxirsGraphQLBridge::new().expect("Failed to create bridge");

        // Generate a minimal schema
        bridge
            .types
            .insert("Person".to_string(), GraphQLObjectType::new("Person"));
        bridge.schema = Some(GraphQLSchema {
            types: bridge.types.clone(),
            query_type: Some("Query".to_string()),
            mutation_type: None,
            sdl: String::new(),
        });

        let result = bridge.execute_query("{ __schema { types { name } } }");
        assert!(result.is_ok());
    }

    #[test]
    fn test_schema_to_symbol_table() {
        let mut bridge = OxirsGraphQLBridge::new().expect("Failed to create bridge");

        let mut person_type = GraphQLObjectType::new("Person");
        person_type.add_field(GraphQLField {
            name: "name".to_string(),
            field_type: GraphQLType::scalar("String"),
            description: None,
            arguments: Vec::new(),
        });
        bridge.types.insert("Person".to_string(), person_type);

        let result = bridge.schema_to_symbol_table();
        assert!(result.is_ok());
    }
}
