//! GraphQL schema integration for TensorLogic
//!
//! This module provides functionality to parse GraphQL schemas and convert them
//! to TensorLogic symbol tables and rules.
//!
//! ## Supported Features
//!
//! - **Type Definitions**: GraphQL types → TensorLogic domains
//! - **Field Definitions**: GraphQL fields → TensorLogic predicates
//! - **Directives**: GraphQL directives → constraint rules (future)
//! - **Interfaces**: GraphQL interfaces → domain hierarchies (future)
//!
//! ## Example
//!
//! ```ignore
//! use tensorlogic_oxirs_bridge::GraphQLConverter;
//!
//! let schema = r#"
//!     type Person {
//!         name: String!
//!         age: Int
//!         friends: [Person!]
//!     }
//!
//!     type Query {
//!         person(id: ID!): Person
//!     }
//! "#;
//!
//! let converter = GraphQLConverter::new();
//! let symbol_table = converter.parse_schema(schema)?;
//! ```

use anyhow::Result;
use std::collections::HashMap;
use tensorlogic_adapters::{DomainInfo, PredicateInfo, SymbolTable};

/// Represents a GraphQL type definition
#[derive(Debug, Clone)]
pub struct GraphQLType {
    pub name: String,
    pub fields: Vec<GraphQLField>,
    pub interfaces: Vec<String>,
    pub description: Option<String>,
}

/// Represents a GraphQL field definition
#[derive(Debug, Clone)]
pub struct GraphQLField {
    pub name: String,
    pub field_type: String,
    pub is_required: bool,
    pub is_list: bool,
    pub arguments: Vec<GraphQLArgument>,
    pub description: Option<String>,
}

/// Represents a GraphQL field argument
#[derive(Debug, Clone)]
pub struct GraphQLArgument {
    pub name: String,
    pub arg_type: String,
    pub is_required: bool,
}

/// GraphQL schema converter
pub struct GraphQLConverter {
    types: HashMap<String, GraphQLType>,
}

impl GraphQLConverter {
    pub fn new() -> Self {
        GraphQLConverter {
            types: HashMap::new(),
        }
    }

    /// Parse a GraphQL schema string (simplified implementation)
    ///
    /// Note: This is a basic implementation that handles simple type definitions.
    /// For full GraphQL parsing, consider using a dedicated GraphQL parser crate.
    pub fn parse_schema(&mut self, schema: &str) -> Result<SymbolTable> {
        self.types.clear();

        // Simple parsing logic - in production, use a proper GraphQL parser
        let lines: Vec<&str> = schema.lines().map(|l| l.trim()).collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                i += 1;
                continue;
            }

            // Parse type definitions
            if line.starts_with("type ") {
                let type_def = self.parse_type_definition(&lines, &mut i)?;
                self.types.insert(type_def.name.clone(), type_def);
            } else {
                i += 1;
            }
        }

        self.to_symbol_table()
    }

    /// Parse a type definition from GraphQL schema
    fn parse_type_definition(&self, lines: &[&str], index: &mut usize) -> Result<GraphQLType> {
        let line = lines[*index];

        // Extract type name from "type TypeName {" or "type TypeName implements Interface {"
        let type_line = line
            .strip_prefix("type ")
            .ok_or_else(|| anyhow::anyhow!("Invalid type definition"))?;

        let mut parts = type_line.split('{');
        let type_header = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("Missing type header"))?
            .trim();

        let (name, interfaces) = if type_header.contains("implements") {
            let mut impl_parts = type_header.split("implements");
            let name = impl_parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("Missing type name"))?
                .trim()
                .to_string();
            let interfaces_str = impl_parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("Missing interfaces"))?
                .trim();
            let interfaces = interfaces_str
                .split('&')
                .map(|s| s.trim().to_string())
                .collect();
            (name, interfaces)
        } else {
            (type_header.to_string(), Vec::new())
        };

        let mut fields = Vec::new();
        *index += 1;

        // Parse fields until we hit closing brace
        while *index < lines.len() {
            let field_line = lines[*index].trim();

            if field_line == "}" {
                *index += 1;
                break;
            }

            if field_line.is_empty() || field_line.starts_with('#') {
                *index += 1;
                continue;
            }

            // Parse field: "fieldName: Type" or "fieldName(args): Type"
            if let Some(field) = self.parse_field(field_line)? {
                fields.push(field);
            }

            *index += 1;
        }

        Ok(GraphQLType {
            name,
            fields,
            interfaces,
            description: None,
        })
    }

    /// Parse a field definition
    fn parse_field(&self, line: &str) -> Result<Option<GraphQLField>> {
        // Handle field with arguments: "fieldName(arg: Type): ReturnType"
        // or simple field: "fieldName: Type"

        let field_line = if line.find(':').is_some() {
            line
        } else {
            return Ok(None); // Not a valid field line
        };

        let (field_part, type_part) = field_line.split_once(':').unwrap();

        let field_part = field_part.trim();
        let type_part = type_part.trim();

        // Extract field name and arguments
        let (field_name, arguments) = if field_part.contains('(') {
            let name_end = field_part.find('(').unwrap();
            let field_name = field_part[..name_end].trim();
            let args_str = field_part[name_end + 1..]
                .strip_suffix(')')
                .unwrap_or("")
                .trim();

            let arguments = if args_str.is_empty() {
                Vec::new()
            } else {
                self.parse_arguments(args_str)?
            };

            (field_name, arguments)
        } else {
            (field_part, Vec::new())
        };

        // Parse field type (handle !, [], etc.)
        let (field_type, is_required, is_list) = self.parse_type(type_part);

        Ok(Some(GraphQLField {
            name: field_name.to_string(),
            field_type,
            is_required,
            is_list,
            arguments,
            description: None,
        }))
    }

    /// Parse field arguments
    fn parse_arguments(&self, args_str: &str) -> Result<Vec<GraphQLArgument>> {
        let mut arguments = Vec::new();

        for arg in args_str.split(',') {
            let arg = arg.trim();
            if arg.is_empty() {
                continue;
            }

            if let Some((name, type_str)) = arg.split_once(':') {
                let name = name.trim().to_string();
                let (arg_type, is_required, _is_list) = self.parse_type(type_str.trim());

                arguments.push(GraphQLArgument {
                    name,
                    arg_type,
                    is_required,
                });
            }
        }

        Ok(arguments)
    }

    /// Parse a GraphQL type string (handles !, [], etc.)
    fn parse_type(&self, type_str: &str) -> (String, bool, bool) {
        let type_str = type_str.trim();
        let is_required = type_str.ends_with('!');
        let type_str = type_str.trim_end_matches('!');

        let is_list = type_str.starts_with('[') && type_str.ends_with(']');
        let type_str = if is_list {
            type_str
                .strip_prefix('[')
                .unwrap()
                .strip_suffix(']')
                .unwrap()
                .trim_end_matches('!')
        } else {
            type_str
        };

        (type_str.to_string(), is_required, is_list)
    }

    /// Convert GraphQL types to a TensorLogic symbol table
    pub fn to_symbol_table(&self) -> Result<SymbolTable> {
        let mut table = SymbolTable::new();

        // First, add scalar types as domains
        for scalar in &["String", "Int", "Float", "Boolean", "ID", "Value"] {
            let domain = DomainInfo::new(*scalar, 1000); // Large cardinality for scalars
            table.add_domain(domain)?;
        }

        // Convert types to domains (skip special GraphQL types)
        for (type_name, type_def) in &self.types {
            // Skip special types like Query, Mutation, Subscription
            if matches!(type_name.as_str(), "Query" | "Mutation" | "Subscription") {
                continue;
            }

            let mut domain = DomainInfo::new(type_name, 100); // Default cardinality

            if let Some(desc) = &type_def.description {
                domain = domain.with_description(desc);
            }

            table.add_domain(domain)?;
        }

        // Convert fields to predicates (excluding special types)
        for type_def in self.types.values() {
            // Skip special types - they're not domains
            if matches!(
                type_def.name.as_str(),
                "Query" | "Mutation" | "Subscription"
            ) {
                continue;
            }
            for field in &type_def.fields {
                let predicate_name = format!("{}_{}", type_def.name, field.name);

                let mut arg_domains = vec![type_def.name.clone()];

                // Add field type as second argument if it's a known type
                if self.types.contains_key(&field.field_type)
                    || self.is_scalar_type(&field.field_type)
                {
                    arg_domains.push(field.field_type.clone());
                } else {
                    arg_domains.push("Value".to_string()); // Default domain for unknown types
                }

                let mut predicate = PredicateInfo::new(&predicate_name, arg_domains);

                if let Some(desc) = &field.description {
                    predicate = predicate.with_description(desc);
                }

                table.add_predicate(predicate)?;
            }
        }

        Ok(table)
    }

    /// Check if a type is a GraphQL scalar type
    fn is_scalar_type(&self, type_name: &str) -> bool {
        matches!(
            type_name,
            "String" | "Int" | "Float" | "Boolean" | "ID" | "Value"
        )
    }

    /// Get parsed types
    pub fn types(&self) -> &HashMap<String, GraphQLType> {
        &self.types
    }
}

impl Default for GraphQLConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphql_converter_basic() {
        let converter = GraphQLConverter::new();
        assert!(converter.types().is_empty());
    }

    #[test]
    fn test_parse_simple_type() {
        let schema = r#"
            type Person {
                name: String!
                age: Int
            }
        "#;

        let mut converter = GraphQLConverter::new();
        let result = converter.parse_schema(schema);
        assert!(result.is_ok());

        assert_eq!(converter.types().len(), 1);
        assert!(converter.types().contains_key("Person"));
    }

    #[test]
    fn test_parse_type_with_fields() {
        let schema = r#"
            type User {
                id: ID!
                name: String!
                email: String
            }
        "#;

        let mut converter = GraphQLConverter::new();
        converter.parse_schema(schema).unwrap();

        let user_type = converter.types().get("User").unwrap();
        assert_eq!(user_type.fields.len(), 3);

        let id_field = &user_type.fields[0];
        assert_eq!(id_field.name, "id");
        assert_eq!(id_field.field_type, "ID");
        assert!(id_field.is_required);
    }

    #[test]
    fn test_parse_type_with_list() {
        let schema = r#"
            type Post {
                tags: [String!]
                comments: [Comment]
            }
        "#;

        let mut converter = GraphQLConverter::new();
        converter.parse_schema(schema).unwrap();

        let post_type = converter.types().get("Post").unwrap();
        assert_eq!(post_type.fields.len(), 2);

        let tags_field = &post_type.fields[0];
        assert!(tags_field.is_list);
    }

    #[test]
    fn test_to_symbol_table() {
        let schema = r#"
            type Person {
                name: String!
                age: Int
            }
        "#;

        let mut converter = GraphQLConverter::new();
        let table = converter.parse_schema(schema).unwrap();

        // Should have Person domain
        assert!(table.domains.contains_key("Person"));

        // Should have predicates for fields
        assert!(table.predicates.contains_key("Person_name"));
        assert!(table.predicates.contains_key("Person_age"));
    }

    #[test]
    fn test_parse_multiple_types() {
        let schema = r#"
            type Author {
                name: String!
            }

            type Book {
                title: String!
                author: Author!
            }
        "#;

        let mut converter = GraphQLConverter::new();
        converter.parse_schema(schema).unwrap();

        assert_eq!(converter.types().len(), 2);
        assert!(converter.types().contains_key("Author"));
        assert!(converter.types().contains_key("Book"));
    }

    #[test]
    fn test_skip_special_types() {
        let schema = r#"
            type Query {
                user(id: ID!): User
            }

            type User {
                name: String!
            }
        "#;

        let mut converter = GraphQLConverter::new();
        let table = converter.parse_schema(schema).unwrap();

        // Query type should not be added as a domain
        assert!(!table.domains.contains_key("Query"));

        // But User should be added
        assert!(table.domains.contains_key("User"));
    }
}
