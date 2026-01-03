//! GraphQL schema integration for TensorLogic
//!
//! This module provides functionality to parse GraphQL schemas and convert them
//! to TensorLogic symbol tables and rules.
//!
//! ## Supported Features
//!
//! - **Type Definitions**: GraphQL types → TensorLogic domains
//! - **Field Definitions**: GraphQL fields → TensorLogic predicates
//! - **Directives**: GraphQL directives → constraint rules
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
use tensorlogic_ir::{TLExpr, Term};

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
    pub directives: Vec<GraphQLDirective>,
    pub description: Option<String>,
}

/// Represents a GraphQL field argument
#[derive(Debug, Clone)]
pub struct GraphQLArgument {
    pub name: String,
    pub arg_type: String,
    pub is_required: bool,
}

/// Represents a GraphQL directive
#[derive(Debug, Clone, PartialEq)]
pub struct GraphQLDirective {
    pub name: String,
    pub arguments: HashMap<String, DirectiveValue>,
}

/// Directive argument value
#[derive(Debug, Clone, PartialEq)]
pub enum DirectiveValue {
    String(String),
    Int(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<DirectiveValue>),
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
        // Handle field with arguments: "fieldName(arg: Type): ReturnType @directive"
        // or simple field: "fieldName: Type @directive"

        let field_line = if line.find(':').is_some() {
            line
        } else {
            return Ok(None); // Not a valid field line
        };

        let (field_part, type_part) = field_line.split_once(':').unwrap();

        let field_part = field_part.trim();
        let mut type_part = type_part.trim();

        // Extract directives from type part
        let directives = self.parse_directives(type_part);

        // Remove directives from type part
        if let Some(at_pos) = type_part.find('@') {
            type_part = type_part[..at_pos].trim();
        }

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
            directives,
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

    /// Parse directives from a field line
    /// Format: @directiveName(arg1: value1, arg2: value2)
    fn parse_directives(&self, line: &str) -> Vec<GraphQLDirective> {
        let mut directives = Vec::new();
        let mut current_pos = 0;

        while let Some(at_pos) = line[current_pos..].find('@') {
            let abs_pos = current_pos + at_pos;
            let remaining = &line[abs_pos + 1..];

            // Extract directive name
            let name_end = remaining
                .find(|c: char| c == '(' || c.is_whitespace() || c == '@')
                .unwrap_or(remaining.len());
            let directive_name = remaining[..name_end].trim().to_string();

            // Parse arguments if present
            let mut arguments = HashMap::new();
            if remaining.len() > name_end && remaining.chars().nth(name_end) == Some('(') {
                if let Some(close_paren) = remaining.find(')') {
                    let args_str = &remaining[name_end + 1..close_paren];
                    arguments = self.parse_directive_arguments(args_str);
                    current_pos = abs_pos + 1 + close_paren + 1;
                } else {
                    current_pos = abs_pos + 1 + name_end;
                }
            } else {
                current_pos = abs_pos + 1 + name_end;
            }

            directives.push(GraphQLDirective {
                name: directive_name,
                arguments,
            });
        }

        directives
    }

    /// Parse directive arguments
    fn parse_directive_arguments(&self, args_str: &str) -> HashMap<String, DirectiveValue> {
        let mut arguments = HashMap::new();

        for arg in args_str.split(',') {
            let arg = arg.trim();
            if arg.is_empty() {
                continue;
            }

            if let Some((name, value_str)) = arg.split_once(':') {
                let name = name.trim().to_string();
                let value_str = value_str.trim();

                if let Some(value) = self.parse_directive_value(value_str) {
                    arguments.insert(name, value);
                }
            }
        }

        arguments
    }

    /// Parse a directive value
    fn parse_directive_value(&self, value_str: &str) -> Option<DirectiveValue> {
        Self::parse_directive_value_impl(value_str)
    }

    /// Parse directive value implementation (static to avoid recursion warning)
    fn parse_directive_value_impl(value_str: &str) -> Option<DirectiveValue> {
        let value_str = value_str.trim();

        // String literal
        if value_str.starts_with('"') && value_str.ends_with('"') {
            let s = value_str[1..value_str.len() - 1].to_string();
            return Some(DirectiveValue::String(s));
        }

        // Boolean
        if value_str == "true" {
            return Some(DirectiveValue::Boolean(true));
        }
        if value_str == "false" {
            return Some(DirectiveValue::Boolean(false));
        }

        // Integer
        if let Ok(i) = value_str.parse::<i64>() {
            return Some(DirectiveValue::Int(i));
        }

        // Float
        if let Ok(f) = value_str.parse::<f64>() {
            return Some(DirectiveValue::Float(f));
        }

        // List (simplified - handle basic cases)
        if value_str.starts_with('[') && value_str.ends_with(']') {
            let inner = &value_str[1..value_str.len() - 1];
            let items: Vec<DirectiveValue> = inner
                .split(',')
                .filter_map(|s| Self::parse_directive_value_impl(s.trim()))
                .collect();
            return Some(DirectiveValue::List(items));
        }

        None
    }

    /// Convert field directives to TensorLogic constraint expressions
    pub fn directives_to_constraints(&self, type_name: &str, field: &GraphQLField) -> Vec<TLExpr> {
        let mut constraints = Vec::new();
        let field_var = format!("{}_{}", type_name, field.name);

        for directive in &field.directives {
            match directive.name.as_str() {
                "constraint" => {
                    constraints.extend(self.constraint_directive_to_expr(&field_var, directive));
                }
                "range" => {
                    constraints.extend(self.range_directive_to_expr(&field_var, directive));
                }
                "length" => {
                    constraints.extend(self.length_directive_to_expr(&field_var, directive));
                }
                "pattern" => {
                    constraints.extend(self.pattern_directive_to_expr(&field_var, directive));
                }
                "uniqueItems" => {
                    constraints.push(self.unique_items_directive_to_expr(&field_var));
                }
                _ => {} // Ignore unknown directives
            }
        }

        constraints
    }

    /// Convert @constraint directive to TL expressions
    fn constraint_directive_to_expr(
        &self,
        field_var: &str,
        directive: &GraphQLDirective,
    ) -> Vec<TLExpr> {
        let mut constraints = Vec::new();

        // Handle minLength
        if let Some(DirectiveValue::Int(min_len)) = directive.arguments.get("minLength") {
            let expr = TLExpr::pred(
                "minLength",
                vec![Term::var(field_var), Term::constant(min_len.to_string())],
            );
            constraints.push(expr);
        }

        // Handle maxLength
        if let Some(DirectiveValue::Int(max_len)) = directive.arguments.get("maxLength") {
            let expr = TLExpr::pred(
                "maxLength",
                vec![Term::var(field_var), Term::constant(max_len.to_string())],
            );
            constraints.push(expr);
        }

        // Handle min (numeric)
        if let Some(DirectiveValue::Int(min)) = directive.arguments.get("min") {
            let expr = TLExpr::pred(
                "greaterOrEqual",
                vec![Term::var(field_var), Term::constant(min.to_string())],
            );
            constraints.push(expr);
        }
        if let Some(DirectiveValue::Float(min)) = directive.arguments.get("min") {
            let expr = TLExpr::pred(
                "greaterOrEqual",
                vec![Term::var(field_var), Term::constant(min.to_string())],
            );
            constraints.push(expr);
        }

        // Handle max (numeric)
        if let Some(DirectiveValue::Int(max)) = directive.arguments.get("max") {
            let expr = TLExpr::pred(
                "lessOrEqual",
                vec![Term::var(field_var), Term::constant(max.to_string())],
            );
            constraints.push(expr);
        }
        if let Some(DirectiveValue::Float(max)) = directive.arguments.get("max") {
            let expr = TLExpr::pred(
                "lessOrEqual",
                vec![Term::var(field_var), Term::constant(max.to_string())],
            );
            constraints.push(expr);
        }

        // Handle pattern (regex)
        if let Some(DirectiveValue::String(pattern)) = directive.arguments.get("pattern") {
            let expr = TLExpr::pred(
                "matches",
                vec![Term::var(field_var), Term::constant(pattern)],
            );
            constraints.push(expr);
        }

        // Handle format (email, url, etc.)
        if let Some(DirectiveValue::String(format)) = directive.arguments.get("format") {
            let expr = TLExpr::pred(
                format!("isValid{}", capitalize_first(format)),
                vec![Term::var(field_var)],
            );
            constraints.push(expr);
        }

        constraints
    }

    /// Convert @range directive to TL expression
    fn range_directive_to_expr(
        &self,
        field_var: &str,
        directive: &GraphQLDirective,
    ) -> Vec<TLExpr> {
        let mut constraints = Vec::new();

        if let Some(DirectiveValue::Int(min)) = directive.arguments.get("min") {
            let expr = TLExpr::pred(
                "greaterOrEqual",
                vec![Term::var(field_var), Term::constant(min.to_string())],
            );
            constraints.push(expr);
        }

        if let Some(DirectiveValue::Int(max)) = directive.arguments.get("max") {
            let expr = TLExpr::pred(
                "lessOrEqual",
                vec![Term::var(field_var), Term::constant(max.to_string())],
            );
            constraints.push(expr);
        }

        constraints
    }

    /// Convert @length directive to TL expression
    fn length_directive_to_expr(
        &self,
        field_var: &str,
        directive: &GraphQLDirective,
    ) -> Vec<TLExpr> {
        let mut constraints = Vec::new();

        if let Some(DirectiveValue::Int(min)) = directive.arguments.get("min") {
            let expr = TLExpr::pred(
                "minLength",
                vec![Term::var(field_var), Term::constant(min.to_string())],
            );
            constraints.push(expr);
        }

        if let Some(DirectiveValue::Int(max)) = directive.arguments.get("max") {
            let expr = TLExpr::pred(
                "maxLength",
                vec![Term::var(field_var), Term::constant(max.to_string())],
            );
            constraints.push(expr);
        }

        constraints
    }

    /// Convert @pattern directive to TL expression
    fn pattern_directive_to_expr(
        &self,
        field_var: &str,
        directive: &GraphQLDirective,
    ) -> Vec<TLExpr> {
        if let Some(DirectiveValue::String(pattern)) = directive.arguments.get("value") {
            vec![TLExpr::pred(
                "matches",
                vec![Term::var(field_var), Term::constant(pattern)],
            )]
        } else {
            vec![]
        }
    }

    /// Convert @uniqueItems directive to TL expression
    fn unique_items_directive_to_expr(&self, field_var: &str) -> TLExpr {
        TLExpr::pred("allUnique", vec![Term::var(field_var)])
    }

    /// Get all constraints for a type
    pub fn get_constraints(&self, type_name: &str) -> Vec<TLExpr> {
        let mut constraints = Vec::new();

        if let Some(type_def) = self.types.get(type_name) {
            for field in &type_def.fields {
                constraints.extend(self.directives_to_constraints(type_name, field));
            }
        }

        constraints
    }

    /// Get parsed types
    pub fn types(&self) -> &HashMap<String, GraphQLType> {
        &self.types
    }
}

/// Capitalize first letter of a string
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
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

    // ====== Directive Tests ======

    #[test]
    fn test_parse_directive_simple() {
        let converter = GraphQLConverter::new();
        let directives = converter.parse_directives("name: String! @deprecated");

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].name, "deprecated");
        assert!(directives[0].arguments.is_empty());
    }

    #[test]
    fn test_parse_directive_with_arguments() {
        let converter = GraphQLConverter::new();
        let directives = converter.parse_directives(r#"age: Int @constraint(min: 0, max: 120)"#);

        assert_eq!(directives.len(), 1);
        assert_eq!(directives[0].name, "constraint");
        assert_eq!(directives[0].arguments.len(), 2);

        assert_eq!(
            directives[0].arguments.get("min"),
            Some(&DirectiveValue::Int(0))
        );
        assert_eq!(
            directives[0].arguments.get("max"),
            Some(&DirectiveValue::Int(120))
        );
    }

    #[test]
    fn test_parse_directive_string_argument() {
        let converter = GraphQLConverter::new();
        let directives =
            converter.parse_directives(r#"email: String @constraint(pattern: "^[a-z]+$")"#);

        assert_eq!(directives.len(), 1);
        assert_eq!(
            directives[0].arguments.get("pattern"),
            Some(&DirectiveValue::String("^[a-z]+$".to_string()))
        );
    }

    #[test]
    fn test_parse_multiple_directives() {
        let converter = GraphQLConverter::new();
        let directives = converter
            .parse_directives(r#"name: String @length(min: 3, max: 50) @pattern(value: "[a-z]+")"#);

        assert_eq!(directives.len(), 2);
        assert_eq!(directives[0].name, "length");
        assert_eq!(directives[1].name, "pattern");
    }

    #[test]
    fn test_field_with_directive_parsing() {
        let schema = r#"
            type User {
                age: Int @constraint(min: 0, max: 120)
            }
        "#;

        let mut converter = GraphQLConverter::new();
        converter.parse_schema(schema).unwrap();

        let user_type = converter.types().get("User").unwrap();
        assert_eq!(user_type.fields.len(), 1);

        let age_field = &user_type.fields[0];
        assert_eq!(age_field.directives.len(), 1);
        assert_eq!(age_field.directives[0].name, "constraint");
    }

    #[test]
    fn test_constraint_directive_to_expr() {
        let converter = GraphQLConverter::new();

        let mut directive_args = HashMap::new();
        directive_args.insert("min".to_string(), DirectiveValue::Int(0));
        directive_args.insert("max".to_string(), DirectiveValue::Int(120));

        let directive = GraphQLDirective {
            name: "constraint".to_string(),
            arguments: directive_args,
        };

        let constraints = converter.constraint_directive_to_expr("User_age", &directive);

        assert_eq!(constraints.len(), 2);
        // Should have greaterOrEqual and lessOrEqual predicates
        let expr_strs: Vec<String> = constraints.iter().map(|e| format!("{:?}", e)).collect();
        assert!(expr_strs
            .iter()
            .any(|s| s.contains("greaterOrEqual") && s.contains("User_age")));
        assert!(expr_strs
            .iter()
            .any(|s| s.contains("lessOrEqual") && s.contains("User_age")));
    }

    #[test]
    fn test_length_directive_to_expr() {
        let converter = GraphQLConverter::new();

        let mut directive_args = HashMap::new();
        directive_args.insert("min".to_string(), DirectiveValue::Int(3));
        directive_args.insert("max".to_string(), DirectiveValue::Int(50));

        let directive = GraphQLDirective {
            name: "length".to_string(),
            arguments: directive_args,
        };

        let constraints = converter.length_directive_to_expr("User_name", &directive);

        assert_eq!(constraints.len(), 2);
        let expr_strs: Vec<String> = constraints.iter().map(|e| format!("{:?}", e)).collect();
        assert!(expr_strs
            .iter()
            .any(|s| s.contains("minLength") && s.contains("User_name")));
        assert!(expr_strs
            .iter()
            .any(|s| s.contains("maxLength") && s.contains("User_name")));
    }

    #[test]
    fn test_pattern_directive_to_expr() {
        let converter = GraphQLConverter::new();

        let mut directive_args = HashMap::new();
        directive_args.insert(
            "value".to_string(),
            DirectiveValue::String("^[a-z]+$".to_string()),
        );

        let directive = GraphQLDirective {
            name: "pattern".to_string(),
            arguments: directive_args,
        };

        let constraints = converter.pattern_directive_to_expr("User_username", &directive);

        assert_eq!(constraints.len(), 1);
        let expr_str = format!("{:?}", constraints[0]);
        assert!(expr_str.contains("matches"));
        assert!(expr_str.contains("User_username"));
        assert!(expr_str.contains("^[a-z]+$"));
    }

    #[test]
    fn test_range_directive_to_expr() {
        let converter = GraphQLConverter::new();

        let mut directive_args = HashMap::new();
        directive_args.insert("min".to_string(), DirectiveValue::Int(18));
        directive_args.insert("max".to_string(), DirectiveValue::Int(65));

        let directive = GraphQLDirective {
            name: "range".to_string(),
            arguments: directive_args,
        };

        let constraints = converter.range_directive_to_expr("User_age", &directive);

        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_unique_items_directive() {
        let converter = GraphQLConverter::new();
        let expr = converter.unique_items_directive_to_expr("User_tags");

        let expr_str = format!("{:?}", expr);
        assert!(expr_str.contains("allUnique"));
        assert!(expr_str.contains("User_tags"));
    }

    #[test]
    fn test_get_constraints_for_type() {
        let schema = r#"
            type Product {
                price: Float @constraint(min: 0.0, max: 10000.0)
                name: String @length(min: 1, max: 100)
                sku: String @pattern(value: "^[A-Z0-9]+$")
            }
        "#;

        let mut converter = GraphQLConverter::new();
        converter.parse_schema(schema).unwrap();

        let constraints = converter.get_constraints("Product");

        // Should have constraints from all three fields
        assert!(constraints.len() >= 5); // 2 from price, 2 from name, 1 from sku
    }

    #[test]
    fn test_format_directive_to_expr() {
        let converter = GraphQLConverter::new();

        let mut directive_args = HashMap::new();
        directive_args.insert(
            "format".to_string(),
            DirectiveValue::String("email".to_string()),
        );

        let directive = GraphQLDirective {
            name: "constraint".to_string(),
            arguments: directive_args,
        };

        let constraints = converter.constraint_directive_to_expr("User_email", &directive);

        assert_eq!(constraints.len(), 1);
        let expr_str = format!("{:?}", constraints[0]);
        assert!(expr_str.contains("isValidEmail"));
        assert!(expr_str.contains("User_email"));
    }

    #[test]
    fn test_directive_value_parsing_boolean() {
        let converter = GraphQLConverter::new();

        assert_eq!(
            converter.parse_directive_value("true"),
            Some(DirectiveValue::Boolean(true))
        );
        assert_eq!(
            converter.parse_directive_value("false"),
            Some(DirectiveValue::Boolean(false))
        );
    }

    #[test]
    fn test_directive_value_parsing_numbers() {
        let converter = GraphQLConverter::new();

        assert_eq!(
            converter.parse_directive_value("42"),
            Some(DirectiveValue::Int(42))
        );
        assert_eq!(
            converter.parse_directive_value("2.5"),
            Some(DirectiveValue::Float(2.5))
        );
    }

    #[test]
    fn test_directive_value_parsing_list() {
        let converter = GraphQLConverter::new();

        if let Some(DirectiveValue::List(items)) = converter.parse_directive_value("[1, 2, 3]") {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0], DirectiveValue::Int(1));
            assert_eq!(items[1], DirectiveValue::Int(2));
            assert_eq!(items[2], DirectiveValue::Int(3));
        } else {
            panic!("Expected List");
        }
    }

    #[test]
    fn test_complex_directive_scenario() {
        let schema = r#"
            type User {
                username: String! @constraint(minLength: 3, maxLength: 20, pattern: "^[a-z0-9_]+$")
                age: Int! @range(min: 13, max: 120)
                email: String! @constraint(format: "email")
                tags: [String!] @uniqueItems
            }
        "#;

        let mut converter = GraphQLConverter::new();
        converter.parse_schema(schema).unwrap();

        let user_type = converter.types().get("User").unwrap();

        // Verify username field has constraint directive
        let username_field = &user_type
            .fields
            .iter()
            .find(|f| f.name == "username")
            .unwrap();
        assert!(!username_field.directives.is_empty());

        // Verify all constraints can be extracted
        let constraints = converter.get_constraints("User");
        assert!(constraints.len() >= 7); // Multiple constraints from multiple fields
    }

    #[test]
    fn test_capitalize_first() {
        assert_eq!(capitalize_first("email"), "Email");
        assert_eq!(capitalize_first("url"), "Url");
        assert_eq!(capitalize_first(""), "");
        assert_eq!(capitalize_first("A"), "A");
    }
}
