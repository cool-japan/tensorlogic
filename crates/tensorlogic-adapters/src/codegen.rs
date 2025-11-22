//! Code generation from schemas.
//!
//! This module provides utilities for generating code in various target languages
//! from TensorLogic schemas, enabling type-safe programming and API generation.
//!
//! Supported targets:
//! - Rust: Type definitions with bounds checking
//! - GraphQL: Schema definitions for API development
//! - TypeScript: Interface and type definitions
//! - Python: Type stubs and PyO3 bindings

use std::fmt::Write as FmtWrite;

use crate::{DomainInfo, PredicateInfo, SymbolTable};

/// Code generator for Rust types from schemas.
pub struct RustCodegen {
    /// Module name for generated code
    module_name: String,
    /// Whether to derive common traits
    derive_common: bool,
    /// Whether to include documentation comments
    include_docs: bool,
}

impl RustCodegen {
    /// Create a new Rust code generator.
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            derive_common: true,
            include_docs: true,
        }
    }

    /// Set whether to derive common traits (Clone, Debug, etc.).
    pub fn with_common_derives(mut self, enable: bool) -> Self {
        self.derive_common = enable;
        self
    }

    /// Set whether to include documentation comments.
    pub fn with_docs(mut self, enable: bool) -> Self {
        self.include_docs = enable;
        self
    }

    /// Generate complete Rust module from a symbol table.
    pub fn generate(&self, table: &SymbolTable) -> String {
        let mut code = String::new();

        // Module header
        writeln!(code, "//! Generated from TensorLogic schema.").unwrap();
        writeln!(code, "//! Module: {}", self.module_name).unwrap();
        writeln!(code, "//!").unwrap();
        writeln!(code, "//! This code was automatically generated.").unwrap();
        writeln!(code, "//! DO NOT EDIT MANUALLY.").unwrap();
        writeln!(code).unwrap();

        // Use statements
        writeln!(code, "#![allow(dead_code)]").unwrap();
        writeln!(code).unwrap();

        // Generate domain types
        writeln!(code, "// ============================================").unwrap();
        writeln!(code, "// Domain Types").unwrap();
        writeln!(code, "// ============================================").unwrap();
        writeln!(code).unwrap();

        for domain in table.domains.values() {
            self.generate_domain(&mut code, domain);
            writeln!(code).unwrap();
        }

        // Generate predicate types
        writeln!(code, "// ============================================").unwrap();
        writeln!(code, "// Predicate Types").unwrap();
        writeln!(code, "// ============================================").unwrap();
        writeln!(code).unwrap();

        for predicate in table.predicates.values() {
            self.generate_predicate(&mut code, predicate, table);
            writeln!(code).unwrap();
        }

        // Generate schema metadata type
        writeln!(code, "// ============================================").unwrap();
        writeln!(code, "// Schema Metadata").unwrap();
        writeln!(code, "// ============================================").unwrap();
        writeln!(code).unwrap();
        self.generate_schema_metadata(&mut code, table);

        code
    }

    /// Generate domain type.
    fn generate_domain(&self, code: &mut String, domain: &DomainInfo) {
        if self.include_docs {
            if let Some(ref desc) = domain.description {
                writeln!(code, "/// {}", desc).unwrap();
            } else {
                writeln!(code, "/// Domain: {}", domain.name).unwrap();
            }
            writeln!(code, "///").unwrap();
            writeln!(code, "/// Cardinality: {}", domain.cardinality).unwrap();
        }

        if self.derive_common {
            writeln!(code, "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]").unwrap();
        }

        let type_name = Self::to_type_name(&domain.name);
        writeln!(code, "pub struct {}(pub usize);", type_name).unwrap();
        writeln!(code).unwrap();

        // Generate constructor and accessors
        writeln!(code, "impl {} {{", type_name).unwrap();
        writeln!(
            code,
            "    /// Maximum valid ID for this domain (exclusive)."
        )
        .unwrap();
        writeln!(
            code,
            "    pub const CARDINALITY: usize = {};",
            domain.cardinality
        )
        .unwrap();
        writeln!(code).unwrap();

        writeln!(code, "    /// Create a new {} instance.", type_name).unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// # Panics").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Panics if `id >= {}`.", domain.cardinality).unwrap();
        writeln!(code, "    pub fn new(id: usize) -> Self {{").unwrap();
        writeln!(code, "        assert!(id < Self::CARDINALITY, \"ID {{}} exceeds cardinality {{}}\", id, Self::CARDINALITY);", ).unwrap();
        writeln!(code, "        Self(id)").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        writeln!(
            code,
            "    /// Create a new {} instance without bounds checking.",
            type_name
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// # Safety").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Caller must ensure `id < {}`.",
            domain.cardinality
        )
        .unwrap();
        writeln!(
            code,
            "    pub unsafe fn new_unchecked(id: usize) -> Self {{"
        )
        .unwrap();
        writeln!(code, "        Self(id)").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "    /// Get the underlying ID.").unwrap();
        writeln!(code, "    pub fn id(&self) -> usize {{").unwrap();
        writeln!(code, "        self.0").unwrap();
        writeln!(code, "    }}").unwrap();

        writeln!(code, "}}").unwrap();
    }

    /// Generate predicate type.
    fn generate_predicate(
        &self,
        code: &mut String,
        predicate: &PredicateInfo,
        _table: &SymbolTable,
    ) {
        if self.include_docs {
            if let Some(ref desc) = predicate.description {
                writeln!(code, "/// {}", desc).unwrap();
            } else {
                writeln!(code, "/// Predicate: {}", predicate.name).unwrap();
            }
            writeln!(code, "///").unwrap();
            writeln!(code, "/// Arity: {}", predicate.arg_domains.len()).unwrap();

            if let Some(ref constraints) = predicate.constraints {
                if !constraints.properties.is_empty() {
                    writeln!(code, "///").unwrap();
                    writeln!(code, "/// Properties:").unwrap();
                    for prop in &constraints.properties {
                        writeln!(code, "/// - {:?}", prop).unwrap();
                    }
                }
            }
        }

        if self.derive_common {
            writeln!(code, "#[derive(Clone, Debug, PartialEq, Eq, Hash)]").unwrap();
        }

        let type_name = Self::to_type_name(&predicate.name);

        // Generate struct with typed fields
        if predicate.arg_domains.is_empty() {
            // Nullary predicate
            writeln!(code, "pub struct {};", type_name).unwrap();
        } else if predicate.arg_domains.len() == 1 {
            // Unary predicate
            let domain_type = Self::to_type_name(&predicate.arg_domains[0]);
            writeln!(code, "pub struct {}(pub {});", type_name, domain_type).unwrap();
        } else {
            // N-ary predicate - use tuple struct
            write!(code, "pub struct {}(", type_name).unwrap();
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                if i > 0 {
                    write!(code, ", ").unwrap();
                }
                write!(code, "pub {}", Self::to_type_name(domain_name)).unwrap();
            }
            writeln!(code, ");").unwrap();
        }

        writeln!(code).unwrap();

        // Generate constructor and accessors
        writeln!(code, "impl {} {{", type_name).unwrap();

        if !predicate.arg_domains.is_empty() {
            // Constructor
            writeln!(code, "    /// Create a new {} instance.", type_name).unwrap();
            write!(code, "    pub fn new(").unwrap();
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                if i > 0 {
                    write!(code, ", ").unwrap();
                }
                write!(code, "arg{}: {}", i, Self::to_type_name(domain_name)).unwrap();
            }
            writeln!(code, ") -> Self {{").unwrap();

            if predicate.arg_domains.len() == 1 {
                writeln!(code, "        Self(arg0)").unwrap();
            } else {
                write!(code, "        Self(").unwrap();
                for i in 0..predicate.arg_domains.len() {
                    if i > 0 {
                        write!(code, ", ").unwrap();
                    }
                    write!(code, "arg{}", i).unwrap();
                }
                writeln!(code, ")").unwrap();
            }
            writeln!(code, "    }}").unwrap();
            writeln!(code).unwrap();

            // Accessor methods
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                writeln!(code, "    /// Get argument {}.", i).unwrap();
                writeln!(
                    code,
                    "    pub fn arg{}(&self) -> {} {{",
                    i,
                    Self::to_type_name(domain_name)
                )
                .unwrap();
                if predicate.arg_domains.len() == 1 {
                    writeln!(code, "        self.0").unwrap();
                } else {
                    writeln!(code, "        self.{}", i).unwrap();
                }
                writeln!(code, "    }}").unwrap();
                writeln!(code).unwrap();
            }
        }

        writeln!(code, "}}").unwrap();
    }

    /// Generate schema metadata type.
    fn generate_schema_metadata(&self, code: &mut String, table: &SymbolTable) {
        writeln!(code, "/// Schema metadata and statistics.").unwrap();
        writeln!(code, "pub struct SchemaMetadata;").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "impl SchemaMetadata {{").unwrap();
        writeln!(code, "    /// Number of domains in the schema.").unwrap();
        writeln!(
            code,
            "    pub const DOMAIN_COUNT: usize = {};",
            table.domains.len()
        )
        .unwrap();
        writeln!(code).unwrap();

        writeln!(code, "    /// Number of predicates in the schema.").unwrap();
        writeln!(
            code,
            "    pub const PREDICATE_COUNT: usize = {};",
            table.predicates.len()
        )
        .unwrap();
        writeln!(code).unwrap();

        writeln!(code, "    /// Total cardinality across all domains.").unwrap();
        let total_card: usize = table.domains.values().map(|d| d.cardinality).sum();
        writeln!(
            code,
            "    pub const TOTAL_CARDINALITY: usize = {};",
            total_card
        )
        .unwrap();

        writeln!(code, "}}").unwrap();
    }

    /// Convert a domain/predicate name to a Rust type name (PascalCase).
    fn to_type_name(name: &str) -> String {
        // Simple conversion: capitalize first letter of each word
        name.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            })
            .collect()
    }
}

/// Code generator for GraphQL schemas from symbol tables.
///
/// This generator creates GraphQL type definitions, queries, and mutations
/// from TensorLogic schemas, enabling API development with type-safe schemas.
pub struct GraphQLCodegen {
    /// Schema name
    schema_name: String,
    /// Whether to include descriptions
    include_descriptions: bool,
    /// Whether to generate Query type
    generate_queries: bool,
    /// Whether to generate Mutation type
    generate_mutations: bool,
}

impl GraphQLCodegen {
    /// Create a new GraphQL code generator.
    pub fn new(schema_name: impl Into<String>) -> Self {
        Self {
            schema_name: schema_name.into(),
            include_descriptions: true,
            generate_queries: true,
            generate_mutations: false,
        }
    }

    /// Set whether to include descriptions.
    pub fn with_descriptions(mut self, enable: bool) -> Self {
        self.include_descriptions = enable;
        self
    }

    /// Set whether to generate Query type.
    pub fn with_queries(mut self, enable: bool) -> Self {
        self.generate_queries = enable;
        self
    }

    /// Set whether to generate Mutation type.
    pub fn with_mutations(mut self, enable: bool) -> Self {
        self.generate_mutations = enable;
        self
    }

    /// Generate complete GraphQL schema from a symbol table.
    pub fn generate(&self, table: &SymbolTable) -> String {
        let mut schema = String::new();

        // Schema header
        writeln!(schema, "# Generated GraphQL Schema").unwrap();
        writeln!(schema, "# Schema: {}", self.schema_name).unwrap();
        writeln!(schema, "#").unwrap();
        writeln!(
            schema,
            "# This schema was automatically generated from TensorLogic."
        )
        .unwrap();
        writeln!(schema, "# DO NOT EDIT MANUALLY.").unwrap();
        writeln!(schema).unwrap();

        // Generate domain types
        writeln!(schema, "# ==========================================").unwrap();
        writeln!(schema, "# Domain Types").unwrap();
        writeln!(schema, "# ==========================================").unwrap();
        writeln!(schema).unwrap();

        for domain in table.domains.values() {
            self.generate_domain_type(&mut schema, domain);
            writeln!(schema).unwrap();
        }

        // Generate predicate types
        writeln!(schema, "# ==========================================").unwrap();
        writeln!(schema, "# Predicate Types").unwrap();
        writeln!(schema, "# ==========================================").unwrap();
        writeln!(schema).unwrap();

        for predicate in table.predicates.values() {
            self.generate_predicate_type(&mut schema, predicate, table);
            writeln!(schema).unwrap();
        }

        // Generate Query type
        if self.generate_queries {
            writeln!(schema, "# ==========================================").unwrap();
            writeln!(schema, "# Query Operations").unwrap();
            writeln!(schema, "# ==========================================").unwrap();
            writeln!(schema).unwrap();
            self.generate_query_type(&mut schema, table);
            writeln!(schema).unwrap();
        }

        // Generate Mutation type
        if self.generate_mutations {
            writeln!(schema, "# ==========================================").unwrap();
            writeln!(schema, "# Mutation Operations").unwrap();
            writeln!(schema, "# ==========================================").unwrap();
            writeln!(schema).unwrap();
            self.generate_mutation_type(&mut schema, table);
            writeln!(schema).unwrap();
        }

        // Schema definition
        writeln!(schema, "# ==========================================").unwrap();
        writeln!(schema, "# Schema Definition").unwrap();
        writeln!(schema, "# ==========================================").unwrap();
        writeln!(schema).unwrap();
        writeln!(schema, "schema {{").unwrap();
        if self.generate_queries {
            writeln!(schema, "  query: Query").unwrap();
        }
        if self.generate_mutations {
            writeln!(schema, "  mutation: Mutation").unwrap();
        }
        writeln!(schema, "}}").unwrap();

        schema
    }

    /// Generate GraphQL type for a domain.
    fn generate_domain_type(&self, schema: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_graphql_type_name(&domain.name);

        if self.include_descriptions {
            if let Some(ref desc) = domain.description {
                writeln!(schema, "\"\"\"\n{}\n\"\"\"", desc).unwrap();
            } else {
                writeln!(schema, "\"\"\"\nDomain: {}\n\"\"\"", domain.name).unwrap();
            }
        }

        writeln!(schema, "type {} {{", type_name).unwrap();
        writeln!(schema, "  \"Unique identifier\"").unwrap();
        writeln!(schema, "  id: ID!").unwrap();
        writeln!(
            schema,
            "  \"Integer index (0 to {})\"",
            domain.cardinality - 1
        )
        .unwrap();
        writeln!(schema, "  index: Int!").unwrap();
        writeln!(schema, "}}").unwrap();
    }

    /// Generate GraphQL type for a predicate.
    fn generate_predicate_type(
        &self,
        schema: &mut String,
        predicate: &PredicateInfo,
        _table: &SymbolTable,
    ) {
        let type_name = Self::to_graphql_type_name(&predicate.name);

        if self.include_descriptions {
            if let Some(ref desc) = predicate.description {
                writeln!(schema, "\"\"\"\n{}\n\"\"\"", desc).unwrap();
            } else {
                writeln!(schema, "\"\"\"\nPredicate: {}\n\"\"\"", predicate.name).unwrap();
            }
        }

        writeln!(schema, "type {} {{", type_name).unwrap();

        // Add ID field
        writeln!(schema, "  \"Unique identifier\"").unwrap();
        writeln!(schema, "  id: ID!").unwrap();

        // Add argument fields
        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            let field_name = format!("arg{}", i);
            let field_type = Self::to_graphql_type_name(domain_name);
            writeln!(schema, "  \"Argument {} of type {}\"", i, domain_name).unwrap();
            writeln!(schema, "  {}: {}!", field_name, field_type).unwrap();
        }

        writeln!(schema, "}}").unwrap();
    }

    /// Generate Query type.
    fn generate_query_type(&self, schema: &mut String, table: &SymbolTable) {
        writeln!(schema, "\"\"\"").unwrap();
        writeln!(schema, "Root query type for retrieving data").unwrap();
        writeln!(schema, "\"\"\"").unwrap();
        writeln!(schema, "type Query {{").unwrap();

        // Domain queries
        for domain in table.domains.values() {
            let type_name = Self::to_graphql_type_name(&domain.name);
            let field_name = Self::to_graphql_field_name(&domain.name);

            writeln!(schema, "  \"Get {} by ID\"", domain.name).unwrap();
            writeln!(schema, "  {}(id: ID!): {}", field_name, type_name).unwrap();
            writeln!(schema).unwrap();

            writeln!(schema, "  \"List all {}s\"", domain.name).unwrap();
            writeln!(schema, "  {}s: [{}!]!", field_name, type_name).unwrap();
            writeln!(schema).unwrap();
        }

        // Predicate queries
        for predicate in table.predicates.values() {
            let type_name = Self::to_graphql_type_name(&predicate.name);
            let field_name = Self::to_graphql_field_name(&predicate.name);

            writeln!(schema, "  \"Query {} predicate\"", predicate.name).unwrap();

            // Build query with argument filters
            write!(schema, "  {}(", field_name).unwrap();
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                if i > 0 {
                    write!(schema, ", ").unwrap();
                }
                let arg_type = Self::to_graphql_type_name(domain_name);
                write!(schema, "arg{}: {}", i, arg_type).unwrap();
            }
            writeln!(schema, "): [{}!]!", type_name).unwrap();
            writeln!(schema).unwrap();
        }

        writeln!(schema, "}}").unwrap();
    }

    /// Generate Mutation type.
    fn generate_mutation_type(&self, schema: &mut String, table: &SymbolTable) {
        writeln!(schema, "\"\"\"").unwrap();
        writeln!(schema, "Root mutation type for modifying data").unwrap();
        writeln!(schema, "\"\"\"").unwrap();
        writeln!(schema, "type Mutation {{").unwrap();

        // Predicate mutations (add/remove)
        for predicate in table.predicates.values() {
            let type_name = Self::to_graphql_type_name(&predicate.name);

            // Add mutation
            writeln!(schema, "  \"Add {} instance\"", predicate.name).unwrap();
            write!(schema, "  add{}(", type_name).unwrap();
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                if i > 0 {
                    write!(schema, ", ").unwrap();
                }
                let arg_type = Self::to_graphql_type_name(domain_name);
                write!(schema, "arg{}: {}!", i, arg_type).unwrap();
            }
            writeln!(schema, "): {}!", type_name).unwrap();
            writeln!(schema).unwrap();

            // Remove mutation
            writeln!(schema, "  \"Remove {} instance\"", predicate.name).unwrap();
            writeln!(schema, "  remove{}(id: ID!): Boolean!", type_name).unwrap();
            writeln!(schema).unwrap();
        }

        writeln!(schema, "}}").unwrap();
    }

    /// Convert a name to GraphQL type name (PascalCase).
    fn to_graphql_type_name(name: &str) -> String {
        RustCodegen::to_type_name(name) // Reuse Rust converter
    }

    /// Convert a name to GraphQL field name (camelCase).
    fn to_graphql_field_name(name: &str) -> String {
        let parts: Vec<&str> = name.split('_').collect();
        if parts.is_empty() {
            return String::new();
        }

        let mut result = parts[0].to_lowercase();
        for part in &parts[1..] {
            if let Some(first_char) = part.chars().next() {
                result.push_str(&first_char.to_uppercase().to_string());
                result.push_str(&part[first_char.len_utf8()..]);
            }
        }
        result
    }
}

/// Code generator for TypeScript definitions from symbol tables.
///
/// This generator creates TypeScript interface and type definitions
/// from TensorLogic schemas, enabling type-safe TypeScript development.
pub struct TypeScriptCodegen {
    /// Module name
    module_name: String,
    /// Whether to export types
    export_types: bool,
    /// Whether to include JSDoc comments
    include_jsdoc: bool,
    /// Whether to generate validation functions
    generate_validators: bool,
}

impl TypeScriptCodegen {
    /// Create a new TypeScript code generator.
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            export_types: true,
            include_jsdoc: true,
            generate_validators: true,
        }
    }

    /// Set whether to export types.
    pub fn with_exports(mut self, enable: bool) -> Self {
        self.export_types = enable;
        self
    }

    /// Set whether to include JSDoc comments.
    pub fn with_jsdoc(mut self, enable: bool) -> Self {
        self.include_jsdoc = enable;
        self
    }

    /// Set whether to generate validator functions.
    pub fn with_validators(mut self, enable: bool) -> Self {
        self.generate_validators = enable;
        self
    }

    /// Generate complete TypeScript module from a symbol table.
    pub fn generate(&self, table: &SymbolTable) -> String {
        let mut code = String::new();

        // Module header
        writeln!(code, "/**").unwrap();
        writeln!(code, " * Generated from TensorLogic schema").unwrap();
        writeln!(code, " * Module: {}", self.module_name).unwrap();
        writeln!(code, " *").unwrap();
        writeln!(code, " * This code was automatically generated.").unwrap();
        writeln!(code, " * DO NOT EDIT MANUALLY.").unwrap();
        writeln!(code, " */").unwrap();
        writeln!(code).unwrap();

        // Generate domain types
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code, "// Domain Types").unwrap();
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code).unwrap();

        for domain in table.domains.values() {
            self.generate_domain_type(&mut code, domain);
            writeln!(code).unwrap();
        }

        // Generate predicate types
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code, "// Predicate Types").unwrap();
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code).unwrap();

        for predicate in table.predicates.values() {
            self.generate_predicate_type(&mut code, predicate, table);
            writeln!(code).unwrap();
        }

        // Generate validator functions if enabled
        if self.generate_validators {
            writeln!(code, "// ==========================================").unwrap();
            writeln!(code, "// Validator Functions").unwrap();
            writeln!(code, "// ==========================================").unwrap();
            writeln!(code).unwrap();

            for domain in table.domains.values() {
                self.generate_domain_validator(&mut code, domain);
                writeln!(code).unwrap();
            }
        }

        // Generate schema metadata
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code, "// Schema Metadata").unwrap();
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code).unwrap();
        self.generate_schema_metadata(&mut code, table);

        code
    }

    /// Generate TypeScript interface for a domain.
    fn generate_domain_type(&self, code: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_typescript_type_name(&domain.name);
        let export = if self.export_types { "export " } else { "" };

        if self.include_jsdoc {
            writeln!(code, "/**").unwrap();
            if let Some(ref desc) = domain.description {
                writeln!(code, " * {}", desc).unwrap();
            } else {
                writeln!(code, " * Domain: {}", domain.name).unwrap();
            }
            writeln!(code, " *").unwrap();
            writeln!(code, " * Cardinality: {}", domain.cardinality).unwrap();
            writeln!(code, " */").unwrap();
        }

        writeln!(code, "{}interface {} {{", export, type_name).unwrap();
        writeln!(code, "  readonly id: number;").unwrap();
        writeln!(code, "}}").unwrap();
        writeln!(code).unwrap();

        // Generate branded type for stronger typing
        writeln!(
            code,
            "{}type {}Id = number & {{ readonly __brand: '{}' }};",
            export, type_name, type_name
        )
        .unwrap();
    }

    /// Generate TypeScript interface for a predicate.
    fn generate_predicate_type(
        &self,
        code: &mut String,
        predicate: &PredicateInfo,
        _table: &SymbolTable,
    ) {
        let type_name = Self::to_typescript_type_name(&predicate.name);
        let export = if self.export_types { "export " } else { "" };

        if self.include_jsdoc {
            writeln!(code, "/**").unwrap();
            if let Some(ref desc) = predicate.description {
                writeln!(code, " * {}", desc).unwrap();
            } else {
                writeln!(code, " * Predicate: {}", predicate.name).unwrap();
            }
            writeln!(code, " *").unwrap();
            writeln!(code, " * Arity: {}", predicate.arg_domains.len()).unwrap();

            if let Some(ref constraints) = predicate.constraints {
                if !constraints.properties.is_empty() {
                    writeln!(code, " *").unwrap();
                    writeln!(code, " * Properties:").unwrap();
                    for prop in &constraints.properties {
                        writeln!(code, " * - {:?}", prop).unwrap();
                    }
                }
            }
            writeln!(code, " */").unwrap();
        }

        writeln!(code, "{}interface {} {{", export, type_name).unwrap();
        writeln!(code, "  readonly id: string;").unwrap();

        // Add argument fields
        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            let field_name = format!("arg{}", i);
            let field_type = format!("{}Id", Self::to_typescript_type_name(domain_name));
            writeln!(code, "  readonly {}: {};", field_name, field_type).unwrap();
        }

        writeln!(code, "}}").unwrap();
    }

    /// Generate validator function for a domain.
    fn generate_domain_validator(&self, code: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_typescript_type_name(&domain.name);
        let export = if self.export_types { "export " } else { "" };

        writeln!(code, "/**").unwrap();
        writeln!(code, " * Validate {} ID", type_name).unwrap();
        writeln!(
            code,
            " * @param id - The ID to validate (must be in range [0, {}))",
            domain.cardinality
        )
        .unwrap();
        writeln!(code, " * @returns true if valid, false otherwise").unwrap();
        writeln!(code, " */").unwrap();

        writeln!(
            code,
            "{}function is{}Id(id: number): id is {}Id {{",
            export, type_name, type_name
        )
        .unwrap();
        writeln!(
            code,
            "  return Number.isInteger(id) && id >= 0 && id < {};",
            domain.cardinality
        )
        .unwrap();
        writeln!(code, "}}").unwrap();
    }

    /// Generate schema metadata constant.
    fn generate_schema_metadata(&self, code: &mut String, table: &SymbolTable) {
        let export = if self.export_types { "export " } else { "" };

        writeln!(code, "/**").unwrap();
        writeln!(code, " * Schema metadata and statistics").unwrap();
        writeln!(code, " */").unwrap();

        writeln!(code, "{}const SCHEMA_METADATA = {{", export).unwrap();
        writeln!(code, "  domainCount: {},", table.domains.len()).unwrap();
        writeln!(code, "  predicateCount: {},", table.predicates.len()).unwrap();

        let total_card: usize = table.domains.values().map(|d| d.cardinality).sum();
        writeln!(code, "  totalCardinality: {},", total_card).unwrap();

        writeln!(code, "  domains: {{").unwrap();
        for domain in table.domains.values() {
            writeln!(
                code,
                "    '{}': {{ cardinality: {} }},",
                domain.name, domain.cardinality
            )
            .unwrap();
        }
        writeln!(code, "  }},").unwrap();

        writeln!(code, "  predicates: {{").unwrap();
        for predicate in table.predicates.values() {
            writeln!(
                code,
                "    '{}': {{ arity: {} }},",
                predicate.name,
                predicate.arg_domains.len()
            )
            .unwrap();
        }
        writeln!(code, "  }},").unwrap();

        writeln!(code, "}} as const;").unwrap();
    }

    /// Convert a name to TypeScript type name (PascalCase).
    fn to_typescript_type_name(name: &str) -> String {
        RustCodegen::to_type_name(name) // Reuse Rust converter
    }
}

/// Code generator for Python type stubs and PyO3 bindings.
///
/// This generator creates Python type stubs (.pyi) and optionally PyO3
/// binding code from TensorLogic schemas.
pub struct PythonCodegen {
    /// Module name
    module_name: String,
    /// Whether to generate PyO3 bindings (vs. just stubs)
    generate_pyo3: bool,
    /// Whether to include docstrings
    include_docs: bool,
    /// Whether to generate dataclass decorators
    use_dataclasses: bool,
}

impl PythonCodegen {
    /// Create a new Python code generator.
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            generate_pyo3: false,
            include_docs: true,
            use_dataclasses: true,
        }
    }

    /// Set whether to generate PyO3 bindings.
    pub fn with_pyo3(mut self, enable: bool) -> Self {
        self.generate_pyo3 = enable;
        self
    }

    /// Set whether to include docstrings.
    pub fn with_docs(mut self, enable: bool) -> Self {
        self.include_docs = enable;
        self
    }

    /// Set whether to use dataclasses.
    pub fn with_dataclasses(mut self, enable: bool) -> Self {
        self.use_dataclasses = enable;
        self
    }

    /// Generate complete Python module from a symbol table.
    pub fn generate(&self, table: &SymbolTable) -> String {
        if self.generate_pyo3 {
            self.generate_pyo3_bindings(table)
        } else {
            self.generate_type_stubs(table)
        }
    }

    /// Generate Python type stubs (.pyi file).
    fn generate_type_stubs(&self, table: &SymbolTable) -> String {
        let mut code = String::new();

        // Module header
        writeln!(code, "\"\"\"").unwrap();
        writeln!(code, "Generated from TensorLogic schema").unwrap();
        writeln!(code, "Module: {}", self.module_name).unwrap();
        writeln!(code).unwrap();
        writeln!(code, "This code was automatically generated.").unwrap();
        writeln!(code, "DO NOT EDIT MANUALLY.").unwrap();
        writeln!(code, "\"\"\"").unwrap();
        writeln!(code).unwrap();

        // Imports
        writeln!(code, "from typing import NewType, Final").unwrap();
        if self.use_dataclasses {
            writeln!(code, "from dataclasses import dataclass").unwrap();
        }
        writeln!(code).unwrap();

        // Generate domain types
        writeln!(code, "# ==========================================").unwrap();
        writeln!(code, "# Domain Types").unwrap();
        writeln!(code, "# ==========================================").unwrap();
        writeln!(code).unwrap();

        for domain in table.domains.values() {
            self.generate_domain_stub(&mut code, domain);
            writeln!(code).unwrap();
        }

        // Generate predicate types
        writeln!(code, "# ==========================================").unwrap();
        writeln!(code, "# Predicate Types").unwrap();
        writeln!(code, "# ==========================================").unwrap();
        writeln!(code).unwrap();

        for predicate in table.predicates.values() {
            self.generate_predicate_stub(&mut code, predicate, table);
            writeln!(code).unwrap();
        }

        // Generate schema metadata
        writeln!(code, "# ==========================================").unwrap();
        writeln!(code, "# Schema Metadata").unwrap();
        writeln!(code, "# ==========================================").unwrap();
        writeln!(code).unwrap();
        self.generate_schema_metadata_stub(&mut code, table);

        code
    }

    /// Generate PyO3 Rust bindings.
    fn generate_pyo3_bindings(&self, table: &SymbolTable) -> String {
        let mut code = String::new();

        // Module header
        writeln!(code, "//! PyO3 bindings for TensorLogic schema").unwrap();
        writeln!(code, "//! Module: {}", self.module_name).unwrap();
        writeln!(code, "//!").unwrap();
        writeln!(code, "//! This code was automatically generated.").unwrap();
        writeln!(code, "//! DO NOT EDIT MANUALLY.").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "use pyo3::prelude::*;").unwrap();
        writeln!(code).unwrap();

        // Generate domain classes
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code, "// Domain Types").unwrap();
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code).unwrap();

        for domain in table.domains.values() {
            self.generate_domain_pyo3(&mut code, domain);
            writeln!(code).unwrap();
        }

        // Generate predicate classes
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code, "// Predicate Types").unwrap();
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code).unwrap();

        for predicate in table.predicates.values() {
            self.generate_predicate_pyo3(&mut code, predicate);
            writeln!(code).unwrap();
        }

        // Generate module registration
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code, "// Module Registration").unwrap();
        writeln!(code, "// ==========================================").unwrap();
        writeln!(code).unwrap();
        self.generate_module_registration(&mut code, table);

        code
    }

    /// Generate Python type stub for a domain.
    fn generate_domain_stub(&self, code: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_python_class_name(&domain.name);

        // NewType for branded ID
        writeln!(code, "{} = NewType('{}', int)", type_name, type_name).unwrap();
        writeln!(code).unwrap();

        // Cardinality constant
        writeln!(
            code,
            "{}_CARDINALITY: Final[int] = {}",
            domain.name.to_uppercase(),
            domain.cardinality
        )
        .unwrap();
        writeln!(code).unwrap();

        // Validator function
        writeln!(code, "def is_valid_{}(id: int) -> bool:", domain.name).unwrap();
        if self.include_docs {
            writeln!(code, "    \"\"\"").unwrap();
            if let Some(ref desc) = domain.description {
                writeln!(code, "    {}", desc).unwrap();
                writeln!(code).unwrap();
            }
            writeln!(code, "    Validate {} ID.", type_name).unwrap();
            writeln!(code).unwrap();
            writeln!(code, "    Args:").unwrap();
            writeln!(code, "        id: The ID to validate").unwrap();
            writeln!(code).unwrap();
            writeln!(code, "    Returns:").unwrap();
            writeln!(
                code,
                "        True if id is in range [0, {}), False otherwise",
                domain.cardinality
            )
            .unwrap();
            writeln!(code, "    \"\"\"").unwrap();
        }
        writeln!(code, "    ...").unwrap();
    }

    /// Generate Python type stub for a predicate.
    fn generate_predicate_stub(
        &self,
        code: &mut String,
        predicate: &PredicateInfo,
        _table: &SymbolTable,
    ) {
        let class_name = Self::to_python_class_name(&predicate.name);

        if self.include_docs {
            writeln!(code, "\"\"\"").unwrap();
            if let Some(ref desc) = predicate.description {
                writeln!(code, "{}", desc).unwrap();
            } else {
                writeln!(code, "Predicate: {}", predicate.name).unwrap();
            }
            writeln!(code).unwrap();
            writeln!(code, "Arity: {}", predicate.arg_domains.len()).unwrap();
            writeln!(code, "\"\"\"").unwrap();
        }

        if self.use_dataclasses {
            writeln!(code, "@dataclass(frozen=True)").unwrap();
        }

        writeln!(code, "class {}:", class_name).unwrap();

        if self.include_docs && predicate.description.is_none() {
            writeln!(code, "    \"\"\"{}\"\"\"", predicate.name).unwrap();
        }

        // Add fields
        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            let field_name = format!("arg{}", i);
            let field_type = Self::to_python_class_name(domain_name);
            writeln!(code, "    {}: {}", field_name, field_type).unwrap();
        }

        if predicate.arg_domains.is_empty() {
            writeln!(code, "    pass").unwrap();
        }
    }

    /// Generate PyO3 class for a domain.
    fn generate_domain_pyo3(&self, code: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_python_class_name(&domain.name);

        writeln!(code, "#[pyclass]").unwrap();
        writeln!(code, "#[derive(Clone, Copy, Debug)]").unwrap();
        writeln!(code, "pub struct {} {{", type_name).unwrap();
        writeln!(code, "    #[pyo3(get)]").unwrap();
        writeln!(code, "    pub id: usize,").unwrap();
        writeln!(code, "}}").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "#[pymethods]").unwrap();
        writeln!(code, "impl {} {{", type_name).unwrap();

        // Constructor
        writeln!(code, "    #[new]").unwrap();
        writeln!(code, "    pub fn new(id: usize) -> PyResult<Self> {{").unwrap();
        writeln!(code, "        if id >= {} {{", domain.cardinality).unwrap();
        writeln!(
            code,
            "            return Err(pyo3::exceptions::PyValueError::new_err("
        )
        .unwrap();
        writeln!(
            code,
            "                format!(\"ID {{}} exceeds cardinality {}\", id)",
            domain.cardinality
        )
        .unwrap();
        writeln!(code, "            ));").unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "        Ok(Self {{ id }})").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code).unwrap();

        // String representation
        writeln!(code, "    fn __repr__(&self) -> String {{").unwrap();
        writeln!(code, "        format!(\"{}({{}})\", self.id)", type_name).unwrap();
        writeln!(code, "    }}").unwrap();

        writeln!(code, "}}").unwrap();
    }

    /// Generate PyO3 class for a predicate.
    fn generate_predicate_pyo3(&self, code: &mut String, predicate: &PredicateInfo) {
        let type_name = Self::to_python_class_name(&predicate.name);

        writeln!(code, "#[pyclass]").unwrap();
        writeln!(code, "#[derive(Clone, Debug)]").unwrap();
        writeln!(code, "pub struct {} {{", type_name).unwrap();

        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            let field_type = Self::to_python_class_name(domain_name);
            writeln!(code, "    #[pyo3(get)]").unwrap();
            writeln!(code, "    pub arg{}: {},", i, field_type).unwrap();
        }

        writeln!(code, "}}").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "#[pymethods]").unwrap();
        writeln!(code, "impl {} {{", type_name).unwrap();

        // Constructor
        writeln!(code, "    #[new]").unwrap();
        write!(code, "    pub fn new(").unwrap();
        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            if i > 0 {
                write!(code, ", ").unwrap();
            }
            write!(
                code,
                "arg{}: {}",
                i,
                Self::to_python_class_name(domain_name)
            )
            .unwrap();
        }
        writeln!(code, ") -> Self {{").unwrap();

        if predicate.arg_domains.is_empty() {
            writeln!(code, "        Self {{}}").unwrap();
        } else {
            write!(code, "        Self {{ ").unwrap();
            for i in 0..predicate.arg_domains.len() {
                if i > 0 {
                    write!(code, ", ").unwrap();
                }
                write!(code, "arg{}", i).unwrap();
            }
            writeln!(code, " }}").unwrap();
        }
        writeln!(code, "    }}").unwrap();

        writeln!(code, "}}").unwrap();
    }

    /// Generate module registration for PyO3.
    fn generate_module_registration(&self, code: &mut String, table: &SymbolTable) {
        writeln!(code, "#[pymodule]").unwrap();
        writeln!(
            code,
            "fn {}(_py: Python, m: &PyModule) -> PyResult<()> {{",
            self.module_name.replace('-', "_")
        )
        .unwrap();

        // Register domain classes
        for domain in table.domains.values() {
            let type_name = Self::to_python_class_name(&domain.name);
            writeln!(code, "    m.add_class::<{}>()?;", type_name).unwrap();
        }

        // Register predicate classes
        for predicate in table.predicates.values() {
            let type_name = Self::to_python_class_name(&predicate.name);
            writeln!(code, "    m.add_class::<{}>()?;", type_name).unwrap();
        }

        writeln!(code, "    Ok(())").unwrap();
        writeln!(code, "}}").unwrap();
    }

    /// Generate schema metadata stub.
    fn generate_schema_metadata_stub(&self, code: &mut String, table: &SymbolTable) {
        writeln!(code, "class SchemaMetadata:").unwrap();
        if self.include_docs {
            writeln!(code, "    \"\"\"Schema metadata and statistics\"\"\"").unwrap();
        }
        writeln!(
            code,
            "    DOMAIN_COUNT: Final[int] = {}",
            table.domains.len()
        )
        .unwrap();
        writeln!(
            code,
            "    PREDICATE_COUNT: Final[int] = {}",
            table.predicates.len()
        )
        .unwrap();

        let total_card: usize = table.domains.values().map(|d| d.cardinality).sum();
        writeln!(code, "    TOTAL_CARDINALITY: Final[int] = {}", total_card).unwrap();
    }

    /// Convert a name to Python class name (PascalCase).
    fn to_python_class_name(name: &str) -> String {
        RustCodegen::to_type_name(name) // Reuse Rust converter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_type_name() {
        assert_eq!(RustCodegen::to_type_name("person"), "Person");
        assert_eq!(RustCodegen::to_type_name("Person"), "Person");
        assert_eq!(RustCodegen::to_type_name("student_record"), "StudentRecord");
        assert_eq!(RustCodegen::to_type_name("HTTP_Request"), "HTTPRequest");
    }

    #[test]
    fn test_generate_simple_schema() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person entity"))
            .unwrap();

        let codegen = RustCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("pub struct Person(pub usize);"));
        assert!(code.contains("CARDINALITY: usize = 100"));
        assert!(code.contains("A person entity"));
    }

    #[test]
    fn test_generate_predicate() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let pred = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
            .with_description("Person knows another person");
        table.add_predicate(pred).unwrap();

        let codegen = RustCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("pub struct Knows(pub Person, pub Person);"));
        assert!(code.contains("Person knows another person"));
        assert!(code.contains("pub fn new(arg0: Person, arg1: Person)"));
    }

    #[test]
    fn test_generate_without_docs() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person"))
            .unwrap();

        let codegen = RustCodegen::new("test_module").with_docs(false);
        let code = codegen.generate(&table);

        // Should not contain descriptive doc comments (module header is ok)
        assert!(!code.contains("/// A person"));
        assert!(!code.contains("/// Cardinality:"));
        // Should still contain the struct
        assert!(code.contains("pub struct Person"));
    }

    #[test]
    fn test_generate_without_derives() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let codegen = RustCodegen::new("test_module").with_common_derives(false);
        let code = codegen.generate(&table);

        // Should not contain derive attributes
        assert!(!code.contains("#[derive("));
        // Should still contain the struct
        assert!(code.contains("pub struct Person"));
    }

    #[test]
    fn test_generate_unary_predicate() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let pred = PredicateInfo::new("adult", vec!["Person".to_string()]);
        table.add_predicate(pred).unwrap();

        let codegen = RustCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("pub struct Adult(pub Person);"));
        assert!(code.contains("pub fn new(arg0: Person)"));
    }

    #[test]
    fn test_generate_metadata() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Course", 50)).unwrap();

        let pred = PredicateInfo::new("enrolled", vec!["Person".to_string(), "Course".to_string()]);
        table.add_predicate(pred).unwrap();

        let codegen = RustCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("DOMAIN_COUNT: usize = 2"));
        assert!(code.contains("PREDICATE_COUNT: usize = 1"));
        assert!(code.contains("TOTAL_CARDINALITY: usize = 150"));
    }

    // GraphQL code generation tests
    #[test]
    fn test_graphql_field_name_conversion() {
        assert_eq!(GraphQLCodegen::to_graphql_field_name("person"), "person");
        assert_eq!(
            GraphQLCodegen::to_graphql_field_name("student_record"),
            "studentRecord"
        );
        assert_eq!(
            GraphQLCodegen::to_graphql_field_name("http_request"),
            "httpRequest"
        );
    }

    #[test]
    fn test_graphql_generate_simple_schema() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person entity"))
            .unwrap();

        let codegen = GraphQLCodegen::new("TestSchema");
        let schema = codegen.generate(&table);

        assert!(schema.contains("# Generated GraphQL Schema"));
        assert!(schema.contains("type Person {"));
        assert!(schema.contains("id: ID!"));
        assert!(schema.contains("index: Int!"));
        assert!(schema.contains("A person entity"));
    }

    #[test]
    fn test_graphql_generate_with_predicate() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let pred = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
            .with_description("Person knows another person");
        table.add_predicate(pred).unwrap();

        let codegen = GraphQLCodegen::new("TestSchema");
        let schema = codegen.generate(&table);

        assert!(schema.contains("type Knows {"));
        assert!(schema.contains("arg0: Person!"));
        assert!(schema.contains("arg1: Person!"));
        assert!(schema.contains("Person knows another person"));
    }

    #[test]
    fn test_graphql_generate_queries() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let pred = PredicateInfo::new("adult", vec!["Person".to_string()]);
        table.add_predicate(pred).unwrap();

        let codegen = GraphQLCodegen::new("TestSchema").with_queries(true);
        let schema = codegen.generate(&table);

        assert!(schema.contains("type Query {"));
        assert!(schema.contains("person(id: ID!): Person"));
        assert!(schema.contains("persons: [Person!]!"));
        assert!(schema.contains("adult(arg0: Person): [Adult!]!"));
    }

    #[test]
    fn test_graphql_generate_mutations() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let pred = PredicateInfo::new("adult", vec!["Person".to_string()]);
        table.add_predicate(pred).unwrap();

        let codegen = GraphQLCodegen::new("TestSchema")
            .with_queries(false)
            .with_mutations(true);
        let schema = codegen.generate(&table);

        assert!(schema.contains("type Mutation {"));
        assert!(schema.contains("addAdult(arg0: Person!): Adult!"));
        assert!(schema.contains("removeAdult(id: ID!): Boolean!"));
        assert!(!schema.contains("type Query"));
    }

    #[test]
    fn test_graphql_without_descriptions() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person"))
            .unwrap();

        let codegen = GraphQLCodegen::new("TestSchema").with_descriptions(false);
        let schema = codegen.generate(&table);

        assert!(!schema.contains("A person"));
        assert!(schema.contains("type Person {"));
    }

    #[test]
    fn test_graphql_schema_definition() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let codegen = GraphQLCodegen::new("TestSchema")
            .with_queries(true)
            .with_mutations(true);
        let schema = codegen.generate(&table);

        assert!(schema.contains("schema {"));
        assert!(schema.contains("query: Query"));
        assert!(schema.contains("mutation: Mutation"));
    }

    #[test]
    fn test_graphql_complex_predicate() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Student", 80)).unwrap();
        table.add_domain(DomainInfo::new("Course", 50)).unwrap();
        table.add_domain(DomainInfo::new("Grade", 5)).unwrap();

        let pred = PredicateInfo::new(
            "grade",
            vec![
                "Student".to_string(),
                "Course".to_string(),
                "Grade".to_string(),
            ],
        );
        table.add_predicate(pred).unwrap();

        let codegen = GraphQLCodegen::new("TestSchema");
        let schema = codegen.generate(&table);

        assert!(schema.contains("type Grade {"));
        assert!(schema.contains("arg0: Student!"));
        assert!(schema.contains("arg1: Course!"));
        assert!(schema.contains("arg2: Grade!"));
    }

    // TypeScript code generation tests
    #[test]
    fn test_typescript_generate_simple_schema() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person entity"))
            .unwrap();

        let codegen = TypeScriptCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("export interface Person {"));
        assert!(code.contains("readonly id: number;"));
        assert!(code.contains("export type PersonId = number"));
        assert!(code.contains("A person entity"));
        assert!(code.contains("Cardinality: 100"));
    }

    #[test]
    fn test_typescript_generate_with_predicate() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let pred = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
            .with_description("Person knows another person");
        table.add_predicate(pred).unwrap();

        let codegen = TypeScriptCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("export interface Knows {"));
        assert!(code.contains("readonly arg0: PersonId;"));
        assert!(code.contains("readonly arg1: PersonId;"));
        assert!(code.contains("Person knows another person"));
    }

    #[test]
    fn test_typescript_generate_validators() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let codegen = TypeScriptCodegen::new("test_module").with_validators(true);
        let code = codegen.generate(&table);

        assert!(code.contains("export function isPersonId(id: number): id is PersonId {"));
        assert!(code.contains("Number.isInteger(id) && id >= 0 && id < 100"));
    }

    #[test]
    fn test_typescript_without_exports() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let codegen = TypeScriptCodegen::new("test_module").with_exports(false);
        let code = codegen.generate(&table);

        // Should not have export keywords
        let export_count = code.matches("export ").count();
        assert_eq!(export_count, 0);
        assert!(code.contains("interface Person {"));
    }

    #[test]
    fn test_typescript_without_jsdoc() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person"))
            .unwrap();

        let codegen = TypeScriptCodegen::new("test_module").with_jsdoc(false);
        let code = codegen.generate(&table);

        assert!(!code.contains("A person"));
        assert!(code.contains("export interface Person {"));
    }

    #[test]
    fn test_typescript_metadata_generation() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Course", 50)).unwrap();

        let pred = PredicateInfo::new("enrolled", vec!["Person".to_string(), "Course".to_string()]);
        table.add_predicate(pred).unwrap();

        let codegen = TypeScriptCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("export const SCHEMA_METADATA = {"));
        assert!(code.contains("domainCount: 2,"));
        assert!(code.contains("predicateCount: 1,"));
        assert!(code.contains("totalCardinality: 150,"));
    }

    // Python code generation tests
    #[test]
    fn test_python_generate_simple_stubs() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person entity"))
            .unwrap();

        let codegen = PythonCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("Person = NewType('Person', int)"));
        assert!(code.contains("PERSON_CARDINALITY: Final[int] = 100"));
        assert!(code.contains("def is_valid_Person(id: int) -> bool:"));
        assert!(code.contains("A person entity"));
    }

    #[test]
    fn test_python_generate_with_predicate() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let pred = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
            .with_description("Person knows another person");
        table.add_predicate(pred).unwrap();

        let codegen = PythonCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("@dataclass(frozen=True)"));
        assert!(code.contains("class Knows:"));
        assert!(code.contains("Person knows another person"));
        assert!(code.contains("arg0: Person"));
        assert!(code.contains("arg1: Person"));
    }

    #[test]
    fn test_python_generate_pyo3_bindings() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let codegen = PythonCodegen::new("test_module").with_pyo3(true);
        let code = codegen.generate(&table);

        assert!(code.contains("use pyo3::prelude::*;"));
        assert!(code.contains("#[pyclass]"));
        assert!(code.contains("pub struct Person {"));
        assert!(code.contains("#[pyo3(get)]"));
        assert!(code.contains("pub id: usize,"));
        assert!(code.contains("#[pymethods]"));
        assert!(code.contains("#[new]"));
        assert!(code.contains("fn __repr__(&self)"));
    }

    #[test]
    fn test_python_without_docs() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person"))
            .unwrap();

        let codegen = PythonCodegen::new("test_module").with_docs(false);
        let code = codegen.generate(&table);

        // Should not contain docstrings (except the module header)
        let docstring_count = code.matches("\"\"\"").count();
        assert_eq!(docstring_count, 2); // Only module header
    }

    #[test]
    fn test_python_without_dataclasses() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let pred = PredicateInfo::new("adult", vec!["Person".to_string()]);
        table.add_predicate(pred).unwrap();

        let codegen = PythonCodegen::new("test_module").with_dataclasses(false);
        let code = codegen.generate(&table);

        assert!(!code.contains("@dataclass"));
        assert!(code.contains("class Adult:"));
    }

    #[test]
    fn test_python_pyo3_module_registration() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Course", 50)).unwrap();

        let pred = PredicateInfo::new("enrolled", vec!["Person".to_string(), "Course".to_string()]);
        table.add_predicate(pred).unwrap();

        let codegen = PythonCodegen::new("test_module").with_pyo3(true);
        let code = codegen.generate(&table);

        assert!(code.contains("#[pymodule]"));
        assert!(code.contains("fn test_module(_py: Python, m: &PyModule)"));
        assert!(code.contains("m.add_class::<Person>()?;"));
        assert!(code.contains("m.add_class::<Course>()?;"));
        assert!(code.contains("m.add_class::<Enrolled>()?;"));
    }

    #[test]
    fn test_python_metadata_generation() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Course", 50)).unwrap();

        let codegen = PythonCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("class SchemaMetadata:"));
        assert!(code.contains("DOMAIN_COUNT: Final[int] = 2"));
        assert!(code.contains("TOTAL_CARDINALITY: Final[int] = 150"));
    }
}
