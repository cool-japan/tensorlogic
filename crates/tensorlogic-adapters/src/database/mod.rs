//! Database integration for schema persistence.
//!
//! This module provides functionality to store and retrieve symbol tables
//! from relational databases. Supported databases:
//! - SQLite (via oxisql-sqlite-compat, Pure Rust) - embedded, file-based
//! - PostgreSQL (via tokio-postgres) - server-based, multi-user
//!
//! The database schema includes tables for:
//! - Domains (with cardinality and metadata)
//! - Predicates (with arity, argument domains, and constraints)
//! - Variables (with domain bindings)
//! - Schema versioning and change history

use serde::{Deserialize, Serialize};

use crate::{AdapterError, SymbolTable};

mod memory;
mod sql;

#[cfg(feature = "sqlite")]
mod sqlite_backend;

#[cfg(feature = "postgres")]
mod postgres_backend;

pub use memory::MemoryDatabase;
pub use sql::{DatabaseStats, SchemaDatabaseSQL};

#[cfg(feature = "sqlite")]
pub use sqlite_backend::SQLiteDatabase;

#[cfg(feature = "postgres")]
pub use postgres_backend::PostgreSQLDatabase;

/// Database storage trait for symbol tables.
///
/// Implementations handle the specifics of different database backends.
pub trait SchemaDatabase {
    /// Store a complete symbol table in the database.
    ///
    /// If a schema with the same name exists, it is updated.
    fn store_schema(&mut self, name: &str, table: &SymbolTable) -> Result<SchemaId, AdapterError>;

    /// Load a symbol table by schema ID.
    fn load_schema(&self, id: SchemaId) -> Result<SymbolTable, AdapterError>;

    /// Load a symbol table by name (returns most recent version).
    fn load_schema_by_name(&self, name: &str) -> Result<SymbolTable, AdapterError>;

    /// List all available schemas.
    fn list_schemas(&self) -> Result<Vec<SchemaMetadata>, AdapterError>;

    /// Delete a schema by ID.
    fn delete_schema(&mut self, id: SchemaId) -> Result<(), AdapterError>;

    /// Search schemas by name pattern.
    fn search_schemas(&self, pattern: &str) -> Result<Vec<SchemaMetadata>, AdapterError>;

    /// Get schema history (all versions).
    fn get_schema_history(&self, name: &str) -> Result<Vec<SchemaVersion>, AdapterError>;
}

/// Unique identifier for a stored schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaId(pub u64);

/// Metadata about a stored schema.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaMetadata {
    /// Unique identifier
    pub id: SchemaId,
    /// Schema name
    pub name: String,
    /// Version number
    pub version: u32,
    /// Creation timestamp (Unix epoch)
    pub created_at: u64,
    /// Last modification timestamp
    pub updated_at: u64,
    /// Number of domains
    pub num_domains: usize,
    /// Number of predicates
    pub num_predicates: usize,
    /// Number of variables
    pub num_variables: usize,
    /// Optional description
    pub description: Option<String>,
}

/// Version information for a schema.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Version number
    pub version: u32,
    /// Timestamp
    pub timestamp: u64,
    /// Change description
    pub description: String,
    /// Schema ID for this version
    pub schema_id: SchemaId,
}

// Tests are in a separate module to keep this file under 2000 lines
#[cfg(test)]
#[path = "../database_tests.rs"]
mod tests;
