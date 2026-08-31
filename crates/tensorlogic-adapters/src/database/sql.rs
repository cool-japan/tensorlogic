//! SQL query generation and database statistics utilities.

use super::SchemaDatabase;
use crate::AdapterError;

/// SQL query generator for schema database operations.
///
/// This utility generates SQL queries for creating tables and CRUD operations
/// on schema databases. Can be used with both SQLite and PostgreSQL with
/// minor dialect adjustments.
pub struct SchemaDatabaseSQL;

impl SchemaDatabaseSQL {
    /// Generate CREATE TABLE statements for schema storage.
    pub fn create_tables_sql() -> Vec<String> {
        vec![
            // Schemas table
            r#"
            CREATE TABLE IF NOT EXISTS schemas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                description TEXT,
                UNIQUE(name, version)
            )
            "#
            .to_string(),
            // Domains table
            r#"
            CREATE TABLE IF NOT EXISTS domains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schema_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                cardinality INTEGER NOT NULL,
                description TEXT,
                metadata TEXT,
                FOREIGN KEY (schema_id) REFERENCES schemas(id) ON DELETE CASCADE,
                UNIQUE(schema_id, name)
            )
            "#
            .to_string(),
            // Predicates table
            r#"
            CREATE TABLE IF NOT EXISTS predicates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schema_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                arity INTEGER NOT NULL,
                description TEXT,
                constraints TEXT,
                metadata TEXT,
                FOREIGN KEY (schema_id) REFERENCES schemas(id) ON DELETE CASCADE,
                UNIQUE(schema_id, name)
            )
            "#
            .to_string(),
            // Predicate arguments table
            r#"
            CREATE TABLE IF NOT EXISTS predicate_arguments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicate_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                domain_name TEXT NOT NULL,
                FOREIGN KEY (predicate_id) REFERENCES predicates(id) ON DELETE CASCADE,
                UNIQUE(predicate_id, position)
            )
            "#
            .to_string(),
            // Variables table
            r#"
            CREATE TABLE IF NOT EXISTS variables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schema_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                domain_name TEXT NOT NULL,
                FOREIGN KEY (schema_id) REFERENCES schemas(id) ON DELETE CASCADE,
                UNIQUE(schema_id, name)
            )
            "#
            .to_string(),
            // Indexes for performance
            "CREATE INDEX IF NOT EXISTS idx_schemas_name ON schemas(name)".to_string(),
            "CREATE INDEX IF NOT EXISTS idx_domains_schema ON domains(schema_id)".to_string(),
            "CREATE INDEX IF NOT EXISTS idx_predicates_schema ON predicates(schema_id)".to_string(),
        ]
    }

    /// Generate INSERT query for storing a domain.
    pub fn insert_domain_sql() -> &'static str {
        r#"
        INSERT INTO domains (schema_id, name, cardinality, description, metadata)
        VALUES ($1, $2, $3, $4, $5)
        "#
    }

    /// Generate INSERT query for storing a predicate.
    pub fn insert_predicate_sql() -> &'static str {
        r#"
        INSERT INTO predicates (schema_id, name, arity, description, constraints, metadata)
        VALUES ($1, $2, $3, $4, $5, $6)
        "#
    }

    /// Generate INSERT query for storing a predicate argument.
    pub fn insert_predicate_arg_sql() -> &'static str {
        r#"
        INSERT INTO predicate_arguments (predicate_id, position, domain_name)
        VALUES ($1, $2, $3)
        "#
    }

    /// Generate INSERT query for storing a variable.
    pub fn insert_variable_sql() -> &'static str {
        r#"
        INSERT INTO variables (schema_id, name, domain_name)
        VALUES ($1, $2, $3)
        "#
    }

    /// Generate SELECT query for loading a schema.
    pub fn select_schema_sql() -> &'static str {
        "SELECT id, name, version, created_at, updated_at, description FROM schemas WHERE id = $1"
    }

    /// Generate SELECT query for loading domains.
    pub fn select_domains_sql() -> &'static str {
        "SELECT name, cardinality, description, metadata FROM domains WHERE schema_id = $1"
    }

    /// Generate SELECT query for loading predicates.
    pub fn select_predicates_sql() -> &'static str {
        "SELECT id, name, arity, description, constraints, metadata FROM predicates WHERE schema_id = $1"
    }

    /// Generate SELECT query for loading predicate arguments.
    pub fn select_predicate_args_sql() -> &'static str {
        "SELECT position, domain_name FROM predicate_arguments WHERE predicate_id = $1 ORDER BY position"
    }
}

/// Statistics about database storage.
#[derive(Clone, Debug)]
pub struct DatabaseStats {
    /// Total number of stored schemas
    pub total_schemas: usize,
    /// Total number of domains across all schemas
    pub total_domains: usize,
    /// Total number of predicates across all schemas
    pub total_predicates: usize,
    /// Total database size in bytes (if applicable)
    pub size_bytes: Option<usize>,
}

impl DatabaseStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self {
            total_schemas: 0,
            total_domains: 0,
            total_predicates: 0,
            size_bytes: None,
        }
    }

    /// Calculate statistics from a database implementation.
    pub fn from_database<D: SchemaDatabase>(db: &D) -> Result<Self, AdapterError> {
        let schemas = db.list_schemas()?;
        let total_schemas = schemas.len();
        let total_domains: usize = schemas.iter().map(|s| s.num_domains).sum();
        let total_predicates: usize = schemas.iter().map(|s| s.num_predicates).sum();

        Ok(Self {
            total_schemas,
            total_domains,
            total_predicates,
            size_bytes: None,
        })
    }

    /// Calculate average domains per schema.
    pub fn avg_domains_per_schema(&self) -> f64 {
        if self.total_schemas == 0 {
            0.0
        } else {
            self.total_domains as f64 / self.total_schemas as f64
        }
    }

    /// Calculate average predicates per schema.
    pub fn avg_predicates_per_schema(&self) -> f64 {
        if self.total_schemas == 0 {
            0.0
        } else {
            self.total_predicates as f64 / self.total_schemas as f64
        }
    }
}

impl Default for DatabaseStats {
    fn default() -> Self {
        Self::new()
    }
}
