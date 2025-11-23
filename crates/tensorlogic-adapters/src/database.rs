//! Database integration for schema persistence.
//!
//! This module provides functionality to store and retrieve symbol tables
//! from relational databases. Supported databases:
//! - SQLite (via rusqlite) - embedded, file-based
//! - PostgreSQL (via tokio-postgres) - server-based, multi-user
//!
//! The database schema includes tables for:
//! - Domains (with cardinality and metadata)
//! - Predicates (with arity, argument domains, and constraints)
//! - Variables (with domain bindings)
//! - Schema versioning and change history

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{AdapterError, SymbolTable};

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

/// In-memory database implementation for testing and development.
///
/// This provides a simple in-memory store that implements the SchemaDatabase trait
/// without requiring external database dependencies. Useful for:
/// - Testing
/// - Development
/// - Small-scale applications
/// - Temporary storage
pub struct MemoryDatabase {
    schemas: HashMap<SchemaId, StoredSchema>,
    next_id: u64,
    name_index: HashMap<String, Vec<SchemaId>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredSchema {
    id: SchemaId,
    name: String,
    version: u32,
    table: SymbolTable,
    created_at: u64,
    updated_at: u64,
    description: Option<String>,
}

impl MemoryDatabase {
    /// Create a new empty memory database.
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            next_id: 1,
            name_index: HashMap::new(),
        }
    }

    /// Get current timestamp (Unix epoch seconds).
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Find latest version for a schema name.
    fn find_latest_version(&self, name: &str) -> Option<SchemaId> {
        self.name_index.get(name).and_then(|ids| {
            ids.iter()
                .filter_map(|id| self.schemas.get(id))
                .max_by_key(|s| s.version)
                .map(|s| s.id)
        })
    }
}

impl Default for MemoryDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaDatabase for MemoryDatabase {
    fn store_schema(&mut self, name: &str, table: &SymbolTable) -> Result<SchemaId, AdapterError> {
        let now = Self::current_timestamp();

        // Check if schema with this name exists
        let version = if let Some(existing_id) = self.find_latest_version(name) {
            if let Some(existing) = self.schemas.get(&existing_id) {
                existing.version + 1
            } else {
                1
            }
        } else {
            1
        };

        let id = SchemaId(self.next_id);
        self.next_id += 1;

        let stored = StoredSchema {
            id,
            name: name.to_string(),
            version,
            table: table.clone(),
            created_at: now,
            updated_at: now,
            description: None,
        };

        self.schemas.insert(id, stored);

        // Update name index
        self.name_index
            .entry(name.to_string())
            .or_default()
            .push(id);

        Ok(id)
    }

    fn load_schema(&self, id: SchemaId) -> Result<SymbolTable, AdapterError> {
        self.schemas
            .get(&id)
            .map(|s| s.table.clone())
            .ok_or_else(|| {
                AdapterError::InvalidOperation(format!("Schema with ID {:?} not found", id))
            })
    }

    fn load_schema_by_name(&self, name: &str) -> Result<SymbolTable, AdapterError> {
        let id = self.find_latest_version(name).ok_or_else(|| {
            AdapterError::InvalidOperation(format!("Schema '{}' not found", name))
        })?;

        self.load_schema(id)
    }

    fn list_schemas(&self) -> Result<Vec<SchemaMetadata>, AdapterError> {
        let mut metadata: Vec<SchemaMetadata> = self
            .schemas
            .values()
            .map(|s| SchemaMetadata {
                id: s.id,
                name: s.name.clone(),
                version: s.version,
                created_at: s.created_at,
                updated_at: s.updated_at,
                num_domains: s.table.domains.len(),
                num_predicates: s.table.predicates.len(),
                num_variables: s.table.variables.len(),
                description: s.description.clone(),
            })
            .collect();

        metadata.sort_by_key(|m| m.name.clone());
        Ok(metadata)
    }

    fn delete_schema(&mut self, id: SchemaId) -> Result<(), AdapterError> {
        if let Some(schema) = self.schemas.remove(&id) {
            // Remove from name index
            if let Some(ids) = self.name_index.get_mut(&schema.name) {
                ids.retain(|&i| i != id);
                if ids.is_empty() {
                    self.name_index.remove(&schema.name);
                }
            }
            Ok(())
        } else {
            Err(AdapterError::InvalidOperation(format!(
                "Schema with ID {:?} not found",
                id
            )))
        }
    }

    fn search_schemas(&self, pattern: &str) -> Result<Vec<SchemaMetadata>, AdapterError> {
        let pattern_lower = pattern.to_lowercase();
        let mut results: Vec<SchemaMetadata> = self
            .schemas
            .values()
            .filter(|s| s.name.to_lowercase().contains(&pattern_lower))
            .map(|s| SchemaMetadata {
                id: s.id,
                name: s.name.clone(),
                version: s.version,
                created_at: s.created_at,
                updated_at: s.updated_at,
                num_domains: s.table.domains.len(),
                num_predicates: s.table.predicates.len(),
                num_variables: s.table.variables.len(),
                description: s.description.clone(),
            })
            .collect();

        results.sort_by_key(|m| m.name.clone());
        Ok(results)
    }

    fn get_schema_history(&self, name: &str) -> Result<Vec<SchemaVersion>, AdapterError> {
        let ids = self.name_index.get(name).ok_or_else(|| {
            AdapterError::InvalidOperation(format!("Schema '{}' not found", name))
        })?;

        let mut versions: Vec<SchemaVersion> = ids
            .iter()
            .filter_map(|id| {
                self.schemas.get(id).map(|s| SchemaVersion {
                    version: s.version,
                    timestamp: s.created_at,
                    description: format!("Version {}", s.version),
                    schema_id: s.id,
                })
            })
            .collect();

        versions.sort_by_key(|v| v.version);
        Ok(versions)
    }
}

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
        VALUES (?, ?, ?, ?, ?)
        "#
    }

    /// Generate INSERT query for storing a predicate.
    pub fn insert_predicate_sql() -> &'static str {
        r#"
        INSERT INTO predicates (schema_id, name, arity, description, constraints, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        "#
    }

    /// Generate INSERT query for storing a predicate argument.
    pub fn insert_predicate_arg_sql() -> &'static str {
        r#"
        INSERT INTO predicate_arguments (predicate_id, position, domain_name)
        VALUES (?, ?, ?)
        "#
    }

    /// Generate INSERT query for storing a variable.
    pub fn insert_variable_sql() -> &'static str {
        r#"
        INSERT INTO variables (schema_id, name, domain_name)
        VALUES (?, ?, ?)
        "#
    }

    /// Generate SELECT query for loading a schema.
    pub fn select_schema_sql() -> &'static str {
        "SELECT id, name, version, created_at, updated_at, description FROM schemas WHERE id = ?"
    }

    /// Generate SELECT query for loading domains.
    pub fn select_domains_sql() -> &'static str {
        "SELECT name, cardinality, description, metadata FROM domains WHERE schema_id = ?"
    }

    /// Generate SELECT query for loading predicates.
    pub fn select_predicates_sql() -> &'static str {
        "SELECT id, name, arity, description, constraints, metadata FROM predicates WHERE schema_id = ?"
    }

    /// Generate SELECT query for loading predicate arguments.
    pub fn select_predicate_args_sql() -> &'static str {
        "SELECT position, domain_name FROM predicate_arguments WHERE predicate_id = ? ORDER BY position"
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

// ============================================================================
// SQLite Backend Implementation
// ============================================================================

#[cfg(feature = "sqlite")]
mod sqlite_backend {
    use super::*;
    use crate::{DomainInfo, PredicateInfo};
    use rusqlite::{params, Connection, Result as SqliteResult};

    /// SQLite database backend for schema storage.
    ///
    /// This implementation provides persistent storage using SQLite.
    /// The database schema is automatically created on first use.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "sqlite")]
    /// # {
    /// use tensorlogic_adapters::{SQLiteDatabase, SchemaDatabase, SymbolTable, DomainInfo};
    ///
    /// let mut db = SQLiteDatabase::new(":memory:").unwrap();
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    ///
    /// let id = db.store_schema("test", &table).unwrap();
    /// let loaded = db.load_schema(id).unwrap();
    /// # }
    /// ```
    pub struct SQLiteDatabase {
        conn: Connection,
    }

    impl SQLiteDatabase {
        /// Create a new SQLite database at the given path.
        ///
        /// Use `:memory:` for an in-memory database (testing).
        pub fn new(path: &str) -> Result<Self, AdapterError> {
            let conn = Connection::open(path).map_err(|e| {
                AdapterError::InvalidOperation(format!("Failed to open SQLite database: {}", e))
            })?;

            let mut db = Self { conn };
            db.initialize_schema()?;
            Ok(db)
        }

        /// Initialize the database schema (create tables if they don't exist).
        fn initialize_schema(&mut self) -> Result<(), AdapterError> {
            for sql in SchemaDatabaseSQL::create_tables_sql() {
                self.conn.execute(&sql, []).map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to create tables: {}", e))
                })?;
            }
            Ok(())
        }

        /// Get current timestamp (Unix epoch seconds).
        fn current_timestamp() -> u64 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        }

        /// Store schema metadata and return schema_id.
        fn store_schema_metadata(&mut self, name: &str) -> Result<(i64, u32), AdapterError> {
            let now = Self::current_timestamp() as i64;

            // Check if schema exists
            let existing_version: Option<u32> = self
                .conn
                .query_row(
                    "SELECT MAX(version) FROM schemas WHERE name = ?1",
                    params![name],
                    |row| row.get(0),
                )
                .ok()
                .flatten();

            let version = existing_version.map(|v| v + 1).unwrap_or(1);

            self.conn
                .execute(
                    "INSERT INTO schemas (name, version, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
                    params![name, version, now, now],
                )
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to insert schema: {}", e))
                })?;

            let schema_id = self.conn.last_insert_rowid();
            Ok((schema_id, version))
        }

        /// Store domains for a schema.
        fn store_domains(
            &mut self,
            schema_id: i64,
            table: &SymbolTable,
        ) -> Result<(), AdapterError> {
            for (name, domain) in &table.domains {
                let metadata_json = serde_json::to_string(&domain.metadata).ok();
                self.conn
                    .execute(
                        SchemaDatabaseSQL::insert_domain_sql(),
                        params![
                            schema_id,
                            name,
                            domain.cardinality as i64,
                            domain.description.as_ref(),
                            metadata_json
                        ],
                    )
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!("Failed to insert domain: {}", e))
                    })?;
            }
            Ok(())
        }

        /// Store predicates for a schema.
        fn store_predicates(
            &mut self,
            schema_id: i64,
            table: &SymbolTable,
        ) -> Result<(), AdapterError> {
            for (name, predicate) in &table.predicates {
                let constraints_json = serde_json::to_string(&predicate.constraints).ok();
                let metadata_json = serde_json::to_string(&predicate.metadata).ok();

                self.conn
                    .execute(
                        SchemaDatabaseSQL::insert_predicate_sql(),
                        params![
                            schema_id,
                            name,
                            predicate.arg_domains.len() as i64,
                            predicate.description.as_ref(),
                            constraints_json,
                            metadata_json
                        ],
                    )
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!("Failed to insert predicate: {}", e))
                    })?;

                let predicate_id = self.conn.last_insert_rowid();

                // Store argument domains
                for (position, domain_name) in predicate.arg_domains.iter().enumerate() {
                    self.conn
                        .execute(
                            SchemaDatabaseSQL::insert_predicate_arg_sql(),
                            params![predicate_id, position as i64, domain_name],
                        )
                        .map_err(|e| {
                            AdapterError::InvalidOperation(format!(
                                "Failed to insert predicate argument: {}",
                                e
                            ))
                        })?;
                }
            }
            Ok(())
        }

        /// Store variables for a schema.
        fn store_variables(
            &mut self,
            schema_id: i64,
            table: &SymbolTable,
        ) -> Result<(), AdapterError> {
            for (var_name, domain_name) in &table.variables {
                self.conn
                    .execute(
                        SchemaDatabaseSQL::insert_variable_sql(),
                        params![schema_id, var_name, domain_name],
                    )
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!("Failed to insert variable: {}", e))
                    })?;
            }
            Ok(())
        }

        /// Load domains for a schema.
        fn load_domains(
            &self,
            schema_id: i64,
        ) -> Result<indexmap::IndexMap<String, DomainInfo>, AdapterError> {
            let mut stmt = self
                .conn
                .prepare(SchemaDatabaseSQL::select_domains_sql())
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to prepare query: {}", e))
                })?;

            let domains = stmt
                .query_map(params![schema_id], |row| {
                    let name: String = row.get(0)?;
                    let cardinality: i64 = row.get(1)?;
                    let description: Option<String> = row.get(2)?;
                    let metadata_json: Option<String> = row.get(3)?;

                    let mut domain = DomainInfo::new(&name, cardinality as usize);
                    if let Some(desc) = description {
                        domain = domain.with_description(desc);
                    }
                    if let Some(meta_str) = metadata_json {
                        if let Ok(metadata) = serde_json::from_str(&meta_str) {
                            domain.metadata = metadata;
                        }
                    }

                    Ok((name, domain))
                })
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query domains: {}", e))
                })?
                .collect::<SqliteResult<_>>()
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to collect domains: {}", e))
                })?;

            Ok(domains)
        }

        /// Load predicates for a schema.
        fn load_predicates(
            &self,
            schema_id: i64,
        ) -> Result<indexmap::IndexMap<String, PredicateInfo>, AdapterError> {
            let mut stmt = self
                .conn
                .prepare(SchemaDatabaseSQL::select_predicates_sql())
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to prepare query: {}", e))
                })?;

            let predicates = stmt
                .query_map(params![schema_id], |row| {
                    let predicate_id: i64 = row.get(0)?;
                    let name: String = row.get(1)?;
                    let _arity: i64 = row.get(2)?;
                    let description: Option<String> = row.get(3)?;
                    let constraints_json: Option<String> = row.get(4)?;
                    let metadata_json: Option<String> = row.get(5)?;

                    Ok((
                        predicate_id,
                        name,
                        description,
                        constraints_json,
                        metadata_json,
                    ))
                })
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query predicates: {}", e))
                })?
                .collect::<SqliteResult<Vec<_>>>()
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to collect predicates: {}", e))
                })?;

            let mut result = indexmap::IndexMap::new();

            for (predicate_id, name, description, constraints_json, metadata_json) in predicates {
                // Load argument domains
                let mut arg_stmt = self
                    .conn
                    .prepare(SchemaDatabaseSQL::select_predicate_args_sql())
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!("Failed to prepare query: {}", e))
                    })?;

                let arg_domains: Vec<String> = arg_stmt
                    .query_map(params![predicate_id], |row| {
                        let _position: i64 = row.get(0)?;
                        let domain_name: String = row.get(1)?;
                        Ok(domain_name)
                    })
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!(
                            "Failed to query predicate args: {}",
                            e
                        ))
                    })?
                    .collect::<SqliteResult<_>>()
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!(
                            "Failed to collect predicate args: {}",
                            e
                        ))
                    })?;

                let mut predicate = PredicateInfo::new(&name, arg_domains);
                if let Some(desc) = description {
                    predicate = predicate.with_description(desc);
                }
                if let Some(constraints_str) = constraints_json {
                    if let Ok(constraints) = serde_json::from_str(&constraints_str) {
                        predicate.constraints = constraints;
                    }
                }
                if let Some(meta_str) = metadata_json {
                    if let Ok(metadata) = serde_json::from_str(&meta_str) {
                        predicate.metadata = metadata;
                    }
                }

                result.insert(name, predicate);
            }

            Ok(result)
        }

        /// Load variables for a schema.
        fn load_variables(
            &self,
            schema_id: i64,
        ) -> Result<indexmap::IndexMap<String, String>, AdapterError> {
            let mut stmt = self
                .conn
                .prepare("SELECT name, domain_name FROM variables WHERE schema_id = ?")
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to prepare query: {}", e))
                })?;

            let variables = stmt
                .query_map(params![schema_id], |row| {
                    let name: String = row.get(0)?;
                    let domain_name: String = row.get(1)?;
                    Ok((name, domain_name))
                })
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query variables: {}", e))
                })?
                .collect::<SqliteResult<_>>()
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to collect variables: {}", e))
                })?;

            Ok(variables)
        }
    }

    impl SchemaDatabase for SQLiteDatabase {
        fn store_schema(
            &mut self,
            name: &str,
            table: &SymbolTable,
        ) -> Result<SchemaId, AdapterError> {
            let (schema_id, _version) = self.store_schema_metadata(name)?;

            self.store_domains(schema_id, table)?;
            self.store_predicates(schema_id, table)?;
            self.store_variables(schema_id, table)?;

            Ok(SchemaId(schema_id as u64))
        }

        fn load_schema(&self, id: SchemaId) -> Result<SymbolTable, AdapterError> {
            let schema_id = id.0 as i64;

            // Verify schema exists
            let _: i64 = self
                .conn
                .query_row(
                    "SELECT id FROM schemas WHERE id = ?",
                    params![schema_id],
                    |row| row.get(0),
                )
                .map_err(|_| {
                    AdapterError::InvalidOperation(format!("Schema with ID {:?} not found", id))
                })?;

            let mut table = SymbolTable::new();
            table.domains = self.load_domains(schema_id)?;
            table.predicates = self.load_predicates(schema_id)?;
            table.variables = self.load_variables(schema_id)?;

            Ok(table)
        }

        fn load_schema_by_name(&self, name: &str) -> Result<SymbolTable, AdapterError> {
            let schema_id: i64 = self
                .conn
                .query_row(
                    "SELECT id FROM schemas WHERE name = ? ORDER BY version DESC LIMIT 1",
                    params![name],
                    |row| row.get(0),
                )
                .map_err(|_| {
                    AdapterError::InvalidOperation(format!("Schema '{}' not found", name))
                })?;

            self.load_schema(SchemaId(schema_id as u64))
        }

        fn list_schemas(&self) -> Result<Vec<SchemaMetadata>, AdapterError> {
            let mut stmt = self
                .conn
                .prepare(
                    r#"
                    SELECT s.id, s.name, s.version, s.created_at, s.updated_at, s.description,
                           (SELECT COUNT(*) FROM domains WHERE schema_id = s.id) as num_domains,
                           (SELECT COUNT(*) FROM predicates WHERE schema_id = s.id) as num_predicates,
                           (SELECT COUNT(*) FROM variables WHERE schema_id = s.id) as num_variables
                    FROM schemas s
                    ORDER BY s.name, s.version DESC
                    "#,
                )
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to prepare query: {}", e))
                })?;

            let schemas = stmt
                .query_map([], |row| {
                    Ok(SchemaMetadata {
                        id: SchemaId(row.get::<_, i64>(0)? as u64),
                        name: row.get(1)?,
                        version: row.get(2)?,
                        created_at: row.get::<_, i64>(3)? as u64,
                        updated_at: row.get::<_, i64>(4)? as u64,
                        num_domains: row.get::<_, i64>(6)? as usize,
                        num_predicates: row.get::<_, i64>(7)? as usize,
                        num_variables: row.get::<_, i64>(8)? as usize,
                        description: row.get(5)?,
                    })
                })
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query schemas: {}", e))
                })?
                .collect::<SqliteResult<_>>()
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to collect schemas: {}", e))
                })?;

            Ok(schemas)
        }

        fn delete_schema(&mut self, id: SchemaId) -> Result<(), AdapterError> {
            let schema_id = id.0 as i64;
            let affected = self
                .conn
                .execute("DELETE FROM schemas WHERE id = ?", params![schema_id])
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to delete schema: {}", e))
                })?;

            if affected == 0 {
                return Err(AdapterError::InvalidOperation(format!(
                    "Schema with ID {:?} not found",
                    id
                )));
            }

            Ok(())
        }

        fn search_schemas(&self, pattern: &str) -> Result<Vec<SchemaMetadata>, AdapterError> {
            let search_pattern = format!("%{}%", pattern);
            let mut stmt = self
                .conn
                .prepare(
                    r#"
                    SELECT s.id, s.name, s.version, s.created_at, s.updated_at, s.description,
                           (SELECT COUNT(*) FROM domains WHERE schema_id = s.id) as num_domains,
                           (SELECT COUNT(*) FROM predicates WHERE schema_id = s.id) as num_predicates,
                           (SELECT COUNT(*) FROM variables WHERE schema_id = s.id) as num_variables
                    FROM schemas s
                    WHERE s.name LIKE ?
                    ORDER BY s.name, s.version DESC
                    "#,
                )
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to prepare query: {}", e))
                })?;

            let schemas = stmt
                .query_map(params![search_pattern], |row| {
                    Ok(SchemaMetadata {
                        id: SchemaId(row.get::<_, i64>(0)? as u64),
                        name: row.get(1)?,
                        version: row.get(2)?,
                        created_at: row.get::<_, i64>(3)? as u64,
                        updated_at: row.get::<_, i64>(4)? as u64,
                        num_domains: row.get::<_, i64>(6)? as usize,
                        num_predicates: row.get::<_, i64>(7)? as usize,
                        num_variables: row.get::<_, i64>(8)? as usize,
                        description: row.get(5)?,
                    })
                })
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query schemas: {}", e))
                })?
                .collect::<SqliteResult<_>>()
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to collect schemas: {}", e))
                })?;

            Ok(schemas)
        }

        fn get_schema_history(&self, name: &str) -> Result<Vec<SchemaVersion>, AdapterError> {
            let mut stmt = self
                .conn
                .prepare(
                    r#"
                    SELECT version, created_at, id
                    FROM schemas
                    WHERE name = ?
                    ORDER BY version ASC
                    "#,
                )
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to prepare query: {}", e))
                })?;

            let versions: Vec<SchemaVersion> = stmt
                .query_map(params![name], |row| {
                    let version: u32 = row.get(0)?;
                    let timestamp: i64 = row.get(1)?;
                    let schema_id: i64 = row.get(2)?;

                    Ok(SchemaVersion {
                        version,
                        timestamp: timestamp as u64,
                        description: format!("Version {}", version),
                        schema_id: SchemaId(schema_id as u64),
                    })
                })
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query versions: {}", e))
                })?
                .collect::<SqliteResult<_>>()
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to collect versions: {}", e))
                })?;

            if versions.is_empty() {
                return Err(AdapterError::InvalidOperation(format!(
                    "Schema '{}' not found",
                    name
                )));
            }

            Ok(versions)
        }
    }
}

#[cfg(feature = "sqlite")]
pub use sqlite_backend::SQLiteDatabase;

// ============================================================================
// PostgreSQL Backend Implementation
// ============================================================================

#[cfg(feature = "postgres")]
mod postgres_backend {
    use super::*;
    use crate::{DomainInfo, PredicateInfo};
    use tokio_postgres::{Client, NoTls};

    /// PostgreSQL database backend for schema storage.
    ///
    /// This implementation provides persistent storage using PostgreSQL
    /// with async support. The database schema is automatically created on first use.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "postgres")]
    /// # {
    /// use tensorlogic_adapters::{PostgreSQLDatabase, SymbolTable, DomainInfo};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut db = PostgreSQLDatabase::new("host=localhost user=postgres").await?;
    /// let mut table = SymbolTable::new();
    /// table.add_domain(DomainInfo::new("Person", 100))?;
    ///
    /// let id = db.store_schema_async("test", &table).await?;
    /// let loaded = db.load_schema_async(id).await?;
    /// # Ok(())
    /// # }
    /// # }
    /// ```
    pub struct PostgreSQLDatabase {
        client: Client,
    }

    impl PostgreSQLDatabase {
        /// Create a new PostgreSQL database connection.
        ///
        /// The connection string should be in the format:
        /// `host=localhost user=postgres password=password dbname=tensorlogic`
        pub async fn new(connection_string: &str) -> Result<Self, AdapterError> {
            let (client, connection) = tokio_postgres::connect(connection_string, NoTls)
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!(
                        "Failed to connect to PostgreSQL: {}",
                        e
                    ))
                })?;

            // Spawn connection in background
            tokio::spawn(async move {
                if let Err(e) = connection.await {
                    eprintln!("PostgreSQL connection error: {}", e);
                }
            });

            let mut db = Self { client };
            db.initialize_schema_async().await?;
            Ok(db)
        }

        /// Initialize the database schema (create tables if they don't exist).
        async fn initialize_schema_async(&mut self) -> Result<(), AdapterError> {
            for sql in Self::create_tables_postgres_sql() {
                self.client.execute(&sql, &[]).await.map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to create tables: {}", e))
                })?;
            }
            Ok(())
        }

        /// Generate CREATE TABLE statements for PostgreSQL.
        ///
        /// PostgreSQL uses SERIAL instead of AUTOINCREMENT.
        fn create_tables_postgres_sql() -> Vec<String> {
            vec![
                // Schemas table
                r#"
                CREATE TABLE IF NOT EXISTS schemas (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at BIGINT NOT NULL,
                    updated_at BIGINT NOT NULL,
                    description TEXT,
                    UNIQUE(name, version)
                )
                "#
                .to_string(),
                // Domains table
                r#"
                CREATE TABLE IF NOT EXISTS domains (
                    id SERIAL PRIMARY KEY,
                    schema_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    cardinality BIGINT NOT NULL,
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
                    id SERIAL PRIMARY KEY,
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
                    id SERIAL PRIMARY KEY,
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
                    id SERIAL PRIMARY KEY,
                    schema_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    domain_name TEXT NOT NULL,
                    FOREIGN KEY (schema_id) REFERENCES schemas(id) ON DELETE CASCADE,
                    UNIQUE(schema_id, name)
                )
                "#
                .to_string(),
                // Indexes
                "CREATE INDEX IF NOT EXISTS idx_schemas_name ON schemas(name)".to_string(),
                "CREATE INDEX IF NOT EXISTS idx_domains_schema ON domains(schema_id)".to_string(),
                "CREATE INDEX IF NOT EXISTS idx_predicates_schema ON predicates(schema_id)"
                    .to_string(),
            ]
        }

        /// Get current timestamp (Unix epoch seconds).
        fn current_timestamp() -> i64 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
        }

        /// Store a schema asynchronously.
        pub async fn store_schema_async(
            &mut self,
            name: &str,
            table: &SymbolTable,
        ) -> Result<SchemaId, AdapterError> {
            let now = Self::current_timestamp();

            // Check if schema exists
            let existing_version: Option<i32> = self
                .client
                .query_opt("SELECT MAX(version) FROM schemas WHERE name = $1", &[&name])
                .await
                .ok()
                .flatten()
                .and_then(|row| row.get(0));

            let version = existing_version.map(|v| v + 1).unwrap_or(1);

            // Insert schema
            let row = self
                .client
                .query_one(
                    "INSERT INTO schemas (name, version, created_at, updated_at) VALUES ($1, $2, $3, $4) RETURNING id",
                    &[&name, &version, &now, &now],
                )
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to insert schema: {}", e))
                })?;

            let schema_id: i32 = row.get(0);

            // Store domains
            for (domain_name, domain) in &table.domains {
                let metadata_json = serde_json::to_string(&domain.metadata).ok();
                self.client
                    .execute(
                        "INSERT INTO domains (schema_id, name, cardinality, description, metadata) VALUES ($1, $2, $3, $4, $5)",
                        &[
                            &schema_id,
                            &domain_name.as_str(),
                            &(domain.cardinality as i64),
                            &domain.description.as_ref(),
                            &metadata_json.as_ref(),
                        ],
                    )
                    .await
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!("Failed to insert domain: {}", e))
                    })?;
            }

            // Store predicates
            for (predicate_name, predicate) in &table.predicates {
                let constraints_json = serde_json::to_string(&predicate.constraints).ok();
                let metadata_json = serde_json::to_string(&predicate.metadata).ok();

                let pred_row = self
                    .client
                    .query_one(
                        "INSERT INTO predicates (schema_id, name, arity, description, constraints, metadata) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                        &[
                            &schema_id,
                            &predicate_name.as_str(),
                            &(predicate.arg_domains.len() as i32),
                            &predicate.description.as_ref(),
                            &constraints_json.as_ref(),
                            &metadata_json.as_ref(),
                        ],
                    )
                    .await
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!("Failed to insert predicate: {}", e))
                    })?;

                let predicate_id: i32 = pred_row.get(0);

                // Store argument domains
                for (position, domain_name) in predicate.arg_domains.iter().enumerate() {
                    self.client
                        .execute(
                            "INSERT INTO predicate_arguments (predicate_id, position, domain_name) VALUES ($1, $2, $3)",
                            &[&predicate_id, &(position as i32), &domain_name.as_str()],
                        )
                        .await
                        .map_err(|e| {
                            AdapterError::InvalidOperation(format!(
                                "Failed to insert predicate argument: {}",
                                e
                            ))
                        })?;
                }
            }

            // Store variables
            for (var_name, domain_name) in &table.variables {
                self.client
                    .execute(
                        "INSERT INTO variables (schema_id, name, domain_name) VALUES ($1, $2, $3)",
                        &[&schema_id, &var_name.as_str(), &domain_name.as_str()],
                    )
                    .await
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!("Failed to insert variable: {}", e))
                    })?;
            }

            Ok(SchemaId(schema_id as u64))
        }

        /// Load a schema by ID asynchronously.
        pub async fn load_schema_async(&self, id: SchemaId) -> Result<SymbolTable, AdapterError> {
            let schema_id = id.0 as i32;

            // Verify schema exists
            self.client
                .query_opt("SELECT id FROM schemas WHERE id = $1", &[&schema_id])
                .await
                .map_err(|e| AdapterError::InvalidOperation(format!("Database error: {}", e)))?
                .ok_or_else(|| {
                    AdapterError::InvalidOperation(format!("Schema with ID {:?} not found", id))
                })?;

            let mut table = SymbolTable::new();

            // Load domains
            let domain_rows = self
                .client
                .query(
                    "SELECT name, cardinality, description, metadata FROM domains WHERE schema_id = $1",
                    &[&schema_id],
                )
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query domains: {}", e))
                })?;

            for row in domain_rows {
                let name: String = row.get(0);
                let cardinality: i64 = row.get(1);
                let description: Option<String> = row.get(2);
                let metadata_json: Option<String> = row.get(3);

                let mut domain = DomainInfo::new(&name, cardinality as usize);
                if let Some(desc) = description {
                    domain = domain.with_description(desc);
                }
                if let Some(meta_str) = metadata_json {
                    if let Ok(metadata) = serde_json::from_str(&meta_str) {
                        domain.metadata = metadata;
                    }
                }

                table.domains.insert(name, domain);
            }

            // Load predicates
            let predicate_rows = self
                .client
                .query(
                    "SELECT id, name, arity, description, constraints, metadata FROM predicates WHERE schema_id = $1",
                    &[&schema_id],
                )
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query predicates: {}", e))
                })?;

            for pred_row in predicate_rows {
                let predicate_id: i32 = pred_row.get(0);
                let name: String = pred_row.get(1);
                let _arity: i32 = pred_row.get(2);
                let description: Option<String> = pred_row.get(3);
                let constraints_json: Option<String> = pred_row.get(4);
                let metadata_json: Option<String> = pred_row.get(5);

                // Load argument domains
                let arg_rows = self
                    .client
                    .query(
                        "SELECT position, domain_name FROM predicate_arguments WHERE predicate_id = $1 ORDER BY position",
                        &[&predicate_id],
                    )
                    .await
                    .map_err(|e| {
                        AdapterError::InvalidOperation(format!(
                            "Failed to query predicate args: {}",
                            e
                        ))
                    })?;

                let arg_domains: Vec<String> = arg_rows.iter().map(|row| row.get(1)).collect();

                let mut predicate = PredicateInfo::new(&name, arg_domains);
                if let Some(desc) = description {
                    predicate = predicate.with_description(desc);
                }
                if let Some(constraints_str) = constraints_json {
                    if let Ok(constraints) = serde_json::from_str(&constraints_str) {
                        predicate.constraints = constraints;
                    }
                }
                if let Some(meta_str) = metadata_json {
                    if let Ok(metadata) = serde_json::from_str(&meta_str) {
                        predicate.metadata = metadata;
                    }
                }

                table.predicates.insert(name, predicate);
            }

            // Load variables
            let var_rows = self
                .client
                .query(
                    "SELECT name, domain_name FROM variables WHERE schema_id = $1",
                    &[&schema_id],
                )
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query variables: {}", e))
                })?;

            for row in var_rows {
                let name: String = row.get(0);
                let domain_name: String = row.get(1);
                table.variables.insert(name, domain_name);
            }

            Ok(table)
        }

        /// Load a schema by name (latest version) asynchronously.
        pub async fn load_schema_by_name_async(
            &self,
            name: &str,
        ) -> Result<SymbolTable, AdapterError> {
            let row = self
                .client
                .query_opt(
                    "SELECT id FROM schemas WHERE name = $1 ORDER BY version DESC LIMIT 1",
                    &[&name],
                )
                .await
                .map_err(|e| AdapterError::InvalidOperation(format!("Database error: {}", e)))?
                .ok_or_else(|| {
                    AdapterError::InvalidOperation(format!("Schema '{}' not found", name))
                })?;

            let schema_id: i32 = row.get(0);
            self.load_schema_async(SchemaId(schema_id as u64)).await
        }

        /// List all schemas asynchronously.
        pub async fn list_schemas_async(&self) -> Result<Vec<SchemaMetadata>, AdapterError> {
            let rows = self
                .client
                .query(
                    r#"
                    SELECT s.id, s.name, s.version, s.created_at, s.updated_at, s.description,
                           (SELECT COUNT(*) FROM domains WHERE schema_id = s.id) as num_domains,
                           (SELECT COUNT(*) FROM predicates WHERE schema_id = s.id) as num_predicates,
                           (SELECT COUNT(*) FROM variables WHERE schema_id = s.id) as num_variables
                    FROM schemas s
                    ORDER BY s.name, s.version DESC
                    "#,
                    &[],
                )
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query schemas: {}", e))
                })?;

            let schemas = rows
                .iter()
                .map(|row| SchemaMetadata {
                    id: SchemaId(row.get::<_, i32>(0) as u64),
                    name: row.get(1),
                    version: row.get::<_, i32>(2) as u32,
                    created_at: row.get::<_, i64>(3) as u64,
                    updated_at: row.get::<_, i64>(4) as u64,
                    num_domains: row.get::<_, i64>(6) as usize,
                    num_predicates: row.get::<_, i64>(7) as usize,
                    num_variables: row.get::<_, i64>(8) as usize,
                    description: row.get(5),
                })
                .collect();

            Ok(schemas)
        }

        /// Delete a schema by ID asynchronously.
        pub async fn delete_schema_async(&mut self, id: SchemaId) -> Result<(), AdapterError> {
            let schema_id = id.0 as i32;
            let affected = self
                .client
                .execute("DELETE FROM schemas WHERE id = $1", &[&schema_id])
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to delete schema: {}", e))
                })?;

            if affected == 0 {
                return Err(AdapterError::InvalidOperation(format!(
                    "Schema with ID {:?} not found",
                    id
                )));
            }

            Ok(())
        }

        /// Search schemas by pattern asynchronously.
        pub async fn search_schemas_async(
            &self,
            pattern: &str,
        ) -> Result<Vec<SchemaMetadata>, AdapterError> {
            let search_pattern = format!("%{}%", pattern);
            let rows = self
                .client
                .query(
                    r#"
                    SELECT s.id, s.name, s.version, s.created_at, s.updated_at, s.description,
                           (SELECT COUNT(*) FROM domains WHERE schema_id = s.id) as num_domains,
                           (SELECT COUNT(*) FROM predicates WHERE schema_id = s.id) as num_predicates,
                           (SELECT COUNT(*) FROM variables WHERE schema_id = s.id) as num_variables
                    FROM schemas s
                    WHERE s.name LIKE $1
                    ORDER BY s.name, s.version DESC
                    "#,
                    &[&search_pattern],
                )
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query schemas: {}", e))
                })?;

            let schemas = rows
                .iter()
                .map(|row| SchemaMetadata {
                    id: SchemaId(row.get::<_, i32>(0) as u64),
                    name: row.get(1),
                    version: row.get::<_, i32>(2) as u32,
                    created_at: row.get::<_, i64>(3) as u64,
                    updated_at: row.get::<_, i64>(4) as u64,
                    num_domains: row.get::<_, i64>(6) as usize,
                    num_predicates: row.get::<_, i64>(7) as usize,
                    num_variables: row.get::<_, i64>(8) as usize,
                    description: row.get(5),
                })
                .collect();

            Ok(schemas)
        }

        /// Get schema history asynchronously.
        pub async fn get_schema_history_async(
            &self,
            name: &str,
        ) -> Result<Vec<SchemaVersion>, AdapterError> {
            let rows = self
                .client
                .query(
                    "SELECT version, created_at, id FROM schemas WHERE name = $1 ORDER BY version ASC",
                    &[&name],
                )
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!("Failed to query versions: {}", e))
                })?;

            if rows.is_empty() {
                return Err(AdapterError::InvalidOperation(format!(
                    "Schema '{}' not found",
                    name
                )));
            }

            let versions = rows
                .iter()
                .map(|row| {
                    let version: i32 = row.get(0);
                    let timestamp: i64 = row.get(1);
                    let schema_id: i32 = row.get(2);

                    SchemaVersion {
                        version: version as u32,
                        timestamp: timestamp as u64,
                        description: format!("Version {}", version),
                        schema_id: SchemaId(schema_id as u64),
                    }
                })
                .collect();

            Ok(versions)
        }
    }
}

#[cfg(feature = "postgres")]
pub use postgres_backend::PostgreSQLDatabase;

// Tests are in a separate module to keep this file under 2000 lines
#[cfg(test)]
#[path = "database_tests.rs"]
mod tests;
