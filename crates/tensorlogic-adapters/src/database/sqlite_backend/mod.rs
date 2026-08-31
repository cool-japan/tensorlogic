//! SQLite database backend for schema storage.
//!
//! This backend is built on [`oxisql_sqlite_compat::SqliteConnection`], the
//! COOLJAPAN Pure-Rust SQLite-compatible engine (Limbo-backed via OxiSQL).
//! It contains no C/C++ dependency — `libsqlite3` is never linked.
//!
//! # Sync ↔ async bridge
//!
//! The [`SchemaDatabase`] trait is synchronous, whereas OxiSQL is asynchronous.
//! Each [`SQLiteDatabase`] therefore owns a dedicated current-thread Tokio runtime
//! and drives every database operation through `runtime.block_on(...)`.  This keeps
//! the backend self-contained: it works in plain synchronous test functions and on
//! worker threads alike, without requiring an ambient runtime.
//!
//! # Parameter placeholders
//!
//! OxiSQL uses `$1`, `$2`, … positional placeholders (the `oxisql-sqlite-compat`
//! layer rewrites them to `?` for Limbo internally).  All SQL in this module and in
//! `sql.rs` uses the `$N` form.

use oxisql_core::{Connection, ToSqlValue, Value};
use oxisql_sqlite_compat::SqliteConnection;
use tokio::runtime::Runtime;

use super::{SchemaDatabase, SchemaDatabaseSQL, SchemaId, SchemaMetadata, SchemaVersion};
use crate::{AdapterError, SymbolTable};

mod io;

/// SQLite database backend for schema storage.
///
/// This implementation provides persistent storage using SQLite via the
/// COOLJAPAN Pure-Rust OxiSQL engine (no C/FFI dependencies).
/// The database schema is automatically created on first use.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "sqlite")]
/// # {
/// use tensorlogic_adapters::{SQLiteDatabase, SchemaDatabase, SymbolTable, DomainInfo};
///
/// let mut db = SQLiteDatabase::new(":memory:").expect("open");
/// let mut table = SymbolTable::new();
/// table.add_domain(DomainInfo::new("Person", 100)).expect("add");
///
/// let id = db.store_schema("test", &table).expect("store");
/// let loaded = db.load_schema(id).expect("load");
/// # }
/// ```
pub struct SQLiteDatabase {
    pub(super) conn: SqliteConnection,
    pub(super) runtime: Runtime,
}

/// Build a current-thread Tokio runtime for the sync↔async bridge.
fn build_runtime() -> Result<Runtime, AdapterError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| AdapterError::InvalidOperation(format!("Failed to build Tokio runtime: {e}")))
}

/// Map an OxiSQL error to [`AdapterError`].
fn map_oxi(e: impl std::fmt::Display) -> AdapterError {
    AdapterError::InvalidOperation(format!("SQLite error: {e}"))
}

/// Extract an `i64` from a result row at the given column index.
pub(super) fn col_i64(row: &oxisql_core::Row, idx: usize) -> Result<i64, AdapterError> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Ok(*n),
        Some(other) => Err(AdapterError::InvalidOperation(format!(
            "column {idx}: expected integer, got {}",
            other.type_name()
        ))),
        None => Err(AdapterError::InvalidOperation(format!(
            "column {idx} missing from result row"
        ))),
    }
}

/// Extract a `String` from a result row at the given column index.
pub(super) fn col_text(row: &oxisql_core::Row, idx: usize) -> Result<String, AdapterError> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(other) => Err(AdapterError::InvalidOperation(format!(
            "column {idx}: expected text, got {}",
            other.type_name()
        ))),
        None => Err(AdapterError::InvalidOperation(format!(
            "column {idx} missing from result row"
        ))),
    }
}

/// Extract an optional `String` from a result row (NULL or missing → `None`).
pub(super) fn col_opt_text(
    row: &oxisql_core::Row,
    idx: usize,
) -> Result<Option<String>, AdapterError> {
    match row.get_by_index(idx) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::Text(s)) => Ok(Some(s.clone())),
        Some(other) => Err(AdapterError::InvalidOperation(format!(
            "column {idx}: expected text or null, got {}",
            other.type_name()
        ))),
    }
}

impl SQLiteDatabase {
    /// Create a new SQLite database at the given path.
    ///
    /// Use `:memory:` for an in-memory database (testing).
    pub fn new(path: &str) -> Result<Self, AdapterError> {
        let runtime = build_runtime()?;
        let conn = runtime
            .block_on(SqliteConnection::open(path))
            .map_err(map_oxi)?;
        let db = Self { conn, runtime };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Initialize the database schema (create tables if they don't exist).
    fn initialize_schema(&self) -> Result<(), AdapterError> {
        for sql in SchemaDatabaseSQL::create_tables_sql() {
            self.exec(&sql, &[])?;
        }
        Ok(())
    }

    /// Execute a statement, returning the number of affected rows.
    pub(super) fn exec(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<u64, AdapterError> {
        self.runtime
            .block_on(self.conn.execute(sql, params))
            .map_err(map_oxi)
    }

    /// Execute a query and return all result rows.
    pub(super) fn query(
        &self,
        sql: &str,
        params: &[&dyn ToSqlValue],
    ) -> Result<Vec<oxisql_core::Row>, AdapterError> {
        self.runtime
            .block_on(self.conn.query(sql, params))
            .map_err(map_oxi)
    }

    /// Count rows in a table that match a given `schema_id`.
    ///
    /// Used to compute `num_domains`, `num_predicates`, and `num_variables`
    /// without correlated subqueries (which the OxiSQL engine does not support).
    fn count_for_schema(&self, table: &str, schema_id: i64) -> Result<usize, AdapterError> {
        let sql = format!("SELECT COUNT(*) FROM {} WHERE schema_id = $1", table);
        let rows = self.query(&sql, &[&schema_id])?;
        match rows.first().and_then(|r| r.get_by_index(0)) {
            Some(Value::I64(n)) => Ok(*n as usize),
            // Aggregate on empty table returns Null.
            Some(Value::Null) | None => Ok(0),
            Some(other) => Err(AdapterError::InvalidOperation(format!(
                "count_for_schema: unexpected value {}",
                other.type_name()
            ))),
        }
    }

    /// Build a [`SchemaMetadata`] from a schema base-row (cols 0–5) plus per-table COUNTs.
    ///
    /// The row must have columns: `id, name, version, created_at, updated_at, description`.
    fn schema_metadata_from_row(
        &self,
        row: &oxisql_core::Row,
    ) -> Result<SchemaMetadata, AdapterError> {
        let schema_id_i64 = col_i64(row, 0)?;
        let name = col_text(row, 1)?;
        let version = col_i64(row, 2)? as u32;
        let created_at = col_i64(row, 3)? as u64;
        let updated_at = col_i64(row, 4)? as u64;
        let description = col_opt_text(row, 5)?;
        let num_domains = self.count_for_schema("domains", schema_id_i64)?;
        let num_predicates = self.count_for_schema("predicates", schema_id_i64)?;
        let num_variables = self.count_for_schema("variables", schema_id_i64)?;
        Ok(SchemaMetadata {
            id: SchemaId(schema_id_i64 as u64),
            name,
            version,
            created_at,
            updated_at,
            description,
            num_domains,
            num_predicates,
            num_variables,
        })
    }

    /// Return the rowid of the most recently inserted row.
    ///
    /// Must be called immediately after a successful `INSERT` on the same
    /// connection, before any subsequent write, to avoid interleaving on the
    /// single-threaded runtime.
    pub(super) fn last_insert_rowid(&self) -> Result<i64, AdapterError> {
        let rows = self.query("SELECT last_insert_rowid()", &[])?;
        match rows.first().and_then(|r| r.get_by_index(0)) {
            Some(Value::I64(n)) => Ok(*n),
            other => Err(AdapterError::InvalidOperation(format!(
                "last_insert_rowid: unexpected value {other:?}"
            ))),
        }
    }
}

impl SchemaDatabase for SQLiteDatabase {
    fn store_schema(&mut self, name: &str, table: &SymbolTable) -> Result<SchemaId, AdapterError> {
        let (schema_id, _version) = io::store_schema_metadata(self, name)?;

        io::store_domains(self, schema_id, table)?;
        io::store_predicates(self, schema_id, table)?;
        io::store_variables(self, schema_id, table)?;

        Ok(SchemaId(schema_id as u64))
    }

    fn load_schema(&self, id: SchemaId) -> Result<SymbolTable, AdapterError> {
        let schema_id = id.0 as i64;

        // Verify schema exists
        let rows = self.query("SELECT id FROM schemas WHERE id = $1", &[&schema_id])?;
        if rows.first().and_then(|r| r.get_by_index(0)).is_none() {
            return Err(AdapterError::InvalidOperation(format!(
                "Schema with ID {:?} not found",
                id
            )));
        }

        let mut table = SymbolTable::new();
        table.domains = io::load_domains(self, schema_id)?;
        table.predicates = io::load_predicates(self, schema_id)?;
        table.variables = io::load_variables(self, schema_id)?;

        Ok(table)
    }

    fn load_schema_by_name(&self, name: &str) -> Result<SymbolTable, AdapterError> {
        let rows = self.query(
            "SELECT id FROM schemas WHERE name = $1 ORDER BY version DESC LIMIT 1",
            &[&name],
        )?;
        let schema_id = match rows.first().and_then(|r| r.get_by_index(0)) {
            Some(Value::I64(n)) => *n,
            _ => {
                return Err(AdapterError::InvalidOperation(format!(
                    "Schema '{}' not found",
                    name
                )))
            }
        };
        self.load_schema(SchemaId(schema_id as u64))
    }

    fn list_schemas(&self) -> Result<Vec<SchemaMetadata>, AdapterError> {
        // Correlated subqueries are not supported by the engine; use separate COUNT queries.
        let schema_rows = self.query(
            "SELECT id, name, version, created_at, updated_at, description \
             FROM schemas \
             ORDER BY name, version DESC",
            &[],
        )?;

        schema_rows
            .iter()
            .map(|row| self.schema_metadata_from_row(row))
            .collect::<Result<Vec<_>, AdapterError>>()
    }

    fn delete_schema(&mut self, id: SchemaId) -> Result<(), AdapterError> {
        let schema_id = id.0 as i64;
        let affected = self.exec("DELETE FROM schemas WHERE id = $1", &[&schema_id])?;

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
        // Correlated subqueries are not supported by the engine; use separate COUNT queries.
        let schema_rows = self.query(
            "SELECT id, name, version, created_at, updated_at, description \
             FROM schemas \
             WHERE name LIKE $1 \
             ORDER BY name, version DESC",
            &[&search_pattern],
        )?;

        schema_rows
            .iter()
            .map(|row| self.schema_metadata_from_row(row))
            .collect::<Result<Vec<_>, AdapterError>>()
    }

    fn get_schema_history(&self, name: &str) -> Result<Vec<SchemaVersion>, AdapterError> {
        let rows = self.query(
            r#"
            SELECT version, created_at, id
            FROM schemas
            WHERE name = $1
            ORDER BY version ASC
            "#,
            &[&name],
        )?;

        let versions: Vec<SchemaVersion> = rows
            .iter()
            .map(|row| {
                let version = col_i64(row, 0)? as u32;
                let timestamp = col_i64(row, 1)? as u64;
                let schema_id = col_i64(row, 2)?;
                Ok(SchemaVersion {
                    version,
                    timestamp,
                    description: format!("Version {}", version),
                    schema_id: SchemaId(schema_id as u64),
                })
            })
            .collect::<Result<Vec<_>, AdapterError>>()?;

        if versions.is_empty() {
            return Err(AdapterError::InvalidOperation(format!(
                "Schema '{}' not found",
                name
            )));
        }

        Ok(versions)
    }
}
