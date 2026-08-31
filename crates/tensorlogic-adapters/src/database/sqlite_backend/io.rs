//! Internal SQLite row serialization and deserialization helpers.

use oxisql_core::ToSqlValue;

use super::{col_i64, col_opt_text, col_text, SQLiteDatabase, SchemaDatabaseSQL};
use crate::{AdapterError, DomainInfo, PredicateInfo, SymbolTable};

/// Get current timestamp (Unix epoch seconds).
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Store schema metadata and return `(schema_id, version)`.
pub(super) fn store_schema_metadata(
    db: &SQLiteDatabase,
    name: &str,
) -> Result<(i64, u32), AdapterError> {
    let now = current_timestamp() as i64;

    // Check if a previous version of this schema exists.
    let rows = db.query("SELECT MAX(version) FROM schemas WHERE name = $1", &[&name])?;
    let existing_version: Option<u32> = match rows.first().and_then(|r| r.get_by_index(0)) {
        Some(oxisql_core::Value::I64(n)) => Some(*n as u32),
        _ => None, // Value::Null (aggregate on empty table) or missing → None
    };

    let version = existing_version.map(|v| v + 1).unwrap_or(1);
    let version_i64 = version as i64; // u32 has no ToSqlValue impl

    db.exec(
        "INSERT INTO schemas (name, version, created_at, updated_at) VALUES ($1, $2, $3, $4)",
        &[&name, &version_i64, &now, &now],
    )?;

    let schema_id = db.last_insert_rowid()?;
    Ok((schema_id, version))
}

/// Store domains for a schema.
pub(super) fn store_domains(
    db: &SQLiteDatabase,
    schema_id: i64,
    table: &SymbolTable,
) -> Result<(), AdapterError> {
    for (name, domain) in &table.domains {
        let cardinality = domain.cardinality as i64;
        let metadata_json: Option<String> = serde_json::to_string(&domain.metadata).ok();
        let description: Option<&str> = domain.description.as_deref();
        let meta_ref: Option<&str> = metadata_json.as_deref();
        let params: &[&dyn ToSqlValue] = &[
            &schema_id,
            &name.as_str(),
            &cardinality,
            &description,
            &meta_ref,
        ];
        db.exec(SchemaDatabaseSQL::insert_domain_sql(), params)
            .map_err(|e| AdapterError::InvalidOperation(format!("Failed to insert domain: {e}")))?;
    }
    Ok(())
}

/// Store predicates for a schema.
pub(super) fn store_predicates(
    db: &SQLiteDatabase,
    schema_id: i64,
    table: &SymbolTable,
) -> Result<(), AdapterError> {
    for (name, predicate) in &table.predicates {
        let arity = predicate.arg_domains.len() as i64;
        let constraints_json: Option<String> = serde_json::to_string(&predicate.constraints).ok();
        let metadata_json: Option<String> = serde_json::to_string(&predicate.metadata).ok();
        let description: Option<&str> = predicate.description.as_deref();
        let constraints_ref: Option<&str> = constraints_json.as_deref();
        let meta_ref: Option<&str> = metadata_json.as_deref();

        let params: &[&dyn ToSqlValue] = &[
            &schema_id,
            &name.as_str(),
            &arity,
            &description,
            &constraints_ref,
            &meta_ref,
        ];
        db.exec(SchemaDatabaseSQL::insert_predicate_sql(), params)
            .map_err(|e| {
                AdapterError::InvalidOperation(format!("Failed to insert predicate: {e}"))
            })?;

        // Capture predicate_id BEFORE inserting argument rows (ordering matters).
        let predicate_id = db.last_insert_rowid()?;

        for (position, domain_name) in predicate.arg_domains.iter().enumerate() {
            let pos = position as i64;
            let arg_params: &[&dyn ToSqlValue] = &[&predicate_id, &pos, &domain_name.as_str()];
            db.exec(SchemaDatabaseSQL::insert_predicate_arg_sql(), arg_params)
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!(
                        "Failed to insert predicate argument: {e}"
                    ))
                })?;
        }
    }
    Ok(())
}

/// Store variables for a schema.
pub(super) fn store_variables(
    db: &SQLiteDatabase,
    schema_id: i64,
    table: &SymbolTable,
) -> Result<(), AdapterError> {
    for (var_name, domain_name) in &table.variables {
        let params: &[&dyn ToSqlValue] = &[&schema_id, &var_name.as_str(), &domain_name.as_str()];
        db.exec(SchemaDatabaseSQL::insert_variable_sql(), params)
            .map_err(|e| {
                AdapterError::InvalidOperation(format!("Failed to insert variable: {e}"))
            })?;
    }
    Ok(())
}

/// Load domains for a schema.
pub(super) fn load_domains(
    db: &SQLiteDatabase,
    schema_id: i64,
) -> Result<indexmap::IndexMap<String, DomainInfo>, AdapterError> {
    let rows = db.query(SchemaDatabaseSQL::select_domains_sql(), &[&schema_id])?;

    let mut domains = indexmap::IndexMap::new();
    for row in rows.iter() {
        let name = col_text(row, 0)?;
        let cardinality = col_i64(row, 1)?;
        let description = col_opt_text(row, 2)?;
        let metadata_json = col_opt_text(row, 3)?;

        let mut domain = DomainInfo::new(&name, cardinality as usize);
        if let Some(desc) = description {
            domain = domain.with_description(desc);
        }
        if let Some(meta_str) = metadata_json {
            if let Ok(metadata) = serde_json::from_str(&meta_str) {
                domain.metadata = metadata;
            }
        }
        domains.insert(name, domain);
    }
    Ok(domains)
}

/// Intermediate representation for a predicate row before arg loading.
type PredicateRow = (i64, String, Option<String>, Option<String>, Option<String>);

/// Load predicates for a schema.
pub(super) fn load_predicates(
    db: &SQLiteDatabase,
    schema_id: i64,
) -> Result<indexmap::IndexMap<String, PredicateInfo>, AdapterError> {
    let rows = db.query(SchemaDatabaseSQL::select_predicates_sql(), &[&schema_id])?;

    // Collect the top-level predicate rows first to avoid nested borrows.
    let predicate_rows: Vec<PredicateRow> = rows
        .iter()
        .map(|row| {
            let predicate_id = col_i64(row, 0)?;
            let name = col_text(row, 1)?;
            let _arity = col_i64(row, 2)?;
            let description = col_opt_text(row, 3)?;
            let constraints_json = col_opt_text(row, 4)?;
            let metadata_json = col_opt_text(row, 5)?;
            Ok((
                predicate_id,
                name,
                description,
                constraints_json,
                metadata_json,
            ))
        })
        .collect::<Result<Vec<_>, AdapterError>>()?;

    let mut result = indexmap::IndexMap::new();

    for (predicate_id, name, description, constraints_json, metadata_json) in predicate_rows {
        let arg_rows = db.query(
            SchemaDatabaseSQL::select_predicate_args_sql(),
            &[&predicate_id],
        )?;

        let arg_domains: Vec<String> = arg_rows
            .iter()
            .map(|row| {
                // col 0 = position (unused), col 1 = domain_name
                col_text(row, 1)
            })
            .collect::<Result<Vec<_>, AdapterError>>()?;

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
pub(super) fn load_variables(
    db: &SQLiteDatabase,
    schema_id: i64,
) -> Result<indexmap::IndexMap<String, String>, AdapterError> {
    let rows = db.query(
        "SELECT name, domain_name FROM variables WHERE schema_id = $1",
        &[&schema_id],
    )?;

    let mut variables = indexmap::IndexMap::new();
    for row in rows.iter() {
        let name = col_text(row, 0)?;
        let domain_name = col_text(row, 1)?;
        variables.insert(name, domain_name);
    }
    Ok(variables)
}
