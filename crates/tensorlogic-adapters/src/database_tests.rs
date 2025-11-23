use super::*;
use crate::{DomainInfo, PredicateInfo};

#[test]
fn test_memory_database_store_load() {
    let mut db = MemoryDatabase::new();
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();

    let id = db.store_schema("test_schema", &table).unwrap();

    let loaded = db.load_schema(id).unwrap();
    assert_eq!(loaded.domains.len(), 1);
    assert!(loaded.domains.contains_key("Person"));
}

#[test]
fn test_memory_database_load_by_name() {
    let mut db = MemoryDatabase::new();
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();

    db.store_schema("test_schema", &table).unwrap();

    let loaded = db.load_schema_by_name("test_schema").unwrap();
    assert_eq!(loaded.domains.len(), 1);
}

#[test]
fn test_memory_database_versioning() {
    let mut db = MemoryDatabase::new();

    let mut table_v1 = SymbolTable::new();
    table_v1.add_domain(DomainInfo::new("Person", 100)).unwrap();

    let id1 = db.store_schema("test_schema", &table_v1).unwrap();

    let mut table_v2 = SymbolTable::new();
    table_v2.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table_v2.add_domain(DomainInfo::new("Course", 50)).unwrap();

    let id2 = db.store_schema("test_schema", &table_v2).unwrap();

    // Should have different IDs
    assert_ne!(id1, id2);

    // Load by name should return latest version
    let latest = db.load_schema_by_name("test_schema").unwrap();
    assert_eq!(latest.domains.len(), 2);

    // Should be able to load old version by ID
    let old = db.load_schema(id1).unwrap();
    assert_eq!(old.domains.len(), 1);
}

#[test]
fn test_memory_database_list_schemas() {
    let mut db = MemoryDatabase::new();

    let mut table1 = SymbolTable::new();
    table1.add_domain(DomainInfo::new("Person", 100)).unwrap();
    db.store_schema("schema1", &table1).unwrap();

    let mut table2 = SymbolTable::new();
    table2.add_domain(DomainInfo::new("Course", 50)).unwrap();
    db.store_schema("schema2", &table2).unwrap();

    let schemas = db.list_schemas().unwrap();
    assert_eq!(schemas.len(), 2);

    // Should be sorted by name
    assert_eq!(schemas[0].name, "schema1");
    assert_eq!(schemas[1].name, "schema2");
}

#[test]
fn test_memory_database_delete() {
    let mut db = MemoryDatabase::new();
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();

    let id = db.store_schema("test_schema", &table).unwrap();

    db.delete_schema(id).unwrap();

    // Should not be able to load deleted schema
    assert!(db.load_schema(id).is_err());
}

#[test]
fn test_memory_database_search() {
    let mut db = MemoryDatabase::new();

    let mut table1 = SymbolTable::new();
    table1.add_domain(DomainInfo::new("Person", 100)).unwrap();
    db.store_schema("user_schema", &table1).unwrap();

    let mut table2 = SymbolTable::new();
    table2.add_domain(DomainInfo::new("Course", 50)).unwrap();
    db.store_schema("course_schema", &table2).unwrap();

    let mut table3 = SymbolTable::new();
    table3.add_domain(DomainInfo::new("Book", 200)).unwrap();
    db.store_schema("library_schema", &table3).unwrap();

    let results = db.search_schemas("schema").unwrap();
    assert_eq!(results.len(), 3); // All contain "schema"

    let results = db.search_schemas("user").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "user_schema");

    let results = db.search_schemas("lib").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "library_schema");
}

#[test]
fn test_memory_database_history() {
    let mut db = MemoryDatabase::new();

    let mut table_v1 = SymbolTable::new();
    table_v1.add_domain(DomainInfo::new("Person", 100)).unwrap();
    db.store_schema("test_schema", &table_v1).unwrap();

    let mut table_v2 = SymbolTable::new();
    table_v2.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table_v2.add_domain(DomainInfo::new("Course", 50)).unwrap();
    db.store_schema("test_schema", &table_v2).unwrap();

    let mut table_v3 = SymbolTable::new();
    table_v3.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table_v3.add_domain(DomainInfo::new("Course", 50)).unwrap();
    table_v3.add_domain(DomainInfo::new("Grade", 5)).unwrap();
    db.store_schema("test_schema", &table_v3).unwrap();

    let history = db.get_schema_history("test_schema").unwrap();
    assert_eq!(history.len(), 3);

    // Should be sorted by version
    assert_eq!(history[0].version, 1);
    assert_eq!(history[1].version, 2);
    assert_eq!(history[2].version, 3);
}

#[test]
fn test_sql_query_generation() {
    let create_tables = SchemaDatabaseSQL::create_tables_sql();
    assert!(!create_tables.is_empty());
    assert!(create_tables[0].contains("CREATE TABLE"));

    assert!(SchemaDatabaseSQL::insert_domain_sql().contains("INSERT INTO domains"));
    assert!(SchemaDatabaseSQL::insert_predicate_sql().contains("INSERT INTO predicates"));
    assert!(SchemaDatabaseSQL::select_schema_sql().contains("SELECT"));
}

#[test]
fn test_schema_metadata_creation() {
    let metadata = SchemaMetadata {
        id: SchemaId(1),
        name: "test".to_string(),
        version: 1,
        created_at: 0,
        updated_at: 0,
        num_domains: 5,
        num_predicates: 10,
        num_variables: 3,
        description: Some("Test schema".to_string()),
    };

    assert_eq!(metadata.name, "test");
    assert_eq!(metadata.num_domains, 5);
}

#[test]
fn test_empty_database() {
    let db = MemoryDatabase::new();
    let schemas = db.list_schemas().unwrap();
    assert!(schemas.is_empty());
}

#[test]
fn test_load_nonexistent_schema() {
    let db = MemoryDatabase::new();
    assert!(db.load_schema(SchemaId(999)).is_err());
    assert!(db.load_schema_by_name("nonexistent").is_err());
}

#[test]
fn test_delete_nonexistent_schema() {
    let mut db = MemoryDatabase::new();
    assert!(db.delete_schema(SchemaId(999)).is_err());
}

#[test]
fn test_search_empty_pattern() {
    let mut db = MemoryDatabase::new();
    let mut table = SymbolTable::new();
    table.add_domain(DomainInfo::new("Person", 100)).unwrap();
    db.store_schema("test", &table).unwrap();

    let results = db.search_schemas("").unwrap();
    assert_eq!(results.len(), 1); // Empty pattern matches all
}

#[test]
fn test_database_stats_empty() {
    let stats = DatabaseStats::new();
    assert_eq!(stats.total_schemas, 0);
    assert_eq!(stats.total_domains, 0);
    assert_eq!(stats.total_predicates, 0);
    assert_eq!(stats.avg_domains_per_schema(), 0.0);
    assert_eq!(stats.avg_predicates_per_schema(), 0.0);
}

#[test]
fn test_database_stats_from_database() {
    let mut db = MemoryDatabase::new();

    // Add first schema
    let mut table1 = SymbolTable::new();
    table1.add_domain(DomainInfo::new("Person", 100)).unwrap();
    table1.add_domain(DomainInfo::new("Course", 50)).unwrap();
    table1
        .add_predicate(PredicateInfo::new(
            "enrolled",
            vec!["Person".to_string(), "Course".to_string()],
        ))
        .unwrap();
    db.store_schema("schema1", &table1).unwrap();

    // Add second schema
    let mut table2 = SymbolTable::new();
    table2.add_domain(DomainInfo::new("Person", 100)).unwrap(); // Add Person domain
    table2.add_domain(DomainInfo::new("Book", 200)).unwrap();
    table2
        .add_predicate(PredicateInfo::new("borrowed", vec!["Person".to_string()]))
        .unwrap();
    table2
        .add_predicate(PredicateInfo::new("returned", vec!["Person".to_string()]))
        .unwrap();
    db.store_schema("schema2", &table2).unwrap();

    let stats = DatabaseStats::from_database(&db).unwrap();
    assert_eq!(stats.total_schemas, 2);
    assert_eq!(stats.total_domains, 4); // 2 + 2
    assert_eq!(stats.total_predicates, 3); // 1 + 2
    assert_eq!(stats.avg_domains_per_schema(), 2.0);
    assert_eq!(stats.avg_predicates_per_schema(), 1.5);
}

#[test]
fn test_database_stats_default() {
    let stats = DatabaseStats::default();
    assert_eq!(stats.total_schemas, 0);
}

// ========================================================================
// SQLite Backend Tests
// ========================================================================

#[cfg(feature = "sqlite")]
mod sqlite_tests {
    use super::*;
    use crate::{PredicateInfo, SQLiteDatabase};

    #[test]
    fn test_sqlite_store_load() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let id = db.store_schema("test_schema", &table).unwrap();

        let loaded = db.load_schema(id).unwrap();
        assert_eq!(loaded.domains.len(), 1);
        assert!(loaded.domains.contains_key("Person"));
    }

    #[test]
    fn test_sqlite_load_by_name() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        db.store_schema("test_schema", &table).unwrap();

        let loaded = db.load_schema_by_name("test_schema").unwrap();
        assert_eq!(loaded.domains.len(), 1);
    }

    #[test]
    fn test_sqlite_versioning() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();

        let mut table_v1 = SymbolTable::new();
        table_v1.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let id1 = db.store_schema("test_schema", &table_v1).unwrap();

        let mut table_v2 = SymbolTable::new();
        table_v2.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table_v2.add_domain(DomainInfo::new("Course", 50)).unwrap();

        let id2 = db.store_schema("test_schema", &table_v2).unwrap();

        // Should have different IDs
        assert_ne!(id1, id2);

        // Load by name should return latest version
        let latest = db.load_schema_by_name("test_schema").unwrap();
        assert_eq!(latest.domains.len(), 2);

        // Should be able to load old version by ID
        let old = db.load_schema(id1).unwrap();
        assert_eq!(old.domains.len(), 1);
    }

    #[test]
    fn test_sqlite_list_schemas() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();

        let mut table1 = SymbolTable::new();
        table1.add_domain(DomainInfo::new("Person", 100)).unwrap();
        db.store_schema("schema1", &table1).unwrap();

        let mut table2 = SymbolTable::new();
        table2.add_domain(DomainInfo::new("Course", 50)).unwrap();
        db.store_schema("schema2", &table2).unwrap();

        let schemas = db.list_schemas().unwrap();
        assert_eq!(schemas.len(), 2);

        // Should be sorted by name
        assert_eq!(schemas[0].name, "schema1");
        assert_eq!(schemas[1].name, "schema2");
    }

    #[test]
    fn test_sqlite_delete() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();

        let id = db.store_schema("test_schema", &table).unwrap();

        db.delete_schema(id).unwrap();

        // Should not be able to load deleted schema
        assert!(db.load_schema(id).is_err());
    }

    #[test]
    fn test_sqlite_search() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();

        let mut table1 = SymbolTable::new();
        table1.add_domain(DomainInfo::new("Person", 100)).unwrap();
        db.store_schema("user_schema", &table1).unwrap();

        let mut table2 = SymbolTable::new();
        table2.add_domain(DomainInfo::new("Course", 50)).unwrap();
        db.store_schema("course_schema", &table2).unwrap();

        let mut table3 = SymbolTable::new();
        table3.add_domain(DomainInfo::new("Book", 200)).unwrap();
        db.store_schema("library_schema", &table3).unwrap();

        let results = db.search_schemas("schema").unwrap();
        assert_eq!(results.len(), 3); // All contain "schema"

        let results = db.search_schemas("user").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "user_schema");

        let results = db.search_schemas("lib").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "library_schema");
    }

    #[test]
    fn test_sqlite_history() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();

        let mut table_v1 = SymbolTable::new();
        table_v1.add_domain(DomainInfo::new("Person", 100)).unwrap();
        db.store_schema("test_schema", &table_v1).unwrap();

        let mut table_v2 = SymbolTable::new();
        table_v2.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table_v2.add_domain(DomainInfo::new("Course", 50)).unwrap();
        db.store_schema("test_schema", &table_v2).unwrap();

        let mut table_v3 = SymbolTable::new();
        table_v3.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table_v3.add_domain(DomainInfo::new("Course", 50)).unwrap();
        table_v3.add_domain(DomainInfo::new("Grade", 5)).unwrap();
        db.store_schema("test_schema", &table_v3).unwrap();

        let history = db.get_schema_history("test_schema").unwrap();
        assert_eq!(history.len(), 3);

        // Should be sorted by version
        assert_eq!(history[0].version, 1);
        assert_eq!(history[1].version, 2);
        assert_eq!(history[2].version, 3);
    }

    #[test]
    fn test_sqlite_with_predicates() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .unwrap();

        let id = db.store_schema("test_schema", &table).unwrap();

        let loaded = db.load_schema(id).unwrap();
        assert_eq!(loaded.domains.len(), 1);
        assert_eq!(loaded.predicates.len(), 1);
        assert!(loaded.predicates.contains_key("knows"));
    }

    #[test]
    fn test_sqlite_with_variables() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.bind_variable("x", "Person").unwrap();

        let id = db.store_schema("test_schema", &table).unwrap();

        let loaded = db.load_schema(id).unwrap();
        assert_eq!(loaded.domains.len(), 1);
        assert_eq!(loaded.variables.len(), 1);
        assert_eq!(loaded.variables.get("x").unwrap(), "Person");
    }

    #[test]
    fn test_sqlite_empty_database() {
        let db = SQLiteDatabase::new(":memory:").unwrap();
        let schemas = db.list_schemas().unwrap();
        assert!(schemas.is_empty());
    }

    #[test]
    fn test_sqlite_load_nonexistent() {
        let db = SQLiteDatabase::new(":memory:").unwrap();
        assert!(db.load_schema(SchemaId(999)).is_err());
        assert!(db.load_schema_by_name("nonexistent").is_err());
    }

    #[test]
    fn test_sqlite_delete_nonexistent() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();
        assert!(db.delete_schema(SchemaId(999)).is_err());
    }

    #[test]
    fn test_sqlite_complex_schema() {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();
        let mut table = SymbolTable::new();

        // Add domains
        table.add_domain(DomainInfo::new("Person", 100)).unwrap();
        table.add_domain(DomainInfo::new("Course", 50)).unwrap();
        table.add_domain(DomainInfo::new("Grade", 5)).unwrap();

        // Add predicates
        table
            .add_predicate(PredicateInfo::new(
                "enrolled",
                vec!["Person".to_string(), "Course".to_string()],
            ))
            .unwrap();
        table
            .add_predicate(PredicateInfo::new(
                "grade",
                vec![
                    "Person".to_string(),
                    "Course".to_string(),
                    "Grade".to_string(),
                ],
            ))
            .unwrap();

        // Add variables
        table.bind_variable("student", "Person").unwrap();
        table.bind_variable("class", "Course").unwrap();

        let id = db.store_schema("university", &table).unwrap();

        let loaded = db.load_schema(id).unwrap();
        assert_eq!(loaded.domains.len(), 3);
        assert_eq!(loaded.predicates.len(), 2);
        assert_eq!(loaded.variables.len(), 2);
    }
}
