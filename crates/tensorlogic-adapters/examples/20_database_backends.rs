//! Demonstrates database backend usage for persistent schema storage.
//!
//! This example shows how to use different database backends to store and
//! retrieve symbol tables:
//! - MemoryDatabase: In-memory storage (default, no features required)
//! - SQLiteDatabase: Persistent SQLite storage (requires 'sqlite' feature)
//! - PostgreSQLDatabase: PostgreSQL storage (requires 'postgres' feature)
//!
//! Run with:
//! ```bash
//! # Memory backend (default)
//! cargo run --example 20_database_backends
//!
//! # SQLite backend
//! cargo run --example 20_database_backends --features sqlite
//!
//! # PostgreSQL backend (requires running PostgreSQL server)
//! cargo run --example 20_database_backends --features postgres
//! ```

use tensorlogic_adapters::{
    DomainInfo, MemoryDatabase, PredicateInfo, SchemaDatabase, SymbolTable,
};

#[cfg(feature = "sqlite")]
use tensorlogic_adapters::SQLiteDatabase;

#[cfg(feature = "postgres")]
use tensorlogic_adapters::PostgreSQLDatabase;

fn main() {
    println!("=== TensorLogic Database Backends Demo ===\n");

    // Demo 1: Memory Database (always available)
    demo_memory_database();

    // Demo 2: SQLite Database (optional)
    #[cfg(feature = "sqlite")]
    demo_sqlite_database();

    // Demo 3: PostgreSQL Database (optional, async)
    #[cfg(feature = "postgres")]
    demo_postgres_database();

    println!("\nDemo complete!");
}

fn demo_memory_database() {
    println!("--- Memory Database Demo ---");

    let mut db = MemoryDatabase::new();

    // Create a schema
    let mut table = create_sample_schema();

    // Store schema
    let id = db
        .store_schema("university_v1", &table)
        .expect("Failed to store schema");
    println!("Stored schema with ID: {:?}", id);

    // Load schema by ID
    let loaded = db.load_schema(id).expect("Failed to load schema");
    println!(
        "Loaded schema with {} domains and {} predicates",
        loaded.domains.len(),
        loaded.predicates.len()
    );

    // Create version 2
    table
        .add_domain(DomainInfo::new("Degree", 10))
        .expect("Failed to add domain");
    let id_v2 = db
        .store_schema("university_v1", &table)
        .expect("Failed to store v2");
    println!("Stored version 2 with ID: {:?}", id_v2);

    // List all schemas
    let schemas = db.list_schemas().expect("Failed to list schemas");
    println!("Total schemas stored: {}", schemas.len());
    for schema in &schemas {
        println!(
            "  - {} (v{}): {} domains, {} predicates",
            schema.name, schema.version, schema.num_domains, schema.num_predicates
        );
    }

    // Get history
    let history = db
        .get_schema_history("university_v1")
        .expect("Failed to get history");
    println!("Schema history ({} versions):", history.len());
    for version in &history {
        println!(
            "  - Version {}: created at {}",
            version.version, version.timestamp
        );
    }

    // Search schemas
    let results = db.search_schemas("university").expect("Failed to search");
    println!("Search results for 'university': {} matches", results.len());

    println!();
}

#[cfg(feature = "sqlite")]
fn demo_sqlite_database() {
    println!("--- SQLite Database Demo ---");

    // Use a temporary file for this demo
    use std::env::temp_dir;
    let db_path = temp_dir().join("tensorlogic_example.db");
    println!("Database path: {:?}", db_path);

    let mut db =
        SQLiteDatabase::new(db_path.to_str().unwrap()).expect("Failed to create SQLite database");

    // Create and store a schema
    let table = create_sample_schema();
    let id = db
        .store_schema("persistent_schema", &table)
        .expect("Failed to store schema");
    println!("Stored schema with ID: {:?}", id);

    // Load it back
    let loaded = db.load_schema(id).expect("Failed to load schema");
    println!(
        "Loaded schema with {} domains and {} predicates",
        loaded.domains.len(),
        loaded.predicates.len()
    );

    // Demonstrate persistence
    println!("Schema is now persisted to disk at: {:?}", db_path);
    println!("You can connect to it later using the same path");

    // List schemas
    let schemas = db.list_schemas().expect("Failed to list schemas");
    println!("Schemas in database: {}", schemas.len());
    for schema in &schemas {
        println!(
            "  - {} (v{}): ID={:?}",
            schema.name, schema.version, schema.id
        );
    }

    // Clean up (optional)
    std::fs::remove_file(&db_path).ok();
    println!("Cleaned up temporary database file");

    println!();
}

#[cfg(feature = "postgres")]
fn demo_postgres_database() {
    println!("--- PostgreSQL Database Demo ---");
    println!("Note: This demo requires a running PostgreSQL server");
    println!("Connection string: host=localhost user=postgres password=postgres");
    println!();

    // Create runtime for async operations
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create runtime");

    runtime.block_on(async {
        // Try to connect (this may fail if PostgreSQL is not running)
        let connection_string = "host=localhost user=postgres password=postgres dbname=tensorlogic";

        match PostgreSQLDatabase::new(connection_string).await {
            Ok(mut db) => {
                println!("Successfully connected to PostgreSQL");

                // Create and store a schema
                let table = create_sample_schema();
                let id = db
                    .store_schema_async("async_schema", &table)
                    .await
                    .expect("Failed to store schema");
                println!("Stored schema with ID: {:?}", id);

                // Load it back
                let loaded = db
                    .load_schema_async(id)
                    .await
                    .expect("Failed to load schema");
                println!(
                    "Loaded schema with {} domains and {} predicates",
                    loaded.domains.len(),
                    loaded.predicates.len()
                );

                // List schemas
                let schemas = db
                    .list_schemas_async()
                    .await
                    .expect("Failed to list schemas");
                println!("Schemas in database: {}", schemas.len());
                for schema in &schemas {
                    println!(
                        "  - {} (v{}): ID={:?}",
                        schema.name, schema.version, schema.id
                    );
                }

                println!();
            }
            Err(e) => {
                println!("Failed to connect to PostgreSQL: {}", e);
                println!("Make sure PostgreSQL is running and accessible");
                println!("Example setup:");
                println!("  docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres");
                println!("  psql -h localhost -U postgres -c 'CREATE DATABASE tensorlogic;'");
                println!();
            }
        }
    });
}

/// Create a sample schema for demonstration.
fn create_sample_schema() -> SymbolTable {
    let mut table = SymbolTable::new();

    // Add domains
    table
        .add_domain(DomainInfo::new("Person", 100).with_description("People in the university"))
        .expect("Failed to add Person domain");

    table
        .add_domain(DomainInfo::new("Course", 50).with_description("Available courses"))
        .expect("Failed to add Course domain");

    table
        .add_domain(DomainInfo::new("Grade", 5).with_description("Grade values (A-F)"))
        .expect("Failed to add Grade domain");

    // Add predicates
    table
        .add_predicate(
            PredicateInfo::new("enrolled", vec!["Person".to_string(), "Course".to_string()])
                .with_description("Person is enrolled in Course"),
        )
        .expect("Failed to add enrolled predicate");

    table
        .add_predicate(
            PredicateInfo::new("teaches", vec!["Person".to_string(), "Course".to_string()])
                .with_description("Person teaches Course"),
        )
        .expect("Failed to add teaches predicate");

    table
        .add_predicate(
            PredicateInfo::new(
                "grade",
                vec![
                    "Person".to_string(),
                    "Course".to_string(),
                    "Grade".to_string(),
                ],
            )
            .with_description("Person received Grade in Course"),
        )
        .expect("Failed to add grade predicate");

    // Add variables
    table
        .bind_variable("student", "Person")
        .expect("Failed to bind student variable");
    table
        .bind_variable("professor", "Person")
        .expect("Failed to bind professor variable");
    table
        .bind_variable("class", "Course")
        .expect("Failed to bind class variable");

    table
}
