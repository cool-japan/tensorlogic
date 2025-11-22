//! Benchmarks for database backend performance.
//!
//! This benchmark suite measures the performance of different database backends
//! for schema storage and retrieval operations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensorlogic_adapters::{
    DomainInfo, MemoryDatabase, PredicateInfo, SchemaDatabase, SymbolTable,
};

#[cfg(feature = "sqlite")]
use tensorlogic_adapters::SQLiteDatabase;

/// Create a small test schema (5 domains, 10 predicates, 5 variables).
fn create_small_schema() -> SymbolTable {
    let mut table = SymbolTable::new();

    // Add 5 domains
    for i in 0..5 {
        let name = format!("Domain{}", i);
        table
            .add_domain(DomainInfo::new(&name, 100 + i * 10))
            .unwrap();
    }

    // Add 10 predicates
    for i in 0..10 {
        let pred_name = format!("pred{}", i);
        table
            .add_predicate(PredicateInfo::new(
                &pred_name,
                vec![format!("Domain{}", i % 5), format!("Domain{}", (i + 1) % 5)],
            ))
            .unwrap();
    }

    // Add 5 variables
    for i in 0..5 {
        let var_name = format!("var{}", i);
        let domain_name = format!("Domain{}", i);
        table.bind_variable(&var_name, &domain_name).unwrap();
    }

    table
}

/// Create a medium test schema (20 domains, 50 predicates, 20 variables).
fn create_medium_schema() -> SymbolTable {
    let mut table = SymbolTable::new();

    // Add 20 domains
    for i in 0..20 {
        let name = format!("Domain{}", i);
        table
            .add_domain(
                DomainInfo::new(&name, 100 + i * 10)
                    .with_description(format!("Description for domain {}", i)),
            )
            .unwrap();
    }

    // Add 50 predicates
    for i in 0..50 {
        let pred_name = format!("predicate{}", i);
        table
            .add_predicate(
                PredicateInfo::new(
                    &pred_name,
                    vec![
                        format!("Domain{}", i % 20),
                        format!("Domain{}", (i + 1) % 20),
                        format!("Domain{}", (i + 2) % 20),
                    ],
                )
                .with_description(format!("Predicate {}", i)),
            )
            .unwrap();
    }

    // Add 20 variables
    for i in 0..20 {
        let var_name = format!("variable{}", i);
        let domain_name = format!("Domain{}", i);
        table.bind_variable(&var_name, &domain_name).unwrap();
    }

    table
}

/// Create a large test schema (100 domains, 200 predicates, 50 variables).
fn create_large_schema() -> SymbolTable {
    let mut table = SymbolTable::new();

    // Add 100 domains
    for i in 0..100 {
        let name = format!("Domain{}", i);
        table
            .add_domain(
                DomainInfo::new(&name, 100 + i * 10)
                    .with_description(format!("Description for domain {}", i)),
            )
            .unwrap();
    }

    // Add 200 predicates
    for i in 0..200 {
        let arity = (i % 4) + 1; // Vary arity from 1 to 4
        let mut args = Vec::new();
        for j in 0..arity {
            args.push(format!("Domain{}", (i + j) % 100));
        }
        let pred_name = format!("predicate{}", i);
        table
            .add_predicate(
                PredicateInfo::new(&pred_name, args).with_description(format!("Predicate {}", i)),
            )
            .unwrap();
    }

    // Add 50 variables
    for i in 0..50 {
        let var_name = format!("variable{}", i);
        let domain_name = format!("Domain{}", i * 2);
        table.bind_variable(&var_name, &domain_name).unwrap();
    }

    table
}

// ============================================================================
// Memory Database Benchmarks
// ============================================================================

fn bench_memory_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_store");

    for &size in &["small", "medium", "large"] {
        let schema = match size {
            "small" => create_small_schema(),
            "medium" => create_medium_schema(),
            "large" => create_large_schema(),
            _ => unreachable!(),
        };

        group.bench_with_input(BenchmarkId::from_parameter(size), &schema, |b, schema| {
            b.iter(|| {
                let mut db = MemoryDatabase::new();
                black_box(db.store_schema("test", schema).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_memory_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_load");

    for &size in &["small", "medium", "large"] {
        let schema = match size {
            "small" => create_small_schema(),
            "medium" => create_medium_schema(),
            "large" => create_large_schema(),
            _ => unreachable!(),
        };

        let mut db = MemoryDatabase::new();
        let id = db.store_schema("test", &schema).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &id, |b, id| {
            b.iter(|| black_box(db.load_schema(*id).unwrap()));
        });
    }

    group.finish();
}

fn bench_memory_list_schemas(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_list_schemas");

    for &count in &[10, 50, 100] {
        let mut db = MemoryDatabase::new();
        let schema = create_small_schema();

        for i in 0..count {
            let name = format!("schema{}", i);
            db.store_schema(&name, &schema).unwrap();
        }

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &db, |b, db| {
            b.iter(|| black_box(db.list_schemas().unwrap()));
        });
    }

    group.finish();
}

fn bench_memory_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_search");

    let mut db = MemoryDatabase::new();
    let schema = create_small_schema();

    // Create 100 schemas with different names
    for i in 0..100 {
        let name = format!("test_schema_{}", i);
        db.store_schema(&name, &schema).unwrap();
    }

    group.bench_function("pattern_match", |b| {
        b.iter(|| black_box(db.search_schemas("test").unwrap()));
    });

    group.finish();
}

// ============================================================================
// SQLite Database Benchmarks
// ============================================================================

#[cfg(feature = "sqlite")]
fn bench_sqlite_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_store");

    for &size in &["small", "medium", "large"] {
        let schema = match size {
            "small" => create_small_schema(),
            "medium" => create_medium_schema(),
            "large" => create_large_schema(),
            _ => unreachable!(),
        };

        group.bench_with_input(BenchmarkId::from_parameter(size), &schema, |b, schema| {
            b.iter(|| {
                let mut db = SQLiteDatabase::new(":memory:").unwrap();
                black_box(db.store_schema("test", schema).unwrap())
            });
        });
    }

    group.finish();
}

#[cfg(feature = "sqlite")]
fn bench_sqlite_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_load");

    for &size in &["small", "medium", "large"] {
        let schema = match size {
            "small" => create_small_schema(),
            "medium" => create_medium_schema(),
            "large" => create_large_schema(),
            _ => unreachable!(),
        };

        let mut db = SQLiteDatabase::new(":memory:").unwrap();
        let id = db.store_schema("test", &schema).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &id, |b, id| {
            b.iter(|| black_box(db.load_schema(*id).unwrap()));
        });
    }

    group.finish();
}

#[cfg(feature = "sqlite")]
fn bench_sqlite_list_schemas(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_list_schemas");

    for &count in &[10, 50, 100] {
        let mut db = SQLiteDatabase::new(":memory:").unwrap();
        let schema = create_small_schema();

        for i in 0..count {
            let name = format!("schema{}", i);
            db.store_schema(&name, &schema).unwrap();
        }

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &db, |b, db| {
            b.iter(|| black_box(db.list_schemas().unwrap()));
        });
    }

    group.finish();
}

#[cfg(feature = "sqlite")]
fn bench_sqlite_vs_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_vs_memory");

    let schema = create_medium_schema();

    group.bench_function("memory_store", |b| {
        b.iter(|| {
            let mut db = MemoryDatabase::new();
            black_box(db.store_schema("test", &schema).unwrap())
        });
    });

    group.bench_function("sqlite_store", |b| {
        b.iter(|| {
            let mut db = SQLiteDatabase::new(":memory:").unwrap();
            black_box(db.store_schema("test", &schema).unwrap())
        });
    });

    group.finish();
}

#[cfg(feature = "sqlite")]
fn bench_sqlite_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqlite_persistence");

    use std::env::temp_dir;
    let db_path = temp_dir().join("bench_persistence.db");

    let schema = create_medium_schema();

    group.bench_function("file_store", |b| {
        b.iter(|| {
            let mut db = SQLiteDatabase::new(db_path.to_str().unwrap()).unwrap();
            black_box(db.store_schema("test", &schema).unwrap())
        });
    });

    // Clean up
    std::fs::remove_file(&db_path).ok();

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    memory_benches,
    bench_memory_store,
    bench_memory_load,
    bench_memory_list_schemas,
    bench_memory_search
);

#[cfg(feature = "sqlite")]
criterion_group!(
    sqlite_benches,
    bench_sqlite_store,
    bench_sqlite_load,
    bench_sqlite_list_schemas,
    bench_sqlite_vs_memory,
    bench_sqlite_persistence
);

#[cfg(feature = "sqlite")]
criterion_main!(memory_benches, sqlite_benches);

#[cfg(not(feature = "sqlite"))]
criterion_main!(memory_benches);
