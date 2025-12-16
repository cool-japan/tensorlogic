//! Benchmarks for utility functions.
//!
//! These benchmarks measure the performance of:
//! - Batch operations (bulk domain/predicate additions)
//! - Conversion utilities (format conversions, data extraction)
//! - Query utilities (filtering, grouping, searching)
//! - Validation utilities (comprehensive validation)
//! - Statistics utilities (metrics collection)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use std::hint::black_box as std_black_box;
use tensorlogic_adapters::{
    BatchOperations, ConversionUtils, DomainInfo, PredicateInfo, QueryUtils, SchemaValidator,
    StatisticsUtils, SymbolTable, ValidationUtils,
};

/// Helper function to create a large symbol table
fn create_large_symbol_table(num_domains: usize, num_predicates: usize) -> SymbolTable {
    let mut table = SymbolTable::new();

    for i in 0..num_domains {
        table
            .add_domain(DomainInfo::new(format!("Domain{}", i), 100 + i))
            .unwrap();
    }

    for i in 0..num_predicates {
        let arity = (i % 4) + 1;
        let domains: Vec<String> = (0..arity)
            .map(|j| format!("Domain{}", (i + j) % num_domains))
            .collect();
        table
            .add_predicate(PredicateInfo::new(format!("pred{}", i), domains))
            .unwrap();
    }

    table
}

/// Benchmark batch domain additions
fn bench_batch_add_domains(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_add_domains");

    for size in [10, 50, 100, 500, 1000].iter() {
        let domains: Vec<DomainInfo> = (0..*size)
            .map(|i| DomainInfo::new(format!("Domain{}", i), 100))
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut table = SymbolTable::new();
                BatchOperations::add_domains(
                    std_black_box(&mut table),
                    std_black_box(domains.clone()),
                )
                .unwrap();
                std_black_box(table);
            });
        });
    }

    group.finish();
}

/// Benchmark batch predicate additions
fn bench_batch_add_predicates(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_add_predicates");

    for size in [10, 50, 100, 500, 1000].iter() {
        // Create a table with domains first
        let mut table = SymbolTable::new();
        for i in 0..10 {
            table
                .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                .unwrap();
        }

        let predicates: Vec<PredicateInfo> = (0..*size)
            .map(|i| PredicateInfo::new(format!("pred{}", i), vec![format!("Domain{}", i % 10)]))
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let mut table = table.clone();
                BatchOperations::add_predicates(
                    std_black_box(&mut table),
                    std_black_box(predicates.clone()),
                )
                .unwrap();
                std_black_box(table);
            });
        });
    }

    group.finish();
}

/// Benchmark batch variable bindings
fn bench_batch_bind_variables(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_bind_variables");

    for size in [10, 50, 100, 500].iter() {
        let mut table = SymbolTable::new();
        table.add_domain(DomainInfo::new("Domain", 100)).unwrap();

        let mut bindings = HashMap::new();
        for i in 0..*size {
            bindings.insert(format!("var{}", i), "Domain".to_string());
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let mut table = table.clone();
                BatchOperations::bind_variables(
                    std_black_box(&mut table),
                    std_black_box(bindings.clone()),
                )
                .unwrap();
                std_black_box(table);
            });
        });
    }

    group.finish();
}

/// Benchmark conversion utilities - summary generation
fn bench_conversion_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_summary");

    for size in [50, 100, 500, 1000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let summary = ConversionUtils::to_summary(std_black_box(table));
                std_black_box(summary);
            });
        });
    }

    group.finish();
}

/// Benchmark conversion utilities - name extraction
fn bench_conversion_extract_names(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_extract_names");

    for size in [100, 500, 1000, 5000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64 * 2));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let domains = ConversionUtils::extract_domain_names(std_black_box(table));
                let predicates = ConversionUtils::extract_predicate_names(std_black_box(table));
                std_black_box((domains, predicates));
            });
        });
    }

    group.finish();
}

/// Benchmark query utilities - predicate filtering by arity
fn bench_query_by_arity(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_by_arity");

    for size in [100, 500, 1000, 5000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let result = QueryUtils::find_predicates_by_arity(std_black_box(table), 2);
                std_black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark query utilities - predicates using domain
fn bench_query_predicates_using_domain(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_predicates_using_domain");

    for size in [100, 500, 1000, 5000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let result =
                    QueryUtils::find_predicates_using_domain(std_black_box(table), "Domain0");
                std_black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark query utilities - grouping by arity
fn bench_query_group_by_arity(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_group_by_arity");

    for size in [100, 500, 1000, 5000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let result = QueryUtils::group_predicates_by_arity(std_black_box(table));
                std_black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark query utilities - domain usage counts
fn bench_query_domain_usage_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_domain_usage_counts");

    for size in [100, 500, 1000, 5000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let result = StatisticsUtils::domain_usage_counts(std_black_box(table));
                std_black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark validation utilities - comprehensive validation
fn bench_validation_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_comprehensive");

    for size in [50, 100, 500, 1000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let validator = SchemaValidator::new(std_black_box(table));
                let report = validator.validate().unwrap();
                std_black_box(report);
            });
        });
    }

    group.finish();
}

/// Benchmark validation utilities - is_valid check
fn bench_validation_is_valid(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_is_valid");

    for size in [100, 500, 1000, 5000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let result = ValidationUtils::is_valid(std_black_box(table));
                std_black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark statistics utilities - average arity calculation
fn bench_statistics_average_arity(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics_average_arity");

    for size in [100, 500, 1000, 5000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let avg = StatisticsUtils::average_predicate_arity(std_black_box(table));
                std_black_box(avg);
            });
        });
    }

    group.finish();
}

/// Benchmark statistics utilities - total cardinality calculation
fn bench_statistics_total_cardinality(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics_total_cardinality");

    for size in [100, 500, 1000, 5000].iter() {
        let table = create_large_symbol_table(*size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, table| {
            b.iter(|| {
                let total = StatisticsUtils::total_domain_cardinality(std_black_box(table));
                std_black_box(total);
            });
        });
    }

    group.finish();
}

/// Benchmark combined operations (realistic workload)
fn bench_combined_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_operations");
    group.sample_size(10);

    for size in [100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                // Create table with batch operations
                let mut table = SymbolTable::new();
                let domains: Vec<DomainInfo> = (0..size)
                    .map(|i| DomainInfo::new(format!("Domain{}", i), 100))
                    .collect();
                BatchOperations::add_domains(&mut table, domains).unwrap();

                let predicates: Vec<PredicateInfo> = (0..size)
                    .map(|i| {
                        PredicateInfo::new(
                            format!("pred{}", i),
                            vec![format!("Domain{}", i % size)],
                        )
                    })
                    .collect();
                BatchOperations::add_predicates(&mut table, predicates).unwrap();

                // Query operations
                let _ = QueryUtils::find_predicates_by_arity(&table, 1);
                let _ = QueryUtils::group_predicates_by_arity(&table);
                let _ = StatisticsUtils::domain_usage_counts(&table);

                // Statistics
                let _ = StatisticsUtils::average_predicate_arity(&table);
                let _ = StatisticsUtils::total_domain_cardinality(&table);

                // Validation
                let _ = ValidationUtils::is_valid(&table);

                std_black_box(table);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_add_domains,
    bench_batch_add_predicates,
    bench_batch_bind_variables,
    bench_conversion_summary,
    bench_conversion_extract_names,
    bench_query_by_arity,
    bench_query_predicates_using_domain,
    bench_query_group_by_arity,
    bench_query_domain_usage_counts,
    bench_validation_comprehensive,
    bench_validation_is_valid,
    bench_statistics_average_arity,
    bench_statistics_total_cardinality,
    bench_combined_operations,
);

criterion_main!(benches);
