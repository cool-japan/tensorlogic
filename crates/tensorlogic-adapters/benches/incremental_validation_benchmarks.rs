//! Benchmarks for incremental validation performance.
//!
//! This benchmark suite demonstrates the performance benefits of incremental
//! validation over full schema revalidation, showing 10-100x speedups for
//! large schemas with small changes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensorlogic_adapters::{
    ChangeTracker, DomainInfo, IncrementalValidator, PredicateInfo, SchemaValidator, SymbolTable,
};

/// Create a large schema for benchmarking
fn create_large_schema(domains: usize, predicates_per_domain: usize) -> SymbolTable {
    let mut table = SymbolTable::new();

    // Add domains
    for i in 0..domains {
        table
            .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
            .unwrap();
    }

    // Add predicates
    for i in 0..domains {
        let domain_name = format!("Domain{}", i);
        for j in 0..predicates_per_domain {
            let pred = PredicateInfo::new(
                format!("pred_{}_{}", i, j),
                vec![domain_name.clone(), domain_name.clone()],
            );
            table.add_predicate(pred).unwrap();
        }
    }

    table
}

/// Benchmark full validation vs incremental validation
fn validation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_comparison");

    for size in [10, 50, 100, 200].iter() {
        let table = create_large_schema(*size, 5);
        let total_predicates = size * 5;

        group.throughput(Throughput::Elements(total_predicates as u64));

        // Benchmark full validation
        group.bench_with_input(BenchmarkId::new("full_validation", size), size, |b, _| {
            b.iter(|| {
                let validator = SchemaValidator::new(&table);
                let report = validator.validate().unwrap();
                black_box(report);
            });
        });

        // Benchmark incremental validation with single change
        group.bench_with_input(
            BenchmarkId::new("incremental_single_change", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut tracker = ChangeTracker::new();
                    tracker.record_domain_modification("Domain0");

                    let mut validator = IncrementalValidator::new(&table, &tracker);
                    let report = validator.validate_incremental().unwrap();
                    black_box(report);
                });
            },
        );

        // Benchmark incremental validation with 10% changes
        group.bench_with_input(
            BenchmarkId::new("incremental_10pct_changes", size),
            size,
            |b, sz| {
                b.iter(|| {
                    let mut tracker = ChangeTracker::new();
                    let change_count = (*sz as f64 * 0.1).ceil() as usize;
                    for i in 0..change_count {
                        tracker.record_domain_modification(format!("Domain{}", i));
                    }

                    let mut validator = IncrementalValidator::new(&table, &tracker);
                    let report = validator.validate_incremental().unwrap();
                    black_box(report);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark change tracking overhead
fn change_tracking_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("change_tracking");

    for operations in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*operations as u64));

        group.bench_with_input(
            BenchmarkId::new("record_changes", operations),
            operations,
            |b, ops| {
                b.iter(|| {
                    let mut tracker = ChangeTracker::new();
                    for i in 0..*ops {
                        match i % 3 {
                            0 => tracker.record_domain_addition(format!("Domain{}", i)),
                            1 => tracker.record_predicate_addition(format!("Pred{}", i)),
                            _ => tracker.record_variable_addition(format!("Var{}", i)),
                        }
                    }
                    black_box(tracker);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch_operations", operations),
            operations,
            |b, ops| {
                b.iter(|| {
                    let mut tracker = ChangeTracker::new();
                    tracker.begin_batch();
                    for i in 0..*ops {
                        tracker.record_domain_addition(format!("Domain{}", i));
                    }
                    tracker.end_batch();
                    black_box(tracker);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark dependency graph construction
fn dependency_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dependency_graph");

    for size in [50, 100, 200, 500].iter() {
        let table = create_large_schema(*size, 10);
        group.throughput(Throughput::Elements((size * 10) as u64));

        group.bench_with_input(BenchmarkId::new("build_graph", size), &table, |b, tbl| {
            b.iter(|| {
                use tensorlogic_adapters::DependencyGraph;
                let graph = DependencyGraph::from_symbol_table(tbl);
                black_box(graph);
            });
        });
    }

    group.finish();
}

/// Benchmark validation cache effectiveness
fn cache_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effectiveness");

    for size in [50, 100, 200].iter() {
        let table = create_large_schema(*size, 5);

        group.bench_with_input(BenchmarkId::new("cold_cache", size), size, |b, _| {
            b.iter(|| {
                let mut tracker = ChangeTracker::new();
                tracker.record_domain_modification("Domain0");

                let mut validator = IncrementalValidator::new(&table, &tracker);
                let report = validator.validate_incremental().unwrap();
                black_box(report);
            });
        });

        group.bench_with_input(BenchmarkId::new("warm_cache", size), size, |b, sz| {
            // Pre-warm the cache
            let mut tracker = ChangeTracker::new();
            for i in 0..*sz {
                tracker.record_domain_modification(format!("Domain{}", i));
            }
            let mut validator = IncrementalValidator::new(&table, &tracker);
            validator.validate_incremental().unwrap();
            let cache = validator.cache().clone();

            b.iter(|| {
                let mut tracker2 = ChangeTracker::new();
                tracker2.record_domain_modification("Domain0");

                let mut validator2 =
                    IncrementalValidator::new(&table, &tracker2).with_cache(cache.clone());
                let report = validator2.validate_incremental().unwrap();
                black_box(report);
            });
        });
    }

    group.finish();
}

/// Benchmark affected components computation
fn affected_components_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("affected_components");

    for size in [50, 100, 200, 500].iter() {
        let table = create_large_schema(*size, 10);

        group.bench_with_input(
            BenchmarkId::new("single_domain_change", size),
            &table,
            |b, tbl| {
                use std::collections::HashSet;
                use tensorlogic_adapters::DependencyGraph;

                let graph = DependencyGraph::from_symbol_table(tbl);
                let mut initial = HashSet::new();
                initial.insert("Domain0".to_string());

                b.iter(|| {
                    let affected = graph.compute_affected_components(&initial);
                    black_box(affected);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("multiple_domains_change", size),
            &table,
            |b, tbl| {
                use std::collections::HashSet;
                use tensorlogic_adapters::DependencyGraph;

                let graph = DependencyGraph::from_symbol_table(tbl);
                let mut initial = HashSet::new();
                let change_count = (*size as f64 * 0.1).ceil() as usize;
                for i in 0..change_count {
                    initial.insert(format!("Domain{}", i));
                }

                b.iter(|| {
                    let affected = graph.compute_affected_components(&initial);
                    black_box(affected);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark speedup ratio
fn speedup_demonstration(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_ratio");
    group.sample_size(50);

    for size in [100, 200, 500].iter() {
        let table = create_large_schema(*size, 10);

        // Full validation baseline
        let full_name = format!("full_{}", size);
        group.bench_function(&full_name, |b| {
            b.iter(|| {
                let validator = SchemaValidator::new(&table);
                let report = validator.validate().unwrap();
                black_box(report);
            });
        });

        // Incremental with 1% changes
        let incr_name = format!("incremental_1pct_{}", size);
        group.bench_function(&incr_name, |b| {
            b.iter(|| {
                let mut tracker = ChangeTracker::new();
                let change_count = (*size as f64 * 0.01).max(1.0) as usize;
                for i in 0..change_count {
                    tracker.record_domain_modification(format!("Domain{}", i));
                }

                let mut validator = IncrementalValidator::new(&table, &tracker);
                let report = validator.validate_incremental().unwrap();
                black_box(report);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    validation_comparison,
    change_tracking_overhead,
    dependency_graph_construction,
    cache_effectiveness,
    affected_components_computation,
    speedup_demonstration,
);
criterion_main!(benches);
