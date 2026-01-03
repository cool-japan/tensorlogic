//! Benchmarks for query planner performance and optimization.
//!
//! This benchmark suite demonstrates the performance benefits of cost-based
//! query planning and plan caching for predicate lookups.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensorlogic_adapters::{
    DomainInfo, PredicateInfo, PredicatePattern, PredicateQuery, QueryPlanner, SymbolTable,
};

/// Create a schema with diverse predicates for benchmarking
fn create_diverse_schema(size: usize) -> SymbolTable {
    let mut table = SymbolTable::new();

    // Add domains
    for i in 0..10 {
        table
            .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
            .unwrap();
    }

    // Add predicates with varying arities and signatures
    for i in 0..size {
        let arity = (i % 4) + 1; // Arity 1-4
        let mut domains = Vec::new();
        for j in 0..arity {
            domains.push(format!("Domain{}", (i + j) % 10));
        }

        let pred = PredicateInfo::new(format!("pred_{}", i), domains);
        table.add_predicate(pred).unwrap();
    }

    table
}

/// Benchmark different query types
fn query_types_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_types");

    for size in [100, 500, 1000, 2000].iter() {
        let table = create_diverse_schema(*size);
        group.throughput(Throughput::Elements(*size as u64));

        // Query by name (O(1))
        group.bench_with_input(BenchmarkId::new("by_name", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let query = PredicateQuery::by_name("pred_0");

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });

        // Query by arity (O(n))
        group.bench_with_input(BenchmarkId::new("by_arity", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let query = PredicateQuery::by_arity(2);

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });

        // Query by signature (O(n))
        group.bench_with_input(BenchmarkId::new("by_signature", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let query =
                PredicateQuery::by_signature(vec!["Domain0".to_string(), "Domain1".to_string()]);

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });

        // Query by domain (O(n))
        group.bench_with_input(BenchmarkId::new("by_domain", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let query = PredicateQuery::by_domain("Domain0");

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });

        // Query by pattern (O(n))
        group.bench_with_input(BenchmarkId::new("by_pattern", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let pattern = PredicatePattern::new()
                .with_name_pattern("pred_*")
                .with_arity_range(2, 3);
            let query = PredicateQuery::by_pattern(pattern);

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark complex queries (AND/OR combinations)
fn complex_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_queries");

    for size in [500, 1000, 2000].iter() {
        let table = create_diverse_schema(*size);

        // AND query
        group.bench_with_input(BenchmarkId::new("and_query", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let query = PredicateQuery::and(vec![
                PredicateQuery::by_arity(2),
                PredicateQuery::by_domain("Domain0"),
            ]);

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });

        // OR query
        group.bench_with_input(BenchmarkId::new("or_query", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let query = PredicateQuery::or(vec![
                PredicateQuery::by_name("pred_0"),
                PredicateQuery::by_name("pred_1"),
            ]);

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });

        // Nested query
        group.bench_with_input(BenchmarkId::new("nested_query", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let query = PredicateQuery::and(vec![
                PredicateQuery::by_arity(2),
                PredicateQuery::or(vec![
                    PredicateQuery::by_domain("Domain0"),
                    PredicateQuery::by_domain("Domain1"),
                ]),
            ]);

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark plan caching effectiveness
fn plan_caching(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_caching");

    for size in [500, 1000, 2000].iter() {
        let table = create_diverse_schema(*size);

        // First execution (cold cache)
        group.bench_with_input(BenchmarkId::new("cold_cache", size), &table, |b, tbl| {
            b.iter(|| {
                let mut planner = QueryPlanner::new(tbl);
                let query = PredicateQuery::by_arity(2);
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });

        // Repeated execution (warm cache)
        group.bench_with_input(BenchmarkId::new("warm_cache", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let query = PredicateQuery::by_arity(2);

            // Pre-warm cache
            planner.execute(&query).unwrap();

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                black_box(results);
            });
        });

        // Multiple different queries (cache utilization)
        group.bench_with_input(BenchmarkId::new("multi_query", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let queries = vec![
                PredicateQuery::by_arity(2),
                PredicateQuery::by_arity(3),
                PredicateQuery::by_domain("Domain0"),
                PredicateQuery::by_signature(vec!["Domain0".to_string(), "Domain1".to_string()]),
            ];

            // Pre-warm cache
            for q in &queries {
                planner.execute(q).unwrap();
            }

            b.iter(|| {
                for q in &queries {
                    let results = planner.execute(q).unwrap();
                    black_box(results);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark pattern matching performance
fn pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");

    for size in [500, 1000, 2000].iter() {
        let table = create_diverse_schema(*size);
        group.throughput(Throughput::Elements(*size as u64));

        // Simple wildcard
        group.bench_with_input(
            BenchmarkId::new("simple_wildcard", size),
            &table,
            |b, tbl| {
                let mut planner = QueryPlanner::new(tbl);
                let pattern = PredicatePattern::new().with_name_pattern("pred_*");
                let query = PredicateQuery::by_pattern(pattern);

                b.iter(|| {
                    let results = planner.execute(&query).unwrap();
                    black_box(results);
                });
            },
        );

        // Complex pattern
        group.bench_with_input(
            BenchmarkId::new("complex_pattern", size),
            &table,
            |b, tbl| {
                let mut planner = QueryPlanner::new(tbl);
                let pattern = PredicatePattern::new()
                    .with_name_pattern("pred_1*")
                    .with_arity_range(2, 3)
                    .with_required_domain("Domain0");
                let query = PredicateQuery::by_pattern(pattern);

                b.iter(|| {
                    let results = planner.execute(&query).unwrap();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark query planning overhead
fn planning_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("planning_overhead");

    for size in [100, 500, 1000].iter() {
        let table = create_diverse_schema(*size);

        group.bench_with_input(
            BenchmarkId::new("plan_generation", size),
            &table,
            |b, tbl| {
                let mut planner = QueryPlanner::new(tbl);
                let query = PredicateQuery::by_arity(2);

                b.iter(|| {
                    planner.clear_cache();
                    let plan = planner.plan(&query).unwrap();
                    black_box(plan);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("direct_execution", size),
            &table,
            |b, tbl| {
                b.iter(|| {
                    let results: Vec<_> = tbl
                        .predicates
                        .iter()
                        .filter(|(_, pred)| pred.arg_domains.len() == 2)
                        .map(|(name, pred)| (name.clone(), pred.clone()))
                        .collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark statistics tracking
fn statistics_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics");

    for size in [100, 500, 1000].iter() {
        let table = create_diverse_schema(*size);

        group.bench_with_input(BenchmarkId::new("with_stats", size), &table, |b, tbl| {
            let mut planner = QueryPlanner::new(tbl);
            let query = PredicateQuery::by_arity(2);

            b.iter(|| {
                let results = planner.execute(&query).unwrap();
                let _stats = planner.statistics();
                black_box(results);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    query_types_comparison,
    complex_queries,
    plan_caching,
    pattern_matching,
    planning_overhead,
    statistics_tracking,
);
criterion_main!(benches);
