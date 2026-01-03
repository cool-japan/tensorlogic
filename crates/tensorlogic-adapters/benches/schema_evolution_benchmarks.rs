//! Benchmarks for schema evolution analysis performance.
//!
//! This benchmark suite measures the performance of breaking change detection,
//! migration plan generation, and compatibility analysis.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensorlogic_adapters::{DomainInfo, EvolutionAnalyzer, PredicateInfo, SymbolTable};

/// Create a schema with specified size
fn create_schema(domains: usize, predicates_per_domain: usize) -> SymbolTable {
    let mut table = SymbolTable::new();

    for i in 0..domains {
        table
            .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
            .unwrap();
    }

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

/// Create a modified version of a schema
fn create_modified_schema(
    base: &SymbolTable,
    additions: usize,
    modifications: usize,
    removals: usize,
) -> SymbolTable {
    let mut table = base.clone();

    // Add new domains
    for i in 0..additions {
        table
            .add_domain(DomainInfo::new(format!("NewDomain{}", i), 50))
            .unwrap();
    }

    // Modify existing domains (change cardinality)
    for i in 0..modifications.min(base.domains.len()) {
        let domain_name = format!("Domain{}", i);
        if table.domains.contains_key(&domain_name) {
            // We can't directly modify, so we'll track this as a conceptual modification
            // In a real scenario, this would involve removing and re-adding
        }
    }

    // Remove some predicates
    let pred_names: Vec<_> = table.predicates.keys().cloned().collect();
    for _i in 0..removals.min(pred_names.len()) {
        // Predicates can't be directly removed in current API
        // This represents the intent
    }

    table
}

/// Benchmark evolution analysis with varying schema sizes
fn analysis_by_schema_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis_by_size");

    for size in [10, 50, 100, 200].iter() {
        let old_schema = create_schema(*size, 5);
        let new_schema = create_modified_schema(&old_schema, 5, 2, 3);
        let total_components = size * 5;

        group.throughput(Throughput::Elements(total_components as u64));

        group.bench_with_input(
            BenchmarkId::new("full_analysis", size),
            &(old_schema, new_schema),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();
                    black_box(report);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark analysis with varying change percentages
fn analysis_by_change_percentage(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis_by_change_pct");

    let base_schema = create_schema(100, 5);

    for change_pct in [1, 5, 10, 25, 50].iter() {
        let changes = (100.0 * (*change_pct as f64) / 100.0) as usize;
        let new_schema = create_modified_schema(&base_schema, changes, changes / 2, changes / 3);

        group.bench_with_input(
            BenchmarkId::new("changes", change_pct),
            &(base_schema.clone(), new_schema),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();
                    black_box(report);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark domain analysis
fn domain_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("domain_analysis");

    for size in [50, 100, 200, 500].iter() {
        let old_schema = create_schema(*size, 5);

        // Create schema with domain additions
        let mut new_with_additions = old_schema.clone();
        for i in 0..(*size / 10) {
            new_with_additions
                .add_domain(DomainInfo::new(format!("NewDomain{}", i), 50))
                .unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("additions", size),
            &(old_schema.clone(), new_with_additions),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();
                    black_box(report);
                });
            },
        );

        // Create schema with modified cardinalities
        let new_with_modifications = old_schema.clone();
        // Modifications would need API support

        group.bench_with_input(
            BenchmarkId::new("modifications", size),
            &(old_schema.clone(), new_with_modifications),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();
                    black_box(report);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark predicate analysis
fn predicate_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_analysis");

    for predicates_count in [100, 500, 1000].iter() {
        let old_schema = create_schema(20, predicates_count / 20);

        // Add new predicates
        let mut new_schema = old_schema.clone();
        for i in 0..(*predicates_count / 10) {
            let pred = PredicateInfo::new(
                format!("new_pred_{}", i),
                vec!["Domain0".to_string(), "Domain1".to_string()],
            );
            new_schema.add_predicate(pred).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("new_predicates", predicates_count),
            &(old_schema, new_schema),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();
                    black_box(report);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark migration plan generation
fn migration_plan_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("migration_plans");

    for size in [50, 100, 200].iter() {
        let old_schema = create_schema(*size, 5);
        let new_schema = create_modified_schema(&old_schema, 10, 5, 5);

        group.bench_with_input(
            BenchmarkId::new("generate_plan", size),
            &(old_schema, new_schema),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();
                    black_box(&report.migration_plan);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark breaking change detection
fn breaking_change_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("breaking_changes");

    for size in [50, 100, 200].iter() {
        let old_schema = create_schema(*size, 5);

        // Create scenario with no breaking changes
        let mut compatible_schema = old_schema.clone();
        compatible_schema
            .add_domain(DomainInfo::new("Compatible", 100))
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("no_breaking", size),
            &(old_schema.clone(), compatible_schema),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();
                    black_box(report.has_breaking_changes());
                });
            },
        );

        // Create scenario with breaking changes (empty new schema)
        let breaking_schema = SymbolTable::new();

        group.bench_with_input(
            BenchmarkId::new("with_breaking", size),
            &(old_schema, breaking_schema),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();
                    black_box(report.has_breaking_changes());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark compatibility report generation
fn compatibility_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("compatibility_report");

    for size in [50, 100, 200, 500].iter() {
        let old_schema = create_schema(*size, 5);
        let new_schema = create_modified_schema(&old_schema, 10, 5, 2);

        group.bench_with_input(
            BenchmarkId::new("full_report", size),
            &(old_schema, new_schema),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();

                    // Access all report components
                    black_box(report.has_breaking_changes());
                    black_box(&report.breaking_changes);
                    black_box(&report.backward_compatible_changes);
                    black_box(&report.migration_plan);
                    black_box(report.suggested_version_bump());
                    black_box(report.max_impact());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark affected predicate detection
fn affected_predicate_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("affected_predicates");

    for size in [50, 100, 200].iter() {
        let mut old_schema = SymbolTable::new();

        // Create schema with interdependent predicates
        for i in 0..*size {
            old_schema
                .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                .unwrap();
        }

        // Create predicates that reference multiple domains
        for i in 0..*size {
            for j in 0..5 {
                let domains = vec![format!("Domain{}", i), format!("Domain{}", (i + 1) % size)];
                let pred = PredicateInfo::new(format!("pred_{}_{}", i, j), domains);
                old_schema.add_predicate(pred).unwrap();
            }
        }

        // Remove one domain in new schema
        let new_schema = SymbolTable::new(); // Simplified for benchmark

        group.bench_with_input(
            BenchmarkId::new("find_affected", size),
            &(old_schema, new_schema),
            |b, (old, new)| {
                b.iter(|| {
                    let analyzer = EvolutionAnalyzer::new(old, new);
                    let report = analyzer.analyze().unwrap();
                    black_box(&report.breaking_changes);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    analysis_by_schema_size,
    analysis_by_change_percentage,
    domain_analysis,
    predicate_analysis,
    migration_plan_generation,
    breaking_change_detection,
    compatibility_report,
    affected_predicate_detection,
);
criterion_main!(benches);
