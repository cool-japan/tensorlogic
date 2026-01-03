//! Benchmarks for schema merging strategies.
//!
//! These benchmarks measure the performance of:
//! - Different merge strategies (KeepFirst, KeepSecond, Union, Intersection, FailOnConflict)
//! - Merging schemas of varying sizes
//! - Conflict detection and resolution
//! - Merge report generation
//! - Compatible predicate detection

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box as std_black_box;
use tensorlogic_adapters::{DomainInfo, MergeStrategy, PredicateInfo, SchemaMerger, SymbolTable};

/// Helper function to create a symbol table with specified size
fn create_symbol_table(
    name_prefix: &str,
    num_domains: usize,
    num_predicates: usize,
) -> SymbolTable {
    let mut table = SymbolTable::new();

    // Add domains
    for i in 0..num_domains {
        table
            .add_domain(DomainInfo::new(
                format!("{}Domain{}", name_prefix, i),
                100 + i,
            ))
            .unwrap();
    }

    // Add predicates
    for i in 0..num_predicates {
        let arity = (i % 3) + 1;
        let domains: Vec<String> = (0..arity)
            .map(|j| format!("{}Domain{}", name_prefix, (i + j) % num_domains))
            .collect();
        table
            .add_predicate(PredicateInfo::new(
                format!("{}Pred{}", name_prefix, i),
                domains,
            ))
            .unwrap();
    }

    table
}

/// Helper function to create overlapping symbol tables
fn create_overlapping_tables(
    overlap_ratio: f64,
    num_domains: usize,
    num_predicates: usize,
) -> (SymbolTable, SymbolTable) {
    let overlap_domains = (num_domains as f64 * overlap_ratio) as usize;
    let overlap_predicates = (num_predicates as f64 * overlap_ratio) as usize;

    let mut base = SymbolTable::new();
    let mut incoming = SymbolTable::new();

    // Add overlapping domains to both
    for i in 0..overlap_domains {
        let domain = DomainInfo::new(format!("Domain{}", i), 100);
        base.add_domain(domain.clone()).unwrap();
        incoming.add_domain(domain).unwrap();
    }

    // Add unique domains to base
    for i in overlap_domains..num_domains {
        base.add_domain(DomainInfo::new(format!("BaseDomain{}", i), 100))
            .unwrap();
    }

    // Add unique domains to incoming
    for i in overlap_domains..num_domains {
        incoming
            .add_domain(DomainInfo::new(format!("IncomingDomain{}", i), 100))
            .unwrap();
    }

    // Add overlapping predicates
    for i in 0..overlap_predicates {
        let domain_idx = if overlap_domains > 0 {
            i % overlap_domains
        } else {
            0
        };
        let pred = PredicateInfo::new(format!("Pred{}", i), vec![format!("Domain{}", domain_idx)]);
        base.add_predicate(pred.clone()).unwrap();
        incoming.add_predicate(pred).unwrap();
    }

    // Add unique predicates to base
    for i in overlap_predicates..num_predicates {
        let domain_name = if overlap_domains > 0 {
            format!("Domain{}", i % overlap_domains)
        } else {
            format!(
                "BaseDomain{}",
                overlap_domains + (i % (num_domains - overlap_domains).max(1))
            )
        };
        base.add_predicate(PredicateInfo::new(
            format!("BasePred{}", i),
            vec![domain_name],
        ))
        .unwrap();
    }

    // Add unique predicates to incoming
    for i in overlap_predicates..num_predicates {
        let domain_name = if overlap_domains > 0 {
            format!("Domain{}", i % overlap_domains)
        } else {
            format!(
                "IncomingDomain{}",
                overlap_domains + (i % (num_domains - overlap_domains).max(1))
            )
        };
        incoming
            .add_predicate(PredicateInfo::new(
                format!("IncomingPred{}", i),
                vec![domain_name],
            ))
            .unwrap();
    }

    (base, incoming)
}

/// Benchmark merge strategies - no conflicts
fn bench_merge_no_conflicts(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_no_conflicts");

    for size in [10, 50, 100, 500].iter() {
        let base = create_symbol_table("Base", *size, *size);
        let incoming = create_symbol_table("Incoming", *size, *size);

        group.throughput(Throughput::Elements(*size as u64 * 2));

        for strategy in [
            MergeStrategy::KeepFirst,
            MergeStrategy::KeepSecond,
            MergeStrategy::Union,
        ]
        .iter()
        {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{:?}", strategy),
                    format!("{}domains_{}preds", size, size),
                ),
                &(base.clone(), incoming.clone()),
                |b, (base, incoming)| {
                    let merger = SchemaMerger::new(*strategy);
                    b.iter(|| {
                        let result = merger
                            .merge(std_black_box(base), std_black_box(incoming))
                            .unwrap();
                        std_black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark merge strategies with conflicts
fn bench_merge_with_conflicts(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_with_conflicts");

    for overlap in [0.25, 0.5, 0.75].iter() {
        let (base, incoming) = create_overlapping_tables(*overlap, 100, 100);

        group.throughput(Throughput::Elements(100));

        for strategy in [
            MergeStrategy::KeepFirst,
            MergeStrategy::KeepSecond,
            MergeStrategy::Union,
        ]
        .iter()
        {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{:?}", strategy),
                    format!("{}%_overlap", overlap * 100.0),
                ),
                &(base.clone(), incoming.clone()),
                |b, (base, incoming)| {
                    let merger = SchemaMerger::new(*strategy);
                    b.iter(|| {
                        let result = merger
                            .merge(std_black_box(base), std_black_box(incoming))
                            .unwrap();
                        std_black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark KeepFirst strategy across different schema sizes
fn bench_keep_first_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("keep_first_scaling");

    for size in [10, 50, 100, 500, 1000].iter() {
        let (base, incoming) = create_overlapping_tables(0.5, *size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(base, incoming),
            |b, (base, incoming)| {
                let merger = SchemaMerger::new(MergeStrategy::KeepFirst);
                b.iter(|| {
                    let result = merger
                        .merge(std_black_box(base), std_black_box(incoming))
                        .unwrap();
                    std_black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Union strategy performance
fn bench_union_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("union_strategy");

    // Test with different overlap ratios
    for overlap in [0.0, 0.25, 0.5, 0.75, 1.0].iter() {
        let (base, incoming) = create_overlapping_tables(*overlap, 200, 200);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%", overlap * 100.0)),
            &(base, incoming),
            |b, (base, incoming)| {
                let merger = SchemaMerger::new(MergeStrategy::Union);
                b.iter(|| {
                    let result = merger
                        .merge(std_black_box(base), std_black_box(incoming))
                        .unwrap();
                    std_black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Intersection strategy performance
fn bench_intersection_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersection_strategy");

    // Test with different overlap ratios
    for overlap in [0.25, 0.5, 0.75].iter() {
        let (base, incoming) = create_overlapping_tables(*overlap, 200, 200);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%", overlap * 100.0)),
            &(base, incoming),
            |b, (base, incoming)| {
                let merger = SchemaMerger::new(MergeStrategy::Intersection);
                b.iter(|| {
                    let result = merger
                        .merge(std_black_box(base), std_black_box(incoming))
                        .unwrap();
                    std_black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark merge report generation overhead
fn bench_merge_report_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_report_generation");

    let (base, incoming) = create_overlapping_tables(0.5, 100, 100);

    group.bench_function("with_report", |b| {
        let merger = SchemaMerger::new(MergeStrategy::Union);
        b.iter(|| {
            let result = merger
                .merge(std_black_box(&base), std_black_box(&incoming))
                .unwrap();
            // Access report to ensure it's generated
            std_black_box(&result.report);
        });
    });

    group.finish();
}

/// Benchmark conflict detection performance
fn bench_conflict_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("conflict_detection");

    for num_conflicts in [10, 50, 100, 200].iter() {
        // Create schemas with specific number of conflicts
        let mut base = SymbolTable::new();
        let mut incoming = SymbolTable::new();

        // Add conflicting domains (same name, different cardinality)
        for i in 0..*num_conflicts {
            base.add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                .unwrap();
            incoming
                .add_domain(DomainInfo::new(format!("Domain{}", i), 200))
                .unwrap();
        }

        // Add some non-conflicting items
        for i in 0..50 {
            base.add_domain(DomainInfo::new(format!("BaseOnly{}", i), 100))
                .unwrap();
            incoming
                .add_domain(DomainInfo::new(format!("IncomingOnly{}", i), 100))
                .unwrap();
        }

        group.throughput(Throughput::Elements(*num_conflicts as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_conflicts),
            &(base, incoming),
            |b, (base, incoming)| {
                let merger = SchemaMerger::new(MergeStrategy::KeepFirst);
                b.iter(|| {
                    let result = merger
                        .merge(std_black_box(base), std_black_box(incoming))
                        .unwrap();
                    std_black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark predicate signature compatibility checking
fn bench_predicate_compatibility(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_compatibility");

    for num_predicates in [50, 100, 500].iter() {
        let mut base = SymbolTable::new();
        let mut incoming = SymbolTable::new();

        // Add shared domain
        base.add_domain(DomainInfo::new("SharedDomain", 100))
            .unwrap();
        incoming
            .add_domain(DomainInfo::new("SharedDomain", 100))
            .unwrap();

        // Add predicates with same signature (compatible)
        for i in 0..*num_predicates {
            let pred = PredicateInfo::new(
                format!("Pred{}", i),
                vec!["SharedDomain".to_string(), "SharedDomain".to_string()],
            );
            base.add_predicate(pred.clone()).unwrap();
            incoming.add_predicate(pred).unwrap();
        }

        group.throughput(Throughput::Elements(*num_predicates as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_predicates),
            &(base, incoming),
            |b, (base, incoming)| {
                let merger = SchemaMerger::new(MergeStrategy::Union);
                b.iter(|| {
                    let result = merger
                        .merge(std_black_box(base), std_black_box(incoming))
                        .unwrap();
                    std_black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark large-scale merge operations
fn bench_large_scale_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_merge");
    group.sample_size(10); // Reduce sample size for expensive operations

    for size in [100, 500, 1000, 2000].iter() {
        let (base, incoming) = create_overlapping_tables(0.3, *size, *size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(base, incoming),
            |b, (base, incoming)| {
                let merger = SchemaMerger::new(MergeStrategy::Union);
                b.iter(|| {
                    let result = merger
                        .merge(std_black_box(base), std_black_box(incoming))
                        .unwrap();
                    std_black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_merge_no_conflicts,
    bench_merge_with_conflicts,
    bench_keep_first_scaling,
    bench_union_strategy,
    bench_intersection_strategy,
    bench_merge_report_generation,
    bench_conflict_detection,
    bench_predicate_compatibility,
    bench_large_scale_merge,
);

criterion_main!(benches);
