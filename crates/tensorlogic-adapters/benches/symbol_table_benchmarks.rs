//! Performance benchmarks for tensorlogic-adapters.
//!
//! Run with: cargo bench -p tensorlogic-adapters
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tensorlogic_adapters::{
    DomainHierarchy, DomainInfo, LookupCache, PredicateInfo, SchemaAnalyzer, SchemaStatistics,
    SchemaValidator, SignatureMatcher, StringInterner, SymbolTable,
};

/// Benchmark domain addition to symbol table
fn bench_domain_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("domain_addition");

    for size in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                SymbolTable::new,
                |mut table| {
                    for i in 0..size {
                        table
                            .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                            .unwrap();
                    }
                    black_box(table)
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

/// Benchmark domain lookup performance
fn bench_domain_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("domain_lookup");

    for size in [10, 100, 1000] {
        // Setup: Create a symbol table with domains
        let mut table = SymbolTable::new();
        for i in 0..size {
            table
                .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                .unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                for i in 0..size {
                    black_box(table.get_domain(&format!("Domain{}", i)));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark predicate addition
fn bench_predicate_addition(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_addition");

    for size in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut table = SymbolTable::new();
                    table.add_domain(DomainInfo::new("Person", 100)).unwrap();
                    table
                },
                |mut table| {
                    for i in 0..size {
                        table
                            .add_predicate(PredicateInfo::new(
                                format!("pred{}", i),
                                vec!["Person".to_string()],
                            ))
                            .unwrap();
                    }
                    black_box(table)
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

/// Benchmark JSON serialization
fn bench_json_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_serialization");

    for size in [10, 100, 1000] {
        let mut table = SymbolTable::new();
        for i in 0..size {
            table
                .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                .unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(table.to_json().unwrap());
            })
        });
    }

    group.finish();
}

/// Benchmark JSON deserialization
fn bench_json_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_deserialization");

    for size in [10, 100, 1000] {
        let mut table = SymbolTable::new();
        for i in 0..size {
            table
                .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                .unwrap();
        }
        let json = table.to_json().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(SymbolTable::from_json(&json).unwrap());
            })
        });
    }

    group.finish();
}

/// Benchmark YAML serialization
fn bench_yaml_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("yaml_serialization");

    for size in [10, 100, 1000] {
        let mut table = SymbolTable::new();
        for i in 0..size {
            table
                .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                .unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(table.to_yaml().unwrap());
            })
        });
    }

    group.finish();
}

/// Benchmark schema validation
fn bench_schema_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_validation");

    for size in [10, 100, 500] {
        let mut table = SymbolTable::new();
        for i in 0..size {
            table
                .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                .unwrap();
        }
        for i in 0..size {
            table
                .add_predicate(PredicateInfo::new(
                    format!("pred{}", i),
                    vec![format!("Domain{}", i % size)],
                ))
                .unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let validator = SchemaValidator::new(&table);
                black_box(validator.validate().unwrap());
            })
        });
    }

    group.finish();
}

/// Benchmark string interning
fn bench_string_interning(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_interning");

    for size in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                StringInterner::new,
                |mut interner| {
                    for i in 0..size {
                        interner.intern(&format!("String{}", i % 100)); // Some repetition
                    }
                    black_box(interner)
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

/// Benchmark string resolution from interner
fn bench_string_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_resolution");

    for size in [100, 1000, 10000] {
        let mut interner = StringInterner::new();
        let ids: Vec<usize> = (0..size)
            .map(|i| interner.intern(&format!("String{}", i)))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                for &id in &ids {
                    black_box(interner.resolve(id));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark lookup cache
fn bench_lookup_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_cache");

    for capacity in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            &capacity,
            |b, &capacity| {
                b.iter_batched(
                    || LookupCache::new(capacity),
                    |mut cache| {
                        for i in 0..capacity * 2 {
                            cache.insert(format!("key{}", i % capacity), i);
                        }
                        for i in 0..capacity {
                            black_box(cache.get(&format!("key{}", i)));
                        }
                        black_box(cache)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark domain hierarchy operations
fn bench_domain_hierarchy(c: &mut Criterion) {
    let mut group = c.benchmark_group("domain_hierarchy");

    for size in [10, 50, 100] {
        let mut hierarchy = DomainHierarchy::new();
        // Create a chain: D0 <: D1 <: D2 <: ... <: Dn
        for i in 0..size {
            hierarchy.add_subtype(format!("D{}", i), format!("D{}", i + 1));
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                // Test subtype checking across the entire chain
                for i in 0..size {
                    black_box(hierarchy.is_subtype(&format!("D{}", i), &format!("D{}", size)));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark hierarchy ancestor queries
fn bench_hierarchy_ancestors(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchy_ancestors");

    for depth in [5, 10, 20] {
        let mut hierarchy = DomainHierarchy::new();
        // Create a chain
        for i in 0..depth {
            hierarchy.add_subtype(format!("D{}", i), format!("D{}", i + 1));
        }

        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, _| {
            b.iter(|| {
                black_box(hierarchy.get_ancestors("D0"));
            })
        });
    }

    group.finish();
}

/// Benchmark memory usage statistics
fn bench_memory_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_stats");

    for size in [100, 1000, 10000] {
        let mut interner = StringInterner::new();
        for i in 0..size {
            interner.intern(&format!("String{}", i));
        }

        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                black_box(interner.memory_usage());
            })
        });
    }

    group.finish();
}

/// Benchmark SignatureMatcher construction
fn bench_signature_matcher_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("signature_matcher_build");

    for size in [10, 50, 100, 500] {
        // Create predicates with varying arities and signatures
        let predicates: Vec<PredicateInfo> = (0..size)
            .map(|i| {
                let arity = (i % 5) + 1; // Arities 1-5
                let domains: Vec<String> = (0..arity).map(|j| format!("Domain{}", j % 3)).collect();
                PredicateInfo::new(format!("pred{}", i), domains)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &predicates,
            |b, preds| {
                b.iter(|| {
                    let matcher = SignatureMatcher::from_predicates(preds.iter());
                    black_box(matcher)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark SignatureMatcher arity lookup
fn bench_signature_matcher_find_by_arity(c: &mut Criterion) {
    let mut group = c.benchmark_group("signature_matcher_find_by_arity");

    for size in [10, 50, 100, 500] {
        // Setup: Build a matcher with predicates
        let predicates: Vec<PredicateInfo> = (0..size)
            .map(|i| {
                let arity = (i % 5) + 1;
                let domains: Vec<String> = (0..arity).map(|j| format!("Domain{}", j % 3)).collect();
                PredicateInfo::new(format!("pred{}", i), domains)
            })
            .collect();

        let matcher = SignatureMatcher::from_predicates(predicates.iter());

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                // Lookup predicates with arity 2
                black_box(matcher.find_by_arity(2))
            })
        });
    }

    group.finish();
}

/// Benchmark SignatureMatcher signature lookup
fn bench_signature_matcher_find_by_signature(c: &mut Criterion) {
    let mut group = c.benchmark_group("signature_matcher_find_by_signature");

    for size in [10, 50, 100, 500] {
        // Setup
        let predicates: Vec<PredicateInfo> = (0..size)
            .map(|i| {
                let arity = (i % 5) + 1;
                let domains: Vec<String> = (0..arity).map(|j| format!("Domain{}", j % 3)).collect();
                PredicateInfo::new(format!("pred{}", i), domains)
            })
            .collect();

        let matcher = SignatureMatcher::from_predicates(predicates.iter());
        let signature = vec!["Domain0".to_string(), "Domain1".to_string()];

        group.bench_with_input(BenchmarkId::from_parameter(size), &signature, |b, sig| {
            b.iter(|| black_box(matcher.find_by_signature(sig)))
        });
    }

    group.finish();
}

/// Benchmark schema statistics computation
fn bench_schema_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_statistics");

    for size in [10, 50, 100] {
        // Setup: Create a schema
        let mut table = SymbolTable::new();

        // Add domains
        for i in 0..size {
            table
                .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                .unwrap();
        }

        // Add predicates
        for i in 0..size * 2 {
            let arity = (i % 3) + 1;
            let domains: Vec<String> = (0..arity).map(|j| format!("Domain{}", j % size)).collect();
            table
                .add_predicate(PredicateInfo::new(format!("pred{}", i), domains))
                .unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, tbl| {
            b.iter(|| black_box(SchemaStatistics::compute(tbl)))
        });
    }

    group.finish();
}

/// Benchmark schema analysis
fn bench_schema_analyzer(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_analyzer");

    for size in [10, 50, 100] {
        // Setup: Create a schema with some issues
        let mut table = SymbolTable::new();

        // Add domains (some unused)
        for i in 0..size {
            table
                .add_domain(DomainInfo::new(
                    format!("Domain{}", i),
                    if i % 5 == 0 { 0 } else { 100 },
                ))
                .unwrap();
        }

        // Add predicates (not using all domains)
        for i in 0..size {
            let arity = (i % 3) + 1;
            let domains: Vec<String> = (0..arity)
                .map(|j| format!("Domain{}", j % (size / 2).max(1)))
                .collect();
            table
                .add_predicate(PredicateInfo::new(format!("pred{}", i), domains))
                .unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &table, |b, tbl| {
            b.iter(|| black_box(SchemaAnalyzer::analyze(tbl)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_domain_addition,
    bench_domain_lookup,
    bench_predicate_addition,
    bench_json_serialization,
    bench_json_deserialization,
    bench_yaml_serialization,
    bench_schema_validation,
    bench_string_interning,
    bench_string_resolution,
    bench_lookup_cache,
    bench_domain_hierarchy,
    bench_hierarchy_ancestors,
    bench_memory_stats,
    bench_signature_matcher_build,
    bench_signature_matcher_find_by_arity,
    bench_signature_matcher_find_by_signature,
    bench_schema_statistics,
    bench_schema_analyzer,
);

criterion_main!(benches);
