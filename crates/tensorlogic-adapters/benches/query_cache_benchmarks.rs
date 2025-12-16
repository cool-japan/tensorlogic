//! Benchmarks for query result caching system.
//!
//! These benchmarks measure the performance of:
//! - Cache insertion and retrieval operations
//! - TTL-based expiration overhead
//! - LRU eviction performance
//! - Cache hit/miss latency
//! - Symbol table query caching speedup
//! - Large-scale cache operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box as std_black_box;
use std::time::Duration;
use tensorlogic_adapters::{
    CacheConfig, CacheKey, DomainInfo, PredicateInfo, QueryCache, SymbolTable, SymbolTableCache,
};

// Helper function to create a large symbol table for benchmarks
fn create_large_symbol_table(num_domains: usize, num_predicates: usize) -> SymbolTable {
    let mut table = SymbolTable::new();

    // Add domains
    for i in 0..num_domains {
        table
            .add_domain(DomainInfo::new(format!("Domain{}", i), 100))
            .unwrap();
    }

    // Add predicates with varying arities
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

/// Benchmark basic cache operations (insert, get)
fn bench_cache_basic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_basic_operations");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut cache: QueryCache<Vec<String>> = QueryCache::new();

                // Insert operations
                for i in 0..size {
                    let key = CacheKey::Custom(format!("key{}", i));
                    let value = vec![format!("value{}", i)];
                    cache.insert(std_black_box(key), std_black_box(value));
                }

                // Get operations
                for i in 0..size {
                    let key = CacheKey::Custom(format!("key{}", i));
                    std_black_box(cache.get(&key));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark cache hit vs miss latency
fn bench_cache_hit_miss_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_hit_miss_latency");

    let mut cache: QueryCache<Vec<String>> = QueryCache::new();

    // Populate cache
    for i in 0..1000 {
        let key = CacheKey::Custom(format!("key{}", i));
        cache.insert(key, vec![format!("value{}", i)]);
    }

    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            let key = CacheKey::Custom(format!("key{}", std_black_box(500)));
            std_black_box(cache.get(&key));
        });
    });

    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            let key = CacheKey::Custom(format!("missing_key{}", std_black_box(2000)));
            std_black_box(cache.get(&key));
        });
    });

    group.finish();
}

/// Benchmark TTL expiration overhead
fn bench_cache_ttl_expiration(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_ttl_expiration");

    // Benchmark with TTL vs without TTL
    group.bench_function("insert_with_ttl", |b| {
        let config = CacheConfig {
            max_entries: 10000,
            default_ttl: Some(Duration::from_secs(60)),
            enable_lru: true,
            enable_stats: true,
        };
        let mut cache: QueryCache<String> = QueryCache::with_config(config);

        b.iter(|| {
            for i in 0..1000 {
                let key = CacheKey::Custom(format!("key{}", i));
                cache.insert(std_black_box(key), std_black_box(format!("value{}", i)));
            }
        });
    });

    group.bench_function("insert_without_ttl", |b| {
        let config = CacheConfig {
            max_entries: 10000,
            default_ttl: None,
            enable_lru: true,
            enable_stats: true,
        };
        let mut cache: QueryCache<String> = QueryCache::with_config(config);

        b.iter(|| {
            for i in 0..1000 {
                let key = CacheKey::Custom(format!("key{}", i));
                cache.insert(std_black_box(key), std_black_box(format!("value{}", i)));
            }
        });
    });

    group.bench_function("cleanup_expired", |b| {
        let config = CacheConfig {
            max_entries: 10000,
            default_ttl: Some(Duration::from_millis(1)), // Very short TTL
            enable_lru: true,
            enable_stats: true,
        };
        let mut cache: QueryCache<String> = QueryCache::with_config(config);

        // Insert many entries
        for i in 0..1000 {
            let key = CacheKey::Custom(format!("key{}", i));
            cache.insert(key, format!("value{}", i));
        }

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        b.iter(|| {
            std_black_box(cache.cleanup_expired());
        });
    });

    group.finish();
}

/// Benchmark LRU eviction performance
fn bench_cache_lru_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_lru_eviction");

    for max_entries in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(max_entries),
            max_entries,
            |b, &max_entries| {
                b.iter(|| {
                    let config = CacheConfig {
                        max_entries,
                        default_ttl: None,
                        enable_lru: true,
                        enable_stats: true,
                    };
                    let mut cache: QueryCache<String> = QueryCache::with_config(config);

                    // Insert more than max_entries to trigger evictions
                    for i in 0..(max_entries * 2) {
                        let key = CacheKey::Custom(format!("key{}", i));
                        cache.insert(std_black_box(key), std_black_box(format!("value{}", i)));
                    }

                    std_black_box(&cache);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark symbol table query caching - arity queries
fn bench_symbol_table_cache_arity(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_table_cache_arity");

    for (domains, predicates) in [(100, 1000), (500, 5000), (1000, 10000)].iter() {
        let table = create_large_symbol_table(*domains, *predicates);
        group.throughput(Throughput::Elements(*predicates as u64));

        // Benchmark without cache
        group.bench_with_input(
            BenchmarkId::new("no_cache", format!("{}d_{}p", domains, predicates)),
            &table,
            |b, table| {
                b.iter(|| {
                    let result: Vec<_> = table
                        .predicates
                        .values()
                        .filter(|p| p.arg_domains.len() == 2)
                        .cloned()
                        .collect();
                    std_black_box(result);
                });
            },
        );

        // Benchmark with cache (first access - miss)
        group.bench_with_input(
            BenchmarkId::new("cache_miss", format!("{}d_{}p", domains, predicates)),
            &table,
            |b, table| {
                b.iter(|| {
                    let mut cache = SymbolTableCache::new();
                    let result = cache.get_predicates_by_arity(std_black_box(table), 2);
                    std_black_box(result);
                });
            },
        );

        // Benchmark with cache (subsequent access - hit)
        group.bench_with_input(
            BenchmarkId::new("cache_hit", format!("{}d_{}p", domains, predicates)),
            &table,
            |b, table| {
                let mut cache = SymbolTableCache::new();
                // Warm up cache
                cache.get_predicates_by_arity(table, 2);

                b.iter(|| {
                    let result = cache.get_predicates_by_arity(std_black_box(table), 2);
                    std_black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark symbol table query caching - domain queries
fn bench_symbol_table_cache_domain(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_table_cache_domain");

    for (domains, predicates) in [(100, 1000), (500, 5000)].iter() {
        let table = create_large_symbol_table(*domains, *predicates);
        let domain_name = "Domain0";

        // Benchmark without cache
        group.bench_with_input(
            BenchmarkId::new("no_cache", format!("{}d_{}p", domains, predicates)),
            &table,
            |b, table| {
                b.iter(|| {
                    let result: Vec<_> = table
                        .predicates
                        .values()
                        .filter(|p| p.arg_domains.contains(&domain_name.to_string()))
                        .cloned()
                        .collect();
                    std_black_box(result);
                });
            },
        );

        // Benchmark with cache (hit)
        group.bench_with_input(
            BenchmarkId::new("cache_hit", format!("{}d_{}p", domains, predicates)),
            &table,
            |b, table| {
                let mut cache = SymbolTableCache::new();
                // Warm up cache
                cache.get_predicates_by_domain(table, domain_name);

                b.iter(|| {
                    let result = cache.get_predicates_by_domain(std_black_box(table), domain_name);
                    std_black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cache statistics tracking overhead
fn bench_cache_statistics_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_statistics_overhead");

    // Benchmark with statistics enabled
    group.bench_function("stats_enabled", |b| {
        let config = CacheConfig {
            max_entries: 10000,
            default_ttl: None,
            enable_lru: true,
            enable_stats: true,
        };
        let mut cache: QueryCache<String> = QueryCache::with_config(config);

        b.iter(|| {
            for i in 0..1000 {
                let key = CacheKey::Custom(format!("key{}", i));
                cache.insert(key.clone(), format!("value{}", i));
                std_black_box(cache.get(&key));
            }
        });
    });

    // Benchmark with statistics disabled
    group.bench_function("stats_disabled", |b| {
        let config = CacheConfig {
            max_entries: 10000,
            default_ttl: None,
            enable_lru: true,
            enable_stats: false,
        };
        let mut cache: QueryCache<String> = QueryCache::with_config(config);

        b.iter(|| {
            for i in 0..1000 {
                let key = CacheKey::Custom(format!("key{}", i));
                cache.insert(key.clone(), format!("value{}", i));
                std_black_box(cache.get(&key));
            }
        });
    });

    group.finish();
}

/// Benchmark cache invalidation operations
fn bench_cache_invalidation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_invalidation");

    group.bench_function("invalidate_single", |b| {
        let mut cache: QueryCache<String> = QueryCache::new();

        // Populate cache
        for i in 0..1000 {
            let key = CacheKey::Custom(format!("key{}", i));
            cache.insert(key, format!("value{}", i));
        }

        b.iter(|| {
            let key = CacheKey::Custom(format!("key{}", std_black_box(500)));
            std_black_box(cache.invalidate(&key));
        });
    });

    group.bench_function("clear_all", |b| {
        b.iter(|| {
            let mut cache: QueryCache<String> = QueryCache::new();

            // Populate cache
            for i in 0..1000 {
                let key = CacheKey::Custom(format!("key{}", i));
                cache.insert(key, format!("value{}", i));
            }

            cache.clear();
            std_black_box(&cache);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_basic_operations,
    bench_cache_hit_miss_latency,
    bench_cache_ttl_expiration,
    bench_cache_lru_eviction,
    bench_symbol_table_cache_arity,
    bench_symbol_table_cache_domain,
    bench_cache_statistics_overhead,
    bench_cache_invalidation,
);

criterion_main!(benches);
