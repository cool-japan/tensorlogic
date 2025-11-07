//! Benchmarks for kernel caching performance
//!
//! This benchmark suite measures the performance benefits of kernel caching
//! and the overhead of cache management.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensorlogic_sklears_kernels::{
    CachedKernel, Kernel, KernelMatrixCache, LinearKernel, RbfKernel, RbfKernelConfig,
};

/// Generate random feature vectors
fn generate_dataset(dim: usize, count: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|i| (0..dim).map(|j| ((i * dim + j) as f64).sin()).collect())
        .collect()
}

/// Benchmark cache hit performance
fn bench_cache_hits(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_hits");

    let data = generate_dataset(50, 10);
    let base_kernel = LinearKernel::new();
    let cached = CachedKernel::new(Box::new(base_kernel));

    // Warm up cache
    for i in 0..data.len() {
        for j in 0..data.len() {
            let _ = cached.compute(&data[i], &data[j]);
        }
    }

    group.bench_function("cached_hits", |b| {
        b.iter(|| {
            black_box(cached.compute(&data[0], &data[1]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark cache miss performance
fn bench_cache_misses(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_misses");

    let dim = 50;
    let base_kernel = LinearKernel::new();
    let cached = CachedKernel::new(Box::new(base_kernel));

    group.bench_function("cache_misses", |b| {
        let mut counter = 0;
        b.iter(|| {
            // Generate unique vectors to force cache misses
            let x: Vec<f64> = (0..dim)
                .map(|i| ((counter * dim + i) as f64).sin())
                .collect();
            counter += 1;
            let y: Vec<f64> = (0..dim)
                .map(|i| ((counter * dim + i) as f64).sin())
                .collect();
            counter += 1;

            black_box(cached.compute(&x, &y).unwrap());
        });
    });

    group.finish();
}

/// Benchmark cached vs uncached kernel
fn bench_cached_vs_uncached(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_vs_uncached");

    let data = generate_dataset(50, 50);

    // Uncached kernel
    group.bench_function("uncached", |b| {
        let kernel = LinearKernel::new();
        b.iter(|| {
            // Compute same pairs repeatedly
            for i in 0..10 {
                for j in 0..10 {
                    black_box(kernel.compute(&data[i], &data[j]).unwrap());
                }
            }
        });
    });

    // Cached kernel (cold cache)
    group.bench_function("cached_cold", |b| {
        b.iter(|| {
            let kernel = LinearKernel::new();
            let cached = CachedKernel::new(Box::new(kernel));
            // Compute same pairs repeatedly
            for i in 0..10 {
                for j in 0..10 {
                    black_box(cached.compute(&data[i], &data[j]).unwrap());
                }
            }
        });
    });

    // Cached kernel (warm cache)
    group.bench_function("cached_warm", |b| {
        let kernel = LinearKernel::new();
        let cached = CachedKernel::new(Box::new(kernel));

        // Warm up cache
        for i in 0..10 {
            for j in 0..10 {
                let _ = cached.compute(&data[i], &data[j]);
            }
        }

        b.iter(|| {
            // Compute same pairs repeatedly (cache hits)
            for i in 0..10 {
                for j in 0..10 {
                    black_box(cached.compute(&data[i], &data[j]).unwrap());
                }
            }
        });
    });

    group.finish();
}

/// Benchmark cache overhead for expensive kernels
fn bench_cache_overhead_expensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_overhead_expensive");

    let data = generate_dataset(100, 20);
    let config = RbfKernelConfig::new(0.5);

    // Uncached RBF kernel
    group.bench_function("rbf_uncached", |b| {
        let kernel = RbfKernel::new(config.clone()).unwrap();
        b.iter(|| {
            black_box(kernel.compute(&data[0], &data[1]).unwrap());
        });
    });

    // Cached RBF kernel (first call - miss)
    group.bench_function("rbf_cached_miss", |b| {
        b.iter(|| {
            let kernel = RbfKernel::new(config.clone()).unwrap();
            let cached = CachedKernel::new(Box::new(kernel));
            black_box(cached.compute(&data[0], &data[1]).unwrap());
        });
    });

    // Cached RBF kernel (subsequent calls - hits)
    group.bench_function("rbf_cached_hit", |b| {
        let kernel = RbfKernel::new(config.clone()).unwrap();
        let cached = CachedKernel::new(Box::new(kernel));
        let _ = cached.compute(&data[0], &data[1]); // Warm up

        b.iter(|| {
            black_box(cached.compute(&data[0], &data[1]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark kernel matrix cache
fn bench_kernel_matrix_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_matrix_cache");

    let kernel = LinearKernel::new();

    for size in [10, 25, 50].iter() {
        let data = generate_dataset(50, *size);

        // Without cache
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("uncached_{}", size)),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.compute_matrix(&data).unwrap());
                });
            },
        );

        // With cache (cold)
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("cached_cold_{}", size)),
            size,
            |b, _| {
                b.iter(|| {
                    let mut cache = KernelMatrixCache::new();
                    black_box(cache.get_or_compute(&data, &kernel).unwrap());
                });
            },
        );

        // With cache (warm)
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("cached_warm_{}", size)),
            size,
            |b, _| {
                let mut cache = KernelMatrixCache::new();
                let _ = cache.get_or_compute(&data, &kernel); // Warm up

                b.iter(|| {
                    black_box(cache.get_or_compute(&data, &kernel).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cache statistics overhead
fn bench_cache_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_stats");

    let data = generate_dataset(50, 10);
    let base_kernel = LinearKernel::new();
    let cached = CachedKernel::new(Box::new(base_kernel));

    // Warm up cache
    for i in 0..data.len() {
        for j in 0..data.len() {
            let _ = cached.compute(&data[i], &data[j]);
        }
    }

    group.bench_function("compute_with_stats", |b| {
        b.iter(|| {
            black_box(cached.compute(&data[0], &data[1]).unwrap());
            let _ = black_box(cached.stats());
        });
    });

    group.bench_function("compute_only", |b| {
        b.iter(|| {
            black_box(cached.compute(&data[0], &data[1]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark cache size impact
fn bench_cache_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_size_impact");

    for num_unique in [10, 50, 100, 500].iter() {
        let data = generate_dataset(50, *num_unique);

        group.throughput(Throughput::Elements(*num_unique as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_unique),
            num_unique,
            |b, _| {
                b.iter(|| {
                    let kernel = LinearKernel::new();
                    let cached = CachedKernel::new(Box::new(kernel));
                    // Access different pairs to fill cache
                    for i in 0..*num_unique {
                        black_box(cached.compute(&data[i], &data[0]).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cache hit rate scenarios
fn bench_cache_hit_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_hit_rates");

    let data = generate_dataset(50, 100);

    // 100% hit rate
    group.bench_function("hit_rate_100", |b| {
        let kernel = LinearKernel::new();
        let cached = CachedKernel::new(Box::new(kernel));
        let _ = cached.compute(&data[0], &data[1]); // Prime cache

        b.iter(|| {
            black_box(cached.compute(&data[0], &data[1]).unwrap());
        });
    });

    // 50% hit rate
    group.bench_function("hit_rate_50", |b| {
        let kernel = LinearKernel::new();
        let cached = CachedKernel::new(Box::new(kernel));
        let _ = cached.compute(&data[0], &data[1]); // Prime cache

        let mut counter = 0;
        b.iter(|| {
            if counter % 2 == 0 {
                black_box(cached.compute(&data[0], &data[1]).unwrap()); // Hit
            } else {
                black_box(cached.compute(&data[0], &data[counter % 10 + 2]).unwrap());
                // Miss
            }
            counter += 1;
        });
    });

    // 0% hit rate (all misses)
    group.bench_function("hit_rate_0", |b| {
        let kernel = LinearKernel::new();
        let cached = CachedKernel::new(Box::new(kernel));

        let mut counter = 0;
        b.iter(|| {
            let idx = counter % data.len();
            black_box(
                cached
                    .compute(&data[idx], &data[(idx + 1) % data.len()])
                    .unwrap(),
            );
            counter += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cache_hits,
    bench_cache_misses,
    bench_cached_vs_uncached,
    bench_cache_overhead_expensive,
    bench_kernel_matrix_cache,
    bench_cache_stats,
    bench_cache_size_impact,
    bench_cache_hit_rates,
);

criterion_main!(benches);
