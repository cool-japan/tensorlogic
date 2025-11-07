//! Benchmarks for kernel matrix operations
//!
//! This benchmark suite measures the performance of computing full kernel
//! matrices for datasets of varying sizes.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensorlogic_ir::TLExpr;
use tensorlogic_sklears_kernels::{
    CosineKernel, Kernel, LinearKernel, PolynomialKernel, PredicateOverlapKernel, RbfKernel,
    RbfKernelConfig, RuleSimilarityConfig, RuleSimilarityKernel, SparseKernelMatrixBuilder,
};

/// Generate random feature vectors
fn generate_dataset(dim: usize, count: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|i| (0..dim).map(|j| ((i * dim + j) as f64).sin()).collect())
        .collect()
}

/// Generate logical rules for RuleSimilarityKernel
fn generate_rules(count: usize) -> Vec<TLExpr> {
    (0..count)
        .map(|i| TLExpr::pred(format!("rule{}", i), vec![]))
        .collect()
}

/// Benchmark linear kernel matrix computation
fn bench_linear_kernel_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_kernel_matrix");

    for size in [10, 25, 50, 100].iter() {
        let data = generate_dataset(50, *size);
        let kernel = LinearKernel::new();

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_matrix(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark RBF kernel matrix computation
fn bench_rbf_kernel_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf_kernel_matrix");

    for size in [10, 25, 50, 100].iter() {
        let data = generate_dataset(50, *size);
        let config = RbfKernelConfig::new(0.5);
        let kernel = RbfKernel::new(config).unwrap();

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_matrix(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark polynomial kernel matrix computation
fn bench_polynomial_kernel_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_kernel_matrix");

    let size = 50;
    let data = generate_dataset(50, size);

    for degree in [2, 3, 4].iter() {
        let kernel = PolynomialKernel::new(*degree, 1.0).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(degree), degree, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_matrix(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark cosine kernel matrix computation
fn bench_cosine_kernel_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_kernel_matrix");

    for size in [10, 25, 50, 100].iter() {
        let data = generate_dataset(50, *size);
        let kernel = CosineKernel::new();

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_matrix(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark rule similarity kernel matrix
fn bench_rule_similarity_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_similarity_matrix");

    let num_rules = 20;
    let rules = generate_rules(num_rules);
    let config = RuleSimilarityConfig::new();
    let kernel = RuleSimilarityKernel::new(rules, config).unwrap();

    for size in [10, 25, 50, 100].iter() {
        let data = vec![vec![0.5; num_rules]; *size];

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_matrix(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark predicate overlap kernel matrix
fn bench_predicate_overlap_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_overlap_matrix");

    let num_preds = 50;
    let kernel = PredicateOverlapKernel::new(num_preds);

    for size in [10, 25, 50, 100].iter() {
        let data = vec![vec![0.5; num_preds]; *size];

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_matrix(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark sparse kernel matrix construction
fn bench_sparse_matrix_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matrix_construction");

    let kernel = LinearKernel::new();

    for size in [50, 100, 200].iter() {
        let data = generate_dataset(50, *size);

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("threshold_0.1_size_{}", size)),
            size,
            |b, _| {
                let builder = SparseKernelMatrixBuilder::new()
                    .with_threshold(0.1)
                    .unwrap();

                b.iter(|| {
                    black_box(builder.build(&data, &kernel).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sparse matrix with max entries constraint
fn bench_sparse_matrix_max_entries(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matrix_max_entries");

    let size = 100;
    let data = generate_dataset(50, size);
    let kernel = LinearKernel::new();

    for max_entries in [10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(max_entries),
            max_entries,
            |b, _| {
                let builder = SparseKernelMatrixBuilder::new()
                    .with_threshold(0.0)
                    .unwrap()
                    .with_max_entries_per_row(*max_entries)
                    .unwrap();

                b.iter(|| {
                    black_box(builder.build(&data, &kernel).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sparse vs dense matrix construction
fn bench_sparse_vs_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vs_dense");

    let size = 100;
    let data = generate_dataset(50, size);
    let kernel = LinearKernel::new();

    // Dense matrix
    group.bench_function("dense", |b| {
        b.iter(|| {
            black_box(kernel.compute_matrix(&data).unwrap());
        });
    });

    // Sparse matrix with threshold 0.5
    group.bench_function("sparse_threshold_0.5", |b| {
        let builder = SparseKernelMatrixBuilder::new()
            .with_threshold(0.5)
            .unwrap();

        b.iter(|| {
            black_box(builder.build(&data, &kernel).unwrap());
        });
    });

    // Sparse matrix with max 20 entries per row
    group.bench_function("sparse_max_20", |b| {
        let builder = SparseKernelMatrixBuilder::new()
            .with_threshold(0.0)
            .unwrap()
            .with_max_entries_per_row(20)
            .unwrap();

        b.iter(|| {
            black_box(builder.build(&data, &kernel).unwrap());
        });
    });

    group.finish();
}

/// Benchmark matrix computation scalability
fn bench_matrix_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_scalability");

    let kernel = LinearKernel::new();

    for (samples, features) in [(50, 20), (100, 50), (200, 100)].iter() {
        let data = generate_dataset(*features, *samples);

        group.throughput(Throughput::Elements((samples * samples) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("samples_{}_features_{}", samples, features)),
            &(samples, features),
            |b, _| {
                b.iter(|| {
                    black_box(kernel.compute_matrix(&data).unwrap());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_kernel_matrix,
    bench_rbf_kernel_matrix,
    bench_polynomial_kernel_matrix,
    bench_cosine_kernel_matrix,
    bench_rule_similarity_matrix,
    bench_predicate_overlap_matrix,
    bench_sparse_matrix_construction,
    bench_sparse_matrix_max_entries,
    bench_sparse_vs_dense,
    bench_matrix_scalability,
);

criterion_main!(benches);
