//! Benchmarks for basic kernel computation operations
//!
//! This benchmark suite measures the performance of individual kernel
//! computations across different kernel types and input sizes.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensorlogic_ir::TLExpr;
use tensorlogic_sklears_kernels::{
    CosineKernel, EditDistanceKernel, Kernel, LinearKernel, NGramKernel, NGramKernelConfig,
    PolynomialKernel, PredicateOverlapKernel, RbfKernel, RbfKernelConfig, RuleSimilarityConfig,
    RuleSimilarityKernel, SubsequenceKernel, SubsequenceKernelConfig,
};

/// Generate random feature vectors
fn generate_vectors(dim: usize, count: usize) -> Vec<Vec<f64>> {
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

/// Benchmark linear kernel computation
fn bench_linear_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_kernel");

    for size in [10, 50, 100, 500, 1000].iter() {
        let vectors = generate_vectors(*size, 2);
        let kernel = LinearKernel::new();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute(&vectors[0], &vectors[1]).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark RBF kernel computation
fn bench_rbf_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf_kernel");

    for size in [10, 50, 100, 500, 1000].iter() {
        let vectors = generate_vectors(*size, 2);
        let config = RbfKernelConfig::new(0.5);
        let kernel = RbfKernel::new(config).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute(&vectors[0], &vectors[1]).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark polynomial kernel computation
fn bench_polynomial_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_kernel");

    for degree in [2, 3, 4, 5].iter() {
        let vectors = generate_vectors(100, 2);
        let kernel = PolynomialKernel::new(*degree, 1.0).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(degree), degree, |b, _| {
            b.iter(|| {
                black_box(kernel.compute(&vectors[0], &vectors[1]).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark cosine kernel computation
fn bench_cosine_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_kernel");

    for size in [10, 50, 100, 500, 1000].iter() {
        let vectors = generate_vectors(*size, 2);
        let kernel = CosineKernel::new();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute(&vectors[0], &vectors[1]).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark rule similarity kernel
fn bench_rule_similarity_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_similarity_kernel");

    for num_rules in [5, 10, 20, 50, 100].iter() {
        let rules = generate_rules(*num_rules);
        let config = RuleSimilarityConfig::new();
        let kernel = RuleSimilarityKernel::new(rules, config).unwrap();

        let x = vec![1.0; *num_rules];
        let y = vec![0.5; *num_rules];

        group.throughput(Throughput::Elements(*num_rules as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_rules), num_rules, |b, _| {
            b.iter(|| {
                black_box(kernel.compute(&x, &y).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark predicate overlap kernel
fn bench_predicate_overlap_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_overlap_kernel");

    for num_preds in [10, 50, 100, 500, 1000].iter() {
        let kernel = PredicateOverlapKernel::new(*num_preds);
        let x = vec![1.0; *num_preds];
        let y = vec![0.5; *num_preds];

        group.throughput(Throughput::Elements(*num_preds as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_preds), num_preds, |b, _| {
            b.iter(|| {
                black_box(kernel.compute(&x, &y).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark n-gram string kernel
fn bench_ngram_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("ngram_kernel");

    let text1 = "the quick brown fox jumps over the lazy dog";
    let text2 = "the quick brown cat jumps over the lazy dog";

    for n in [2, 3, 4, 5].iter() {
        let config = NGramKernelConfig::new(*n).unwrap();
        let kernel = NGramKernel::new(config);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_strings(text1, text2).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark subsequence string kernel
fn bench_subsequence_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("subsequence_kernel");

    let text1 = "machine learning";
    let text2 = "machine_learning";

    for max_len in [2, 3, 4].iter() {
        let config = SubsequenceKernelConfig::new()
            .with_max_length(*max_len)
            .unwrap()
            .with_decay(0.5)
            .unwrap();
        let kernel = SubsequenceKernel::new(config);

        group.bench_with_input(BenchmarkId::from_parameter(max_len), max_len, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_strings(text1, text2).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark edit distance kernel
fn bench_edit_distance_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("edit_distance_kernel");

    let texts = [
        ("color", "colour"),
        ("algorithm", "logarithm"),
        ("tensorlogic", "tensorflux"),
    ];

    for (i, (text1, text2)) in texts.iter().enumerate() {
        let kernel = EditDistanceKernel::new(0.1).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("pair_{}", i)),
            &(text1, text2),
            |b, (t1, t2)| {
                b.iter(|| {
                    black_box(kernel.compute_strings(t1, t2).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark kernel comparison across types
fn bench_kernel_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_comparison");

    let size = 100;
    let vectors = generate_vectors(size, 2);

    // Linear
    group.bench_function("linear", |b| {
        let kernel = LinearKernel::new();
        b.iter(|| {
            black_box(kernel.compute(&vectors[0], &vectors[1]).unwrap());
        });
    });

    // RBF
    group.bench_function("rbf", |b| {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        b.iter(|| {
            black_box(kernel.compute(&vectors[0], &vectors[1]).unwrap());
        });
    });

    // Polynomial (degree 3)
    group.bench_function("polynomial", |b| {
        let kernel = PolynomialKernel::new(3, 1.0).unwrap();
        b.iter(|| {
            black_box(kernel.compute(&vectors[0], &vectors[1]).unwrap());
        });
    });

    // Cosine
    group.bench_function("cosine", |b| {
        let kernel = CosineKernel::new();
        b.iter(|| {
            black_box(kernel.compute(&vectors[0], &vectors[1]).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_kernel,
    bench_rbf_kernel,
    bench_polynomial_kernel,
    bench_cosine_kernel,
    bench_rule_similarity_kernel,
    bench_predicate_overlap_kernel,
    bench_ngram_kernel,
    bench_subsequence_kernel,
    bench_edit_distance_kernel,
    bench_kernel_comparison,
);

criterion_main!(benches);
