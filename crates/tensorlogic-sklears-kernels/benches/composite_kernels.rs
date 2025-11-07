//! Benchmarks for composite kernel operations
//!
//! This benchmark suite measures the performance of kernel composition
//! operations including weighted sums, products, and kernel alignment.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensorlogic_sklears_kernels::{
    CosineKernel, Kernel, KernelAlignment, LinearKernel, PolynomialKernel, ProductKernel,
    RbfKernel, RbfKernelConfig, WeightedSumKernel,
};

/// Generate random feature vectors
fn generate_dataset(dim: usize, count: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|i| (0..dim).map(|j| ((i * dim + j) as f64).sin()).collect())
        .collect()
}

/// Generate a kernel matrix
fn generate_kernel_matrix(size: usize) -> Vec<Vec<f64>> {
    let data = generate_dataset(50, size);
    let kernel = LinearKernel::new();
    kernel.compute_matrix(&data).unwrap()
}

/// Benchmark weighted sum kernel with 2 components
fn bench_weighted_sum_two(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_sum_two");

    let data = generate_dataset(50, 2);
    let k1 = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
    let k2 = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;
    let weights = vec![0.7, 0.3];
    let kernel = WeightedSumKernel::new(vec![k1, k2], weights).unwrap();

    group.bench_function("compute", |b| {
        b.iter(|| {
            black_box(kernel.compute(&data[0], &data[1]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark weighted sum kernel with multiple components
fn bench_weighted_sum_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_sum_many");

    let data = generate_dataset(50, 2);

    for num_kernels in [2, 4, 8].iter() {
        let mut kernels: Vec<Box<dyn Kernel>> = Vec::new();
        let mut weights = Vec::new();

        for i in 0..*num_kernels {
            if i % 2 == 0 {
                kernels.push(Box::new(LinearKernel::new()));
            } else {
                kernels.push(Box::new(CosineKernel::new()));
            }
            weights.push(1.0 / (*num_kernels as f64));
        }

        let kernel = WeightedSumKernel::new(kernels, weights).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_kernels),
            num_kernels,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.compute(&data[0], &data[1]).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark weighted sum kernel matrix
fn bench_weighted_sum_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_sum_matrix");

    for size in [10, 25, 50].iter() {
        let data = generate_dataset(50, *size);

        let k1 = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let k2 = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;
        let weights = vec![0.7, 0.3];
        let kernel = WeightedSumKernel::new(vec![k1, k2], weights).unwrap();

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_matrix(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark product kernel with 2 components
fn bench_product_kernel_two(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_kernel_two");

    let data = generate_dataset(50, 2);
    let k1 = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
    let k2 = Box::new(CosineKernel::new()) as Box<dyn Kernel>;
    let kernel = ProductKernel::new(vec![k1, k2]).unwrap();

    group.bench_function("compute", |b| {
        b.iter(|| {
            black_box(kernel.compute(&data[0], &data[1]).unwrap());
        });
    });

    group.finish();
}

/// Benchmark product kernel with multiple components
fn bench_product_kernel_many(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_kernel_many");

    let data = generate_dataset(50, 2);

    for num_kernels in [2, 4, 8].iter() {
        let mut kernels: Vec<Box<dyn Kernel>> = Vec::new();

        for i in 0..*num_kernels {
            if i % 3 == 0 {
                kernels.push(Box::new(LinearKernel::new()));
            } else if i % 3 == 1 {
                kernels.push(Box::new(CosineKernel::new()));
            } else {
                kernels.push(Box::new(PolynomialKernel::new(2, 1.0).unwrap()));
            }
        }

        let kernel = ProductKernel::new(kernels).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_kernels),
            num_kernels,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.compute(&data[0], &data[1]).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark product kernel matrix
fn bench_product_kernel_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_kernel_matrix");

    for size in [10, 25, 50].iter() {
        let data = generate_dataset(50, *size);

        let k1 = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let k2 = Box::new(CosineKernel::new()) as Box<dyn Kernel>;
        let kernel = ProductKernel::new(vec![k1, k2]).unwrap();

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_matrix(&data).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark kernel alignment computation
fn bench_kernel_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_alignment");

    for size in [10, 25, 50, 100].iter() {
        let k1 = generate_kernel_matrix(*size);
        let k2 = generate_kernel_matrix(*size);

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(KernelAlignment::compute_alignment(&k1, &k2).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark kernel alignment with different matrix sizes
fn bench_kernel_alignment_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_alignment_sizes");

    for size in [10, 25, 50, 100].iter() {
        let k1 = generate_kernel_matrix(*size);
        let k2 = generate_kernel_matrix(*size);

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(KernelAlignment::compute_alignment(&k1, &k2).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark composition vs individual kernels
fn bench_composition_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("composition_overhead");

    let data = generate_dataset(50, 50);

    // Individual kernels
    group.bench_function("linear_only", |b| {
        let kernel = LinearKernel::new();
        b.iter(|| {
            black_box(kernel.compute_matrix(&data).unwrap());
        });
    });

    group.bench_function("rbf_only", |b| {
        let kernel = RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap();
        b.iter(|| {
            black_box(kernel.compute_matrix(&data).unwrap());
        });
    });

    // Weighted sum (70% linear + 30% RBF)
    group.bench_function("weighted_sum", |b| {
        let k1 = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let k2 = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;
        let weights = vec![0.7, 0.3];
        let kernel = WeightedSumKernel::new(vec![k1, k2], weights).unwrap();

        b.iter(|| {
            black_box(kernel.compute_matrix(&data).unwrap());
        });
    });

    // Product kernel (linear * RBF)
    group.bench_function("product", |b| {
        let k1 = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let k2 = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;
        let kernel = ProductKernel::new(vec![k1, k2]).unwrap();

        b.iter(|| {
            black_box(kernel.compute_matrix(&data).unwrap());
        });
    });

    group.finish();
}

/// Benchmark complex composite kernels
fn bench_complex_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_composition");

    let data = generate_dataset(50, 25);

    // Nested composition: (0.5*Linear + 0.3*RBF) * Cosine
    group.bench_function("nested_composition", |b| {
        let k1 = Box::new(LinearKernel::new()) as Box<dyn Kernel>;
        let k2 = Box::new(RbfKernel::new(RbfKernelConfig::new(0.5)).unwrap()) as Box<dyn Kernel>;
        let weighted = Box::new(WeightedSumKernel::new(vec![k1, k2], vec![0.5, 0.3]).unwrap())
            as Box<dyn Kernel>;
        let k3 = Box::new(CosineKernel::new()) as Box<dyn Kernel>;
        let kernel = ProductKernel::new(vec![weighted, k3]).unwrap();

        b.iter(|| {
            black_box(kernel.compute_matrix(&data).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_weighted_sum_two,
    bench_weighted_sum_many,
    bench_weighted_sum_matrix,
    bench_product_kernel_two,
    bench_product_kernel_many,
    bench_product_kernel_matrix,
    bench_kernel_alignment,
    bench_kernel_alignment_sizes,
    bench_composition_overhead,
    bench_complex_composition,
);

criterion_main!(benches);
