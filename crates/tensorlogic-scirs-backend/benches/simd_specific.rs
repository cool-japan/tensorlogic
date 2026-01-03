//! SIMD-Specific Performance Benchmarks
//!
//! These benchmarks measure the performance impact of SIMD optimizations
//! for various tensor operations. They help quantify the speedup gained
//! from vectorization.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use std::hint::black_box;
use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_scirs_backend::Scirs2Exec;

/// Benchmark element-wise operations with SIMD.
fn bench_elementwise_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_elementwise");

    for size in [100, 1000, 10000, 100000] {
        group.throughput(Throughput::Elements(size as u64));

        // ReLU activation
        group.bench_with_input(BenchmarkId::new("relu", size), &size, |b, &size| {
            let data: Vec<f64> = (0..size)
                .map(|i| (i as f64 - size as f64 / 2.0) * 0.01)
                .collect();
            let tensor = Array2::from_shape_vec((size, 1), data).unwrap();

            b.iter(|| {
                let activated: Array2<f64> = tensor.mapv(|x| x.max(0.0));
                black_box(activated);
            });
        });

        // Sigmoid activation
        group.bench_with_input(BenchmarkId::new("sigmoid", size), &size, |b, &size| {
            let data: Vec<f64> = (0..size)
                .map(|i| (i as f64 - size as f64 / 2.0) * 0.01)
                .collect();
            let tensor = Array2::from_shape_vec((size, 1), data).unwrap();

            b.iter(|| {
                let activated: Array2<f64> = tensor.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                black_box(activated);
            });
        });

        // Element-wise multiplication
        group.bench_with_input(BenchmarkId::new("multiply", size), &size, |b, &size| {
            let data1: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
            let data2: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.01).collect();
            let tensor1 = Array2::from_shape_vec((size, 1), data1).unwrap();
            let tensor2 = Array2::from_shape_vec((size, 1), data2).unwrap();

            b.iter(|| {
                let result = &tensor1 * &tensor2;
                black_box(result);
            });
        });

        // Element-wise addition
        group.bench_with_input(BenchmarkId::new("add", size), &size, |b, &size| {
            let data1: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
            let data2: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.01).collect();
            let tensor1 = Array2::from_shape_vec((size, 1), data1).unwrap();
            let tensor2 = Array2::from_shape_vec((size, 1), data2).unwrap();

            b.iter(|| {
                let result = &tensor1 + &tensor2;
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark reductions with SIMD.
fn bench_reductions_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_reductions");

    for size in [100, 1000, 10000, 100000] {
        group.throughput(Throughput::Elements(size as u64));

        // Sum reduction
        group.bench_with_input(BenchmarkId::new("sum", size), &size, |b, &size| {
            let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
            let tensor = Array2::from_shape_vec((size, 1), data).unwrap();

            b.iter(|| {
                let sum = tensor.sum();
                black_box(sum);
            });
        });

        // Max reduction
        group.bench_with_input(BenchmarkId::new("max", size), &size, |b, &size| {
            let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
            let tensor = Array2::from_shape_vec((size, 1), data).unwrap();

            b.iter(|| {
                let max = tensor.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                black_box(max);
            });
        });

        // Mean reduction
        group.bench_with_input(BenchmarkId::new("mean", size), &size, |b, &size| {
            let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.01).collect();
            let tensor = Array2::from_shape_vec((size, 1), data).unwrap();

            b.iter(|| {
                let mean = tensor.mean().unwrap();
                black_box(mean);
            });
        });
    }

    group.finish();
}

/// Benchmark matrix operations with SIMD.
fn bench_matrix_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_matrix");

    for size in [10, 50, 100, 200] {
        group.throughput(Throughput::Elements((size * size) as u64));

        // Matrix multiplication
        group.bench_with_input(BenchmarkId::new("matmul", size), &size, |b, &size| {
            let data1: Vec<f64> = (0..size * size).map(|i| i as f64 * 0.01).collect();
            let data2: Vec<f64> = (0..size * size)
                .map(|i| (size * size - i) as f64 * 0.01)
                .collect();
            let mat1 = Array2::from_shape_vec((size, size), data1).unwrap();
            let mat2 = Array2::from_shape_vec((size, size), data2).unwrap();

            b.iter(|| {
                let result = mat1.dot(&mat2);
                black_box(result);
            });
        });

        // Transpose
        group.bench_with_input(BenchmarkId::new("transpose", size), &size, |b, &size| {
            let data: Vec<f64> = (0..size * size).map(|i| i as f64 * 0.01).collect();
            let mat = Array2::from_shape_vec((size, size), data).unwrap();

            b.iter(|| {
                let result = mat.t();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark vectorization efficiency.
fn bench_vectorization_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vectorization");

    // Benchmark loop unrolling benefit
    for size in [1024, 4096, 16384] {
        group.throughput(Throughput::Elements(size as u64));

        // Scalar loop (baseline)
        group.bench_with_input(BenchmarkId::new("scalar_loop", size), &size, |b, &size| {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

            b.iter(|| {
                let mut result = Vec::with_capacity(size);
                for &x in &data {
                    result.push(x * 2.0 + 1.0);
                }
                black_box(result);
            });
        });

        // Vectorized operation
        group.bench_with_input(BenchmarkId::new("vectorized", size), &size, |b, &size| {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let tensor = Array2::from_shape_vec((size, 1), data).unwrap();

            b.iter(|| {
                let result = tensor.mapv(|x| x * 2.0 + 1.0);
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark cache efficiency with different access patterns.
fn bench_cache_patterns_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_cache");

    let size = 1000;

    // Sequential access (cache-friendly)
    group.bench_function("sequential", |b| {
        let data: Vec<f64> = (0..size * size).map(|i| i as f64).collect();
        let mat = Array2::from_shape_vec((size, size), data).unwrap();

        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..size {
                for j in 0..size {
                    sum += mat[[i, j]];
                }
            }
            black_box(sum);
        });
    });

    // Strided access (less cache-friendly)
    group.bench_function("strided", |b| {
        let data: Vec<f64> = (0..size * size).map(|i| i as f64).collect();
        let mat = Array2::from_shape_vec((size, size), data).unwrap();

        b.iter(|| {
            let mut sum = 0.0;
            for j in 0..size {
                for i in 0..size {
                    sum += mat[[i, j]];
                }
            }
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark TensorLogic operations with SIMD.
fn bench_tensorlogic_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_tensorlogic");

    group.bench_function("and_1000", |b| {
        // Compile a simple AND operation
        let expr = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("x")]),
        );
        let graph = compile_to_einsum(&expr).unwrap();

        let mut executor = Scirs2Exec::new();

        // Create input tensors
        let p_data: Vec<f64> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let q_data: Vec<f64> = (0..1000)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();

        let p_tensor = Scirs2Exec::from_vec(p_data, vec![1000]).unwrap();
        let q_tensor = Scirs2Exec::from_vec(q_data, vec![1000]).unwrap();

        executor.add_tensor("P[a]", p_tensor);
        executor.add_tensor("Q[a]", q_tensor);

        b.iter(|| {
            let result = executor.forward(&graph).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_elementwise_simd,
    bench_reductions_simd,
    bench_matrix_simd,
    bench_vectorization_efficiency,
    bench_cache_patterns_simd,
    bench_tensorlogic_simd
);
criterion_main!(benches);
