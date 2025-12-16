//! Benchmarks for tensor operations in the SciRS2 backend.
//!
//! Run benchmarks with:
//! ```bash
//! cargo bench -p tensorlogic-scirs-backend
//! ```

#![allow(unused_imports)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::ArrayD;
use std::hint::black_box;
use tensorlogic_scirs_backend::Scirs2Exec;

#[allow(dead_code)]
fn create_tensor(shape: &[usize]) -> ArrayD<f64> {
    let size: usize = shape.iter().product();
    ArrayD::from_shape_vec(shape.to_vec(), (0..size).map(|i| i as f64 * 0.01).collect()).unwrap()
}

// NOTE: Disabled - uses einsum_raw which no longer exists
#[cfg(any())]
#[allow(dead_code)]
fn bench_einsum_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_matmul");

    for size in [32, 64, 128, 256, 512] {
        let a = create_tensor(&[size, size]);
        let b = create_tensor(&[size, size]);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| {
                let mut exec = Scirs2Exec::new();
                exec.einsum_raw(black_box("ij,jk->ik"), black_box(&[a.clone(), b.clone()]))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// NOTE: Disabled - uses outdated API
#[cfg(any())]
#[allow(dead_code)]
fn bench_einsum_batch_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_batch_matmul");

    for batch in [1, 4, 8, 16] {
        let a = create_tensor(&[batch, 64, 64]);
        let b = create_tensor(&[batch, 64, 64]);

        group.throughput(Throughput::Elements((batch * 64 * 64) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |bench, _| {
            bench.iter(|| {
                let mut exec = Scirs2Exec::new();
                exec.einsum_raw(
                    black_box("bij,bjk->bik"),
                    black_box(&[a.clone(), b.clone()]),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

// NOTE: Disabled - uses outdated API
#[cfg(any())]
#[allow(dead_code)]
fn bench_einsum_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_transpose");

    for size in [64, 128, 256, 512, 1024] {
        let a = create_tensor(&[size, size]);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| {
                let mut exec = Scirs2Exec::new();
                exec.einsum_raw(black_box("ij->ji"), black_box(&[a.clone()]))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// NOTE: Disabled - uses outdated API
#[cfg(any())]
#[allow(dead_code)]
fn bench_einsum_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_trace");

    for size in [64, 128, 256, 512] {
        let a = create_tensor(&[size, size]);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| {
                let mut exec = Scirs2Exec::new();
                exec.einsum_raw(black_box("ii->"), black_box(&[a.clone()]))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// NOTE: Disabled - uses outdated API
#[cfg(any())]
#[allow(dead_code)]
fn bench_unary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary_ops");

    let size = 10000;
    let tensor = create_tensor(&[size]);

    group.throughput(Throughput::Elements(size as u64));

    for op in ["relu", "sigmoid", "oneminus", "tanh"] {
        group.bench_with_input(BenchmarkId::from_parameter(op), &op, |bench, op| {
            bench.iter(|| {
                let mut exec = Scirs2Exec::new();
                exec.unary_op(black_box(op), black_box(&tensor)).unwrap()
            });
        });
    }

    group.finish();
}

// NOTE: Disabled - uses outdated API
#[cfg(any())]
#[allow(dead_code)]
fn bench_binary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_ops");

    let size = 10000;
    let a = create_tensor(&[size]);
    let b = create_tensor(&[size]);

    group.throughput(Throughput::Elements(size as u64));

    for op in ["add", "subtract", "multiply", "divide"] {
        group.bench_with_input(BenchmarkId::from_parameter(op), &op, |bench, op| {
            bench.iter(|| {
                let mut exec = Scirs2Exec::new();
                exec.binary_op(black_box(op), black_box(&a), black_box(&b))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// NOTE: Disabled - uses outdated API
#[cfg(any())]
#[allow(dead_code)]
fn bench_reduce_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_ops");

    for size in [1000, 10000, 100000] {
        let tensor = create_tensor(&[size]);

        group.throughput(Throughput::Elements(size as u64));

        for op in ["sum", "max", "min", "mean"] {
            let id = format!("{}/{}", op, size);
            group.bench_with_input(BenchmarkId::from_parameter(&id), &(), |bench, _| {
                bench.iter(|| {
                    let mut exec = Scirs2Exec::new();
                    exec.reduce_op(black_box(op), black_box(&tensor), black_box(&[0]))
                        .unwrap()
                });
            });
        }
    }

    group.finish();
}

// NOTE: Disabled - uses outdated API
#[cfg(any())]
#[allow(dead_code)]
fn bench_reduce_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_2d");

    let tensor = create_tensor(&[1000, 1000]);
    let elements = 1000 * 1000;

    group.throughput(Throughput::Elements(elements as u64));

    // Reduce along axis 0
    group.bench_function("sum_axis0", |bench| {
        bench.iter(|| {
            let mut exec = Scirs2Exec::new();
            exec.reduce_op(black_box("sum"), black_box(&tensor), black_box(&[0]))
                .unwrap()
        });
    });

    // Reduce along axis 1
    group.bench_function("sum_axis1", |bench| {
        bench.iter(|| {
            let mut exec = Scirs2Exec::new();
            exec.reduce_op(black_box("sum"), black_box(&tensor), black_box(&[1]))
                .unwrap()
        });
    });

    // Reduce all
    group.bench_function("sum_all", |bench| {
        bench.iter(|| {
            let mut exec = Scirs2Exec::new();
            exec.reduce_op(black_box("sum"), black_box(&tensor), black_box(&[0, 1]))
                .unwrap()
        });
    });

    group.finish();
}

// NOTE: Disabled - uses unary_op which no longer exists
#[cfg(any())]
#[allow(dead_code)]
fn bench_memory_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool");

    let size = 10000;
    let tensor = create_tensor(&[size]);

    group.throughput(Throughput::Elements(size as u64));

    // Without pooling
    group.bench_function("no_pool", |bench| {
        bench.iter(|| {
            let mut exec = Scirs2Exec::new();
            for _ in 0..10 {
                let _ = exec
                    .unary_op(black_box("relu"), black_box(&tensor))
                    .unwrap();
            }
        });
    });

    // With pooling
    group.bench_function("with_pool", |bench| {
        bench.iter(|| {
            let mut exec = Scirs2Exec::with_memory_pool();
            for _ in 0..10 {
                let _ = exec
                    .unary_op(black_box("relu"), black_box(&tensor))
                    .unwrap();
            }
        });
    });

    group.finish();
}

// NOTE: Disabled - uses outdated API
#[cfg(any())]
#[allow(dead_code)]
fn bench_logical_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("logical_ops");

    let size = 10000;
    // Values in [0, 1] for logical operations
    let a = ArrayD::from_shape_vec(
        vec![size],
        (0..size).map(|i| i as f64 / size as f64).collect(),
    )
    .unwrap();
    let b = ArrayD::from_shape_vec(
        vec![size],
        (0..size).map(|i| 1.0 - i as f64 / size as f64).collect(),
    )
    .unwrap();

    group.throughput(Throughput::Elements(size as u64));

    for op in ["or_max", "or_prob_sum", "nand", "nor", "xor"] {
        group.bench_with_input(BenchmarkId::from_parameter(op), &op, |bench, op| {
            bench.iter(|| {
                let mut exec = Scirs2Exec::new();
                exec.binary_op(black_box(op), black_box(&a), black_box(&b))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// NOTE: Disabled - uses outdated API
#[cfg(any())]
#[allow(dead_code)]
fn bench_complex_einsum(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_einsum");

    // Attention-like pattern: Q * K^T
    let batch = 4;
    let heads = 8;
    let seq_len = 128;
    let head_dim = 64;

    let q = create_tensor(&[batch, heads, seq_len, head_dim]);
    let k = create_tensor(&[batch, heads, seq_len, head_dim]);

    group.throughput(Throughput::Elements(
        (batch * heads * seq_len * seq_len) as u64,
    ));

    group.bench_function("attention_scores", |bench| {
        bench.iter(|| {
            let mut exec = Scirs2Exec::new();
            exec.einsum_raw(
                black_box("bhqd,bhkd->bhqk"),
                black_box(&[q.clone(), k.clone()]),
            )
            .unwrap()
        });
    });

    // Outer product
    let v1 = create_tensor(&[1000]);
    let v2 = create_tensor(&[1000]);

    group.throughput(Throughput::Elements(1000 * 1000));

    group.bench_function("outer_product", |bench| {
        bench.iter(|| {
            let mut exec = Scirs2Exec::new();
            exec.einsum_raw(black_box("i,j->ij"), black_box(&[v1.clone(), v2.clone()]))
                .unwrap()
        });
    });

    group.finish();
}

// Dummy benchmark to satisfy criterion_group! macro requirement
fn bench_dummy(_c: &mut Criterion) {
    // All real benchmarks are disabled due to outdated API usage
    // This file needs to be updated to use the current executor API
}

criterion_group!(benches, bench_dummy);

criterion_main!(benches);
