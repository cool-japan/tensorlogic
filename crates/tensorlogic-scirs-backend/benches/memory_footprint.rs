//! Memory Footprint Benchmarks
//!
//! Measures memory allocation and usage patterns for different graph sizes and operations.
//! Run with: cargo bench --bench memory_footprint
//!
//! **Note**: This benchmark requires the `integration-tests` feature to avoid
//! circular dev-dependencies with tensorlogic-compiler.

#![cfg(feature = "integration-tests")]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::ArrayD;
use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_scirs_backend::Scirs2Exec;

fn create_test_tensor(shape: &[usize]) -> ArrayD<f64> {
    ArrayD::from_shape_vec(
        shape.to_vec(),
        (0..shape.iter().product())
            .map(|i| (i as f64) * 0.01)
            .collect(),
    )
    .unwrap()
}

/// Benchmark memory usage for simple expressions
fn bench_memory_simple_expressions(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_simple_expressions");

    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));

        // Single predicate
        group.bench_with_input(
            BenchmarkId::new("single_predicate", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let expr = TLExpr::pred("P", vec![Term::var("i")]);
                    let graph = compile_to_einsum(&expr).unwrap();

                    let mut executor = Scirs2Exec::new();
                    let data = create_test_tensor(&[size]);
                    executor.add_tensor(graph.tensors[0].clone(), data);

                    black_box(executor.forward(&graph).unwrap())
                });
            },
        );

        // AND operation
        group.bench_with_input(
            BenchmarkId::new("and_operation", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let p = TLExpr::pred("P", vec![Term::var("i")]);
                    let q = TLExpr::pred("Q", vec![Term::var("i")]);
                    let expr = TLExpr::and(p, q);
                    let graph = compile_to_einsum(&expr).unwrap();

                    let mut executor = Scirs2Exec::new();
                    let p_data = create_test_tensor(&[size]);
                    let q_data = create_test_tensor(&[size]);
                    executor.add_tensor(graph.tensors[0].clone(), p_data);
                    executor.add_tensor(graph.tensors[1].clone(), q_data);

                    black_box(executor.forward(&graph).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage for matrix operations
fn bench_memory_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_matrix_operations");

    for size in [10, 50, 100] {
        group.throughput(Throughput::Elements((size * size) as u64));

        // Two predicates with 2D tensors (simulates matrix-like operations)
        group.bench_with_input(
            BenchmarkId::new("matrix_like_operations", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let a = TLExpr::pred("A", vec![Term::var("i"), Term::var("j")]);
                    let b = TLExpr::pred("B", vec![Term::var("i"), Term::var("j")]);
                    let expr = TLExpr::and(a, b);
                    let graph = compile_to_einsum(&expr).unwrap();

                    let mut executor = Scirs2Exec::new();
                    let a_data = create_test_tensor(&[size, size]);
                    let b_data = create_test_tensor(&[size, size]);
                    executor.add_tensor(graph.tensors[0].clone(), a_data);
                    executor.add_tensor(graph.tensors[1].clone(), b_data);

                    black_box(executor.forward(&graph).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage for complex expressions
fn bench_memory_complex_expressions(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_complex_expressions");

    for size in [100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("nested_operations", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    // (A AND B) OR (C AND D)
                    let a = TLExpr::pred("A", vec![Term::var("i")]);
                    let b = TLExpr::pred("B", vec![Term::var("i")]);
                    let c = TLExpr::pred("C", vec![Term::var("i")]);
                    let d = TLExpr::pred("D", vec![Term::var("i")]);
                    let ab = TLExpr::and(a, b);
                    let cd = TLExpr::and(c, d);
                    let expr = TLExpr::or(ab, cd);
                    let graph = compile_to_einsum(&expr).unwrap();

                    let mut executor = Scirs2Exec::new();
                    for i in 0..4 {
                        let data = create_test_tensor(&[size]);
                        executor.add_tensor(graph.tensors[i].clone(), data);
                    }

                    black_box(executor.forward(&graph).unwrap())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_simple_expressions,
    bench_memory_matrix_operations,
    bench_memory_complex_expressions
);
criterion_main!(benches);
