//! SIMD vs Non-SIMD Performance Benchmarks
//!
//! This benchmark suite compares performance with and without SIMD acceleration.
//! Run with: cargo bench --bench simd_comparison --features simd
//!
//! **Note**: This benchmark requires the `integration-tests` feature to avoid
//! circular dev-dependencies with tensorlogic-compiler.

#[cfg(feature = "integration-tests")]
use std::hint::black_box;

#[cfg(feature = "integration-tests")]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "integration-tests")]
use scirs2_core::ndarray::ArrayD;
#[cfg(feature = "integration-tests")]
use tensorlogic_compiler::compile_to_einsum;
#[cfg(feature = "integration-tests")]
use tensorlogic_infer::TlAutodiff;
#[cfg(feature = "integration-tests")]
use tensorlogic_ir::{TLExpr, Term};
#[cfg(feature = "integration-tests")]
use tensorlogic_scirs_backend::Scirs2Exec;

#[cfg(feature = "integration-tests")]
fn create_test_tensor(shape: &[usize]) -> ArrayD<f64> {
    ArrayD::from_shape_vec(
        shape.to_vec(),
        (0..shape.iter().product())
            .map(|i| (i as f64) * 0.01)
            .collect(),
    )
    .unwrap()
}

/// Benchmark element-wise operations
#[cfg(feature = "integration-tests")]
fn bench_elemwise_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("elemwise_operations");

    for size in [100, 1000, 10000, 100000] {
        group.bench_with_input(BenchmarkId::new("add", size), &size, |bench, &size| {
            let a = TLExpr::pred("a", vec![Term::var("i")]);
            let b = TLExpr::pred("b", vec![Term::var("i")]);
            let expr = TLExpr::or(a, b); // OR uses addition
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let a_data = create_test_tensor(&[size]);
            let b_data = create_test_tensor(&[size]);
            executor.add_tensor(graph.tensors[0].clone(), a_data);
            executor.add_tensor(graph.tensors[1].clone(), b_data);

            bench.iter(|| black_box(executor.forward(&graph).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("mul", size), &size, |bench, &size| {
            let a = TLExpr::pred("a", vec![Term::var("i")]);
            let b = TLExpr::pred("b", vec![Term::var("i")]);
            let expr = TLExpr::and(a, b); // AND uses multiplication
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let a_data = create_test_tensor(&[size]);
            let b_data = create_test_tensor(&[size]);
            executor.add_tensor(graph.tensors[0].clone(), a_data);
            executor.add_tensor(graph.tensors[1].clone(), b_data);

            bench.iter(|| black_box(executor.forward(&graph).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark reduction operations
#[cfg(feature = "integration-tests")]
fn bench_reduction_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_operations");

    for size in [100, 1000, 10000, 100000] {
        group.bench_with_input(BenchmarkId::new("sum", size), &size, |bench, &size| {
            let pred = TLExpr::pred("p", vec![Term::var("i"), Term::var("j")]);
            let expr = TLExpr::exists("j", "domain", pred); // EXISTS uses sum
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let data = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), data);

            bench.iter(|| black_box(executor.forward(&graph).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark matrix operations
#[cfg(feature = "integration-tests")]
fn bench_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");

    for size in [10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::new("hadamard", size), &size, |bench, &size| {
            let a = TLExpr::pred("a", vec![Term::var("i"), Term::var("j")]);
            let b = TLExpr::pred("b", vec![Term::var("i"), Term::var("j")]);
            let expr = TLExpr::and(a, b);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let a_data = create_test_tensor(&[size, size]);
            let b_data = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), a_data);
            executor.add_tensor(graph.tensors[1].clone(), b_data);

            bench.iter(|| black_box(executor.forward(&graph).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark logical operations (fuzzy AND)
#[cfg(feature = "integration-tests")]
fn bench_logical_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("logical_operations");

    for size in [100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("fuzzy_and_product", size),
            &size,
            |bench, &size| {
                let a = TLExpr::pred("a", vec![Term::var("i")]);
                let b = TLExpr::pred("b", vec![Term::var("i")]);
                let expr = TLExpr::and(a, b);
                let graph = compile_to_einsum(&expr).unwrap();

                let mut executor = Scirs2Exec::new();
                let a_data = create_test_tensor(&[size]);
                let b_data = create_test_tensor(&[size]);
                executor.add_tensor(graph.tensors[0].clone(), a_data);
                executor.add_tensor(graph.tensors[1].clone(), b_data);

                bench.iter(|| black_box(executor.forward(&graph).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark complex operations
#[cfg(feature = "integration-tests")]
fn bench_complex_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_operations");

    for size in [100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("nested_and", size),
            &size,
            |bench, &size| {
                // P0 AND P1 AND P2 AND P3 AND P4
                let mut expr = TLExpr::pred("P0", vec![Term::var("i")]);
                for i in 1..5 {
                    let next = TLExpr::pred(format!("P{}", i), vec![Term::var("i")]);
                    expr = TLExpr::and(expr, next);
                }
                let graph = compile_to_einsum(&expr).unwrap();

                let mut executor = Scirs2Exec::new();
                for i in 0..5 {
                    let data = create_test_tensor(&[size]);
                    executor.add_tensor(graph.tensors[i].clone(), data);
                }

                bench.iter(|| black_box(executor.forward(&graph).unwrap()));
            },
        );
    }

    group.finish();
}

#[cfg(feature = "integration-tests")]
criterion_group!(
    benches,
    bench_elemwise_operations,
    bench_reduction_operations,
    bench_matrix_operations,
    bench_logical_operations,
    bench_complex_operations
);
#[cfg(feature = "integration-tests")]
criterion_main!(benches);

// Fallback main when integration-tests feature is not enabled
#[cfg(not(feature = "integration-tests"))]
fn main() {
    eprintln!("This benchmark requires the 'integration-tests' feature to be enabled.");
    eprintln!("Run with: cargo bench --features integration-tests");
}
