//! Forward pass throughput benchmarks.
//!
//! Measures the performance of forward pass execution for various graph sizes
//! and operation types.
//!
//! **Note**: This benchmark requires the `integration-tests` feature to avoid
//! circular dev-dependencies with tensorlogic-compiler.

#![cfg(feature = "integration-tests")]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::ArrayD;
use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_infer::{TlAutodiff, TlExecutor};
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

fn bench_simple_predicate(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_predicate");

    for size in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64 * *size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Create a simple predicate expression
            let expr = TLExpr::pred("x", vec![Term::var("i"), Term::var("j")]);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let tensor = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), tensor);

            b.iter(|| {
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_and_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("and_operation");

    for size in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64 * *size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let x = TLExpr::pred("x", vec![Term::var("i"), Term::var("j")]);
            let y = TLExpr::pred("y", vec![Term::var("i"), Term::var("j")]);
            let expr = TLExpr::and(x, y);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let tensor1 = create_test_tensor(&[size, size]);
            let tensor2 = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), tensor1);
            executor.add_tensor(graph.tensors[1].clone(), tensor2);

            b.iter(|| {
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_exists_quantifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("exists_quantifier");

    for size in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64 * *size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let pred = TLExpr::pred("p", vec![Term::var("i"), Term::var("j")]);
            let expr = TLExpr::exists("j", "domain", pred);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let tensor = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), tensor);

            b.iter(|| {
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_complex_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_expression");

    for size in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*size as u64 * *size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // (x AND y) OR (NOT z)
            let x = TLExpr::pred("x", vec![Term::var("i"), Term::var("j")]);
            let y = TLExpr::pred("y", vec![Term::var("i"), Term::var("j")]);
            let z = TLExpr::pred("z", vec![Term::var("i"), Term::var("j")]);
            let and_expr = TLExpr::and(x, y);
            let not_z = TLExpr::negate(z);
            let expr = TLExpr::or(and_expr, not_z);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let tensor1 = create_test_tensor(&[size, size]);
            let tensor2 = create_test_tensor(&[size, size]);
            let tensor3 = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), tensor1);
            executor.add_tensor(graph.tensors[1].clone(), tensor2);
            executor.add_tensor(graph.tensors[2].clone(), tensor3);

            b.iter(|| {
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_einsum_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_matmul");

    for size in [10, 32, 64, 128].iter() {
        group.throughput(Throughput::Elements(
            *size as u64 * *size as u64 * *size as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            let mut executor = Scirs2Exec::new();
            let tensor_a = create_test_tensor(&[size, size]);
            let tensor_b = create_test_tensor(&[size, size]);

            bencher.iter(|| {
                let result = executor
                    .einsum("ij,jk->ik", &[tensor_a.clone(), tensor_b.clone()])
                    .unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_memory_pooling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pooling");

    for size in [50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("with_pool", size), size, |b, &size| {
            let expr = TLExpr::pred("x", vec![Term::var("i"), Term::var("j")]);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::with_memory_pool();
            let tensor = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), tensor.clone());

            b.iter(|| {
                executor.add_tensor(graph.tensors[0].clone(), tensor.clone());
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("without_pool", size), size, |b, &size| {
            let expr = TLExpr::pred("x", vec![Term::var("i"), Term::var("j")]);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let tensor = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), tensor.clone());

            b.iter(|| {
                executor.add_tensor(graph.tensors[0].clone(), tensor.clone());
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_simple_predicate,
    bench_and_operation,
    bench_exists_quantifier,
    bench_complex_expression,
    bench_einsum_matmul,
    bench_memory_pooling,
);
criterion_main!(benches);
