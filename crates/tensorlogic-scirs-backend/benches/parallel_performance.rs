//! Parallel vs Sequential execution performance benchmarks.
//!
//! Compares the performance of parallel vs sequential execution
//! for graphs with varying levels of parallelism.
//!
//! **Note**: This benchmark requires the `integration-tests` feature to avoid
//! circular dev-dependencies with tensorlogic-compiler.

#![cfg(feature = "integration-tests")]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::ArrayD;
use tensorlogic_compiler::{compile_to_einsum, compile_to_einsum_with_context, CompilerContext};
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{TLExpr, Term};
#[cfg(feature = "parallel")]
use tensorlogic_scirs_backend::ParallelScirs2Exec;
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

/// Benchmark with high parallelism: multiple independent operations
fn bench_high_parallelism(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_parallelism");

    for size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64 * *size as u64));

        // Create an expression with many independent predicates combined
        let p1 = TLExpr::pred("p1", vec![Term::var("i"), Term::var("j")]);
        let p2 = TLExpr::pred("p2", vec![Term::var("i"), Term::var("j")]);
        let p3 = TLExpr::pred("p3", vec![Term::var("i"), Term::var("j")]);
        let p4 = TLExpr::pred("p4", vec![Term::var("i"), Term::var("j")]);

        // AND(p1, p2, p3, p4) creates multiple levels
        let and_12 = TLExpr::and(p1, p2);
        let and_34 = TLExpr::and(p3, p4);
        let expr = TLExpr::and(and_12, and_34);

        let graph = compile_to_einsum(&expr).unwrap();

        // Sequential benchmark
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |b, &size| {
            let mut executor = Scirs2Exec::new();
            for i in 0..4 {
                let tensor = create_test_tensor(&[size, size]);
                executor.add_tensor(format!("p{}", i + 1), tensor);
            }

            b.iter(|| {
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });

        // Parallel benchmark
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |b, &size| {
            let mut executor = ParallelScirs2Exec::new();
            for i in 0..4 {
                let tensor = create_test_tensor(&[size, size]);
                executor.add_tensor(format!("p{}", i + 1), tensor);
            }

            b.iter(|| {
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark with low parallelism: mostly sequential operations
fn bench_low_parallelism(c: &mut Criterion) {
    let mut group = c.benchmark_group("low_parallelism");

    for size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64 * *size as u64));

        // Create a sequential chain: p1 -> exists -> negate
        let pred = TLExpr::pred("p", vec![Term::var("i"), Term::var("j")]);
        let exists_expr = TLExpr::exists("j", "domain", pred);
        let expr = TLExpr::negate(exists_expr);

        // Create context with domain
        let mut ctx = CompilerContext::new();
        ctx.add_domain("domain", *size);
        let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

        // Sequential benchmark
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |b, &size| {
            let mut executor = Scirs2Exec::new();
            let tensor = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), tensor);

            b.iter(|| {
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });

        // Parallel benchmark (should be similar to sequential)
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |b, &size| {
            let mut executor = ParallelScirs2Exec::new();
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

/// Benchmark with medium parallelism: balanced workload
fn bench_medium_parallelism(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_parallelism");

    for size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64 * *size as u64));

        // Create an expression with some independent operations
        let p1 = TLExpr::pred("p1", vec![Term::var("i"), Term::var("j")]);
        let p2 = TLExpr::pred("p2", vec![Term::var("i"), Term::var("j")]);
        let expr = TLExpr::or(p1, p2);

        let graph = compile_to_einsum(&expr).unwrap();

        // Sequential benchmark
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |b, &size| {
            let mut executor = Scirs2Exec::new();
            let tensor1 = create_test_tensor(&[size, size]);
            let tensor2 = create_test_tensor(&[size, size]);
            executor.add_tensor("p1".to_string(), tensor1);
            executor.add_tensor("p2".to_string(), tensor2);

            b.iter(|| {
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });

        // Parallel benchmark
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |b, &size| {
            let mut executor = ParallelScirs2Exec::new();
            let tensor1 = create_test_tensor(&[size, size]);
            let tensor2 = create_test_tensor(&[size, size]);
            executor.add_tensor("p1".to_string(), tensor1);
            executor.add_tensor("p2".to_string(), tensor2);

            b.iter(|| {
                let result = executor.forward(black_box(&graph)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

/// Benchmark thread scaling: measure performance with different thread counts
#[cfg(feature = "parallel")]
fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");

    let size = 100;
    group.throughput(Throughput::Elements((size * size) as u64));

    // Create an expression with high parallelism
    let p1 = TLExpr::pred("p1", vec![Term::var("i"), Term::var("j")]);
    let p2 = TLExpr::pred("p2", vec![Term::var("i"), Term::var("j")]);
    let p3 = TLExpr::pred("p3", vec![Term::var("i"), Term::var("j")]);
    let p4 = TLExpr::pred("p4", vec![Term::var("i"), Term::var("j")]);

    let and_12 = TLExpr::and(p1, p2);
    let and_34 = TLExpr::and(p3, p4);
    let expr = TLExpr::and(and_12, and_34);

    let graph = compile_to_einsum(&expr).unwrap();

    for num_threads in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &num_threads| {
                let mut executor = ParallelScirs2Exec::new();
                executor.set_num_threads(num_threads);

                for i in 0..4 {
                    let tensor = create_test_tensor(&[size, size]);
                    executor.add_tensor(format!("p{}", i + 1), tensor);
                }

                b.iter(|| {
                    let result = executor.forward(black_box(&graph)).unwrap();
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark backward pass: parallel vs sequential
fn bench_backward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_pass");

    for size in [50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*size as u64 * *size as u64));

        let p1 = TLExpr::pred("p1", vec![Term::var("i"), Term::var("j")]);
        let p2 = TLExpr::pred("p2", vec![Term::var("i"), Term::var("j")]);
        let expr = TLExpr::and(p1, p2);

        let graph = compile_to_einsum(&expr).unwrap();

        // Sequential benchmark
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |b, &size| {
            let mut executor = Scirs2Exec::new();
            let tensor1 = create_test_tensor(&[size, size]);
            let tensor2 = create_test_tensor(&[size, size]);
            executor.add_tensor("p1".to_string(), tensor1);
            executor.add_tensor("p2".to_string(), tensor2);

            // Forward pass first
            executor.forward(&graph).unwrap();

            b.iter(|| {
                let loss_grad = create_test_tensor(&[size, size]);
                let result = executor
                    .backward(black_box(&graph), black_box(&loss_grad))
                    .unwrap();
                black_box(result);
            });
        });

        // Parallel benchmark
        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |b, &size| {
            let mut executor = ParallelScirs2Exec::new();
            let tensor1 = create_test_tensor(&[size, size]);
            let tensor2 = create_test_tensor(&[size, size]);
            executor.add_tensor("p1".to_string(), tensor1);
            executor.add_tensor("p2".to_string(), tensor2);

            // Forward pass first
            executor.forward(&graph).unwrap();

            b.iter(|| {
                let loss_grad = create_test_tensor(&[size, size]);
                let result = executor
                    .backward(black_box(&graph), black_box(&loss_grad))
                    .unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_high_parallelism,
    bench_low_parallelism,
    bench_medium_parallelism,
    bench_thread_scaling,
    bench_backward_pass
);

#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    bench_high_parallelism,
    bench_low_parallelism,
    bench_medium_parallelism,
    bench_backward_pass
);

criterion_main!(benches);
