//! End-to-End TensorLogic Benchmarks
//!
//! These benchmarks measure the complete pipeline from TLExpr compilation
//! to execution, simulating realistic usage scenarios.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensorlogic_compiler::{compile_to_einsum, compile_to_einsum_with_context, CompilerContext};
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_scirs_backend::Scirs2Exec;

fn create_test_tensor(shape: &[usize]) -> scirs2_core::ndarray::ArrayD<f64> {
    let size = shape.iter().product();
    let data: Vec<f64> = (0..size)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    scirs2_core::ndarray::ArrayD::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), data).unwrap()
}

/// Benchmark simple predicate evaluation.
fn bench_simple_predicate(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_simple_predicate");

    for domain_size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(domain_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                // Create expression: P(x)
                let expr = TLExpr::pred("P", vec![Term::var("x")]);
                let graph = compile_to_einsum(&expr).unwrap();

                let mut executor = Scirs2Exec::new();
                let tensor = create_test_tensor(&[size]);
                executor.add_tensor("P[a]", tensor);

                b.iter(|| {
                    let result = executor.forward(&graph).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark AND operation.
fn bench_and_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_and");

    for domain_size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(domain_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                // P(x) AND Q(x)
                let expr = TLExpr::and(
                    TLExpr::pred("P", vec![Term::var("x")]),
                    TLExpr::pred("Q", vec![Term::var("x")]),
                );
                let graph = compile_to_einsum(&expr).unwrap();

                let mut executor = Scirs2Exec::new();
                executor.add_tensor("P[a]", create_test_tensor(&[size]));
                executor.add_tensor("Q[a]", create_test_tensor(&[size]));

                b.iter(|| {
                    let result = executor.forward(&graph).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark EXISTS quantifier.
fn bench_exists_quantifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_exists");

    for domain_size in [10, 50, 100] {
        group.throughput(Throughput::Elements((domain_size * domain_size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                // EXISTS y: knows(x, y)
                let pred = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
                let expr = TLExpr::exists("y", "Person", pred);

                // Create context with domain
                let mut ctx = CompilerContext::new();
                ctx.add_domain("Person".to_string(), size);

                let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

                let mut executor = Scirs2Exec::new();
                let tensor = create_test_tensor(&[size, size]);
                executor.add_tensor("knows[ab]", tensor);

                b.iter(|| {
                    let result = executor.forward(&graph).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full training iteration (forward + backward).
fn bench_training_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_training");

    for domain_size in [10, 50, 100] {
        group.throughput(Throughput::Elements((domain_size * domain_size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                // P(x) AND Q(x) - simpler for training benchmark
                let expr = TLExpr::and(
                    TLExpr::pred("P", vec![Term::var("x")]),
                    TLExpr::pred("Q", vec![Term::var("x")]),
                );
                let graph = compile_to_einsum(&expr).unwrap();

                let p_tensor = create_test_tensor(&[size]);
                let q_tensor = create_test_tensor(&[size]);

                b.iter(|| {
                    // Create fresh executor for each iteration
                    let mut executor = Scirs2Exec::new();
                    executor.add_tensor("P[a]", p_tensor.clone());
                    executor.add_tensor("Q[a]", q_tensor.clone());

                    // Forward pass
                    let result = executor.forward(&graph).unwrap();

                    // Backward pass
                    let grad_out = Scirs2Exec::ones(result.shape().to_vec());
                    let _input_grads = executor.backward(&graph, &grad_out).unwrap();

                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch processing.
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_batch");

    for batch_size in [1, 10, 100] {
        group.throughput(Throughput::Elements((batch_size * 100) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let domain_size = 100;
                let expr = TLExpr::pred("P", vec![Term::var("x")]);
                let graph = compile_to_einsum(&expr).unwrap();

                b.iter(|| {
                    for _ in 0..batch_size {
                        let mut executor = Scirs2Exec::new();
                        executor.add_tensor("P[a]", create_test_tensor(&[domain_size]));
                        let result = executor.forward(&graph).unwrap();
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark graph scaling with operation count.
fn bench_graph_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_scaling");

    let domain_size = 50;

    for num_ops in [1, 3, 5, 10] {
        group.throughput(Throughput::Elements((domain_size * num_ops) as u64));

        group.bench_with_input(
            BenchmarkId::new("operations", num_ops),
            &num_ops,
            |b, &num_ops| {
                // Build expression with multiple ANDs
                let mut expr = TLExpr::pred("P0", vec![Term::var("x")]);
                for i in 1..num_ops {
                    expr = TLExpr::and(expr, TLExpr::pred(format!("P{}", i), vec![Term::var("x")]));
                }
                let graph = compile_to_einsum(&expr).unwrap();

                let mut executor = Scirs2Exec::new();
                for i in 0..num_ops {
                    executor.add_tensor(format!("P{}[a]", i), create_test_tensor(&[domain_size]));
                }

                b.iter(|| {
                    let result = executor.forward(&graph).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark OR operation.
fn bench_or_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_or");

    for domain_size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(domain_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                // P(x) OR Q(x)
                let expr = TLExpr::or(
                    TLExpr::pred("P", vec![Term::var("x")]),
                    TLExpr::pred("Q", vec![Term::var("x")]),
                );
                let graph = compile_to_einsum(&expr).unwrap();

                let mut executor = Scirs2Exec::new();
                executor.add_tensor("P[a]", create_test_tensor(&[size]));
                executor.add_tensor("Q[a]", create_test_tensor(&[size]));

                b.iter(|| {
                    let result = executor.forward(&graph).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark NOT operation.
fn bench_not_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_not");

    for domain_size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(domain_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                // NOT P(x)
                let expr = TLExpr::negate(TLExpr::pred("P", vec![Term::var("x")]));
                let graph = compile_to_einsum(&expr).unwrap();

                let mut executor = Scirs2Exec::new();
                executor.add_tensor("P[a]", create_test_tensor(&[size]));

                b.iter(|| {
                    let result = executor.forward(&graph).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FORALL quantifier.
fn bench_forall_quantifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_forall");

    for domain_size in [10, 50, 100] {
        group.throughput(Throughput::Elements((domain_size * domain_size) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                // FORALL y: knows(x, y)
                let pred = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
                let expr = TLExpr::forall("y", "Person", pred);

                // Create context with domain
                let mut ctx = CompilerContext::new();
                ctx.add_domain("Person".to_string(), size);

                let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

                let mut executor = Scirs2Exec::new();
                let tensor = create_test_tensor(&[size, size]);
                executor.add_tensor("knows[ab]", tensor);

                b.iter(|| {
                    let result = executor.forward(&graph).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark implication operation.
fn bench_implication(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_implies");

    for domain_size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(domain_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                // P(x) => Q(x)
                let expr = TLExpr::imply(
                    TLExpr::pred("P", vec![Term::var("x")]),
                    TLExpr::pred("Q", vec![Term::var("x")]),
                );
                let graph = compile_to_einsum(&expr).unwrap();

                let mut executor = Scirs2Exec::new();
                executor.add_tensor("P[a]", create_test_tensor(&[size]));
                executor.add_tensor("Q[a]", create_test_tensor(&[size]));

                b.iter(|| {
                    let result = executor.forward(&graph).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark complex nested operations.
fn bench_complex_nested(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_nested");

    for domain_size in [10, 50, 100] {
        group.throughput(Throughput::Elements(domain_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(domain_size),
            &domain_size,
            |b, &size| {
                // (P(x) AND Q(x)) OR (R(x) AND S(x))
                let left = TLExpr::and(
                    TLExpr::pred("P", vec![Term::var("x")]),
                    TLExpr::pred("Q", vec![Term::var("x")]),
                );
                let right = TLExpr::and(
                    TLExpr::pred("R", vec![Term::var("x")]),
                    TLExpr::pred("S", vec![Term::var("x")]),
                );
                let expr = TLExpr::or(left, right);
                let graph = compile_to_einsum(&expr).unwrap();

                let mut executor = Scirs2Exec::new();
                executor.add_tensor("P[a]", create_test_tensor(&[size]));
                executor.add_tensor("Q[a]", create_test_tensor(&[size]));
                executor.add_tensor("R[a]", create_test_tensor(&[size]));
                executor.add_tensor("S[a]", create_test_tensor(&[size]));

                b.iter(|| {
                    let result = executor.forward(&graph).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simple_predicate,
    bench_and_operation,
    bench_or_operation,
    bench_not_operation,
    bench_exists_quantifier,
    bench_forall_quantifier,
    bench_implication,
    bench_complex_nested,
    bench_training_iteration,
    bench_batch_processing,
    bench_graph_scaling
);
criterion_main!(benches);
