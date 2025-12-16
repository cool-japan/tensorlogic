//! Gradient Stability Benchmarks
//!
//! Measures numerical stability and accuracy of gradient computations.
//! Run with: cargo bench --bench gradient_stability
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
            .map(|i| (i as f64) * 0.01 + 0.1)
            .collect(),
    )
    .unwrap()
}

#[cfg(feature = "integration-tests")]
fn create_grad_tensor(shape: &[usize]) -> ArrayD<f64> {
    ArrayD::ones(shape.to_vec())
}

/// Benchmark gradient computation for simple operations
#[cfg(feature = "integration-tests")]
fn bench_gradient_simple_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_simple_ops");

    for size in [100, 1000, 10000] {
        // AND operation gradients
        group.bench_with_input(
            BenchmarkId::new("and_gradient", size),
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

                    // Forward pass
                    let output = executor.forward(&graph).unwrap();

                    // Backward pass
                    let grad_output = create_grad_tensor(output.shape());
                    black_box(executor.backward(&graph, &grad_output).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark gradient computation for nested operations
#[cfg(feature = "integration-tests")]
fn bench_gradient_nested_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_nested_ops");

    for depth in [2, 3, 5] {
        group.bench_with_input(
            BenchmarkId::new("nested_and", depth),
            &depth,
            |bench, &depth| {
                let size = 1000;
                bench.iter(|| {
                    // Build nested AND expression: P AND Q AND R AND...
                    let mut expr = TLExpr::pred("P0", vec![Term::var("i")]);
                    for i in 1..depth {
                        let next = TLExpr::pred(format!("P{}", i), vec![Term::var("i")]);
                        expr = TLExpr::and(expr, next);
                    }
                    let graph = compile_to_einsum(&expr).unwrap();

                    let mut executor = Scirs2Exec::new();
                    for i in 0..depth {
                        let data = create_test_tensor(&[size]);
                        executor.add_tensor(graph.tensors[i].clone(), data);
                    }

                    let output = executor.forward(&graph).unwrap();
                    let grad_output = create_grad_tensor(output.shape());
                    black_box(executor.backward(&graph, &grad_output).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark gradient computation for matrix-like operations
#[cfg(feature = "integration-tests")]
fn bench_gradient_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_matrix_ops");

    for size in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("2d_and_gradient", size),
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

                    let output = executor.forward(&graph).unwrap();
                    let grad_output = create_grad_tensor(output.shape());
                    black_box(executor.backward(&graph, &grad_output).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark gradient computation for quantifiers
#[cfg(feature = "integration-tests")]
fn bench_gradient_quantifiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_quantifiers");

    for size in [100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("exists_gradient", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let pred = TLExpr::pred("P", vec![Term::var("i"), Term::var("j")]);
                    let expr = TLExpr::exists("j", "domain", pred);
                    let graph = compile_to_einsum(&expr).unwrap();

                    let mut executor = Scirs2Exec::new();
                    let data = create_test_tensor(&[size, size]);
                    executor.add_tensor(graph.tensors[0].clone(), data);

                    let output = executor.forward(&graph).unwrap();
                    let grad_output = create_grad_tensor(output.shape());
                    black_box(executor.backward(&graph, &grad_output).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark gradient computation for complex expressions
#[cfg(feature = "integration-tests")]
fn bench_gradient_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_complex");

    for size in [100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("complex_expression", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    // (A AND B) OR (NOT C)
                    let a = TLExpr::pred("A", vec![Term::var("i")]);
                    let b = TLExpr::pred("B", vec![Term::var("i")]);
                    let c = TLExpr::pred("C", vec![Term::var("i")]);
                    let ab = TLExpr::and(a, b);
                    let not_c = TLExpr::negate(c);
                    let expr = TLExpr::or(ab, not_c);
                    let graph = compile_to_einsum(&expr).unwrap();

                    let mut executor = Scirs2Exec::new();
                    let a_data = create_test_tensor(&[size]);
                    let b_data = create_test_tensor(&[size]);
                    let c_data = create_test_tensor(&[size]);
                    executor.add_tensor(graph.tensors[0].clone(), a_data);
                    executor.add_tensor(graph.tensors[1].clone(), b_data);
                    executor.add_tensor(graph.tensors[2].clone(), c_data);

                    let output = executor.forward(&graph).unwrap();
                    let grad_output = create_grad_tensor(output.shape());
                    black_box(executor.backward(&graph, &grad_output).unwrap())
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "integration-tests")]
criterion_group!(
    benches,
    bench_gradient_simple_ops,
    bench_gradient_nested_ops,
    bench_gradient_matrix_ops,
    bench_gradient_quantifiers,
    bench_gradient_complex
);

#[cfg(feature = "integration-tests")]
criterion_main!(benches);

// Fallback main when integration-tests feature is not enabled
#[cfg(not(feature = "integration-tests"))]
fn main() {
    eprintln!("This benchmark requires the 'integration-tests' feature to be enabled.");
    eprintln!("Run with: cargo bench --features integration-tests");
}
