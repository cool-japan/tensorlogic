//! Forward Pass Throughput Benchmarks
//!
//! Measures operations per second for various graph complexities.
//! Run with: cargo bench --bench throughput
//!
//! **Note**: This benchmark requires the `integration-tests` feature to avoid
//! circular dev-dependencies with tensorlogic-compiler.

#[cfg(feature = "integration-tests")]
use std::hint::black_box;

#[cfg(feature = "integration-tests")]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
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

/// Benchmark throughput for element-wise operations
#[cfg(feature = "integration-tests")]
fn bench_throughput_elemwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_elemwise");

    for size in [1000, 10000, 100000, 1000000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("and", size), &size, |bench, &size| {
            let p = TLExpr::pred("P", vec![Term::var("i")]);
            let q = TLExpr::pred("Q", vec![Term::var("i")]);
            let expr = TLExpr::and(p, q);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let p_data = create_test_tensor(&[size]);
            let q_data = create_test_tensor(&[size]);
            executor.add_tensor(graph.tensors[0].clone(), p_data);
            executor.add_tensor(graph.tensors[1].clone(), q_data);

            bench.iter(|| black_box(executor.forward(&graph).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("or", size), &size, |bench, &size| {
            let p = TLExpr::pred("P", vec![Term::var("i")]);
            let q = TLExpr::pred("Q", vec![Term::var("i")]);
            let expr = TLExpr::or(p, q);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let p_data = create_test_tensor(&[size]);
            let q_data = create_test_tensor(&[size]);
            executor.add_tensor(graph.tensors[0].clone(), p_data);
            executor.add_tensor(graph.tensors[1].clone(), q_data);

            bench.iter(|| black_box(executor.forward(&graph).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark throughput for matrix operations
#[cfg(feature = "integration-tests")]
fn bench_throughput_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_matrix");

    for size in [32, 64, 128, 256] {
        let ops = (size * size * size) as u64; // Approximate operations
        group.throughput(Throughput::Elements(ops));

        group.bench_with_input(BenchmarkId::new("2d_and", size), &size, |bench, &size| {
            let a = TLExpr::pred("A", vec![Term::var("i"), Term::var("j")]);
            let b = TLExpr::pred("B", vec![Term::var("i"), Term::var("j")]);
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

/// Benchmark throughput for reduction operations
#[cfg(feature = "integration-tests")]
fn bench_throughput_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_reductions");

    for size in [1000, 10000, 100000, 1000000] {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("exists", size), &size, |bench, &size| {
            let pred = TLExpr::pred("P", vec![Term::var("i"), Term::var("j")]);
            let expr = TLExpr::exists("j", "domain", pred);
            let graph = compile_to_einsum(&expr).unwrap();

            let mut executor = Scirs2Exec::new();
            let data = create_test_tensor(&[size, size]);
            executor.add_tensor(graph.tensors[0].clone(), data);

            bench.iter(|| black_box(executor.forward(&graph).unwrap()));
        });
    }

    group.finish();
}

/// Benchmark throughput for complex graphs
#[cfg(feature = "integration-tests")]
fn bench_throughput_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_complex");

    // Rule evaluation simulation with multiple predicates
    group.bench_function("rule_evaluation_5_predicates", |bench| {
        let size = 1000;

        // Create expression: P0 AND P1 AND P2 AND P3 AND P4
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
    });

    // Complex logical expression
    group.bench_function("complex_logical_expression", |bench| {
        let size = 1000;

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

        bench.iter(|| black_box(executor.forward(&graph).unwrap()));
    });

    group.finish();
}

/// Benchmark throughput for batch operations
#[cfg(feature = "integration-tests")]
fn bench_throughput_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_batch");

    for batch_size in [1, 10, 100, 1000] {
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_inference", batch_size),
            &batch_size,
            |bench, &batch_size| {
                let feature_size = 100;

                bench.iter(|| {
                    // Simulate batch inference
                    for _ in 0..batch_size {
                        let p = TLExpr::pred("P", vec![Term::var("i")]);
                        let graph = compile_to_einsum(&p).unwrap();

                        let mut executor = Scirs2Exec::new();
                        let input = create_test_tensor(&[feature_size]);
                        executor.add_tensor(graph.tensors[0].clone(), input);

                        black_box(executor.forward(&graph).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "integration-tests")]
criterion_group!(
    benches,
    bench_throughput_elemwise,
    bench_throughput_matrix,
    bench_throughput_reductions,
    bench_throughput_complex,
    bench_throughput_batch
);
#[cfg(feature = "integration-tests")]
criterion_main!(benches);

// Fallback main when integration-tests feature is not enabled
#[cfg(not(feature = "integration-tests"))]
fn main() {
    eprintln!("This benchmark requires the 'integration-tests' feature to be enabled.");
    eprintln!("Run with: cargo bench --features integration-tests");
}
