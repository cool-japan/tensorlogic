//! Benchmarks for factor operations
//!
//! This benchmark suite measures the performance of core factor operations including:
//! - Factor product (combining factors)
//! - Marginalization (summing out variables)
//! - Division (message quotients)
//! - Reduction (conditioning on evidence)
//! - Maximization (max-product operations)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array;
use std::hint::black_box;
use tensorlogic_quantrs_hooks::Factor;

/// Create a binary factor for benchmarking
fn create_binary_factor(name: &str, vars: Vec<String>, values: Vec<f64>) -> Factor {
    let shape = vec![2, 2];
    let array = Array::from_shape_vec(shape, values).unwrap().into_dyn();
    Factor::new(name.to_string(), vars, array).unwrap()
}

/// Create a ternary factor (3 values per variable)
fn create_ternary_factor(name: &str, vars: Vec<String>) -> Factor {
    let size = 3_usize.pow(vars.len() as u32);
    let values: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) / size as f64).collect();
    let shape: Vec<usize> = vec![3; vars.len()];
    let array = Array::from_shape_vec(shape, values).unwrap().into_dyn();
    Factor::new(name.to_string(), vars, array).unwrap()
}

/// Create a factor with specified cardinality per variable
fn create_factor_with_card(name: &str, vars: Vec<String>, card: usize) -> Factor {
    let size = card.pow(vars.len() as u32);
    let values: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) / size as f64).collect();
    let shape: Vec<usize> = vec![card; vars.len()];
    let array = Array::from_shape_vec(shape, values).unwrap().into_dyn();
    Factor::new(name.to_string(), vars, array).unwrap()
}

/// Benchmark factor product operations
fn bench_factor_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("factor_product");

    // Binary factors: 2x2 and 2x2 -> 2x2x2
    let f1 = create_binary_factor("f1", vec!["X".to_string()], vec![0.6, 0.4]);
    let f2 = create_binary_factor("f2", vec!["Y".to_string()], vec![0.7, 0.3]);

    group.throughput(Throughput::Elements(1));
    group.bench_function("binary_2x2", |b| {
        b.iter(|| {
            black_box(f1.product(&f2).unwrap());
        });
    });

    // Overlapping variables: (X,Y) x (Y,Z)
    let f3 = create_binary_factor(
        "f3",
        vec!["X".to_string(), "Y".to_string()],
        vec![0.1, 0.2, 0.3, 0.4],
    );
    let f4 = create_binary_factor(
        "f4",
        vec!["Y".to_string(), "Z".to_string()],
        vec![0.5, 0.6, 0.7, 0.8],
    );

    group.bench_function("overlapping_binary", |b| {
        b.iter(|| {
            black_box(f3.product(&f4).unwrap());
        });
    });

    // Ternary factors
    let f5 = create_ternary_factor("f5", vec!["A".to_string(), "B".to_string()]);
    let f6 = create_ternary_factor("f6", vec!["B".to_string(), "C".to_string()]);

    group.bench_function("ternary_3x3", |b| {
        b.iter(|| {
            black_box(f5.product(&f6).unwrap());
        });
    });

    // Large factors with varying cardinality
    for card in [5, 10, 20] {
        let large_f1 =
            create_factor_with_card("large1", vec!["V1".to_string(), "V2".to_string()], card);
        let large_f2 =
            create_factor_with_card("large2", vec!["V2".to_string(), "V3".to_string()], card);

        group.throughput(Throughput::Elements((card * card * card) as u64));
        group.bench_with_input(
            BenchmarkId::new("large_card", card),
            &(large_f1, large_f2),
            |b, (f1, f2)| {
                b.iter(|| {
                    black_box(f1.product(f2).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark marginalization operations
fn bench_marginalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("marginalization");

    // Binary factor marginalization
    let f1 = create_binary_factor(
        "joint",
        vec!["X".to_string(), "Y".to_string()],
        vec![0.1, 0.2, 0.3, 0.4],
    );

    group.throughput(Throughput::Elements(1));
    group.bench_function("binary_2x2", |b| {
        b.iter(|| {
            black_box(f1.marginalize_out("Y").unwrap());
        });
    });

    // Ternary factor
    let f2 = create_ternary_factor(
        "f2",
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    );

    group.bench_function("ternary_3x3x3_marginalize_one", |b| {
        b.iter(|| {
            black_box(f2.marginalize_out("C").unwrap());
        });
    });

    group.bench_function("ternary_3x3x3_marginalize_two", |b| {
        b.iter(|| {
            let temp = f2.marginalize_out("C").unwrap();
            black_box(temp.marginalize_out("B").unwrap());
        });
    });

    // Large factors
    for card in [5, 10, 15] {
        let large_f = create_factor_with_card(
            "large",
            vec!["V1".to_string(), "V2".to_string(), "V3".to_string()],
            card,
        );

        group.throughput(Throughput::Elements((card * card * card) as u64));
        group.bench_with_input(BenchmarkId::new("large_card", card), &large_f, |b, f| {
            b.iter(|| {
                black_box(f.marginalize_out("V3").unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark factor division
fn bench_division(c: &mut Criterion) {
    let mut group = c.benchmark_group("division");

    let f1 = create_binary_factor("numerator", vec!["X".to_string()], vec![0.6, 0.4]);
    let f2 = create_binary_factor("denominator", vec!["X".to_string()], vec![0.3, 0.2]);

    group.throughput(Throughput::Elements(1));
    group.bench_function("binary_division", |b| {
        b.iter(|| {
            black_box(f1.divide(&f2).unwrap());
        });
    });

    // Larger factors
    for card in [5, 10, 20] {
        let large_f1 = create_factor_with_card("num", vec!["V1".to_string()], card);
        let large_f2 = create_factor_with_card("den", vec!["V1".to_string()], card);

        group.throughput(Throughput::Elements(card as u64));
        group.bench_with_input(
            BenchmarkId::new("large_card", card),
            &(large_f1, large_f2),
            |b, (f1, f2)| {
                b.iter(|| {
                    black_box(f1.divide(f2).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark factor reduction (conditioning)
fn bench_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction");

    let f1 = create_binary_factor(
        "joint",
        vec!["X".to_string(), "Y".to_string()],
        vec![0.1, 0.2, 0.3, 0.4],
    );

    group.throughput(Throughput::Elements(1));
    group.bench_function("binary_reduce", |b| {
        b.iter(|| {
            black_box(f1.reduce("Y", 1).unwrap());
        });
    });

    // Ternary factor reduction
    let f2 = create_ternary_factor(
        "f2",
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    );

    group.bench_function("ternary_reduce_one", |b| {
        b.iter(|| {
            black_box(f2.reduce("C", 1).unwrap());
        });
    });

    group.bench_function("ternary_reduce_two", |b| {
        b.iter(|| {
            let temp = f2.reduce("C", 1).unwrap();
            black_box(temp.reduce("B", 1).unwrap());
        });
    });

    group.finish();
}

/// Benchmark maximization operations (for max-product)
fn bench_maximization(c: &mut Criterion) {
    let mut group = c.benchmark_group("maximization");

    let f1 = create_binary_factor(
        "joint",
        vec!["X".to_string(), "Y".to_string()],
        vec![0.1, 0.2, 0.3, 0.4],
    );

    group.throughput(Throughput::Elements(1));
    group.bench_function("binary_maximize", |b| {
        b.iter(|| {
            black_box(f1.maximize_out("Y").unwrap());
        });
    });

    // Ternary maximization
    let f2 = create_ternary_factor(
        "f2",
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    );

    group.bench_function("ternary_maximize", |b| {
        b.iter(|| {
            black_box(f2.maximize_out("C").unwrap());
        });
    });

    // Large factors
    for card in [5, 10, 15] {
        let large_f = create_factor_with_card(
            "large",
            vec!["V1".to_string(), "V2".to_string(), "V3".to_string()],
            card,
        );

        group.throughput(Throughput::Elements((card * card * card) as u64));
        group.bench_with_input(BenchmarkId::new("large_card", card), &large_f, |b, f| {
            b.iter(|| {
                black_box(f.maximize_out("V3").unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark normalization
fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");

    for card in [2, 5, 10, 20] {
        let factor = create_factor_with_card(
            "to_normalize",
            vec!["V1".to_string(), "V2".to_string()],
            card,
        );

        group.throughput(Throughput::Elements((card * card) as u64));
        group.bench_with_input(BenchmarkId::new("card", card), &factor, |b, f| {
            b.iter(|| {
                let mut temp = f.clone();
                temp.normalize();
                black_box(temp);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_factor_product,
    bench_marginalization,
    bench_division,
    bench_reduction,
    bench_maximization,
    bench_normalization
);
criterion_main!(benches);
