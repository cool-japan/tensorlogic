//! Benchmarks for ToRSh tensor interoperability
//!
//! Measures performance of bidirectional conversions between TensorLogic and ToRSh,
//! which are critical for neurosymbolic AI workflows.

#![cfg(feature = "torsh")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::ArrayD;
use std::hint::black_box;
use tensorlogic_scirs_backend::torsh_interop::*;
use torsh_core::device::DeviceType;

/// Benchmark TensorLogic → ToRSh conversion (f64)
fn bench_tl_to_torsh(c: &mut Criterion) {
    let mut group = c.benchmark_group("tl_to_torsh_f64");

    for size in [10, 100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64) / (*size as f64)).collect();
        let shape = vec![*size];
        let tensor =
            ArrayD::from_shape_vec(shape, data).expect("Failed to create benchmark tensor");

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = tl_to_torsh(black_box(&tensor), DeviceType::Cpu)
                    .expect("Benchmark conversion failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark TensorLogic → ToRSh conversion (f32)
fn bench_tl_to_torsh_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("tl_to_torsh_f32");

    for size in [10, 100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64) / (*size as f64)).collect();
        let shape = vec![*size];
        let tensor =
            ArrayD::from_shape_vec(shape, data).expect("Failed to create benchmark tensor");

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = tl_to_torsh_f32(black_box(&tensor), DeviceType::Cpu)
                    .expect("Benchmark f32 conversion failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark ToRSh → TensorLogic conversion (f64)
fn bench_torsh_to_tl(c: &mut Criterion) {
    use torsh_tensor::Tensor;

    let mut group = c.benchmark_group("torsh_to_tl_f64");

    for size in [10, 100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64) / (*size as f64)).collect();
        let tensor = Tensor::from_data(data, vec![*size], DeviceType::Cpu)
            .expect("Failed to create ToRSh benchmark tensor");

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = torsh_to_tl(black_box(&tensor)).expect("Benchmark ToRSh to TL failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark ToRSh → TensorLogic conversion (f32)
fn bench_torsh_f32_to_tl(c: &mut Criterion) {
    use torsh_tensor::Tensor;

    let mut group = c.benchmark_group("torsh_f32_to_tl");

    for size in [10, 100, 1000, 10000].iter() {
        let data: Vec<f32> = (0..*size).map(|i| (i as f32) / (*size as f32)).collect();
        let tensor = Tensor::from_data(data, vec![*size], DeviceType::Cpu)
            .expect("Failed to create ToRSh f32 benchmark tensor");

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result =
                    torsh_f32_to_tl(black_box(&tensor)).expect("Benchmark ToRSh f32 to TL failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark roundtrip conversion (TensorLogic → ToRSh → TensorLogic)
fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip_conversion");

    for size in [10, 100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64) / (*size as f64)).collect();
        let shape = vec![*size];
        let tensor = ArrayD::from_shape_vec(shape, data)
            .expect("Failed to create roundtrip benchmark tensor");

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let torsh = tl_to_torsh(black_box(&tensor), DeviceType::Cpu)
                    .expect("Roundtrip TL to ToRSh failed");
                let back = torsh_to_tl(black_box(&torsh)).expect("Roundtrip ToRSh to TL failed");
                black_box(back)
            });
        });
    }

    group.finish();
}

/// Benchmark matrix conversions (2D tensors)
fn bench_matrix_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_conversion");

    for dim in [10, 50, 100, 200].iter() {
        let size = dim * dim;
        let data: Vec<f64> = (0..size).map(|i| (i as f64) / (size as f64)).collect();
        let tensor = ArrayD::from_shape_vec(vec![*dim, *dim], data)
            .expect("Failed to create matrix benchmark tensor");

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, _| {
            b.iter(|| {
                let result = tl_to_torsh(black_box(&tensor), DeviceType::Cpu)
                    .expect("Matrix conversion failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark hybrid logic-neural workflow (realistic scenario)
fn bench_hybrid_workflow(c: &mut Criterion) {
    use torsh_tensor::Tensor;

    let mut group = c.benchmark_group("hybrid_workflow");

    // Simulate knowledge graph scenario: entities × entities adjacency matrix
    for num_entities in [10, 50, 100, 200].iter() {
        let size = num_entities * num_entities;

        // Step 1: Logic execution produces tensor
        let logic_results: Vec<f64> = (0..size)
            .map(|i| if i % num_entities == 0 { 1.0 } else { 0.0 })
            .collect();
        let logic_tensor =
            ArrayD::from_shape_vec(vec![*num_entities, *num_entities], logic_results)
                .expect("Failed to create logic tensor for hybrid workflow");

        // Step 2: Neural embeddings
        let embedding_size = 4;
        let embeddings: Vec<f32> = (0..num_entities * embedding_size)
            .map(|i| (i as f32) / (num_entities * embedding_size) as f32)
            .collect();
        let embedding_tensor = Tensor::from_data(
            embeddings,
            vec![*num_entities, embedding_size],
            DeviceType::Cpu,
        )
        .expect("Failed to create embedding tensor for hybrid workflow");

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_entities),
            num_entities,
            |b, _| {
                b.iter(|| {
                    // Convert logic to ToRSh
                    let logic_torsh = tl_to_torsh_f32(black_box(&logic_tensor), DeviceType::Cpu)
                        .expect("Hybrid workflow: logic to ToRSh failed");

                    // Compute embedding similarity (matrix multiplication)
                    let emb_t = embedding_tensor
                        .transpose(0, 1)
                        .expect("Hybrid workflow: transpose failed");
                    let similarity = embedding_tensor
                        .matmul(&emb_t)
                        .expect("Hybrid workflow: matmul failed");

                    // Combine logic and neural scores (hybrid reasoning)
                    let alpha = 0.7_f32;
                    let hybrid = logic_torsh
                        .mul_scalar(alpha)
                        .expect("Hybrid workflow: logic scaling failed")
                        .add(
                            &similarity
                                .mul_scalar(1.0 - alpha)
                                .expect("Hybrid workflow: neural scaling failed"),
                        )
                        .expect("Hybrid workflow: addition failed");

                    // Convert back to logic for constraint checking
                    let result = torsh_f32_to_tl(black_box(&hybrid))
                        .expect("Hybrid workflow: ToRSh to TL failed");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark type conversion overhead (f64 ↔ f32)
fn bench_type_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_conversion_overhead");

    for size in [100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64) / (*size as f64)).collect();
        let tensor = ArrayD::from_shape_vec(vec![*size], data)
            .expect("Failed to create type conversion benchmark tensor");

        // Benchmark f64 → f64 (no conversion)
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("f64_to_f64", size), size, |b, _| {
            b.iter(|| {
                let result = tl_to_torsh(black_box(&tensor), DeviceType::Cpu)
                    .expect("Type conversion f64→f64 failed");
                black_box(result)
            });
        });

        // Benchmark f64 → f32 (with conversion)
        group.bench_with_input(BenchmarkId::new("f64_to_f32", size), size, |b, _| {
            b.iter(|| {
                let result = tl_to_torsh_f32(black_box(&tensor), DeviceType::Cpu)
                    .expect("Type conversion f64→f32 failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    torsh_interop,
    bench_tl_to_torsh,
    bench_tl_to_torsh_f32,
    bench_torsh_to_tl,
    bench_torsh_f32_to_tl,
    bench_roundtrip,
    bench_matrix_conversion,
    bench_hybrid_workflow,
    bench_type_conversion,
);

criterion_main!(torsh_interop);
