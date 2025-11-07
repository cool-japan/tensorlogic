//! Benchmark training performance across different configurations.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use tensorlogic_train::{
    AdamOptimizer, AdamWOptimizer, BatchConfig, MseLoss, OptimizerConfig, SgdOptimizer, Trainer,
    TrainerConfig,
};

fn generate_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array2<f64>) {
    let data = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
        ((i + j) as f64 * 0.01).sin()
    });

    let targets = Array2::from_shape_fn((n_samples, 1), |(i, _)| {
        data.row(i).sum() / n_features as f64
    });

    (data, targets)
}

fn benchmark_optimizer_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_comparison");

    let (train_data, train_targets) = generate_data(1000, 20);

    for optimizer_name in &["SGD", "Adam", "AdamW"] {
        group.bench_with_input(
            BenchmarkId::from_parameter(optimizer_name),
            optimizer_name,
            |b, &name| {
                b.iter(|| {
                    let loss = Box::new(MseLoss);

                    let optimizer: Box<dyn tensorlogic_train::Optimizer> = match name {
                        "SGD" => Box::new(SgdOptimizer::new(OptimizerConfig {
                            learning_rate: 0.01,
                            ..Default::default()
                        })),
                        "Adam" => Box::new(AdamOptimizer::new(OptimizerConfig {
                            learning_rate: 0.001,
                            ..Default::default()
                        })),
                        "AdamW" => Box::new(AdamWOptimizer::new(OptimizerConfig {
                            learning_rate: 0.001,
                            weight_decay: 0.01,
                            ..Default::default()
                        })),
                        _ => unreachable!(),
                    };

                    let config = TrainerConfig {
                        num_epochs: 5,
                        batch_config: BatchConfig {
                            batch_size: 32,
                            shuffle: false,
                            ..Default::default()
                        },
                        ..Default::default()
                    };

                    let mut trainer = Trainer::new(config, loss, optimizer);
                    let mut parameters = HashMap::new();
                    parameters.insert("weights".to_string(), Array2::zeros((20, 1)));
                    parameters.insert("bias".to_string(), Array2::zeros((1, 1)));

                    trainer
                        .train(
                            &train_data.view(),
                            &train_targets.view(),
                            None,
                            None,
                            &mut parameters,
                        )
                        .unwrap();

                    black_box(parameters);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_scaling");

    let (train_data, train_targets) = generate_data(1000, 20);

    for &batch_size in &[16, 32, 64, 128] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    let loss = Box::new(MseLoss);
                    let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig::default()));

                    let config = TrainerConfig {
                        num_epochs: 5,
                        batch_config: BatchConfig {
                            batch_size: bs,
                            shuffle: false,
                            ..Default::default()
                        },
                        ..Default::default()
                    };

                    let mut trainer = Trainer::new(config, loss, optimizer);
                    let mut parameters = HashMap::new();
                    parameters.insert("weights".to_string(), Array2::zeros((20, 1)));
                    parameters.insert("bias".to_string(), Array2::zeros((1, 1)));

                    trainer
                        .train(
                            &train_data.view(),
                            &train_targets.view(),
                            None,
                            None,
                            &mut parameters,
                        )
                        .unwrap();

                    black_box(parameters);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_dataset_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_scaling");

    for &n_samples in &[100, 500, 1000, 2000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, &n| {
                let (train_data, train_targets) = generate_data(n, 20);

                b.iter(|| {
                    let loss = Box::new(MseLoss);
                    let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig::default()));

                    let config = TrainerConfig {
                        num_epochs: 5,
                        batch_config: BatchConfig {
                            batch_size: 32,
                            shuffle: false,
                            ..Default::default()
                        },
                        ..Default::default()
                    };

                    let mut trainer = Trainer::new(config, loss, optimizer);
                    let mut parameters = HashMap::new();
                    parameters.insert("weights".to_string(), Array2::zeros((20, 1)));
                    parameters.insert("bias".to_string(), Array2::zeros((1, 1)));

                    trainer
                        .train(
                            &train_data.view(),
                            &train_targets.view(),
                            None,
                            None,
                            &mut parameters,
                        )
                        .unwrap();

                    black_box(parameters);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_model_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_size_scaling");

    let (train_data, _) = generate_data(1000, 100);

    for &output_dim in &[1, 10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(output_dim),
            &output_dim,
            |b, &dim| {
                let targets = Array2::zeros((1000, dim));

                b.iter(|| {
                    let loss = Box::new(MseLoss);
                    let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig::default()));

                    let config = TrainerConfig {
                        num_epochs: 5,
                        batch_config: BatchConfig {
                            batch_size: 32,
                            shuffle: false,
                            ..Default::default()
                        },
                        ..Default::default()
                    };

                    let mut trainer = Trainer::new(config, loss, optimizer);
                    let mut parameters = HashMap::new();
                    parameters.insert("weights".to_string(), Array2::zeros((100, dim)));
                    parameters.insert("bias".to_string(), Array2::zeros((1, dim)));

                    trainer
                        .train(
                            &train_data.view(),
                            &targets.view(),
                            None,
                            None,
                            &mut parameters,
                        )
                        .unwrap();

                    black_box(parameters);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_gradient_clipping(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_clipping");

    let (train_data, train_targets) = generate_data(1000, 20);

    for has_clipping in &[false, true] {
        group.bench_with_input(
            BenchmarkId::from_parameter(if *has_clipping {
                "with_clipping"
            } else {
                "no_clipping"
            }),
            has_clipping,
            |b, &clip| {
                b.iter(|| {
                    let loss = Box::new(MseLoss);
                    let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig {
                        learning_rate: 0.001,
                        grad_clip: if clip { Some(1.0) } else { None },
                        ..Default::default()
                    }));

                    let config = TrainerConfig {
                        num_epochs: 5,
                        batch_config: BatchConfig {
                            batch_size: 32,
                            shuffle: false,
                            ..Default::default()
                        },
                        ..Default::default()
                    };

                    let mut trainer = Trainer::new(config, loss, optimizer);
                    let mut parameters = HashMap::new();
                    parameters.insert("weights".to_string(), Array2::zeros((20, 1)));
                    parameters.insert("bias".to_string(), Array2::zeros((1, 1)));

                    trainer
                        .train(
                            &train_data.view(),
                            &train_targets.view(),
                            None,
                            None,
                            &mut parameters,
                        )
                        .unwrap();

                    black_box(parameters);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_optimizer_comparison,
    benchmark_batch_sizes,
    benchmark_dataset_scaling,
    benchmark_model_sizes,
    benchmark_gradient_clipping,
);
criterion_main!(benches);
