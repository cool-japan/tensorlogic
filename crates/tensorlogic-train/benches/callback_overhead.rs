//! Benchmark callback overhead during training.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::hint::black_box;
use tensorlogic_train::{
    AdamOptimizer, BatchConfig, CallbackList, EarlyStoppingCallback, EpochCallback,
    GradientMonitor, HistogramCallback, ModelEMACallback, MseLoss, OptimizerConfig,
    ProfilingCallback, Trainer, TrainerConfig, ValidationCallback,
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

fn benchmark_no_callbacks(c: &mut Criterion) {
    c.bench_function("training_no_callbacks", |b| {
        let (train_data, train_targets) = generate_data(1000, 20);

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
    });
}

fn benchmark_single_callback_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_callback_overhead");

    let (train_data, train_targets) = generate_data(1000, 20);

    // Epoch callback
    group.bench_function("epoch_callback", |b| {
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

            let mut callbacks = CallbackList::new();
            callbacks.add(Box::new(EpochCallback::new(false)));

            let mut trainer = Trainer::new(config, loss, optimizer).with_callbacks(callbacks);

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
    });

    // Validation callback
    group.bench_function("validation_callback", |b| {
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

            let mut callbacks = CallbackList::new();
            callbacks.add(Box::new(ValidationCallback::new(1)));

            let mut trainer = Trainer::new(config, loss, optimizer).with_callbacks(callbacks);

            let mut parameters = HashMap::new();
            parameters.insert("weights".to_string(), Array2::zeros((20, 1)));
            parameters.insert("bias".to_string(), Array2::zeros((1, 1)));

            trainer
                .train(
                    &train_data.view(),
                    &train_targets.view(),
                    Some(&train_data.view()),
                    Some(&train_targets.view()),
                    &mut parameters,
                )
                .unwrap();

            black_box(parameters);
        });
    });

    // Early stopping callback
    group.bench_function("early_stopping", |b| {
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

            let mut callbacks = CallbackList::new();
            callbacks.add(Box::new(EarlyStoppingCallback::new(3, 1e-4)));

            let mut trainer = Trainer::new(config, loss, optimizer).with_callbacks(callbacks);

            let mut parameters = HashMap::new();
            parameters.insert("weights".to_string(), Array2::zeros((20, 1)));
            parameters.insert("bias".to_string(), Array2::zeros((1, 1)));

            trainer
                .train(
                    &train_data.view(),
                    &train_targets.view(),
                    Some(&train_data.view()),
                    Some(&train_targets.view()),
                    &mut parameters,
                )
                .unwrap();

            black_box(parameters);
        });
    });

    // Model EMA callback
    group.bench_function("model_ema", |b| {
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

            let mut callbacks = CallbackList::new();
            callbacks.add(Box::new(ModelEMACallback::new(0.999, true)));

            let mut trainer = Trainer::new(config, loss, optimizer).with_callbacks(callbacks);

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
    });

    group.finish();
}

fn benchmark_multiple_callbacks(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_callbacks");

    let (train_data, train_targets) = generate_data(1000, 20);

    for &n_callbacks in &[1, 3, 5, 10] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_callbacks),
            &n_callbacks,
            |b, &n| {
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

                    let mut callbacks = CallbackList::new();

                    // Add n callbacks
                    for _ in 0..n {
                        callbacks.add(Box::new(EpochCallback::new(false)));
                    }

                    let mut trainer =
                        Trainer::new(config, loss, optimizer).with_callbacks(callbacks);

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

fn benchmark_heavy_callbacks(c: &mut Criterion) {
    let mut group = c.benchmark_group("heavy_callback_overhead");

    let (train_data, train_targets) = generate_data(1000, 20);

    // Gradient monitor (computes statistics every batch)
    group.bench_function("gradient_monitor", |b| {
        b.iter(|| {
            let loss = Box::new(MseLoss);
            let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig::default()));

            let config = TrainerConfig {
                num_epochs: 3, // Reduced for heavy callback
                batch_config: BatchConfig {
                    batch_size: 32,
                    shuffle: false,
                    ..Default::default()
                },
                ..Default::default()
            };

            let mut callbacks = CallbackList::new();
            callbacks.add(Box::new(GradientMonitor::new(1, 1e-7, 100.0)));

            let mut trainer = Trainer::new(config, loss, optimizer).with_callbacks(callbacks);

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
    });

    // Histogram callback (tracks weight distributions)
    group.bench_function("histogram_callback", |b| {
        b.iter(|| {
            let loss = Box::new(MseLoss);
            let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig::default()));

            let config = TrainerConfig {
                num_epochs: 3,
                batch_config: BatchConfig {
                    batch_size: 32,
                    shuffle: false,
                    ..Default::default()
                },
                ..Default::default()
            };

            let mut callbacks = CallbackList::new();
            callbacks.add(Box::new(HistogramCallback::new(1, 10, false)));

            let mut trainer = Trainer::new(config, loss, optimizer).with_callbacks(callbacks);

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
    });

    // Profiling callback
    group.bench_function("profiling_callback", |b| {
        b.iter(|| {
            let loss = Box::new(MseLoss);
            let optimizer = Box::new(AdamOptimizer::new(OptimizerConfig::default()));

            let config = TrainerConfig {
                num_epochs: 3,
                batch_config: BatchConfig {
                    batch_size: 32,
                    shuffle: false,
                    ..Default::default()
                },
                ..Default::default()
            };

            let mut callbacks = CallbackList::new();
            callbacks.add(Box::new(ProfilingCallback::new(false, 1)));

            let mut trainer = Trainer::new(config, loss, optimizer).with_callbacks(callbacks);

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
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_no_callbacks,
    benchmark_single_callback_overhead,
    benchmark_multiple_callbacks,
    benchmark_heavy_callbacks,
);
criterion_main!(benches);
