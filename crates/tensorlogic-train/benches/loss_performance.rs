//! Benchmark loss function performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::Array2;
use std::hint::black_box;
use tensorlogic_train::{
    BCEWithLogitsLoss, ContrastiveLoss, CrossEntropyLoss, DiceLoss, FocalLoss, HingeLoss,
    HuberLoss, KLDivergenceLoss, Loss, MseLoss, TripletLoss, TverskyLoss,
};

fn generate_predictions_targets(n_samples: usize, n_classes: usize) -> (Array2<f64>, Array2<f64>) {
    let predictions = Array2::from_shape_fn((n_samples, n_classes), |(i, j)| {
        ((i + j) as f64 * 0.1).sin()
    });

    let targets = Array2::from_shape_fn((n_samples, n_classes), |(i, j)| {
        if j == i % n_classes {
            1.0
        } else {
            0.0
        }
    });

    (predictions, targets)
}

fn benchmark_loss_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_computation");

    let (predictions, targets) = generate_predictions_targets(1000, 10);

    // MSE Loss
    group.bench_function("mse_loss", |b| {
        let loss = MseLoss;
        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    // Cross-Entropy Loss
    group.bench_function("cross_entropy", |b| {
        let loss = CrossEntropyLoss::default();
        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    // BCE with Logits
    group.bench_function("bce_with_logits", |b| {
        let loss = BCEWithLogitsLoss;
        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    // Focal Loss
    group.bench_function("focal_loss", |b| {
        let loss = FocalLoss::default();
        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    // Dice Loss
    group.bench_function("dice_loss", |b| {
        let loss = DiceLoss::default();
        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    // Tversky Loss
    group.bench_function("tversky_loss", |b| {
        let loss = TverskyLoss::default();
        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    // Hinge Loss
    group.bench_function("hinge_loss", |b| {
        let loss = HingeLoss::default();
        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    // Huber Loss
    group.bench_function("huber_loss", |b| {
        let loss = HuberLoss::default();
        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    // KL Divergence
    group.bench_function("kl_divergence", |b| {
        let loss = KLDivergenceLoss::default();
        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_loss_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_data_scaling");

    for &n_samples in &[100, 500, 1000, 5000, 10000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, &n| {
                let (predictions, targets) = generate_predictions_targets(n, 10);
                let loss = CrossEntropyLoss::default();

                b.iter(|| {
                    let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_loss_class_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_class_scaling");

    for &n_classes in &[2, 5, 10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_classes),
            &n_classes,
            |b, &n| {
                let (predictions, targets) = generate_predictions_targets(1000, n);
                let loss = CrossEntropyLoss::default();

                b.iter(|| {
                    let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_contrastive_triplet(c: &mut Criterion) {
    let mut group = c.benchmark_group("metric_learning_losses");

    let n_pairs = 1000;

    // Contrastive Loss - expects [N, 2] for predictions (distances), [N, 1] for targets (labels)
    group.bench_function("contrastive_loss", |b| {
        let loss = ContrastiveLoss::default();
        let predictions =
            Array2::from_shape_fn((n_pairs, 2), |(i, j)| ((i + j) as f64 * 0.01).sin().abs());
        let targets =
            Array2::from_shape_fn((n_pairs, 1), |(i, _)| if i % 2 == 0 { 1.0 } else { 0.0 });

        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    // Triplet Loss - expects [N, 2] for predictions (pos_dist, neg_dist)
    group.bench_function("triplet_loss", |b| {
        let loss = TripletLoss::default();
        let predictions =
            Array2::from_shape_fn((n_pairs, 2), |(i, j)| ((i + j) as f64 * 0.01).cos().abs());
        let targets = Array2::zeros((n_pairs, 1)); // Not used for triplet loss

        b.iter(|| {
            let result = loss.compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_loss_computation,
    benchmark_loss_scaling,
    benchmark_loss_class_scaling,
    benchmark_contrastive_triplet,
);
criterion_main!(benches);
