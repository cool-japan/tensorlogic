//! Benchmark metric computation performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::Array2;
use std::hint::black_box;
use tensorlogic_train::{
    Accuracy, BalancedAccuracy, CohensKappa, ConfusionMatrix, F1Score,
    MatthewsCorrelationCoefficient, Metric, PerClassMetrics, Precision, Recall, RocCurve,
    TopKAccuracy,
};

fn generate_classification_data(n_samples: usize, n_classes: usize) -> (Array2<f64>, Array2<f64>) {
    // Generate one-hot encoded predictions and targets
    let mut predictions = Array2::zeros((n_samples, n_classes));
    let mut targets = Array2::zeros((n_samples, n_classes));

    for i in 0..n_samples {
        let pred_class = i % n_classes;
        let true_class = (i + 1) % n_classes;
        predictions[[i, pred_class]] = 1.0;
        targets[[i, true_class]] = 1.0;
    }

    (predictions, targets)
}

fn benchmark_basic_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_metrics");

    let (predictions, targets) = generate_classification_data(1000, 10);

    group.bench_function("accuracy", |b| {
        let metric = Accuracy::default();
        b.iter(|| {
            let result = metric
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("precision", |b| {
        let metric = Precision::default();
        b.iter(|| {
            let result = metric
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("recall", |b| {
        let metric = Recall::default();
        b.iter(|| {
            let result = metric
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("f1_score", |b| {
        let metric = F1Score::default();
        b.iter(|| {
            let result = metric
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("balanced_accuracy", |b| {
        let metric = BalancedAccuracy;
        b.iter(|| {
            let result = metric
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_confusion_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("confusion_matrix");

    for &n_classes in &[5, 10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_classes),
            &n_classes,
            |b, &n| {
                let (predictions, targets) = generate_classification_data(1000, n);

                b.iter(|| {
                    let result =
                        ConfusionMatrix::compute(&predictions.view(), &targets.view()).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_advanced_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_metrics");

    let (predictions, targets) = generate_classification_data(1000, 10);

    group.bench_function("cohens_kappa", |b| {
        let metric = CohensKappa;
        b.iter(|| {
            let result = metric
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("matthews_correlation", |b| {
        let metric = MatthewsCorrelationCoefficient;
        b.iter(|| {
            let result = metric
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("per_class_metrics", |b| {
        b.iter(|| {
            let result = PerClassMetrics::compute(&predictions.view(), &targets.view()).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn benchmark_top_k_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_accuracy");

    let (probs, targets) = generate_classification_data(1000, 100);

    for &k in &[1, 3, 5, 10] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k_val| {
            b.iter(|| {
                let metric = TopKAccuracy::new(k_val);
                let result = metric.compute(&probs.view(), &targets.view()).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn benchmark_roc_curve(c: &mut Criterion) {
    let mut group = c.benchmark_group("roc_curve");

    for &n_samples in &[100, 500, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, &n| {
                let scores: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
                let targets: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();

                b.iter(|| {
                    let result = RocCurve::compute(&scores, &targets).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_metric_computation_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("metric_batch_computation");

    let (predictions, targets) = generate_classification_data(1000, 10);

    group.bench_function("compute_all_basic_metrics", |b| {
        b.iter(|| {
            let accuracy = Accuracy::default();
            let precision = Precision::default();
            let recall = Recall::default();
            let f1 = F1Score::default();

            let acc_val = accuracy
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            let prec_val = precision
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            let rec_val = recall
                .compute(&predictions.view(), &targets.view())
                .unwrap();
            let f1_val = f1.compute(&predictions.view(), &targets.view()).unwrap();

            black_box((acc_val, prec_val, rec_val, f1_val));
        });
    });

    group.finish();
}

fn benchmark_metric_data_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("metric_data_scaling");

    for &n_samples in &[100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, &n| {
                let (predictions, targets) = generate_classification_data(n, 10);

                b.iter(|| {
                    let metric = Accuracy::default();
                    let result = metric
                        .compute(&predictions.view(), &targets.view())
                        .unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_basic_metrics,
    benchmark_confusion_matrix,
    benchmark_advanced_metrics,
    benchmark_top_k_accuracy,
    benchmark_roc_curve,
    benchmark_metric_computation_batch,
    benchmark_metric_data_scaling,
);
criterion_main!(benches);
