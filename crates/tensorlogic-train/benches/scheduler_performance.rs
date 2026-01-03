//! Benchmark learning rate scheduler performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use tensorlogic_train::{
    CosineAnnealingLrScheduler, CyclicLrMode, CyclicLrScheduler, ExponentialLrScheduler,
    LrScheduler, MultiStepLrScheduler, NoamScheduler, OneCycleLrScheduler, OptimizerConfig,
    PlateauMode, PolynomialDecayLrScheduler, ReduceLROnPlateauScheduler, SgdOptimizer,
    StepLrScheduler, WarmupCosineLrScheduler,
};

fn benchmark_scheduler_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_step_overhead");

    // Step scheduler
    group.bench_function("step_lr", |b| {
        let mut scheduler = StepLrScheduler::new(0.1, 10, 0.5);
        let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());
        b.iter(|| {
            for _ in 0..100 {
                scheduler.step(&mut optimizer);
            }
            black_box(&scheduler);
        });
    });

    // Exponential scheduler
    group.bench_function("exponential_lr", |b| {
        let mut scheduler = ExponentialLrScheduler::new(0.1, 0.95);
        let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());
        b.iter(|| {
            for _ in 0..100 {
                scheduler.step(&mut optimizer);
            }
            black_box(&scheduler);
        });
    });

    // Cosine annealing
    group.bench_function("cosine_annealing", |b| {
        let mut scheduler = CosineAnnealingLrScheduler::new(0.1, 0.001, 100);
        let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());
        b.iter(|| {
            for _ in 0..100 {
                scheduler.step(&mut optimizer);
            }
            black_box(&scheduler);
        });
    });

    // Multi-step scheduler
    group.bench_function("multistep_lr", |b| {
        let mut scheduler = MultiStepLrScheduler::new(0.1, vec![30, 60, 90], 0.1);
        let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());
        b.iter(|| {
            for _ in 0..100 {
                scheduler.step(&mut optimizer);
            }
            black_box(&scheduler);
        });
    });

    // Noam scheduler
    group.bench_function("noam_scheduler", |b| {
        let mut scheduler = NoamScheduler::new(512, 4000, 1.0);
        let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());
        b.iter(|| {
            for _ in 0..100 {
                scheduler.step(&mut optimizer);
            }
            black_box(&scheduler);
        });
    });

    // Polynomial decay
    group.bench_function("polynomial_decay", |b| {
        let mut scheduler = PolynomialDecayLrScheduler::new(0.1, 0.001, 2.0, 100);
        let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());
        b.iter(|| {
            for _ in 0..100 {
                scheduler.step(&mut optimizer);
            }
            black_box(&scheduler);
        });
    });

    // One cycle
    group.bench_function("one_cycle", |b| {
        let mut scheduler = OneCycleLrScheduler::new(0.01, 0.1, 0.001, 100, 0.3);
        let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());
        b.iter(|| {
            for _ in 0..100 {
                scheduler.step(&mut optimizer);
            }
            black_box(&scheduler);
        });
    });

    // Cyclic LR
    group.bench_function("cyclic_lr_triangular", |b| {
        let mut scheduler = CyclicLrScheduler::new(0.001, 0.1, 10, CyclicLrMode::Triangular);
        let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());
        b.iter(|| {
            for _ in 0..100 {
                scheduler.step(&mut optimizer);
            }
            black_box(&scheduler);
        });
    });

    // Warmup cosine
    group.bench_function("warmup_cosine", |b| {
        let mut scheduler = WarmupCosineLrScheduler::new(0.1, 0.001, 10, 100);
        let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());
        b.iter(|| {
            for _ in 0..100 {
                scheduler.step(&mut optimizer);
            }
            black_box(&scheduler);
        });
    });

    group.finish();
}

fn benchmark_reduce_on_plateau(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_on_plateau");

    for mode in &[PlateauMode::Min, PlateauMode::Max] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", mode)),
            mode,
            |b, &mode| {
                b.iter(|| {
                    let mut scheduler = ReduceLROnPlateauScheduler::new(
                        0.1,   // initial_lr
                        mode,  // mode
                        0.5,   // factor
                        10,    // patience
                        1e-4,  // threshold
                        0.001, // min_lr
                        2,     // cooldown
                    );
                    let mut optimizer = SgdOptimizer::new(OptimizerConfig::default());

                    // Simulate training with varying metrics
                    let metrics = vec![
                        1.0, 0.95, 0.9, 0.88, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, // Plateau
                        0.85, 0.83, 0.8, 0.78, 0.76, 0.76, 0.76, 0.76, 0.76,
                        0.76, // Another plateau
                    ];

                    for metric in metrics {
                        scheduler.step_with_metric(&mut optimizer, metric);
                    }

                    black_box(&scheduler);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_scheduler_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_memory");

    group.bench_function("create_step_lr", |b| {
        b.iter(|| {
            let scheduler = StepLrScheduler::new(0.1, 10, 0.5);
            black_box(scheduler);
        });
    });

    group.bench_function("create_cosine_annealing", |b| {
        b.iter(|| {
            let scheduler = CosineAnnealingLrScheduler::new(0.1, 0.001, 100);
            black_box(scheduler);
        });
    });

    group.bench_function("create_multistep", |b| {
        b.iter(|| {
            let milestones = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
            let scheduler = MultiStepLrScheduler::new(0.1, milestones, 0.1);
            black_box(scheduler);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_scheduler_overhead,
    benchmark_reduce_on_plateau,
    benchmark_scheduler_memory,
);
criterion_main!(benches);
