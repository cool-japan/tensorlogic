//! Benchmarks for profile-guided optimization

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::time::Duration;
use tensorlogic_ir::{EinsumGraph, ExecutionProfile, NodeStats, ProfileGuidedOptimizer};

fn bench_profile_recording(c: &mut Criterion) {
    c.bench_function("profile_record_node", |b| {
        b.iter(|| {
            let mut profile = ExecutionProfile::new();
            profile.record_node(black_box(0), Duration::from_millis(10), black_box(1024));
            black_box(profile);
        })
    });

    c.bench_function("profile_record_tensor_access", |b| {
        b.iter(|| {
            let mut profile = ExecutionProfile::new();
            profile.record_tensor_access(black_box(0), black_box(1024));
            black_box(profile);
        })
    });

    c.bench_function("profile_record_multiple_nodes", |b| {
        b.iter(|| {
            let mut profile = ExecutionProfile::new();
            for i in 0..10 {
                profile.record_node(
                    black_box(i),
                    Duration::from_millis(10 + i as u64),
                    black_box((1024 * (i + 1)) as u64),
                );
            }
            black_box(profile);
        })
    });
}

fn bench_profile_analysis(c: &mut Criterion) {
    let mut profile = ExecutionProfile::new();
    for i in 0..100 {
        profile.record_node(i % 10, Duration::from_millis(10), 1024);
    }

    c.bench_function("profile_get_hot_nodes", |b| {
        b.iter(|| {
            let _ = black_box(&profile).get_hot_nodes(black_box(5));
        })
    });

    c.bench_function("profile_get_memory_intensive", |b| {
        b.iter(|| {
            let _ = black_box(&profile).get_memory_intensive_nodes(black_box(100 * 1024 * 1024));
        })
    });

    c.bench_function("profile_merge", |b| {
        let mut profile2 = ExecutionProfile::new();
        for i in 0..50 {
            profile2.record_node(i % 5, Duration::from_millis(5), 512);
        }

        b.iter(|| {
            let mut p = profile.clone();
            p.merge(black_box(&profile2));
            black_box(p);
        })
    });
}

fn bench_node_stats(c: &mut Criterion) {
    c.bench_function("node_stats_record_execution", |b| {
        b.iter(|| {
            let mut stats = NodeStats::new();
            stats.record_execution(black_box(Duration::from_millis(10)), black_box(1024));
            black_box(stats);
        })
    });

    c.bench_function("node_stats_avg_time", |b| {
        let mut stats = NodeStats::new();
        for _ in 0..100 {
            stats.record_execution(Duration::from_millis(10), 1024);
        }

        b.iter(|| {
            let _ = black_box(&stats).avg_time();
        })
    });

    c.bench_function("node_stats_performance_score", |b| {
        let mut stats = NodeStats::new();
        for _ in 0..100 {
            stats.record_execution(Duration::from_millis(10), 1024);
        }

        b.iter(|| {
            let _ = black_box(&stats).performance_score();
        })
    });
}

fn bench_pgo_optimizer(c: &mut Criterion) {
    let mut profile = ExecutionProfile::new();
    for i in 0..50 {
        profile.record_node(
            i % 10,
            Duration::from_millis(10 + i as u64),
            (1024 * (i + 1)) as u64,
        );
    }
    for i in 0..100 {
        profile.record_tensor_access(i % 5, 4096);
    }

    c.bench_function("pgo_optimizer_analyze", |b| {
        let graph = EinsumGraph::new();
        let optimizer = ProfileGuidedOptimizer::new(profile.clone());

        b.iter(|| {
            let _ = black_box(&optimizer).analyze(black_box(&graph));
        })
    });

    c.bench_function("pgo_optimizer_creation", |b| {
        b.iter(|| {
            let _ = ProfileGuidedOptimizer::new(black_box(profile.clone()));
        })
    });

    c.bench_function("pgo_optimizer_with_thresholds", |b| {
        b.iter(|| {
            let _ = ProfileGuidedOptimizer::new(black_box(profile.clone()))
                .with_hot_threshold(black_box(10))
                .with_memory_threshold(black_box(100 * 1024 * 1024));
        })
    });
}

fn bench_profile_serialization(c: &mut Criterion) {
    let mut profile = ExecutionProfile::new();
    for i in 0..100 {
        profile.record_node(i % 10, Duration::from_millis(10), 1024);
    }
    for i in 0..50 {
        profile.record_tensor_access(i % 5, 4096);
    }

    c.bench_function("profile_to_json", |b| {
        b.iter(|| {
            let _ = black_box(&profile).to_json();
        })
    });

    let json = profile.to_json().unwrap();

    c.bench_function("profile_from_json", |b| {
        b.iter(|| {
            let _ = ExecutionProfile::from_json(black_box(&json));
        })
    });
}

criterion_group!(
    pgo,
    bench_profile_recording,
    bench_profile_analysis,
    bench_node_stats,
    bench_pgo_optimizer,
    bench_profile_serialization
);
criterion_main!(pgo);
