//! Benchmarks for distributed schema synchronization

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensorlogic_adapters::{
    ConflictResolution, DomainInfo, NodeId, PredicateInfo, SymbolTable, SyncChangeType, SyncEvent,
    SynchronizationManager, VectorClock,
};

// Benchmark: Vector Clock Operations
fn vector_clock_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_clock");

    // Benchmark: Clock increment
    group.bench_function("increment", |b| {
        let mut clock = VectorClock::new();
        let node = NodeId::new("test-node");
        b.iter(|| {
            clock.increment(black_box(&node));
        });
    });

    // Benchmark: Clock merge
    group.bench_function("merge", |b| {
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");

        let mut clock1 = VectorClock::new();
        clock1.increment(&node1);

        let mut clock2 = VectorClock::new();
        clock2.increment(&node2);

        b.iter(|| {
            let mut c1 = clock1.clone();
            c1.merge(black_box(&clock2));
            c1
        });
    });

    // Benchmark: Happens-before check
    group.bench_function("happens_before", |b| {
        let node = NodeId::new("node");

        let mut clock1 = VectorClock::new();
        clock1.increment(&node);

        let mut clock2 = VectorClock::new();
        clock2.increment(&node);
        clock2.increment(&node);

        b.iter(|| black_box(clock1.happens_before(&clock2)));
    });

    // Benchmark: Concurrent check
    group.bench_function("is_concurrent", |b| {
        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");

        let mut clock1 = VectorClock::new();
        clock1.increment(&node1);

        let mut clock2 = VectorClock::new();
        clock2.increment(&node2);

        b.iter(|| black_box(clock1.is_concurrent(&clock2)));
    });

    group.finish();
}

// Benchmark: Event Creation
fn event_creation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_creation");

    let node = NodeId::new("test-node");
    let clock = VectorClock::new();

    // Benchmark: Simple event creation
    group.bench_function("simple", |b| {
        b.iter(|| {
            SyncEvent::new(
                black_box(node.clone()),
                black_box(clock.clone()),
                SyncChangeType::DomainAdded,
                "TestDomain".to_string(),
                None,
            )
        });
    });

    // Benchmark: Event with serialized data
    group.bench_function("with_data", |b| {
        let domain = DomainInfo::new("TestDomain", 100);
        let data = serde_json::to_string(&domain).unwrap();

        b.iter(|| {
            SyncEvent::new(
                black_box(node.clone()),
                black_box(clock.clone()),
                SyncChangeType::DomainAdded,
                "TestDomain".to_string(),
                Some(data.clone()),
            )
        });
    });

    group.finish();
}

// Benchmark: Adding Domains to Sync Manager
fn add_domain_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_domain");

    // Benchmark: Single domain addition
    group.bench_function("single", |b| {
        b.iter_batched(
            || {
                let node = NodeId::new("test-node");
                let table = SymbolTable::new();
                SynchronizationManager::new(node, table)
            },
            |mut mgr| {
                mgr.add_domain(black_box(DomainInfo::new("Person", 100)))
                    .unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Benchmark: Multiple domain additions
    for count in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter_batched(
                || {
                    let node = NodeId::new("test-node");
                    let table = SymbolTable::new();
                    SynchronizationManager::new(node, table)
                },
                |mut mgr| {
                    for i in 0..count {
                        mgr.add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                            .unwrap();
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// Benchmark: Event Application
fn apply_event_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_event");

    let node1 = NodeId::new("node1");
    let node2 = NodeId::new("node2");

    // Benchmark: Apply single event
    group.bench_function("single", |b| {
        b.iter_batched(
            || {
                let mut mgr1 = SynchronizationManager::new(node1.clone(), SymbolTable::new());
                mgr1.add_domain(DomainInfo::new("Person", 100)).unwrap();
                let events = mgr1.pending_events();

                let mgr2 = SynchronizationManager::new(node2.clone(), SymbolTable::new());

                (mgr2, events)
            },
            |(mut mgr2, events)| {
                mgr2.apply_event(black_box(events[0].clone())).unwrap();
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Benchmark: Apply multiple events
    for count in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter_batched(
                || {
                    let mut mgr1 = SynchronizationManager::new(node1.clone(), SymbolTable::new());
                    for i in 0..count {
                        mgr1.add_domain(DomainInfo::new(format!("Domain{}", i), 100))
                            .unwrap();
                    }
                    let events = mgr1.pending_events();

                    let mgr2 = SynchronizationManager::new(node2.clone(), SymbolTable::new());

                    (mgr2, events)
                },
                |(mut mgr2, events)| {
                    for event in events {
                        mgr2.apply_event(event).unwrap();
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// Benchmark: Conflict Resolution
fn conflict_resolution_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("conflict_resolution");

    for strategy in &[
        ConflictResolution::FirstWriteWins,
        ConflictResolution::LastWriteWins,
        ConflictResolution::VectorClock,
    ] {
        group.bench_with_input(
            BenchmarkId::new("strategy", format!("{:?}", strategy)),
            strategy,
            |b, &strategy| {
                b.iter_batched(
                    || {
                        let node1 = NodeId::new("node1");
                        let node2 = NodeId::new("node2");

                        // Setup node with existing domain
                        let mut mgr =
                            SynchronizationManager::new(node2.clone(), SymbolTable::new());
                        mgr.set_resolution_strategy(strategy);
                        mgr.add_domain(DomainInfo::new("Product", 100)).unwrap();

                        // Create conflicting event from node1
                        let mut mgr1 =
                            SynchronizationManager::new(node1.clone(), SymbolTable::new());
                        mgr1.add_domain(DomainInfo::new("Product", 200)).unwrap();
                        let events = mgr1.pending_events();

                        (mgr, events[0].clone())
                    },
                    |(mut mgr, event)| {
                        let _ = mgr.apply_event(black_box(event));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// Benchmark: Predicate Synchronization
fn predicate_sync_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_sync");

    let node1 = NodeId::new("node1");
    let node2 = NodeId::new("node2");

    // Setup base domains
    let setup = || {
        let mut mgr1 = SynchronizationManager::new(node1.clone(), SymbolTable::new());
        let mut mgr2 = SynchronizationManager::new(node2.clone(), SymbolTable::new());

        // Add domains to both
        mgr1.add_domain(DomainInfo::new("Person", 100)).unwrap();
        mgr1.add_domain(DomainInfo::new("Organization", 50))
            .unwrap();

        let events = mgr1.pending_events();
        for event in events {
            mgr2.apply_event(event).unwrap();
        }

        (mgr1, mgr2)
    };

    // Benchmark: Add and sync single predicate
    group.bench_function("single_predicate", |b| {
        b.iter_batched(
            setup,
            |(mut mgr1, mut mgr2)| {
                mgr1.add_predicate(PredicateInfo::new(
                    "worksAt",
                    vec!["Person".to_string(), "Organization".to_string()],
                ))
                .unwrap();

                let events = mgr1.pending_events();
                for event in events {
                    mgr2.apply_event(event).unwrap();
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Benchmark: Add and sync multiple predicates
    group.bench_function("multiple_predicates", |b| {
        b.iter_batched(
            setup,
            |(mut mgr1, mut mgr2)| {
                mgr1.add_predicate(PredicateInfo::new(
                    "worksAt",
                    vec!["Person".to_string(), "Organization".to_string()],
                ))
                .unwrap();
                mgr1.add_predicate(PredicateInfo::new(
                    "manages",
                    vec!["Person".to_string(), "Person".to_string()],
                ))
                .unwrap();
                mgr1.add_predicate(PredicateInfo::new(
                    "founded",
                    vec!["Person".to_string(), "Organization".to_string()],
                ))
                .unwrap();

                let events = mgr1.pending_events();
                for event in events {
                    mgr2.apply_event(event).unwrap();
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    vector_clock_benchmarks,
    event_creation_benchmarks,
    add_domain_benchmarks,
    apply_event_benchmarks,
    conflict_resolution_benchmarks,
    predicate_sync_benchmarks,
);

criterion_main!(benches);
