//! Benchmarks for graph kernel operations
//!
//! This benchmark suite measures the performance of graph-based kernels
//! including subgraph matching, random walks, and Weisfeiler-Lehman.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensorlogic_ir::TLExpr;
use tensorlogic_sklears_kernels::{
    Graph, RandomWalkKernel, SubgraphMatchingConfig, SubgraphMatchingKernel, WalkKernelConfig,
    WeisfeilerLehmanConfig, WeisfeilerLehmanKernel,
};

/// Generate a simple linear expression chain
fn generate_linear_expr(depth: usize) -> TLExpr {
    let mut expr = TLExpr::pred("p0", vec![]);
    for i in 1..depth {
        expr = TLExpr::and(expr, TLExpr::pred(format!("p{}", i), vec![]));
    }
    expr
}

/// Generate a balanced binary tree expression
fn generate_tree_expr(depth: usize) -> TLExpr {
    if depth == 0 {
        TLExpr::pred("leaf", vec![])
    } else {
        TLExpr::and(generate_tree_expr(depth - 1), generate_tree_expr(depth - 1))
    }
}

/// Generate a complex nested expression
fn generate_complex_expr(depth: usize) -> TLExpr {
    if depth == 0 {
        TLExpr::pred("base", vec![])
    } else {
        TLExpr::or(
            TLExpr::and(
                generate_complex_expr(depth - 1),
                TLExpr::pred(format!("pred{}", depth), vec![]),
            ),
            TLExpr::negate(generate_complex_expr(depth - 1)),
        )
    }
}

/// Benchmark graph construction from TLExpr
fn bench_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction");

    for depth in [5, 10, 15, 20].iter() {
        let expr = generate_linear_expr(*depth);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("linear_{}", depth)),
            depth,
            |b, _| {
                b.iter(|| {
                    black_box(Graph::from_tlexpr(&expr));
                });
            },
        );
    }

    for depth in [2, 3, 4].iter() {
        let expr = generate_tree_expr(*depth);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("tree_{}", depth)),
            depth,
            |b, _| {
                b.iter(|| {
                    black_box(Graph::from_tlexpr(&expr));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark subgraph matching kernel
fn bench_subgraph_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("subgraph_matching");

    for depth in [3, 5, 7].iter() {
        let expr1 = generate_linear_expr(*depth);
        let expr2 = generate_linear_expr(*depth);
        let graph1 = Graph::from_tlexpr(&expr1);
        let graph2 = Graph::from_tlexpr(&expr2);

        for max_size in [2, 3].iter() {
            let config = SubgraphMatchingConfig::new().with_max_size(*max_size);
            let kernel = SubgraphMatchingKernel::new(config);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("depth_{}_maxsize_{}", depth, max_size)),
                &(depth, max_size),
                |b, _| {
                    b.iter(|| {
                        black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark random walk kernel
fn bench_random_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_walk");

    for depth in [5, 10, 15].iter() {
        let expr1 = generate_linear_expr(*depth);
        let expr2 = generate_linear_expr(*depth);
        let graph1 = Graph::from_tlexpr(&expr1);
        let graph2 = Graph::from_tlexpr(&expr2);

        for max_length in [3, 4, 5].iter() {
            let config = WalkKernelConfig::new()
                .with_max_length(*max_length)
                .with_decay(0.8);
            let kernel = RandomWalkKernel::new(config).unwrap();

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("depth_{}_maxlen_{}", depth, max_length)),
                &(depth, max_length),
                |b, _| {
                    b.iter(|| {
                        black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark Weisfeiler-Lehman kernel
fn bench_weisfeiler_lehman(c: &mut Criterion) {
    let mut group = c.benchmark_group("weisfeiler_lehman");

    for depth in [5, 10, 15, 20].iter() {
        let expr1 = generate_linear_expr(*depth);
        let expr2 = generate_linear_expr(*depth);
        let graph1 = Graph::from_tlexpr(&expr1);
        let graph2 = Graph::from_tlexpr(&expr2);

        for iterations in [1, 2, 3, 4].iter() {
            let config = WeisfeilerLehmanConfig::new().with_iterations(*iterations);
            let kernel = WeisfeilerLehmanKernel::new(config);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("depth_{}_iters_{}", depth, iterations)),
                &(depth, iterations),
                |b, _| {
                    b.iter(|| {
                        black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark graph kernel comparison
fn bench_graph_kernel_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_kernel_comparison");

    let depth = 10;
    let expr1 = generate_linear_expr(depth);
    let expr2 = generate_linear_expr(depth);
    let graph1 = Graph::from_tlexpr(&expr1);
    let graph2 = Graph::from_tlexpr(&expr2);

    // Subgraph matching
    group.bench_function("subgraph_matching", |b| {
        let config = SubgraphMatchingConfig::new().with_max_size(3);
        let kernel = SubgraphMatchingKernel::new(config);
        b.iter(|| {
            black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
        });
    });

    // Random walk
    group.bench_function("random_walk", |b| {
        let config = WalkKernelConfig::new().with_max_length(4).with_decay(0.8);
        let kernel = RandomWalkKernel::new(config).unwrap();
        b.iter(|| {
            black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
        });
    });

    // Weisfeiler-Lehman
    group.bench_function("weisfeiler_lehman", |b| {
        let config = WeisfeilerLehmanConfig::new().with_iterations(3);
        let kernel = WeisfeilerLehmanKernel::new(config);
        b.iter(|| {
            black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
        });
    });

    group.finish();
}

/// Benchmark graph structure complexity
fn bench_graph_structure_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_structure_complexity");

    let config = WeisfeilerLehmanConfig::new().with_iterations(2);
    let kernel = WeisfeilerLehmanKernel::new(config);

    // Linear chains
    for depth in [10, 20, 30].iter() {
        let expr1 = generate_linear_expr(*depth);
        let expr2 = generate_linear_expr(*depth);
        let graph1 = Graph::from_tlexpr(&expr1);
        let graph2 = Graph::from_tlexpr(&expr2);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("linear_{}", depth)),
            depth,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
                });
            },
        );
    }

    // Binary trees
    for depth in [2, 3, 4].iter() {
        let expr1 = generate_tree_expr(*depth);
        let expr2 = generate_tree_expr(*depth);
        let graph1 = Graph::from_tlexpr(&expr1);
        let graph2 = Graph::from_tlexpr(&expr2);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("tree_{}", depth)),
            depth,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
                });
            },
        );
    }

    // Complex nested structures
    for depth in [2, 3, 4].iter() {
        let expr1 = generate_complex_expr(*depth);
        let expr2 = generate_complex_expr(*depth);
        let graph1 = Graph::from_tlexpr(&expr1);
        let graph2 = Graph::from_tlexpr(&expr2);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("complex_{}", depth)),
            depth,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark graph kernel with different similarities
fn bench_graph_similarity_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_similarity_levels");

    let config = WeisfeilerLehmanConfig::new().with_iterations(3);
    let kernel = WeisfeilerLehmanKernel::new(config);

    let base_expr = generate_linear_expr(10);
    let graph1 = Graph::from_tlexpr(&base_expr);

    // Identical graphs
    group.bench_function("identical", |b| {
        let graph2 = Graph::from_tlexpr(&base_expr);
        b.iter(|| {
            black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
        });
    });

    // Slightly different graphs
    group.bench_function("similar", |b| {
        let expr2 = TLExpr::and(generate_linear_expr(9), TLExpr::pred("different", vec![]));
        let graph2 = Graph::from_tlexpr(&expr2);
        b.iter(|| {
            black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
        });
    });

    // Very different graphs
    group.bench_function("different", |b| {
        let expr2 = generate_tree_expr(3);
        let graph2 = Graph::from_tlexpr(&expr2);
        b.iter(|| {
            black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
        });
    });

    group.finish();
}

/// Benchmark walk kernel decay parameter
fn bench_walk_kernel_decay(c: &mut Criterion) {
    let mut group = c.benchmark_group("walk_kernel_decay");

    let expr1 = generate_linear_expr(10);
    let expr2 = generate_linear_expr(10);
    let graph1 = Graph::from_tlexpr(&expr1);
    let graph2 = Graph::from_tlexpr(&expr2);

    for decay in [0.5, 0.7, 0.9].iter() {
        let config = WalkKernelConfig::new()
            .with_max_length(4)
            .with_decay(*decay);
        let kernel = RandomWalkKernel::new(config).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("decay_{:.1}", decay)),
            decay,
            |b, _| {
                b.iter(|| {
                    black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark scalability with graph size
fn bench_graph_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_scalability");

    let config = WeisfeilerLehmanConfig::new().with_iterations(2);
    let kernel = WeisfeilerLehmanKernel::new(config);

    for num_nodes in [10, 20, 50, 100].iter() {
        let expr1 = generate_linear_expr(*num_nodes);
        let expr2 = generate_linear_expr(*num_nodes);
        let graph1 = Graph::from_tlexpr(&expr1);
        let graph2 = Graph::from_tlexpr(&expr2);

        group.throughput(Throughput::Elements(*num_nodes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(num_nodes), num_nodes, |b, _| {
            b.iter(|| {
                black_box(kernel.compute_graphs(&graph1, &graph2).unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_graph_construction,
    bench_subgraph_matching,
    bench_random_walk,
    bench_weisfeiler_lehman,
    bench_graph_kernel_comparison,
    bench_graph_structure_complexity,
    bench_graph_similarity_levels,
    bench_walk_kernel_decay,
    bench_graph_scalability,
);

criterion_main!(benches);
