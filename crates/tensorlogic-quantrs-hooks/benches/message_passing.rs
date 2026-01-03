//! Benchmarks for message passing algorithms
//!
//! This benchmark suite measures the performance of message passing algorithms including:
//! - Sum-product belief propagation (marginal inference)
//! - Max-product algorithm (MAP inference)
//! - Loopy BP with damping
//! - Convergence characteristics

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array;
use std::hint::black_box;
use tensorlogic_quantrs_hooks::{
    Factor, FactorGraph, MaxProductAlgorithm, MessagePassingAlgorithm, SumProductAlgorithm,
};

/// Create a simple chain graph: X -- Y -- Z
fn create_chain_graph(card: usize) -> FactorGraph {
    let mut graph = FactorGraph::new();

    // Add variables
    graph.add_variable_with_card("X".to_string(), "Domain".to_string(), card);
    graph.add_variable_with_card("Y".to_string(), "Domain".to_string(), card);
    graph.add_variable_with_card("Z".to_string(), "Domain".to_string(), card);

    // Add factors
    let size_xy = card * card;
    let values_xy: Vec<f64> = (0..size_xy)
        .map(|i| (i as f64 + 1.0) / size_xy as f64)
        .collect();
    let shape_xy = vec![card, card];
    let array_xy = Array::from_shape_vec(shape_xy, values_xy)
        .unwrap()
        .into_dyn();
    let f_xy = Factor::new(
        "P(X,Y)".to_string(),
        vec!["X".to_string(), "Y".to_string()],
        array_xy,
    )
    .unwrap();
    graph.add_factor(f_xy).unwrap();

    let size_yz = card * card;
    let values_yz: Vec<f64> = (0..size_yz)
        .map(|i| (i as f64 + 1.0) / size_yz as f64)
        .collect();
    let shape_yz = vec![card, card];
    let array_yz = Array::from_shape_vec(shape_yz, values_yz)
        .unwrap()
        .into_dyn();
    let f_yz = Factor::new(
        "P(Y,Z)".to_string(),
        vec!["Y".to_string(), "Z".to_string()],
        array_yz,
    )
    .unwrap();
    graph.add_factor(f_yz).unwrap();

    graph
}

/// Create a grid graph: 2D lattice MRF
fn create_grid_graph(rows: usize, cols: usize, card: usize) -> FactorGraph {
    let mut graph = FactorGraph::new();

    // Add variables
    for i in 0..rows {
        for j in 0..cols {
            graph.add_variable_with_card(format!("X_{}_{}", i, j), "Pixel".to_string(), card);
        }
    }

    // Add pairwise factors (horizontal and vertical edges)
    let size = card * card;
    let values: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) / size as f64).collect();
    let shape = vec![card, card];

    // Horizontal edges
    for i in 0..rows {
        for j in 0..(cols - 1) {
            let array = Array::from_shape_vec(shape.clone(), values.clone())
                .unwrap()
                .into_dyn();
            let factor = Factor::new(
                format!("edge_{}_{}_{}_{}", i, j, i, j + 1),
                vec![format!("X_{}_{}", i, j), format!("X_{}_{}", i, j + 1)],
                array,
            )
            .unwrap();
            graph.add_factor(factor).unwrap();
        }
    }

    // Vertical edges
    for i in 0..(rows - 1) {
        for j in 0..cols {
            let array = Array::from_shape_vec(shape.clone(), values.clone())
                .unwrap()
                .into_dyn();
            let factor = Factor::new(
                format!("edge_{}_{}_{}_{}", i, j, i + 1, j),
                vec![format!("X_{}_{}", i, j), format!("X_{}_{}", i + 1, j)],
                array,
            )
            .unwrap();
            graph.add_factor(factor).unwrap();
        }
    }

    graph
}

/// Create a star graph: one central node connected to many leaf nodes
fn create_star_graph(num_leaves: usize, card: usize) -> FactorGraph {
    let mut graph = FactorGraph::new();

    // Add central variable
    graph.add_variable_with_card("Center".to_string(), "Domain".to_string(), card);

    // Add leaf variables and factors
    let size = card * card;
    let values: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) / size as f64).collect();
    let shape = vec![card, card];

    for i in 0..num_leaves {
        let leaf_name = format!("Leaf_{}", i);
        graph.add_variable_with_card(leaf_name.clone(), "Domain".to_string(), card);

        let array = Array::from_shape_vec(shape.clone(), values.clone())
            .unwrap()
            .into_dyn();
        let factor = Factor::new(
            format!("edge_center_{}", i),
            vec!["Center".to_string(), leaf_name],
            array,
        )
        .unwrap();
        graph.add_factor(factor).unwrap();
    }

    graph
}

/// Benchmark sum-product BP on chain graphs
fn bench_sum_product_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_product_chain");

    for card in [2, 5, 10] {
        let graph = create_chain_graph(card);
        let algorithm = SumProductAlgorithm::default();

        group.throughput(Throughput::Elements(3)); // 3 variables
        group.bench_with_input(BenchmarkId::new("card", card), &graph, |b, g| {
            b.iter(|| {
                black_box(algorithm.run(g).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark sum-product BP on grid graphs (loopy)
fn bench_sum_product_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_product_grid");

    for (rows, cols) in [(2, 2), (3, 3), (4, 4)] {
        let graph = create_grid_graph(rows, cols, 2);
        let algorithm = SumProductAlgorithm::new(20, 1e-4, 0.0); // Limited iterations for benchmark

        let num_vars = rows * cols;
        group.throughput(Throughput::Elements(num_vars as u64));
        group.bench_with_input(
            BenchmarkId::new("grid_size", format!("{}x{}", rows, cols)),
            &graph,
            |b, g| {
                b.iter(|| {
                    black_box(algorithm.run(g).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sum-product BP with damping
fn bench_sum_product_damping(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_product_damping");

    let graph = create_grid_graph(3, 3, 2);

    for damping in [0.0, 0.3, 0.5, 0.7] {
        let algorithm = SumProductAlgorithm::new(20, 1e-4, damping);

        group.bench_with_input(
            BenchmarkId::new("damping", format!("{:.1}", damping)),
            &graph,
            |b, g| {
                b.iter(|| {
                    black_box(algorithm.run(g).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark max-product algorithm
fn bench_max_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_product");

    for card in [2, 5, 10] {
        let graph = create_chain_graph(card);
        let algorithm = MaxProductAlgorithm::default();

        group.throughput(Throughput::Elements(3));
        group.bench_with_input(BenchmarkId::new("card", card), &graph, |b, g| {
            b.iter(|| {
                black_box(algorithm.run(g).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark max-product on grid graphs
fn bench_max_product_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_product_grid");

    for (rows, cols) in [(2, 2), (3, 3)] {
        let graph = create_grid_graph(rows, cols, 2);
        let algorithm = MaxProductAlgorithm::new(20, 1e-4);

        let num_vars = rows * cols;
        group.throughput(Throughput::Elements(num_vars as u64));
        group.bench_with_input(
            BenchmarkId::new("grid_size", format!("{}x{}", rows, cols)),
            &graph,
            |b, g| {
                b.iter(|| {
                    black_box(algorithm.run(g).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark convergence characteristics
fn bench_convergence_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence_iterations");

    let graph = create_grid_graph(3, 3, 2);

    for max_iter in [5, 10, 20, 50] {
        let algorithm = SumProductAlgorithm::new(max_iter, 1e-6, 0.0);

        group.bench_with_input(
            BenchmarkId::new("max_iterations", max_iter),
            &graph,
            |b, g| {
                b.iter(|| {
                    black_box(algorithm.run(g).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark star topology (tree structure)
fn bench_star_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("star_topology");

    for num_leaves in [5, 10, 20] {
        let graph = create_star_graph(num_leaves, 2);
        let algorithm = SumProductAlgorithm::default();

        group.throughput(Throughput::Elements((num_leaves + 1) as u64));
        group.bench_with_input(
            BenchmarkId::new("num_leaves", num_leaves),
            &graph,
            |b, g| {
                b.iter(|| {
                    black_box(algorithm.run(g).unwrap());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sum_product_chain,
    bench_sum_product_grid,
    bench_sum_product_damping,
    bench_max_product,
    bench_max_product_grid,
    bench_convergence_iterations,
    bench_star_topology,
);
criterion_main!(benches);
