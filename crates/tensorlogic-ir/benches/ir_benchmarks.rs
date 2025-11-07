//! Performance benchmarks for TensorLogic IR core operations
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensorlogic_ir::{DomainInfo, DomainRegistry, EinsumGraph, EinsumNode, TLExpr, Term};

// ===== Expression Construction Benchmarks =====

fn bench_expr_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_construction");

    group.bench_function("simple_predicate", |b| {
        b.iter(|| {
            let _expr = TLExpr::pred(black_box("Person"), vec![Term::var(black_box("x"))]);
        });
    });

    group.bench_function("binary_logical_and", |b| {
        b.iter(|| {
            let p1 = TLExpr::pred("P", vec![Term::var("x")]);
            let p2 = TLExpr::pred("Q", vec![Term::var("y")]);
            let _expr = TLExpr::and(black_box(p1), black_box(p2));
        });
    });

    group.bench_function("nested_expression", |b| {
        b.iter(|| {
            let p1 = TLExpr::pred("P", vec![Term::var("x")]);
            let p2 = TLExpr::pred("Q", vec![Term::var("y")]);
            let p3 = TLExpr::pred("R", vec![Term::var("z")]);
            let and1 = TLExpr::and(p1, p2);
            let _expr = TLExpr::and(black_box(and1), black_box(p3));
        });
    });

    group.bench_function("quantified_expression", |b| {
        b.iter(|| {
            let pred = TLExpr::pred("P", vec![Term::var("x")]);
            let _expr = TLExpr::exists(black_box("x"), black_box("Domain"), black_box(pred));
        });
    });

    group.bench_function("arithmetic_expression", |b| {
        b.iter(|| {
            let x = TLExpr::pred("x", vec![Term::var("i")]);
            let y = TLExpr::pred("y", vec![Term::var("i")]);
            let _expr = TLExpr::add(black_box(x), black_box(y));
        });
    });

    group.finish();
}

// ===== Free Variable Analysis Benchmarks =====

fn bench_free_vars(c: &mut Criterion) {
    let mut group = c.benchmark_group("free_vars_analysis");

    // Simple predicate
    let simple = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    group.bench_function("simple_predicate", |b| {
        b.iter(|| {
            let _vars = black_box(&simple).free_vars();
        });
    });

    // Nested AND
    let nested = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::and(
            TLExpr::pred("Q", vec![Term::var("y")]),
            TLExpr::pred("R", vec![Term::var("z")]),
        ),
    );
    group.bench_function("nested_and", |b| {
        b.iter(|| {
            let _vars = black_box(&nested).free_vars();
        });
    });

    // Quantified expression
    let quantified = TLExpr::exists(
        "x",
        "D",
        TLExpr::forall(
            "y",
            "D",
            TLExpr::pred("R", vec![Term::var("x"), Term::var("y"), Term::var("z")]),
        ),
    );
    group.bench_function("quantified_expression", |b| {
        b.iter(|| {
            let _vars = black_box(&quantified).free_vars();
        });
    });

    // Deep nesting (10 levels)
    let mut deep = TLExpr::pred("P", vec![Term::var("x0")]);
    for i in 1..10 {
        let pred = TLExpr::pred("P", vec![Term::var(format!("x{}", i))]);
        deep = TLExpr::and(deep, pred);
    }
    group.bench_function("deep_nesting_10_levels", |b| {
        b.iter(|| {
            let _vars = black_box(&deep).free_vars();
        });
    });

    group.finish();
}

// ===== Arity Validation Benchmarks =====

fn bench_arity_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("arity_validation");

    // Valid expression
    let valid = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]),
        TLExpr::pred("P", vec![Term::var("a"), Term::var("b")]),
    );
    group.bench_function("valid_arity", |b| {
        b.iter(|| {
            let _result = black_box(&valid).validate_arity();
        });
    });

    // Invalid expression (catches error)
    let invalid = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]),
        TLExpr::pred("P", vec![Term::var("z")]),
    );
    group.bench_function("invalid_arity", |b| {
        b.iter(|| {
            let _result = black_box(&invalid).validate_arity();
        });
    });

    // Complex nested expression
    let complex = TLExpr::and(
        TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::pred("Q", vec![Term::var("y"), Term::var("z")]),
        ),
        TLExpr::and(
            TLExpr::pred("P", vec![Term::var("a")]),
            TLExpr::pred("Q", vec![Term::var("b"), Term::var("c")]),
        ),
    );
    group.bench_function("complex_nested", |b| {
        b.iter(|| {
            let _result = black_box(&complex).validate_arity();
        });
    });

    group.finish();
}

// ===== Graph Construction Benchmarks =====

fn bench_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction");

    group.bench_function("add_tensor", |b| {
        b.iter(|| {
            let mut graph = EinsumGraph::new();
            let _idx = graph.add_tensor(black_box("tensor"));
        });
    });

    group.bench_function("add_node", |bencher| {
        let mut graph = EinsumGraph::new();
        let a = graph.add_tensor("a");
        let b_tensor = graph.add_tensor("b");
        let c = graph.add_tensor("c");

        bencher.iter_with_setup(
            || graph.clone(),
            |mut g| {
                let _idx = g.add_node(EinsumNode::elem_binary("add", a, b_tensor, c));
            },
        );
    });

    // Build a small graph
    group.bench_function("build_small_graph", |b| {
        b.iter(|| {
            let mut graph = EinsumGraph::new();
            let a = graph.add_tensor("a");
            let b = graph.add_tensor("b");
            let c = graph.add_tensor("c");

            graph
                .add_node(EinsumNode::elem_binary("add", a, b, c))
                .unwrap();
            graph.add_output(c).unwrap();
        });
    });

    // Build a medium graph (10 layers)
    group.bench_function("build_medium_graph_10_layers", |b| {
        b.iter(|| {
            let mut graph = EinsumGraph::new();
            let mut current = graph.add_tensor("input");

            for i in 0..10 {
                let next = graph.add_tensor(format!("layer_{}", i));
                graph
                    .add_node(EinsumNode::elem_unary("relu", current, next))
                    .unwrap();
                current = next;
            }

            graph.add_output(current).unwrap();
        });
    });

    group.finish();
}

// ===== Graph Validation Benchmarks =====

fn bench_graph_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_validation");

    // Small valid graph
    let mut small_graph = EinsumGraph::new();
    let a = small_graph.add_tensor("a");
    let b = small_graph.add_tensor("b");
    let c = small_graph.add_tensor("c");
    small_graph
        .add_node(EinsumNode::elem_binary("add", a, b, c))
        .unwrap();
    small_graph.add_output(c).unwrap();

    group.bench_function("small_graph", |b| {
        b.iter(|| {
            let _result = black_box(&small_graph).validate();
        });
    });

    // Medium graph (10 layers)
    let mut medium_graph = EinsumGraph::new();
    let mut current = medium_graph.add_tensor("input");
    for i in 0..10 {
        let next = medium_graph.add_tensor(format!("layer_{}", i));
        medium_graph
            .add_node(EinsumNode::elem_unary("relu", current, next))
            .unwrap();
        current = next;
    }
    medium_graph.add_output(current).unwrap();

    group.bench_function("medium_graph_10_layers", |b| {
        b.iter(|| {
            let _result = black_box(&medium_graph).validate();
        });
    });

    // Large graph (50 layers)
    let mut large_graph = EinsumGraph::new();
    let mut current = large_graph.add_tensor("input");
    for i in 0..50 {
        let next = large_graph.add_tensor(format!("layer_{}", i));
        large_graph
            .add_node(EinsumNode::elem_unary("relu", current, next))
            .unwrap();
        current = next;
    }
    large_graph.add_output(current).unwrap();

    group.bench_function("large_graph_50_layers", |b| {
        b.iter(|| {
            let _result = black_box(&large_graph).validate();
        });
    });

    group.finish();
}

// ===== Serialization Benchmarks =====

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    // Expression serialization
    let expr = TLExpr::forall(
        "x",
        "Person",
        TLExpr::imply(
            TLExpr::pred("Person", vec![Term::var("x")]),
            TLExpr::pred("Mortal", vec![Term::var("x")]),
        ),
    );

    group.bench_function("expr_to_json", |b| {
        b.iter(|| {
            let _json = serde_json::to_string(black_box(&expr)).unwrap();
        });
    });

    let json = serde_json::to_string(&expr).unwrap();
    group.bench_function("expr_from_json", |b| {
        b.iter(|| {
            let _expr: TLExpr = serde_json::from_str(black_box(&json)).unwrap();
        });
    });

    group.bench_function("expr_to_binary", |b| {
        b.iter(|| {
            let _binary =
                bincode::serde::encode_to_vec(black_box(&expr), bincode::config::standard())
                    .unwrap();
        });
    });

    let binary = bincode::serde::encode_to_vec(&expr, bincode::config::standard()).unwrap();
    group.bench_function("expr_from_binary", |b| {
        b.iter(|| {
            let (_expr, _): (TLExpr, usize) =
                bincode::serde::decode_from_slice(black_box(&binary), bincode::config::standard())
                    .unwrap();
        });
    });

    // Graph serialization
    let mut graph = EinsumGraph::new();
    let mut current = graph.add_tensor("input");
    for i in 0..10 {
        let next = graph.add_tensor(format!("layer_{}", i));
        graph
            .add_node(EinsumNode::elem_unary("relu", current, next))
            .unwrap();
        current = next;
    }
    graph.add_output(current).unwrap();

    group.bench_function("graph_to_json", |b| {
        b.iter(|| {
            let _json = serde_json::to_string(black_box(&graph)).unwrap();
        });
    });

    let graph_json = serde_json::to_string(&graph).unwrap();
    group.bench_function("graph_from_json", |b| {
        b.iter(|| {
            let _graph: EinsumGraph = serde_json::from_str(black_box(&graph_json)).unwrap();
        });
    });

    group.bench_function("graph_to_binary", |b| {
        b.iter(|| {
            let _binary =
                bincode::serde::encode_to_vec(black_box(&graph), bincode::config::standard())
                    .unwrap();
        });
    });

    let graph_binary = bincode::serde::encode_to_vec(&graph, bincode::config::standard()).unwrap();
    group.bench_function("graph_from_binary", |b| {
        b.iter(|| {
            let (_graph, _): (EinsumGraph, usize) = bincode::serde::decode_from_slice(
                black_box(&graph_binary),
                bincode::config::standard(),
            )
            .unwrap();
        });
    });

    group.finish();
}

// ===== Domain Operations Benchmarks =====

fn bench_domain_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("domain_operations");

    group.bench_function("create_domain_registry", |b| {
        b.iter(|| {
            let _registry = DomainRegistry::with_builtins();
        });
    });

    let mut registry = DomainRegistry::with_builtins();
    group.bench_function("register_domain", |b| {
        b.iter(|| {
            let mut reg = registry.clone();
            let _result = reg.register(DomainInfo::finite(black_box("TestDomain"), 100));
        });
    });

    registry
        .register(DomainInfo::finite("TestDomain", 100))
        .unwrap();
    group.bench_function("lookup_domain", |b| {
        b.iter(|| {
            let _domain = black_box(&registry).get(black_box("TestDomain"));
        });
    });

    let expr = TLExpr::exists("x", "Int", TLExpr::pred("P", vec![Term::var("x")]));
    group.bench_function("validate_domains", |b| {
        b.iter(|| {
            let _result = black_box(&expr).validate_domains(black_box(&registry));
        });
    });

    group.finish();
}

// ===== Cloning Benchmarks =====

fn bench_cloning(c: &mut Criterion) {
    let mut group = c.benchmark_group("cloning");

    // Simple expression
    let simple = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    group.bench_function("simple_expr", |b| {
        b.iter(|| {
            let _cloned = black_box(&simple).clone();
        });
    });

    // Complex expression
    let complex = TLExpr::forall(
        "x",
        "Domain",
        TLExpr::exists(
            "y",
            "Domain",
            TLExpr::imply(
                TLExpr::and(
                    TLExpr::pred("P", vec![Term::var("x")]),
                    TLExpr::pred("Q", vec![Term::var("y")]),
                ),
                TLExpr::pred("R", vec![Term::var("x"), Term::var("y")]),
            ),
        ),
    );
    group.bench_function("complex_expr", |b| {
        b.iter(|| {
            let _cloned = black_box(&complex).clone();
        });
    });

    // Graph
    let mut graph = EinsumGraph::new();
    let mut current = graph.add_tensor("input");
    for i in 0..20 {
        let next = graph.add_tensor(format!("layer_{}", i));
        graph
            .add_node(EinsumNode::elem_unary("relu", current, next))
            .unwrap();
        current = next;
    }
    graph.add_output(current).unwrap();

    group.bench_function("graph_20_layers", |b| {
        b.iter(|| {
            let _cloned = black_box(&graph).clone();
        });
    });

    group.finish();
}

// ===== Throughput Benchmarks =====

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Expressions per second
    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("create_exprs", size), &size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    let _expr = TLExpr::pred(
                        black_box("P"),
                        vec![Term::var(black_box(&format!("x{}", i)))],
                    );
                }
            });
        });
    }

    // Graph nodes per second
    for size in [10, 100, 500] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("add_graph_nodes", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut graph = EinsumGraph::new();
                    let mut current = graph.add_tensor("input");
                    for i in 0..size {
                        let next = graph.add_tensor(format!("layer_{}", i));
                        graph
                            .add_node(EinsumNode::elem_unary("relu", current, next))
                            .unwrap();
                        current = next;
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_expr_construction,
    bench_free_vars,
    bench_arity_validation,
    bench_graph_construction,
    bench_graph_validation,
    bench_serialization,
    bench_domain_operations,
    bench_cloning,
    bench_throughput
);

criterion_main!(benches);
