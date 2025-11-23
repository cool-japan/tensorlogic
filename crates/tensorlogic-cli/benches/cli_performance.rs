//! CLI Performance Benchmarks
//!
//! Benchmarks for tensorlogic-cli components to measure and track performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use tensorlogic_cli::{analysis, parser, CompilationContext};
use tensorlogic_compiler::{compile_to_einsum_with_context, CompilationConfig};

/// Benchmark expression parsing
fn bench_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser");

    let expressions = vec![
        ("simple_predicate", "knows(x, y)"),
        ("and_operation", "knows(x, y) AND likes(y, z)"),
        ("or_operation", "person(x) OR robot(x)"),
        (
            "complex_nested",
            "(knows(x, y) AND likes(y, z)) OR (friend(x, z) AND NOT enemy(x, z))",
        ),
        ("quantifier", "EXISTS x IN Person. knows(x, alice)"),
        (
            "nested_quantifiers",
            "EXISTS x IN Person. FORALL y IN Person. knows(x, y)",
        ),
        ("arithmetic", "age(x) + 10 * 2"),
        ("comparison", "age(x) > 18 AND score(x) <= 100"),
        ("conditional", "IF age(x) >= 18 THEN adult(x) ELSE child(x)"),
        ("implication", "knows(x, y) -> likes(x, y)"),
    ];

    for (name, expr) in expressions {
        group.bench_with_input(BenchmarkId::new("parse", name), expr, |b, expr| {
            b.iter(|| parser::parse_expression(black_box(expr)).unwrap());
        });
    }

    group.finish();
}

/// Benchmark compilation
fn bench_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation");

    let test_cases = vec![
        ("simple_predicate", "knows(x, y)"),
        ("and_chain", "a(x) AND b(x) AND c(x) AND d(x)"),
        ("or_chain", "a(x) OR b(x) OR c(x) OR d(x)"),
        ("mixed_logic", "(a(x) AND b(x)) OR (c(x) AND d(x))"),
        ("deep_nesting", "((((a AND b) OR c) AND d) OR e)"),
        ("quantifier", "EXISTS x IN D. knows(x, y)"),
        (
            "multi_quantifier",
            "EXISTS x IN D. EXISTS y IN D. knows(x, y)",
        ),
    ];

    for (name, expr_str) in test_cases {
        let expr = parser::parse_expression(expr_str).unwrap();

        group.bench_with_input(BenchmarkId::new("compile", name), &expr, |b, expr| {
            b.iter(|| {
                let config = CompilationConfig::soft_differentiable();
                let mut ctx = CompilationContext::with_config(config);
                ctx.add_domain("D", 100);
                compile_to_einsum_with_context(black_box(expr), &mut ctx).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark different compilation strategies
fn bench_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategies");

    let expr = parser::parse_expression("a(x) AND b(y) OR c(z)").unwrap();

    let strategies = vec![
        (
            "soft_differentiable",
            CompilationConfig::soft_differentiable(),
        ),
        ("hard_boolean", CompilationConfig::hard_boolean()),
        ("fuzzy_godel", CompilationConfig::fuzzy_godel()),
        ("fuzzy_product", CompilationConfig::fuzzy_product()),
        ("fuzzy_lukasiewicz", CompilationConfig::fuzzy_lukasiewicz()),
        ("probabilistic", CompilationConfig::probabilistic()),
    ];

    for (name, config) in strategies {
        group.bench_with_input(BenchmarkId::new("strategy", name), &config, |b, config| {
            b.iter(|| {
                let mut ctx = CompilationContext::with_config(config.clone());
                ctx.add_domain("D", 100);
                compile_to_einsum_with_context(black_box(&expr), &mut ctx).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark graph analysis
fn bench_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis");

    // Pre-compile graphs of varying complexity
    let test_cases = vec![
        ("simple", "knows(x, y)"),
        ("medium", "knows(x, y) AND likes(y, z)"),
        (
            "complex",
            "(knows(x, y) AND likes(y, z)) OR (friend(x, z) AND NOT enemy(x, z))",
        ),
        ("quantifier", "EXISTS x IN D. FORALL y IN D. knows(x, y)"),
    ];

    for (name, expr_str) in test_cases {
        let expr = parser::parse_expression(expr_str).unwrap();
        let config = CompilationConfig::soft_differentiable();
        let mut ctx = CompilationContext::with_config(config);
        ctx.add_domain("D", 100);
        let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

        group.bench_with_input(BenchmarkId::new("analyze", name), &graph, |b, graph| {
            b.iter(|| analysis::GraphMetrics::analyze(black_box(graph)));
        });
    }

    group.finish();
}

/// Benchmark parse -> compile -> analyze pipeline
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    let expressions = vec![
        ("simple", "knows(x, y)"),
        ("medium", "(knows(x, y) AND likes(y, z)) OR friend(x, z)"),
        (
            "complex",
            "EXISTS x IN D. (knows(x, y) AND likes(y, z) OR friend(x, z))",
        ),
    ];

    for (name, expr_str) in expressions {
        group.bench_with_input(
            BenchmarkId::new("pipeline", name),
            expr_str,
            |b, expr_str| {
                b.iter(|| {
                    // Parse
                    let expr = parser::parse_expression(black_box(expr_str)).unwrap();

                    // Compile
                    let config = CompilationConfig::soft_differentiable();
                    let mut ctx = CompilationContext::with_config(config);
                    ctx.add_domain("D", 100);
                    let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

                    // Analyze
                    let _metrics = analysis::GraphMetrics::analyze(&graph);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark domain sizes impact
fn bench_domain_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("domain_sizes");

    let expr = parser::parse_expression("EXISTS x IN D. knows(x, y)").unwrap();

    for size in [10, 50, 100, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("domain_size", size), size, |b, size| {
            b.iter(|| {
                let config = CompilationConfig::soft_differentiable();
                let mut ctx = CompilationContext::with_config(config);
                ctx.add_domain("D", *size);
                compile_to_einsum_with_context(black_box(&expr), &mut ctx).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark expression complexity scaling
fn bench_expression_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("expression_scaling");

    // Generate expressions of increasing complexity
    for chain_length in [2, 4, 8, 16].iter() {
        let expr_parts: Vec<String> = (0..*chain_length).map(|i| format!("p{}(x)", i)).collect();
        let expr_str = expr_parts.join(" AND ");
        let expr = parser::parse_expression(&expr_str).unwrap();

        group.bench_with_input(
            BenchmarkId::new("and_chain", chain_length),
            &expr,
            |b, expr| {
                b.iter(|| {
                    let config = CompilationConfig::soft_differentiable();
                    let mut ctx = CompilationContext::with_config(config);
                    ctx.add_domain("D", 100);
                    compile_to_einsum_with_context(black_box(expr), &mut ctx).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_parser,
    bench_compilation,
    bench_strategies,
    bench_analysis,
    bench_full_pipeline,
    bench_domain_sizes,
    bench_expression_scaling
);
criterion_main!(benches);
