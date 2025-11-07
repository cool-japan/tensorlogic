//! Compilation performance benchmarks.
//!
//! Measures compilation time for different expression complexities and strategies.
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tensorlogic_compiler::{
    compile_to_einsum, compile_to_einsum_with_context, CompilationConfig, CompilerContext,
};
use tensorlogic_ir::{TLExpr, Term};

// ============================================================================
// Simple Expression Benchmarks
// ============================================================================

fn bench_simple_predicate(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_predicate");

    let expr = TLExpr::pred("person", vec![Term::var("x")]);

    group.bench_function("compile", |b| {
        b.iter(|| {
            let graph = compile_to_einsum(black_box(&expr)).unwrap();
            black_box(graph);
        });
    });

    group.finish();
}

fn bench_simple_and(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_and");

    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::and(p, q);

    group.bench_function("compile", |b| {
        b.iter(|| {
            let graph = compile_to_einsum(black_box(&expr)).unwrap();
            black_box(graph);
        });
    });

    group.finish();
}

fn bench_simple_or(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_or");

    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::or(p, q);

    group.bench_function("compile", |b| {
        b.iter(|| {
            let graph = compile_to_einsum(black_box(&expr)).unwrap();
            black_box(graph);
        });
    });

    group.finish();
}

// ============================================================================
// Complex Expression Benchmarks
// ============================================================================

fn bench_nested_and_or(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_and_or");

    // (P AND Q) OR (R AND S)
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let r = TLExpr::pred("R", vec![Term::var("x")]);
    let s = TLExpr::pred("S", vec![Term::var("x")]);
    let and1 = TLExpr::and(p, q);
    let and2 = TLExpr::and(r, s);
    let expr = TLExpr::or(and1, and2);

    group.bench_function("compile", |b| {
        b.iter(|| {
            let graph = compile_to_einsum(black_box(&expr)).unwrap();
            black_box(graph);
        });
    });

    group.finish();
}

fn bench_deep_nesting(c: &mut Criterion) {
    let mut group = c.benchmark_group("deep_nesting");

    for depth in [3, 5, 7, 10].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            // Create deeply nested AND expressions
            let mut expr = TLExpr::pred("P0", vec![Term::var("x")]);
            for i in 1..depth {
                let next = TLExpr::pred(format!("P{}", i), vec![Term::var("x")]);
                expr = TLExpr::and(expr, next);
            }

            b.iter(|| {
                let graph = compile_to_einsum(black_box(&expr)).unwrap();
                black_box(graph);
            });
        });
    }

    group.finish();
}

fn bench_wide_expression(c: &mut Criterion) {
    let mut group = c.benchmark_group("wide_expression");

    for width in [5, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(width), width, |b, &width| {
            // Create wide AND expression (P1 AND P2 AND ... AND Pn)
            let mut expr = TLExpr::pred("P0", vec![Term::var("x")]);
            for i in 1..width {
                let next = TLExpr::pred(format!("P{}", i), vec![Term::var("x")]);
                expr = TLExpr::and(expr, next);
            }

            b.iter(|| {
                let graph = compile_to_einsum(black_box(&expr)).unwrap();
                black_box(graph);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Quantifier Benchmarks
// ============================================================================

fn bench_exists_quantifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("exists_quantifier");

    let pred = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::exists("y", "D", pred);

    group.bench_function("compile", |b| {
        b.iter(|| {
            let mut ctx = CompilerContext::new();
            ctx.add_domain("D", 100);
            let graph = compile_to_einsum_with_context(black_box(&expr), &mut ctx).unwrap();
            black_box(graph);
        });
    });

    group.finish();
}

fn bench_nested_quantifiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_quantifiers");

    // ∃x.∃y.∃z.P(x,y,z)
    let pred = TLExpr::pred(
        "related",
        vec![Term::var("x"), Term::var("y"), Term::var("z")],
    );
    let exists_z = TLExpr::exists("z", "D", pred);
    let exists_y = TLExpr::exists("y", "D", exists_z);
    let expr = TLExpr::exists("x", "D", exists_y);

    group.bench_function("compile", |b| {
        b.iter(|| {
            let mut ctx = CompilerContext::new();
            ctx.add_domain("D", 100);
            let graph = compile_to_einsum_with_context(black_box(&expr), &mut ctx).unwrap();
            black_box(graph);
        });
    });

    group.finish();
}

// ============================================================================
// Strategy Comparison Benchmarks
// ============================================================================

fn bench_strategy_comparison_and(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_and");

    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::and(p, q);

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
        group.bench_with_input(BenchmarkId::new("compile", name), &config, |b, config| {
            b.iter(|| {
                let mut ctx = CompilerContext::new();
                ctx.config = config.clone();
                let graph = compile_to_einsum_with_context(black_box(&expr), &mut ctx).unwrap();
                black_box(graph);
            });
        });
    }

    group.finish();
}

fn bench_strategy_comparison_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_complex");

    // Complex expression: ((P AND Q) OR (R AND S)) AND NOT(T)
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let r = TLExpr::pred("R", vec![Term::var("x")]);
    let s = TLExpr::pred("S", vec![Term::var("x")]);
    let t = TLExpr::pred("T", vec![Term::var("x")]);
    let and1 = TLExpr::and(p, q);
    let and2 = TLExpr::and(r, s);
    let or = TLExpr::or(and1, and2);
    let not_t = TLExpr::negate(t);
    let expr = TLExpr::and(or, not_t);

    let strategies = vec![
        (
            "soft_differentiable",
            CompilationConfig::soft_differentiable(),
        ),
        ("hard_boolean", CompilationConfig::hard_boolean()),
        ("fuzzy_godel", CompilationConfig::fuzzy_godel()),
    ];

    for (name, config) in strategies {
        group.bench_with_input(BenchmarkId::new("compile", name), &config, |b, config| {
            b.iter(|| {
                let mut ctx = CompilerContext::new();
                ctx.config = config.clone();
                let graph = compile_to_einsum_with_context(black_box(&expr), &mut ctx).unwrap();
                black_box(graph);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Multi-arity Predicate Benchmarks
// ============================================================================

fn bench_multi_arity_predicates(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_arity");

    for arity in [2, 3, 4, 5].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(arity), arity, |b, &arity| {
            // Create predicate with given arity
            let args: Vec<Term> = (0..arity).map(|i| Term::var(format!("x{}", i))).collect();
            let expr = TLExpr::pred("relation", args);

            b.iter(|| {
                let graph = compile_to_einsum(black_box(&expr)).unwrap();
                black_box(graph);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Negation Benchmarks
// ============================================================================

fn bench_negation(c: &mut Criterion) {
    let mut group = c.benchmark_group("negation");

    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let expr = TLExpr::negate(p);

    group.bench_function("compile", |b| {
        b.iter(|| {
            let graph = compile_to_einsum(black_box(&expr)).unwrap();
            black_box(graph);
        });
    });

    group.finish();
}

fn bench_double_negation(c: &mut Criterion) {
    let mut group = c.benchmark_group("double_negation");

    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let not_p = TLExpr::negate(p);
    let expr = TLExpr::negate(not_p);

    group.bench_function("compile", |b| {
        b.iter(|| {
            let graph = compile_to_einsum(black_box(&expr)).unwrap();
            black_box(graph);
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group!(
    simple_benches,
    bench_simple_predicate,
    bench_simple_and,
    bench_simple_or,
    bench_negation,
    bench_double_negation
);

criterion_group!(
    complex_benches,
    bench_nested_and_or,
    bench_deep_nesting,
    bench_wide_expression
);

criterion_group!(
    quantifier_benches,
    bench_exists_quantifier,
    bench_nested_quantifiers
);

criterion_group!(
    strategy_benches,
    bench_strategy_comparison_and,
    bench_strategy_comparison_complex
);

criterion_group!(multi_arity_benches, bench_multi_arity_predicates);

criterion_main!(
    simple_benches,
    complex_benches,
    quantifier_benches,
    strategy_benches,
    multi_arity_benches
);
