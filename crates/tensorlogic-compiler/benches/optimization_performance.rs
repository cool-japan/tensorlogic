//! Optimization pipeline performance benchmarks.
//!
//! Measures the performance of individual optimization passes and the full pipeline
//! across different expression complexities.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tensorlogic_compiler::optimize::{
    analyze_complexity, eliminate_dead_code, fold_constants, optimize_distributivity,
    optimize_negations, optimize_quantifiers, reduce_strength, simplify_algebraic,
    OptimizationPipeline, PipelineConfig,
};
use tensorlogic_ir::{TLExpr, Term};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simple expression for benchmarking
fn create_simple_expr() -> TLExpr {
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    TLExpr::add(x, TLExpr::Constant(0.0))
}

/// Create a moderately complex expression
fn create_moderate_expr() -> TLExpr {
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    TLExpr::negate(TLExpr::negate(TLExpr::add(
        TLExpr::mul(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
        TLExpr::mul(x, TLExpr::Constant(1.0)),
    )))
}

/// Create a complex expression with multiple optimization opportunities
fn create_complex_expr() -> TLExpr {
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let a = TLExpr::pred("a", vec![Term::var("i")]);
    let b = TLExpr::pred("b", vec![Term::var("i")]);
    let c = TLExpr::pred("c", vec![Term::var("i")]);

    // Complex expression benefiting from all passes:
    // if true then (NOT(NOT(x^2 + 0)) AND (a*b + a*c)) else FALSE
    TLExpr::IfThenElse {
        condition: Box::new(TLExpr::Constant(1.0)),
        then_branch: Box::new(TLExpr::and(
            TLExpr::negate(TLExpr::negate(TLExpr::add(
                TLExpr::pow(x, TLExpr::Constant(2.0)),
                TLExpr::Constant(0.0),
            ))),
            TLExpr::add(
                TLExpr::mul(a.clone(), b.clone()),
                TLExpr::mul(a.clone(), c.clone()),
            ),
        )),
        else_branch: Box::new(TLExpr::Constant(0.0)),
    }
}

/// Create a deep nested expression
fn create_deep_nested_expr(depth: usize) -> TLExpr {
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let mut expr = x;
    for _ in 0..depth {
        expr = TLExpr::negate(TLExpr::negate(TLExpr::add(expr, TLExpr::Constant(0.0))));
    }
    expr
}

/// Create an expression with quantifiers for optimization
fn create_quantifier_expr() -> TLExpr {
    let a = TLExpr::pred("a", vec![Term::var("i")]);
    let p_x = TLExpr::pred("p", vec![Term::var("x")]);
    TLExpr::Exists {
        var: "x".to_string(),
        domain: "D".to_string(),
        body: Box::new(TLExpr::add(a, p_x)),
    }
}

// ============================================================================
// Individual Pass Benchmarks
// ============================================================================

fn bench_negation_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("negation_opt");

    // Simple double negation
    let simple = TLExpr::negate(TLExpr::negate(TLExpr::pred("x", vec![Term::var("i")])));
    group.bench_function("double_negation", |b| {
        b.iter(|| {
            let (result, _) = optimize_negations(black_box(&simple));
            black_box(result);
        });
    });

    // De Morgan's law
    let demorgan = TLExpr::negate(TLExpr::and(
        TLExpr::pred("p", vec![Term::var("i")]),
        TLExpr::pred("q", vec![Term::var("i")]),
    ));
    group.bench_function("demorgans_law", |b| {
        b.iter(|| {
            let (result, _) = optimize_negations(black_box(&demorgan));
            black_box(result);
        });
    });

    // Complex nested negations
    let complex = create_moderate_expr();
    group.bench_function("complex", |b| {
        b.iter(|| {
            let (result, _) = optimize_negations(black_box(&complex));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_constant_folding(c: &mut Criterion) {
    let mut group = c.benchmark_group("constant_folding");

    // Simple constant expression
    let simple = TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0));
    group.bench_function("simple", |b| {
        b.iter(|| {
            let (result, _) = fold_constants(black_box(&simple));
            black_box(result);
        });
    });

    // Nested constants
    let nested = TLExpr::mul(
        TLExpr::add(TLExpr::Constant(2.0), TLExpr::Constant(3.0)),
        TLExpr::sub(TLExpr::Constant(10.0), TLExpr::Constant(4.0)),
    );
    group.bench_function("nested", |b| {
        b.iter(|| {
            let (result, _) = fold_constants(black_box(&nested));
            black_box(result);
        });
    });

    // Mixed with variables
    let mixed = create_moderate_expr();
    group.bench_function("mixed", |b| {
        b.iter(|| {
            let (result, _) = fold_constants(black_box(&mixed));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_algebraic_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("algebraic_simp");

    // Identity elimination (x + 0)
    let identity = create_simple_expr();
    group.bench_function("identity", |b| {
        b.iter(|| {
            let (result, _) = simplify_algebraic(black_box(&identity));
            black_box(result);
        });
    });

    // Multiple identities
    let multiple = TLExpr::mul(
        TLExpr::add(
            TLExpr::pred("x", vec![Term::var("i")]),
            TLExpr::Constant(0.0),
        ),
        TLExpr::Constant(1.0),
    );
    group.bench_function("multiple", |b| {
        b.iter(|| {
            let (result, _) = simplify_algebraic(black_box(&multiple));
            black_box(result);
        });
    });

    // Annihilation (x * 0)
    let annihilation = TLExpr::mul(
        TLExpr::pred("x", vec![Term::var("i")]),
        TLExpr::Constant(0.0),
    );
    group.bench_function("annihilation", |b| {
        b.iter(|| {
            let (result, _) = simplify_algebraic(black_box(&annihilation));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_strength_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("strength_reduction");

    // Power to multiplication (x^2 → x*x)
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let power = TLExpr::pow(x.clone(), TLExpr::Constant(2.0));
    group.bench_function("power_to_mul", |b| {
        b.iter(|| {
            let (result, _) = reduce_strength(black_box(&power));
            black_box(result);
        });
    });

    // exp(log(x)) → x
    let exp_log = TLExpr::exp(TLExpr::log(x.clone()));
    group.bench_function("exp_log", |b| {
        b.iter(|| {
            let (result, _) = reduce_strength(black_box(&exp_log));
            black_box(result);
        });
    });

    // Complex expression
    let complex = TLExpr::add(
        TLExpr::pow(x.clone(), TLExpr::Constant(2.0)),
        TLExpr::exp(TLExpr::log(x)),
    );
    group.bench_function("complex", |b| {
        b.iter(|| {
            let (result, _) = reduce_strength(black_box(&complex));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_distributivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributivity");

    let a = TLExpr::pred("a", vec![Term::var("i")]);
    let b = TLExpr::pred("b", vec![Term::var("i")]);
    let c = TLExpr::pred("c", vec![Term::var("i")]);

    // Simple factoring: a*b + a*c → a*(b+c)
    let simple = TLExpr::add(
        TLExpr::mul(a.clone(), b.clone()),
        TLExpr::mul(a.clone(), c.clone()),
    );
    group.bench_function("simple_factor", |b| {
        b.iter(|| {
            let (result, _) = optimize_distributivity(black_box(&simple));
            black_box(result);
        });
    });

    // Logical factoring: (a OR b) AND (a OR c)
    let logical = TLExpr::and(
        TLExpr::or(a.clone(), b.clone()),
        TLExpr::or(a.clone(), c.clone()),
    );
    group.bench_function("logical_factor", |b| {
        b.iter(|| {
            let (result, _) = optimize_distributivity(black_box(&logical));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_quantifier_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantifier_opt");

    // Simple loop-invariant hoisting
    let simple = create_quantifier_expr();
    group.bench_function("hoist_invariant", |b| {
        b.iter(|| {
            let (result, _) = optimize_quantifiers(black_box(&simple));
            black_box(result);
        });
    });

    // Nested quantifiers
    let nested = TLExpr::Exists {
        var: "x".to_string(),
        domain: "D1".to_string(),
        body: Box::new(TLExpr::Exists {
            var: "y".to_string(),
            domain: "D2".to_string(),
            body: Box::new(TLExpr::pred("p", vec![Term::var("x"), Term::var("y")])),
        }),
    };
    group.bench_function("nested", |b| {
        b.iter(|| {
            let (result, _) = optimize_quantifiers(black_box(&nested));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_dead_code_elimination(c: &mut Criterion) {
    let mut group = c.benchmark_group("dead_code");

    // Constant condition: if true then A else B → A
    let a = TLExpr::pred("a", vec![Term::var("i")]);
    let b = TLExpr::pred("b", vec![Term::var("i")]);
    let const_cond = TLExpr::IfThenElse {
        condition: Box::new(TLExpr::Constant(1.0)),
        then_branch: Box::new(a.clone()),
        else_branch: Box::new(b),
    };
    group.bench_function("const_branch", |b| {
        b.iter(|| {
            let (result, _) = eliminate_dead_code(black_box(&const_cond));
            black_box(result);
        });
    });

    // Short-circuit: AND(false, x) → false
    let short_circuit = TLExpr::and(TLExpr::Constant(0.0), a.clone());
    group.bench_function("short_circuit", |b| {
        b.iter(|| {
            let (result, _) = eliminate_dead_code(black_box(&short_circuit));
            black_box(result);
        });
    });

    group.finish();
}

// ============================================================================
// Full Pipeline Benchmarks
// ============================================================================

fn bench_pipeline_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_simple");

    let expr = create_simple_expr();
    let pipeline = OptimizationPipeline::new();

    group.bench_function("optimize", |b| {
        b.iter(|| {
            let (result, _) = pipeline.optimize(black_box(&expr));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_pipeline_moderate(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_moderate");

    let expr = create_moderate_expr();
    let pipeline = OptimizationPipeline::new();

    group.bench_function("optimize", |b| {
        b.iter(|| {
            let (result, _) = pipeline.optimize(black_box(&expr));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_pipeline_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_complex");

    let expr = create_complex_expr();
    let pipeline = OptimizationPipeline::new();

    group.bench_function("optimize", |b| {
        b.iter(|| {
            let (result, _) = pipeline.optimize(black_box(&expr));
            black_box(result);
        });
    });

    group.finish();
}

fn bench_pipeline_nested_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_nested_depth");
    let pipeline = OptimizationPipeline::new();

    for depth in [1, 2, 4, 8].iter() {
        let expr = create_deep_nested_expr(*depth);
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, _| {
            b.iter(|| {
                let (result, _) = pipeline.optimize(black_box(&expr));
                black_box(result);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Configuration Comparison Benchmarks
// ============================================================================

fn bench_pipeline_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_configs");
    let expr = create_complex_expr();

    // Default configuration
    let default = OptimizationPipeline::new();
    group.bench_function("default", |b| {
        b.iter(|| {
            let (result, _) = default.optimize(black_box(&expr));
            black_box(result);
        });
    });

    // Aggressive configuration
    let aggressive = OptimizationPipeline::with_config(PipelineConfig::aggressive());
    group.bench_function("aggressive", |b| {
        b.iter(|| {
            let (result, _) = aggressive.optimize(black_box(&expr));
            black_box(result);
        });
    });

    // Minimal configuration (only constant folding)
    let minimal = OptimizationPipeline::with_config(PipelineConfig::constant_folding_only());
    group.bench_function("minimal", |b| {
        b.iter(|| {
            let (result, _) = minimal.optimize(black_box(&expr));
            black_box(result);
        });
    });

    group.finish();
}

// ============================================================================
// Complexity Analysis Benchmarks
// ============================================================================

fn bench_complexity_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_analysis");

    let simple = create_simple_expr();
    group.bench_function("simple", |b| {
        b.iter(|| {
            let result = analyze_complexity(black_box(&simple));
            black_box(result);
        });
    });

    let moderate = create_moderate_expr();
    group.bench_function("moderate", |b| {
        b.iter(|| {
            let result = analyze_complexity(black_box(&moderate));
            black_box(result);
        });
    });

    let complex = create_complex_expr();
    group.bench_function("complex", |b| {
        b.iter(|| {
            let result = analyze_complexity(black_box(&complex));
            black_box(result);
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group!(
    individual_passes,
    bench_negation_optimization,
    bench_constant_folding,
    bench_algebraic_simplification,
    bench_strength_reduction,
    bench_distributivity,
    bench_quantifier_optimization,
    bench_dead_code_elimination,
);

criterion_group!(
    full_pipeline,
    bench_pipeline_simple,
    bench_pipeline_moderate,
    bench_pipeline_complex,
    bench_pipeline_nested_depth,
);

criterion_group!(configurations, bench_pipeline_configs,);

criterion_group!(analysis, bench_complexity_analysis,);

criterion_main!(individual_passes, full_pipeline, configurations, analysis,);
