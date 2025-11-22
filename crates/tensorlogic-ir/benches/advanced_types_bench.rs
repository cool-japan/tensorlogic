//! Benchmarks for advanced type systems (dependent, linear, refinement types)

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tensorlogic_ir::dependent::{DependentType, DependentTypeContext, DimConstraint, IndexExpr};
use tensorlogic_ir::linear::{LinearContext, LinearType, LinearityChecker};
use tensorlogic_ir::refinement::{LiquidTypeInference, RefinementContext, RefinementType};
use tensorlogic_ir::TLExpr;

fn bench_dependent_types(c: &mut Criterion) {
    c.bench_function("dependent_type_vector_creation", |b| {
        b.iter(|| {
            let n = IndexExpr::var("n");
            let _ = DependentType::vector(black_box(n), "Int");
        })
    });

    c.bench_function("dependent_type_matrix_creation", |b| {
        b.iter(|| {
            let m = IndexExpr::var("m");
            let n = IndexExpr::var("n");
            let _ = DependentType::matrix(black_box(m), black_box(n), "Float");
        })
    });

    c.bench_function("index_expr_arithmetic", |b| {
        b.iter(|| {
            let n = IndexExpr::var("n");
            let m = IndexExpr::var("m");
            let sum = IndexExpr::add(black_box(n.clone()), black_box(m.clone()));
            let product = IndexExpr::mul(black_box(n), black_box(m));
            let _ = IndexExpr::add(sum, product);
        })
    });

    c.bench_function("index_expr_simplification", |b| {
        let complex = IndexExpr::add(
            IndexExpr::mul(IndexExpr::var("n"), IndexExpr::constant(1)),
            IndexExpr::constant(0),
        );
        b.iter(|| {
            let _ = black_box(complex.clone()).simplify();
        })
    });

    c.bench_function("dim_constraint_creation", |b| {
        let n = IndexExpr::var("n");
        b.iter(|| {
            let _ = DimConstraint::lte(black_box(n.clone()), IndexExpr::constant(100));
        })
    });

    c.bench_function("dependent_context_checking", |b| {
        let mut ctx = DependentTypeContext::new();
        ctx.bind_index("n", 50);
        let n = IndexExpr::var("n");
        let constraint = DimConstraint::lte(n, IndexExpr::constant(100));
        ctx.add_constraint(constraint);

        b.iter(|| {
            let _ = black_box(&ctx).is_satisfiable();
        })
    });
}

fn bench_linear_types(c: &mut Criterion) {
    c.bench_function("linear_type_creation", |b| {
        b.iter(|| {
            let _ = LinearType::linear(black_box("Tensor"));
        })
    });

    c.bench_function("linear_context_bind", |b| {
        b.iter(|| {
            let mut ctx = LinearContext::new();
            ctx.bind("x", LinearType::linear("Tensor"));
            black_box(ctx);
        })
    });

    c.bench_function("linear_context_use", |b| {
        let mut ctx = LinearContext::new();
        ctx.bind("x", LinearType::linear("Tensor"));

        b.iter(|| {
            let mut c = ctx.clone();
            let _ = c.use_var(black_box("x"));
            black_box(c);
        })
    });

    c.bench_function("linear_context_validation", |b| {
        let mut ctx = LinearContext::new();
        ctx.bind("x", LinearType::linear("Tensor"));
        ctx.use_var("x").unwrap();

        b.iter(|| {
            let _ = black_box(&ctx).validate();
        })
    });

    c.bench_function("linear_context_merge", |b| {
        let mut ctx1 = LinearContext::new();
        ctx1.bind("x", LinearType::unrestricted("Int"));
        ctx1.use_var("x").unwrap();

        let mut ctx2 = LinearContext::new();
        ctx2.bind("x", LinearType::unrestricted("Int"));
        ctx2.use_var("x").unwrap();

        b.iter(|| {
            let _ = black_box(&ctx1).merge(black_box(&ctx2));
        })
    });

    c.bench_function("linearity_checker_full", |b| {
        b.iter(|| {
            let mut checker = LinearityChecker::new();
            checker.bind("x", LinearType::linear("Tensor"));
            checker.bind("y", LinearType::unrestricted("Int"));
            checker.use_var("x");
            checker.use_var("y");
            checker.use_var("y");
            let _ = checker.check();
            black_box(checker);
        })
    });
}

fn bench_refinement_types(c: &mut Criterion) {
    c.bench_function("refinement_type_positive_int", |b| {
        b.iter(|| {
            let _ = RefinementType::positive_int(black_box("x"));
        })
    });

    c.bench_function("refinement_type_probability", |b| {
        b.iter(|| {
            let _ = RefinementType::probability(black_box("p"));
        })
    });

    c.bench_function("refinement_type_non_empty_vec", |b| {
        b.iter(|| {
            let _ = RefinementType::non_empty_vec(black_box("v"), "Int");
        })
    });

    c.bench_function("refinement_context_bind", |b| {
        b.iter(|| {
            let mut ctx = RefinementContext::new();
            let pos_int = RefinementType::positive_int("x");
            ctx.bind("x", pos_int);
            black_box(ctx);
        })
    });

    c.bench_function("refinement_type_strengthen", |b| {
        let pos_int = RefinementType::positive_int("x");
        let constraint = TLExpr::lt(TLExpr::pred("x", vec![]), TLExpr::constant(100.0));

        b.iter(|| {
            let _ = black_box(pos_int.clone()).strengthen(black_box(constraint.clone()));
        })
    });

    c.bench_function("liquid_type_inference", |b| {
        let candidates = vec![
            TLExpr::gt(TLExpr::pred("x", vec![]), TLExpr::constant(0.0)),
            TLExpr::gte(TLExpr::pred("x", vec![]), TLExpr::constant(0.0)),
            TLExpr::lt(TLExpr::pred("x", vec![]), TLExpr::constant(100.0)),
        ];

        b.iter(|| {
            let mut inference = LiquidTypeInference::new();
            inference.add_unknown("x_refinement", black_box(candidates.clone()));
            let _ = inference.infer();
            black_box(inference);
        })
    });
}

criterion_group!(
    advanced_types,
    bench_dependent_types,
    bench_linear_types,
    bench_refinement_types
);
criterion_main!(advanced_types);
