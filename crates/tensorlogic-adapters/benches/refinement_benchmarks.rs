//! Benchmarks for refinement type operations.
//!
//! Measures performance of:
//! - Type checking with refinement predicates
//! - Subtyping checks
//! - Predicate evaluation
//! - Registry lookups

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tensorlogic_adapters::{
    RefinementContext, RefinementPredicate, RefinementRegistry, RefinementType,
};

fn benchmark_refinement_checking(c: &mut Criterion) {
    let mut group = c.benchmark_group("refinement_checking");

    // Single predicate checking
    let positive = RefinementType::new("Int").with_predicate(RefinementPredicate::GreaterThan(0.0));

    group.bench_function("check_single_predicate", |b| {
        b.iter(|| std::hint::black_box(positive.check(std::hint::black_box(42.0))))
    });

    // Multiple predicates (conjunction)
    let bounded = RefinementType::new("Int")
        .with_predicate(RefinementPredicate::GreaterThan(0.0))
        .with_predicate(RefinementPredicate::LessThan(100.0))
        .with_predicate(RefinementPredicate::Modulo {
            divisor: 2,
            remainder: 0,
        });

    group.bench_function("check_multiple_predicates", |b| {
        b.iter(|| std::hint::black_box(bounded.check(std::hint::black_box(42.0))))
    });

    // Range checking
    let probability = RefinementType::new("Float")
        .with_predicate(RefinementPredicate::Range { min: 0.0, max: 1.0 });

    group.bench_function("check_range", |b| {
        b.iter(|| std::hint::black_box(probability.check(std::hint::black_box(0.5))))
    });

    // Complex nested predicates
    let complex = RefinementType::new("Int").with_predicate(RefinementPredicate::And(vec![
        RefinementPredicate::GreaterThan(10.0),
        RefinementPredicate::LessThan(100.0),
        RefinementPredicate::Or(vec![
            RefinementPredicate::Modulo {
                divisor: 3,
                remainder: 0,
            },
            RefinementPredicate::Modulo {
                divisor: 5,
                remainder: 0,
            },
        ]),
    ]));

    group.bench_function("check_complex_nested", |b| {
        b.iter(|| std::hint::black_box(complex.check(std::hint::black_box(45.0))))
    });

    group.finish();
}

fn benchmark_subtyping(c: &mut Criterion) {
    let mut group = c.benchmark_group("subtyping");

    // Simple subtyping
    let stricter =
        RefinementType::new("Int").with_predicate(RefinementPredicate::GreaterThan(10.0));
    let looser = RefinementType::new("Int").with_predicate(RefinementPredicate::GreaterThan(5.0));

    group.bench_function("simple_subtype_check", |b| {
        b.iter(|| std::hint::black_box(stricter.is_subtype_of(&looser)))
    });

    // Range subtyping
    let narrow_range = RefinementType::new("Int").with_predicate(RefinementPredicate::Range {
        min: 10.0,
        max: 20.0,
    });
    let wide_range = RefinementType::new("Int").with_predicate(RefinementPredicate::Range {
        min: 0.0,
        max: 100.0,
    });

    group.bench_function("range_subtype_check", |b| {
        b.iter(|| std::hint::black_box(narrow_range.is_subtype_of(&wide_range)))
    });

    // Modulo subtyping
    let div_by_4 = RefinementType::new("Int").with_predicate(RefinementPredicate::Modulo {
        divisor: 4,
        remainder: 0,
    });
    let div_by_2 = RefinementType::new("Int").with_predicate(RefinementPredicate::Modulo {
        divisor: 2,
        remainder: 0,
    });

    group.bench_function("modulo_subtype_check", |b| {
        b.iter(|| std::hint::black_box(div_by_4.is_subtype_of(&div_by_2)))
    });

    // Conjunction subtyping
    let bounded = RefinementType::new("Int")
        .with_predicate(RefinementPredicate::GreaterThan(5.0))
        .with_predicate(RefinementPredicate::LessThan(10.0));
    let positive = RefinementType::new("Int").with_predicate(RefinementPredicate::GreaterThan(0.0));

    group.bench_function("conjunction_subtype_check", |b| {
        b.iter(|| std::hint::black_box(bounded.is_subtype_of(&positive)))
    });

    group.finish();
}

fn benchmark_registry(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry");

    let registry = RefinementRegistry::with_builtins();

    group.bench_function("registry_lookup_exists", |b| {
        b.iter(|| std::hint::black_box(registry.contains(std::hint::black_box("PositiveInt"))))
    });

    group.bench_function("registry_check_builtin", |b| {
        b.iter(|| {
            std::hint::black_box(registry.check(
                std::hint::black_box("PositiveInt"),
                std::hint::black_box(42.0),
            ))
        })
    });

    // Add custom types for scaling test
    let mut large_registry = RefinementRegistry::new();
    for i in 0..100 {
        large_registry.register(
            RefinementType::new("Int")
                .with_name(format!("Type{}", i))
                .with_predicate(RefinementPredicate::GreaterThan(i as f64)),
        );
    }

    group.bench_function("registry_lookup_100_types", |b| {
        b.iter(|| std::hint::black_box(large_registry.contains(std::hint::black_box("Type50"))))
    });

    group.finish();
}

fn benchmark_predicate_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicate_evaluation");

    // Simple predicates
    let gt = RefinementPredicate::GreaterThan(10.0);
    group.bench_function("eval_greater_than", |b| {
        b.iter(|| std::hint::black_box(gt.check(std::hint::black_box(15.0))))
    });

    let range = RefinementPredicate::Range {
        min: 0.0,
        max: 100.0,
    };
    group.bench_function("eval_range", |b| {
        b.iter(|| std::hint::black_box(range.check(std::hint::black_box(50.0))))
    });

    let modulo = RefinementPredicate::Modulo {
        divisor: 7,
        remainder: 3,
    };
    group.bench_function("eval_modulo", |b| {
        b.iter(|| std::hint::black_box(modulo.check(std::hint::black_box(31.0))))
    });

    // Set membership
    let in_set = RefinementPredicate::InSet(vec![1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]);
    group.bench_function("eval_in_set_small", |b| {
        b.iter(|| std::hint::black_box(in_set.check(std::hint::black_box(13.0))))
    });

    let large_set = RefinementPredicate::InSet((0..1000).map(|i| i as f64).collect());
    group.bench_function("eval_in_set_large", |b| {
        b.iter(|| std::hint::black_box(large_set.check(std::hint::black_box(500.0))))
    });

    // Custom predicate
    let is_prime = RefinementPredicate::custom("is_prime", "Check primality", |n| {
        if n < 2.0 {
            return false;
        }
        let n = n as i64;
        for i in 2..=((n as f64).sqrt() as i64) {
            if n % i == 0 {
                return false;
            }
        }
        true
    });
    group.bench_function("eval_custom_prime", |b| {
        b.iter(|| std::hint::black_box(is_prime.check(std::hint::black_box(97.0))))
    });

    group.finish();
}

fn benchmark_context_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_evaluation");

    let mut context = RefinementContext::new();
    context.set_value("x", 10.0);
    context.set_value("y", 20.0);
    context.set_value("z", 30.0);

    let dependent = RefinementPredicate::Dependent {
        variable: "x".to_string(),
        relation: tensorlogic_adapters::DependentRelation::LessThan,
    };

    group.bench_function("eval_dependent_predicate", |b| {
        b.iter(|| {
            std::hint::black_box(
                dependent
                    .check_with_context(std::hint::black_box(5.0), std::hint::black_box(&context)),
            )
        })
    });

    // Complex type with context
    let complex_type = RefinementType::new("Int")
        .with_predicate(RefinementPredicate::GreaterThan(0.0))
        .with_predicate(dependent);

    group.bench_function("check_with_context", |b| {
        b.iter(|| {
            std::hint::black_box(
                complex_type
                    .check_with_context(std::hint::black_box(5.0), std::hint::black_box(&context)),
            )
        })
    });

    group.finish();
}

fn benchmark_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    for num_predicates in [1, 5, 10, 20, 50].iter() {
        let mut refinement = RefinementType::new("Int");
        for i in 0..*num_predicates {
            refinement = refinement.with_predicate(RefinementPredicate::GreaterThan(i as f64));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(num_predicates),
            num_predicates,
            |b, _| b.iter(|| std::hint::black_box(refinement.check(std::hint::black_box(100.0)))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_refinement_checking,
    benchmark_subtyping,
    benchmark_registry,
    benchmark_predicate_evaluation,
    benchmark_context_evaluation,
    benchmark_scaling,
);
criterion_main!(benches);
