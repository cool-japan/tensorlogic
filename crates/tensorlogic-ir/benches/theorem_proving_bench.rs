//! Benchmarks for Automated Theorem Proving Modules
//!
//! This benchmark suite measures the performance of:
//! - Unification algorithms
//! - Resolution-based theorem proving
//! - Sequent calculus proof search
//! - Constraint logic programming

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use tensorlogic_ir::{
    anti_unify_terms, lgg_terms, unify_term_list, unify_terms, Clause, Literal, ResolutionProver,
    ResolutionStrategy, Sequent, TLExpr, Term,
};

// ============================================================================
// Unification Benchmarks
// ============================================================================

fn bench_unification_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("unification_simple");

    // Variable-constant unification
    group.bench_function("var_constant", |b| {
        let x = Term::var("x");
        let a = Term::constant("a");
        b.iter(|| {
            let _ = unify_terms(black_box(&x), black_box(&a));
        });
    });

    // Same variable unification
    group.bench_function("same_var", |b| {
        let x = Term::var("x");
        b.iter(|| {
            let _ = unify_terms(black_box(&x), black_box(&x));
        });
    });

    // Different constants (should fail)
    group.bench_function("different_constants", |b| {
        let a = Term::constant("a");
        let b_const = Term::constant("b");
        b.iter(|| {
            let _ = unify_terms(black_box(&a), black_box(&b_const));
        });
    });

    group.finish();
}

fn bench_unification_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("unification_complex");

    // Multiple term pairs
    for size in [2, 5, 10, 20].iter() {
        group.bench_with_input(BenchmarkId::new("term_list", size), size, |b, &size| {
            let pairs: Vec<(Term, Term)> = (0..size)
                .map(|i| {
                    (
                        Term::var(format!("x{}", i)),
                        Term::constant(format!("c{}", i)),
                    )
                })
                .collect();
            b.iter(|| {
                let _ = unify_term_list(black_box(&pairs));
            });
        });
    }

    group.finish();
}

fn bench_anti_unification(c: &mut Criterion) {
    let mut group = c.benchmark_group("anti_unification");

    // Simple anti-unification
    group.bench_function("different_constants", |b| {
        let a = Term::constant("a");
        let b_const = Term::constant("b");
        b.iter(|| {
            let _ = anti_unify_terms(black_box(&a), black_box(&b_const));
        });
    });

    // LGG of multiple terms
    for size in [2, 5, 10].iter() {
        group.bench_with_input(BenchmarkId::new("lgg", size), size, |b, &size| {
            let terms: Vec<Term> = (0..size)
                .map(|i| Term::constant(format!("c{}", i)))
                .collect();
            b.iter(|| {
                let _ = lgg_terms(black_box(&terms));
            });
        });
    }

    group.finish();
}

// ============================================================================
// Resolution Benchmarks
// ============================================================================

fn bench_resolution_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolution_simple");

    // Basic contradiction: {P}, {¬P}
    group.bench_function("basic_contradiction", |b| {
        b.iter(|| {
            let mut prover = ResolutionProver::new();
            let p = Literal::positive(TLExpr::pred("P", vec![Term::constant("a")]));
            let not_p = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));
            prover.add_clause(Clause::from_literals(vec![p]));
            prover.add_clause(Clause::from_literals(vec![not_p]));
            let _ = black_box(prover.prove());
        });
    });

    // Modus ponens: {P}, {¬P ∨ Q}, {¬Q}
    group.bench_function("modus_ponens", |b| {
        b.iter(|| {
            let mut prover = ResolutionProver::new();
            let p = Literal::positive(TLExpr::pred("P", vec![]));
            let not_p = Literal::negative(TLExpr::pred("P", vec![]));
            let q = Literal::positive(TLExpr::pred("Q", vec![]));
            let not_q = Literal::negative(TLExpr::pred("Q", vec![]));

            prover.add_clause(Clause::from_literals(vec![p]));
            prover.add_clause(Clause::from_literals(vec![not_p, q.clone()]));
            prover.add_clause(Clause::from_literals(vec![not_q]));
            let _ = black_box(prover.prove());
        });
    });

    group.finish();
}

fn bench_resolution_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolution_strategies");

    // Test different strategies on the same problem
    let clauses = vec![
        Clause::from_literals(vec![
            Literal::positive(TLExpr::pred("P", vec![])),
            Literal::positive(TLExpr::pred("Q", vec![])),
        ]),
        Clause::from_literals(vec![Literal::negative(TLExpr::pred("P", vec![]))]),
        Clause::from_literals(vec![Literal::negative(TLExpr::pred("Q", vec![]))]),
    ];

    let strategies = vec![
        (
            "Saturation",
            ResolutionStrategy::Saturation { max_clauses: 10000 },
        ),
        (
            "SetOfSupport",
            ResolutionStrategy::SetOfSupport { max_steps: 1000 },
        ),
        ("Linear", ResolutionStrategy::Linear { max_depth: 100 }),
        (
            "UnitResolution",
            ResolutionStrategy::UnitResolution { max_steps: 1000 },
        ),
    ];

    for (name, strategy) in strategies {
        group.bench_with_input(
            BenchmarkId::new("strategy", name),
            &strategy,
            |b, strategy| {
                b.iter(|| {
                    let mut prover = ResolutionProver::with_strategy(strategy.clone());
                    for clause in &clauses {
                        prover.add_clause(clause.clone());
                    }
                    let _ = black_box(prover.prove());
                });
            },
        );
    }

    group.finish();
}

fn bench_resolution_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolution_scaling");

    // Benchmark with increasing number of clauses
    for num_predicates in [3, 5, 7, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("predicates", num_predicates),
            num_predicates,
            |b, &num_predicates| {
                b.iter(|| {
                    let mut prover = ResolutionProver::new();

                    // Create a chain of implications: P1 → P2 → ... → Pn
                    for i in 0..num_predicates {
                        let pi = Literal::positive(TLExpr::pred(format!("P{}", i), vec![]));
                        let pi_plus_1 =
                            Literal::positive(TLExpr::pred(format!("P{}", i + 1), vec![]));
                        let not_pi = pi.negate();

                        // Pi → Pi+1 becomes ¬Pi ∨ Pi+1
                        prover.add_clause(Clause::from_literals(vec![not_pi, pi_plus_1]));
                    }

                    // Add P0 as fact
                    prover.add_clause(Clause::from_literals(vec![Literal::positive(
                        TLExpr::pred("P0", vec![]),
                    )]));

                    // Negate goal: ¬Pn
                    prover.add_clause(Clause::from_literals(vec![Literal::negative(
                        TLExpr::pred(format!("P{}", num_predicates), vec![]),
                    )]));

                    let _ = black_box(prover.prove());
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Sequent Calculus Benchmarks
// ============================================================================

fn bench_sequent_axioms(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequent_axioms");

    // Identity axiom checking
    group.bench_function("is_axiom", |b| {
        let p = TLExpr::pred("P", vec![Term::var("x")]);
        let sequent = Sequent::identity(p);
        b.iter(|| {
            let _ = black_box(sequent.is_axiom());
        });
    });

    // Weakening operations
    group.bench_function("weakening_left", |b| {
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let sequent = Sequent::identity(p.clone());
        b.iter(|| {
            let _ = black_box(sequent.clone().weaken_left(q.clone()));
        });
    });

    group.finish();
}

fn bench_sequent_proof_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequent_proof_search");

    // Simple AND decomposition
    group.bench_function("and_decomposition", |b| {
        use tensorlogic_ir::{ProofSearchEngine, ProofSearchStrategy};

        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let and_pq = TLExpr::and(p.clone(), q);
        let goal = Sequent::new(vec![and_pq], vec![p]);

        b.iter(|| {
            let mut engine =
                ProofSearchEngine::new(ProofSearchStrategy::DepthFirst { max_depth: 10 }, 1000);
            let _ = black_box(engine.search(&goal));
        });
    });

    // Nested conjunctions
    for depth in [2, 3, 4].iter() {
        group.bench_with_input(BenchmarkId::new("nested_and", depth), depth, |b, &depth| {
            use tensorlogic_ir::{ProofSearchEngine, ProofSearchStrategy};

            let p = TLExpr::pred("P", vec![]);

            // Build nested AND: ((P ∧ P) ∧ P) ∧ ... ∧ P
            let mut nested = p.clone();
            for _ in 0..depth {
                nested = TLExpr::and(nested, p.clone());
            }

            let goal = Sequent::new(vec![nested], vec![p.clone()]);

            b.iter(|| {
                let mut engine =
                    ProofSearchEngine::new(ProofSearchStrategy::DepthFirst { max_depth: 20 }, 1000);
                let _ = black_box(engine.search(&goal));
            });
        });
    }

    group.finish();
}

fn bench_sequent_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequent_strategies");

    use tensorlogic_ir::{ProofSearchEngine, ProofSearchStrategy};

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);
    let r = TLExpr::pred("R", vec![]);

    // (P ∧ Q) ∧ R ⊢ P
    let nested = TLExpr::and(TLExpr::and(p.clone(), q), r);
    let goal = Sequent::new(vec![nested], vec![p]);

    let strategies = vec![
        (
            "DepthFirst",
            ProofSearchStrategy::DepthFirst { max_depth: 15 },
        ),
        (
            "BreadthFirst",
            ProofSearchStrategy::BreadthFirst { max_depth: 15 },
        ),
        (
            "IterativeDeepening",
            ProofSearchStrategy::IterativeDeepening { max_depth: 15 },
        ),
    ];

    for (name, strategy) in strategies {
        group.bench_with_input(
            BenchmarkId::new("strategy", name),
            &strategy,
            |b, strategy| {
                b.iter(|| {
                    let mut engine = ProofSearchEngine::new(strategy.clone(), 1000);
                    let _ = black_box(engine.search(&goal));
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Cross-Module Integration Benchmarks
// ============================================================================

fn bench_resolution_with_unification(c: &mut Criterion) {
    let mut group = c.benchmark_group("resolution_unification");

    // Resolution with variables requiring unification
    group.bench_function("variable_resolution", |b| {
        b.iter(|| {
            let mut prover = ResolutionProver::new();

            // P(x) - universal fact
            let p_x = Literal::positive(TLExpr::pred("P", vec![Term::var("x")]));

            // ¬P(a) - negated goal
            let not_p_a = Literal::negative(TLExpr::pred("P", vec![Term::constant("a")]));

            prover.add_clause(Clause::from_literals(vec![p_x]));
            prover.add_clause(Clause::from_literals(vec![not_p_a]));

            let _ = black_box(prover.prove());
        });
    });

    // Multiple variables
    group.bench_function("multiple_variables", |b| {
        b.iter(|| {
            let mut prover = ResolutionProver::new();

            // R(x, y)
            let r_xy = Literal::positive(TLExpr::pred("R", vec![Term::var("x"), Term::var("y")]));

            // ¬R(a, b)
            let not_r_ab = Literal::negative(TLExpr::pred(
                "R",
                vec![Term::constant("a"), Term::constant("b")],
            ));

            prover.add_clause(Clause::from_literals(vec![r_xy]));
            prover.add_clause(Clause::from_literals(vec![not_r_ab]));

            let _ = black_box(prover.prove());
        });
    });

    group.finish();
}

fn bench_proof_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_complexity");

    // Measure complexity vs problem size
    for size in [3, 5, 7].iter() {
        group.bench_with_input(BenchmarkId::new("chain_length", size), size, |b, &size| {
            b.iter(|| {
                let mut prover = ResolutionProver::new();

                // Create chain: P1 → P2 → ... → Pn
                for i in 0..size {
                    let pi = Literal::positive(TLExpr::pred(format!("P{}", i), vec![]));
                    let pi_next = Literal::positive(TLExpr::pred(format!("P{}", i + 1), vec![]));

                    // Pi → Pi+1 becomes ¬Pi ∨ Pi+1
                    prover.add_clause(Clause::from_literals(vec![pi.negate(), pi_next]));
                }

                // P0
                prover.add_clause(Clause::from_literals(vec![Literal::positive(
                    TLExpr::pred("P0", vec![]),
                )]));

                // ¬Pn
                prover.add_clause(Clause::from_literals(vec![Literal::negative(
                    TLExpr::pred(format!("P{}", size), vec![]),
                )]));

                let _ = black_box(prover.prove());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_unification_simple,
    bench_unification_complex,
    bench_anti_unification,
    bench_resolution_simple,
    bench_resolution_strategies,
    bench_resolution_scaling,
    bench_sequent_axioms,
    bench_sequent_proof_search,
    bench_sequent_strategies,
    bench_resolution_with_unification,
    bench_proof_complexity,
);

criterion_main!(benches);
