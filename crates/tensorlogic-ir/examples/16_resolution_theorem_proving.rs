//! # Example 16: Resolution-Based Theorem Proving
//!
//! This example demonstrates the resolution-based automated theorem proving capabilities
//! of TensorLogic's resolution module.
//!
//! ## What You'll Learn
//!
//! - Creating clauses and literals
//! - Binary resolution between clauses
//! - Different proof strategies (saturation, set-of-support, unit resolution)
//! - Converting formulas to Clausal Normal Form (CNF)
//! - Classical logic proofs (modus ponens, contrapositive, etc.)
//! - Analyzing proof statistics
//!
//! ## Key Concepts
//!
//! **Resolution Principle**: From clauses `C₁ ∨ L` and `C₂ ∨ ¬L`, derive `C₁ ∨ C₂`
//! - **Refutation-based**: Prove `Γ ⊢ φ` by showing `Γ ∪ {¬φ}` is unsatisfiable
//! - **Complete**: If formula is unsatisfiable, resolution will derive empty clause
//! - **Sound**: Every derived clause is a logical consequence

use tensorlogic_ir::{Clause, Literal, ResolutionProver, ResolutionStrategy, TLExpr};

fn main() {
    println!("=== Resolution-Based Theorem Proving Examples ===\n");

    // Example 1: Basic Resolution
    example_1_basic_resolution();

    // Example 2: Modus Ponens
    example_2_modus_ponens();

    // Example 3: Contrapositive
    example_3_contrapositive();

    // Example 4: Resolution Strategies
    example_4_resolution_strategies();

    // Example 5: Horn Clause Resolution
    example_5_horn_clauses();

    // Example 6: Proof Statistics
    example_6_proof_statistics();

    // Example 7: Satisfiable Formulas
    example_7_satisfiable();

    // Example 8: Three-Clause Resolution
    example_8_three_clauses();
}

fn example_1_basic_resolution() {
    println!("Example 1: Basic Resolution");
    println!("Prove contradiction from P and ¬P\n");

    let mut prover = ResolutionProver::new();

    // Clause 1: {P}
    let p = TLExpr::pred("P", vec![]);
    prover.add_clause(Clause::unit(Literal::positive(p.clone())));

    // Clause 2: {¬P}
    prover.add_clause(Clause::unit(Literal::negative(p)));

    println!("  Clause set:");
    println!("    {{P}}");
    println!("    {{¬P}}");

    let result = prover.prove();

    match result {
        tensorlogic_ir::ProofResult::Unsatisfiable { steps, .. } => {
            println!("  ✓ Unsatisfiable (empty clause derived)");
            println!("  Resolution steps: {}", steps);
        }
        _ => println!("  ✗ Unexpected result: {:?}", result),
    }
    println!();
}

fn example_2_modus_ponens() {
    println!("Example 2: Modus Ponens");
    println!("From P and P → Q, prove Q\n");

    // In CNF:
    // P → Q ≡ ¬P ∨ Q
    // So clauses are: {P}, {¬P, Q}
    // To prove Q, negate it and derive contradiction: {P}, {¬P, Q}, {¬Q}

    let mut prover = ResolutionProver::new();

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);

    // Clause 1: {P}
    prover.add_clause(Clause::unit(Literal::positive(p.clone())));

    // Clause 2: {¬P, Q} (P → Q in CNF)
    prover.add_clause(Clause::from_literals(vec![
        Literal::negative(p),
        Literal::positive(q.clone()),
    ]));

    // Clause 3: {¬Q} (negation of goal)
    prover.add_clause(Clause::unit(Literal::negative(q)));

    println!("  Clauses:");
    println!("    {{P}}         (premise)");
    println!("    {{¬P, Q}}     (P → Q in CNF)");
    println!("    {{¬Q}}        (negated goal)");

    let result = prover.prove();

    match result {
        tensorlogic_ir::ProofResult::Unsatisfiable { steps, .. } => {
            println!("  ✓ Q is provable from P and P → Q");
            println!("  Resolution steps: {}", steps);
        }
        _ => println!("  ✗ Unexpected result"),
    }
    println!();
}

fn example_3_contrapositive() {
    println!("Example 3: Contrapositive");
    println!("From P → Q, prove ¬Q → ¬P\n");

    // To prove (P → Q) → (¬Q → ¬P), we assume P → Q and ¬Q → ¬P is false
    // In CNF:
    // P → Q ≡ {¬P, Q}
    // ¬(¬Q → ¬P) ≡ ¬Q ∧ P ≡ {¬Q}, {P}
    // If unsatisfiable, contrapositive is valid

    let mut prover = ResolutionProver::new();

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);

    // Assume P → Q: {¬P, Q}
    prover.add_clause(Clause::from_literals(vec![
        Literal::negative(p.clone()),
        Literal::positive(q.clone()),
    ]));

    // Assume ¬(¬Q → ¬P) ≡ ¬Q ∧ P
    prover.add_clause(Clause::unit(Literal::negative(q)));
    prover.add_clause(Clause::unit(Literal::positive(p)));

    println!("  Clauses:");
    println!("    {{¬P, Q}}     (P → Q)");
    println!("    {{¬Q}}        (assume ¬Q)");
    println!("    {{P}}         (assume P)");

    let result = prover.prove();

    match result {
        tensorlogic_ir::ProofResult::Unsatisfiable { steps, .. } => {
            println!("  ✓ Contrapositive is valid");
            println!("  Resolution steps: {}", steps);
        }
        _ => println!("  ✗ Unexpected result"),
    }
    println!();
}

fn example_4_resolution_strategies() {
    println!("Example 4: Different Resolution Strategies");
    println!("Compare performance of different strategies\n");

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);

    // Same problem: P, P → Q, ¬Q (should derive contradiction)
    let clauses = vec![
        Clause::unit(Literal::positive(p.clone())),
        Clause::from_literals(vec![Literal::negative(p), Literal::positive(q.clone())]),
        Clause::unit(Literal::negative(q)),
    ];

    let strategies = vec![
        (
            "Saturation",
            ResolutionStrategy::Saturation { max_clauses: 1000 },
        ),
        (
            "Set-of-Support",
            ResolutionStrategy::SetOfSupport { max_steps: 100 },
        ),
        (
            "Unit Resolution",
            ResolutionStrategy::UnitResolution { max_steps: 100 },
        ),
    ];

    for (name, strategy) in strategies {
        let mut prover = ResolutionProver::with_strategy(strategy);
        prover.add_clauses(clauses.clone());

        let result = prover.prove();

        println!("  {} Strategy:", name);
        match result {
            tensorlogic_ir::ProofResult::Unsatisfiable { steps, .. } => {
                println!("    ✓ Unsatisfiable");
                println!("    Steps: {}", steps);
                println!("    Resolution steps: {}", prover.stats.resolution_steps);
            }
            _ => println!("    Result: {:?}", result),
        }
    }
    println!();
}

fn example_5_horn_clauses() {
    println!("Example 5: Horn Clause Resolution");
    println!("Solve a Horn clause problem\n");

    // Horn clauses: rules with at most one positive literal
    // Example: animal(X) ← mammal(X)
    //          mammal(dog)
    //          Prove: animal(dog)

    // In propositional logic (simplified):
    // {¬mammal, animal}  (mammal → animal)
    // {mammal}           (mammal holds)
    // {¬animal}          (negated goal)

    let mut prover = ResolutionProver::new();

    let mammal = TLExpr::pred("mammal", vec![]);
    let animal = TLExpr::pred("animal", vec![]);

    // Rule: mammal → animal
    let rule = Clause::from_literals(vec![
        Literal::negative(mammal.clone()),
        Literal::positive(animal.clone()),
    ]);
    assert!(rule.is_horn());
    prover.add_clause(rule);

    // Fact: mammal
    prover.add_clause(Clause::unit(Literal::positive(mammal)));

    // Negated goal: ¬animal
    prover.add_clause(Clause::unit(Literal::negative(animal)));

    println!("  Horn clauses:");
    println!("    {{¬mammal, animal}}  (mammal → animal)");
    println!("    {{mammal}}           (fact)");
    println!("    {{¬animal}}          (negated goal)");

    let result = prover.prove();

    match result {
        tensorlogic_ir::ProofResult::Unsatisfiable { .. } => {
            println!("  ✓ Goal 'animal' is provable");
        }
        _ => println!("  ✗ Unexpected result"),
    }
    println!();
}

fn example_6_proof_statistics() {
    println!("Example 6: Proof Statistics");
    println!("Analyze proof search statistics\n");

    let mut prover = ResolutionProver::new();

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);
    let r = TLExpr::pred("R", vec![]);

    // Complex problem: P ∨ Q, ¬P ∨ R, ¬Q, ¬R
    prover.add_clause(Clause::from_literals(vec![
        Literal::positive(p.clone()),
        Literal::positive(q.clone()),
    ]));
    prover.add_clause(Clause::from_literals(vec![
        Literal::negative(p),
        Literal::positive(r.clone()),
    ]));
    prover.add_clause(Clause::unit(Literal::negative(q)));
    prover.add_clause(Clause::unit(Literal::negative(r)));

    println!("  Clause set:");
    println!("    {{P, Q}}");
    println!("    {{¬P, R}}");
    println!("    {{¬Q}}");
    println!("    {{¬R}}");

    let result = prover.prove();

    println!("\n  Statistics:");
    println!("    Clauses generated: {}", prover.stats.clauses_generated);
    println!("    Resolution steps: {}", prover.stats.resolution_steps);
    println!(
        "    Tautologies removed: {}",
        prover.stats.tautologies_removed
    );
    println!("    Clauses subsumed: {}", prover.stats.clauses_subsumed);
    println!(
        "    Empty clause found: {}",
        prover.stats.empty_clause_found
    );

    match result {
        tensorlogic_ir::ProofResult::Unsatisfiable { steps, derivation } => {
            println!("\n  ✓ Unsatisfiable");
            println!("  Total steps to empty clause: {}", steps);
            println!("  Derivation length: {}", derivation.len());
        }
        _ => println!("\n  Result: {:?}", result),
    }
    println!();
}

fn example_7_satisfiable() {
    println!("Example 7: Satisfiable Formulas");
    println!("Detect when no contradiction exists\n");

    let mut prover = ResolutionProver::new();

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);

    // Satisfiable: {P}, {Q} (no complementary literals)
    prover.add_clause(Clause::unit(Literal::positive(p)));
    prover.add_clause(Clause::unit(Literal::positive(q)));

    println!("  Clause set:");
    println!("    {{P}}");
    println!("    {{Q}}");
    println!("  (No complementary literals - should be satisfiable)");

    let result = prover.prove();

    match result {
        tensorlogic_ir::ProofResult::Satisfiable => {
            println!("  ✓ Satisfiable (no contradiction)");
        }
        tensorlogic_ir::ProofResult::Saturated { clauses_generated } => {
            println!("  ✓ Saturated without finding contradiction");
            println!("  Clauses generated: {}", clauses_generated);
        }
        _ => println!("  Result: {:?}", result),
    }
    println!();
}

fn example_8_three_clauses() {
    println!("Example 8: Three-Clause Resolution Chain");
    println!("Demonstrate multi-step resolution\n");

    // Prove: (P ∨ Q) ∧ (¬P ∨ R) ∧ ¬Q ∧ ¬R is unsatisfiable
    // This requires multiple resolution steps

    let mut prover = ResolutionProver::new();

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);
    let r = TLExpr::pred("R", vec![]);

    prover.add_clause(Clause::from_literals(vec![
        Literal::positive(p.clone()),
        Literal::positive(q.clone()),
    ]));
    prover.add_clause(Clause::from_literals(vec![
        Literal::negative(p),
        Literal::positive(r.clone()),
    ]));
    prover.add_clause(Clause::unit(Literal::negative(q)));
    prover.add_clause(Clause::unit(Literal::negative(r)));

    println!("  Clause set:");
    println!("    {{P, Q}}");
    println!("    {{¬P, R}}");
    println!("    {{¬Q}}");
    println!("    {{¬R}}");
    println!("\n  Resolution chain:");
    println!("    {{P, Q}} + {{¬Q}} → {{P}}       (resolve on Q)");
    println!("    {{P}} + {{¬P, R}} → {{R}}       (resolve on P)");
    println!("    {{R}} + {{¬R}} → {{}}           (resolve on R, empty clause!)");

    let result = prover.prove();

    match result {
        tensorlogic_ir::ProofResult::Unsatisfiable { steps, .. } => {
            println!("\n  ✓ Empty clause derived through resolution chain");
            println!("  Steps: {}", steps);
        }
        _ => println!("\n  ✗ Unexpected result"),
    }
    println!();
}
