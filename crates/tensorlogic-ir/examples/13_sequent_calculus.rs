//! # Example 13: Sequent Calculus and Automated Proof Search
//!
//! This example demonstrates the sequent calculus module for proof-theoretic reasoning.
//! Sequent calculus provides a formal foundation for constructing and verifying logical proofs.
//!
//! ## What You'll Learn
//!
//! - Creating sequents (Γ ⊢ Δ) with antecedents and consequents
//! - Applying inference rules (Identity, Weakening, AND/OR/NOT rules)
//! - Building proof trees manually
//! - Automated proof search with different strategies
//! - Cut elimination for proof normalization
//!
//! ## Key Concepts
//!
//! A **sequent** `Γ ⊢ Δ` represents: "if all formulas in Γ are true, then at least one formula in Δ is true"
//! - Γ (antecedents): hypotheses on the left
//! - Δ (consequents): conclusions on the right
//! - ⊢ (turnstile): entailment relation

use tensorlogic_ir::{
    CutElimination, InferenceRule, ProofSearchEngine, ProofSearchStrategy, ProofTree, Sequent,
    TLExpr, Term,
};

fn main() {
    println!("=== Sequent Calculus Examples ===\n");

    // Example 1: Identity Axiom
    example_1_identity();

    // Example 2: Weakening Rules
    example_2_weakening();

    // Example 3: AND Rules
    example_3_and_rules();

    // Example 4: OR Rules
    example_4_or_rules();

    // Example 5: NOT Rules
    example_5_not_rules();

    // Example 6: Automated Proof Search
    example_6_proof_search();

    // Example 7: Different Search Strategies
    example_7_search_strategies();

    // Example 8: Cut Elimination
    example_8_cut_elimination();

    // Example 9: Complex Proof
    example_9_complex_proof();
}

fn example_1_identity() {
    println!("Example 1: Identity Axiom");
    println!("The most basic sequent: P ⊢ P\n");

    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let sequent = Sequent::identity(p.clone());

    println!("  Sequent: {:?}", sequent);
    println!("  Is axiom: {}", sequent.is_axiom());
    println!("  Free variables: {:?}\n", sequent.free_vars());

    // Create a proof tree
    let proof = ProofTree::identity(p);
    println!("  Proof valid: {}", proof.is_valid());
    println!("  Proof depth: {}", proof.depth());
    println!("  Proof size: {}\n", proof.size());
}

fn example_2_weakening() {
    println!("Example 2: Weakening Rules");
    println!("Add formulas without changing validity\n");

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);

    // Start with P ⊢ P
    let sequent = Sequent::identity(p.clone());
    println!("  Original: P ⊢ P");

    // Apply weakening left: add Q to antecedents
    let weakened_left = sequent.clone().weaken_left(q.clone());
    println!("  After weakening left: P, Q ⊢ P");
    println!("  Antecedents: {}", weakened_left.antecedents.len());

    // Apply weakening right: add Q to consequents
    let weakened_right = sequent.weaken_right(q);
    println!("  After weakening right: P ⊢ P, Q");
    println!("  Consequents: {}\n", weakened_right.consequents.len());
}

fn example_3_and_rules() {
    println!("Example 3: AND Rules");
    println!("Decompose conjunctions in proofs\n");

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);
    let and_pq = TLExpr::and(p.clone(), q.clone());

    // AND-Left: From P ∧ Q ⊢ P, we can derive by splitting the conjunction
    let sequent = Sequent::new(vec![and_pq.clone()], vec![p.clone()]);
    println!("  Goal sequent: P ∧ Q ⊢ P");
    println!("  This is provable by AND-Left rule");

    // The premise would be: P, Q ⊢ P (which is an axiom)
    let premise = ProofTree::identity(p.clone());

    let proof = ProofTree::new(sequent, InferenceRule::AndLeft { index: 0 }, vec![premise]);

    println!("  Proof valid: {}", proof.is_valid());
    println!("  Proof depth: {}\n", proof.depth());
}

fn example_4_or_rules() {
    println!("Example 4: OR Rules");
    println!("Handle disjunctions in proofs\n");

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);
    let or_pq = TLExpr::or(p.clone(), q.clone());

    // OR-Right: From P ⊢ P ∨ Q
    let _sequent = Sequent::new(vec![p.clone()], vec![or_pq]);
    println!("  Goal sequent: P ⊢ P ∨ Q");
    println!("  This is provable by OR-Right rule");

    // The premise would be: P ⊢ P, Q
    let premise_seq = Sequent::new(vec![p.clone()], vec![p.clone(), q]);
    let _premise = ProofTree::identity(p);

    println!(
        "  Premise sequent has {} consequents\n",
        premise_seq.consequents.len()
    );
}

fn example_5_not_rules() {
    println!("Example 5: NOT Rules");
    println!("Handle negation in proofs\n");

    let p = TLExpr::pred("P", vec![]);
    let not_p = TLExpr::negate(p.clone());

    // NOT-Left: ¬P ⊢ (empty) is not provable
    // NOT-Right: (empty) ⊢ ¬P means we need to prove P ⊢ (empty), i.e., P leads to contradiction

    let sequent = Sequent::new(vec![not_p.clone()], vec![]);
    println!("  Sequent: ¬P ⊢ (empty)");
    println!("  Is axiom: {}", sequent.is_axiom());

    let sequent2 = Sequent::identity(not_p);
    println!("  Identity: ¬P ⊢ ¬P");
    println!("  Is axiom: {}\n", sequent2.is_axiom());
}

fn example_6_proof_search() {
    println!("Example 6: Automated Proof Search");
    println!("Let the engine find proofs automatically\n");

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);
    let and_pq = TLExpr::and(p.clone(), q);

    // Goal: Prove P ∧ Q ⊢ P
    let sequent = Sequent::new(vec![and_pq], vec![p]);

    println!("  Goal: P ∧ Q ⊢ P");

    let mut engine =
        ProofSearchEngine::new(ProofSearchStrategy::DepthFirst { max_depth: 10 }, 1000);

    match engine.search(&sequent) {
        Some(proof) => {
            println!("  ✓ Proof found!");
            println!("  Depth: {}", proof.depth());
            println!("  Size: {} steps", proof.size());
            println!("  Uses cut: {}", proof.uses_cut());
            println!("  Statistics:");
            println!(
                "    - Sequents explored: {}",
                engine.stats.sequents_explored
            );
            println!("    - Proofs generated: {}", engine.stats.proofs_generated);
            println!("    - Backtracks: {}", engine.stats.backtracks);
        }
        None => println!("  ✗ No proof found"),
    }
    println!();
}

fn example_7_search_strategies() {
    println!("Example 7: Different Search Strategies");
    println!("Compare different proof search approaches\n");

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);
    let and_pq = TLExpr::and(p.clone(), q.clone());

    let sequent = Sequent::new(vec![and_pq], vec![p]);

    let strategies = vec![
        (
            "Depth-First",
            ProofSearchStrategy::DepthFirst { max_depth: 10 },
        ),
        (
            "Breadth-First",
            ProofSearchStrategy::BreadthFirst { max_depth: 10 },
        ),
        (
            "Iterative Deepening",
            ProofSearchStrategy::IterativeDeepening { max_depth: 10 },
        ),
    ];

    for (name, strategy) in strategies {
        let mut engine = ProofSearchEngine::new(strategy, 1000);
        match engine.search(&sequent) {
            Some(proof) => {
                println!("  {} Strategy:", name);
                println!("    Proof depth: {}", proof.depth());
                println!("    Sequents explored: {}", engine.stats.sequents_explored);
                println!("    Backtracks: {}", engine.stats.backtracks);
            }
            None => println!("  {} Strategy: No proof found", name),
        }
    }
    println!();
}

fn example_8_cut_elimination() {
    println!("Example 8: Cut Elimination");
    println!("Normalize proofs by removing cut rules\n");

    let p = TLExpr::pred("P", vec![]);

    // Create a simple proof without cuts
    let proof = ProofTree::identity(p);

    println!("  Original proof:");
    println!("    Uses cut: {}", proof.uses_cut());
    println!("    Is cut-free: {}", CutElimination::is_cut_free(&proof));

    // Eliminate cuts (should be unchanged for cut-free proofs)
    let eliminated = CutElimination::eliminate(proof.clone());

    println!("  After cut elimination:");
    println!("    Uses cut: {}", eliminated.uses_cut());
    println!(
        "    Depth unchanged: {}",
        eliminated.depth() == proof.depth()
    );
    println!();
}

fn example_9_complex_proof() {
    println!("Example 9: Complex Proof with Multiple Rules");
    println!("Combine several inference rules\n");

    let p = TLExpr::pred("P", vec![]);
    let q = TLExpr::pred("Q", vec![]);
    let r = TLExpr::pred("R", vec![]);

    // Goal: (P ∧ Q) ∧ R ⊢ P
    let pq = TLExpr::and(p.clone(), q.clone());
    let pqr = TLExpr::and(pq, r);

    let sequent = Sequent::new(vec![pqr], vec![p]);

    println!("  Goal: (P ∧ Q) ∧ R ⊢ P");
    println!("  This requires nested AND-Left applications\n");

    let mut engine =
        ProofSearchEngine::new(ProofSearchStrategy::DepthFirst { max_depth: 15 }, 2000);

    match engine.search(&sequent) {
        Some(proof) => {
            println!("  ✓ Complex proof found!");
            println!("  Depth: {}", proof.depth());
            println!("  Size: {} inference steps", proof.size());
            println!("  Statistics:");
            println!(
                "    - Sequents explored: {}",
                engine.stats.sequents_explored
            );
            println!("    - Proofs generated: {}", engine.stats.proofs_generated);
            println!("    - Backtracks: {}", engine.stats.backtracks);

            if let Some(depth) = engine.stats.proof_depth {
                println!("    - Final depth: {}", depth);
            }
        }
        None => println!("  ✗ Proof search failed"),
    }
    println!();
}
