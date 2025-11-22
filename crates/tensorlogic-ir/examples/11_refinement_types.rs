//! Refinement Types Example
//!
//! Demonstrates refinement types that extend base types with logical predicates.
//! This enables more precise type checking and verification by constraining valid values.

use tensorlogic_ir::refinement::{LiquidTypeInference, RefinementContext, RefinementType};
use tensorlogic_ir::TLExpr;

fn main() {
    println!("=== Refinement Types in TensorLogic ===\n");

    // Example 1: Built-in refinement types
    example_builtin_refinements();

    // Example 2: Refinement context
    example_refinement_context();

    // Example 3: Type strengthening and weakening
    example_strengthening_weakening();

    // Example 4: Non-empty collections
    example_non_empty_collections();

    // Example 5: Liquid type inference
    example_liquid_type_inference();

    // Example 6: Refinement composition
    example_refinement_composition();
}

fn example_builtin_refinements() {
    println!("--- Example 1: Built-in Refinement Types ---");

    // Positive integers
    let pos_int = RefinementType::positive_int("x");
    println!("Positive int: {}", pos_int);

    // Natural numbers (non-negative)
    let nat = RefinementType::nat("n");
    println!("Natural number: {}", nat);

    // Probability (0.0 to 1.0)
    let prob = RefinementType::probability("p");
    println!("Probability: {}", prob);

    // Non-empty vector
    let non_empty = RefinementType::non_empty_vec("v", "Int");
    println!("Non-empty vector: {}\n", non_empty);
}

fn example_refinement_context() {
    println!("--- Example 2: Refinement Context ---");

    let mut ctx = RefinementContext::new();

    // Bind a refined variable
    let pos_int = RefinementType::positive_int("x");
    ctx.bind("x", pos_int.clone());

    println!("Bound variable 'x' with refinement: {}", pos_int);

    // Get the type
    if let Some(typ) = ctx.get_type("x") {
        println!("Type of 'x': {}", typ);
    }

    // Add assumptions
    let assumption = TLExpr::lt(TLExpr::pred("x", vec![]), TLExpr::constant(100.0));
    ctx.assume(assumption.clone());

    println!("Added assumption: {}", assumption);
    println!(
        "Assumption is satisfied: {}\n",
        ctx.check_refinement(&assumption)
    );
}

fn example_strengthening_weakening() {
    println!("--- Example 3: Type Strengthening and Weakening ---");

    // Start with positive int: {x: Int | x > 0}
    let pos_int = RefinementType::positive_int("x");
    println!("Original: {}", pos_int);

    // Strengthen: add upper bound
    let upper_bound = TLExpr::lt(TLExpr::pred("x", vec![]), TLExpr::constant(100.0));
    let bounded_pos_int = pos_int.clone().strengthen(upper_bound);
    println!("Strengthened (with upper bound): {}", bounded_pos_int);

    // Weaken: remove refinement
    let weakened = pos_int.weaken();
    println!("Weakened: {}", weakened);
    println!("Weakened refinement is trivial: {}\n", weakened.refinement);
}

fn example_non_empty_collections() {
    println!("--- Example 4: Non-Empty Collections ---");

    // Non-empty vector of integers
    let non_empty_ints = RefinementType::non_empty_vec("numbers", "Int");
    println!("Non-empty vector: {}", non_empty_ints);

    // Non-empty vector of floats
    let non_empty_floats = RefinementType::non_empty_vec("values", "Float");
    println!("Non-empty vector: {}\n", non_empty_floats);
}

fn example_liquid_type_inference() {
    println!("--- Example 5: Liquid Type Inference ---");

    let mut inference = LiquidTypeInference::new();

    // Define candidates for refinement
    let candidates = vec![
        TLExpr::gt(TLExpr::pred("x", vec![]), TLExpr::constant(0.0)),
        TLExpr::gte(TLExpr::pred("x", vec![]), TLExpr::constant(0.0)),
        TLExpr::lt(TLExpr::pred("x", vec![]), TLExpr::constant(100.0)),
    ];

    inference.add_unknown("x_refinement", candidates);

    println!("Added unknown refinement with 3 candidates");

    // Infer refinements
    let inferred = inference.infer();

    println!("Inferred refinements:");
    for (name, refinement) in &inferred {
        println!("  {}: {}", name, refinement);
    }

    println!();
}

fn example_refinement_composition() {
    println!("--- Example 6: Refinement Composition ---");

    // Start with natural numbers
    let nat = RefinementType::nat("n");
    println!("Base: {}", nat);

    // Add upper bound to make it bounded nat
    let bounded_nat = nat.strengthen(TLExpr::lt(
        TLExpr::pred("n", vec![]),
        TLExpr::constant(1000.0),
    ));
    println!("Strengthened (bounded): {}", bounded_nat);

    // Matrix dimensions: positive integers for rows and columns
    let m_positive = RefinementType::positive_int("m");
    let n_positive = RefinementType::positive_int("n");

    println!("\nMatrix dimensions:");
    println!("  Rows (m): {}", m_positive);
    println!("  Cols (n): {}", n_positive);

    // Additional constraint: total size <= 1 million
    let size_constraint = TLExpr::lte(
        TLExpr::mul(TLExpr::pred("m", vec![]), TLExpr::pred("n", vec![])),
        TLExpr::constant(1000000.0),
    );

    let constrained_m = m_positive.strengthen(size_constraint);
    println!("  With size constraint: {}\n", constrained_m);
}
