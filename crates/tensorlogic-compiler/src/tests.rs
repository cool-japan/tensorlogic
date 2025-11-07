//! Unit tests for the compiler.

use crate::{
    compile_to_einsum, compile_to_einsum_with_context, passes::validate_arity, CompilerContext,
};
use tensorlogic_ir::{TLExpr, Term};

#[test]
fn test_simple_predicate() {
    let pred = TLExpr::pred("Parent", vec![Term::var("x"), Term::var("y")]);
    let graph = compile_to_einsum(&pred).unwrap();

    assert_eq!(graph.tensors.len(), 1);
    assert!(graph.tensors[0].starts_with("Parent"));
}

#[test]
fn test_and_expression() {
    let p1 = TLExpr::pred("Parent", vec![Term::var("x"), Term::var("y")]);
    let p2 = TLExpr::pred("Child", vec![Term::var("x"), Term::var("y")]);
    let and_expr = TLExpr::and(p1, p2);

    let graph = compile_to_einsum(&and_expr).unwrap();
    assert!(graph.tensors.len() >= 2);
    assert!(!graph.nodes.is_empty());
}

#[test]
fn test_exists_quantifier() {
    let pred = TLExpr::pred("Parent", vec![Term::var("x"), Term::var("z")]);
    let exists_expr = TLExpr::exists("z", "Person", pred);

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);

    let graph = compile_to_einsum_with_context(&exists_expr, &mut ctx).unwrap();
    assert!(!graph.nodes.is_empty());
}

#[test]
fn test_arity_validation() {
    let p1 = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    let p2 = TLExpr::pred("P", vec![Term::var("a"), Term::var("b")]);
    let expr = TLExpr::and(p1, p2);

    assert!(validate_arity(&expr).is_ok());

    let p3 = TLExpr::pred("P", vec![Term::var("x")]);
    let bad_expr = TLExpr::and(expr, p3);

    assert!(validate_arity(&bad_expr).is_err());
}

#[test]
fn test_implication() {
    let premise = TLExpr::pred("Parent", vec![Term::var("x"), Term::var("y")]);
    let conclusion = TLExpr::pred("Ancestor", vec![Term::var("x"), Term::var("y")]);
    let imply_expr = TLExpr::imply(premise, conclusion);

    let graph = compile_to_einsum(&imply_expr).unwrap();
    assert!(graph.tensors.len() >= 2);
}

#[test]
fn test_compiler_context() {
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);
    ctx.add_domain("City", 50);

    assert!(ctx.bind_var("x", "Person").is_ok());
    assert!(ctx.bind_var("y", "City").is_ok());
    assert!(ctx.bind_var("z", "NonExistent").is_err());

    let axis_x = ctx.assign_axis("x");
    let axis_y = ctx.assign_axis("y");
    assert_ne!(axis_x, axis_y);

    let axis_x2 = ctx.assign_axis("x");
    assert_eq!(axis_x, axis_x2);
}

#[test]
fn test_transitivity_rule_shared_variables() {
    // Transitivity: knows(x,y) ∧ knows(y,z) → knows(x,z)
    // This tests variable sharing in AND (y appears in both predicates)
    let knows_xy = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let knows_yz = TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")]);
    let knows_xz = TLExpr::pred("knows", vec![Term::var("x"), Term::var("z")]);

    // Build the rule: knows(x,y) ∧ knows(y,z) → knows(x,z)
    let premise = TLExpr::and(knows_xy, knows_yz);
    let rule = TLExpr::imply(premise, knows_xz);

    let graph = compile_to_einsum(&rule).unwrap();

    // Check that compilation succeeded
    assert!(graph.tensors.len() >= 3, "Should have at least 3 tensors");
    assert!(!graph.nodes.is_empty(), "Should have einsum nodes");

    // Verify that the first AND operation produces correct einsum spec
    // Should be something like "ab,bc->abc" (contraction over shared 'b'/'y')
    let and_node = &graph.nodes[0];
    if let tensorlogic_ir::OpType::Einsum { spec } = &and_node.op {
        assert!(
            spec.contains("->"),
            "Einsum spec should have output: {}",
            spec
        );
    } else {
        panic!("Expected Einsum operation");
    }

    // The output axes should be the union of input axes (x, y, z = a, b, c)
    // After contraction: should have axes for x and z only
}

#[test]
fn test_and_with_different_axes() {
    // Test AND with partially overlapping variables
    // P(x,y) ∧ Q(y,z) should produce output with axes for x,y,z
    let p = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    let q = TLExpr::pred("Q", vec![Term::var("y"), Term::var("z")]);
    let and_expr = TLExpr::and(p, q);

    let graph = compile_to_einsum(&and_expr).unwrap();

    // Should successfully compile with shared variable 'y'
    assert!(graph.tensors.len() >= 2);
    assert_eq!(graph.nodes.len(), 1, "Should have one AND operation");

    // Check the einsum spec
    let einsum_node = &graph.nodes[0];
    // Should be something like "ab,bc->abc" where b is shared (y)
    if let tensorlogic_ir::OpType::Einsum { spec } = &einsum_node.op {
        assert!(spec.contains(","), "Should have comma in spec");
        assert!(spec.contains("->"), "Should have arrow in spec");
    } else {
        panic!("Expected Einsum operation");
    }
}

#[test]
fn test_and_with_disjoint_variables() {
    // Test AND with completely disjoint variables
    // P(x) ∧ Q(y) should produce outer product with axes x,y
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("y")]);
    let and_expr = TLExpr::and(p, q);

    let graph = compile_to_einsum(&and_expr).unwrap();

    // Should successfully compile
    assert!(graph.tensors.len() >= 2);
    assert_eq!(graph.nodes.len(), 1);

    // Einsum spec should be "a,b->ab" (outer product)
    let einsum_node = &graph.nodes[0];
    if let tensorlogic_ir::OpType::Einsum { spec } = &einsum_node.op {
        let parts: Vec<&str> = spec.split("->").collect();
        assert_eq!(parts.len(), 2, "Einsum spec should have input and output");

        // Output should contain both axes
        let output_axes = parts[1];
        assert_eq!(
            output_axes.len(),
            2,
            "Output should have 2 axes for x and y"
        );
    } else {
        panic!("Expected Einsum operation");
    }
}

#[test]
fn test_transitivity_complete() {
    // Test: ∀x,y,z. knows(x,y) ∧ knows(y,z) → knows(x,z)
    // This is a classic transitivity rule that should work with our new broadcasting

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);

    // Build the premise: knows(x,y) ∧ knows(y,z)
    let knows_xy = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let knows_yz = TLExpr::pred("knows", vec![Term::var("y"), Term::var("z")]);
    let premise = TLExpr::and(knows_xy, knows_yz);

    // Build the conclusion: knows(x,z)
    let conclusion = TLExpr::pred("knows", vec![Term::var("x"), Term::var("z")]);

    // Create the implication
    let transitivity_rule = TLExpr::imply(premise, conclusion);

    // Compile it
    let result = compile_to_einsum_with_context(&transitivity_rule, &mut ctx);

    // Should compile successfully with our new broadcasting approach
    assert!(
        result.is_ok(),
        "Transitivity rule should compile successfully"
    );

    let graph = result.unwrap();

    // Verify the graph has operations (marginalization, broadcasting, subtraction, relu)
    assert!(
        !graph.nodes.is_empty(),
        "Should have generated computation nodes"
    );

    // Should have at least:
    // - 2 predicates (knows_xy, knows_yz)
    // - 1 AND (einsum contraction)
    // - 1 marginalization (sum over y)
    // - 1 subtraction
    // - 1 relu
    assert!(
        graph.nodes.len() >= 4,
        "Should have sufficient operations for transitivity"
    );
}

#[test]
fn test_implication_with_broadcasting() {
    // Test: P(x) → Q(x, y)
    // Conclusion has extra variable y, so premise should be broadcast

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);
    ctx.add_domain("Location", 50);

    let premise = TLExpr::pred("P", vec![Term::var("x")]);
    let conclusion = TLExpr::pred("Q", vec![Term::var("x"), Term::var("y")]);

    let implication = TLExpr::imply(premise, conclusion);

    let result = compile_to_einsum_with_context(&implication, &mut ctx);

    assert!(
        result.is_ok(),
        "Should handle conclusion with extra variables"
    );

    let graph = result.unwrap();
    assert!(!graph.nodes.is_empty());
}

#[test]
fn test_or_with_different_variables() {
    // Test: P(x) ∨ Q(y)
    // Different free variables should be broadcast to common shape

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);

    let p_x = TLExpr::pred("P", vec![Term::var("x")]);
    let q_y = TLExpr::pred("Q", vec![Term::var("y")]);

    let or_expr = TLExpr::or(p_x, q_y);

    let result = compile_to_einsum_with_context(&or_expr, &mut ctx);

    if let Err(e) = &result {
        eprintln!("Error compiling OR with different variables: {:?}", e);
    }
    assert!(
        result.is_ok(),
        "OR should handle different free variables via broadcasting: {:?}",
        result.err()
    );

    let graph = result.unwrap();
    assert!(!graph.nodes.is_empty());
}
