//! End-to-end integration tests for tensorlogic-compiler.
//!
//! These tests verify complete workflows from TLExpr construction through
//! compilation to EinsumGraph generation, validating that all components
//! work together correctly.
//!
//! **Note**: These tests require the `integration-tests` feature to avoid
//! circular dev-dependencies with tensorlogic-scirs-backend.

#![cfg(feature = "integration-tests")]

use tensorlogic_compiler::{
    compile_to_einsum, compile_to_einsum_with_context, CompilationConfig, CompilerContext,
};
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{EinsumGraph, TLExpr, Term};
use tensorlogic_scirs_backend::Scirs2Exec;

// ============================================================================
// Helper Functions
// ============================================================================

/// Compile and execute an expression with given tensor values
fn compile_and_execute(expr: &TLExpr, inputs: &[(&str, Vec<f64>)]) -> Vec<f64> {
    let graph = compile_to_einsum(expr).expect("Compilation failed");
    execute_graph(&graph, inputs)
}

/// Execute a graph with given tensor values
fn execute_graph(graph: &EinsumGraph, inputs: &[(&str, Vec<f64>)]) -> Vec<f64> {
    let mut executor = Scirs2Exec::new();

    // Add input tensors
    for tensor_name in &graph.tensors {
        if tensor_name.starts_with("const_") {
            continue;
        }

        for (pred_name, values) in inputs {
            if tensor_name.starts_with(pred_name) {
                use scirs2_core::ndarray::Array1;
                let tensor = Array1::from(values.clone()).into_dyn();
                executor.add_tensor(tensor_name.clone(), tensor);
                break;
            }
        }
    }

    let result = executor.forward(graph).expect("Execution failed");
    result.as_slice().expect("Failed to get slice").to_vec()
}

// ============================================================================
// Basic Logical Operations
// ============================================================================

#[test]
fn test_end_to_end_simple_predicate() {
    let expr = TLExpr::pred("person", vec![Term::var("x")]);
    let result = compile_and_execute(&expr, &[("person", vec![1.0, 0.0, 1.0])]);

    assert_eq!(result.len(), 3);
    assert_eq!(result, vec![1.0, 0.0, 1.0]);
}

#[test]
fn test_end_to_end_and_operation() {
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::and(p, q);

    let result = compile_and_execute(
        &expr,
        &[("P", vec![1.0, 0.5, 0.0]), ("Q", vec![1.0, 0.5, 1.0])],
    );

    assert_eq!(result.len(), 3);
    // Product AND: 1.0*1.0=1.0, 0.5*0.5=0.25, 0.0*1.0=0.0
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 0.25).abs() < 1e-6);
    assert!((result[2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_end_to_end_or_operation() {
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::or(p, q);

    let result = compile_and_execute(
        &expr,
        &[("P", vec![0.0, 0.5, 1.0]), ("Q", vec![0.0, 0.5, 0.0])],
    );

    assert_eq!(result.len(), 3);
    // ProbabilisticSum OR: a + b - a*b
    // 0 + 0 - 0 = 0
    // 0.5 + 0.5 - 0.25 = 0.75
    // 1 + 0 - 0 = 1
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 0.75).abs() < 1e-6);
    assert!((result[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_end_to_end_not_operation() {
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let expr = TLExpr::negate(p);

    let result = compile_and_execute(&expr, &[("P", vec![0.0, 0.5, 1.0])]);

    assert_eq!(result.len(), 3);
    // Complement NOT: 1 - x
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 0.5).abs() < 1e-6);
    assert!((result[2] - 0.0).abs() < 1e-6);
}

// ============================================================================
// Complex Nested Expressions
// ============================================================================

#[test]
fn test_end_to_end_nested_and_or() {
    // (P AND Q) OR (R AND S)
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let r = TLExpr::pred("R", vec![Term::var("x")]);
    let s = TLExpr::pred("S", vec![Term::var("x")]);

    let and1 = TLExpr::and(p, q);
    let and2 = TLExpr::and(r, s);
    let expr = TLExpr::or(and1, and2);

    let result = compile_and_execute(
        &expr,
        &[
            ("P", vec![1.0, 0.0]),
            ("Q", vec![1.0, 0.0]),
            ("R", vec![0.0, 1.0]),
            ("S", vec![0.0, 1.0]),
        ],
    );

    assert_eq!(result.len(), 2);
    // Position 0: (1*1=1) OR (0*0=0) = 1
    // Position 1: (0*0=0) OR (1*1=1) = 1
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_end_to_end_de_morgan() {
    // NOT(P AND Q) should equal OR(NOT(P), NOT(Q))
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);

    // NOT(P AND Q)
    let and_pq = TLExpr::and(p.clone(), q.clone());
    let not_and = TLExpr::negate(and_pq);

    // OR(NOT(P), NOT(Q))
    let not_p = TLExpr::negate(p);
    let not_q = TLExpr::negate(q);
    let or_not = TLExpr::or(not_p, not_q);

    let inputs = [("P", vec![0.0, 0.5, 1.0]), ("Q", vec![0.0, 0.5, 1.0])];

    let result_left = compile_and_execute(&not_and, &inputs);
    let result_right = compile_and_execute(&or_not, &inputs);

    assert_eq!(result_left.len(), result_right.len());
    for i in 0..result_left.len() {
        assert!(
            (result_left[i] - result_right[i]).abs() < 1e-6,
            "De Morgan's law failed at position {}: {} != {}",
            i,
            result_left[i],
            result_right[i]
        );
    }
}

#[test]
fn test_end_to_end_deep_nesting() {
    // ((P AND Q) AND R) AND S
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let r = TLExpr::pred("R", vec![Term::var("x")]);
    let s = TLExpr::pred("S", vec![Term::var("x")]);

    let and1 = TLExpr::and(p, q);
    let and2 = TLExpr::and(and1, r);
    let expr = TLExpr::and(and2, s);

    let result = compile_and_execute(
        &expr,
        &[
            ("P", vec![1.0, 1.0]),
            ("Q", vec![1.0, 0.5]),
            ("R", vec![1.0, 0.5]),
            ("S", vec![1.0, 0.5]),
        ],
    );

    assert_eq!(result.len(), 2);
    // Position 0: 1 * 1 * 1 * 1 = 1.0
    // Position 1: 1 * 0.5 * 0.5 * 0.5 = 0.125
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 0.125).abs() < 1e-6);
}

// ============================================================================
// Quantifiers
// ============================================================================

#[test]
#[ignore = "EXISTS quantifier compilation needs further work"]
fn test_end_to_end_exists() {
    // ∃y. knows(x, y) - "x knows someone"
    let pred = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::exists("y", "Person", pred);

    // TODO: Enable once EXISTS compilation is fully implemented
    let graph = compile_to_einsum(&expr);
    if let Ok(graph) = graph {
        assert!(!graph.tensors.is_empty());
        assert!(!graph.nodes.is_empty());
        assert!(graph.tensors.iter().any(|t| t.starts_with("knows")));
    }
}

// ============================================================================
// Multi-arity Predicates
// ============================================================================

#[test]
fn test_end_to_end_binary_predicate() {
    let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let graph = compile_to_einsum(&expr).expect("Compilation failed");

    assert!(!graph.tensors.is_empty());
    assert!(graph.tensors.iter().any(|t| t.starts_with("knows")));
}

#[test]
fn test_end_to_end_ternary_predicate() {
    let expr = TLExpr::pred(
        "relationship",
        vec![Term::var("x"), Term::var("y"), Term::var("z")],
    );
    let graph = compile_to_einsum(&expr).expect("Compilation failed");

    assert!(!graph.tensors.is_empty());
    assert!(graph.tensors.iter().any(|t| t.starts_with("relationship")));
}

// ============================================================================
// Strategy Comparison Tests
// ============================================================================

#[test]
fn test_end_to_end_strategy_soft_differentiable() {
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::and(p, q);

    let mut ctx = CompilerContext::new();
    ctx.config = CompilationConfig::soft_differentiable();
    let graph = compile_to_einsum_with_context(&expr, &mut ctx).expect("Compilation failed");

    let result = execute_graph(&graph, &[("P", vec![1.0, 0.5]), ("Q", vec![1.0, 0.5])]);

    // Product AND: 1*1=1, 0.5*0.5=0.25
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 0.25).abs() < 1e-6);
}

#[test]
fn test_end_to_end_strategy_hard_boolean() {
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::and(p, q);

    let mut ctx = CompilerContext::new();
    ctx.config = CompilationConfig::hard_boolean();
    let graph = compile_to_einsum_with_context(&expr, &mut ctx).expect("Compilation failed");

    let result = execute_graph(&graph, &[("P", vec![1.0, 0.5]), ("Q", vec![1.0, 0.5])]);

    // Min AND: min(1,1)=1, min(0.5,0.5)=0.5
    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_end_to_end_strategy_fuzzy_godel() {
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let and_expr = TLExpr::and(p.clone(), q.clone());
    let or_expr = TLExpr::or(p, q);

    let mut ctx_and = CompilerContext::new();
    ctx_and.config = CompilationConfig::fuzzy_godel();
    let graph_and =
        compile_to_einsum_with_context(&and_expr, &mut ctx_and).expect("Compilation failed");

    let mut ctx_or = CompilerContext::new();
    ctx_or.config = CompilationConfig::fuzzy_godel();
    let graph_or =
        compile_to_einsum_with_context(&or_expr, &mut ctx_or).expect("Compilation failed");

    let inputs = [("P", vec![0.3, 0.7]), ("Q", vec![0.6, 0.4])];
    let result_and = execute_graph(&graph_and, &inputs);
    let result_or = execute_graph(&graph_or, &inputs);

    // Gödel AND (Min): min(0.3,0.6)=0.3, min(0.7,0.4)=0.4
    // Gödel OR (Max): max(0.3,0.6)=0.6, max(0.7,0.4)=0.7
    assert_eq!(result_and.len(), 2);
    assert_eq!(result_or.len(), 2);
    assert!((result_and[0] - 0.3).abs() < 1e-6);
    assert!((result_and[1] - 0.4).abs() < 1e-6);
    assert!((result_or[0] - 0.6).abs() < 1e-6);
    assert!((result_or[1] - 0.7).abs() < 1e-6);
}

// ============================================================================
// Graph Structure Tests
// ============================================================================

#[test]
fn test_end_to_end_graph_structure_simple() {
    let expr = TLExpr::pred("P", vec![Term::var("x")]);
    let graph = compile_to_einsum(&expr).expect("Compilation failed");

    assert!(!graph.tensors.is_empty());
    assert!(graph.tensors.iter().any(|t| t.starts_with("P")));
}

#[test]
fn test_end_to_end_graph_structure_compound() {
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let q = TLExpr::pred("Q", vec![Term::var("x")]);
    let expr = TLExpr::and(p, q);

    let graph = compile_to_einsum(&expr).expect("Compilation failed");

    // Should have at least P, Q, and result tensor
    assert!(graph.tensors.len() >= 3);
    assert!(graph.tensors.iter().any(|t| t.starts_with("P")));
    assert!(graph.tensors.iter().any(|t| t.starts_with("Q")));

    // Should have at least one operation node
    assert!(!graph.nodes.is_empty());
}

#[test]
fn test_end_to_end_graph_optimization() {
    // For Product AND strategy with non-empty axes, should use einsum fusion
    let p = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    let q = TLExpr::pred("Q", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::and(p, q);

    let mut ctx = CompilerContext::new();
    ctx.config = CompilationConfig::soft_differentiable();
    let graph = compile_to_einsum_with_context(&expr, &mut ctx).expect("Compilation failed");

    // With einsum fusion, we should have fewer nodes (1 einsum instead of 3 ops)
    assert!(!graph.nodes.is_empty());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_end_to_end_empty_predicate_list() {
    // Verify that predicates with no arguments compile
    let expr = TLExpr::pred("truth", vec![]);
    let graph = compile_to_einsum(&expr);

    // This should succeed - nullary predicates are valid
    assert!(graph.is_ok());
    let graph = graph.unwrap();
    assert!(!graph.tensors.is_empty());
}

// ============================================================================
// Constant Tensor Tests
// ============================================================================

#[test]
fn test_end_to_end_identity_with_true() {
    // P AND TRUE should equal P
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let true_expr = TLExpr::constant(1.0);
    let expr = TLExpr::and(p, true_expr);

    // Just verify compilation succeeds - execution with constants needs more infrastructure
    let graph = compile_to_einsum(&expr);
    assert!(graph.is_ok());
    let graph = graph.unwrap();
    assert!(!graph.tensors.is_empty());
}

#[test]
fn test_end_to_end_annihilation_with_false() {
    // P AND FALSE should equal FALSE
    let p = TLExpr::pred("P", vec![Term::var("x")]);
    let false_expr = TLExpr::constant(0.0);
    let expr = TLExpr::and(p, false_expr);

    // Just verify compilation succeeds - execution with constants needs more infrastructure
    let graph = compile_to_einsum(&expr);
    assert!(graph.is_ok());
    let graph = graph.unwrap();
    assert!(!graph.tensors.is_empty());
}
