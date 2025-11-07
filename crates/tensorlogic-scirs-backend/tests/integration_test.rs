//! Integration test: Compile TLExpr → EinsumGraph → Execute with SciRS2
//!
//! **Note**: These tests require the `integration-tests` feature to avoid
//! circular dev-dependencies with tensorlogic-compiler.

#![cfg(feature = "integration-tests")]

use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_scirs_backend::Scirs2Exec;

#[test]
fn test_simple_predicate_execution() {
    // Define: knows(x, y)
    let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute (input tensor should be provided externally in real usage)
    let mut executor = Scirs2Exec::new();

    // Add a 3x3 "knows" relation matrix
    let knows_matrix = Scirs2Exec::from_vec(
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        vec![3, 3],
    )
    .unwrap();

    // Use dynamic tensor name from the compiled graph
    executor.add_tensor(graph.tensors[0].clone(), knows_matrix);

    // Forward pass
    let result = executor.forward(&graph).unwrap();

    // Verify shape
    assert_eq!(result.shape(), &[3, 3]);
}

#[test]
fn test_exists_quantifier_execution() {
    // Define: ∃y. knows(x, y) - "Find persons who know someone"
    use tensorlogic_compiler::CompilerContext;

    let expr = TLExpr::exists(
        "y",
        "Person",
        TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]),
    );

    // Compile with domain context
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 3); // 3 persons
    ctx.bind_var("x", "Person").unwrap();
    ctx.bind_var("y", "Person").unwrap();

    let graph = tensorlogic_compiler::compile_to_einsum_with_context(&expr, &mut ctx).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    // Input: 3 persons, knows matrix
    let knows_matrix = Scirs2Exec::from_vec(
        vec![
            1.0, 0.0, 1.0, // Person 0 knows 0 and 2
            0.0, 1.0, 0.0, // Person 1 knows only 1
            0.0, 0.0, 0.0, // Person 2 knows nobody
        ],
        vec![3, 3],
    )
    .unwrap();

    executor.add_tensor("knows[ab]", knows_matrix);

    // Forward pass (should sum over axis 1 - the 'y' dimension)
    let result = executor.forward(&graph).unwrap();

    // Verify shape: should be [3] (one value per person)
    assert_eq!(result.shape(), &[3]);

    // Person 0 knows 2 people → sum = 2.0
    // Person 1 knows 1 person → sum = 1.0
    // Person 2 knows 0 people → sum = 0.0
    assert_eq!(result[[0]], 2.0);
    assert_eq!(result[[1]], 1.0);
    assert_eq!(result[[2]], 0.0);
}

#[test]
fn test_and_operation_execution() {
    // Simplified test: P(x, y) ∧ Q(x, y) - same axes, element-wise
    let p = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    let q = TLExpr::pred("Q", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::and(p, q);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();
    println!("Graph tensors: {:?}", graph.tensors);
    println!("Graph nodes: {:?}", graph.nodes);

    // Execute
    let mut executor = Scirs2Exec::new();

    // Input: P and Q matrices
    let p_matrix = Scirs2Exec::from_vec(vec![1.0, 0.5, 0.5, 1.0], vec![2, 2]).unwrap();

    let q_matrix = Scirs2Exec::from_vec(vec![0.8, 0.6, 0.6, 0.8], vec![2, 2]).unwrap();

    // Use dynamic tensor names from the compiled graph
    executor.add_tensor(graph.tensors[0].clone(), p_matrix);
    executor.add_tensor(graph.tensors[1].clone(), q_matrix);

    // Forward pass (einsum: "ab,ab->ab", element-wise product)
    let result = executor.forward(&graph).unwrap();

    // Verify shape: should be [2, 2] (x, y dimensions)
    assert_eq!(result.shape(), &[2, 2]);

    // Verify element-wise multiplication: P * Q
    // [1.0*0.8, 0.5*0.6, 0.5*0.6, 1.0*0.8]
    assert!((result[[0, 0]] - 0.8).abs() < 1e-6);
    assert!((result[[0, 1]] - 0.3).abs() < 1e-6);
    assert!((result[[1, 0]] - 0.3).abs() < 1e-6);
    assert!((result[[1, 1]] - 0.8).abs() < 1e-6);
}

#[test]
fn test_implication_execution() {
    // Define: knows(x, y) → friends(x, y)
    let knows = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let friends = TLExpr::pred("friends", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::imply(knows, friends);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();
    println!("Implication graph tensors: {:?}", graph.tensors);
    println!("Implication graph nodes: {:?}", graph.nodes);

    // Execute
    let mut executor = Scirs2Exec::new();

    // Inputs
    let knows_matrix = Scirs2Exec::from_vec(vec![0.8, 0.2, 0.1, 0.9], vec![2, 2]).unwrap();

    let friends_matrix = Scirs2Exec::from_vec(vec![0.9, 0.1, 0.0, 1.0], vec![2, 2]).unwrap();

    // Match compiler-generated names
    executor.add_tensor(graph.tensors[0].clone(), knows_matrix);
    executor.add_tensor(graph.tensors[1].clone(), friends_matrix);

    // Forward pass (ReLU(friends - knows))
    let result = executor.forward(&graph).unwrap();

    // Verify shape
    assert_eq!(result.shape(), &[2, 2]);

    // Result should be ReLU(friends - knows)
    // [0.9-0.8, 0.1-0.2, 0.0-0.1, 1.0-0.9]
    // = ReLU([0.1, -0.1, -0.1, 0.1])
    // = [0.1, 0.0, 0.0, 0.1]
    assert!((result[[0, 0]] - 0.1).abs() < 1e-6);
    assert_eq!(result[[0, 1]], 0.0);
    assert_eq!(result[[1, 0]], 0.0);
    assert!((result[[1, 1]] - 0.1).abs() < 1e-6);
}

#[test]
fn test_arithmetic_add_execution() {
    // Define: score(x, y) + bonus(x, y)
    let score = TLExpr::pred("score", vec![Term::var("x"), Term::var("y")]);
    let bonus = TLExpr::pred("bonus", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::add(score, bonus);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    let score_matrix = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let bonus_matrix = Scirs2Exec::from_vec(vec![0.5, 0.5, 1.0, 1.0], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), score_matrix);
    executor.add_tensor(graph.tensors[1].clone(), bonus_matrix);

    let result = executor.forward(&graph).unwrap();

    // Verify element-wise addition
    assert_eq!(result.shape(), &[2, 2]);
    assert!((result[[0, 0]] - 1.5).abs() < 1e-6);
    assert!((result[[0, 1]] - 2.5).abs() < 1e-6);
    assert!((result[[1, 0]] - 4.0).abs() < 1e-6);
    assert!((result[[1, 1]] - 5.0).abs() < 1e-6);
}

#[test]
fn test_arithmetic_sub_execution() {
    // Define: total(x, y) - penalty(x, y)
    let total = TLExpr::pred("total", vec![Term::var("x"), Term::var("y")]);
    let penalty = TLExpr::pred("penalty", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::sub(total, penalty);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    let total_matrix = Scirs2Exec::from_vec(vec![10.0, 8.0, 6.0, 4.0], vec![2, 2]).unwrap();
    let penalty_matrix = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 1.0], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), total_matrix);
    executor.add_tensor(graph.tensors[1].clone(), penalty_matrix);

    let result = executor.forward(&graph).unwrap();

    // Verify element-wise subtraction
    assert_eq!(result.shape(), &[2, 2]);
    assert!((result[[0, 0]] - 9.0).abs() < 1e-6);
    assert!((result[[0, 1]] - 6.0).abs() < 1e-6);
    assert!((result[[1, 0]] - 3.0).abs() < 1e-6);
    assert!((result[[1, 1]] - 3.0).abs() < 1e-6);
}

#[test]
fn test_arithmetic_mul_execution() {
    // Define: weight(x, y) * value(x, y)
    let weight = TLExpr::pred("weight", vec![Term::var("x"), Term::var("y")]);
    let value = TLExpr::pred("value", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::mul(weight, value);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    let weight_matrix = Scirs2Exec::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
    let value_matrix = Scirs2Exec::from_vec(vec![1.5, 2.0, 2.5, 3.0], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), weight_matrix);
    executor.add_tensor(graph.tensors[1].clone(), value_matrix);

    let result = executor.forward(&graph).unwrap();

    // Verify element-wise multiplication
    assert_eq!(result.shape(), &[2, 2]);
    assert!((result[[0, 0]] - 3.0).abs() < 1e-6);
    assert!((result[[0, 1]] - 6.0).abs() < 1e-6);
    assert!((result[[1, 0]] - 10.0).abs() < 1e-6);
    assert!((result[[1, 1]] - 15.0).abs() < 1e-6);
}

#[test]
fn test_arithmetic_div_execution() {
    // Define: numerator(x, y) / denominator(x, y)
    let numerator = TLExpr::pred("numerator", vec![Term::var("x"), Term::var("y")]);
    let denominator = TLExpr::pred("denominator", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::div(numerator, denominator);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    let numerator_matrix = Scirs2Exec::from_vec(vec![10.0, 15.0, 20.0, 25.0], vec![2, 2]).unwrap();
    let denominator_matrix = Scirs2Exec::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), numerator_matrix);
    executor.add_tensor(graph.tensors[1].clone(), denominator_matrix);

    let result = executor.forward(&graph).unwrap();

    // Verify element-wise division
    assert_eq!(result.shape(), &[2, 2]);
    assert!((result[[0, 0]] - 5.0).abs() < 1e-6);
    assert!((result[[0, 1]] - 5.0).abs() < 1e-6);
    assert!((result[[1, 0]] - 5.0).abs() < 1e-6);
    assert!((result[[1, 1]] - 5.0).abs() < 1e-6);
}

#[test]
fn test_comparison_eq_execution() {
    // Define: score(x, y) == target(x, y)
    let score = TLExpr::pred("score", vec![Term::var("x"), Term::var("y")]);
    let target = TLExpr::pred("target", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::eq(score, target);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    let score_matrix = Scirs2Exec::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let target_matrix = Scirs2Exec::from_vec(vec![1.0, 2.5, 3.0, 4.5], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), score_matrix);
    executor.add_tensor(graph.tensors[1].clone(), target_matrix);

    let result = executor.forward(&graph).unwrap();

    // Verify comparison returns 0.0 or 1.0
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 1.0); // 1.0 == 1.0 → 1.0
    assert_eq!(result[[0, 1]], 0.0); // 2.0 == 2.5 → 0.0
    assert_eq!(result[[1, 0]], 1.0); // 3.0 == 3.0 → 1.0
    assert_eq!(result[[1, 1]], 0.0); // 4.0 == 4.5 → 0.0
}

#[test]
fn test_comparison_lt_execution() {
    // Define: actual(x, y) < threshold(x, y)
    let actual = TLExpr::pred("actual", vec![Term::var("x"), Term::var("y")]);
    let threshold = TLExpr::pred("threshold", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::lt(actual, threshold);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    let actual_matrix = Scirs2Exec::from_vec(vec![1.0, 3.0, 5.0, 7.0], vec![2, 2]).unwrap();
    let threshold_matrix = Scirs2Exec::from_vec(vec![2.0, 3.0, 4.0, 8.0], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), actual_matrix);
    executor.add_tensor(graph.tensors[1].clone(), threshold_matrix);

    let result = executor.forward(&graph).unwrap();

    // Verify less-than comparison
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 1.0); // 1.0 < 2.0 → 1.0
    assert_eq!(result[[0, 1]], 0.0); // 3.0 < 3.0 → 0.0
    assert_eq!(result[[1, 0]], 0.0); // 5.0 < 4.0 → 0.0
    assert_eq!(result[[1, 1]], 1.0); // 7.0 < 8.0 → 1.0
}

#[test]
fn test_comparison_gt_execution() {
    // Define: score(x, y) > minimum(x, y)
    let score = TLExpr::pred("score", vec![Term::var("x"), Term::var("y")]);
    let minimum = TLExpr::pred("minimum", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::gt(score, minimum);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    let score_matrix = Scirs2Exec::from_vec(vec![5.0, 3.0, 2.0, 8.0], vec![2, 2]).unwrap();
    let minimum_matrix = Scirs2Exec::from_vec(vec![4.0, 3.0, 3.0, 7.0], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), score_matrix);
    executor.add_tensor(graph.tensors[1].clone(), minimum_matrix);

    let result = executor.forward(&graph).unwrap();

    // Verify greater-than comparison
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 1.0); // 5.0 > 4.0 → 1.0
    assert_eq!(result[[0, 1]], 0.0); // 3.0 > 3.0 → 0.0
    assert_eq!(result[[1, 0]], 0.0); // 2.0 > 3.0 → 0.0
    assert_eq!(result[[1, 1]], 1.0); // 8.0 > 7.0 → 1.0
}

#[test]
fn test_comparison_lte_execution() {
    // Define: value(x, y) <= limit(x, y)
    let value = TLExpr::pred("value", vec![Term::var("x"), Term::var("y")]);
    let limit = TLExpr::pred("limit", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::lte(value, limit);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    let value_matrix = Scirs2Exec::from_vec(vec![1.0, 5.0, 5.0, 9.0], vec![2, 2]).unwrap();
    let limit_matrix = Scirs2Exec::from_vec(vec![2.0, 5.0, 4.0, 8.0], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), value_matrix);
    executor.add_tensor(graph.tensors[1].clone(), limit_matrix);

    let result = executor.forward(&graph).unwrap();

    // Verify less-than-or-equal comparison
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 1.0); // 1.0 <= 2.0 → 1.0
    assert_eq!(result[[0, 1]], 1.0); // 5.0 <= 5.0 → 1.0
    assert_eq!(result[[1, 0]], 0.0); // 5.0 <= 4.0 → 0.0
    assert_eq!(result[[1, 1]], 0.0); // 9.0 <= 8.0 → 0.0
}

#[test]
fn test_comparison_gte_execution() {
    // Define: result(x, y) >= threshold(x, y)
    let result_pred = TLExpr::pred("result", vec![Term::var("x"), Term::var("y")]);
    let threshold = TLExpr::pred("threshold", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::gte(result_pred, threshold);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();

    // Execute
    let mut executor = Scirs2Exec::new();

    let result_matrix = Scirs2Exec::from_vec(vec![8.0, 5.0, 5.0, 3.0], vec![2, 2]).unwrap();
    let threshold_matrix = Scirs2Exec::from_vec(vec![7.0, 5.0, 6.0, 4.0], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), result_matrix);
    executor.add_tensor(graph.tensors[1].clone(), threshold_matrix);

    let result = executor.forward(&graph).unwrap();

    // Verify greater-than-or-equal comparison
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 1.0); // 8.0 >= 7.0 → 1.0
    assert_eq!(result[[0, 1]], 1.0); // 5.0 >= 5.0 → 1.0
    assert_eq!(result[[1, 0]], 0.0); // 5.0 >= 6.0 → 0.0
    assert_eq!(result[[1, 1]], 0.0); // 3.0 >= 4.0 → 0.0
}

#[test]
fn test_conditional_if_then_else_execution() {
    // Define: if condition(x, y) then value_a(x, y) else value_b(x, y)
    let condition = TLExpr::pred("condition", vec![Term::var("x"), Term::var("y")]);
    let value_a = TLExpr::pred("value_a", vec![Term::var("x"), Term::var("y")]);
    let value_b = TLExpr::pred("value_b", vec![Term::var("x"), Term::var("y")]);
    let expr = TLExpr::if_then_else(condition, value_a, value_b);

    // Compile
    let graph = compile_to_einsum(&expr).unwrap();
    println!("Conditional graph tensors: {:?}", graph.tensors);
    println!("Conditional graph nodes: {:?}", graph.nodes);

    // Execute
    let mut executor = Scirs2Exec::new();

    // Condition: 0.0 or 1.0 (false or true)
    let condition_matrix = Scirs2Exec::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let value_a_matrix = Scirs2Exec::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
    let value_b_matrix = Scirs2Exec::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    executor.add_tensor(graph.tensors[0].clone(), condition_matrix);
    executor.add_tensor(graph.tensors[1].clone(), value_a_matrix);
    executor.add_tensor(graph.tensors[2].clone(), value_b_matrix);

    // Forward pass: condition * value_a + (1 - condition) * value_b
    let result = executor.forward(&graph).unwrap();

    // Verify conditional selection
    assert_eq!(result.shape(), &[2, 2]);
    assert!((result[[0, 0]] - 10.0).abs() < 1e-6); // condition=1 → value_a=10.0
    assert!((result[[0, 1]] - 6.0).abs() < 1e-6); // condition=0 → value_b=6.0
    assert!((result[[1, 0]] - 7.0).abs() < 1e-6); // condition=0 → value_b=7.0
    assert!((result[[1, 1]] - 40.0).abs() < 1e-6); // condition=1 → value_a=40.0
}
