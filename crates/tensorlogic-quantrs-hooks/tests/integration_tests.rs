//! Integration tests for tensorlogic-quantrs-hooks

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::Array;
use tensorlogic_quantrs_hooks::*;

/// Test a simple chain graph: X -> Y -> Z
#[test]
fn test_chain_graph() {
    let mut graph = FactorGraph::new();

    // Add variables
    graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
    graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

    // Add factor P(X)
    let px_values = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
        .unwrap()
        .into_dyn();
    let px = Factor::new("P(X)".to_string(), vec!["x".to_string()], px_values).unwrap();
    graph.add_factor(px).unwrap();

    // Add factor P(Y|X)
    let py_given_x_values = Array::from_shape_vec(
        vec![2, 2],
        vec![
            0.8, 0.2, // P(Y|X=0)
            0.3, 0.7, // P(Y|X=1)
        ],
    )
    .unwrap()
    .into_dyn();
    let py_given_x = Factor::new(
        "P(Y|X)".to_string(),
        vec!["x".to_string(), "y".to_string()],
        py_given_x_values,
    )
    .unwrap();
    graph.add_factor(py_given_x).unwrap();

    // Run inference
    let algorithm = Box::new(SumProductAlgorithm::default());
    let engine = InferenceEngine::new(graph, algorithm);

    // Compute marginals
    let query_x = MarginalizationQuery {
        variable: "x".to_string(),
    };
    let marginal_x = engine.marginalize(&query_x);
    assert!(marginal_x.is_ok());

    let marginal_x_values = marginal_x.unwrap();
    let sum: f64 = marginal_x_values.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
}

/// Test factor product operation
#[test]
fn test_factor_product_integration() {
    let f1_values = Array::from_shape_vec(vec![2], vec![0.6, 0.4])
        .unwrap()
        .into_dyn();
    let f1 = Factor::new("f1".to_string(), vec!["x".to_string()], f1_values).unwrap();

    let f2_values = Array::from_shape_vec(vec![2], vec![0.7, 0.3])
        .unwrap()
        .into_dyn();
    let f2 = Factor::new("f2".to_string(), vec!["y".to_string()], f2_values).unwrap();

    let product = f1.product(&f2).unwrap();

    assert_eq!(product.variables.len(), 2);
    assert_eq!(product.values.shape(), &[2, 2]);

    // Values should be outer product
    assert_abs_diff_eq!(product.values[[0, 0]], 0.6 * 0.7, epsilon = 1e-10);
    assert_abs_diff_eq!(product.values[[0, 1]], 0.6 * 0.3, epsilon = 1e-10);
    assert_abs_diff_eq!(product.values[[1, 0]], 0.4 * 0.7, epsilon = 1e-10);
    assert_abs_diff_eq!(product.values[[1, 1]], 0.4 * 0.3, epsilon = 1e-10);
}

/// Test factor marginalization
#[test]
fn test_factor_marginalization_integration() {
    // Joint distribution P(X, Y)
    let joint_values = Array::from_shape_vec(
        vec![2, 2],
        vec![
            0.24, 0.06, // X=0: Y=0, Y=1
            0.56, 0.14, // X=1: Y=0, Y=1
        ],
    )
    .unwrap()
    .into_dyn();
    let joint = Factor::new(
        "P(X,Y)".to_string(),
        vec!["x".to_string(), "y".to_string()],
        joint_values,
    )
    .unwrap();

    // Marginalize out Y to get P(X)
    let marginal_x = joint.marginalize_out("y").unwrap();

    assert_eq!(marginal_x.variables.len(), 1);
    assert_eq!(marginal_x.variables[0], "x");

    // P(X=0) = 0.24 + 0.06 = 0.30
    // P(X=1) = 0.56 + 0.14 = 0.70
    assert_abs_diff_eq!(marginal_x.values[[0]], 0.30, epsilon = 1e-10);
    assert_abs_diff_eq!(marginal_x.values[[1]], 0.70, epsilon = 1e-10);
}

/// Test factor reduce (evidence)
#[test]
fn test_factor_reduce_integration() {
    let joint_values = Array::from_shape_vec(
        vec![2, 2],
        vec![
            0.24, 0.06, // X=0
            0.56, 0.14, // X=1
        ],
    )
    .unwrap()
    .into_dyn();
    let joint = Factor::new(
        "P(X,Y)".to_string(),
        vec!["x".to_string(), "y".to_string()],
        joint_values,
    )
    .unwrap();

    // Reduce with evidence Y=0
    let reduced = joint.reduce("y", 0).unwrap();

    assert_eq!(reduced.variables.len(), 1);
    assert_eq!(reduced.variables[0], "x");

    // Should extract column Y=0: [0.24, 0.56]
    assert_abs_diff_eq!(reduced.values[[0]], 0.24, epsilon = 1e-10);
    assert_abs_diff_eq!(reduced.values[[1]], 0.56, epsilon = 1e-10);
}

/// Test message passing convergence
#[test]
fn test_message_passing_convergence() {
    let mut graph = FactorGraph::new();
    graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

    let algorithm = Box::new(SumProductAlgorithm::new(10, 1e-6, 0.0));
    let engine = InferenceEngine::new(graph, algorithm);

    let query = MarginalizationQuery {
        variable: "x".to_string(),
    };
    let result = engine.marginalize(&query);

    assert!(result.is_ok());
    let marginal = result.unwrap();

    // Should be uniform for single variable with no factors
    assert_abs_diff_eq!(marginal[[0]], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(marginal[[1]], 0.5, epsilon = 1e-6);
}

/// Test TLExpr to factor graph conversion
#[test]
fn test_tlexpr_to_factor_graph() {
    use tensorlogic_ir::{TLExpr, Term};

    // Create expression: P(x) ∧ Q(y)
    let expr = TLExpr::and(
        TLExpr::pred("P", vec![Term::var("x")]),
        TLExpr::pred("Q", vec![Term::var("y")]),
    );

    let graph = tensorlogic_quantrs_hooks::expr_to_factor_graph(&expr);
    assert!(graph.is_ok());

    let graph = graph.unwrap();
    assert_eq!(graph.num_variables(), 2);
    assert_eq!(graph.num_factors(), 2);
}

/// Test TLExpr with quantifier
#[test]
fn test_tlexpr_with_exists() {
    use tensorlogic_ir::{TLExpr, Term};

    // ∃x. P(x)
    let expr = TLExpr::exists("x", "Domain", TLExpr::pred("P", vec![Term::var("x")]));

    let graph = tensorlogic_quantrs_hooks::expr_to_factor_graph(&expr);
    assert!(graph.is_ok());

    let graph = graph.unwrap();
    assert!(graph.num_variables() > 0);
}

/// Test marginalization query
#[test]
fn test_marginalization_lib_function() {
    use scirs2_core::ndarray::Array;

    let joint = Array::from_shape_vec(
        vec![2, 2],
        vec![
            0.25, 0.25, // X=0
            0.25, 0.25, // X=1
        ],
    )
    .unwrap()
    .into_dyn();

    // Marginalize out Y (axis 1)
    let marginal = tensorlogic_quantrs_hooks::marginalize(&joint, 0, &[0, 1]).unwrap();

    assert_eq!(marginal.ndim(), 1);
    let sum: f64 = marginal.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
}

/// Test conditioning
#[test]
fn test_conditioning_lib_function() {
    use scirs2_core::ndarray::Array;
    use std::collections::HashMap;

    let joint = Array::from_shape_vec(
        vec![2, 2],
        vec![
            0.2, 0.3, // X=0
            0.4, 0.1, // X=1
        ],
    )
    .unwrap()
    .into_dyn();

    let mut evidence = HashMap::new();
    evidence.insert(1, 0); // Y=0

    let conditional = tensorlogic_quantrs_hooks::condition(&joint, &evidence).unwrap();

    // Should have one dimension less
    assert_eq!(conditional.ndim(), 1);

    // Should be normalized
    let sum: f64 = conditional.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

    // P(X|Y=0) ∝ [0.2, 0.4] → [1/3, 2/3]
    assert_abs_diff_eq!(conditional[[0]], 1.0 / 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(conditional[[1]], 2.0 / 3.0, epsilon = 1e-6);
}

/// Test joint probability computation
#[test]
fn test_joint_computation() {
    let mut graph = FactorGraph::new();
    graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);
    graph.add_variable_with_card("y".to_string(), "Binary".to_string(), 2);

    let algorithm = Box::new(SumProductAlgorithm::default());
    let engine = InferenceEngine::new(graph, algorithm);

    let joint = engine.joint();
    assert!(joint.is_ok());

    let joint_dist = joint.unwrap();
    let sum: f64 = joint_dist.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
}

/// Test loopy belief propagation with damping
#[test]
fn test_loopy_bp_with_damping() {
    let mut graph = FactorGraph::new();
    graph.add_variable_with_card("x".to_string(), "Binary".to_string(), 2);

    // Use damping factor 0.5
    let algorithm = Box::new(SumProductAlgorithm::new(50, 1e-6, 0.5));
    let engine = InferenceEngine::new(graph, algorithm);

    let query = MarginalizationQuery {
        variable: "x".to_string(),
    };
    let result = engine.marginalize(&query);

    assert!(result.is_ok());
}

/// Test factor division
#[test]
fn test_factor_division() {
    let f1_values = Array::from_shape_vec(vec![2], vec![0.6, 0.8])
        .unwrap()
        .into_dyn();
    let f1 = Factor::new("f1".to_string(), vec!["x".to_string()], f1_values).unwrap();

    let f2_values = Array::from_shape_vec(vec![2], vec![0.3, 0.4])
        .unwrap()
        .into_dyn();
    let f2 = Factor::new("f2".to_string(), vec!["x".to_string()], f2_values).unwrap();

    let result = f1.divide(&f2).unwrap();

    assert_abs_diff_eq!(result.values[[0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result.values[[1]], 2.0, epsilon = 1e-10);
}

/// Test message passing with actual factors
#[test]
fn test_message_passing_with_factors() {
    let mut graph = FactorGraph::new();
    graph.add_variable_with_card("var_0".to_string(), "Binary".to_string(), 2);

    // Add a factor
    let factor_values = Array::from_shape_vec(vec![2], vec![0.7, 0.3])
        .unwrap()
        .into_dyn();
    let factor = Factor::new(
        "factor_0".to_string(),
        vec!["var_0".to_string()],
        factor_values,
    )
    .unwrap();
    graph.add_factor(factor).unwrap();

    let algorithm = SumProductAlgorithm::default();
    let result = algorithm.run(&graph);

    assert!(result.is_ok());
    let beliefs = result.unwrap();

    if let Some(belief) = beliefs.get("var_0") {
        let sum: f64 = belief.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        // Should reflect the factor distribution (normalized)
        assert_abs_diff_eq!(belief[[0]], 0.7, epsilon = 1e-6);
        assert_abs_diff_eq!(belief[[1]], 0.3, epsilon = 1e-6);
    }
}
