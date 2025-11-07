//! Unit tests for the IR.

use crate::{
    expr::TLExpr,
    graph::{EinsumGraph, EinsumNode},
    term::Term,
};

#[test]
fn test_term_constructors() {
    let var = Term::var("x");
    assert!(var.is_var());
    assert_eq!(var.name(), "x");

    let constant = Term::constant("alice");
    assert!(constant.is_const());
    assert_eq!(constant.name(), "alice");
}

#[test]
fn test_tlexpr_builders() {
    let pred = TLExpr::pred("Parent", vec![Term::var("x"), Term::var("y")]);
    let pred2 = TLExpr::pred("Child", vec![Term::var("y"), Term::var("x")]);

    let and_expr = TLExpr::and(pred.clone(), pred2);
    assert!(matches!(and_expr, TLExpr::And(_, _)));

    let or_expr = TLExpr::or(pred.clone(), TLExpr::negate(pred.clone()));
    assert!(matches!(or_expr, TLExpr::Or(_, _)));
}

#[test]
fn test_free_vars() {
    let pred = TLExpr::pred("P", vec![Term::var("x"), Term::var("y")]);
    let free = pred.free_vars();
    assert_eq!(free.len(), 2);
    assert!(free.contains("x"));
    assert!(free.contains("y"));

    let exists_expr = TLExpr::exists("x", "Domain", pred.clone());
    let free2 = exists_expr.free_vars();
    assert_eq!(free2.len(), 1);
    assert!(free2.contains("y"));
    assert!(!free2.contains("x"));
}

#[test]
fn test_all_predicates() {
    let p1 = TLExpr::pred("Parent", vec![Term::var("x"), Term::var("y")]);
    let p2 = TLExpr::pred("Age", vec![Term::var("x")]);
    let expr = TLExpr::and(p1, p2);

    let preds = expr.all_predicates();
    assert_eq!(preds.len(), 2);
    assert_eq!(preds.get("Parent"), Some(&2));
    assert_eq!(preds.get("Age"), Some(&1));
}

#[test]
fn test_einsum_graph_builder() {
    let mut graph = EinsumGraph::new();

    let t0 = graph.add_tensor("A");
    let t1 = graph.add_tensor("B");
    let t2 = graph.add_tensor("C");

    assert_eq!(t0, 0);
    assert_eq!(t1, 1);
    assert_eq!(t2, 2);

    let node = EinsumNode::new("ij,jk->ik", vec![t0, t1], vec![t2]);
    let node_idx = graph.add_node(node).unwrap();
    assert_eq!(node_idx, 0);

    graph.add_output(t2).unwrap();
    assert_eq!(graph.outputs, vec![2]);

    assert!(graph.validate().is_ok());
}

#[test]
fn test_einsum_node_validation() {
    let node = EinsumNode::new("ij,jk->ik", vec![0, 1], vec![2]);
    assert!(node.validate(3).is_ok());
    assert!(node.validate(2).is_err()); // output idx 2 is out of bounds for num_tensors=2

    let empty_spec = EinsumNode::new("", vec![0], vec![1]);
    assert!(empty_spec.validate(2).is_err());

    // Test with invalid output index
    let invalid_output_node = EinsumNode::new("i->i", vec![0], vec![10]);
    assert!(invalid_output_node.validate(2).is_err());
}

#[test]
fn test_graph_validation() {
    let mut graph = EinsumGraph::new();
    graph.add_tensor("A");
    graph.add_tensor("B");
    let t2 = graph.add_tensor("C");

    let invalid_node = EinsumNode::new("ij,jk->ik", vec![0, 5], vec![t2]);
    assert!(graph.add_node(invalid_node).is_err());

    // Valid node
    let valid_node = EinsumNode::new("ij,jk->ik", vec![0, 1], vec![t2]);
    assert!(graph.add_node(valid_node).is_ok());

    assert!(graph.add_output(10).is_err());
    assert!(graph.add_output(0).is_ok());
}

#[test]
fn test_arithmetic_operations() {
    let const1 = TLExpr::constant(5.0);
    let const2 = TLExpr::constant(3.0);

    let add_expr = TLExpr::add(const1.clone(), const2.clone());
    assert!(matches!(add_expr, TLExpr::Add(_, _)));

    let sub_expr = TLExpr::sub(const1.clone(), const2.clone());
    assert!(matches!(sub_expr, TLExpr::Sub(_, _)));

    let mul_expr = TLExpr::mul(const1.clone(), const2.clone());
    assert!(matches!(mul_expr, TLExpr::Mul(_, _)));

    let div_expr = TLExpr::div(const1, const2);
    assert!(matches!(div_expr, TLExpr::Div(_, _)));
}

#[test]
fn test_comparison_operations() {
    let pred1 = TLExpr::pred("Score", vec![Term::var("x")]);
    let const1 = TLExpr::constant(10.0);

    let eq_expr = TLExpr::eq(pred1.clone(), const1.clone());
    assert!(matches!(eq_expr, TLExpr::Eq(_, _)));

    let lt_expr = TLExpr::lt(pred1.clone(), const1.clone());
    assert!(matches!(lt_expr, TLExpr::Lt(_, _)));

    let gt_expr = TLExpr::gt(pred1.clone(), const1.clone());
    assert!(matches!(gt_expr, TLExpr::Gt(_, _)));

    let lte_expr = TLExpr::lte(pred1.clone(), const1.clone());
    assert!(matches!(lte_expr, TLExpr::Lte(_, _)));

    let gte_expr = TLExpr::gte(pred1, const1);
    assert!(matches!(gte_expr, TLExpr::Gte(_, _)));
}

#[test]
fn test_conditional_expression() {
    let condition = TLExpr::pred("IsAdult", vec![Term::var("x")]);
    let then_branch = TLExpr::constant(1.0);
    let else_branch = TLExpr::constant(0.0);

    let if_expr = TLExpr::if_then_else(condition, then_branch, else_branch);
    assert!(matches!(if_expr, TLExpr::IfThenElse { .. }));
}

#[test]
fn test_constant_expression() {
    let const_expr = TLExpr::constant(42.5);
    assert!(matches!(const_expr, TLExpr::Constant(v) if v == 42.5));

    // Constants should have no free variables
    let free = const_expr.free_vars();
    assert_eq!(free.len(), 0);

    // Constants should have no predicates
    let preds = const_expr.all_predicates();
    assert_eq!(preds.len(), 0);
}

#[test]
fn test_complex_arithmetic_expression() {
    // (x + y) * (x - y)
    let x = TLExpr::pred("X", vec![Term::var("i")]);
    let y = TLExpr::pred("Y", vec![Term::var("i")]);

    let sum = TLExpr::add(x.clone(), y.clone());
    let diff = TLExpr::sub(x, y);
    let product = TLExpr::mul(sum, diff);

    // Should still track free variables correctly
    let free = product.free_vars();
    assert_eq!(free.len(), 1);
    assert!(free.contains("i"));

    // Should track predicates
    let preds = product.all_predicates();
    assert_eq!(preds.len(), 2);
    assert_eq!(preds.get("X"), Some(&1));
    assert_eq!(preds.get("Y"), Some(&1));
}

#[test]
fn test_comparison_free_vars() {
    let pred = TLExpr::pred("Score", vec![Term::var("x"), Term::var("y")]);
    let threshold = TLExpr::constant(0.5);

    let comparison = TLExpr::gt(pred, threshold);

    // Should only have free vars from pred, not from constant
    let free = comparison.free_vars();
    assert_eq!(free.len(), 2);
    assert!(free.contains("x"));
    assert!(free.contains("y"));
}

#[test]
fn test_conditional_free_vars() {
    let condition = TLExpr::pred("P", vec![Term::var("x")]);
    let then_branch = TLExpr::pred("Q", vec![Term::var("y")]);
    let else_branch = TLExpr::pred("R", vec![Term::var("z")]);

    let if_expr = TLExpr::if_then_else(condition, then_branch, else_branch);

    // Should collect free vars from all three branches
    let free = if_expr.free_vars();
    assert_eq!(free.len(), 3);
    assert!(free.contains("x"));
    assert!(free.contains("y"));
    assert!(free.contains("z"));
}

#[test]
fn test_arity_validation_with_arithmetic() {
    // Same predicate used in arithmetic should still validate arity
    let p1 = TLExpr::pred("Score", vec![Term::var("x")]);
    let p2 = TLExpr::pred("Score", vec![Term::var("y")]);

    let sum = TLExpr::add(p1, p2);
    assert!(sum.validate_arity().is_ok());

    // Mismatched arity should fail
    let p3 = TLExpr::pred("Score", vec![Term::var("x")]);
    let p4 = TLExpr::pred("Score", vec![Term::var("y"), Term::var("z")]);

    let bad_sum = TLExpr::add(p3, p4);
    assert!(bad_sum.validate_arity().is_err());
}

#[test]
fn test_aggregate_operations() {
    use crate::expr::AggregateOp;

    // Test count
    let count_expr = TLExpr::count("x", "Domain", TLExpr::pred("P", vec![Term::var("x")]));
    assert!(matches!(
        count_expr,
        TLExpr::Aggregate {
            op: AggregateOp::Count,
            ..
        }
    ));

    // Test sum
    let sum_expr = TLExpr::sum("x", "Domain", TLExpr::pred("Value", vec![Term::var("x")]));
    assert!(matches!(
        sum_expr,
        TLExpr::Aggregate {
            op: AggregateOp::Sum,
            ..
        }
    ));

    // Test average
    let avg_expr = TLExpr::average("x", "Domain", TLExpr::pred("Score", vec![Term::var("x")]));
    assert!(matches!(
        avg_expr,
        TLExpr::Aggregate {
            op: AggregateOp::Average,
            ..
        }
    ));

    // Test max
    let max_expr = TLExpr::max_agg("x", "Domain", TLExpr::pred("Max", vec![Term::var("x")]));
    assert!(matches!(
        max_expr,
        TLExpr::Aggregate {
            op: AggregateOp::Max,
            ..
        }
    ));

    // Test min
    let min_expr = TLExpr::min_agg("x", "Domain", TLExpr::pred("Min", vec![Term::var("x")]));
    assert!(matches!(
        min_expr,
        TLExpr::Aggregate {
            op: AggregateOp::Min,
            ..
        }
    ));

    // Test product
    let product_expr = TLExpr::product("x", "Domain", TLExpr::pred("P", vec![Term::var("x")]));
    assert!(matches!(
        product_expr,
        TLExpr::Aggregate {
            op: AggregateOp::Product,
            ..
        }
    ));
}

#[test]
fn test_aggregate_with_group_by() {
    use crate::expr::AggregateOp;

    let agg_expr = TLExpr::aggregate_with_group_by(
        AggregateOp::Sum,
        "x",
        "Domain",
        TLExpr::pred("Value", vec![Term::var("x"), Term::var("y")]),
        vec!["y".to_string()],
    );

    match agg_expr {
        TLExpr::Aggregate {
            op,
            var,
            domain,
            group_by,
            ..
        } => {
            assert!(matches!(op, AggregateOp::Sum));
            assert_eq!(var, "x");
            assert_eq!(domain, "Domain");
            assert_eq!(group_by, Some(vec!["y".to_string()]));
        }
        _ => panic!("Expected Aggregate variant"),
    }
}

#[test]
fn test_aggregate_free_vars() {
    // Aggregate variable should be bound
    let agg = TLExpr::sum("x", "Domain", TLExpr::pred("P", vec![Term::var("x")]));
    let free = agg.free_vars();
    assert_eq!(free.len(), 0);

    // Group-by variables should be free
    let agg_with_group = TLExpr::aggregate_with_group_by(
        crate::expr::AggregateOp::Sum,
        "x",
        "Domain",
        TLExpr::pred("Value", vec![Term::var("x"), Term::var("y")]),
        vec!["y".to_string()],
    );
    let free = agg_with_group.free_vars();
    assert_eq!(free.len(), 1);
    assert!(free.contains("y"));
}

#[test]
fn test_aggregate_domain_validation() {
    use crate::domain::DomainRegistry;

    let registry = DomainRegistry::with_builtins();

    // Valid aggregate with domain in registry
    let agg = TLExpr::sum("x", "Int", TLExpr::pred("Value", vec![Term::var("x")]));
    assert!(agg.validate_domains(&registry).is_ok());

    // Invalid aggregate with unknown domain
    let bad_agg = TLExpr::sum(
        "x",
        "UnknownDomain",
        TLExpr::pred("Value", vec![Term::var("x")]),
    );
    assert!(bad_agg.validate_domains(&registry).is_err());
}

#[test]
fn test_einsum_node_spec_parsing() {
    use crate::graph::EinsumNode;

    let node = EinsumNode::new("ij,jk->ik", vec![0, 1], vec![2]);

    // Parse and validate the spec
    let spec_opt = node.parse_einsum_spec().unwrap();
    assert!(spec_opt.is_some());

    let spec = spec_opt.unwrap();
    assert_eq!(spec.num_inputs(), 2);
    assert_eq!(spec.output_ndim(), 2);
    assert!(spec.is_reduction());
    assert_eq!(spec.summed_indices.len(), 1);

    // Test operation description
    let desc = node.operation_description();
    assert_eq!(desc, "Einsum(ij,jk->ik)");
}

#[test]
fn test_einsum_node_invalid_spec() {
    use crate::graph::EinsumNode;

    // Node with mismatched input count
    let node = EinsumNode::new("ij,jk->ik", vec![0], vec![1]); // Only 1 input, expects 2

    let result = node.parse_einsum_spec();
    assert!(result.is_err());
}

#[test]
fn test_non_einsum_node_operations() {
    use crate::graph::EinsumNode;

    let unary_node = EinsumNode::elem_unary("neg", 0, 1);
    assert_eq!(unary_node.operation_description(), "ElemUnary(neg)");
    assert!(unary_node.parse_einsum_spec().unwrap().is_none());

    let binary_node = EinsumNode::elem_binary("add", 0, 1, 2);
    assert_eq!(binary_node.operation_description(), "ElemBinary(add)");

    let reduce_node = EinsumNode::reduce("sum", vec![0, 1], 0, 1);
    assert_eq!(
        reduce_node.operation_description(),
        "Reduce(sum, axes=[0, 1])"
    );
}
