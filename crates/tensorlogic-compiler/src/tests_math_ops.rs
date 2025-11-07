//! Tests for new mathematical operations (Pow, Mod, Min, Max, Abs, Floor, Ceil, Round, Sqrt, Exp, Log, Sin, Cos, Tan, Let).

use crate::{compile_to_einsum, compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

/// Test compilation of power operation
#[test]
fn test_pow_compilation() {
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let expr = TLExpr::Pow(Box::new(a), Box::new(b));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of modulo operation
#[test]
fn test_mod_compilation() {
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let expr = TLExpr::Mod(Box::new(a), Box::new(b));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of min operation
#[test]
fn test_min_compilation() {
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let expr = TLExpr::Min(Box::new(a), Box::new(b));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of max operation
#[test]
fn test_max_compilation() {
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let expr = TLExpr::Max(Box::new(a), Box::new(b));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of absolute value operation
#[test]
fn test_abs_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Abs(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of floor operation
#[test]
fn test_floor_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Floor(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of ceil operation
#[test]
fn test_ceil_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Ceil(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of round operation
#[test]
fn test_round_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Round(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of square root operation
#[test]
fn test_sqrt_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Sqrt(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of exponential operation
#[test]
fn test_exp_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Exp(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of natural logarithm operation
#[test]
fn test_log_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Log(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of sine operation
#[test]
fn test_sin_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Sin(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of cosine operation
#[test]
fn test_cos_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Cos(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of tangent operation
#[test]
fn test_tan_compilation() {
    let inner = TLExpr::pred("a", vec![Term::var("x")]);
    let expr = TLExpr::Tan(Box::new(inner));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test compilation of let binding
#[test]
fn test_let_binding_compilation() {
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Number", 10);

    // let temp = a + b in temp * c
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let c = TLExpr::pred("c", vec![Term::var("x")]);

    let value = TLExpr::Add(Box::new(a), Box::new(b));
    // In body, we would ideally reference temp but for now just use c
    let body = TLExpr::Mul(Box::new(value.clone()), Box::new(c));

    let expr = TLExpr::Let {
        var: "temp".to_string(),
        value: Box::new(value),
        body: Box::new(body),
    };

    let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
    assert!(!graph.tensors.is_empty());
    // Let binding should produce nodes
    assert_eq!(graph.outputs.len(), 1);
}

/// Test nested mathematical operations
#[test]
fn test_nested_math_operations() {
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);

    // sqrt(abs(a + b))
    let add = TLExpr::Add(Box::new(a), Box::new(b));
    let abs = TLExpr::Abs(Box::new(add));
    let expr = TLExpr::Sqrt(Box::new(abs));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test complex expression with multiple new operations
#[test]
fn test_complex_math_expression() {
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let c = TLExpr::pred("c", vec![Term::var("x")]);

    // min(pow(a, b), max(floor(c), 2.0))
    let pow = TLExpr::Pow(Box::new(a), Box::new(b));
    let floor_c = TLExpr::Floor(Box::new(c));
    let const_2 = TLExpr::Constant(2.0);
    let max_expr = TLExpr::Max(Box::new(floor_c), Box::new(const_2));
    let expr = TLExpr::Min(Box::new(pow), Box::new(max_expr));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test trigonometric expressions
#[test]
fn test_trig_expressions() {
    let x = TLExpr::pred("x", vec![Term::var("i")]);

    // sin(x)^2 + cos(x)^2 = 1 (Pythagorean identity)
    let sin_x = TLExpr::Sin(Box::new(x.clone()));
    let cos_x = TLExpr::Cos(Box::new(x));
    let two = TLExpr::Constant(2.0);

    let sin_squared = TLExpr::Pow(Box::new(sin_x), Box::new(two.clone()));
    let cos_squared = TLExpr::Pow(Box::new(cos_x), Box::new(two));
    let expr = TLExpr::Add(Box::new(sin_squared), Box::new(cos_squared));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test let binding with nested let
#[test]
fn test_nested_let_bindings() {
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Number", 10);

    // let x = a in (let y = b in a + b)
    let a = TLExpr::pred("a", vec![Term::var("i")]);
    let b = TLExpr::pred("b", vec![Term::var("i")]);

    let a2 = TLExpr::pred("a", vec![Term::var("i")]);
    let b2 = TLExpr::pred("b", vec![Term::var("i")]);
    let sum = TLExpr::Add(Box::new(a2), Box::new(b2));

    let inner_let = TLExpr::Let {
        var: "y".to_string(),
        value: Box::new(b),
        body: Box::new(sum),
    };

    let expr = TLExpr::Let {
        var: "x".to_string(),
        value: Box::new(a),
        body: Box::new(inner_let),
    };

    let graph = compile_to_einsum_with_context(&expr, &mut ctx).unwrap();
    assert!(!graph.tensors.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test modulo with constants
#[test]
fn test_mod_with_constants() {
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let divisor = TLExpr::Constant(10.0);
    let expr = TLExpr::Mod(Box::new(a), Box::new(divisor));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}

/// Test exponential and logarithm composition
#[test]
fn test_exp_log_composition() {
    let a = TLExpr::pred("a", vec![Term::var("x")]);

    // log(exp(a)) should approximately equal a
    let exp_a = TLExpr::Exp(Box::new(a));
    let expr = TLExpr::Log(Box::new(exp_a));

    let graph = compile_to_einsum(&expr).unwrap();
    assert!(!graph.tensors.is_empty());
    assert!(!graph.nodes.is_empty());
    assert_eq!(graph.outputs.len(), 1);
}
