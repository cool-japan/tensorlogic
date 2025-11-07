//! Example demonstrating new mathematical operations, Let bindings, and optimizations.
//!
//! This example showcases the enhanced features added to tensorlogic-ir:
//! - Mathematical operations (trigonometric, power, modulo, etc.)
//! - Min/Max operations
//! - Let bindings for local variables
//! - Constant folding optimization
//! - Algebraic simplification

use tensorlogic_ir::{algebraic_simplify, constant_fold, optimize_expr, TLExpr, Term};

fn main() {
    println!("=== TensorLogic IR: New Operations Demo ===\n");

    // 1. Mathematical Operations
    println!("1. Mathematical Operations");
    println!("--------------------------");

    // Power operation: x^2
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let power = TLExpr::pow(x.clone(), TLExpr::constant(2.0));
    println!("Power: x^2 = {}", power);

    // Trigonometric: sin(x) + cos(x)
    let trig = TLExpr::add(TLExpr::sin(x.clone()), TLExpr::cos(x.clone()));
    println!("Trigonometric: sin(x) + cos(x) = {}", trig);

    // Square root: sqrt(x)
    let sqrt_expr = TLExpr::sqrt(x.clone());
    println!("Square root: sqrt(x) = {}", sqrt_expr);

    // Floor and Ceil
    let floor_expr = TLExpr::floor(x.clone());
    let ceil_expr = TLExpr::ceil(x.clone());
    println!("Floor: floor(x) = {}", floor_expr);
    println!("Ceil: ceil(x) = {}", ceil_expr);

    // Absolute value
    let abs_expr = TLExpr::abs(TLExpr::sub(x.clone(), TLExpr::constant(5.0)));
    println!("Absolute: abs(x - 5) = {}", abs_expr);

    println!();

    // 2. Min/Max Operations
    println!("2. Min/Max Operations");
    println!("---------------------");

    let a = TLExpr::pred("a", vec![Term::var("i")]);
    let b = TLExpr::pred("b", vec![Term::var("i")]);

    let min_expr = TLExpr::min(a.clone(), b.clone());
    let max_expr = TLExpr::max(a.clone(), b.clone());

    println!("Min: min(a, b) = {}", min_expr);
    println!("Max: max(a, b) = {}", max_expr);

    println!();

    // 3. Let Bindings
    println!("3. Let Bindings");
    println!("---------------");

    // let temp = x + 1 in temp * temp
    let let_expr = TLExpr::let_binding(
        "temp",
        TLExpr::add(x.clone(), TLExpr::constant(1.0)),
        TLExpr::mul(TLExpr::pred("temp", vec![]), TLExpr::pred("temp", vec![])),
    );
    println!("Let binding: let temp = x + 1 in temp * temp");
    println!("  = {}", let_expr);

    // Nested let bindings
    let nested_let = TLExpr::let_binding(
        "x",
        TLExpr::constant(5.0),
        TLExpr::let_binding(
            "y",
            TLExpr::add(TLExpr::pred("x", vec![]), TLExpr::constant(3.0)),
            TLExpr::mul(TLExpr::pred("x", vec![]), TLExpr::pred("y", vec![])),
        ),
    );
    println!("Nested: let x = 5 in (let y = x + 3 in x * y)");
    println!("  = {}", nested_let);

    println!();

    // 4. Constant Folding
    println!("4. Constant Folding");
    println!("-------------------");

    // (2 + 3) * 4 = 20
    let expr1 = TLExpr::mul(
        TLExpr::add(TLExpr::constant(2.0), TLExpr::constant(3.0)),
        TLExpr::constant(4.0),
    );
    println!("Original: (2 + 3) * 4 = {}", expr1);
    let folded1 = constant_fold(&expr1);
    println!("Folded:   {}", folded1);
    println!();

    // sin(0) = 0
    let expr2 = TLExpr::sin(TLExpr::constant(0.0));
    println!("Original: sin(0) = {}", expr2);
    let folded2 = constant_fold(&expr2);
    println!("Folded:   {}", folded2);
    println!();

    // sqrt(16) = 4
    let expr3 = TLExpr::sqrt(TLExpr::constant(16.0));
    println!("Original: sqrt(16) = {}", expr3);
    let folded3 = constant_fold(&expr3);
    println!("Folded:   {}", folded3);
    println!();

    // 2^3 = 8
    let expr4 = TLExpr::pow(TLExpr::constant(2.0), TLExpr::constant(3.0));
    println!("Original: 2^3 = {}", expr4);
    let folded4 = constant_fold(&expr4);
    println!("Folded:   {}", folded4);

    println!();

    // 5. Algebraic Simplification
    println!("5. Algebraic Simplification");
    println!("---------------------------");

    // x + 0 = x
    let expr5 = TLExpr::add(x.clone(), TLExpr::constant(0.0));
    println!("Original: x + 0 = {}", expr5);
    let simplified5 = algebraic_simplify(&expr5);
    println!("Simplified: {}", simplified5);
    println!();

    // x * 1 = x
    let expr6 = TLExpr::mul(x.clone(), TLExpr::constant(1.0));
    println!("Original: x * 1 = {}", expr6);
    let simplified6 = algebraic_simplify(&expr6);
    println!("Simplified: {}", simplified6);
    println!();

    // x * 0 = 0
    let expr7 = TLExpr::mul(x.clone(), TLExpr::constant(0.0));
    println!("Original: x * 0 = {}", expr7);
    let simplified7 = algebraic_simplify(&expr7);
    println!("Simplified: {}", simplified7);
    println!();

    // x^0 = 1
    let expr8 = TLExpr::pow(x.clone(), TLExpr::constant(0.0));
    println!("Original: x^0 = {}", expr8);
    let simplified8 = algebraic_simplify(&expr8);
    println!("Simplified: {}", simplified8);
    println!();

    // NOT(NOT(x)) = x
    let expr9 = TLExpr::negate(TLExpr::negate(x.clone()));
    println!("Original: NOT(NOT(x)) = {}", expr9);
    let simplified9 = algebraic_simplify(&expr9);
    println!("Simplified: {}", simplified9);

    println!();

    // 6. Combined Optimization
    println!("6. Combined Optimization");
    println!("------------------------");

    // (2 + 3) * 1 should become 5
    let expr10 = TLExpr::mul(
        TLExpr::add(TLExpr::constant(2.0), TLExpr::constant(3.0)),
        TLExpr::constant(1.0),
    );
    println!("Original: (2 + 3) * 1 = {}", expr10);
    let optimized10 = optimize_expr(&expr10);
    println!("Optimized: {}", optimized10);
    println!();

    // (x + 0) * (y * 1) should become x * y
    let y = TLExpr::pred("y", vec![Term::var("j")]);
    let expr11 = TLExpr::mul(
        TLExpr::add(x.clone(), TLExpr::constant(0.0)),
        TLExpr::mul(y.clone(), TLExpr::constant(1.0)),
    );
    println!("Original: (x + 0) * (y * 1) = {}", expr11);
    let optimized11 = optimize_expr(&expr11);
    println!("Optimized: {}", optimized11);

    println!();

    // 7. Complex Expression
    println!("7. Complex Expression Example");
    println!("-----------------------------");

    // let radius = sqrt(x^2 + y^2) in
    //   if radius < 1.0 then sin(radius * π) else 0
    let complex = TLExpr::let_binding(
        "radius",
        TLExpr::sqrt(TLExpr::add(
            TLExpr::pow(x.clone(), TLExpr::constant(2.0)),
            TLExpr::pow(y.clone(), TLExpr::constant(2.0)),
        )),
        TLExpr::if_then_else(
            TLExpr::lt(TLExpr::pred("radius", vec![]), TLExpr::constant(1.0)),
            TLExpr::sin(TLExpr::mul(
                TLExpr::pred("radius", vec![]),
                TLExpr::constant(std::f64::consts::PI),
            )),
            TLExpr::constant(0.0),
        ),
    );

    println!("Complex expression:");
    println!("  let radius = sqrt(x^2 + y^2) in");
    println!("    if radius < 1.0 then sin(radius * π) else 0");
    println!("\nExpression structure:");
    println!("{}", complex);

    // Check free variables
    let free_vars = complex.free_vars();
    println!("\nFree variables: {:?}", free_vars);

    // Optimize
    let optimized_complex = optimize_expr(&complex);
    println!("\nOptimized: {}", optimized_complex);

    println!();
    println!("=== Demo Complete ===");
}
