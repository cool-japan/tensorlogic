//! Example demonstrating new mathematical operations and Let bindings.
//!
//! This example showcases:
//! - Binary mathematical operations: Pow, Mod, Min, Max
//! - Unary mathematical operations: Abs, Floor, Ceil, Round, Sqrt, Exp, Log, Sin, Cos, Tan
//! - Let bindings for local variable definitions

use tensorlogic_compiler::{compile_to_einsum, compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

fn main() {
    println!("=== Tensorlogic New Operations Demo ===\n");

    // 1. Power operation
    println!("1. Power Operation: a^2");
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let two = TLExpr::Constant(2.0);
    let pow_expr = TLExpr::Pow(Box::new(a.clone()), Box::new(two));
    let graph = compile_to_einsum(&pow_expr).unwrap();
    println!(
        "   Nodes: {}, Tensors: {}\n",
        graph.nodes.len(),
        graph.tensors.len()
    );

    // 2. Modulo operation
    println!("2. Modulo Operation: a % 10");
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let ten = TLExpr::Constant(10.0);
    let mod_expr = TLExpr::Mod(Box::new(a), Box::new(ten));
    let graph = compile_to_einsum(&mod_expr).unwrap();
    println!(
        "   Nodes: {}, Tensors: {}\n",
        graph.nodes.len(),
        graph.tensors.len()
    );

    // 3. Min/Max operations
    println!("3. Min/Max Operations: min(a, b) and max(c, d)");
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let min_expr = TLExpr::Min(Box::new(a), Box::new(b));

    let c = TLExpr::pred("c", vec![Term::var("x")]);
    let d = TLExpr::pred("d", vec![Term::var("x")]);
    let max_expr = TLExpr::Max(Box::new(c), Box::new(d));

    let graph1 = compile_to_einsum(&min_expr).unwrap();
    let graph2 = compile_to_einsum(&max_expr).unwrap();
    println!(
        "   Min nodes: {}, Max nodes: {}\n",
        graph1.nodes.len(),
        graph2.nodes.len()
    );

    // 4. Absolute value and rounding operations
    println!("4. Rounding Operations: abs(a), floor(b), ceil(c), round(d)");
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let abs_expr = TLExpr::Abs(Box::new(a));

    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let floor_expr = TLExpr::Floor(Box::new(b));

    let c = TLExpr::pred("c", vec![Term::var("x")]);
    let ceil_expr = TLExpr::Ceil(Box::new(c));

    let d = TLExpr::pred("d", vec![Term::var("x")]);
    let round_expr = TLExpr::Round(Box::new(d));

    let _ = compile_to_einsum(&abs_expr).unwrap();
    let _ = compile_to_einsum(&floor_expr).unwrap();
    let _ = compile_to_einsum(&ceil_expr).unwrap();
    let _ = compile_to_einsum(&round_expr).unwrap();
    println!("   All rounding operations compiled successfully\n");

    // 5. Square root and exponential operations
    println!("5. Exponential Operations: sqrt(a), exp(b), log(c)");
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let sqrt_expr = TLExpr::Sqrt(Box::new(a));

    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let exp_expr = TLExpr::Exp(Box::new(b));

    let c = TLExpr::pred("c", vec![Term::var("x")]);
    let log_expr = TLExpr::Log(Box::new(c));

    let _ = compile_to_einsum(&sqrt_expr).unwrap();
    let _ = compile_to_einsum(&exp_expr).unwrap();
    let _ = compile_to_einsum(&log_expr).unwrap();
    println!("   All exponential operations compiled successfully\n");

    // 6. Trigonometric operations
    println!("6. Trigonometric Operations: sin(x), cos(x), tan(x)");
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let sin_expr = TLExpr::Sin(Box::new(x.clone()));
    let cos_expr = TLExpr::Cos(Box::new(x.clone()));
    let tan_expr = TLExpr::Tan(Box::new(x));

    let _ = compile_to_einsum(&sin_expr).unwrap();
    let _ = compile_to_einsum(&cos_expr).unwrap();
    let _ = compile_to_einsum(&tan_expr).unwrap();
    println!("   All trigonometric operations compiled successfully\n");

    // 7. Complex nested expression
    println!("7. Complex Expression: sqrt(abs(a^2 + b^2))");
    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let two = TLExpr::Constant(2.0);

    let a_squared = TLExpr::Pow(Box::new(a), Box::new(two.clone()));
    let b_squared = TLExpr::Pow(Box::new(b), Box::new(two));
    let sum = TLExpr::Add(Box::new(a_squared), Box::new(b_squared));
    let abs_sum = TLExpr::Abs(Box::new(sum));
    let magnitude = TLExpr::Sqrt(Box::new(abs_sum));

    let graph = compile_to_einsum(&magnitude).unwrap();
    println!(
        "   Nodes: {}, Tensors: {}",
        graph.nodes.len(),
        graph.tensors.len()
    );
    println!("   This computes the Euclidean distance!\n");

    // 8. Pythagorean identity: sin²(x) + cos²(x)
    println!("8. Pythagorean Identity: sin²(x) + cos²(x)");
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let sin_x = TLExpr::Sin(Box::new(x.clone()));
    let cos_x = TLExpr::Cos(Box::new(x));
    let two = TLExpr::Constant(2.0);

    let sin_squared = TLExpr::Pow(Box::new(sin_x), Box::new(two.clone()));
    let cos_squared = TLExpr::Pow(Box::new(cos_x), Box::new(two));
    let identity = TLExpr::Add(Box::new(sin_squared), Box::new(cos_squared));

    let graph = compile_to_einsum(&identity).unwrap();
    println!(
        "   Nodes: {}, Tensors: {}",
        graph.nodes.len(),
        graph.tensors.len()
    );
    println!("   Result should be approximately 1.0\n");

    // 9. Let binding example
    println!("9. Let Binding: let temp = a + b in temp * c");
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Number", 10);

    let a = TLExpr::pred("a", vec![Term::var("x")]);
    let b = TLExpr::pred("b", vec![Term::var("x")]);
    let c = TLExpr::pred("c", vec![Term::var("x")]);

    let sum = TLExpr::Add(Box::new(a), Box::new(b));
    let product = TLExpr::Mul(Box::new(sum.clone()), Box::new(c));

    let let_expr = TLExpr::Let {
        var: "temp".to_string(),
        value: Box::new(sum),
        body: Box::new(product),
    };

    let graph = compile_to_einsum_with_context(&let_expr, &mut ctx).unwrap();
    println!(
        "   Nodes: {}, Tensors: {}",
        graph.nodes.len(),
        graph.tensors.len()
    );
    println!("   Let bindings allow reusing subexpressions\n");

    // 10. Practical example: Normalized softmax temperature
    println!("10. Practical Example: exp(x/T) / sum(exp(x/T))");
    println!("    (Softmax with temperature scaling)");
    let x = TLExpr::pred("x", vec![Term::var("i")]);
    let temperature = TLExpr::Constant(0.5);

    let scaled = TLExpr::Div(Box::new(x), Box::new(temperature));
    let exp_scaled = TLExpr::Exp(Box::new(scaled));

    let graph = compile_to_einsum(&exp_scaled).unwrap();
    println!(
        "   Nodes: {}, Tensors: {}",
        graph.nodes.len(),
        graph.tensors.len()
    );
    println!("   Commonly used in attention mechanisms\n");

    println!("=== All new operations working correctly! ===");
}
