//! Example demonstrating expression optimization passes.
//!
//! This example showcases the compiler's optimization capabilities:
//! - Constant folding: evaluating constant expressions at compile time
//! - Algebraic simplification: applying mathematical identities
//! - Negation optimization: pushing negations inward (De Morgan's laws)

use tensorlogic_compiler::optimize::{fold_constants, optimize_negations, simplify_algebraic};
use tensorlogic_ir::{TLExpr, Term};

fn main() {
    println!("=== Tensorlogic Optimization Demo ===\n");

    // 1. Constant Folding
    println!("1. Constant Folding Optimization");
    println!("   Original: (2.0 + 3.0) * 4.0");
    let expr1 = TLExpr::Mul(
        Box::new(TLExpr::Add(
            Box::new(TLExpr::Constant(2.0)),
            Box::new(TLExpr::Constant(3.0)),
        )),
        Box::new(TLExpr::Constant(4.0)),
    );
    let (optimized1, stats1) = fold_constants(&expr1);
    println!("   Optimized: {:?}", optimized1);
    println!("   Stats: {} binary ops folded\n", stats1.binary_ops_folded);

    // 2. Complex Constant Expression
    println!("2. Complex Constant Folding");
    println!("   Original: sqrt(16.0) + exp(0.0) * 2.0");
    let expr2 = TLExpr::Add(
        Box::new(TLExpr::Sqrt(Box::new(TLExpr::Constant(16.0)))),
        Box::new(TLExpr::Mul(
            Box::new(TLExpr::Exp(Box::new(TLExpr::Constant(0.0)))),
            Box::new(TLExpr::Constant(2.0)),
        )),
    );
    let (optimized2, stats2) = fold_constants(&expr2);
    println!("   Optimized: {:?}", optimized2);
    println!(
        "   Stats: {} unary ops folded, {} binary ops folded\n",
        stats2.unary_ops_folded, stats2.binary_ops_folded
    );

    // 3. Algebraic Simplification - Identity Laws
    println!("3. Algebraic Simplification (Identity Laws)");
    let x = TLExpr::pred("x", vec![Term::var("i")]);

    println!("   a) x + 0 = x");
    let expr3a = TLExpr::Add(Box::new(x.clone()), Box::new(TLExpr::Constant(0.0)));
    let (optimized3a, stats3a) = simplify_algebraic(&expr3a);
    println!("      Optimized: {:?}", optimized3a);
    println!(
        "      Stats: {} identities eliminated\n",
        stats3a.identities_eliminated
    );

    println!("   b) x * 1 = x");
    let expr3b = TLExpr::Mul(Box::new(x.clone()), Box::new(TLExpr::Constant(1.0)));
    let (optimized3b, stats3b) = simplify_algebraic(&expr3b);
    println!("      Optimized: {:?}", optimized3b);
    println!(
        "      Stats: {} identities eliminated\n",
        stats3b.identities_eliminated
    );

    println!("   c) x / 1 = x");
    let expr3c = TLExpr::Div(Box::new(x.clone()), Box::new(TLExpr::Constant(1.0)));
    let (optimized3c, stats3c) = simplify_algebraic(&expr3c);
    println!("      Optimized: {:?}", optimized3c);
    println!(
        "      Stats: {} identities eliminated\n",
        stats3c.identities_eliminated
    );

    // 4. Algebraic Simplification - Annihilation
    println!("4. Algebraic Simplification (Annihilation)");
    println!("   a) x * 0 = 0");
    let expr4a = TLExpr::Mul(Box::new(x.clone()), Box::new(TLExpr::Constant(0.0)));
    let (optimized4a, stats4a) = simplify_algebraic(&expr4a);
    println!("      Optimized: {:?}", optimized4a);
    println!(
        "      Stats: {} annihilations applied\n",
        stats4a.annihilations_applied
    );

    println!("   b) 0 / x = 0");
    let expr4b = TLExpr::Div(Box::new(TLExpr::Constant(0.0)), Box::new(x.clone()));
    let (optimized4b, stats4b) = simplify_algebraic(&expr4b);
    println!("      Optimized: {:?}", optimized4b);
    println!(
        "      Stats: {} annihilations applied\n",
        stats4b.annihilations_applied
    );

    // 5. Power Identities
    println!("5. Power Identities");
    println!("   a) x^0 = 1");
    let expr5a = TLExpr::Pow(Box::new(x.clone()), Box::new(TLExpr::Constant(0.0)));
    let (optimized5a, stats5a) = simplify_algebraic(&expr5a);
    println!("      Optimized: {:?}", optimized5a);
    println!(
        "      Stats: {} identities eliminated\n",
        stats5a.identities_eliminated
    );

    println!("   b) x^1 = x");
    let expr5b = TLExpr::Pow(Box::new(x.clone()), Box::new(TLExpr::Constant(1.0)));
    let (optimized5b, stats5b) = simplify_algebraic(&expr5b);
    println!("      Optimized: {:?}", optimized5b);
    println!(
        "      Stats: {} identities eliminated\n",
        stats5b.identities_eliminated
    );

    // 6. Idempotent Operations
    println!("6. Idempotent Operations");
    println!("   a) min(x, x) = x");
    let expr6a = TLExpr::Min(Box::new(x.clone()), Box::new(x.clone()));
    let (optimized6a, stats6a) = simplify_algebraic(&expr6a);
    println!("      Optimized: {:?}", optimized6a);
    println!(
        "      Stats: {} idempotent simplified\n",
        stats6a.idempotent_simplified
    );

    println!("   b) max(x, x) = x");
    let expr6b = TLExpr::Max(Box::new(x.clone()), Box::new(x));
    let (optimized6b, stats6b) = simplify_algebraic(&expr6b);
    println!("      Optimized: {:?}", optimized6b);
    println!(
        "      Stats: {} idempotent simplified\n",
        stats6b.idempotent_simplified
    );

    // 7. Negation Optimization (De Morgan's Laws)
    println!("7. Negation Optimization (De Morgan's Laws)");
    let a = TLExpr::pred("a", vec![Term::var("i")]);
    let b = TLExpr::pred("b", vec![Term::var("i")]);

    println!("   a) NOT(a AND b) = (NOT a) OR (NOT b)");
    let expr7a = TLExpr::Not(Box::new(TLExpr::And(
        Box::new(a.clone()),
        Box::new(b.clone()),
    )));
    let (_optimized7a, stats7a) = optimize_negations(&expr7a);
    println!("      Original depth: calculated");
    println!(
        "      Stats: {} De Morgan applications\n",
        stats7a.demorgans_applied
    );

    println!("   b) NOT(NOT(a)) = a (double negation elimination)");
    let expr7b = TLExpr::Not(Box::new(TLExpr::Not(Box::new(a.clone()))));
    let (_optimized7b, stats7b) = optimize_negations(&expr7b);
    println!(
        "      Stats: {} double negations eliminated\n",
        stats7b.double_negations_eliminated
    );

    // 8. Combined Optimizations
    println!("8. Combined Optimizations");
    println!("   Original: (x + 0) * 1 + (2.0 * 3.0)");
    let expr8 = TLExpr::Add(
        Box::new(TLExpr::Mul(
            Box::new(TLExpr::Add(
                Box::new(a.clone()),
                Box::new(TLExpr::Constant(0.0)),
            )),
            Box::new(TLExpr::Constant(1.0)),
        )),
        Box::new(TLExpr::Mul(
            Box::new(TLExpr::Constant(2.0)),
            Box::new(TLExpr::Constant(3.0)),
        )),
    );

    // Apply constant folding first
    let (after_const_fold, cf_stats) = fold_constants(&expr8);
    println!(
        "   After constant folding: {} ops folded",
        cf_stats.binary_ops_folded
    );

    // Then apply algebraic simplification
    let (final_opt, alg_stats) = simplify_algebraic(&after_const_fold);
    println!(
        "   After algebraic simplification: {} identities eliminated",
        alg_stats.identities_eliminated
    );
    println!("   Final result: {:?}\n", final_opt);

    // 9. Real-world Example: Softmax Optimization
    println!("9. Real-world Example: Temperature-scaled Softmax");
    println!("   Original: exp((x - max) / 1.0) (temperature = 1.0)");
    let x_pred = TLExpr::pred("x", vec![Term::var("i")]);
    let max_pred = TLExpr::pred("max", vec![]);
    let temp = TLExpr::Constant(1.0);

    let softmax_expr = TLExpr::Exp(Box::new(TLExpr::Div(
        Box::new(TLExpr::Sub(Box::new(x_pred), Box::new(max_pred))),
        Box::new(temp),
    )));

    let (opt_softmax, softmax_stats) = simplify_algebraic(&softmax_expr);
    println!("   Optimized (division by 1 eliminated): {:?}", opt_softmax);
    println!(
        "   Stats: {} identities eliminated\n",
        softmax_stats.identities_eliminated
    );

    println!("=== All optimization passes demonstrated! ===");
    println!("\nKey Takeaways:");
    println!("- Constant folding reduces compile-time computation");
    println!("- Algebraic simplification eliminates redundant operations");
    println!("- Negation optimization produces more efficient logical circuits");
    println!("- Combining optimizations yields significant improvements");
}
