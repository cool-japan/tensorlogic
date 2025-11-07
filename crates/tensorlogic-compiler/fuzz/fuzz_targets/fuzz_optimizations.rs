//! Fuzz target for optimization passes.
//!
//! This tests CSE, DCE, negation optimization, and einsum optimization with
//! randomly generated graphs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_compiler::{compile_to_einsum, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > 400 {
        return;
    }

    // Build complex expressions that will trigger optimization passes
    let expr = build_optimizable_expr_from_bytes(data);

    // Compile with default context (all optimizations enabled)
    let mut ctx = CompilerContext::new();
    ctx.add_domain("D1".to_string(), 15);
    ctx.add_domain("D2".to_string(), 20);

    let _ = compile_to_einsum(&expr, &mut ctx);
});

/// Build expressions specifically designed to trigger optimizations.
fn build_optimizable_expr_from_bytes(data: &[u8]) -> TLExpr {
    if data.is_empty() {
        return TLExpr::Constant(1.0);
    }

    let discriminant = data[0] % 15;
    let rest = &data[1..];

    match discriminant {
        0 if rest.len() >= 2 => {
            // Double negation: NOT(NOT(expr))
            let inner = build_optimizable_expr_from_bytes(&rest[1..]);
            TLExpr::Not(Box::new(TLExpr::Not(Box::new(inner))))
        }
        1 if rest.len() >= 4 => {
            // De Morgan's law: NOT(AND(a, b))
            let mid = rest.len() / 2;
            let left = build_optimizable_expr_from_bytes(&rest[..mid]);
            let right = build_optimizable_expr_from_bytes(&rest[mid..]);
            TLExpr::Not(Box::new(TLExpr::And(Box::new(left), Box::new(right))))
        }
        2 if rest.len() >= 4 => {
            // De Morgan's law: NOT(OR(a, b))
            let mid = rest.len() / 2;
            let left = build_optimizable_expr_from_bytes(&rest[..mid]);
            let right = build_optimizable_expr_from_bytes(&rest[mid..]);
            TLExpr::Not(Box::new(TLExpr::Or(Box::new(left), Box::new(right))))
        }
        3 if rest.len() >= 2 => {
            // Common subexpression: AND(expr, expr)
            let inner = build_optimizable_expr_from_bytes(rest);
            TLExpr::And(Box::new(inner.clone()), Box::new(inner))
        }
        4 if rest.len() >= 3 => {
            // CSE opportunity: AND(AND(a, b), AND(a, b))
            let third = rest.len() / 3;
            let a = build_optimizable_expr_from_bytes(&rest[..third]);
            let b = build_optimizable_expr_from_bytes(&rest[third..2 * third]);

            let sub1 = TLExpr::And(Box::new(a.clone()), Box::new(b.clone()));
            let sub2 = TLExpr::And(Box::new(a), Box::new(b));

            TLExpr::And(Box::new(sub1), Box::new(sub2))
        }
        5 if rest.len() >= 1 => {
            // Quantifier with negation: NOT(EXISTS x. P(x))
            let var_name = format!("x{}", rest[0] % 5);
            let body = build_optimizable_expr_from_bytes(&rest[1..]);

            TLExpr::Not(Box::new(TLExpr::Exists {
                var: var_name,
                body: Box::new(body),
            }))
        }
        6 if rest.len() >= 1 => {
            // Quantifier with negation: NOT(FORALL x. P(x))
            let var_name = format!("y{}", rest[0] % 5);
            let body = build_optimizable_expr_from_bytes(&rest[1..]);

            TLExpr::Not(Box::new(TLExpr::Forall {
                var: var_name,
                body: Box::new(body),
            }))
        }
        7 if rest.len() >= 2 => {
            // Identity operations: a + 0
            let expr = build_optimizable_expr_from_bytes(rest);
            TLExpr::Add(Box::new(expr), Box::new(TLExpr::Constant(0.0)))
        }
        8 if rest.len() >= 2 => {
            // Identity operations: a * 1
            let expr = build_optimizable_expr_from_bytes(rest);
            TLExpr::Multiply(Box::new(expr), Box::new(TLExpr::Constant(1.0)))
        }
        9 if rest.len() >= 4 => {
            // Complex nested expression with multiple optimization opportunities
            let quarter = rest.len() / 4;
            let a = build_optimizable_expr_from_bytes(&rest[..quarter]);
            let b = build_optimizable_expr_from_bytes(&rest[quarter..2 * quarter]);
            let c = build_optimizable_expr_from_bytes(&rest[2 * quarter..3 * quarter]);

            // ((a AND b) OR (a AND b)) AND c
            let sub = TLExpr::And(Box::new(a.clone()), Box::new(b.clone()));
            let or_expr = TLExpr::Or(Box::new(sub.clone()), Box::new(sub));
            TLExpr::And(Box::new(or_expr), Box::new(c))
        }
        10 if rest.len() >= 3 => {
            // Implication with negations: NOT(a) -> NOT(b)
            let mid = rest.len() / 2;
            let a = build_optimizable_expr_from_bytes(&rest[..mid]);
            let b = build_optimizable_expr_from_bytes(&rest[mid..]);

            TLExpr::Implication(
                Box::new(TLExpr::Not(Box::new(a))),
                Box::new(TLExpr::Not(Box::new(b))),
            )
        }
        11 => {
            // AND with TRUE (should be optimized to just expr)
            let expr = build_optimizable_expr_from_bytes(rest);
            TLExpr::And(Box::new(expr), Box::new(TLExpr::Constant(1.0)))
        }
        12 => {
            // OR with FALSE (should be optimized to just expr)
            let expr = build_optimizable_expr_from_bytes(rest);
            TLExpr::Or(Box::new(expr), Box::new(TLExpr::Constant(0.0)))
        }
        13 if rest.len() >= 2 => {
            // Conditional with same branches: if cond then a else a
            let mid = rest.len() / 2;
            let cond = build_optimizable_expr_from_bytes(&rest[..mid]);
            let branch = build_optimizable_expr_from_bytes(&rest[mid..]);

            TLExpr::IfThenElse {
                condition: Box::new(cond),
                then_expr: Box::new(branch.clone()),
                else_expr: Box::new(branch),
            }
        }
        _ => {
            // Base case: simple predicate
            let var_name = format!("x{}", data[0] % 5);
            TLExpr::Predicate {
                name: "P".to_string(),
                args: vec![Term::Variable(var_name)],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_negation_optimization() {
        let data = &[0, 1, 2, 3];
        let expr = build_optimizable_expr_from_bytes(data);

        let mut ctx = CompilerContext::new();
        ctx.add_domain("D1".to_string(), 10);

        // Should optimize and not panic
        let _ = compile_to_einsum(&expr, &mut ctx);
    }

    #[test]
    fn test_cse_optimization() {
        let data = &[4, 1, 2, 3, 4, 5, 6];
        let expr = build_optimizable_expr_from_bytes(data);

        let mut ctx = CompilerContext::new();
        ctx.add_domain("D1".to_string(), 10);

        // Should detect and eliminate common subexpressions
        let _ = compile_to_einsum(&expr, &mut ctx);
    }
}
