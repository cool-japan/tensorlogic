//! Fuzz target for quantifier handling and scope analysis.
//!
//! This tests nested quantifiers, variable scoping, and bound/free variable detection.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_compiler::{compile_to_einsum, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > 300 {
        return;
    }

    // Build expressions with nested quantifiers
    let expr = build_quantified_expr_from_bytes(data, 0);

    // Test compilation
    let mut ctx = CompilerContext::new();
    ctx.add_domain("D".to_string(), 10);

    let _ = compile_to_einsum(&expr, &mut ctx);
});

/// Build deeply nested quantified expressions.
fn build_quantified_expr_from_bytes(data: &[u8], depth: usize) -> TLExpr {
    // Limit nesting depth to avoid stack overflow
    if depth > 10 || data.is_empty() {
        return TLExpr::Predicate {
            name: "P".to_string(),
            args: vec![Term::Variable("x".to_string())],
        };
    }

    let discriminant = data[0] % 10;
    let rest = &data[1..];

    match discriminant {
        0..=3 if rest.len() >= 1 => {
            // EXISTS quantifier (higher probability)
            let var_name = format!("x{}", rest[0] % 8);
            TLExpr::Exists {
                var: var_name,
                body: Box::new(build_quantified_expr_from_bytes(rest, depth + 1)),
            }
        }
        4..=6 if rest.len() >= 1 => {
            // FORALL quantifier
            let var_name = format!("y{}", rest[0] % 8);
            TLExpr::Forall {
                var: var_name,
                body: Box::new(build_quantified_expr_from_bytes(rest, depth + 1)),
            }
        }
        7 if rest.len() >= 2 => {
            // AND with nested quantifiers
            let mid = rest.len() / 2;
            TLExpr::And(
                Box::new(build_quantified_expr_from_bytes(&rest[..mid], depth + 1)),
                Box::new(build_quantified_expr_from_bytes(&rest[mid..], depth + 1)),
            )
        }
        8 if rest.len() >= 2 => {
            // Implication with quantifiers
            let mid = rest.len() / 2;
            TLExpr::Implication(
                Box::new(build_quantified_expr_from_bytes(&rest[..mid], depth + 1)),
                Box::new(build_quantified_expr_from_bytes(&rest[mid..], depth + 1)),
            )
        }
        _ => {
            // Predicate with multiple variables (some may be free)
            let num_vars = ((data[0] % 4) + 1) as usize;
            let args: Vec<Term> = (0..num_vars)
                .map(|i| {
                    let var_name = format!("v{}", (data[0] + i as u8) % 8);
                    Term::Variable(var_name)
                })
                .collect();

            TLExpr::Predicate {
                name: "R".to_string(),
                args,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_quantifier_fuzzing() {
        let data = &[1, 2, 3, 4, 5];
        let expr = build_quantified_expr_from_bytes(data, 0);

        let mut ctx = CompilerContext::new();
        ctx.add_domain("D".to_string(), 10);

        // Should not panic
        let _ = compile_to_einsum(&expr, &mut ctx);
    }

    #[test]
    fn test_deep_nesting() {
        let data = &[0; 50]; // All EXISTS
        let expr = build_quantified_expr_from_bytes(data, 0);

        let mut ctx = CompilerContext::new();
        ctx.add_domain("D".to_string(), 10);

        // Should not panic even with deep nesting
        let _ = compile_to_einsum(&expr, &mut ctx);
    }
}
