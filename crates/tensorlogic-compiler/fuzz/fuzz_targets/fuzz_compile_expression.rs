//! Fuzz target for compiling arbitrary TLExpr expressions.
//!
//! This tests the compiler's robustness against malformed or unexpected inputs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_compiler::{compile_to_einsum, CompilerContext};
use tensorlogic_ir::{TLExpr, Term};

/// Generate a fuzzing harness for TLExpr compilation.
fuzz_target!(|data: &[u8]| {
    // Only process reasonable-sized inputs to avoid timeout
    if data.len() > 1000 || data.is_empty() {
        return;
    }

    // Use the fuzzer data to construct expressions
    let expr = build_expr_from_bytes(data);

    // Attempt compilation - we don't care if it fails, just that it doesn't panic
    let mut ctx = CompilerContext::new();
    let _ = compile_to_einsum(&expr, &mut ctx);
});

/// Build a TLExpr from raw bytes.
///
/// This creates increasingly complex expressions based on byte patterns.
fn build_expr_from_bytes(data: &[u8]) -> TLExpr {
    if data.is_empty() {
        return TLExpr::Predicate {
            name: "P".to_string(),
            args: vec![Term::Variable("x".to_string())],
        };
    }

    let discriminant = data[0] % 13;
    let rest = &data[1..];

    match discriminant {
        0 => {
            // Simple predicate
            let var_name = format!("x{}", data[0] % 5);
            TLExpr::Predicate {
                name: "P".to_string(),
                args: vec![Term::Variable(var_name)],
            }
        }
        1 if rest.len() >= 2 => {
            // AND of two subexpressions
            let mid = rest.len() / 2;
            TLExpr::And(
                Box::new(build_expr_from_bytes(&rest[..mid])),
                Box::new(build_expr_from_bytes(&rest[mid..])),
            )
        }
        2 if rest.len() >= 2 => {
            // OR of two subexpressions
            let mid = rest.len() / 2;
            TLExpr::Or(
                Box::new(build_expr_from_bytes(&rest[..mid])),
                Box::new(build_expr_from_bytes(&rest[mid..])),
            )
        }
        3 => {
            // NOT
            TLExpr::Not(Box::new(build_expr_from_bytes(rest)))
        }
        4 if rest.len() >= 1 => {
            // EXISTS quantifier
            let var_name = format!("x{}", rest[0] % 5);
            TLExpr::Exists {
                var: var_name,
                body: Box::new(build_expr_from_bytes(&rest[1..])),
            }
        }
        5 if rest.len() >= 1 => {
            // FORALL quantifier
            let var_name = format!("y{}", rest[0] % 5);
            TLExpr::Forall {
                var: var_name,
                body: Box::new(build_expr_from_bytes(&rest[1..])),
            }
        }
        6 if rest.len() >= 2 => {
            // Implication
            let mid = rest.len() / 2;
            TLExpr::Implication(
                Box::new(build_expr_from_bytes(&rest[..mid])),
                Box::new(build_expr_from_bytes(&rest[mid..])),
            )
        }
        7 if rest.len() >= 2 => {
            // Binary predicate
            let var1 = format!("x{}", rest[0] % 3);
            let var2 = format!("y{}", rest[1] % 3);
            TLExpr::Predicate {
                name: "R".to_string(),
                args: vec![Term::Variable(var1), Term::Variable(var2)],
            }
        }
        8 if rest.len() >= 2 => {
            // Arithmetic: Add
            let mid = rest.len() / 2;
            TLExpr::Add(
                Box::new(build_expr_from_bytes(&rest[..mid])),
                Box::new(build_expr_from_bytes(&rest[mid..])),
            )
        }
        9 if rest.len() >= 2 => {
            // Arithmetic: Multiply
            let mid = rest.len() / 2;
            TLExpr::Multiply(
                Box::new(build_expr_from_bytes(&rest[..mid])),
                Box::new(build_expr_from_bytes(&rest[mid..])),
            )
        }
        10 => {
            // Constant
            let value = (data[0] as f64) / 255.0;
            TLExpr::Constant(value)
        }
        11 if rest.len() >= 2 => {
            // Comparison: LessThan
            let mid = rest.len() / 2;
            TLExpr::LessThan(
                Box::new(build_expr_from_bytes(&rest[..mid])),
                Box::new(build_expr_from_bytes(&rest[mid..])),
            )
        }
        12 if rest.len() >= 3 => {
            // Conditional: If-then-else
            let third = rest.len() / 3;
            TLExpr::IfThenElse {
                condition: Box::new(build_expr_from_bytes(&rest[..third])),
                then_expr: Box::new(build_expr_from_bytes(&rest[third..2 * third])),
                else_expr: Box::new(build_expr_from_bytes(&rest[2 * third..])),
            }
        }
        _ => {
            // Default: simple predicate
            TLExpr::Predicate {
                name: "Q".to_string(),
                args: vec![Term::Variable("z".to_string())],
            }
        }
    }
}
