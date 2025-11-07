//! Fuzz target for type checking with random signatures.
//!
//! This tests the type checker's ability to handle arbitrary predicate signatures
//! and detect type mismatches.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_adapters::{DomainInfo, PredicateInfo};
use tensorlogic_compiler::CompilerContext;
use tensorlogic_ir::{PredicateSignature, TLExpr, Term};

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > 500 {
        return;
    }

    // Build a context with random domain and signature information
    let mut ctx = build_context_from_bytes(data);

    // Build an expression using some of the domains
    let expr = build_typed_expr_from_bytes(&data[data.len() / 2..], &ctx);

    // Attempt to compile - should not panic even with type mismatches
    let _ = tensorlogic_compiler::compile_to_einsum(&expr, &mut ctx);
});

/// Build a CompilerContext with random domains and signatures.
fn build_context_from_bytes(data: &[u8]) -> CompilerContext {
    let mut ctx = CompilerContext::new();

    // Add a few domains
    let num_domains = (data[0] % 5) + 1;
    for i in 0..num_domains {
        let domain_name = format!("Domain{}", i);
        let size = ((data[i as usize + 1] as usize) % 20) + 1;
        ctx.add_domain(domain_name, size);
    }

    // Add some predicate signatures
    let domain_names: Vec<String> = ctx.domains.keys().cloned().collect();
    if !domain_names.is_empty() {
        let num_predicates = (data[0] % 4) + 1;
        for i in 0..num_predicates {
            let pred_name = format!("P{}", i);
            let arity = ((data[i as usize + 2] % 3) + 1) as usize;

            // Build arg types from available domains
            let arg_types: Vec<String> = (0..arity)
                .map(|j| {
                    let idx = (data[(i + j) as usize + 3] as usize) % domain_names.len();
                    domain_names[idx].clone()
                })
                .collect();

            let signature = PredicateSignature::new(pred_name.clone(), arg_types);
            ctx.predicate_signatures.insert(pred_name, signature);
        }
    }

    ctx
}

/// Build a typed expression from bytes, using domains from context.
fn build_typed_expr_from_bytes(data: &[u8], ctx: &CompilerContext) -> TLExpr {
    if data.is_empty() {
        return TLExpr::Constant(1.0);
    }

    let discriminant = data[0] % 8;
    let rest = &data[1..];

    let domain_names: Vec<String> = ctx.domains.keys().cloned().collect();
    if domain_names.is_empty() {
        return TLExpr::Constant(0.5);
    }

    match discriminant {
        0 => {
            // Predicate with typed term
            let domain_idx = (data[0] as usize) % domain_names.len();
            let domain_type = &domain_names[domain_idx];

            let term = Term::Typed {
                var: format!("x{}", data[0] % 5),
                ty: domain_type.clone(),
            };

            TLExpr::Predicate {
                name: "P0".to_string(),
                args: vec![term],
            }
        }
        1 if rest.len() >= 2 => {
            // AND
            let mid = rest.len() / 2;
            TLExpr::And(
                Box::new(build_typed_expr_from_bytes(&rest[..mid], ctx)),
                Box::new(build_typed_expr_from_bytes(&rest[mid..], ctx)),
            )
        }
        2 if rest.len() >= 2 => {
            // OR
            let mid = rest.len() / 2;
            TLExpr::Or(
                Box::new(build_typed_expr_from_bytes(&rest[..mid], ctx)),
                Box::new(build_typed_expr_from_bytes(&rest[mid..], ctx)),
            )
        }
        3 => {
            // NOT
            TLExpr::Not(Box::new(build_typed_expr_from_bytes(rest, ctx)))
        }
        4 if rest.len() >= 1 => {
            // EXISTS with type
            let domain_idx = (rest[0] as usize) % domain_names.len();
            let var_name = format!("x{}", rest[0] % 5);

            TLExpr::Exists {
                var: var_name,
                body: Box::new(build_typed_expr_from_bytes(&rest[1..], ctx)),
            }
        }
        5 if rest.len() >= 2 => {
            // Binary predicate with two typed args
            let domain_idx1 = (rest[0] as usize) % domain_names.len();
            let domain_idx2 = (rest[1] as usize) % domain_names.len();

            let term1 = Term::Typed {
                var: format!("x{}", rest[0] % 3),
                ty: domain_names[domain_idx1].clone(),
            };
            let term2 = Term::Typed {
                var: format!("y{}", rest[1] % 3),
                ty: domain_names[domain_idx2].clone(),
            };

            TLExpr::Predicate {
                name: "P1".to_string(),
                args: vec![term1, term2],
            }
        }
        6 if rest.len() >= 2 => {
            // Implication with type annotations
            let mid = rest.len() / 2;
            TLExpr::Implication(
                Box::new(build_typed_expr_from_bytes(&rest[..mid], ctx)),
                Box::new(build_typed_expr_from_bytes(&rest[mid..], ctx)),
            )
        }
        _ => {
            // Constant
            TLExpr::Constant((data[0] as f64) / 255.0)
        }
    }
}
