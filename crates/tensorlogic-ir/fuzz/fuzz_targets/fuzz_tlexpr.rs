#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_ir::{Term, TLExpr};

fuzz_target!(|data: &[u8]| {
    // Generate random expressions from fuzz data
    if data.len() < 2 {
        return;
    }

    let op_type = data[0] % 10;
    let rest = &data[1..];

    // Build variable and constant terms
    let var_x = Term::Var("x".to_string());
    let var_y = Term::Var("y".to_string());
    let const_true = Term::Const(1.0);
    let const_false = Term::Const(0.0);

    // Create predicates
    let pred1 = TLExpr::Predicate {
        name: "P".to_string(),
        args: vec![var_x.clone()],
    };

    let pred2 = TLExpr::Predicate {
        name: "Q".to_string(),
        args: vec![var_y.clone()],
    };

    // Fuzz different expression types based on op_type
    let expr = match op_type {
        0 => {
            // AND operation
            TLExpr::And(Box::new(pred1.clone()), Box::new(pred2.clone()))
        }
        1 => {
            // OR operation
            TLExpr::Or(Box::new(pred1.clone()), Box::new(pred2.clone()))
        }
        2 => {
            // NOT operation
            TLExpr::Not(Box::new(pred1.clone()))
        }
        3 => {
            // EXISTS quantifier
            TLExpr::Exists {
                var: "x".to_string(),
                body: Box::new(pred1.clone()),
            }
        }
        4 => {
            // FORALL quantifier
            TLExpr::Forall {
                var: "y".to_string(),
                body: Box::new(pred2.clone()),
            }
        }
        5 => {
            // IMPLIES operation
            TLExpr::Implies(Box::new(pred1.clone()), Box::new(pred2.clone()))
        }
        6 => {
            // EQUIV operation
            TLExpr::Equiv(Box::new(pred1.clone()), Box::new(pred2.clone()))
        }
        7 => {
            // Nested AND-OR
            TLExpr::And(
                Box::new(TLExpr::Or(Box::new(pred1.clone()), Box::new(pred2.clone()))),
                Box::new(pred1.clone()),
            )
        }
        8 => {
            // Nested NOT
            TLExpr::Not(Box::new(TLExpr::Not(Box::new(pred1.clone()))))
        }
        _ => {
            // Complex nested expression
            TLExpr::Implies(
                Box::new(TLExpr::And(Box::new(pred1.clone()), Box::new(pred2.clone()))),
                Box::new(TLExpr::Or(Box::new(pred1.clone()), Box::new(pred2.clone()))),
            )
        }
    };

    // Test serialization/deserialization
    if let Ok(serialized) = serde_json::to_string(&expr) {
        let _: Result<TLExpr, _> = serde_json::from_str(&serialized);
    }

    // Test bincode serialization
    if let Ok(encoded) = bincode::encode_to_vec(&expr, bincode::config::standard()) {
        let _: Result<(TLExpr, _), _> =
            bincode::decode_from_slice(&encoded, bincode::config::standard());
    }

    // Test display formatting (shouldn't panic)
    let _ = format!("{:?}", expr);

    // Test free variable extraction (if available)
    // This exercises internal logic without panicking
    drop(expr);
});
