#![no_main]

use libfuzzer_sys::fuzz_target;
use tensorlogic_ir::{Term, TLExpr};

fuzz_target!(|data: &[u8]| {
    // Limit input size to avoid excessive memory usage
    if data.len() < 2 || data.len() > 1000 {
        return;
    }

    // Use first bytes to determine test scenario
    let choice = data[0] % 10;

    // Create base predicates using valid Term constructors
    let term1 = Term::var("x");
    let term2 = Term::var("y");
    let pred1 = TLExpr::pred("P", vec![term1.clone()]);
    let pred2 = TLExpr::pred("Q", vec![term2.clone()]);

    // Build different expression types based on input
    let expr = match choice {
        0 => {
            // Simple AND operation
            TLExpr::and(pred1.clone(), pred2.clone())
        }
        1 => {
            // Simple OR operation
            TLExpr::or(pred1.clone(), pred2.clone())
        }
        2 => {
            // Simple NOT operation
            TLExpr::negate(pred1.clone())
        }
        3 => {
            // EXISTS quantifier
            TLExpr::exists("x", "D", pred1.clone())
        }
        4 => {
            // FORALL quantifier
            TLExpr::forall("y", "D", pred2.clone())
        }
        5 => {
            // IMPLIES operation
            TLExpr::imply(pred1.clone(), pred2.clone())
        }
        6 => {
            // EQUIV operation (A ↔ B = (A → B) ∧ (B → A))
            let forward = TLExpr::imply(pred1.clone(), pred2.clone());
            let backward = TLExpr::imply(pred2.clone(), pred1.clone());
            TLExpr::and(forward, backward)
        }
        7 => {
            // Nested AND-OR
            let inner_or = TLExpr::or(pred1.clone(), pred2.clone());
            TLExpr::and(inner_or, pred1.clone())
        }
        8 => {
            // Nested NOT
            let inner_not = TLExpr::negate(pred1.clone());
            TLExpr::negate(inner_not)
        }
        _ => {
            // Complex nested expression
            let inner_and = TLExpr::and(pred1.clone(), pred2.clone());
            let inner_or = TLExpr::or(pred1.clone(), pred2.clone());
            TLExpr::imply(inner_and, inner_or)
        }
    };

    // Test serialization/deserialization (JSON)
    if let Ok(serialized) = serde_json::to_string(&expr) {
        let _: Result<TLExpr, _> = serde_json::from_str(&serialized);
    }

    // Test display formatting (shouldn't panic)
    let _ = format!("{:?}", expr);
    let _ = format!("{}", expr);

    // Test cloning (shouldn't panic)
    let _ = expr.clone();

    // Test equality (shouldn't panic)
    let _ = expr == expr.clone();
});
