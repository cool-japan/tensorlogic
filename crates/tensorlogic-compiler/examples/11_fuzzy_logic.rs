//! Comprehensive demonstration of fuzzy logic operators.
//!
//! This example showcases all fuzzy logic operators implemented in tensorlogic-compiler:
//! - T-norms (fuzzy AND)
//! - T-conorms (fuzzy OR)
//! - Fuzzy negations
//! - Fuzzy implications
//!
//! We demonstrate practical use cases in fuzzy reasoning, such as temperature control,
//! risk assessment, and decision-making under uncertainty.

use tensorlogic_compiler::{compile_to_einsum_with_context, CompilerContext};
use tensorlogic_ir::{
    FuzzyImplicationKind, FuzzyNegationKind, TCoNormKind, TLExpr, TNormKind, Term,
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     TensorLogic: Fuzzy Logic Operators Demonstration     ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Initialize compiler context
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Temperature", 100);
    ctx.add_domain("Decision", 50);

    // Example 1: T-Norms (Fuzzy AND)
    println!("═══ Example 1: T-Norms (Fuzzy AND) ═══\n");
    demo_tnorms(&mut ctx);

    // Example 2: T-Conorms (Fuzzy OR)
    println!("\n═══ Example 2: T-Conorms (Fuzzy OR) ═══\n");
    demo_tconorms(&mut ctx);

    // Example 3: Fuzzy Negations
    println!("\n═══ Example 3: Fuzzy Negations ═══\n");
    demo_fuzzy_negations(&mut ctx);

    // Example 4: Fuzzy Implications
    println!("\n═══ Example 4: Fuzzy Implications ═══\n");
    demo_fuzzy_implications(&mut ctx);

    // Example 5: Practical Application - Temperature Control
    println!("\n═══ Example 5: Practical Application - HVAC Control ═══\n");
    demo_temperature_control(&mut ctx);

    // Example 6: Risk Assessment with Fuzzy Logic
    println!("\n═══ Example 6: Risk Assessment ═══\n");
    demo_risk_assessment(&mut ctx);

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                  All examples complete!                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
}

/// Demonstrates all T-norm operators (fuzzy AND).
fn demo_tnorms(ctx: &mut CompilerContext) {
    println!("T-norms generalize logical AND to the fuzzy domain [0,1].\n");

    let warm = TLExpr::pred("warm", vec![Term::var("x")]);
    let comfortable = TLExpr::pred("comfortable", vec![Term::var("x")]);

    // 1. Minimum T-norm (Gödel)
    println!("1. Minimum T-norm: min(a, b)");
    println!("   Use case: Conservative fuzzy AND (most restrictive)");
    let min_expr = TLExpr::tnorm(TNormKind::Minimum, warm.clone(), comfortable.clone());
    compile_and_report("minimum_tnorm", &min_expr, ctx);

    // 2. Product T-norm
    println!("\n2. Product T-norm: a * b");
    println!("   Use case: Probabilistic interpretation (independence)");
    let prod_expr = TLExpr::tnorm(TNormKind::Product, warm.clone(), comfortable.clone());
    compile_and_report("product_tnorm", &prod_expr, ctx);

    // 3. Łukasiewicz T-norm
    println!("\n3. Łukasiewicz T-norm: max(0, a + b - 1)");
    println!("   Use case: Strong conjunction (strict requirements)");
    let luk_expr = TLExpr::tnorm(TNormKind::Lukasiewicz, warm.clone(), comfortable.clone());
    compile_and_report("lukasiewicz_tnorm", &luk_expr, ctx);

    // 4. Nilpotent Minimum
    println!("\n4. Nilpotent Minimum: min(a,b) if a+b>1, else 0");
    println!("   Use case: Threshold-based conjunction");
    let nilp_expr = TLExpr::tnorm(
        TNormKind::NilpotentMinimum,
        warm.clone(),
        comfortable.clone(),
    );
    compile_and_report("nilpotent_minimum_tnorm", &nilp_expr, ctx);

    // 5. Hamacher Product
    println!("\n5. Hamacher Product: ab/(a+b-ab)");
    println!("   Use case: Parametric family for flexible conjunction");
    let ham_expr = TLExpr::tnorm(TNormKind::Hamacher, warm, comfortable);
    compile_and_report("hamacher_tnorm", &ham_expr, ctx);
}

/// Demonstrates all T-conorm operators (fuzzy OR).
fn demo_tconorms(ctx: &mut CompilerContext) {
    println!("T-conorms generalize logical OR to the fuzzy domain [0,1].\n");

    let hot = TLExpr::pred("hot", vec![Term::var("x")]);
    let cold = TLExpr::pred("cold", vec![Term::var("x")]);

    // 1. Maximum T-conorm (Gödel)
    println!("1. Maximum T-conorm: max(a, b)");
    println!("   Use case: Optimistic fuzzy OR (least restrictive)");
    let max_expr = TLExpr::tconorm(TCoNormKind::Maximum, hot.clone(), cold.clone());
    compile_and_report("maximum_tconorm", &max_expr, ctx);

    // 2. Probabilistic Sum
    println!("\n2. Probabilistic Sum: a + b - ab");
    println!("   Use case: Probabilistic union of events");
    let prob_expr = TLExpr::tconorm(TCoNormKind::ProbabilisticSum, hot.clone(), cold.clone());
    compile_and_report("probabilistic_sum_tconorm", &prob_expr, ctx);

    // 3. Bounded Sum
    println!("\n3. Bounded Sum: min(1, a + b)");
    println!("   Use case: Additive disjunction with saturation");
    let bounded_expr = TLExpr::tconorm(TCoNormKind::BoundedSum, hot.clone(), cold.clone());
    compile_and_report("bounded_sum_tconorm", &bounded_expr, ctx);

    // 4. Nilpotent Maximum
    println!("\n4. Nilpotent Maximum: max(a,b) if a+b<1, else 1");
    println!("   Use case: Threshold-based disjunction");
    let nilp_max_expr = TLExpr::tconorm(TCoNormKind::NilpotentMaximum, hot.clone(), cold.clone());
    compile_and_report("nilpotent_maximum_tconorm", &nilp_max_expr, ctx);

    // 5. Hamacher Sum
    println!("\n5. Hamacher Sum: (a+b-2ab)/(1-ab)");
    println!("   Use case: Dual of Hamacher product");
    let ham_sum_expr = TLExpr::tconorm(TCoNormKind::Hamacher, hot, cold);
    compile_and_report("hamacher_tconorm", &ham_sum_expr, ctx);
}

/// Demonstrates fuzzy negation operators.
fn demo_fuzzy_negations(ctx: &mut CompilerContext) {
    println!("Fuzzy negations generalize logical NOT to continuous values.\n");

    let likely = TLExpr::pred("likely", vec![Term::var("x")]);

    // 1. Standard Negation
    println!("1. Standard Negation: 1 - a");
    println!("   Use case: Classical complement (most common)");
    let std_not = TLExpr::fuzzy_not(FuzzyNegationKind::Standard, likely.clone());
    compile_and_report("standard_negation", &std_not, ctx);

    // 2. Sugeno Negation
    println!("\n2. Sugeno Negation: (1-a)/(1+λa) for λ > -1");
    println!("   Use case: Parametric negation (λ=50 means λ=0.5)");
    let sugeno_not = TLExpr::fuzzy_not(FuzzyNegationKind::Sugeno { lambda: 50 }, likely.clone());
    compile_and_report("sugeno_negation", &sugeno_not, ctx);

    // 3. Yager Negation
    println!("\n3. Yager Negation: (1-a^w)^(1/w) for w > 0");
    println!("   Use case: Power-based negation (w=20 means w=2.0)");
    let yager_not = TLExpr::fuzzy_not(FuzzyNegationKind::Yager { w: 20 }, likely);
    compile_and_report("yager_negation", &yager_not, ctx);
}

/// Demonstrates fuzzy implication operators.
fn demo_fuzzy_implications(ctx: &mut CompilerContext) {
    println!("Fuzzy implications generalize logical implication (a → b).\n");

    let rain = TLExpr::pred("rain", vec![Term::var("x")]);
    let umbrella = TLExpr::pred("umbrella", vec![Term::var("x")]);

    // 1. Gödel Implication
    println!("1. Gödel Implication: 1 if a≤b, else b");
    println!("   Use case: Residuum of minimum t-norm");
    let godel = TLExpr::fuzzy_imply(FuzzyImplicationKind::Godel, rain.clone(), umbrella.clone());
    compile_and_report("godel_implication", &godel, ctx);

    // 2. Łukasiewicz Implication
    println!("\n2. Łukasiewicz Implication: min(1, 1-a+b)");
    println!("   Use case: Residuum of Łukasiewicz t-norm");
    let luk = TLExpr::fuzzy_imply(
        FuzzyImplicationKind::Lukasiewicz,
        rain.clone(),
        umbrella.clone(),
    );
    compile_and_report("lukasiewicz_implication", &luk, ctx);

    // 3. Reichenbach Implication
    println!("\n3. Reichenbach Implication: 1 - a + ab");
    println!("   Use case: Probabilistic implication");
    let reich = TLExpr::fuzzy_imply(
        FuzzyImplicationKind::Reichenbach,
        rain.clone(),
        umbrella.clone(),
    );
    compile_and_report("reichenbach_implication", &reich, ctx);

    // 4. Kleene-Dienes Implication
    println!("\n4. Kleene-Dienes Implication: max(1-a, b)");
    println!("   Use case: Material implication in fuzzy logic");
    let kd = TLExpr::fuzzy_imply(
        FuzzyImplicationKind::KleeneDienes,
        rain.clone(),
        umbrella.clone(),
    );
    compile_and_report("kleene_dienes_implication", &kd, ctx);

    // 5. Rescher Implication
    println!("\n5. Rescher Implication: 1 if a≤b, else 0");
    println!("   Use case: Crisp (binary) implication");
    let rescher = TLExpr::fuzzy_imply(
        FuzzyImplicationKind::Rescher,
        rain.clone(),
        umbrella.clone(),
    );
    compile_and_report("rescher_implication", &rescher, ctx);

    // 6. Goguen Implication
    println!("\n6. Goguen Implication: 1 if a≤b, else b/a");
    println!("   Use case: Residuum of product t-norm");
    let goguen = TLExpr::fuzzy_imply(FuzzyImplicationKind::Goguen, rain, umbrella);
    compile_and_report("goguen_implication", &goguen, ctx);
}

/// Practical example: HVAC temperature control system.
fn demo_temperature_control(ctx: &mut CompilerContext) {
    println!("Scenario: Smart HVAC system using fuzzy logic\n");

    // Temperature predicates
    let too_hot = TLExpr::pred("too_hot", vec![Term::var("temp")]);
    let too_cold = TLExpr::pred("too_cold", vec![Term::var("temp")]);
    let humid = TLExpr::pred("humid", vec![Term::var("temp")]);

    // Rule 1: If (too_hot AND humid) OR too_cold, then activate climate control
    println!("Rule: Climate control = (too_hot AND humid) OR too_cold");

    // Using Product t-norm for AND (probabilistic)
    let hot_and_humid = TLExpr::tnorm(TNormKind::Product, too_hot, humid);

    // Using Maximum t-conorm for OR
    let need_control = TLExpr::tconorm(TCoNormKind::Maximum, hot_and_humid, too_cold);

    compile_and_report("climate_control_rule", &need_control, ctx);

    println!("\nInterpretation:");
    println!("  • Product t-norm: Hot AND humid treated as independent events");
    println!("  • Maximum t-conorm: Activate if EITHER condition is strong");
    println!("  • Result: Smooth, continuous control signal");
}

/// Practical example: Risk assessment with fuzzy logic.
fn demo_risk_assessment(ctx: &mut CompilerContext) {
    println!("Scenario: Investment risk assessment\n");

    // Risk factors
    let market_volatile = TLExpr::pred("market_volatile", vec![Term::var("d")]);
    let high_leverage = TLExpr::pred("high_leverage", vec![Term::var("d")]);
    let regulatory_risk = TLExpr::pred("regulatory_risk", vec![Term::var("d")]);

    // Compound risk: If market is volatile AND (high leverage OR regulatory risk)
    println!("Risk = volatile AND (high_leverage OR regulatory)");

    // Use Łukasiewicz t-conorm for OR (additive with saturation)
    let financial_risk = TLExpr::tconorm(TCoNormKind::BoundedSum, high_leverage, regulatory_risk);

    // Use Minimum t-norm for AND (conservative)
    let total_risk = TLExpr::tnorm(TNormKind::Minimum, market_volatile, financial_risk);

    compile_and_report("investment_risk", &total_risk, ctx);

    println!("\nInterpretation:");
    println!("  • BoundedSum: Multiple risks accumulate (max out at 1.0)");
    println!("  • Minimum t-norm: Conservative risk estimation");
    println!("  • Result: Safe, worst-case risk assessment");

    // Risk mitigation rule using implication
    println!("\n--- Risk Mitigation Rule ---");
    let high_risk = TLExpr::pred("high_risk", vec![Term::var("d")]);
    let hedge = TLExpr::pred("apply_hedge", vec![Term::var("d")]);

    // If high risk, then hedge (using Gödel implication)
    let mitigation = TLExpr::fuzzy_imply(FuzzyImplicationKind::Godel, high_risk, hedge);

    compile_and_report("risk_mitigation", &mitigation, ctx);

    println!("\nInterpretation:");
    println!("  • Gödel implication: Strong consequence when premise is true");
    println!("  • Result: Clear hedging signal when risk exceeds threshold");
}

/// Helper function to compile an expression and report results.
fn compile_and_report(_name: &str, expr: &TLExpr, ctx: &mut CompilerContext) {
    match compile_to_einsum_with_context(expr, ctx) {
        Ok(graph) => {
            println!("   ✓ Compiled successfully:");
            println!("     - {} tensors", graph.tensors.len());
            println!("     - {} operations", graph.nodes.len());
            println!("     - {} outputs", graph.outputs.len());

            // Validate graph
            if let Err(e) = graph.validate() {
                println!("     ⚠ Validation warning: {}", e);
            }
        }
        Err(e) => {
            println!("   ✗ Compilation failed: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_tnorms_compile() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);

        let left = TLExpr::pred("P", vec![Term::var("x")]);
        let right = TLExpr::pred("Q", vec![Term::var("x")]);

        for kind in [
            TNormKind::Minimum,
            TNormKind::Product,
            TNormKind::Lukasiewicz,
            TNormKind::NilpotentMinimum,
            TNormKind::Hamacher,
        ] {
            let expr = TLExpr::tnorm(kind, left.clone(), right.clone());
            let result = compile_to_einsum_with_context(&expr, &mut ctx);
            assert!(result.is_ok(), "TNorm {:?} should compile", kind);
        }
    }

    #[test]
    fn test_all_tconorms_compile() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);

        let left = TLExpr::pred("P", vec![Term::var("x")]);
        let right = TLExpr::pred("Q", vec![Term::var("x")]);

        for kind in [
            TCoNormKind::Maximum,
            TCoNormKind::ProbabilisticSum,
            TCoNormKind::BoundedSum,
            TCoNormKind::NilpotentMaximum,
            TCoNormKind::Hamacher,
        ] {
            let expr = TLExpr::tconorm(kind, left.clone(), right.clone());
            let result = compile_to_einsum_with_context(&expr, &mut ctx);
            assert!(result.is_ok(), "TCoNorm {:?} should compile", kind);
        }
    }

    #[test]
    fn test_complex_fuzzy_expression() {
        let mut ctx = CompilerContext::new();
        ctx.add_domain("D", 10);

        // Complex: (A tnorm B) tconorm NOT(C)
        let a = TLExpr::pred("A", vec![Term::var("x")]);
        let b = TLExpr::pred("B", vec![Term::var("x")]);
        let c = TLExpr::pred("C", vec![Term::var("x")]);

        let a_and_b = TLExpr::tnorm(TNormKind::Product, a, b);
        let not_c = TLExpr::fuzzy_not(FuzzyNegationKind::Standard, c);
        let expr = TLExpr::tconorm(TCoNormKind::Maximum, a_and_b, not_c);

        let result = compile_to_einsum_with_context(&expr, &mut ctx);
        assert!(result.is_ok());

        let graph = result.unwrap();
        assert!(graph.validate().is_ok());
    }
}
