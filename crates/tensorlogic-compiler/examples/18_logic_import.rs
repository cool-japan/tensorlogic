//! # Logic Expression Import Example
//!
//! This example demonstrates how to import logic expressions from various
//! external logic frameworks and formats.
//!
//! ## Supported Formats
//!
//! 1. **Prolog**: Standard Prolog syntax with rules and facts
//! 2. **S-Expression**: Lisp-like syntax for nested logic
//! 3. **TPTP**: Automated theorem proving format
//!
//! ## Usage
//!
//! Import logic from different formats, compile to einsum graphs, and execute.
//!
//! ## Running this Example
//!
//! ```bash
//! cargo run --example 18_logic_import
//! ```

use anyhow::Result;
use tensorlogic_compiler::{
    compile_to_einsum_with_context,
    import::{parse_auto, parse_prolog, parse_sexpr, parse_tptp},
    CompilerContext,
};

fn main() -> Result<()> {
    println!("=== Logic Expression Import Example ===\n");

    example_1_prolog_import()?;
    example_2_sexpr_import()?;
    example_3_tptp_import()?;
    example_4_auto_detection()?;
    example_5_complex_rules()?;

    println!("\n=== Summary ===");
    println!("✓ All import examples completed successfully");
    println!("\nSupported formats:");
    println!("  • Prolog: Facts, rules, conjunctions, disjunctions, negation");
    println!("  • S-Expression: Nested logic with quantifiers");
    println!("  • TPTP: FOF/CNF formulas for theorem proving");
    println!("  • Auto-detect: Automatic format detection");

    Ok(())
}

fn example_1_prolog_import() -> Result<()> {
    println!("## Example 1: Prolog Import");
    println!();

    // Simple fact
    println!("Prolog fact: mortal(socrates).");
    let expr = parse_prolog("mortal(socrates).")?;
    println!("  ✓ Parsed as predicate");
    println!("  Expression: {:?}", expr);
    println!();

    // Rule with implication
    println!("Prolog rule: mortal(X) :- human(X).");
    let expr = parse_prolog("mortal(X) :- human(X).")?;
    println!("  ✓ Parsed as implication");
    println!("  Expression: {:?}", expr);

    // Compile to einsum graph
    let graph = compile_to_einsum_with_context(&expr, &mut CompilerContext::new())?;
    println!("  ✓ Compiled to graph with {} nodes", graph.nodes.len());
    println!();

    // Conjunction
    println!("Prolog conjunction: human(X), mortal(X).");
    let expr = parse_prolog("human(X), mortal(X).")?;
    println!("  ✓ Parsed as conjunction");
    println!("  Expression: {:?}", expr);
    println!();

    // Disjunction
    println!("Prolog disjunction: human(X) ; god(X).");
    let expr = parse_prolog("human(X) ; god(X).")?;
    println!("  ✓ Parsed as disjunction");
    println!("  Expression: {:?}", expr);
    println!();

    // Negation
    println!("Prolog negation: \\+ god(X).");
    let expr = parse_prolog("\\+ god(X).")?;
    println!("  ✓ Parsed as negation");
    println!("  Expression: {:?}", expr);
    println!();

    Ok(())
}

fn example_2_sexpr_import() -> Result<()> {
    println!("## Example 2: S-Expression Import");
    println!();

    // Simple predicate
    println!("S-expr: (mortal socrates)");
    let expr = parse_sexpr("(mortal socrates)")?;
    println!("  ✓ Parsed as predicate");
    println!("  Expression: {:?}", expr);
    println!();

    // Conjunction
    println!("S-expr: (and (human x) (mortal x))");
    let expr = parse_sexpr("(and (human x) (mortal x))")?;
    println!("  ✓ Parsed as conjunction");
    println!("  Expression: {:?}", expr);
    println!();

    // Implication
    println!("S-expr: (=> (human x) (mortal x))");
    let expr = parse_sexpr("(=> (human x) (mortal x))")?;
    println!("  ✓ Parsed as implication");
    println!("  Expression: {:?}", expr);
    println!();

    // Universal quantification
    println!("S-expr: (forall (x Person) (=> (human x) (mortal x)))");
    let expr = parse_sexpr("(forall (x Person) (=> (human x) (mortal x)))")?;
    println!("  ✓ Parsed as universal quantifier");
    println!("  Expression: {:?}", expr);

    // Compile to einsum graph
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;
    println!("  ✓ Compiled to graph with {} nodes", graph.nodes.len());
    println!();

    // Existential quantification
    println!("S-expr: (exists (x Person) (mortal x))");
    let expr = parse_sexpr("(exists (x Person) (mortal x))")?;
    println!("  ✓ Parsed as existential quantifier");
    println!("  Expression: {:?}", expr);
    println!();

    Ok(())
}

fn example_3_tptp_import() -> Result<()> {
    println!("## Example 3: TPTP Import");
    println!();

    // FOF formula (First-Order Formula)
    println!("TPTP FOF: fof(mortality, axiom, ![X]: (human(X) => mortal(X))).");
    let expr = parse_tptp("fof(mortality, axiom, ![X]: (human(X) => mortal(X))).")?;
    println!("  ✓ Parsed as FOF formula");
    println!("  Expression: {:?}", expr);

    // Compile to einsum graph
    let mut ctx = CompilerContext::new();
    ctx.add_domain("Entity", 50);
    let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;
    println!("  ✓ Compiled to graph with {} nodes", graph.nodes.len());
    println!();

    // Existential quantifier
    println!("TPTP FOF: fof(existence, axiom, ?[X]: mortal(X)).");
    let expr = parse_tptp("fof(existence, axiom, ?[X]: mortal(X)).")?;
    println!("  ✓ Parsed with existential quantifier");
    println!("  Expression: {:?}", expr);
    println!();

    // Conjunction
    println!("TPTP FOF: fof(both, axiom, human(socrates) & mortal(socrates)).");
    let expr = parse_tptp("fof(both, axiom, human(socrates) & mortal(socrates)).")?;
    println!("  ✓ Parsed as conjunction");
    println!("  Expression: {:?}", expr);
    println!();

    // Negation
    println!("TPTP FOF: fof(not_god, axiom, ~god(socrates)).");
    let expr = parse_tptp("fof(not_god, axiom, ~god(socrates)).")?;
    println!("  ✓ Parsed as negation");
    println!("  Expression: {:?}", expr);
    println!();

    Ok(())
}

fn example_4_auto_detection() -> Result<()> {
    println!("## Example 4: Auto-Detection");
    println!();

    // Auto-detect Prolog
    println!("Auto-detecting: mortal(socrates).");
    let expr = parse_auto("mortal(socrates).")?;
    println!("  ✓ Detected as Prolog");
    println!("  Expression: {:?}", expr);
    println!();

    // Auto-detect S-expression
    println!("Auto-detecting: (and (human x) (mortal x))");
    let expr = parse_auto("(and (human x) (mortal x))")?;
    println!("  ✓ Detected as S-expression");
    println!("  Expression: {:?}", expr);
    println!();

    // Auto-detect TPTP
    println!("Auto-detecting: fof(test, axiom, mortal(X)).");
    let expr = parse_auto("fof(test, axiom, mortal(X)).")?;
    println!("  ✓ Detected as TPTP");
    println!("  Expression: {:?}", expr);
    println!();

    Ok(())
}

fn example_5_complex_rules() -> Result<()> {
    println!("## Example 5: Complex Rules from Different Formats");
    println!();

    // Prolog: Transitivity rule
    println!("Prolog transitivity:");
    println!("  ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).");
    let expr = parse_prolog("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).")?;
    println!("  ✓ Parsed complex recursive rule");
    println!("  Expression: {:?}", expr);
    println!();

    // S-expr: Complex nested quantifiers
    println!("S-expr nested quantifiers:");
    println!("  (forall (x Person) (exists (y Person) (knows x y)))");
    let expr = parse_sexpr("(forall (x Person) (exists (y Person) (knows x y)))")?;
    println!("  ✓ Parsed nested quantifiers");
    println!("  Expression: {:?}", expr);

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Person", 100);
    let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;
    println!("  ✓ Compiled to graph with {} nodes", graph.nodes.len());
    println!();

    // TPTP: Complex formula with multiple operators
    println!("TPTP complex formula:");
    println!("  fof(complex, axiom, ![X, Y]: ((human(X) & knows(X, Y)) => (human(Y) | god(Y)))).");
    let expr = parse_tptp(
        "fof(complex, axiom, ![X, Y]: ((human(X) & knows(X, Y)) => (human(Y) | god(Y)))).",
    )?;
    println!("  ✓ Parsed complex FOF formula");
    println!("  Expression: {:?}", expr);

    let mut ctx = CompilerContext::new();
    ctx.add_domain("Entity", 50);
    let graph = compile_to_einsum_with_context(&expr, &mut ctx)?;
    println!("  ✓ Compiled to graph with {} nodes", graph.nodes.len());
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1() {
        assert!(example_1_prolog_import().is_ok());
    }

    #[test]
    fn test_example_2() {
        assert!(example_2_sexpr_import().is_ok());
    }

    #[test]
    fn test_example_3() {
        assert!(example_3_tptp_import().is_ok());
    }

    #[test]
    fn test_example_4() {
        assert!(example_4_auto_detection().is_ok());
    }

    #[test]
    fn test_example_5() {
        assert!(example_5_complex_rules().is_ok());
    }
}
